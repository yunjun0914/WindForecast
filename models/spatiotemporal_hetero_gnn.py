from __future__ import annotations

import torch
from torch import nn


class StaticMessageLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        edge_size: int,
        dropout: float,
        edge_dropout: float,
    ) -> None:
        super().__init__()
        self.edge_dropout = float(edge_dropout)
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_size + edge_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_wind: torch.Tensor,
    ) -> torch.Tensor:
        del node_wind
        source, target = edge_index
        if self.training and self.edge_dropout > 0:
            keep = torch.rand(len(source), device=nodes.device) >= self.edge_dropout
            source, target, edge_attr = source[keep], target[keep], edge_attr[keep]
        edge_values = edge_attr.unsqueeze(0).expand(nodes.shape[0], -1, -1)
        messages = self.message(
            torch.cat([nodes[:, source], nodes[:, target], edge_values], dim=-1)
        )
        return self._update(nodes, target, messages)

    def _update(
        self, nodes: torch.Tensor, target: torch.Tensor, messages: torch.Tensor
    ) -> torch.Tensor:
        aggregated = torch.zeros_like(nodes)
        indices = target.view(1, -1, 1).expand(nodes.shape[0], -1, nodes.shape[-1])
        aggregated.scatter_add_(1, indices, messages)
        degree = torch.zeros(nodes.shape[1], device=nodes.device, dtype=nodes.dtype)
        degree.scatter_add_(0, target, torch.ones_like(target, dtype=nodes.dtype))
        aggregated /= degree.clamp_min(1.0).view(1, -1, 1)
        update = self.update(torch.cat([nodes, aggregated], dim=-1))
        return self.norm(nodes + update)


class DynamicWindMessageLayer(StaticMessageLayer):
    DYNAMIC_EDGE_SIZE = 5

    def __init__(
        self,
        hidden_size: int,
        edge_size: int,
        dropout: float,
        edge_dropout: float,
    ) -> None:
        super().__init__(hidden_size, edge_size + self.DYNAMIC_EDGE_SIZE, dropout, edge_dropout)
        self.gate = nn.Sequential(
            nn.Linear(self.DYNAMIC_EDGE_SIZE, max(8, hidden_size // 4)),
            nn.GELU(),
            nn.Linear(max(8, hidden_size // 4), 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_wind: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        if self.training and self.edge_dropout > 0:
            keep = torch.rand(len(source), device=nodes.device) >= self.edge_dropout
            source, target, edge_attr = source[keep], target[keep], edge_attr[keep]
        source_wind = node_wind[:, source]
        edge_x = edge_attr[:, 4].view(1, -1)
        edge_y = edge_attr[:, 3].view(1, -1)
        alignment = source_wind[:, :, 0] * edge_x + source_wind[:, :, 1] * edge_y
        crosswind = source_wind[:, :, 0] * edge_y - source_wind[:, :, 1] * edge_x
        speed = source_wind[:, :, 2]
        distance = edge_attr[:, 2].view(1, -1).expand_as(speed)
        travel_time = torch.clamp(distance / (0.10 + speed), 0.0, 5.0) / 5.0
        dynamic = torch.stack(
            [alignment, torch.abs(crosswind), torch.relu(alignment), speed, travel_time],
            dim=-1,
        )
        static = edge_attr.unsqueeze(0).expand(nodes.shape[0], -1, -1)
        messages = self.message(
            torch.cat([nodes[:, source], nodes[:, target], static, dynamic], dim=-1)
        )
        messages *= self.gate(dynamic)
        return self._update(nodes, target, messages)


class SpatioTemporalHeteroGNN(nn.Module):
    def __init__(
        self,
        node_size: int,
        edge_size: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_type: torch.Tensor,
        turbine_indices: torch.Tensor,
        group_aggregation: torch.Tensor,
        hidden_size: int = 64,
        num_layers: int = 2,
        gru_hidden_size: int = 48,
        dropout: float = 0.20,
        edge_dropout: float = 0.15,
        dynamic_edges: bool = False,
    ) -> None:
        super().__init__()
        self.dynamic_edges = bool(dynamic_edges)
        self.register_buffer("edge_index", edge_index.long())
        self.register_buffer("edge_attr", edge_attr.float())
        self.register_buffer("node_type", node_type.long())
        self.register_buffer("turbine_indices", turbine_indices.long())
        self.register_buffer("group_aggregation", group_aggregation.float())
        self.type_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(node_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                )
                for _ in range(3)
            ]
        )
        layer_class = DynamicWindMessageLayer if dynamic_edges else StaticMessageLayer
        self.layers = nn.ModuleList(
            [
                layer_class(hidden_size, edge_size, dropout, edge_dropout)
                for _ in range(num_layers)
            ]
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * gru_hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def encode_nodes(self, features: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(
            *features.shape[:-1],
            self.type_encoders[0][0].out_features,
            device=features.device,
            dtype=features.dtype,
        )
        for node_type, encoder in enumerate(self.type_encoders):
            mask = self.node_type == node_type
            output[:, mask] = encoder(features[:, mask])
        return output

    def forward(
        self, node_features: torch.Tensor, node_wind: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, node_count, feature_size = node_features.shape
        flat_features = node_features.reshape(
            batch_size * sequence_length, node_count, feature_size
        )
        flat_wind = node_wind.reshape(batch_size * sequence_length, node_count, 3)
        nodes = self.encode_nodes(flat_features)
        for layer in self.layers:
            nodes = layer(nodes, self.edge_index, self.edge_attr, flat_wind)
        turbine_nodes = nodes[:, self.turbine_indices]
        turbine_count = turbine_nodes.shape[1]
        sequence = turbine_nodes.reshape(
            batch_size, sequence_length, turbine_count, -1
        ).permute(0, 2, 1, 3)
        sequence = sequence.reshape(batch_size * turbine_count, sequence_length, -1)
        encoded, _ = self.gru(sequence)
        logits = self.head(encoded).squeeze(-1)
        turbine_power = torch.sigmoid(logits).reshape(
            batch_size, turbine_count, sequence_length
        ).permute(0, 2, 1)
        group_power = turbine_power @ self.group_aggregation
        return turbine_power, group_power
