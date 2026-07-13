from __future__ import annotations

import torch
from torch import nn


class EdgeMessageLayer(nn.Module):
    def __init__(
        self, hidden_size: int, edge_size: int, dropout: float, edge_dropout: float = 0.0
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
    ) -> torch.Tensor:
        source, target = edge_index
        if self.training and self.edge_dropout > 0:
            keep = torch.rand(len(source), device=source.device) >= self.edge_dropout
            source = source[keep]
            target = target[keep]
            edge_attr = edge_attr[keep]
        source_nodes = nodes[:, source]
        target_nodes = nodes[:, target]
        edge_values = edge_attr.unsqueeze(0).expand(nodes.shape[0], -1, -1)
        messages = self.message(torch.cat([source_nodes, target_nodes, edge_values], dim=-1))
        aggregated = torch.zeros_like(nodes)
        indices = target.view(1, -1, 1).expand(nodes.shape[0], -1, nodes.shape[-1])
        aggregated.scatter_add_(1, indices, messages)
        degree = torch.zeros(nodes.shape[1], device=nodes.device, dtype=nodes.dtype)
        degree.scatter_add_(0, target, torch.ones_like(target, dtype=nodes.dtype))
        aggregated = aggregated / degree.clamp_min(1.0).view(1, -1, 1)
        update = self.update(torch.cat([nodes, aggregated], dim=-1))
        return self.norm(nodes + update)


class TurbineOutputMixin:
    residual_amplitude: float
    direct_output: bool
    turbine_group_index: torch.Tensor
    group_aggregation: torch.Tensor
    group_scale_theta: nn.Parameter

    def _power_output(
        self, residual_logits: torch.Tensor, teacher_base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.direct_output:
            turbine_power = torch.sigmoid(residual_logits)
        else:
            group_scale = 1.0 + 0.25 * torch.tanh(self.group_scale_theta)
            turbine_scale = group_scale[self.turbine_group_index].unsqueeze(0)
            turbine_power = torch.clamp(
                turbine_scale * teacher_base
                + self.residual_amplitude * torch.tanh(residual_logits),
                0.0,
                1.0,
            )
        group_power = turbine_power @ self.group_aggregation
        return turbine_power, group_power


class SpatialTeacherGNN(nn.Module, TurbineOutputMixin):
    def __init__(
        self,
        node_size: int,
        edge_size: int,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        turbine_indices: torch.Tensor,
        turbine_group_index: torch.Tensor,
        group_aggregation: torch.Tensor,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.10,
        edge_dropout: float = 0.0,
        residual_amplitude: float = 0.25,
        direct_output: bool = False,
    ) -> None:
        super().__init__()
        self.residual_amplitude = float(residual_amplitude)
        self.direct_output = bool(direct_output)
        self.register_buffer("edge_index", edge_index.long())
        self.register_buffer("edge_attr", edge_attr.float())
        self.register_buffer("turbine_indices", turbine_indices.long())
        self.register_buffer("turbine_group_index", turbine_group_index.long())
        self.register_buffer("group_aggregation", group_aggregation.float())
        self.encoder = nn.Sequential(
            nn.Linear(node_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                EdgeMessageLayer(hidden_size, edge_size, dropout, edge_dropout)
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.group_scale_theta = nn.Parameter(torch.zeros(group_aggregation.shape[1]))
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self, node_features: torch.Tensor, teacher_base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self.encoder(node_features)
        for layer in self.layers:
            nodes = layer(nodes, self.edge_index, self.edge_attr)
        residual = self.head(nodes[:, self.turbine_indices]).squeeze(-1)
        return self._power_output(residual, teacher_base)


class TeacherTurbineMLP(nn.Module, TurbineOutputMixin):
    def __init__(
        self,
        node_size: int,
        turbine_indices: torch.Tensor,
        turbine_group_index: torch.Tensor,
        group_aggregation: torch.Tensor,
        hidden_size: int = 64,
        dropout: float = 0.10,
        residual_amplitude: float = 0.25,
        direct_output: bool = False,
    ) -> None:
        super().__init__()
        self.residual_amplitude = float(residual_amplitude)
        self.direct_output = bool(direct_output)
        self.register_buffer("turbine_indices", turbine_indices.long())
        self.register_buffer("turbine_group_index", turbine_group_index.long())
        self.register_buffer("group_aggregation", group_aggregation.float())
        self.head = nn.Sequential(
            nn.Linear(node_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden_size // 2), 1),
        )
        self.group_scale_theta = nn.Parameter(torch.zeros(group_aggregation.shape[1]))
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self, node_features: torch.Tensor, teacher_base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = self.head(node_features[:, self.turbine_indices]).squeeze(-1)
        return self._power_output(residual, teacher_base)
