from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from models.issue_block_tcn import FullContextTemporalBlock


class SpatialMLPEncoder(nn.Module):
    """Encode valid cells from one weather source without erasing grid identity."""

    def __init__(
        self,
        input_channels: int,
        spatial_mask: torch.Tensor,
        hidden_size: int,
        embedding_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        mask = torch.as_tensor(spatial_mask, dtype=torch.bool)
        if mask.ndim != 2 or not bool(mask.any()):
            raise ValueError("spatial_mask must be a non-empty 2D mask")
        self.register_buffer("spatial_mask", mask)
        input_size = int(input_channels) * int(mask.sum().item())
        self.network = nn.Sequential(
            nn.Linear(input_size, int(hidden_size)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size), int(embedding_size)),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5:
            raise ValueError(
                f"Expected [batch,time,channel,height,width], got {values.shape}"
            )
        if tuple(values.shape[-2:]) != tuple(self.spatial_mask.shape):
            raise ValueError("Input grid shape differs from the configured spatial mask")
        valid_cells = values[..., self.spatial_mask]
        flattened = valid_cells.flatten(start_dim=2)
        return self.network(flattened)


class SourceExpertTCN(nn.Module):
    """Source-specific spatial encoder followed by a full-issue temporal TCN."""

    def __init__(
        self,
        component_channels: Sequence[int],
        spatial_masks: Sequence[torch.Tensor],
        component_hidden_sizes: Sequence[int],
        component_embedding_sizes: Sequence[int],
        time_feature_size: int = 5,
        temporal_hidden_size: int = 64,
        num_temporal_blocks: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.10,
        output_size: int = 3,
    ) -> None:
        super().__init__()
        component_count = len(component_channels)
        lengths = {
            component_count,
            len(spatial_masks),
            len(component_hidden_sizes),
            len(component_embedding_sizes),
        }
        if component_count == 0 or len(lengths) != 1:
            raise ValueError("All component configuration lists must have equal length")
        if kernel_size % 2 == 0:
            raise ValueError("Full-context TCN requires an odd kernel size")

        self.component_encoders = nn.ModuleList(
            [
                SpatialMLPEncoder(
                    input_channels=channels,
                    spatial_mask=mask,
                    hidden_size=hidden,
                    embedding_size=embedding,
                    dropout=dropout,
                )
                for channels, mask, hidden, embedding in zip(
                    component_channels,
                    spatial_masks,
                    component_hidden_sizes,
                    component_embedding_sizes,
                )
            ]
        )
        merged_size = int(sum(component_embedding_sizes))
        temporal_hidden_size = int(temporal_hidden_size)
        if merged_size == temporal_hidden_size:
            self.source_fusion = nn.Identity()
        else:
            self.source_fusion = nn.Sequential(
                nn.Linear(merged_size, temporal_hidden_size),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            )

        temporal_layers = []
        input_size = temporal_hidden_size + int(time_feature_size)
        for layer_index in range(int(num_temporal_blocks)):
            temporal_layers.append(
                FullContextTemporalBlock(
                    input_size,
                    temporal_hidden_size,
                    kernel_size=int(kernel_size),
                    dilation=2**layer_index,
                    dropout=float(dropout),
                )
            )
            input_size = temporal_hidden_size
        self.temporal_encoder = nn.Sequential(*temporal_layers)
        head_hidden = max(16, temporal_hidden_size // 2)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(temporal_hidden_size),
                    nn.Linear(temporal_hidden_size, head_hidden),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(head_hidden, 1),
                )
                for _ in range(int(output_size))
            ]
        )
        self.receptive_field = 1 + 2 * (int(kernel_size) - 1) * sum(
            2**index for index in range(int(num_temporal_blocks))
        )

    def forward(
        self,
        components: Sequence[torch.Tensor],
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        if len(components) != len(self.component_encoders):
            raise ValueError("Input component count differs from model configuration")
        if time_features.ndim != 3:
            raise ValueError(
                f"Expected [batch,time,feature] time features, got {time_features.shape}"
            )
        embeddings = [
            encoder(component)
            for encoder, component in zip(self.component_encoders, components)
        ]
        if any(embedding.shape[:2] != time_features.shape[:2] for embedding in embeddings):
            raise ValueError("Source components and time features are not aligned")
        source_embedding = self.source_fusion(torch.cat(embeddings, dim=-1))
        temporal_input = torch.cat([source_embedding, time_features], dim=-1)
        hidden = self.temporal_encoder(temporal_input.transpose(1, 2)).transpose(1, 2)
        logits = torch.cat([head(hidden) for head in self.heads], dim=-1)
        return torch.sigmoid(logits)
