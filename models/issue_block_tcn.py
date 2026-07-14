from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.seqnn import TemporalBlock


class FullContextTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Full-context TCN requires an odd kernel size")
        padding = ((kernel_size - 1) * dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return F.relu(out + residual)


class IssueBlockTCN(nn.Module):
    """Map one complete forecast issue to one power trajectory per output head."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.10,
        full_context: bool = True,
    ) -> None:
        super().__init__()
        block = FullContextTemporalBlock if full_context else TemporalBlock
        layers = []
        in_channels = int(input_size)
        for layer_index in range(int(num_layers)):
            layers.append(
                block(
                    in_channels,
                    int(hidden_size),
                    int(kernel_size),
                    dilation=2**layer_index,
                    dropout=float(dropout),
                )
            )
            in_channels = int(hidden_size)
        self.encoder = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(int(hidden_size)),
                    nn.Linear(int(hidden_size), max(16, int(hidden_size) // 2)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(max(16, int(hidden_size) // 2), 1),
                )
                for _ in range(int(output_size))
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected [batch, time, feature], got {features.shape}")
        hidden = self.encoder(features.transpose(1, 2)).transpose(1, 2)
        return torch.cat([head(hidden) for head in self.heads], dim=-1)
