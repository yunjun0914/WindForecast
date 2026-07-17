from __future__ import annotations

import math

import torch
from torch import nn

from models.issue_block_tcn import FullContextTemporalBlock


class IssueDecisionQTCN(nn.Module):
    """Estimate the expected official reward for each candidate group forecast."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.10,
        value_size: int = 64,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = int(input_size)
        for layer_index in range(int(num_layers)):
            layers.append(
                FullContextTemporalBlock(
                    in_channels,
                    int(hidden_size),
                    int(kernel_size),
                    dilation=2**layer_index,
                    dropout=float(dropout),
                )
            )
            in_channels = int(hidden_size)
        self.encoder = nn.Sequential(*layers)
        self.context_value = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), int(value_size)),
            nn.Tanh(),
        )
        self.context_bias = nn.Linear(int(hidden_size), 1)
        self.candidate_value = nn.Sequential(
            nn.Linear(3, int(value_size)),
            nn.Tanh(),
            nn.Linear(int(value_size), int(value_size)),
        )
        self.candidate_bias = nn.Sequential(
            nn.Linear(3, max(8, int(value_size) // 2)),
            nn.Tanh(),
            nn.Linear(max(8, int(value_size) // 2), 1),
        )
        self.value_scale = math.sqrt(float(value_size))

    @staticmethod
    def candidate_features(candidates: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [candidates, candidates.square(), candidates.pow(3)], dim=-1
        )

    def forward(
        self,
        features: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected [batch, time, feature], got {features.shape}")
        if candidates.ndim != 1:
            raise ValueError(f"Expected one-dimensional candidates, got {candidates.shape}")
        hidden = self.encoder(features.transpose(1, 2)).transpose(1, 2)
        context_value = self.context_value(hidden)
        candidate_features = self.candidate_features(candidates)
        candidate_value = self.candidate_value(candidate_features)
        interaction = torch.einsum(
            "bth,mh->btm", context_value, candidate_value
        ) / self.value_scale
        return (
            interaction
            + self.context_bias(hidden)
            + self.candidate_bias(candidate_features).transpose(0, 1)
        )
