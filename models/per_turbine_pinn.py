from __future__ import annotations

import torch
from torch import nn

from models.seqnn import TemporalBlock


class PerTurbineResidualPINN(nn.Module):
    """Turbine-level physical power curve with bounded calendar and NN corrections."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        residual_amplitude: float = 0.15,
        max_lead_hour: int = 240,
    ):
        super().__init__()
        self.residual_amplitude = float(residual_amplitude)
        self.max_lead_hour = int(max_lead_hour)
        self.physics_scale_theta = nn.Parameter(torch.zeros(()))
        self.lead_bias = nn.Embedding(self.max_lead_hour + 1, 1)
        self.month_bias = nn.Embedding(12, 1)
        self.residual = nn.Sequential(
            nn.Linear(int(input_size), int(hidden_size)),
            nn.Tanh(),
            nn.Linear(int(hidden_size), max(8, int(hidden_size) // 2)),
            nn.Tanh(),
            nn.Linear(max(8, int(hidden_size) // 2), 1),
        )
        nn.init.zeros_(self.lead_bias.weight)
        nn.init.zeros_(self.month_bias.weight)
        nn.init.normal_(self.residual[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual[-1].bias)

    @property
    def physics_scale(self) -> torch.Tensor:
        return 1.0 + 0.5 * torch.tanh(self.physics_scale_theta)

    def forward(
        self,
        features: torch.Tensor,
        physical_norm: torch.Tensor,
        lead_hour: torch.Tensor,
        month_index: torch.Tensor,
        return_parts: bool = False,
    ):
        lead_hour = lead_hour.clamp(0, self.max_lead_hour)
        month_index = month_index.clamp(0, 11)
        physical = self.physics_scale * physical_norm
        calendar = self.lead_bias(lead_hour).squeeze(-1) + self.month_bias(month_index).squeeze(-1)
        residual = self.residual_amplitude * torch.tanh(self.residual(features).squeeze(-1))
        prediction = torch.clamp(physical + calendar + residual, 0.0, 1.0)
        if return_parts:
            return prediction, {
                "physical": physical,
                "calendar": calendar,
                "residual": residual,
            }
        return prediction

    def regularization(self) -> dict[str, torch.Tensor]:
        residual_l2 = torch.stack([p.pow(2).mean() for p in self.residual.parameters()]).mean()
        bias_l2 = 0.5 * (
            self.lead_bias.weight.pow(2).mean() + self.month_bias.weight.pow(2).mean()
        )
        scale_l2 = (self.physics_scale - 1.0).pow(2)
        return {
            "residual_l2": residual_l2,
            "bias_l2": bias_l2,
            "scale_l2": scale_l2,
        }


class TemporalResidualNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = int(input_size)
        for layer_index in range(int(num_layers)):
            layers.append(
                TemporalBlock(
                    in_channels,
                    int(hidden_size),
                    int(kernel_size),
                    dilation=2**layer_index,
                    dropout=float(dropout),
                )
            )
            in_channels = int(hidden_size)
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), max(16, int(hidden_size) // 2)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(16, int(hidden_size) // 2), 1),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(
                f"Temporal residual expects [batch, time, feature], got {features.shape}"
            )
        hidden = self.network(features.transpose(1, 2))
        return self.head(hidden[:, :, -1])


class PerTurbineResidualTCNPINN(PerTurbineResidualPINN):
    """Physical-anchor PINN whose bounded residual is a causal TCN."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        residual_amplitude: float = 0.15,
        max_lead_hour: int = 240,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            residual_amplitude=residual_amplitude,
            max_lead_hour=max_lead_hour,
        )
        self.residual = TemporalResidualNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
