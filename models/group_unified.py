from __future__ import annotations

import math

import torch
from torch import nn

from models.seqnn import TemporalBlock


def soft_unit_price(error_rate: torch.Tensor, gamma: float) -> torch.Tensor:
    return 4.0 - torch.sigmoid((error_rate - 0.06) / gamma) - 3.0 * torch.sigmoid(
        (error_rate - 0.08) / gamma
    )


def normalized_metric_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gamma: float,
    nmae_weight: float,
    ficr_weight: float,
    min_output_ratio: float = 0.10,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(target) & (target >= min_output_ratio)
    if not bool(valid.any()):
        zero = prediction.sum() * 0.0
        return zero, zero.detach(), zero.detach()

    pred = prediction[valid]
    actual = target[valid]
    error = torch.abs(pred - actual)
    if sample_weight is None:
        nmae = error.mean()
    else:
        weight = sample_weight[valid]
        nmae = (error * weight).sum() / weight.sum().clamp_min(1e-6)

    price = soft_unit_price(error, gamma)
    ficr = (actual * price).sum() / (actual * 4.0).sum().clamp_min(1e-6)
    loss = nmae_weight * nmae + ficr_weight * (1.0 - ficr)
    return loss, nmae.detach(), ficr.detach()


class GroupPhysicsPINN(nn.Module):
    """Group power curve with a bounded weather residual around first principles."""

    def __init__(
        self,
        n_turbines: int,
        rotor_area_m2: float,
        rated_power_w: float,
        c_max: float,
        hidden_size: int = 48,
        residual_amplitude: float = 0.12,
    ) -> None:
        super().__init__()
        self.n_turbines = int(n_turbines)
        self.rotor_area_m2 = float(rotor_area_m2)
        self.rated_power_w = float(rated_power_w)
        self.c_max = float(c_max)
        self.residual_amplitude = float(residual_amplitude)
        self.c_eff_net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.residual_net = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.Tanh(),
            nn.Linear(max(16, hidden_size // 2), 1),
        )
        nn.init.normal_(self.residual_net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_net[-1].bias)

    def c_eff(self, wind_speed: torch.Tensor) -> torch.Tensor:
        normalized = wind_speed.unsqueeze(-1) / 25.0
        return self.c_max * torch.sigmoid(self.c_eff_net(normalized).squeeze(-1))

    def physical_ratio(
        self,
        wind_speed: torch.Tensor,
        air_density: torch.Tensor,
    ) -> torch.Tensor:
        coefficient = self.c_eff(wind_speed)
        physical_w = (
            0.5
            * air_density
            * (self.n_turbines * self.rotor_area_m2)
            * wind_speed.clamp_min(0.0).pow(3)
            * coefficient
        )
        operating = (wind_speed >= 3.5) & (wind_speed < 25.0)
        physical_w = torch.where(operating, physical_w, torch.zeros_like(physical_w))
        return torch.clamp(physical_w / self.rated_power_w, 0.0, 1.0)

    def forward(
        self,
        teacher: torch.Tensor,
        air_density: torch.Tensor,
        doy: torch.Tensor,
        lead_hour: torch.Tensor,
        return_parts: bool = False,
    ):
        wind_speed = teacher[:, 0].clamp(0.0, 35.0)
        wind_std = teacher[:, 1].clamp(0.0, 15.0)
        wd_sin = teacher[:, 2].clamp(-1.0, 1.0)
        wd_cos = teacher[:, 3].clamp(-1.0, 1.0)
        physical = self.physical_ratio(wind_speed, air_density)
        angle = 2.0 * math.pi * doy / 365.0
        residual_features = torch.stack(
            [
                wind_speed / 25.0,
                wind_std / 8.0,
                air_density - 1.2,
                wd_sin,
                wd_cos,
                torch.sin(angle),
                torch.cos(angle),
                lead_hour / 36.0,
                physical,
            ],
            dim=1,
        )
        residual = self.residual_amplitude * torch.tanh(
            self.residual_net(residual_features).squeeze(-1)
        )
        prediction = torch.clamp(physical + residual, 0.0, 1.0)
        if return_parts:
            return prediction, {"physical": physical, "residual": residual}
        return prediction

    def physics_regularization(self, device: torch.device) -> dict[str, torch.Tensor]:
        wind = torch.linspace(0.0, 30.0, 121, device=device, requires_grad=True)
        coefficient = self.c_eff(wind)
        first = torch.autograd.grad(coefficient.sum(), wind, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), wind, create_graph=True)[0]
        boundary = self.c_eff(torch.tensor([3.5, 25.0], device=device)).pow(2).mean()
        smoothness = second.pow(2).mean()

        rated_wind = torch.linspace(12.0, 24.5, 48, device=device, requires_grad=True)
        rho = torch.full_like(rated_wind, 1.2)
        rated_power = self.physical_ratio(rated_wind, rho)
        rated_grad = torch.autograd.grad(rated_power.sum(), rated_wind, create_graph=True)[0]
        flatness = rated_grad.pow(2).mean()
        residual_l2 = torch.stack(
            [parameter.pow(2).mean() for parameter in self.residual_net.parameters()]
        ).mean()
        return {
            "boundary": boundary,
            "smoothness": smoothness,
            "flatness": flatness,
            "residual_l2": residual_l2,
        }


class BoundedResidualMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        dropout: float = 0.10,
        max_delta: float = 0.12,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(16, hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, hidden_size // 2), 1),
        )
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.max_delta * torch.tanh(self.network(features).squeeze(-1))


class MultiHeadTCNPowerRegressor(nn.Module):
    """Shared temporal encoder with one direct output head per group."""

    def __init__(
        self,
        input_size: int,
        n_groups: int = 3,
        hidden_size: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.n_groups = int(n_groups)

        layers = []
        in_channels = int(input_size)
        for layer_index in range(num_layers):
            layers.append(
                TemporalBlock(
                    in_channels,
                    hidden_size,
                    kernel_size,
                    dilation=2**layer_index,
                    dropout=dropout,
                )
            )
            in_channels = hidden_size
        self.network = nn.Sequential(*layers)
        head_size = max(16, hidden_size // 2)
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, head_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_size, 1),
                )
                for _ in range(self.n_groups)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.network(features.transpose(1, 2))[:, :, -1]
        return torch.cat([head(encoded) for head in self.heads], dim=1)
