import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EMA_WINDOW = 24  # K: truncation of the infinite EMA sum, see docs/pinn_plan.md section 3


class CAllNet(nn.Module):
    """C_all(v): MLP mapping wind speed -> physical coefficient over the full cut-in..
    cut-out domain, hard-bounded to [0, c_max] via a scaled sigmoid so it can never
    violate the empirical Betz ceiling on its own (the small periodic corrections added
    on top can still push the *total* C_eff slightly over -- that's what the separate
    L_Betz loss term guards against)."""

    def __init__(self, c_max, hidden=32):
        super().__init__()
        self.c_max = c_max
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, v):
        # v: (...,) -> (..., 1) -> (...,)
        raw = self.net(v.unsqueeze(-1))
        return self.c_max * torch.sigmoid(raw).squeeze(-1)


class HarmonicTerm(nn.Module):
    """1st-harmonic periodic correction: a*sin(2*pi*x/period) + b*cos(2*pi*x/period).
    Used for both g_doy (period=365) and g_moy (period=12). Initialized near zero so it
    starts as a negligible perturbation on top of C_all."""

    def __init__(self, period):
        super().__init__()
        self.period = period
        self.a = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        angle = 2 * math.pi * x / self.period
        return self.a * torch.sin(angle) + self.b * torch.cos(angle)


class ResponseTime(nn.Module):
    """Learnable turbine response time constant tau (hours), reparameterized through
    softplus so it stays positive, then converted to the EMA decay rate alpha."""

    def __init__(self, init_tau=2.0):
        super().__init__()
        # softplus(theta) = init_tau  =>  theta = log(exp(init_tau) - 1)
        init_theta = math.log(math.expm1(init_tau))
        self.theta = nn.Parameter(torch.tensor(float(init_theta)))

    @property
    def tau(self):
        return F.softplus(self.theta)

    @property
    def alpha(self):
        return 1 - torch.exp(-1.0 / self.tau)


class WindDistribution(nn.Module):
    """Convert forecast grid spread into a per-hour wind-speed distribution width.

    The LDAPS 16-grid standard deviation is only a proxy for turbine-to-turbine wind
    spread, so learn a floor plus a positive scale multiplier instead of trusting it
    literally.
    """

    def __init__(self, init_floor=0.5, init_scale=1.0):
        super().__init__()
        self.floor_theta = nn.Parameter(torch.tensor(float(math.log(math.expm1(init_floor)))))
        self.scale_theta = nn.Parameter(torch.tensor(float(math.log(math.expm1(init_scale)))))

    @property
    def floor(self):
        return F.softplus(self.floor_theta)

    @property
    def scale(self):
        return F.softplus(self.scale_theta)

    def forward(self, v_std):
        return self.floor + self.scale * torch.clamp(v_std, min=0)


def ema_smooth(p_phys, alpha, k_max=EMA_WINDOW):
    """Exact convolution solution of tau*dP/dt + P = p_phys(v(t)):
    P_t = sum_{k=0}^{k_max} alpha*(1-alpha)^k * p_phys[t-k].
    The infinite tail beyond k_max is folded into the oldest available lag, which
    preserves steady-state gain while keeping the fixed 24-hour window.
    p_phys: 1D tensor, chronologically ordered with no gaps. Values before the start
    of the sequence are approximated by replicating the first value."""
    k = torch.arange(k_max + 1, device=p_phys.device, dtype=p_phys.dtype)
    w = alpha * (1 - alpha) ** k  # w[0] = alpha (largest, weight on the current step)
    w = w.clone()
    w[-1] = (1 - alpha) ** k_max

    padded = F.pad(p_phys.view(1, 1, -1), (k_max, 0), mode="replicate").view(-1)
    windows = padded.unfold(0, k_max + 1, 1)  # (T, k_max+1); windows[t, j] = p_phys[t + j - k_max]
    w_rev = w.flip(0)  # windows[t, j] should be weighted by w[k_max - j]
    return (windows * w_rev.view(1, -1)).sum(dim=1)


class TurbineGroupBias(nn.Module):
    """Residual bias for one KPX group.

    hod_bias and moy_bias are used for train/validation/test because hour-of-day and
    month-of-year repeat. hour_bias and year_bias are optional train-only residual
    absorption terms; they are deliberately ignored whenever their indices are not
    supplied (validation/test). All bias terms are parameterized in capacity-normalized
    units and converted to kWh in forward calls."""

    def __init__(self, capacity, n_train_rows=None, n_train_years=None):
        super().__init__()
        self.register_buffer("capacity", torch.tensor(float(capacity)))
        self.hod_bias = nn.Embedding(24, 1)
        self.moy_bias = nn.Embedding(12, 1)
        nn.init.zeros_(self.hod_bias.weight)
        nn.init.zeros_(self.moy_bias.weight)
        self.hour_bias = nn.Embedding(n_train_rows, 1) if n_train_rows is not None else None
        self.year_bias = nn.Embedding(n_train_years, 1) if n_train_years is not None else None
        if self.hour_bias is not None:
            nn.init.zeros_(self.hour_bias.weight)
        if self.year_bias is not None:
            nn.init.zeros_(self.year_bias.weight)

    def calendar(self, hod, moy=None):
        out = self.hod_bias(hod).squeeze(-1)
        if moy is not None:
            out = out + self.moy_bias(moy - 1).squeeze(-1)
        return out * self.capacity

    def train_only(self, row_idx):
        if self.hour_bias is None:
            raise RuntimeError("train_only bias was requested but hour_bias is not initialized")
        return self.hour_bias(row_idx).squeeze(-1) * self.capacity

    def train_year(self, year_idx):
        if self.year_bias is None:
            raise RuntimeError("year bias was requested but year_bias is not initialized")
        return self.year_bias(year_idx).squeeze(-1) * self.capacity


class PowerCurvePINN(nn.Module):
    """One manufacturer's shared physics backbone: C_all(v) + g_doy + g_moy, response
    time tau/alpha, assembled into P_phys via the governing equation. Bias terms live
    separately per KPX group (see TurbineGroupBias) since they absorb group-specific
    noise, not manufacturer-level aerodynamics.

    This is the best-scoring configuration found this session (time-holdout score
    ~0.548/0.550/0.505 for group_1/2/3 with tuned lambdas + 2000-epoch bias stage) --
    restored here after a later architectural experiment (closed-form rated region,
    hod-only bias, MSE training loss) failed to beat it. See docs/pinn_plan.md section
    11 for the full experiment log of what was tried and why it didn't pan out yet."""

    def __init__(self, c_max, area):
        super().__init__()
        self.area = area
        self.c_all = CAllNet(c_max)
        self.g_doy = HarmonicTerm(period=365.0)
        self.g_moy = HarmonicTerm(period=12.0)
        self.response = ResponseTime()
        self.wind_dist = WindDistribution()

    def c_eff(self, v, doy, moy):
        return self.c_all(v) + self.g_doy(doy) + self.g_moy(moy)

    def p_phys(self, v, rho, doy, moy):
        return 0.5 * rho * self.area * v**3 * self.c_eff(v, doy, moy)

    def expected_p_phys(self, v, rho, doy, moy, v_std):
        sigma = self.wind_dist(v_std)
        # 5-point Gauss-Hermite quadrature for E[f(v + sigma*Z)], Z~N(0,1).
        nodes = torch.tensor(
            [-2.0201828704560856, -0.9585724646138185, 0.0, 0.9585724646138185, 2.0201828704560856],
            device=v.device,
            dtype=v.dtype,
        )
        weights = torch.tensor(
            [0.01995324205904591, 0.3936193231522412, 0.9453087204829419, 0.3936193231522412, 0.01995324205904591],
            device=v.device,
            dtype=v.dtype,
        ) / math.sqrt(math.pi)
        v_q = torch.clamp(v.unsqueeze(-1) + math.sqrt(2.0) * sigma.unsqueeze(-1) * nodes, min=0)
        p_q = self.p_phys(v_q, rho.unsqueeze(-1), doy.unsqueeze(-1), moy.unsqueeze(-1))
        return (p_q * weights.view(1, -1)).sum(dim=1)

    def forward(self, v, rho, doy, moy, v_std=None):
        """Inputs are chronologically ordered 1D tensors with no gaps."""
        if v_std is None:
            p = self.p_phys(v, rho, doy, moy)
        else:
            p = self.expected_p_phys(v, rho, doy, moy, v_std)
        return ema_smooth(p, self.response.alpha)
