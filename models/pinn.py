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


class CAllGRUNet(nn.Module):
    """Temporal variant of C_all(v).

    The static C_all backbone is still exposed through forward(v), so the existing
    physics losses can keep probing a clean pointwise power curve. For chronological
    train/test sequences, temporal(v, ...) applies a bounded GRU modulation to the
    coefficient itself, not to the final power output.
    """

    def __init__(self, c_max, hidden=32, gru_hidden=32, window=24, temporal_scale=0.25):
        super().__init__()
        self.c_max = c_max
        self.window = int(window)
        self.temporal_scale = float(temporal_scale)
        self.static = CAllNet(c_max, hidden=hidden)
        self.gru = nn.GRU(input_size=7, hidden_size=gru_hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, v):
        return self.static(v)

    def _features(self, v, rho, doy, moy, v_std):
        if v_std is None:
            v_std = torch.zeros_like(v)
        doy_angle = 2 * math.pi * doy / 365.0
        moy_angle = 2 * math.pi * moy / 12.0
        return torch.stack(
            [
                v / 25.0,
                rho - 1.2,
                v_std / 5.0,
                torch.sin(doy_angle),
                torch.cos(doy_angle),
                torch.sin(moy_angle),
                torch.cos(moy_angle),
            ],
            dim=1,
        )

    def temporal(self, v, rho, doy, moy, v_std=None):
        if v.ndim != 1:
            return self.static(v)
        if v.numel() <= 1 or self.window <= 1:
            return self.static(v)

        base = self.static(v)
        features = self._features(v, rho, doy, moy, v_std)
        outputs = []
        h = None
        for start in range(0, features.shape[0], self.window):
            chunk = features[start : start + self.window].unsqueeze(0)
            out, h = self.gru(chunk, h)
            outputs.append(out.squeeze(0))
            h = h.detach()
        temporal_state = torch.cat(outputs, dim=0)
        modulation = torch.tanh(self.head(temporal_state).squeeze(-1))
        return torch.clamp(base * (1.0 + self.temporal_scale * modulation), 0.0, self.c_max)


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


class DirectionFourierCorrection(nn.Module):
    """Small group-specific C_eff correction from wind direction.

    Inputs are sin(theta), cos(theta), typically predicted from SCADA wind direction
    by a weather-only teacher. A tanh amplitude cap keeps this term from replacing
    the main wind-speed power curve.
    """

    def __init__(self, amplitude=0.02):
        super().__init__()
        self.amplitude = float(amplitude)
        self.coeff = nn.Parameter(torch.zeros(4))

    def forward(self, wd_sin, wd_cos):
        sin2 = 2.0 * wd_sin * wd_cos
        cos2 = wd_cos.pow(2) - wd_sin.pow(2)
        raw = (
            self.coeff[0] * wd_sin
            + self.coeff[1] * wd_cos
            + self.coeff[2] * sin2
            + self.coeff[3] * cos2
        )
        return self.amplitude * torch.tanh(raw)


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

    hod_bias, dow_bias, and moy_bias are used for train/validation/test because
    hour-of-day, day-of-week, and month-of-year repeat. hour_bias and year_bias are optional train-only residual
    absorption terms; they are deliberately ignored whenever their indices are not
    supplied (validation/test). All bias terms are parameterized in capacity-normalized
    units and converted to kWh in forward calls."""

    def __init__(
        self,
        capacity,
        n_train_rows=None,
        n_train_years=None,
        use_direction_correction=False,
        direction_amplitude=0.02,
    ):
        super().__init__()
        self.register_buffer("capacity", torch.tensor(float(capacity)))
        self.hod_bias = nn.Embedding(24, 1)
        self.dow_bias = nn.Embedding(7, 1)
        self.moy_bias = nn.Embedding(12, 1)
        nn.init.zeros_(self.hod_bias.weight)
        nn.init.zeros_(self.dow_bias.weight)
        nn.init.zeros_(self.moy_bias.weight)
        self.hour_bias = nn.Embedding(n_train_rows, 1) if n_train_rows is not None else None
        self.year_bias = nn.Embedding(n_train_years, 1) if n_train_years is not None else None
        self.direction = DirectionFourierCorrection(direction_amplitude) if use_direction_correction else None
        if self.hour_bias is not None:
            nn.init.zeros_(self.hour_bias.weight)
        if self.year_bias is not None:
            nn.init.zeros_(self.year_bias.weight)

    def calendar(self, hod, moy=None, dow=None):
        out = self.hod_bias(hod).squeeze(-1)
        if dow is not None:
            out = out + self.dow_bias(dow).squeeze(-1)
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

    def direction_c_eff(self, wd_sin, wd_cos):
        if self.direction is None:
            raise RuntimeError("direction correction was requested but it is not initialized")
        return self.direction(wd_sin, wd_cos)


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

    def c_eff(self, v, doy, moy, c_eff_add=None):
        out = self.c_all(v) + self.g_doy(doy) + self.g_moy(moy)
        if c_eff_add is not None:
            out = out + c_eff_add
        return out

    def p_phys(self, v, rho, doy, moy, c_eff_add=None):
        return 0.5 * rho * self.area * v**3 * self.c_eff(v, doy, moy, c_eff_add=c_eff_add)

    def expected_p_phys(self, v, rho, doy, moy, v_std, c_eff_add=None):
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
        add_q = c_eff_add.unsqueeze(-1) if c_eff_add is not None else None
        p_q = self.p_phys(v_q, rho.unsqueeze(-1), doy.unsqueeze(-1), moy.unsqueeze(-1), c_eff_add=add_q)
        return (p_q * weights.view(1, -1)).sum(dim=1)

    def forward(self, v, rho, doy, moy, v_std=None, c_eff_add=None):
        """Inputs are chronologically ordered 1D tensors with no gaps."""
        if v_std is None:
            p = self.p_phys(v, rho, doy, moy, c_eff_add=c_eff_add)
        else:
            p = self.expected_p_phys(v, rho, doy, moy, v_std, c_eff_add=c_eff_add)
        return ema_smooth(p, self.response.alpha)


class PowerCurveGRUPINN(PowerCurvePINN):
    """PINN with a GRU-backed temporal C_eff net.

    This keeps the original physical scaffold, EMA response, wind-distribution moment,
    and bias path intact. The only change is that chronological forward passes use a
    temporal C_all coefficient, while collocation losses still see the static curve.
    """

    def __init__(self, c_max, area, gru_hidden=32, window=24, temporal_scale=0.25):
        super().__init__(c_max, area)
        self.c_all = CAllGRUNet(
            c_max,
            hidden=32,
            gru_hidden=gru_hidden,
            window=window,
            temporal_scale=temporal_scale,
        )

    def c_eff_temporal(self, v, rho, doy, moy, v_std=None, c_eff_add=None):
        out = self.c_all.temporal(v, rho, doy, moy, v_std) + self.g_doy(doy) + self.g_moy(moy)
        if c_eff_add is not None:
            out = out + c_eff_add
        return out

    def p_phys_temporal(self, v, rho, doy, moy, v_std=None, c_eff_add=None):
        return 0.5 * rho * self.area * v**3 * self.c_eff_temporal(
            v, rho, doy, moy, v_std, c_eff_add=c_eff_add
        )

    def expected_p_phys_temporal(self, v, rho, doy, moy, v_std, c_eff_add=None):
        sigma = self.wind_dist(v_std)
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
        c_eff = self.c_eff_temporal(v, rho, doy, moy, v_std, c_eff_add=c_eff_add).unsqueeze(-1)
        p_q = 0.5 * rho.unsqueeze(-1) * self.area * v_q**3 * c_eff
        return (p_q * weights.view(1, -1)).sum(dim=1)

    def forward(self, v, rho, doy, moy, v_std=None, c_eff_add=None):
        if v_std is None:
            p = self.p_phys_temporal(v, rho, doy, moy, c_eff_add=c_eff_add)
        else:
            p = self.expected_p_phys_temporal(v, rho, doy, moy, v_std, c_eff_add=c_eff_add)
        return ema_smooth(p, self.response.alpha)
