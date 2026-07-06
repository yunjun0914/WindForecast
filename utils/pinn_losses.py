import torch

from utils.pinn_data import CUT_IN_SPEED, CUT_OUT_SPEED

GAMMA = 0.01  # fixed sigmoid temperature for the FICR price-step relaxation (not learned)


def soft_unit_price(error_rate, gamma=GAMMA):
    """Differentiable relaxation of the step-function settlement price
    (<=6%: 4.0, <=8%: 3.0, else: 0.0)."""
    return 4.0 - torch.sigmoid((error_rate - 0.06) / gamma) - 3.0 * torch.sigmoid((error_rate - 0.08) / gamma)


def data_loss(y, p_hat, capacity, min_output_ratio=0.10, gamma=GAMMA):
    """0.5*NMAE + 0.5*(1-FICR_soft), restricted to hours with actual output >= 10% of
    capacity (matches the official metric's own filter). Equivalent (up to a sign flip
    and constant offset) to maximizing the official total_score directly.

    (Plain MSE was tried instead -- see docs/pinn_plan.md 11.4 -- and scored lower:
    physics_only dropped from ~0.548/0.550/0.505 to ~0.538/0.518/0.498, since MSE
    optimizes a different objective than what's actually being scored. Reverted.)"""
    valid = y >= capacity * min_output_ratio
    y_v, p_v = y[valid], p_hat[valid]
    error_rate = torch.abs(p_v - y_v) / capacity

    l_nmae = error_rate.mean()

    price = soft_unit_price(error_rate, gamma=gamma)
    ficr_soft = (y_v * price).sum() / (y_v * 4.0).sum()
    l_ficr = 1 - ficr_soft

    loss = 0.5 * l_nmae + 0.5 * l_ficr
    return loss, l_nmae.detach(), ficr_soft.detach()


def betz_loss(model, v_c, doy_c, moy_c, c_max):
    """Penalize the *total* C_eff (base + periodic corrections) for exceeding the
    empirical Betz ceiling -- C_all alone is already hard-bounded by its sigmoid output
    activation, but the small doy/moy perturbations added on top could still push the
    sum slightly over."""
    c_eff = model.c_eff(v_c, doy_c, moy_c)
    return torch.clamp(c_eff - c_max, min=0).pow(2).mean()


def boundary_condition_loss(model, doy_c, moy_c):
    """C_eff should be ~0 right at cut-in and cut-out (no power generated there)."""
    v_cutin = torch.full_like(doy_c, CUT_IN_SPEED)
    v_cutout = torch.full_like(doy_c, CUT_OUT_SPEED)
    return model.c_eff(v_cutin, doy_c, moy_c).pow(2).mean() + model.c_eff(v_cutout, doy_c, moy_c).pow(2).mean()


def flatness_loss(model, v_c, rho_c, doy_c, moy_c, turbine_capacity_w):
    """dP/dv should be ~0 in the rated-to-cutout range (pitch control caps output).
    v_c must already be restricted to that range by the caller. p_phys is in watts
    (megawatt-scale per turbine), so its v-derivative squared would otherwise dwarf
    every other loss term -- normalize by the single-turbine rated capacity first to
    keep this on the same O(1) scale as the rest of the loss."""
    v_c = v_c.clone().requires_grad_(True)
    p = model.p_phys(v_c, rho_c, doy_c, moy_c) / turbine_capacity_w
    (grad,) = torch.autograd.grad(p.sum(), v_c, create_graph=True)
    return grad.pow(2).mean()


def smoothness_loss(model, v_c):
    """Curvature penalty on C_all(v) -- discourage unphysical wiggles, especially where
    training data is sparse (very low/high wind speed)."""
    v_c = v_c.clone().requires_grad_(True)
    c = model.c_all(v_c)
    (grad1,) = torch.autograd.grad(c.sum(), v_c, create_graph=True)
    (grad2,) = torch.autograd.grad(grad1.sum(), v_c, create_graph=True)
    return grad2.pow(2).mean()


def bias_l2(bias_module):
    """Diagnostic-only: mean squared magnitude of the hod_bias embedding, for logging how
    large the learned correction is. Not part of the training loss -- the actual
    shrinkage is applied via AdamW's decoupled weight_decay (see train_pinn.py)."""
    out = {
        "hod": bias_module.hod_bias.weight.pow(2).mean(),
        "moy": bias_module.moy_bias.weight.pow(2).mean(),
    }
    if bias_module.hour_bias is not None:
        out["hour"] = bias_module.hour_bias.weight.pow(2).mean()
    if bias_module.year_bias is not None:
        out["year"] = bias_module.year_bias.weight.pow(2).mean()
    return out
