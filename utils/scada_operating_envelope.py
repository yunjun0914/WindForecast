from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


class OperatingEnvelopeFitError(ValueError):
    """Raised when SCADA cannot support a stable operating envelope estimate."""


@dataclass(frozen=True)
class OperatingEnvelope:
    cut_in_speed: float
    cut_out_speed: float
    cut_out_detected: bool
    active_power_ratio: float
    crossing_probability: float
    n_observations: int
    n_high_wind: int
    bin_centers: np.ndarray
    bin_counts: np.ndarray
    active_probability: np.ndarray
    lower_probability: np.ndarray
    upper_probability: np.ndarray


def _interpolated_crossing(
    x: np.ndarray,
    y: np.ndarray,
    level: float,
    *,
    increasing: bool,
) -> float | None:
    crossed = y >= level if increasing else y <= level
    indices = np.flatnonzero(crossed)
    if len(indices) == 0:
        return None
    index = int(indices[0])
    if index == 0:
        # For the upper envelope, an already-low first high-wind bin does not
        # demonstrate a falling transition inside the observed tail.
        return float(x[index]) if increasing else None
    x0, x1 = float(x[index - 1]), float(x[index])
    y0, y1 = float(y[index - 1]), float(y[index])
    if abs(y1 - y0) <= 1e-12:
        return x1
    fraction = float(np.clip((level - y0) / (y1 - y0), 0.0, 1.0))
    return x0 + fraction * (x1 - x0)


def fit_operating_envelope(
    wind_speed: np.ndarray,
    power_ratio: np.ndarray,
    *,
    bin_width: float = 0.5,
    min_bin_count: int = 30,
    active_power_ratio: float = 0.01,
    crossing_probability: float = 0.50,
    lower_search_max: float = 10.0,
    upper_search_min: float = 15.0,
    max_speed: float = 35.0,
) -> OperatingEnvelope:
    """Estimate model-specific cut-in/out speeds from SCADA operating probability.

    A row is considered operating when its 10-minute power is at least
    ``active_power_ratio`` of rated 10-minute energy. The lower envelope is fitted
    with an increasing isotonic curve and the upper envelope with a decreasing one.
    Cut-in/out are the corresponding probability crossings. A cut-out is deliberately
    left undetected when the observed high-wind tail never crosses the requested
    probability; callers should not silently replace it with a global constant.
    """

    wind = np.asarray(wind_speed, dtype=float).reshape(-1)
    power = np.asarray(power_ratio, dtype=float).reshape(-1)
    if wind.shape != power.shape:
        raise ValueError("wind_speed and power_ratio must have the same shape")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    if min_bin_count < 1:
        raise ValueError("min_bin_count must be positive")
    if not 0.0 < active_power_ratio < 1.0:
        raise ValueError("active_power_ratio must be in (0, 1)")
    if not 0.0 < crossing_probability < 1.0:
        raise ValueError("crossing_probability must be in (0, 1)")
    if not 0.0 < lower_search_max < upper_search_min < max_speed:
        raise ValueError("search ranges must satisfy 0 < lower_max < upper_min < max")

    valid = (
        np.isfinite(wind)
        & np.isfinite(power)
        & (wind >= 0.0)
        & (wind <= max_speed)
        & (power >= 0.0)
    )
    wind = wind[valid]
    power = np.clip(power[valid], 0.0, 1.0)
    if len(wind) < max(500, 8 * min_bin_count):
        raise OperatingEnvelopeFitError(
            f"Too few finite SCADA observations: n={len(wind)}"
        )

    edges = np.arange(0.0, max_speed + bin_width, bin_width, dtype=float)
    if edges[-1] < max_speed:
        edges = np.append(edges, max_speed)
    bin_index = np.clip(np.digitize(wind, edges, right=False) - 1, 0, len(edges) - 2)
    active = power >= active_power_ratio
    counts = np.bincount(bin_index, minlength=len(edges) - 1).astype(float)
    active_counts = np.bincount(
        bin_index, weights=active.astype(float), minlength=len(edges) - 1
    )
    centers_all = 0.5 * (edges[:-1] + edges[1:])
    supported = counts >= float(min_bin_count)
    centers = centers_all[supported]
    counts = counts[supported]
    probability = active_counts[supported] / counts
    if len(centers) < 8:
        raise OperatingEnvelopeFitError("Too few supported SCADA wind bins")

    lower_keep = centers <= lower_search_max
    upper_keep = centers >= upper_search_min
    if int(lower_keep.sum()) < 4:
        raise OperatingEnvelopeFitError("Too few supported bins for cut-in")
    lower_fit = IsotonicRegression(
        increasing=True,
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    ).fit_transform(
        centers[lower_keep],
        probability[lower_keep],
        sample_weight=counts[lower_keep],
    )
    cut_in = _interpolated_crossing(
        centers[lower_keep],
        np.asarray(lower_fit, dtype=float),
        crossing_probability,
        increasing=True,
    )
    if cut_in is None:
        raise OperatingEnvelopeFitError(
            "SCADA operating probability never crosses the cut-in level"
        )

    lower_probability = np.full(len(centers), np.nan, dtype=float)
    lower_probability[lower_keep] = lower_fit
    upper_probability = np.full(len(centers), np.nan, dtype=float)
    cut_out = np.nan
    cut_out_detected = False
    if int(upper_keep.sum()) >= 4:
        upper_fit = IsotonicRegression(
            increasing=False,
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        ).fit_transform(
            centers[upper_keep],
            probability[upper_keep],
            sample_weight=counts[upper_keep],
        )
        upper_probability[upper_keep] = upper_fit
        crossing = _interpolated_crossing(
            centers[upper_keep],
            np.asarray(upper_fit, dtype=float),
            crossing_probability,
            increasing=False,
        )
        if crossing is not None:
            cut_out = float(crossing)
            cut_out_detected = True

    n_high_wind = int(np.sum(wind >= upper_search_min))
    return OperatingEnvelope(
        cut_in_speed=float(cut_in),
        cut_out_speed=float(cut_out),
        cut_out_detected=bool(cut_out_detected),
        active_power_ratio=float(active_power_ratio),
        crossing_probability=float(crossing_probability),
        n_observations=int(len(wind)),
        n_high_wind=n_high_wind,
        bin_centers=np.asarray(centers, dtype=float),
        bin_counts=np.asarray(counts, dtype=float),
        active_probability=np.asarray(probability, dtype=float),
        lower_probability=lower_probability,
        upper_probability=upper_probability,
    )


def soft_operating_gate(
    wind_speed: np.ndarray,
    cut_in_speed: float,
    cut_out_speed: float = np.nan,
    *,
    tau_in: float = 1.0,
    tau_out: float = 1.0,
    include_cut_in: bool = True,
    include_cut_out: bool = True,
) -> np.ndarray:
    """Return a smooth train/test-observable operating gate from forecast wind."""

    if tau_in <= 0 or tau_out <= 0:
        raise ValueError("tau_in and tau_out must be positive")
    wind = np.asarray(wind_speed, dtype=float)
    gate = np.ones_like(wind, dtype=float)
    if include_cut_in:
        lower_logit = np.clip((wind - float(cut_in_speed)) / tau_in, -60.0, 60.0)
        gate *= 1.0 / (1.0 + np.exp(-lower_logit))
    if include_cut_out and np.isfinite(cut_out_speed):
        upper_logit = np.clip((float(cut_out_speed) - wind) / tau_out, -60.0, 60.0)
        gate *= 1.0 / (1.0 + np.exp(-upper_logit))
    gate[~np.isfinite(wind)] = np.nan
    return gate
