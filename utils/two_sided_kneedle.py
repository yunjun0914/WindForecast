from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


class KneedleFitError(ValueError):
    """Raised when a stable lower/upper knee pair cannot be identified."""


@dataclass(frozen=True)
class TwoSidedKneedleResult:
    lower_wind: float
    upper_wind: float
    lower_power_ratio: float
    upper_power_ratio: float
    wind_bins: np.ndarray
    raw_power_bins: np.ndarray
    monotone_power_bins: np.ndarray
    difference: np.ndarray
    bin_counts: np.ndarray
    lower_index: int
    upper_index: int


def _equal_frequency_bins(
    wind_speed: np.ndarray,
    power_ratio: np.ndarray,
    n_bins: int,
    min_bin_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(wind_speed, kind="mergesort")
    wind = wind_speed[order]
    power = power_ratio[order]
    effective_bins = min(int(n_bins), len(wind) // int(min_bin_count))
    if effective_bins < 8:
        raise KneedleFitError(
            f"Too few observations for two-sided Kneedle: n={len(wind)}"
        )

    wind_bins = []
    power_bins = []
    counts = []
    for indices in np.array_split(np.arange(len(wind)), effective_bins):
        if len(indices) == 0:
            continue
        wind_bins.append(float(np.median(wind[indices])))
        power_bins.append(float(np.median(power[indices])))
        counts.append(int(len(indices)))

    wind_bins = np.asarray(wind_bins, dtype=float)
    power_bins = np.asarray(power_bins, dtype=float)
    counts = np.asarray(counts, dtype=float)

    # Repeated wind medians can occur at calm conditions. Collapse them before
    # isotonic fitting so the normalized x-axis is strictly increasing.
    unique_wind, inverse = np.unique(wind_bins, return_inverse=True)
    if len(unique_wind) != len(wind_bins):
        collapsed_power = np.zeros(len(unique_wind), dtype=float)
        collapsed_counts = np.zeros(len(unique_wind), dtype=float)
        for index in range(len(unique_wind)):
            keep = inverse == index
            collapsed_power[index] = float(
                np.average(power_bins[keep], weights=counts[keep])
            )
            collapsed_counts[index] = float(counts[keep].sum())
        wind_bins = unique_wind
        power_bins = collapsed_power
        counts = collapsed_counts

    if len(wind_bins) < 8 or wind_bins[-1] <= wind_bins[0]:
        raise KneedleFitError("Wind bins do not span a usable increasing range")
    return wind_bins, power_bins, counts


def fit_two_sided_kneedle(
    wind_speed: np.ndarray,
    power_ratio: np.ndarray,
    *,
    n_bins: int = 64,
    min_bin_count: int = 30,
    min_knee_bins: int = 3,
) -> TwoSidedKneedleResult:
    """Fit the lower and upper knees of an empirical S-shaped power curve.

    The empirical curve is equal-frequency binned, made monotone with weighted
    isotonic regression, and normalized to the unit square. For an increasing
    S-curve, ``y - x`` reaches its minimum at the lower knee and its maximum at
    the upper knee.
    """

    wind = np.asarray(wind_speed, dtype=float).reshape(-1)
    power = np.asarray(power_ratio, dtype=float).reshape(-1)
    if wind.shape != power.shape:
        raise ValueError("wind_speed and power_ratio must have the same shape")
    valid = np.isfinite(wind) & np.isfinite(power) & (wind >= 0.0)
    wind = wind[valid]
    power = np.clip(power[valid], 0.0, 1.0)
    if len(wind) < max(8 * min_bin_count, 200):
        raise KneedleFitError(
            f"Too few finite observations for two-sided Kneedle: n={len(wind)}"
        )

    wind_bins, raw_power_bins, counts = _equal_frequency_bins(
        wind,
        power,
        n_bins=n_bins,
        min_bin_count=min_bin_count,
    )
    monotone = IsotonicRegression(
        increasing=True,
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    ).fit_transform(wind_bins, raw_power_bins, sample_weight=counts)

    wind_span = float(wind_bins[-1] - wind_bins[0])
    power_span = float(monotone[-1] - monotone[0])
    if wind_span <= 1e-8 or power_span <= 1e-4:
        raise KneedleFitError("Empirical power curve has insufficient dynamic range")
    x_norm = (wind_bins - wind_bins[0]) / wind_span
    y_norm = (monotone - monotone[0]) / power_span
    difference = y_norm - x_norm

    # End points lie on the reference chord by construction and cannot be knees.
    interior = np.arange(1, len(wind_bins) - 1)
    if len(interior) < 2 * min_knee_bins + 1:
        raise KneedleFitError("Too few interior bins for two ordered knees")
    lower_index = int(interior[np.argmin(difference[interior])])
    upper_index = int(interior[np.argmax(difference[interior])])
    if upper_index - lower_index < int(min_knee_bins):
        raise KneedleFitError(
            "Two-sided Kneedle did not find an ordered, separated knee pair"
        )
    if difference[lower_index] >= 0.0 or difference[upper_index] <= 0.0:
        raise KneedleFitError(
            "Empirical curve does not cross both sides of its reference chord"
        )

    return TwoSidedKneedleResult(
        lower_wind=float(wind_bins[lower_index]),
        upper_wind=float(wind_bins[upper_index]),
        lower_power_ratio=float(monotone[lower_index]),
        upper_power_ratio=float(monotone[upper_index]),
        wind_bins=wind_bins,
        raw_power_bins=raw_power_bins,
        monotone_power_bins=np.asarray(monotone, dtype=float),
        difference=np.asarray(difference, dtype=float),
        bin_counts=np.asarray(counts, dtype=float),
        lower_index=lower_index,
        upper_index=upper_index,
    )


def kneedle_mid_mask(
    wind_speed: np.ndarray,
    result: TwoSidedKneedleResult,
) -> np.ndarray:
    wind = np.asarray(wind_speed, dtype=float)
    return (
        np.isfinite(wind)
        & (wind >= result.lower_wind)
        & (wind <= result.upper_wind)
    )
