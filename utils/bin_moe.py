from __future__ import annotations

import numpy as np
import pandas as pd


BIN_BOUNDARIES = np.asarray([0.25, 0.50, 0.75], dtype=float)
N_BINS = len(BIN_BOUNDARIES) + 1
QUARTILE_CENTERS = np.asarray([0.125, 0.375, 0.625, 0.875], dtype=float)


def capacity_bin_indices(values, capacity: float) -> np.ndarray:
    ratio = np.asarray(values, dtype=float) / float(capacity)
    bins = np.digitize(ratio, BIN_BOUNDARIES, right=False)
    return np.clip(bins, 0, N_BINS - 1).astype(np.int64)


def add_centered_weather_regime(
    table: pd.DataFrame,
    wind_col: str = "optgrid_ws_calibrated",
    output_col: str = "weather_regime_ws",
) -> pd.DataFrame:
    required = [
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "turbine_id",
        wind_col,
    ]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Weather regime input missing columns: {missing}")

    out = table.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["data_available_kst_dtm"] = pd.to_datetime(
        out["data_available_kst_dtm"]
    )
    out[wind_col] = pd.to_numeric(out[wind_col], errors="coerce")
    out = out.sort_values(
        ["turbine_id", "data_available_kst_dtm", "forecast_kst_dtm"]
    ).copy()
    grouped = out.groupby(
        ["turbine_id", "data_available_kst_dtm"], sort=False
    )[wind_col]
    out[output_col] = grouped.transform(
        lambda values: values.rolling(window=5, center=True, min_periods=1).median()
    )
    return out.reset_index(drop=True)


def fit_weather_quantile_boundaries(values) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("Cannot fit weather quantiles without finite values")
    return np.quantile(finite, BIN_BOUNDARIES).astype(float)


def weather_quantile_bins(values, boundaries) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    boundaries = np.asarray(boundaries, dtype=float)
    if boundaries.shape != BIN_BOUNDARIES.shape:
        raise ValueError(
            f"Weather boundaries must have shape {BIN_BOUNDARIES.shape}, "
            f"got {boundaries.shape}"
        )
    if np.any(~np.isfinite(boundaries)) or np.any(np.diff(boundaries) < 0):
        raise ValueError("Weather boundaries must be finite and nondecreasing")
    bins = np.digitize(values, boundaries, right=False)
    return np.clip(bins, 0, N_BINS - 1).astype(np.int64)


def empirical_weather_percentiles(values, reference) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    reference = np.sort(reference[np.isfinite(reference)])
    if len(reference) == 0:
        raise ValueError("Cannot compute weather percentiles without a reference")
    if np.any(~np.isfinite(values)):
        raise ValueError("Weather percentile values must be finite")
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return np.clip((left + right) / (2.0 * len(reference)), 0.0, 1.0)


def adjacent_quartile_weights(percentiles) -> np.ndarray:
    percentiles = np.asarray(percentiles, dtype=float)
    if percentiles.ndim != 1 or np.any(~np.isfinite(percentiles)):
        raise ValueError("Percentiles must be a finite one-dimensional array")
    clipped = np.clip(percentiles, 0.0, 1.0)
    weights = np.zeros((len(clipped), N_BINS), dtype=float)
    weights[clipped <= QUARTILE_CENTERS[0], 0] = 1.0
    weights[clipped >= QUARTILE_CENTERS[-1], -1] = 1.0
    for index in range(N_BINS - 1):
        lower = QUARTILE_CENTERS[index]
        upper = QUARTILE_CENTERS[index + 1]
        selected = (clipped >= lower) & (clipped <= upper)
        fraction = (clipped[selected] - lower) / (upper - lower)
        weights[selected, index] = 1.0 - fraction
        weights[selected, index + 1] = fraction
    return weights


def normalize_gate_probabilities(probabilities) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != N_BINS:
        raise ValueError(
            f"Gate probabilities must have shape (n, {N_BINS}), got {values.shape}"
        )
    values = np.clip(values, 0.0, None)
    denominator = values.sum(axis=1, keepdims=True)
    if np.any(denominator <= 0):
        raise ValueError("Gate probabilities contain a row with zero total mass")
    return values / denominator


def mix_expert_predictions(expert_predictions, probabilities) -> np.ndarray:
    experts = np.asarray(expert_predictions, dtype=float)
    gate = normalize_gate_probabilities(probabilities)
    if experts.shape != gate.shape:
        raise ValueError(
            f"Expert/gate shape mismatch: {experts.shape} != {gate.shape}"
        )
    return np.sum(experts * gate, axis=1)


def hard_mix_expert_predictions(expert_predictions, probabilities) -> np.ndarray:
    experts = np.asarray(expert_predictions, dtype=float)
    gate = normalize_gate_probabilities(probabilities)
    if experts.shape != gate.shape:
        raise ValueError(
            f"Expert/gate shape mismatch: {experts.shape} != {gate.shape}"
        )
    return experts[np.arange(len(experts)), gate.argmax(axis=1)]


def oracle_mix_expert_predictions(expert_predictions, actual_bins) -> np.ndarray:
    experts = np.asarray(expert_predictions, dtype=float)
    bins = np.asarray(actual_bins, dtype=np.int64)
    if experts.ndim != 2 or experts.shape[1] != N_BINS:
        raise ValueError(
            f"Expert predictions must have shape (n, {N_BINS}), got {experts.shape}"
        )
    if bins.shape != (len(experts),):
        raise ValueError(f"Actual bin shape mismatch: {bins.shape}")
    if np.any((bins < 0) | (bins >= N_BINS)):
        raise ValueError("Actual bins are outside the supported range")
    return experts[np.arange(len(experts)), bins]
