from __future__ import annotations

import numpy as np


BIN_BOUNDARIES = np.asarray([0.25, 0.50, 0.75], dtype=float)
N_BINS = len(BIN_BOUNDARIES) + 1


def capacity_bin_indices(values, capacity: float) -> np.ndarray:
    ratio = np.asarray(values, dtype=float) / float(capacity)
    bins = np.digitize(ratio, BIN_BOUNDARIES, right=False)
    return np.clip(bins, 0, N_BINS - 1).astype(np.int64)


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
