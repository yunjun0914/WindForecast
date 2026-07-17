from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
import experiments.evaluate_group_decision_q_tcn_oof as baseline
from utils.tcn_scada_cutin_features import (
    GROUP_MODEL,
    attach_cutin_features,
    build_cutin_group_panel,
    fit_model_cutin,
)


TEACHER_CACHE_ROOT = Path(
    os.environ.get("SCADA_CUTIN_TEACHER_CACHE_ROOT", "cache/per_turbine_teacher_v1")
)
TEACHER_CACHE_TAG = os.environ.get(
    "SCADA_CUTIN_TEACHER_CACHE_TAG", "optimal_grid_replace_local16_v1"
)
CUTIN_TAU = float(os.environ.get("SCADA_CUTIN_TAU", "1.0"))
DATA_ROOT = Path(os.environ.get("SCADA_CUTIN_DATA_ROOT", "data/train"))

_ORIGINAL_BUILD_FOLD_FEATURES = baseline.build_fold_features
_ENVELOPE_CACHE: dict[tuple[str, tuple[int, ...]], object] = {}
_SCADA_CACHE: dict[str, pd.DataFrame] = {}


def _scada_for_model(model: str) -> pd.DataFrame:
    if model not in _SCADA_CACHE:
        file_name = (
            "scada_vestas_train.csv"
            if model == "vestas"
            else "scada_unison_train.csv"
        )
        _SCADA_CACHE[model] = pd.read_csv(
            DATA_ROOT / file_name,
            encoding="utf-8-sig",
        )
    return _SCADA_CACHE[model]


def _fold_envelope(group: str, train_years: list[int]):
    model = GROUP_MODEL[group]
    key = model, tuple(sorted(map(int, train_years)))
    if key not in _ENVELOPE_CACHE:
        envelope = fit_model_cutin(_scada_for_model(model), model, list(key[1]))
        _ENVELOPE_CACHE[key] = envelope
        print(
            f"SCADA cut-in model={model} train_years={list(key[1])} "
            f"speed={envelope.cut_in_speed:.4f}m/s n={envelope.n_observations}",
            flush=True,
        )
    return _ENVELOPE_CACHE[key]


def build_fold_features_with_cutin(
    base_features,
    candidates,
    targets,
    group,
    pred_year,
    train_years,
    args,
):
    features, feature_cols, selections = _ORIGINAL_BUILD_FOLD_FEATURES(
        base_features,
        candidates,
        targets,
        group,
        pred_year,
        train_years,
        args,
    )
    teacher_path = TEACHER_CACHE_ROOT / (
        f"{group}_pred{pred_year}_{TEACHER_CACHE_TAG}.pkl"
    )
    if not teacher_path.exists():
        raise FileNotFoundError(f"Missing fold-safe teacher cache: {teacher_path}")
    teacher = pd.read_pickle(teacher_path)
    expected_splits = {"train_oob", "validation"}
    actual_splits = set(map(str, teacher["split"].dropna().unique()))
    if actual_splits != expected_splits:
        raise ValueError(
            f"Unexpected teacher splits for {group} pred{pred_year}: {actual_splits}"
        )
    envelope = _fold_envelope(group, train_years)
    features = attach_cutin_features(
        features,
        teacher,
        cut_in_speed=envelope.cut_in_speed,
        tau=CUTIN_TAU,
    )
    selections = selections.copy()
    selections["scada_cutin_speed"] = envelope.cut_in_speed
    selections["scada_cutin_tau"] = CUTIN_TAU
    return features, feature_cols, selections


baseline.build_fold_features = build_fold_features_with_cutin
baseline.build_group_local_panel = build_cutin_group_panel


if __name__ == "__main__":
    print(
        f"TCN SCADA cut-in input enabled: tau={CUTIN_TAU:g} "
        f"teacher_cache={TEACHER_CACHE_ROOT}",
        flush=True,
    )
    baseline.main()
