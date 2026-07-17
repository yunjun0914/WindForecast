from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from utils.group_local_panel import (
    LOCAL_PANEL_FEATURES,
    GroupLocalPanel,
    build_group_local_panel,
)
from utils.per_turbine_scada import clean_power_10m, turbine_capacity_kwh
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.scada_operating_envelope import OperatingEnvelope, fit_operating_envelope


MODEL_GROUPS = {
    "vestas": ("kpx_group_1", "kpx_group_2"),
    "unison": ("kpx_group_3",),
}
GROUP_MODEL = {
    group: model for model, groups in MODEL_GROUPS.items() for group in groups
}
CUTIN_LOCAL_FEATURES = ("scada_cutin_gate", "scada_cutin_margin")
CUTIN_AGGREGATE_FEATURES = (
    "scada_cutin_active_turbines",
    "scada_cutin_margin_mean",
    "scada_cutin_margin_min",
    "scada_cutin_below_fraction",
)


def collect_model_scada_samples(
    scada: pd.DataFrame,
    model: str,
    train_years: list[int] | tuple[int, ...],
    *,
    isolated_zero_peer_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool same-model turbines and return clean wind/power-ratio samples."""

    if model not in MODEL_GROUPS:
        raise ValueError(f"Unknown turbine model: {model}")
    source = scada.copy()
    source["kst_dtm"] = pd.to_datetime(source["kst_dtm"])
    source = source.loc[source["kst_dtm"].dt.year.isin(train_years)].reset_index(
        drop=True
    )
    wind_parts: list[np.ndarray] = []
    power_parts: list[np.ndarray] = []
    for group in MODEL_GROUPS[model]:
        turbines = GROUP_TURBINE_PREFIXES[group]
        wind_table = pd.DataFrame(
            {
                turbine: pd.to_numeric(source[f"{turbine}_ws"], errors="coerce")
                for turbine in turbines
            }
        )
        for turbine in turbines:
            peer_median = wind_table.drop(columns=turbine).median(axis=1, skipna=True)
            wind = wind_table[turbine]
            isolated_zero = wind.le(0.1) & peer_median.ge(
                isolated_zero_peer_threshold
            )
            power = clean_power_10m(source[f"{turbine}_power_kw10m"], group)
            rated_10m = turbine_capacity_kwh(group) / 6.0
            valid = wind.between(0.0, 35.0) & power.notna() & ~isolated_zero
            wind_parts.append(wind.loc[valid].to_numpy(float))
            power_parts.append((power.loc[valid] / rated_10m).to_numpy(float))
    if not wind_parts:
        raise ValueError(f"No SCADA samples for model={model} years={train_years}")
    return np.concatenate(wind_parts), np.concatenate(power_parts)


def fit_model_cutin(
    scada: pd.DataFrame,
    model: str,
    train_years: list[int] | tuple[int, ...],
) -> OperatingEnvelope:
    wind, power_ratio = collect_model_scada_samples(scada, model, train_years)
    return fit_operating_envelope(
        wind,
        power_ratio,
        bin_width=0.5,
        min_bin_count=30,
        active_power_ratio=0.01,
        crossing_probability=0.50,
    )


def attach_cutin_features(
    features: pd.DataFrame,
    teacher: pd.DataFrame,
    *,
    cut_in_speed: float,
    tau: float = 1.0,
) -> pd.DataFrame:
    """Attach fold-safe RF wind-derived cut-in gate and signed margin."""

    if tau <= 0:
        raise ValueError("tau must be positive")
    keys = ["forecast_kst_dtm", "turbine_id"]
    required = [*keys, "teacher_ws_cubic"]
    missing = [column for column in required if column not in teacher.columns]
    if missing:
        raise ValueError(f"Teacher table missing columns: {missing}")
    teacher_one = teacher[required].copy()
    teacher_one["forecast_kst_dtm"] = pd.to_datetime(
        teacher_one["forecast_kst_dtm"]
    )
    if teacher_one.duplicated(keys).any():
        raise ValueError("Teacher table has duplicate forecast/turbine rows")

    out = features.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out = out.merge(teacher_one, on=keys, how="left", validate="one_to_one")
    coverage = float(out["teacher_ws_cubic"].notna().mean())
    if coverage < 0.95:
        raise ValueError(f"Low teacher wind coverage: {coverage:.4f}")
    margin = out["teacher_ws_cubic"].to_numpy(float) - float(cut_in_speed)
    logit = np.clip(margin / float(tau), -60.0, 60.0)
    out["scada_cutin_margin"] = margin
    out["scada_cutin_gate"] = 1.0 / (1.0 + np.exp(-logit))
    return out.drop(columns="teacher_ws_cubic")


def build_cutin_group_panel(features: pd.DataFrame, group: str) -> GroupLocalPanel:
    """Build the existing group panel plus turbine-local and aggregate cut-in state."""

    local_features = [*LOCAL_PANEL_FEATURES, *CUTIN_LOCAL_FEATURES]
    panel = build_group_local_panel(
        features,
        group,
        local_feature_cols=local_features,
    )
    table = panel.table.copy()
    turbines = GROUP_TURBINE_PREFIXES[group]
    gate_cols = [f"{turbine}__scada_cutin_gate" for turbine in turbines]
    margin_cols = [f"{turbine}__scada_cutin_margin" for turbine in turbines]
    gates = table[gate_cols].to_numpy(float)
    margins = table[margin_cols].to_numpy(float)
    table["scada_cutin_active_turbines"] = np.sum(gates, axis=1)
    table["scada_cutin_margin_mean"] = np.mean(margins, axis=1)
    table["scada_cutin_margin_min"] = np.min(margins, axis=1)
    table["scada_cutin_below_fraction"] = np.mean(margins < 0.0, axis=1)
    return replace(
        panel,
        table=table,
        full_feature_cols=tuple(
            [*panel.full_feature_cols, *CUTIN_AGGREGATE_FEATURES]
        ),
    )
