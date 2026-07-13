from __future__ import annotations

import numpy as np
import pandas as pd

from utils.metrics import GROUP_CAPACITY_KWH
from utils.power_curve import GROUP_MANUFACTURER, GROUP_TURBINE_PREFIXES


SCADA_ALIGNMENT = {
    "vestas": ("ceil", 0),
    "unison": ("floor", 1),
}
SCADA_UPPER_TOLERANCE = 1.02
SCADA_MIN_VALID_10M = 4


def turbine_capacity_kwh(group: str) -> float:
    return float(GROUP_CAPACITY_KWH[group]) / len(GROUP_TURBINE_PREFIXES[group])


def power_10m_upper_kwh(group: str, tolerance: float = SCADA_UPPER_TOLERANCE) -> float:
    return turbine_capacity_kwh(group) / 6.0 * tolerance


def align_scada_hour(timestamps: pd.Series, group: str) -> pd.Series:
    manufacturer = GROUP_MANUFACTURER[group]
    alignment, shift_hours = SCADA_ALIGNMENT[manufacturer]
    times = pd.to_datetime(timestamps)
    if alignment == "ceil":
        hours = times.dt.ceil("h")
    elif alignment == "floor":
        hours = times.dt.floor("h")
    else:
        raise ValueError(f"Unknown SCADA alignment: {alignment}")
    return hours + pd.Timedelta(hours=shift_hours)


def clean_power_10m(values: pd.Series, group: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    cleaned = numeric.clip(lower=0.0)
    return cleaned.mask(cleaned > power_10m_upper_kwh(group))


def _hourly_energy(cleaned: pd.Series, hours: pd.Series, min_valid_10m: int) -> pd.DataFrame:
    table = pd.DataFrame({"forecast_kst_dtm": hours, "power": cleaned})
    grouped = table.groupby("forecast_kst_dtm", sort=True)["power"]
    count = grouped.count()
    energy = grouped.sum(min_count=1).mul(6.0).div(count)
    energy = energy.where(count >= min_valid_10m)
    return pd.DataFrame(
        {
            "forecast_kst_dtm": energy.index,
            "scada_power_kwh": energy.to_numpy(float),
            "valid_10m_count": count.reindex(energy.index).to_numpy(int),
        }
    )


def _hourly_wind(scada: pd.DataFrame, turbine: str, hours: pd.Series) -> pd.DataFrame:
    ws = pd.to_numeric(scada[f"{turbine}_ws"], errors="coerce").where(lambda s: s.between(0, 40))
    wd = pd.to_numeric(scada[f"{turbine}_wd"], errors="coerce")
    radians = np.radians(wd)
    table = pd.DataFrame(
        {
            "forecast_kst_dtm": hours,
            "ws": ws,
            "ws_cube": ws**3,
            "wd_sin": np.sin(radians),
            "wd_cos": np.cos(radians),
        }
    )
    grouped = table.groupby("forecast_kst_dtm", sort=True)
    hourly = grouped[["ws", "ws_cube", "wd_sin", "wd_cos"]].mean()
    hourly["scada_ws_cubic"] = np.cbrt(hourly.pop("ws_cube").clip(lower=0))
    return hourly.rename(
        columns={
            "ws": "scada_ws_mean",
            "wd_sin": "scada_wd_sin",
            "wd_cos": "scada_wd_cos",
        }
    ).reset_index()


def build_turbine_scada_hourly(
    scada: pd.DataFrame,
    group: str,
    min_valid_10m: int = SCADA_MIN_VALID_10M,
) -> pd.DataFrame:
    hours = align_scada_hour(scada["kst_dtm"], group)
    parts = []
    for turbine in GROUP_TURBINE_PREFIXES[group]:
        power = clean_power_10m(scada[f"{turbine}_power_kw10m"], group)
        hourly = _hourly_energy(power, hours, min_valid_10m)
        hourly = hourly.merge(_hourly_wind(scada, turbine, hours), on="forecast_kst_dtm", how="left")
        hourly["group"] = group
        hourly["turbine_id"] = turbine
        parts.append(hourly)
    return pd.concat(parts, ignore_index=True)


def build_official_aligned_turbine_targets(
    scada: pd.DataFrame,
    labels: pd.DataFrame,
    group: str,
    min_valid_10m: int = SCADA_MIN_VALID_10M,
) -> pd.DataFrame:
    hourly = build_turbine_scada_hourly(scada, group, min_valid_10m=min_valid_10m)
    labels_one = labels[["kst_dtm", group]].copy()
    labels_one["kst_dtm"] = pd.to_datetime(labels_one["kst_dtm"])
    labels_one = labels_one.rename(columns={"kst_dtm": "forecast_kst_dtm", group: "official_target"})
    hourly = hourly.merge(labels_one, on="forecast_kst_dtm", how="left")

    n_turbines = len(GROUP_TURBINE_PREFIXES[group])
    group_stats = (
        hourly.groupby("forecast_kst_dtm", as_index=False)
        .agg(
            scada_group_sum=("scada_power_kwh", "sum"),
            valid_turbines=("scada_power_kwh", "count"),
        )
    )
    hourly = hourly.merge(group_stats, on="forecast_kst_dtm", how="left")
    complete = hourly["valid_turbines"].eq(n_turbines) & hourly["scada_group_sum"].gt(0)
    hourly["scada_share"] = np.where(
        complete,
        hourly["scada_power_kwh"] / hourly["scada_group_sum"],
        np.nan,
    )
    hourly["turbine_target"] = hourly["official_target"] * hourly["scada_share"]
    hourly["year"] = pd.to_datetime(hourly["forecast_kst_dtm"]).dt.year
    return hourly.sort_values(["forecast_kst_dtm", "turbine_id"]).reset_index(drop=True)


def target_integrity_summary(targets: pd.DataFrame) -> pd.DataFrame:
    valid = targets.dropna(subset=["turbine_target", "official_target"])
    grouped = (
        valid.groupby(["group", "forecast_kst_dtm"], as_index=False)
        .agg(
            official_target=("official_target", "first"),
            turbine_target_sum=("turbine_target", "sum"),
            n_turbines=("turbine_id", "nunique"),
        )
    )
    grouped["abs_error"] = (grouped["turbine_target_sum"] - grouped["official_target"]).abs()
    return (
        grouped.groupby("group", as_index=False)
        .agg(
            hours=("forecast_kst_dtm", "count"),
            max_abs_error=("abs_error", "max"),
            mean_abs_error=("abs_error", "mean"),
        )
    )
