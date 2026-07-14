from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.metrics import TARGET_COLS


@dataclass
class IssueBlockData:
    features: np.ndarray
    targets: np.ndarray
    forecast_times: np.ndarray
    issue_times: np.ndarray
    years: np.ndarray
    feature_cols: tuple[str, ...]
    target_cols: tuple[str, ...]


@dataclass
class PerTurbineIssueBlockData:
    features: np.ndarray
    targets: np.ndarray
    official_targets: np.ndarray
    forecast_times: np.ndarray
    issue_times: np.ndarray
    years: np.ndarray
    feature_cols: tuple[str, ...]


def make_issue_blocks(
    weather: pd.DataFrame,
    labels: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str] = TARGET_COLS,
    expected_leads: tuple[int, ...] = tuple(range(12, 36)),
) -> IssueBlockData:
    required_weather = ["forecast_kst_dtm", "data_available_kst_dtm", *feature_cols]
    missing_weather = [col for col in required_weather if col not in weather.columns]
    if missing_weather:
        raise ValueError(f"Issue weather missing columns: {missing_weather}")
    required_labels = ["kst_dtm", *target_cols]
    missing_labels = [col for col in required_labels if col not in labels.columns]
    if missing_labels:
        raise ValueError(f"Issue labels missing columns: {missing_labels}")

    table = weather[required_weather].copy()
    table["forecast_kst_dtm"] = pd.to_datetime(table["forecast_kst_dtm"])
    table["data_available_kst_dtm"] = pd.to_datetime(table["data_available_kst_dtm"])
    label_table = labels[required_labels].copy()
    label_table["kst_dtm"] = pd.to_datetime(label_table["kst_dtm"])
    table = table.merge(
        label_table,
        left_on="forecast_kst_dtm",
        right_on="kst_dtm",
        how="left",
        validate="one_to_one",
    )
    if table.duplicated(["data_available_kst_dtm", "forecast_kst_dtm"]).any():
        raise ValueError("Duplicate forecast rows inside an issue")

    feature_parts = []
    target_parts = []
    time_parts = []
    issue_times = []
    years = []
    expected_leads_array = np.asarray(expected_leads, dtype=np.int16)

    for issue_time, issue in table.groupby("data_available_kst_dtm", sort=True):
        issue = issue.sort_values("forecast_kst_dtm").reset_index(drop=True)
        leads = (
            (issue["forecast_kst_dtm"] - issue["data_available_kst_dtm"])
            .dt.total_seconds()
            .div(3600)
            .to_numpy(dtype=np.float64)
        )
        if len(issue) != len(expected_leads) or not np.array_equal(
            leads.astype(np.int16), expected_leads_array
        ):
            raise ValueError(
                f"Incomplete issue {issue_time}: rows={len(issue)} leads={leads.tolist()}"
            )

        forecast_years = issue["forecast_kst_dtm"].dt.year.unique()
        if len(forecast_years) != 1:
            continue

        features = issue[feature_cols].to_numpy(dtype=np.float32)
        targets = issue[target_cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        feature_parts.append(
            np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        )
        target_parts.append(targets)
        time_parts.append(issue["forecast_kst_dtm"].to_numpy(dtype="datetime64[ns]"))
        issue_times.append(np.datetime64(issue_time, "ns"))
        years.append(int(forecast_years[0]))

    if not feature_parts:
        raise ValueError("No complete single-year forecast issues were found")

    return IssueBlockData(
        features=np.stack(feature_parts).astype(np.float32),
        targets=np.stack(target_parts).astype(np.float32),
        forecast_times=np.stack(time_parts),
        issue_times=np.asarray(issue_times, dtype="datetime64[ns]"),
        years=np.asarray(years, dtype=np.int16),
        feature_cols=tuple(feature_cols),
        target_cols=tuple(target_cols),
    )


def make_per_turbine_issue_blocks(
    table: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "turbine_target",
    official_col: str = "official_target",
    expected_leads: tuple[int, ...] = tuple(range(12, 36)),
) -> PerTurbineIssueBlockData:
    required = [
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        target_col,
        official_col,
        *feature_cols,
    ]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Per-turbine issue table missing columns: {missing}")

    work = table[required].copy()
    work["forecast_kst_dtm"] = pd.to_datetime(work["forecast_kst_dtm"])
    work["data_available_kst_dtm"] = pd.to_datetime(
        work["data_available_kst_dtm"]
    )
    if work.duplicated(["data_available_kst_dtm", "forecast_kst_dtm"]).any():
        raise ValueError("Duplicate per-turbine forecast rows inside an issue")

    feature_parts = []
    target_parts = []
    official_parts = []
    time_parts = []
    issue_times = []
    years = []
    expected_leads_array = np.asarray(expected_leads, dtype=np.int16)
    for issue_time, issue in work.groupby("data_available_kst_dtm", sort=True):
        issue = issue.sort_values("forecast_kst_dtm").reset_index(drop=True)
        leads = (
            (issue["forecast_kst_dtm"] - issue["data_available_kst_dtm"])
            .dt.total_seconds()
            .div(3600)
            .to_numpy(dtype=np.float64)
        )
        if len(issue) != len(expected_leads) or not np.array_equal(
            leads.astype(np.int16), expected_leads_array
        ):
            raise ValueError(
                f"Incomplete per-turbine issue {issue_time}: "
                f"rows={len(issue)} leads={leads.tolist()}"
            )

        forecast_years = issue["forecast_kst_dtm"].dt.year.unique()
        if len(forecast_years) != 1:
            continue
        features = issue[feature_cols].to_numpy(dtype=np.float32)
        feature_parts.append(
            np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32
            )
        )
        target_parts.append(
            pd.to_numeric(issue[target_col], errors="coerce").to_numpy(np.float32)
        )
        official_parts.append(
            pd.to_numeric(issue[official_col], errors="coerce").to_numpy(np.float32)
        )
        time_parts.append(issue["forecast_kst_dtm"].to_numpy(dtype="datetime64[ns]"))
        issue_times.append(np.datetime64(issue_time, "ns"))
        years.append(int(forecast_years[0]))

    if not feature_parts:
        raise ValueError("No complete single-year per-turbine forecast issues were found")
    return PerTurbineIssueBlockData(
        features=np.stack(feature_parts).astype(np.float32),
        targets=np.stack(target_parts).astype(np.float32),
        official_targets=np.stack(official_parts).astype(np.float32),
        forecast_times=np.stack(time_parts),
        issue_times=np.asarray(issue_times, dtype="datetime64[ns]"),
        years=np.asarray(years, dtype=np.int16),
        feature_cols=tuple(feature_cols),
    )
