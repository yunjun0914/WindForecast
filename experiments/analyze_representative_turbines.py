from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import spearmanr

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # type: ignore[no-redef]  # noqa: F401

    sys.modules.setdefault("_bootstrap", _bootstrap)

from experiments.analyze_jointmix_oof import (
    PINN_STEM,
    TCN_STEM,
    aggregate_training,
    build_branch_predictions,
)
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS
from utils.per_turbine_scada import (
    build_official_aligned_turbine_targets,
    turbine_capacity_kwh,
)
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import haversine_km
from utils.site_metadata import load_turbine_metadata


YEARS = (2022, 2023, 2024)
OPTIMAL_GRID_CACHE_VERSION = "per_turbine_optimal_grid_v1"
TEACHER_CACHE_VERSION = "per_turbine_teacher_v1"
TEACHER_CACHE_TAG = "optimal_grid_replace_local16_v1"
REPRESENTATIVE_TURBINES = {
    "kpx_group_1": "vestas_wtg03",
    "kpx_group_2": "vestas_wtg10",
    "kpx_group_3": "unison_wtg05",
}
HIGH_WIND_THRESHOLD = 5.0
LOW_POWER_RATIO = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument(
        "--labels", type=Path, default=Path("data/train/train_labels.csv")
    )
    parser.add_argument("--info", type=Path, default=Path("data/info.xlsx"))
    parser.add_argument(
        "--scada-vestas",
        type=Path,
        default=Path("data/train/scada_vestas_train.csv"),
    )
    parser.add_argument(
        "--scada-unison",
        type=Path,
        default=Path("data/train/scada_unison_train.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/representative_turbine_diagnosis.csv"),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("results/representative_turbine_diagnosis.png"),
    )
    return parser.parse_args()


def l3_error(
    actual: pd.Series | np.ndarray,
    prediction: pd.Series | np.ndarray,
) -> float:
    actual_array = np.asarray(actual, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    valid = np.isfinite(actual_array) & np.isfinite(prediction_array)
    if not valid.any():
        return float("nan")
    cubed_error = np.abs(actual_array[valid] - prediction_array[valid]) ** 3
    return float(np.mean(cubed_error) ** (1.0 / 3.0))


def safe_spearman(
    left: pd.Series | np.ndarray,
    right: pd.Series | np.ndarray,
) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    if int(valid.sum()) < 3:
        return float("nan")
    value = spearmanr(left_array[valid], right_array[valid]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def turbine_score(actual: np.ndarray, prediction: np.ndarray, capacity: float) -> float:
    actual = np.asarray(actual, dtype=float)
    prediction = np.clip(np.asarray(prediction, dtype=float), 0.0, capacity)
    valid = np.isfinite(actual) & np.isfinite(prediction)
    actual = actual[valid]
    prediction = prediction[valid]
    if len(actual) == 0:
        return float("nan")
    error_rate = np.abs(prediction - actual) / capacity
    nmae = float(error_rate.mean())
    unit_price = np.select(
        [error_rate <= 0.06, error_rate <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    denominator = float(np.sum(actual * 4.0))
    ficr = float(np.sum(actual * unit_price) / denominator) if denominator > 0 else 0.0
    return 0.5 * (1.0 - nmae) + 0.5 * ficr


def binned_power_curve_noise(wind: pd.Series, power_ratio: pd.Series) -> float:
    table = pd.DataFrame(
        {
            "wind": pd.to_numeric(wind, errors="coerce"),
            "power_ratio": pd.to_numeric(power_ratio, errors="coerce"),
        }
    ).dropna()
    table = table.loc[table["wind"].between(0.0, 35.0)]
    if len(table) < 100:
        return float("nan")
    table["wind_bin"] = np.floor(table["wind"] * 2.0) / 2.0
    counts = table.groupby("wind_bin")["power_ratio"].transform("count")
    medians = table.groupby("wind_bin")["power_ratio"].transform("median")
    valid = counts >= 30
    if not valid.any():
        return float("nan")
    return float(np.abs(table.loc[valid, "power_ratio"] - medians.loc[valid]).mean())


def load_teacher_oof(cache_root: Path) -> pd.DataFrame:
    parts = []
    for group in TARGET_COLS:
        for pred_year in YEARS:
            path = (
                cache_root
                / TEACHER_CACHE_VERSION
                / f"{group}_pred{pred_year}_{TEACHER_CACHE_TAG}.pkl"
            )
            if not path.exists():
                continue
            table = pd.read_pickle(path)
            table["forecast_kst_dtm"] = pd.to_datetime(table["forecast_kst_dtm"])
            heldout = table.loc[
                table["split"].eq("validation")
                & table["forecast_kst_dtm"].dt.year.eq(pred_year)
            ].copy()
            heldout["group"] = group
            heldout["pred_year"] = pred_year
            parts.append(heldout)
    if not parts:
        raise FileNotFoundError("No outer-year RF teacher cache was found")
    output = pd.concat(parts, ignore_index=True)
    keys = ["forecast_kst_dtm", "group", "turbine_id", "pred_year"]
    if output.duplicated(keys).any():
        raise ValueError("RF teacher OOF cache contains duplicate rows")
    return output


def load_optimal_grid_oof(cache_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_parts = []
    selection_parts = []
    cache_dir = cache_root / OPTIMAL_GRID_CACHE_VERSION
    for group in TARGET_COLS:
        for pred_year in YEARS:
            feature_path = cache_dir / f"{group}_pred{pred_year}_features.pkl"
            selection_path = cache_dir / f"{group}_pred{pred_year}_selection.csv"
            if not feature_path.exists() or not selection_path.exists():
                continue
            features = pd.read_pickle(feature_path)
            features["forecast_kst_dtm"] = pd.to_datetime(
                features["forecast_kst_dtm"]
            )
            features = features.loc[
                features["forecast_kst_dtm"].dt.year.eq(pred_year)
            ].copy()
            features["group"] = group
            features["pred_year"] = pred_year
            feature_parts.append(features)

            selection = pd.read_csv(selection_path, encoding="utf-8-sig")
            selection["pred_year"] = pred_year
            selection_parts.append(selection)
    if not feature_parts:
        raise FileNotFoundError("No outer-year optimal-grid cache was found")
    features = pd.concat(feature_parts, ignore_index=True)
    selections = pd.concat(selection_parts, ignore_index=True)
    keys = ["forecast_kst_dtm", "group", "turbine_id", "pred_year"]
    if features.duplicated(keys).any():
        raise ValueError("Optimal-grid OOF cache contains duplicate rows")
    return features, selections


def build_aligned_oof(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(args.labels, encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv(args.scada_vestas, encoding="utf-8-sig")
    scada_unison = pd.read_csv(args.scada_unison, encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }
    targets = pd.concat(
        [
            build_official_aligned_turbine_targets(
                scada_by_group[group], labels, group
            )
            for group in TARGET_COLS
        ],
        ignore_index=True,
    )
    targets["forecast_kst_dtm"] = pd.to_datetime(targets["forecast_kst_dtm"])

    predictions = build_branch_predictions(args.results_dir)
    teacher = load_teacher_oof(args.cache_root)
    optimal_grid, selections = load_optimal_grid_oof(args.cache_root)
    keys = ["forecast_kst_dtm", "group", "turbine_id", "pred_year"]
    target_columns = [
        "forecast_kst_dtm",
        "group",
        "turbine_id",
        "scada_power_kwh",
        "scada_ws_mean",
        "scada_ws_cubic",
        "scada_share",
        "turbine_target",
        "official_target",
    ]
    aligned = predictions.merge(
        targets[target_columns],
        on=["forecast_kst_dtm", "group", "turbine_id"],
        how="left",
        validate="many_to_one",
    )
    aligned = aligned.merge(
        teacher[
            keys
            + [
                "teacher_ws_mean",
                "teacher_ws_cubic",
                "teacher_power_curve_kwh",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    aligned = aligned.merge(
        optimal_grid[
            keys + ["optgrid_ws_raw", "optgrid_ws_calibrated"]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    coverage = aligned[
        ["turbine_target", "teacher_ws_cubic", "optgrid_ws_calibrated"]
    ].notna().mean()
    if (
        float(coverage["turbine_target"]) < 0.80
        or float(coverage[["teacher_ws_cubic", "optgrid_ws_calibrated"]].min())
        < 0.95
    ):
        raise ValueError(f"OOF diagnostic coverage is too low: {coverage.to_dict()}")
    print(f"OOF alignment coverage={coverage.to_dict()}")
    return aligned, selections


def add_peer_values(aligned: pd.DataFrame) -> pd.DataFrame:
    output_parts = []
    for group, group_table in aligned.groupby("group", sort=False):
        keys = ["forecast_kst_dtm", "pred_year"]
        columns = list(GROUP_TURBINE_PREFIXES[group])
        power = group_table.pivot(
            index=keys, columns="turbine_id", values="scada_power_kwh"
        ).reindex(columns=columns)
        wind = group_table.pivot(
            index=keys, columns="turbine_id", values="scada_ws_cubic"
        ).reindex(columns=columns)
        peer_power = power.rsub(power.sum(axis=1), axis=0).div(len(columns) - 1)
        peer_wind = wind.rsub(wind.sum(axis=1), axis=0).div(len(columns) - 1)
        peer_power = peer_power.stack().rename("peer_power_kwh").reset_index()
        peer_wind = peer_wind.stack().rename("peer_ws_cubic").reset_index()
        peer = peer_power.merge(peer_wind, on=[*keys, "turbine_id"], how="inner")
        output_parts.append(
            group_table.merge(
                peer,
                on=[*keys, "turbine_id"],
                how="left",
                validate="many_to_one",
            )
        )
    return pd.concat(output_parts, ignore_index=True)


def selection_summary(selections: pd.DataFrame) -> pd.DataFrame:
    def mode(values: pd.Series) -> str:
        modes = values.dropna().astype(str).mode()
        return modes.iloc[0] if len(modes) else ""

    return selections.groupby(["group", "turbine_id"], as_index=False).agg(
        selected_source=("source", mode),
        selected_level=("level", mode),
        selected_grid_id=("grid_id", mode),
        selected_candidate_nunique=("candidate", "nunique"),
        selected_distance_km=("distance_km", "mean"),
        selection_cv_ws_mae=("cv_ws_mae", "mean"),
    )


def location_summary(info_path: Path) -> pd.DataFrame:
    metadata = load_turbine_metadata(info_path)
    rows = []
    for group in TARGET_COLS:
        one = metadata.loc[metadata["group"].eq(group)].set_index("turbine_id")
        for turbine in GROUP_TURBINE_PREFIXES[group]:
            origin = one.loc[turbine]
            peers = one.drop(index=turbine)
            distance = haversine_km(
                float(origin["latitude"]),
                float(origin["longitude"]),
                peers["latitude"].to_numpy(float),
                peers["longitude"].to_numpy(float),
            )
            rows.append(
                {
                    "group": group,
                    "turbine_id": turbine,
                    "mean_peer_distance_km": float(np.mean(distance)),
                }
            )
    return pd.DataFrame(rows)


def summarize_turbines(
    aligned: pd.DataFrame,
    selections: pd.DataFrame,
    results_dir: Path,
    info_path: Path,
) -> pd.DataFrame:
    aligned = add_peer_values(aligned)
    rows = []
    for (group, turbine), one in aligned.groupby(
        ["group", "turbine_id"], sort=False
    ):
        one = one.sort_values("forecast_kst_dtm").copy()
        capacity = turbine_capacity_kwh(group)
        model_valid = (
            one["official_target"].ge(0.10 * GROUP_CAPACITY_KWH[group])
            & one["turbine_target"].notna()
        )
        model = one.loc[model_valid]
        power_ratio = one["scada_power_kwh"] / capacity
        high_wind = one["scada_ws_cubic"].ge(HIGH_WIND_THRESHOLD)
        stopped = high_wind & power_ratio.le(LOW_POWER_RATIO)
        time_gap = one["forecast_kst_dtm"].diff().dt.total_seconds().div(3600.0)
        ramp = power_ratio.diff().abs().where(time_gap.eq(1.0))
        share_mean = float(one["scada_share"].mean())
        share_std = float(one["scada_share"].std(ddof=0))
        rows.append(
            {
                "group": group,
                "turbine_id": turbine,
                "is_representative": turbine == REPRESENTATIVE_TURBINES[group],
                "n_oof_rows": len(one),
                "mean_power_ratio": float(power_ratio.mean()),
                "low_power_rate": float(power_ratio.le(0.10).mean()),
                "rated_power_rate": float(power_ratio.ge(0.80).mean()),
                "zero_power_rate": float(power_ratio.le(LOW_POWER_RATIO).mean()),
                "high_wind_hours": int(high_wind.sum()),
                "high_wind_low_power_rate": (
                    float(stopped.sum() / high_wind.sum())
                    if high_wind.any()
                    else float("nan")
                ),
                "hourly_ramp_mae": float(ramp.mean()),
                "power_curve_noise": binned_power_curve_noise(
                    one["scada_ws_cubic"], power_ratio
                ),
                "wind_power_spearman": safe_spearman(
                    one["scada_ws_cubic"], power_ratio
                ),
                "share_mean": share_mean,
                "share_std": share_std,
                "share_cv": share_std / share_mean if share_mean > 0 else float("nan"),
                "target_group_spearman": safe_spearman(
                    one["turbine_target"], one["official_target"]
                ),
                "peer_power_spearman": safe_spearman(
                    one["scada_power_kwh"], one["peer_power_kwh"]
                ),
                "peer_power_nmae": float(
                    np.abs(one["scada_power_kwh"] - one["peer_power_kwh"]).mean()
                    / capacity
                ),
                "peer_wind_spearman": safe_spearman(
                    one["scada_ws_cubic"], one["peer_ws_cubic"]
                ),
                "peer_wind_l3": l3_error(
                    one["scada_ws_cubic"], one["peer_ws_cubic"]
                ),
                "optgrid_wind_l3": l3_error(
                    one["scada_ws_mean"], one["optgrid_ws_calibrated"]
                ),
                "optgrid_wind_spearman": safe_spearman(
                    one["scada_ws_mean"], one["optgrid_ws_calibrated"]
                ),
                "teacher_wind_l3": l3_error(
                    one["scada_ws_cubic"], one["teacher_ws_cubic"]
                ),
                "teacher_wind_spearman": safe_spearman(
                    one["scada_ws_cubic"], one["teacher_ws_cubic"]
                ),
                "teacher_curve_nmae": float(
                    np.abs(
                        one["scada_power_kwh"] - one["teacher_power_curve_kwh"]
                    ).mean()
                    / capacity
                ),
                "pinn_oof_score": turbine_score(
                    model["turbine_target"], model["pinn_mix"], capacity
                ),
                "tcn_oof_score": turbine_score(
                    model["turbine_target"], model["tcn_mix"], capacity
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary["combined_oof_score"] = summary[
        ["pinn_oof_score", "tcn_oof_score"]
    ].mean(axis=1)

    pinn_training = aggregate_training(
        results_dir / f"{PINN_STEM}_training.csv", "pinn"
    )
    tcn_training = aggregate_training(
        results_dir / f"{TCN_STEM}_training.csv", "tcn"
    )
    fold_stability = pinn_training.merge(
        tcn_training, on=["group", "turbine_id"], validate="one_to_one"
    )[
        [
            "group",
            "turbine_id",
            "pinn_score_std",
            "tcn_score_std",
            "pinn_best_epoch_median",
            "tcn_best_epoch_median",
        ]
    ]
    output = (
        summary.merge(
            selection_summary(selections),
            on=["group", "turbine_id"],
            validate="one_to_one",
        )
        .merge(
            location_summary(info_path),
            on=["group", "turbine_id"],
            validate="one_to_one",
        )
        .merge(
            fold_stability,
            on=["group", "turbine_id"],
            validate="one_to_one",
        )
        .sort_values(["group", "combined_oof_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    rank_directions = {
        "combined_oof_score": False,
        "mean_power_ratio": False,
        "share_cv": True,
        "target_group_spearman": False,
        "peer_power_spearman": False,
        "high_wind_low_power_rate": True,
        "power_curve_noise": True,
        "optgrid_wind_l3": True,
        "teacher_wind_l3": True,
    }
    for column, ascending in rank_directions.items():
        output[f"{column}_group_rank"] = (
            output.groupby("group")[column]
            .rank(method="min", ascending=ascending)
            .astype(int)
        )
    return output


def turbine_label(turbine_id: str) -> str:
    return turbine_id.replace("vestas_wtg", "V").replace("unison_wtg", "U")


def configure_font() -> None:
    path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    family = "DejaVu Sans"
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams.update(
        {"font.family": family, "axes.unicode_minus": False, "font.size": 9}
    )


def plot_diagnosis(summary: pd.DataFrame, output_path: Path) -> None:
    configure_font()
    colors = {
        "kpx_group_1": "#2878B5",
        "kpx_group_2": "#E07A2D",
        "kpx_group_3": "#339966",
    }
    panels = [
        ("mean_power_ratio", "평균 SCADA 출력 / 정격", True),
        ("optgrid_wind_l3", "최적격자 풍속 L3 오차", False),
        ("teacher_wind_l3", "RF teacher 풍속 L3 오차", False),
        ("share_cv", "터빈 출력 점유율 변동계수", False),
        ("target_group_spearman", "터빈 타깃과 그룹 발전량 순위상관", True),
        ("peer_power_spearman", "나머지 터빈 평균과 출력 순위상관", True),
        ("high_wind_low_power_rate", "강풍·저출력 비율", False),
        ("power_curve_noise", "SCADA power curve 잔차", False),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    score_group_rank = summary.groupby("group")["combined_oof_score"].rank(
        pct=True
    )
    for axis, (column, label, higher_is_better) in zip(axes.ravel(), panels):
        for group, one in summary.groupby("group", sort=False):
            representative = one["is_representative"]
            axis.scatter(
                one.loc[~representative, column],
                one.loc[~representative, "combined_oof_score"],
                s=42,
                color=colors[group],
                alpha=0.78,
                label=group.replace("kpx_group_", "Group "),
            )
            axis.scatter(
                one.loc[representative, column],
                one.loc[representative, "combined_oof_score"],
                s=150,
                marker="*",
                color=colors[group],
                edgecolor="#111111",
                linewidth=0.8,
                zorder=4,
            )
        for row in summary.itertuples(index=False):
            axis.annotate(
                turbine_label(row.turbine_id),
                (getattr(row, column), row.combined_oof_score),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                fontweight="bold" if row.is_representative else "normal",
            )
        rho = safe_spearman(summary[column], summary["combined_oof_score"])
        driver_group_rank = summary.groupby("group")[column].rank(pct=True)
        within_group_rho = safe_spearman(driver_group_rank, score_group_rank)
        direction = "높을수록 유리" if higher_is_better else "낮을수록 유리"
        axis.set_title(
            f"{label}\n전체 ρ={rho:+.2f} · 그룹 내 순위 ρ={within_group_rho:+.2f}, "
            f"{direction}"
        )
        axis.set_xlabel(label)
        axis.set_ylabel("PINN·TCN 평균 OOF score")
        axis.grid(color="#E5E5E5", linewidth=0.7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(
        handles[:3], labels[:3], loc="lower right", frameon=False, fontsize=8
    )
    figure.suptitle(
        "대표 터빈이 쉬운 이유 진단: V03 · V10 · U05",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    figure.text(
        0.5,
        -0.01,
        "별표는 현재 대표 터빈. 풍속 오차는 mean(|관측-예측|³)^(1/3), 모델 점수는 outer-year OOF만 사용.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def print_driver_correlations(summary: pd.DataFrame) -> None:
    columns = [
        "mean_power_ratio",
        "share_mean",
        "low_power_rate",
        "rated_power_rate",
        "optgrid_wind_l3",
        "teacher_wind_l3",
        "share_cv",
        "target_group_spearman",
        "peer_power_spearman",
        "peer_wind_spearman",
        "high_wind_low_power_rate",
        "power_curve_noise",
        "teacher_curve_nmae",
        "hourly_ramp_mae",
        "mean_peer_distance_km",
    ]
    score_group_rank = summary.groupby("group")["combined_oof_score"].rank(pct=True)
    rows = []
    for column in columns:
        driver_group_rank = summary.groupby("group")[column].rank(pct=True)
        rows.append(
            {
                "driver": column,
                "score_spearman": safe_spearman(
                    summary[column], summary["combined_oof_score"]
                ),
                "within_group_rank_spearman": safe_spearman(
                    driver_group_rank, score_group_rank
                ),
            }
        )
    correlation = pd.DataFrame(rows)
    correlation["abs_correlation"] = correlation["score_spearman"].abs()
    correlation = correlation.sort_values("abs_correlation", ascending=False)
    print("\n=== score-driver Spearman across 17 turbines ===")
    print(correlation.drop(columns="abs_correlation").to_string(index=False))


def main() -> None:
    args = parse_args()
    aligned, selections = build_aligned_oof(args)
    summary = summarize_turbines(aligned, selections, args.results_dir, args.info)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding="utf-8-sig")
    plot_diagnosis(summary, args.figure_output)
    print_driver_correlations(summary)
    columns = [
        "group",
        "turbine_id",
        "is_representative",
        "combined_oof_score",
        "mean_power_ratio",
        "mean_power_ratio_group_rank",
        "optgrid_wind_l3",
        "optgrid_wind_l3_group_rank",
        "teacher_wind_l3",
        "teacher_wind_l3_group_rank",
        "share_cv",
        "share_cv_group_rank",
        "peer_power_spearman",
        "peer_power_spearman_group_rank",
        "high_wind_low_power_rate",
        "high_wind_low_power_rate_group_rank",
        "power_curve_noise",
        "power_curve_noise_group_rank",
    ]
    print("\n=== representative diagnosis ===")
    print(summary.loc[summary["is_representative"], columns].to_string(index=False))
    print(f"\nsaved {args.output}")
    print(f"saved {args.figure_output}")


if __name__ == "__main__":
    main()
