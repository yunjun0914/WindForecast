from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # type: ignore[no-redef]  # noqa: F401

    sys.modules.setdefault("_bootstrap", _bootstrap)

from experiments.predict_group_joint_family_submission import build_turbine_mix
from experiments.tune_power_lgbm_hyperparams import prepare_fold_cache, sample_weight
from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
)
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.tree_feature_profiles import FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_V1


PINN_STEM = "per_turbine_pinn_optimal_grid_replace_v1"
TCN_STEM = "per_turbine_tcn_optgrid_weather_only_groupjoint_h64_l1_e3_v1"
PREDICTION_KEYS = ["forecast_kst_dtm", "group", "turbine_id", "pred_year"]
TREE_PARAM_COLUMNS = [
    "random_state",
    "n_jobs",
    "verbose",
    "objective",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "min_split_gain",
]
ISOTONIC_ALPHA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
REPRESENTATIVE_TURBINES = {
    "kpx_group_1": "vestas_wtg03",
    "kpx_group_2": "vestas_wtg10",
    "kpx_group_3": "unison_wtg05",
}
RESIDUAL_LAMBDA_GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--labels", type=Path, default=Path("data/train/train_labels.csv")
    )
    parser.add_argument(
        "--tree-best-csv",
        type=Path,
        default=Path("configs/group_quota_lgbm_complete_nested_v1_best.csv"),
    )
    parser.add_argument(
        "--tree-predictions",
        type=Path,
        default=Path("results/group_quota_lgbm_complete_nested_v1_predictions.csv"),
    )
    parser.add_argument("--rebuild-tree", action="store_true")
    parser.add_argument(
        "--difficulty-output",
        type=Path,
        default=Path("results/jointmix_restore_turbine_difficulty.csv"),
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=Path("results/jointmix_restore_turbine_difficulty.png"),
    )
    parser.add_argument("--isotonic-comparison-output", type=Path)
    parser.add_argument("--isotonic-submission-input", type=Path)
    parser.add_argument("--isotonic-submission-output", type=Path)
    parser.add_argument("--representative-comparison-output", type=Path)
    return parser.parse_args()


def load_prediction(path: Path, output_column: str) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in [*PREDICTION_KEYS, "pred"] if column not in table]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    table["forecast_kst_dtm"] = pd.to_datetime(table["forecast_kst_dtm"])
    if table.duplicated(PREDICTION_KEYS).any():
        raise ValueError(f"{path} contains duplicate turbine predictions")
    return table[PREDICTION_KEYS + ["pred"]].rename(
        columns={"pred": output_column}
    )


def build_branch_predictions(results_dir: Path) -> pd.DataFrame:
    paths = {
        "pinn_base": results_dir / f"{PINN_STEM}_baseline_turbine_predictions.csv",
        "pinn_joint": results_dir / f"{PINN_STEM}_turbine_predictions.csv",
        "tcn_base": results_dir / f"{TCN_STEM}_baseline_turbine_predictions.csv",
        "tcn_joint": results_dir / f"{TCN_STEM}_turbine_predictions.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"OOF turbine predictions are missing: {missing}")

    combined = load_prediction(paths["pinn_base"], "pinn_base")
    for column in ("pinn_joint", "tcn_base", "tcn_joint"):
        combined = combined.merge(
            load_prediction(paths[column], column),
            on=PREDICTION_KEYS,
            how="inner",
            validate="one_to_one",
        )
    return build_turbine_mix(combined)


def tree_params(row: pd.Series) -> dict[str, object]:
    params = {column: row[column] for column in TREE_PARAM_COLUMNS}
    integer_columns = {
        "random_state",
        "n_jobs",
        "verbose",
        "n_estimators",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample_freq",
    }
    for column in integer_columns:
        params[column] = int(params[column])
    for column in set(TREE_PARAM_COLUMNS) - integer_columns - {"objective"}:
        params[column] = float(params[column])
    return params


def get_or_build_tree_predictions(
    best_path: Path,
    prediction_path: Path,
    rebuild: bool,
) -> pd.DataFrame:
    if prediction_path.exists() and not rebuild:
        predictions = pd.read_csv(prediction_path, encoding="utf-8-sig")
        predictions["forecast_kst_dtm"] = pd.to_datetime(
            predictions["forecast_kst_dtm"]
        )
        return predictions

    best = pd.read_csv(best_path, encoding="utf-8-sig")
    cache = prepare_fold_cache(
        TARGET_COLS,
        feature_profile=FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_V1,
    )
    rows = []
    for group in TARGET_COLS:
        selected = best.loc[best["group"].eq(group)]
        if len(selected) != 1:
            raise ValueError(f"Expected one TREE parameter row for {group}")
        selected = selected.iloc[0]
        for fold in cache[group]:
            y_train = fold["y_train"]
            keep = y_train.to_numpy(float) >= (
                GROUP_CAPACITY_KWH[group] * float(selected["min_output_ratio"])
            )
            x_train = fold["x_train"].loc[keep]
            y_used = y_train.loc[keep]
            model = LGBMRegressor(**tree_params(selected))
            model.fit(
                x_train,
                y_used,
                sample_weight=sample_weight(
                    y_used, group, str(selected["weight_policy"])
                ),
            )
            prediction = np.clip(
                model.predict(fold["x_val"]),
                0.0,
                GROUP_CAPACITY_KWH[group],
            )
            rows.append(
                pd.DataFrame(
                    {
                        "forecast_kst_dtm": pd.to_datetime(fold["time_val"]),
                        "group": group,
                        "pred_year": int(fold["pred_year"]),
                        "actual": fold["y_val"].to_numpy(float),
                        "pred": prediction,
                    }
                )
            )
    predictions = pd.concat(rows, ignore_index=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    return predictions


def build_group_oof_predictions(
    turbine: pd.DataFrame,
    tree_predictions: pd.DataFrame,
    labels_path: Path,
) -> pd.DataFrame:
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    group_predictions = turbine.groupby(
        ["forecast_kst_dtm", "group", "pred_year"], as_index=False
    )[
        [
            "pinn_base",
            "pinn_joint",
            "pinn_mix",
            "tcn_base",
            "tcn_joint",
            "tcn_mix",
        ]
    ].sum()
    tree = tree_predictions.copy()
    tree["forecast_kst_dtm"] = pd.to_datetime(tree["forecast_kst_dtm"])
    tree_keys = ["forecast_kst_dtm", "group", "pred_year"]
    if tree.duplicated(tree_keys).any():
        raise ValueError("TREE OOF predictions contain duplicate rows")
    group_predictions = group_predictions.merge(
        tree[tree_keys + ["pred"]].rename(columns={"pred": "tree"}),
        on=tree_keys,
        how="inner",
        validate="one_to_one",
    )
    group_predictions["jointmix_raw"] = (
        0.50 * group_predictions["pinn_mix"]
        + 0.05 * group_predictions["tree"]
        + 0.45 * group_predictions["tcn_mix"]
    )
    capacities = group_predictions["group"].map(GROUP_CAPACITY_KWH).to_numpy(float)
    group_predictions["jointmix_floor10"] = np.clip(
        group_predictions["jointmix_raw"].to_numpy(float),
        0.10 * capacities,
        capacities,
    )
    parts = []
    for group in TARGET_COLS:
        one = group_predictions.loc[group_predictions["group"].eq(group)].merge(
            labels[["kst_dtm", group]],
            left_on="forecast_kst_dtm",
            right_on="kst_dtm",
            how="inner",
        )
        one = one.dropna(subset=[group]).copy()
        one["official_target"] = one[group].to_numpy(float)
        parts.append(one.drop(columns=["kst_dtm", group]))
    return pd.concat(parts, ignore_index=True)


def score_branches(group_predictions: pd.DataFrame) -> pd.DataFrame:
    prediction_parts = []
    variants = {
        "jointmix_p50_t5_c45_floor10": "jointmix_floor10",
        "jointmix_p50_t5_c45_raw": "jointmix_raw",
        "pinn_base": "pinn_base",
        "pinn_joint": "pinn_joint",
        "pinn_mix_50_50_floor20": "pinn_mix",
        "tree_quota65": "tree",
        "tcn_base": "tcn_base",
        "tcn_joint": "tcn_joint",
        "tcn_mix_25_75": "tcn_mix",
    }
    for variant, prediction_column in variants.items():
        for group in TARGET_COLS:
            one = group_predictions.loc[
                group_predictions["group"].eq(group)
            ].copy()
            one["pred"] = np.clip(
                one[prediction_column].to_numpy(float),
                0.0,
                GROUP_CAPACITY_KWH[group],
            )
            one["variant"] = variant
            prediction_parts.append(
                one[["variant", "group", "official_target", "pred"]]
            )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    summary, group_scores = pooled_oof_summary(predictions)
    return summary.merge(
        group_scores.groupby("variant", as_index=False).agg(
            min_group_score=("score", "min"),
            max_group_score=("score", "max"),
        ),
        on="variant",
        how="left",
    ).sort_values("mean_score", ascending=False)


def representative_residual_prediction(
    frame: pd.DataFrame,
    residual_lambda: float,
) -> np.ndarray:
    capacities = frame["group"].map(GROUP_CAPACITY_KWH).to_numpy(float)
    raw = (
        0.95
        * (
            frame["representative_baseline"].to_numpy(float)
            + float(residual_lambda) * frame["spatial_residual"].to_numpy(float)
        )
        + 0.05 * frame["tree"].to_numpy(float)
    )
    return np.clip(raw, 0.10 * capacities, capacities)


def build_representative_residual_frame(
    turbine: pd.DataFrame,
    group_oof: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["forecast_kst_dtm", "group", "pred_year"]
    work = turbine[
        [*keys, "turbine_id", "pinn_mix", "tcn_mix"]
    ].copy()
    work["neural_turbine"] = (
        0.50 * work["pinn_mix"].to_numpy(float)
        + 0.45 * work["tcn_mix"].to_numpy(float)
    ) / 0.95
    summed = (
        work.groupby(keys, as_index=False)["neural_turbine"]
        .sum()
        .rename(columns={"neural_turbine": "neural_sum"})
    )

    representative_parts = []
    for group, turbine_id in REPRESENTATIVE_TURBINES.items():
        one = work.loc[
            work["group"].eq(group) & work["turbine_id"].eq(turbine_id),
            [*keys, "neural_turbine"],
        ].copy()
        expected_rows = int(work.loc[work["group"].eq(group), keys].drop_duplicates().shape[0])
        if len(one) != expected_rows:
            raise ValueError(
                f"Representative turbine coverage differs for {group}/{turbine_id}: "
                f"{len(one)} != {expected_rows}"
            )
        one["representative_turbine"] = turbine_id
        one["n_turbines"] = len(GROUP_TURBINE_PREFIXES[group])
        representative_parts.append(one)
    representative = pd.concat(representative_parts, ignore_index=True)

    base = group_oof[
        [*keys, "official_target", "tree", "jointmix_floor10"]
    ].copy()
    output = base.merge(summed, on=keys, validate="one_to_one").merge(
        representative,
        on=keys,
        validate="one_to_one",
    )
    output["representative_baseline"] = (
        output["n_turbines"].to_numpy(float)
        * output["neural_turbine"].to_numpy(float)
    )
    output["spatial_residual"] = (
        output["neural_sum"] - output["representative_baseline"]
    )
    lambda_one = representative_residual_prediction(output, 1.0)
    max_difference = float(
        np.max(np.abs(lambda_one - output["jointmix_floor10"].to_numpy(float)))
    )
    if max_difference > 1e-6:
        raise ValueError(
            f"Representative residual lambda=1 differs from jointmix: {max_difference}"
        )
    return output


def select_residual_lambda(
    frame: pd.DataFrame,
    group: str,
) -> tuple[float, dict[float, float]]:
    capacity = float(GROUP_CAPACITY_KWH[group])
    actual = frame["official_target"].to_numpy(float)
    scores = {}
    for residual_lambda in RESIDUAL_LAMBDA_GRID:
        prediction = representative_residual_prediction(frame, residual_lambda)
        nmae, ficr = group_nmae_ficr(actual, prediction, capacity)
        scores[residual_lambda] = 0.5 * (1.0 - nmae) + 0.5 * ficr
    selected = max(
        RESIDUAL_LAMBDA_GRID,
        key=lambda value: (scores[value], value),
    )
    return float(selected), scores


def crossfit_representative_residual(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for group in TARGET_COLS:
        group_frame = frame.loc[frame["group"].eq(group)].copy()
        years = sorted(int(value) for value in group_frame["pred_year"].unique())
        if len(years) < 2:
            raise ValueError(
                f"Representative residual cross-fit needs two years: {group}"
            )
        for held_year in years:
            train = group_frame.loc[group_frame["pred_year"].ne(held_year)]
            heldout = group_frame.loc[
                group_frame["pred_year"].eq(held_year)
            ].copy()
            selected_lambda, _ = select_residual_lambda(train, group)
            heldout["representative_residual_pred"] = (
                representative_residual_prediction(heldout, selected_lambda)
            )
            heldout["selected_residual_lambda"] = selected_lambda
            parts.append(heldout)
    return pd.concat(parts, ignore_index=True)


def representative_residual_comparison(crossfit: pd.DataFrame) -> pd.DataFrame:
    variants = [
        (f"fixed_lambda_{value:.2f}", value, None)
        for value in RESIDUAL_LAMBDA_GRID
    ]
    variants.append(
        (
            "nested_selected_lambda",
            None,
            "representative_residual_pred",
        )
    )
    rows = []
    for variant, residual_lambda, prediction_column in variants:
        for group in TARGET_COLS:
            one = crossfit.loc[crossfit["group"].eq(group)]
            capacity = float(GROUP_CAPACITY_KWH[group])
            prediction = (
                representative_residual_prediction(one, residual_lambda)
                if prediction_column is None
                else one[prediction_column].to_numpy(float)
            )
            actual = one["official_target"].to_numpy(float)
            nmae, ficr = group_nmae_ficr(actual, prediction, capacity)
            hit6, hit8 = weighted_hit_rates(actual, prediction, capacity)
            rows.append(
                {
                    "scope": "group",
                    "group": group,
                    "variant": variant,
                    "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                    "one_minus_nmae": 1.0 - nmae,
                    "weighted_hit6": hit6,
                    "weighted_hit8": hit8,
                    "ficr": ficr,
                }
            )
    per_group = pd.DataFrame(rows)
    overall = (
        per_group.groupby("variant", as_index=False)
        .agg(
            one_minus_nmae=("one_minus_nmae", "mean"),
            weighted_hit6=("weighted_hit6", "mean"),
            weighted_hit8=("weighted_hit8", "mean"),
            ficr=("ficr", "mean"),
        )
    )
    overall["score"] = 0.5 * overall["one_minus_nmae"] + 0.5 * overall["ficr"]
    overall["scope"] = "overall"
    overall["group"] = "all"
    comparison = pd.concat([overall, per_group], ignore_index=True)

    diagnostics = []
    for group in TARGET_COLS:
        one = crossfit.loc[crossfit["group"].eq(group)]
        selected = one.groupby("pred_year", sort=True)[
            "selected_residual_lambda"
        ].first()
        diagnostics.append(
            {
                "group": group,
                "mean_selected_lambda": float(selected.mean()),
                "selected_lambda_by_year": ",".join(
                    f"{int(year)}:{float(value):.2f}"
                    for year, value in selected.items()
                ),
                "representative_turbine": REPRESENTATIVE_TURBINES[group],
                "mean_abs_spatial_residual": float(
                    one["spatial_residual"].abs().mean()
                ),
            }
        )
    comparison = comparison.merge(pd.DataFrame(diagnostics), on="group", how="left")
    return comparison.sort_values(
        ["scope", "score"], ascending=[False, False]
    ).reset_index(drop=True)


def fit_jointmix_isotonic(frame: pd.DataFrame, group: str) -> IsotonicRegression:
    capacity = float(GROUP_CAPACITY_KWH[group])
    valid = (
        np.isfinite(frame["jointmix_floor10"])
        & np.isfinite(frame["official_target"])
        & frame["official_target"].ge(0.10 * capacity)
    )
    if int(valid.sum()) < 500:
        raise ValueError(
            f"Too few jointmix isotonic rows for {group}: {int(valid.sum())}"
        )
    calibrator = IsotonicRegression(
        increasing=True,
        y_min=0.10 * capacity,
        y_max=capacity,
        out_of_bounds="clip",
    )
    calibrator.fit(
        frame.loc[valid, "jointmix_floor10"].to_numpy(float),
        frame.loc[valid, "official_target"].to_numpy(float),
    )
    return calibrator


def isotonic_inner_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    years = sorted(int(value) for value in frame["pred_year"].unique())
    if len(years) >= 2:
        return [
            (
                np.flatnonzero(frame["pred_year"].ne(year).to_numpy()),
                np.flatnonzero(frame["pred_year"].eq(year).to_numpy()),
            )
            for year in years
        ]
    months = pd.to_datetime(frame["forecast_kst_dtm"]).dt.month
    blocks = (months.sub(1) // 4).astype(int)
    return [
        (
            np.flatnonzero(blocks.ne(block).to_numpy()),
            np.flatnonzero(blocks.eq(block).to_numpy()),
        )
        for block in sorted(int(value) for value in blocks.unique())
    ]


def crossfit_isotonic_values(frame: pd.DataFrame, group: str) -> np.ndarray:
    prediction = np.full(len(frame), np.nan, dtype=float)
    for fit_indices, heldout_indices in isotonic_inner_folds(frame):
        calibrator = fit_jointmix_isotonic(frame.iloc[fit_indices], group)
        prediction[heldout_indices] = calibrator.predict(
            frame.iloc[heldout_indices]["jointmix_floor10"].to_numpy(float)
        )
    if not np.isfinite(prediction).all():
        raise ValueError(f"Incomplete isotonic inner OOF for {group}")
    return prediction


def select_ficr_aware_alpha(
    frame: pd.DataFrame,
    isotonic_prediction: np.ndarray,
    group: str,
) -> tuple[float, dict[float, float]]:
    capacity = float(GROUP_CAPACITY_KWH[group])
    actual = frame["official_target"].to_numpy(float)
    raw = frame["jointmix_floor10"].to_numpy(float)
    scores = {}
    for alpha in ISOTONIC_ALPHA_GRID:
        blended = np.clip(
            raw + alpha * (isotonic_prediction - raw),
            0.10 * capacity,
            capacity,
        )
        nmae, ficr = group_nmae_ficr(actual, blended, capacity)
        scores[alpha] = 0.5 * (1.0 - nmae) + 0.5 * ficr
    selected = max(ISOTONIC_ALPHA_GRID, key=lambda value: (scores[value], -value))
    return float(selected), scores


def crossfit_jointmix_isotonic(group_oof: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for group in TARGET_COLS:
        group_frame = group_oof.loc[group_oof["group"].eq(group)].copy()
        years = sorted(int(value) for value in group_frame["pred_year"].unique())
        if len(years) < 2:
            raise ValueError(f"Isotonic cross-fit needs at least two years: {group}")
        capacity = float(GROUP_CAPACITY_KWH[group])
        for held_year in years:
            train = group_frame.loc[
                group_frame["pred_year"].ne(held_year)
            ].reset_index(drop=True)
            heldout = group_frame.loc[
                group_frame["pred_year"].eq(held_year)
            ].copy()
            inner_isotonic = crossfit_isotonic_values(train, group)
            selected_alpha, _ = select_ficr_aware_alpha(
                train, inner_isotonic, group
            )
            calibrator = fit_jointmix_isotonic(train, group)
            raw = heldout["jointmix_floor10"].to_numpy(float)
            full_isotonic = calibrator.predict(raw)
            heldout["isotonic_full_pred"] = np.clip(
                full_isotonic,
                0.10 * capacity,
                capacity,
            )
            heldout["isotonic_pred"] = np.clip(
                raw + selected_alpha * (full_isotonic - raw),
                0.10 * capacity,
                capacity,
            )
            heldout["isotonic_alpha"] = selected_alpha
            parts.append(heldout)
    return pd.concat(parts, ignore_index=True)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return np.nan
    if np.unique(left[valid]).size < 2 or np.unique(right[valid]).size < 2:
        return np.nan
    return float(spearmanr(left[valid], right[valid]).statistic)


def weighted_hit_rates(
    actual: np.ndarray,
    prediction: np.ndarray,
    capacity: float,
) -> tuple[float, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction) & (
        actual >= 0.10 * capacity
    )
    actual = actual[valid]
    error_rate = np.abs(prediction[valid] - actual) / capacity
    weight_sum = float(actual.sum())
    if weight_sum <= 0.0:
        return np.nan, np.nan
    hit6 = float(actual[error_rate <= 0.06].sum() / weight_sum)
    hit8 = float(actual[error_rate <= 0.08].sum() / weight_sum)
    return hit6, hit8


def isotonic_comparison(crossfit: pd.DataFrame) -> pd.DataFrame:
    group_rows = []
    for variant, column in (
        ("jointmix_raw", "jointmix_floor10"),
        ("jointmix_isotonic_full", "isotonic_full_pred"),
        ("jointmix_isotonic_ficr_selected", "isotonic_pred"),
    ):
        for group in TARGET_COLS:
            one = crossfit.loc[crossfit["group"].eq(group)]
            capacity = float(GROUP_CAPACITY_KWH[group])
            actual = one["official_target"].to_numpy(float)
            prediction = one[column].to_numpy(float)
            nmae, ficr = group_nmae_ficr(actual, prediction, capacity)
            hit6, hit8 = weighted_hit_rates(actual, prediction, capacity)
            group_rows.append(
                {
                    "scope": "group",
                    "group": group,
                    "variant": variant,
                    "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                    "one_minus_nmae": 1.0 - nmae,
                    "weighted_hit6": hit6,
                    "weighted_hit8": hit8,
                    "ficr": ficr,
                }
            )
    per_group = pd.DataFrame(group_rows)
    overall = (
        per_group.groupby("variant", as_index=False)
        .agg(
            one_minus_nmae=("one_minus_nmae", "mean"),
            weighted_hit6=("weighted_hit6", "mean"),
            weighted_hit8=("weighted_hit8", "mean"),
            ficr=("ficr", "mean"),
        )
    )
    overall["score"] = 0.5 * overall["one_minus_nmae"] + 0.5 * overall["ficr"]
    overall["scope"] = "overall"
    overall["group"] = "all"
    comparison = pd.concat([overall, per_group], ignore_index=True)

    diagnostic_rows = []
    for group in TARGET_COLS:
        one = crossfit.loc[crossfit["group"].eq(group)]
        capacity = float(GROUP_CAPACITY_KWH[group])
        valid = one["official_target"].ge(0.10 * capacity).to_numpy()
        raw = one.loc[valid, "jointmix_floor10"].to_numpy(float)
        calibrated = one.loc[valid, "isotonic_pred"].to_numpy(float)
        actual = one.loc[valid, "official_target"].to_numpy(float)
        fold_inversions = 0
        fold_unique_retention = []
        for _, fold in one.loc[valid].groupby("pred_year", sort=False):
            fold_raw = fold["jointmix_floor10"].to_numpy(float)
            fold_calibrated = fold["isotonic_pred"].to_numpy(float)
            mapping = (
                pd.DataFrame(
                    {"raw": fold_raw, "calibrated": fold_calibrated}
                )
                .groupby("raw", sort=True)["calibrated"]
                .first()
                .to_numpy(float)
            )
            fold_inversions += int(np.sum(np.diff(mapping) < -1e-8))
            fold_unique_retention.append(
                int(np.unique(fold_calibrated).size)
                / max(int(np.unique(fold_raw).size), 1)
            )
        diagnostic_rows.append(
            {
                "group": group,
                "raw_actual_spearman": safe_spearman(raw, actual),
                "calibrated_actual_spearman": safe_spearman(calibrated, actual),
                "raw_calibrated_spearman": safe_spearman(raw, calibrated),
                "mean_selected_alpha": float(one["isotonic_alpha"].mean()),
                "selected_alpha_by_year": ",".join(
                    f"{int(year)}:{float(alpha):.2f}"
                    for year, alpha in one.groupby("pred_year", sort=True)[
                        "isotonic_alpha"
                    ].first().items()
                ),
                "mean_fold_unique_retention": float(
                    np.mean(fold_unique_retention)
                ),
                "within_fold_monotonic_inversions": fold_inversions,
                "mean_abs_adjustment": float(np.mean(np.abs(calibrated - raw))),
            }
        )
    diagnostics = pd.DataFrame(diagnostic_rows)
    comparison = comparison.merge(diagnostics, on="group", how="left")
    return comparison[
        [
            "scope",
            "group",
            "variant",
            "score",
            "one_minus_nmae",
            "weighted_hit6",
            "weighted_hit8",
            "ficr",
            "raw_actual_spearman",
            "calibrated_actual_spearman",
            "raw_calibrated_spearman",
            "mean_selected_alpha",
            "selected_alpha_by_year",
            "mean_fold_unique_retention",
            "within_fold_monotonic_inversions",
            "mean_abs_adjustment",
        ]
    ]


def apply_jointmix_isotonic_submission(
    group_oof: pd.DataFrame,
    submission_path: Path,
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    submission = pd.read_csv(submission_path, encoding="utf-8-sig")
    required = ["forecast_id", "forecast_kst_dtm", *TARGET_COLS]
    if list(submission.columns) != required:
        raise ValueError(
            f"Submission columns differ from expected: {submission.columns.tolist()}"
        )
    output = submission.copy()
    selected_alphas = {}
    for group in TARGET_COLS:
        group_frame = group_oof.loc[group_oof["group"].eq(group)].reset_index(
            drop=True
        )
        crossfit_isotonic = crossfit_isotonic_values(group_frame, group)
        selected_alpha, _ = select_ficr_aware_alpha(
            group_frame, crossfit_isotonic, group
        )
        selected_alphas[group] = selected_alpha
        calibrator = fit_jointmix_isotonic(
            group_frame, group
        )
        capacity = float(GROUP_CAPACITY_KWH[group])
        raw = output[group].to_numpy(float)
        isotonic_prediction = calibrator.predict(raw)
        output[group] = np.clip(
            raw + selected_alpha * (isotonic_prediction - raw),
            0.10 * capacity,
            capacity,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output, selected_alphas


def aggregate_training(path: Path, prefix: str) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig")
    required = ["group", "turbine_id", "best_epoch", "turbine_val_score"]
    missing = [column for column in required if column not in table]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return (
        table.groupby(["group", "turbine_id"], as_index=False)
        .agg(
            score_mean=("turbine_val_score", "mean"),
            score_std=("turbine_val_score", lambda values: values.std(ddof=0)),
            score_min=("turbine_val_score", "min"),
            best_epoch_median=("best_epoch", "median"),
            best_epoch_max=("best_epoch", "max"),
            n_folds=("pred_year", "nunique"),
        )
        .rename(
            columns={
                column: f"{prefix}_{column}"
                for column in (
                    "score_mean",
                    "score_std",
                    "score_min",
                    "best_epoch_median",
                    "best_epoch_max",
                    "n_folds",
                )
            }
        )
    )


def classify_turbines(difficulty: pd.DataFrame) -> pd.DataFrame:
    out = difficulty.copy()
    out["combined_score"] = out[["pinn_score_mean", "tcn_score_mean"]].mean(
        axis=1
    )
    out["max_fold_std"] = out[["pinn_score_std", "tcn_score_std"]].max(axis=1)
    out["tcn_minus_pinn"] = out["tcn_score_mean"] - out["pinn_score_mean"]
    lower = float(out["combined_score"].quantile(1.0 / 3.0))
    upper = float(out["combined_score"].quantile(2.0 / 3.0))

    difficulty_label = np.select(
        [out["combined_score"].le(lower), out["combined_score"].ge(upper)],
        ["공통 난도 높음", "공통 학습 용이"],
        default="중간 난도",
    )
    representation_label = np.select(
        [out["tcn_minus_pinn"].ge(0.015), out["tcn_minus_pinn"].le(-0.015)],
        ["TCN 우세", "PINN 우세"],
        default="표현 차이 작음",
    )
    stability_label = np.where(out["max_fold_std"].ge(0.020), "연도 민감", "안정")
    out["difficulty_class"] = difficulty_label
    out["representation_class"] = representation_label
    out["stability_class"] = stability_label
    out["difficulty_rank"] = out["combined_score"].rank(
        method="min", ascending=True
    ).astype(int)
    return out.sort_values(["difficulty_rank", "turbine_id"]).reset_index(drop=True)


def turbine_label(turbine_id: str) -> str:
    return turbine_id.replace("vestas_wtg", "V").replace("unison_wtg", "U")


def plot_difficulty(difficulty: pd.DataFrame, output_path: Path) -> None:
    plot = difficulty.sort_values("combined_score", ascending=True).reset_index(drop=True)
    labels = [turbine_label(value) for value in plot["turbine_id"]]
    y = np.arange(len(plot), dtype=float)
    group_colors = {
        "kpx_group_1": "#2878B5",
        "kpx_group_2": "#E07A2D",
        "kpx_group_3": "#339966",
    }

    korean_font_path = Path(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    font_family = "DejaVu Sans"
    if korean_font_path.exists():
        font_manager.fontManager.addfont(str(korean_font_path))
        font_family = font_manager.FontProperties(fname=korean_font_path).get_name()
    plt.rcParams.update(
        {
            "font.family": font_family,
            "axes.unicode_minus": False,
            "font.size": 10,
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 10),
        sharey=True,
        gridspec_kw={"width_ratios": [2.4, 1.25, 1.45], "wspace": 0.08},
    )

    score_axis, gap_axis, epoch_axis = axes
    score_axis.errorbar(
        plot["pinn_score_mean"],
        y + 0.13,
        xerr=plot["pinn_score_std"],
        fmt="o",
        color="#1769AA",
        ecolor="#8FB9D9",
        capsize=2,
        label="PINN 평균 ± fold 표준편차",
    )
    score_axis.errorbar(
        plot["tcn_score_mean"],
        y - 0.13,
        xerr=plot["tcn_score_std"],
        fmt="s",
        color="#D76B1F",
        ecolor="#E9B58F",
        capsize=2,
        label="TCN 평균 ± fold 표준편차",
    )
    score_axis.set_yticks(y, labels)
    for tick, group in zip(score_axis.get_yticklabels(), plot["group"]):
        tick.set_color(group_colors[group])
        tick.set_fontweight("bold")
    score_axis.set_xlabel("터빈 validation score (높을수록 쉬움)")
    score_axis.set_title("학습 성능과 연도 안정성")
    score_axis.grid(axis="x", color="#DDDDDD", linewidth=0.8)
    score_axis.legend(loc="lower right", frameon=False)

    gap = plot["tcn_minus_pinn"].to_numpy(float)
    gap_axis.barh(
        y,
        gap,
        height=0.55,
        color=np.where(gap >= 0.0, "#4C9F70", "#B95C5C"),
    )
    gap_axis.axvline(0.0, color="#333333", linewidth=1.0)
    gap_axis.axvline(0.015, color="#888888", linewidth=0.8, linestyle="--")
    gap_axis.axvline(-0.015, color="#888888", linewidth=0.8, linestyle="--")
    gap_axis.set_xlabel("TCN - PINN")
    gap_axis.set_title("표현별 우세")
    gap_axis.grid(axis="x", color="#E5E5E5", linewidth=0.7)

    epoch_axis.scatter(
        plot["pinn_best_epoch_median"],
        y + 0.13,
        marker="o",
        color="#1769AA",
        label="PINN",
    )
    epoch_axis.scatter(
        plot["tcn_best_epoch_median"],
        y - 0.13,
        marker="s",
        color="#D76B1F",
        label="TCN",
    )
    epoch_axis.set_xscale("symlog", linthresh=5)
    epoch_axis.set_xlabel("median best epoch (symlog)")
    epoch_axis.set_title("수렴 시점")
    epoch_axis.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    epoch_axis.legend(loc="lower right", frameon=False)

    figure.suptitle(
        "터빈별 학습 난이도: PINN teacher 입력 vs TCN weather-only 입력",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.01,
        0.01,
        "터빈명 색상: 파랑=Group 1, 주황=Group 2, 초록=Group 3 | "
        "점수는 outer-year fold의 터빈 단위 지표",
        fontsize=9,
        color="#555555",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if (args.isotonic_submission_input is None) != (
        args.isotonic_submission_output is None
    ):
        raise ValueError(
            "--isotonic-submission-input and --isotonic-submission-output "
            "must be provided together"
        )
    turbine_predictions = build_branch_predictions(args.results_dir)
    tree_predictions = get_or_build_tree_predictions(
        args.tree_best_csv,
        args.tree_predictions,
        args.rebuild_tree,
    )
    group_oof = build_group_oof_predictions(
        turbine_predictions,
        tree_predictions,
        args.labels,
    )
    branch_scores = score_branches(group_oof)
    print("\n=== branch pooled OOF ===")
    print(
        branch_scores[
            [
                "variant",
                "mean_score",
                "mean_nmae",
                "mean_ficr",
                "min_group_score",
            ]
        ].to_string(index=False)
    )

    if args.representative_comparison_output is not None:
        representative_frame = build_representative_residual_frame(
            turbine_predictions,
            group_oof,
        )
        representative_oof = crossfit_representative_residual(
            representative_frame
        )
        representative_comparison = representative_residual_comparison(
            representative_oof
        )
        print("\n=== representative turbine residual OOF: overall ===")
        print(
            representative_comparison.loc[
                representative_comparison["scope"].eq("overall")
            ][
                [
                    "variant",
                    "score",
                    "one_minus_nmae",
                    "weighted_hit6",
                    "weighted_hit8",
                    "ficr",
                ]
            ].to_string(index=False)
        )
        print("\n=== representative turbine residual OOF: nested selection ===")
        print(
            representative_comparison.loc[
                representative_comparison["variant"].eq(
                    "nested_selected_lambda"
                )
                & representative_comparison["scope"].eq("group")
            ].to_string(index=False)
        )
        args.representative_comparison_output.parent.mkdir(
            parents=True, exist_ok=True
        )
        representative_comparison.to_csv(
            args.representative_comparison_output,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"saved {args.representative_comparison_output}")

    isotonic_oof = crossfit_jointmix_isotonic(group_oof)
    comparison = isotonic_comparison(isotonic_oof)
    print("\n=== jointmix isotonic OOF ===")
    print(comparison.to_string(index=False))
    if args.isotonic_comparison_output is not None:
        args.isotonic_comparison_output.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(
            args.isotonic_comparison_output,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"saved {args.isotonic_comparison_output}")
    if args.isotonic_submission_input is not None:
        calibrated_submission, selected_alphas = apply_jointmix_isotonic_submission(
            group_oof,
            args.isotonic_submission_input,
            args.isotonic_submission_output,
        )
        mean_adjustment = np.mean(
            np.abs(
                calibrated_submission[TARGET_COLS].to_numpy(float)
                - pd.read_csv(
                    args.isotonic_submission_input, encoding="utf-8-sig"
                )[TARGET_COLS].to_numpy(float)
            )
        )
        print(
            f"saved {args.isotonic_submission_output}: "
            f"mean_abs_adjustment={mean_adjustment:.6f} "
            f"selected_alphas={selected_alphas}"
        )

    pinn = aggregate_training(args.results_dir / f"{PINN_STEM}_training.csv", "pinn")
    tcn = aggregate_training(args.results_dir / f"{TCN_STEM}_training.csv", "tcn")
    difficulty = classify_turbines(
        pinn.merge(tcn, on=["group", "turbine_id"], validate="one_to_one")
    )
    args.difficulty_output.parent.mkdir(parents=True, exist_ok=True)
    difficulty.to_csv(args.difficulty_output, index=False, encoding="utf-8-sig")
    plot_difficulty(difficulty, args.figure_output)

    columns = [
        "difficulty_rank",
        "turbine_id",
        "combined_score",
        "pinn_score_mean",
        "tcn_score_mean",
        "max_fold_std",
        "tcn_minus_pinn",
        "difficulty_class",
        "representation_class",
        "stability_class",
    ]
    print("\n=== hardest turbines ===")
    print(difficulty[columns].head(6).to_string(index=False))
    print("\n=== easiest turbines ===")
    print(difficulty[columns].tail(6).sort_values("difficulty_rank").to_string(index=False))
    print(f"\nsaved {args.difficulty_output}")
    print(f"saved {args.figure_output}")


if __name__ == "__main__":
    main()
