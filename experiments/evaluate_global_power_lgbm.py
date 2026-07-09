import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import _bootstrap  # noqa: F401
from tune_power_lgbm_hyperparams import prepare_fold_cache
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr
from utils.power_curve import GROUP_N_TURBINES


RESULTS_DIR = Path("results")
GROUP_META = {
    "kpx_group_1": {"group_id": 0, "manufacturer_id": 0, "is_vestas": 1, "is_unison": 0},
    "kpx_group_2": {"group_id": 1, "manufacturer_id": 0, "is_vestas": 1, "is_unison": 0},
    "kpx_group_3": {"group_id": 2, "manufacturer_id": 1, "is_vestas": 0, "is_unison": 1},
}


def parse_float_list(value):
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def log_uniform(rng, low, high):
    return float(10 ** rng.uniform(np.log10(low), np.log10(high)))


def sample_params(rng, seed):
    return {
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
        "objective": "regression_l1",
        "n_estimators": int(rng.integers(700, 2201)),
        "learning_rate": log_uniform(rng, 0.012, 0.055),
        "num_leaves": int(rng.choice([24, 32, 48, 64, 96, 128])),
        "max_depth": int(rng.choice([5, 6, 7, 8, 10, -1])),
        "min_child_samples": int(rng.choice([60, 80, 120, 160, 220, 300])),
        "subsample": float(rng.uniform(0.65, 0.98)),
        "colsample_bytree": float(rng.uniform(0.60, 0.98)),
        "reg_alpha": log_uniform(rng, 1e-4, 5.0),
        "reg_lambda": log_uniform(rng, 1e-3, 20.0),
        "min_split_gain": log_uniform(rng, 1e-5, 0.2),
    }


def add_group_features(x, group):
    out = x.copy()
    meta = GROUP_META[group]
    for key, value in meta.items():
        out[key] = value
    out["capacity_kwh"] = GROUP_CAPACITY_KWH[group]
    out["n_turbines"] = GROUP_N_TURBINES[group]
    return out


def group_weight(n_rows, n_groups):
    if n_rows <= 0:
        return 1.0
    return 1.0 / (n_rows * n_groups)


def fit_predict_fold(cache, pred_year, params):
    x_parts, y_parts, w_parts = [], [], []
    val_parts = []
    groups_in_fold = [group for group in TARGET_COLS if any(fold["pred_year"] == pred_year for fold in cache[group])]
    n_groups = len(TARGET_COLS)

    for group in TARGET_COLS:
        fold_rows = [fold for fold in cache[group] if fold["pred_year"] == pred_year]
        if not fold_rows:
            continue
        fold = fold_rows[0]
        capacity = GROUP_CAPACITY_KWH[group]
        y = fold["y_train"].to_numpy(float)
        mask = y >= capacity * 0.10
        x_group = add_group_features(fold["x_train"].loc[mask], group)
        y_norm = fold["y_train"].loc[mask].to_numpy(float) / capacity
        base_weight = 0.5 + np.sqrt(np.clip(y_norm, 0, 1))
        base_weight = base_weight * group_weight(len(y_norm), n_groups)
        x_parts.append(x_group)
        y_parts.append(y_norm)
        w_parts.append(base_weight)

        val_parts.append(
            {
                "group": group,
                "fold": fold,
                "x_val": add_group_features(fold["x_val"], group),
            }
        )

    if not x_parts:
        return []

    x_train = pd.concat(x_parts, ignore_index=True)
    y_train = np.concatenate(y_parts)
    weights = np.concatenate(w_parts)
    feature_cols = sorted(x_train.columns)
    x_train = x_train.reindex(columns=feature_cols, fill_value=0)

    model = LGBMRegressor(**params)
    model.fit(x_train, y_train, sample_weight=weights)

    pred_rows = []
    for item in val_parts:
        group = item["group"]
        fold = item["fold"]
        x_val = item["x_val"].reindex(columns=feature_cols, fill_value=0)
        pred_ratio = np.clip(model.predict(x_val), 0, 1)
        pred = pred_ratio * GROUP_CAPACITY_KWH[group]
        pred_rows.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": pd.to_datetime(fold["time_val"]).to_numpy(),
                    "pred_year": pred_year,
                    "train_years": fold["train_years"],
                    "train_policy": "global_lgbm",
                    "model": "global_lgbm",
                    "group": group,
                    "actual": fold["y_val"].to_numpy(float),
                    "pred": pred,
                }
            )
        )
    return pred_rows


def score_predictions(predictions, model_name):
    rows = []
    for (pred_year, group), df in predictions.groupby(["pred_year", "group"]):
        score, nmae, ficr = score_group(df["actual"], df["pred"], group)
        rows.append(
            {
                "pred_year": pred_year,
                "train_policy": model_name,
                "model": model_name,
                "group": group,
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n": len(df),
            }
        )
    scores = pd.DataFrame(rows)
    means = (
        scores.groupby(["pred_year", "train_policy", "model"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"), n=("n", "sum"))
    )
    means["group"] = "fold_mean"
    scores = pd.concat([scores, means[scores.columns]], ignore_index=True)
    summary = (
        scores[scores["group"] == "fold_mean"]
        .groupby(["train_policy", "model"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
            n_folds=("score", "count"),
        )
    )
    return scores, summary


def score_group(actual, pred, group):
    cap = GROUP_CAPACITY_KWH[group]
    pred = np.clip(np.asarray(pred, dtype=float), 0, cap)
    nmae, ficr = group_nmae_ficr(actual, pred, cap)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def blend_with_specific(global_pred, specific_pred, weights, group3_only=True):
    specific = specific_pred.copy()
    specific["forecast_kst_dtm"] = pd.to_datetime(specific["forecast_kst_dtm"])
    global_pred = global_pred.copy()
    global_pred["forecast_kst_dtm"] = pd.to_datetime(global_pred["forecast_kst_dtm"])
    key = ["forecast_kst_dtm", "pred_year", "group"]
    merged = specific[key + ["actual", "pred"]].rename(columns={"pred": "specific_pred"}).merge(
        global_pred[key + ["pred"]].rename(columns={"pred": "global_pred"}), on=key, how="inner"
    )
    rows = []
    pred_frames = []
    for weight in weights:
        out = merged.copy()
        use_global = out["group"].eq("kpx_group_3") if group3_only else pd.Series(True, index=out.index)
        out["pred"] = out["specific_pred"]
        out.loc[use_global, "pred"] = (
            (1 - weight) * out.loc[use_global, "specific_pred"] + weight * out.loc[use_global, "global_pred"]
        )
        scores, summary = score_predictions(
            out[["forecast_kst_dtm", "pred_year", "group", "actual", "pred"]], f"blend_global_{weight:g}"
        )
        for row in scores.to_dict("records"):
            row["global_weight"] = weight
            row["group3_only"] = group3_only
            rows.append(row)
        pred_frames.append(out.assign(global_weight=weight, group3_only=group3_only))
    blend_scores = pd.DataFrame(rows)
    blend_summary = (
        blend_scores[blend_scores["group"] == "fold_mean"]
        .groupby(["global_weight", "group3_only"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
            n_folds=("score", "count"),
        )
        .sort_values("mean_score", ascending=False)
    )
    return blend_scores, blend_summary, pd.concat(pred_frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--stem", default="global_power_lgbm_v1")
    parser.add_argument("--specific-predictions", default="results/power_lgbm_best_v2_l1_predictions.csv")
    parser.add_argument("--blend-weights", default="0,0.25,0.5,0.75,1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cache = prepare_fold_cache(TARGET_COLS)
    trial_rows = []
    all_trial_predictions = {}

    for trial in range(args.trials):
        params = sample_params(rng, args.seed * 1000 + trial)
        pred_parts = []
        for pred_year in [2022, 2023, 2024]:
            pred_parts.extend(fit_predict_fold(cache, pred_year, params))
        predictions = pd.concat(pred_parts, ignore_index=True)
        scores, summary = score_predictions(predictions, "global_lgbm")
        row = summary.iloc[0].to_dict()
        row.update({"trial": trial, **params})
        trial_rows.append(row)
        all_trial_predictions[trial] = predictions
        print(
            f"trial={trial:03d} score={row['mean_score']:.5f} "
            f"ficr={row['mean_ficr']:.5f} worst={row['worst_fold']:.5f}"
        )

    trials = pd.DataFrame(trial_rows).sort_values("mean_score", ascending=False)
    best_trial = int(trials.iloc[0]["trial"])
    best_predictions = all_trial_predictions[best_trial]
    best_scores, best_summary = score_predictions(best_predictions, "global_lgbm")

    specific = pd.read_csv(args.specific_predictions, encoding="utf-8-sig")
    specific = specific[(specific["train_policy"] == "best_lgbm_v2_l1") & (specific["model"] == "lgbm_best")].copy()
    blend_scores, blend_summary, blend_predictions = blend_with_specific(
        best_predictions,
        specific,
        parse_float_list(args.blend_weights),
        group3_only=True,
    )

    trials_path = RESULTS_DIR / f"{args.stem}_trials.csv"
    pred_path = RESULTS_DIR / f"{args.stem}_predictions.csv"
    scores_path = RESULTS_DIR / f"{args.stem}_scores.csv"
    summary_path = RESULTS_DIR / f"{args.stem}_summary.csv"
    blend_scores_path = RESULTS_DIR / f"{args.stem}_group3_blend_scores.csv"
    blend_summary_path = RESULTS_DIR / f"{args.stem}_group3_blend_summary.csv"
    blend_pred_path = RESULTS_DIR / f"{args.stem}_group3_blend_predictions.csv"

    trials.to_csv(trials_path, index=False, encoding="utf-8-sig")
    best_predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    best_scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    best_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    blend_scores.to_csv(blend_scores_path, index=False, encoding="utf-8-sig")
    blend_summary.to_csv(blend_summary_path, index=False, encoding="utf-8-sig")
    blend_predictions.to_csv(blend_pred_path, index=False, encoding="utf-8-sig")

    print("\n=== global best ===")
    print(best_summary.to_string(index=False))
    print("\n=== group3 blend summary ===")
    print(blend_summary.to_string(index=False))
    print(f"saved {trials_path}")
    print(f"saved {pred_path}")
    print(f"saved {blend_summary_path}")
    return trials, best_predictions, best_summary, blend_summary


if __name__ == "__main__":
    main()
