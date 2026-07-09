import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from xgboost import XGBRegressor

import _bootstrap  # noqa: F401
from tune_power_lgbm_hyperparams import (
    parse_list,
    prepare_fold_cache,
    sample_weight,
    score_one,
)
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS


RESULTS_DIR = Path("results")


def log_uniform(rng, low, high):
    return float(10 ** rng.uniform(np.log10(low), np.log10(high)))


def sample_xgb_params(rng, seed):
    seed = int(seed % (2**32 - 1))
    return {
        "random_state": seed,
        "n_jobs": -1,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_estimators": int(rng.integers(500, 1801)),
        "learning_rate": log_uniform(rng, 0.012, 0.07),
        "max_depth": int(rng.choice([3, 4, 5, 6, 7, 8])),
        "min_child_weight": log_uniform(rng, 1.0, 40.0),
        "subsample": float(rng.uniform(0.60, 0.98)),
        "colsample_bytree": float(rng.uniform(0.55, 0.98)),
        "reg_alpha": log_uniform(rng, 1e-4, 5.0),
        "reg_lambda": log_uniform(rng, 0.05, 50.0),
        "gamma": log_uniform(rng, 1e-5, 5.0),
    }


def sample_extra_params(rng, seed):
    seed = int(seed % (2**32 - 1))
    return {
        "random_state": seed,
        "n_jobs": -1,
        "n_estimators": int(rng.integers(400, 1001)),
        "min_samples_leaf": int(rng.choice([1, 2, 3, 5, 8, 12, 20])),
        "max_features": float(rng.uniform(0.35, 1.0)),
        "max_depth": rng.choice([None, 8, 12, 16, 24, 32]),
        "bootstrap": bool(rng.choice([False, True])),
    }


def sample_rf_params(rng, seed):
    seed = int(seed % (2**32 - 1))
    return {
        "random_state": seed,
        "n_jobs": -1,
        "n_estimators": int(rng.integers(400, 1001)),
        "min_samples_leaf": int(rng.choice([1, 2, 3, 5, 8, 12, 20])),
        "max_features": float(rng.uniform(0.35, 1.0)),
        "max_depth": rng.choice([None, 8, 12, 16, 24, 32]),
        "bootstrap": True,
    }


def sample_policy(rng):
    return {
        "min_output_ratio": float(rng.choice([0.03, 0.05, 0.10, 0.15])),
        "weight_policy": rng.choice(["none", "metric_x2", "actual_sqrt"]),
    }


def make_model(model_type, params):
    if model_type == "xgb":
        return XGBRegressor(**params)
    if model_type == "extra":
        return ExtraTreesRegressor(**params)
    if model_type == "rf":
        return RandomForestRegressor(**params)
    raise ValueError(f"unknown model_type: {model_type}")


def sample_params(model_type, rng, seed):
    if model_type == "xgb":
        return sample_xgb_params(rng, seed)
    if model_type == "extra":
        return sample_extra_params(rng, seed)
    if model_type == "rf":
        return sample_rf_params(rng, seed)
    raise ValueError(f"unknown model_type: {model_type}")


def evaluate_trial(model_type, group, folds, params, policy):
    rows = []
    pred_rows = []
    capacity = GROUP_CAPACITY_KWH[group]
    for fold in folds:
        train_mask = fold["y_train"].to_numpy(float) >= capacity * policy["min_output_ratio"]
        x_train = fold["x_train"].loc[train_mask]
        y_train = fold["y_train"].loc[train_mask]
        if len(y_train) < 500:
            continue
        model = make_model(model_type, params)
        weights = sample_weight(y_train, group, policy["weight_policy"])
        model.fit(x_train, y_train, sample_weight=weights)
        pred = model.predict(fold["x_val"])
        score, nmae, ficr = score_one(fold["y_val"], pred, group)
        rows.append(
            {
                "group": group,
                "pred_year": fold["pred_year"],
                "train_years": fold["train_years"],
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n_train": len(y_train),
                "n_val": len(fold["y_val"]),
            }
        )
        pred_rows.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": pd.to_datetime(fold["time_val"]).to_numpy(),
                    "pred_year": fold["pred_year"],
                    "train_years": fold["train_years"],
                    "train_policy": f"best_{model_type}",
                    "model": model_type,
                    "group": group,
                    "actual": fold["y_val"].to_numpy(float),
                    "pred": pred,
                }
            )
        )
    return rows, pred_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", required=True, choices=["xgb", "extra", "rf"])
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stem", default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"power_{args.model_type}_hyperparams"
    groups = parse_list(args.groups)
    cache = prepare_fold_cache(groups)
    rng = np.random.default_rng(args.seed)

    trial_rows = []
    fold_rows = []
    trial_pred_parts = {}
    for group in groups:
        folds = cache[group]
        print(f"\n=== tune {args.model_type} {group}: folds={len(folds)} trials={args.trials} ===")
        for trial in range(args.trials):
            params = sample_params(args.model_type, rng, args.seed * 1000 + trial)
            policy = sample_policy(rng)
            rows, pred_rows = evaluate_trial(args.model_type, group, folds, params, policy)
            if not rows:
                continue
            scores = pd.DataFrame(rows)
            summary = {
                "model_type": args.model_type,
                "group": group,
                "trial": trial,
                "mean_score": scores["score"].mean(),
                "mean_nmae": scores["nmae"].mean(),
                "mean_ficr": scores["ficr"].mean(),
                "worst_fold": scores["score"].min(),
                "std_score": scores["score"].std(ddof=0),
                **policy,
                **params,
            }
            trial_rows.append(summary)
            for row in rows:
                fold_rows.append({**{"model_type": args.model_type, "trial": trial}, **policy, **params, **row})
            trial_pred_parts[(group, trial)] = pred_rows
            print(
                f"{group} trial={trial:03d} score={summary['mean_score']:.5f} "
                f"ficr={summary['mean_ficr']:.5f} worst={summary['worst_fold']:.5f} policy={policy}"
            )

    trials = pd.DataFrame(trial_rows).sort_values(["group", "mean_score"], ascending=[True, False])
    folds = pd.DataFrame(fold_rows)
    best = trials.groupby("group", as_index=False).head(1).reset_index(drop=True)

    best_pred_parts = []
    for row in best.itertuples():
        for pred_df in trial_pred_parts[(row.group, int(row.trial))]:
            best_pred_parts.append(pred_df)
    predictions = pd.concat(best_pred_parts, ignore_index=True) if best_pred_parts else pd.DataFrame()
    summary = (
        best.groupby("model_type", as_index=False)
        .agg(
            mean_score=("mean_score", "mean"),
            mean_nmae=("mean_nmae", "mean"),
            mean_ficr=("mean_ficr", "mean"),
            worst_group=("mean_score", "min"),
        )
    )

    trials_path = RESULTS_DIR / f"{stem}_trials.csv"
    folds_path = RESULTS_DIR / f"{stem}_folds.csv"
    best_path = RESULTS_DIR / f"{stem}_best.csv"
    pred_path = RESULTS_DIR / f"{stem}_best_predictions.csv"
    summary_path = RESULTS_DIR / f"{stem}_best_summary.csv"
    trials.to_csv(trials_path, index=False, encoding="utf-8-sig")
    folds.to_csv(folds_path, index=False, encoding="utf-8-sig")
    best.to_csv(best_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n=== best ===")
    print(best[["model_type", "group", "trial", "mean_score", "mean_nmae", "mean_ficr", "worst_fold"]].to_string(index=False))
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print(f"saved {trials_path}")
    print(f"saved {pred_path}")
    return trials, folds, best, predictions, summary


if __name__ == "__main__":
    main()
