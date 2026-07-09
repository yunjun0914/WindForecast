import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import _bootstrap  # noqa: F401
from tune_power_lgbm_hyperparams import parse_list, prepare_fold_cache, sample_weight, score_one
from utils.metrics import TARGET_COLS
from utils.tree_feature_profiles import FEATURE_PROFILE_FULL_V2, FEATURE_PROFILES


RESULTS_DIR = Path("results")
PARAM_COLS = [
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
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "min_split_gain",
]


def clean_params(row):
    params = {col: row[col] for col in PARAM_COLS}
    for col in ["random_state", "n_jobs", "verbose", "n_estimators", "num_leaves", "max_depth", "min_child_samples"]:
        params[col] = int(params[col])
    for col in ["learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "min_split_gain"]:
        params[col] = float(params[col])
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-csv", default="results/power_lgbm_hyperparams_v2_l1_20_best.csv")
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--stem", default="power_lgbm_best_v2_l1")
    parser.add_argument("--train-policy-name", default="best_lgbm_v2_l1")
    parser.add_argument("--feature-profile", default=FEATURE_PROFILE_FULL_V2, choices=FEATURE_PROFILES)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = parse_list(args.groups)
    best = pd.read_csv(args.best_csv, encoding="utf-8-sig")
    cache = prepare_fold_cache(groups, feature_profile=args.feature_profile)

    score_rows = []
    pred_rows = []
    for group in groups:
        row = best[best["group"] == group]
        if row.empty:
            print(f"{group}: skip no best row")
            continue
        row = row.iloc[0]
        params = clean_params(row)
        policy = {
            "min_output_ratio": float(row["min_output_ratio"]),
            "weight_policy": row["weight_policy"],
        }
        print(f"\n=== evaluate best {group}: trial={row['trial']} policy={policy} ===")
        for fold in cache[group]:
            y_train = fold["y_train"]
            train_mask = y_train.to_numpy(float) >= float(row["min_output_ratio"]) * {
                "kpx_group_1": 21600,
                "kpx_group_2": 21600,
                "kpx_group_3": 21000,
            }[group]
            x_train = fold["x_train"].loc[train_mask]
            y_used = y_train.loc[train_mask]
            model = LGBMRegressor(**params)
            weights = sample_weight(y_used, group, policy["weight_policy"])
            model.fit(x_train, y_used, sample_weight=weights)
            pred = model.predict(fold["x_val"])
            score, nmae, ficr = score_one(fold["y_val"], pred, group)
            score_rows.append(
                {
                    "pred_year": fold["pred_year"],
                    "train_years": fold["train_years"],
                    "train_policy": args.train_policy_name,
                    "model": "lgbm_best",
                    "feature_profile": args.feature_profile,
                    "group": group,
                    "score": score,
                    "nmae": nmae,
                    "ficr": ficr,
                    "n": len(fold["y_val"]),
                    "n_train": len(y_train),
                    "n_used": len(y_used),
                }
            )
            pred_rows.append(
                pd.DataFrame(
                    {
                        "forecast_kst_dtm": pd.to_datetime(fold["time_val"]).to_numpy(),
                        "pred_year": fold["pred_year"],
                        "train_years": fold["train_years"],
                        "train_policy": args.train_policy_name,
                        "model": "lgbm_best",
                        "feature_profile": args.feature_profile,
                        "group": group,
                        "actual": fold["y_val"].to_numpy(float),
                        "pred": pred,
                    }
                )
            )
            print(f"{group} pred_year={fold['pred_year']}: score={score:.5f}, nmae={nmae:.5f}, ficr={ficr:.5f}")

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    means = (
        scores.groupby(["pred_year", "train_policy", "model", "feature_profile"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"), n=("n", "sum"))
    )
    means["group"] = "fold_mean"
    means["train_years"] = ""
    means["n_train"] = np.nan
    means["n_used"] = np.nan
    scores = pd.concat([scores, means[scores.columns]], ignore_index=True)
    summary = (
        scores[scores["group"] == "fold_mean"]
        .groupby(["train_policy", "model", "feature_profile"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
            n_folds=("score", "count"),
        )
    )

    scores_path = RESULTS_DIR / f"{args.stem}_scores.csv"
    pred_path = RESULTS_DIR / f"{args.stem}_predictions.csv"
    summary_path = RESULTS_DIR / f"{args.stem}_summary.csv"
    scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(pred_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print(f"saved {scores_path}")
    print(f"saved {pred_path}")
    print(f"saved {summary_path}")
    return scores, predictions, summary


if __name__ == "__main__":
    main()
