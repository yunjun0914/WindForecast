import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr


RESULTS_DIR = Path("results")


def parse_float_list(value):
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def score_group(actual, pred, group):
    cap = GROUP_CAPACITY_KWH[group]
    pred = np.clip(pred, 0, cap)
    nmae, ficr = group_nmae_ficr(actual, pred, cap)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def load_actual():
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    return labels[["forecast_kst_dtm", *TARGET_COLS]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-predictions", default="results/tree_compact_v2_multi_year_lgbm_policy_predictions.csv")
    parser.add_argument("--pinn-oof", default="results/pinn_effective_grid_g1_year_bagging_oof_predictions.csv")
    parser.add_argument("--train-policy", default="metric_valid")
    parser.add_argument("--tree-model", default="model_mean")
    parser.add_argument("--tree-weights", default="0,0.25,0.4,0.5,0.6,0.75,1.0")
    parser.add_argument("--stem", default="pinn_tree_compact_v2_metric_valid_blend")
    args = parser.parse_args()

    tree_weights = parse_float_list(args.tree_weights)
    actual = load_actual()
    pinn = pd.read_csv(args.pinn_oof, encoding="utf-8-sig")
    pinn["forecast_kst_dtm"] = pd.to_datetime(pinn["forecast_kst_dtm"])
    tree = pd.read_csv(args.tree_predictions, encoding="utf-8-sig")
    tree["forecast_kst_dtm"] = pd.to_datetime(tree["forecast_kst_dtm"])
    tree = tree[(tree["train_policy"] == args.train_policy) & (tree["model"] == args.tree_model)].copy()

    rows = []
    pred_frames = []
    for pred_year in sorted(tree["pred_year"].unique()):
        fold_rows = []
        for group in TARGET_COLS:
            t = tree[(tree["pred_year"] == pred_year) & (tree["group"] == group)][
                ["forecast_kst_dtm", "actual", "pred"]
            ].rename(columns={"pred": "tree_pred"})
            p = pinn[pinn["pred_year"] == pred_year][["forecast_kst_dtm", group]].rename(columns={group: "pinn_pred"})
            merged = t.merge(p, on="forecast_kst_dtm", how="inner")
            if merged.empty:
                continue
            for weight in tree_weights:
                pred = (1 - weight) * merged["pinn_pred"].to_numpy(float) + weight * merged["tree_pred"].to_numpy(float)
                score, nmae, ficr = score_group(merged["actual"], pred, group)
                rows.append(
                    {
                        "pred_year": pred_year,
                        "tree_weight": weight,
                        "group": group,
                        "score": score,
                        "nmae": nmae,
                        "ficr": ficr,
                        "n": len(merged),
                    }
                )
            fold_rows.append(merged.assign(group=group, pred_year=pred_year))
        if fold_rows:
            pred_frames.append(pd.concat(fold_rows, ignore_index=True))

    results = pd.DataFrame(rows)
    means = (
        results.groupby(["pred_year", "tree_weight"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"), n=("n", "sum"))
    )
    means["group"] = "fold_mean"
    results = pd.concat([results, means[results.columns]], ignore_index=True)
    summary = (
        results[results["group"] == "fold_mean"]
        .groupby("tree_weight", as_index=False)
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

    pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    scores_path = RESULTS_DIR / f"{args.stem}_scores.csv"
    summary_path = RESULTS_DIR / f"{args.stem}_summary.csv"
    pred_path = RESULTS_DIR / f"{args.stem}_aligned_predictions.csv"
    results.to_csv(scores_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print(f"saved {scores_path}")
    print(f"saved {summary_path}")
    print(f"saved {pred_path}")
    return results, summary


if __name__ == "__main__":
    main()
