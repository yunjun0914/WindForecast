import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr


RESULTS_DIR = Path("results")


def read_pred(path, name):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    return df[["forecast_kst_dtm", "pred_year", "group", "actual", "pred"]].rename(columns={"pred": name})


def load_predictions(args):
    base = read_pred(args.lgbm, "lgbm")
    for name, path in [("xgb", args.xgb), ("extra", args.extra)]:
        other = read_pred(path, name)
        base = base.merge(other[["forecast_kst_dtm", "pred_year", "group", name]], on=["forecast_kst_dtm", "pred_year", "group"], how="inner")
    return base


def score_group(actual, pred, group):
    capacity = GROUP_CAPACITY_KWH[group]
    pred = np.clip(np.asarray(pred, dtype=float), 0, capacity)
    nmae, ficr = group_nmae_ficr(actual, pred, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def score_predictions(df, pred_col):
    rows = []
    for (pred_year, group), part in df.groupby(["pred_year", "group"]):
        score, nmae, ficr = score_group(part["actual"], part[pred_col], group)
        rows.append({"pred_year": pred_year, "group": group, "score": score, "nmae": nmae, "ficr": ficr, "n": len(part)})
    scores = pd.DataFrame(rows)
    means = (
        scores.groupby("pred_year", as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"), n=("n", "sum"))
    )
    means["group"] = "fold_mean"
    scores = pd.concat([scores, means[scores.columns]], ignore_index=True)
    fold_mean = scores[scores["group"] == "fold_mean"]
    summary = pd.DataFrame(
        [
            {
                "mean_score": fold_mean["score"].mean(),
                "mean_nmae": fold_mean["nmae"].mean(),
                "mean_ficr": fold_mean["ficr"].mean(),
                "worst_fold": fold_mean["score"].min(),
                "std_score": fold_mean["score"].std(),
            }
        ]
    )
    return scores, summary


def weight_grid(step):
    values = np.round(np.arange(0, 1 + step / 2, step), 10)
    for w_lgbm, w_xgb in product(values, values):
        w_extra = round(1.0 - float(w_lgbm) - float(w_xgb), 10)
        if w_extra < -1e-9:
            continue
        yield float(w_lgbm), float(w_xgb), max(0.0, w_extra)


def evaluate_global(df, step):
    rows = []
    for w_lgbm, w_xgb, w_extra in weight_grid(step):
        out = df.copy()
        out["ens_pred"] = w_lgbm * out["lgbm"] + w_xgb * out["xgb"] + w_extra * out["extra"]
        _, summary = score_predictions(out, "ens_pred")
        row = summary.iloc[0].to_dict()
        row.update({"mode": "global", "group": "all", "w_lgbm": w_lgbm, "w_xgb": w_xgb, "w_extra": w_extra})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_score", ascending=False)


def evaluate_group_specific(df, step):
    group_best = []
    group_detail = []
    for group in TARGET_COLS:
        part = df[df["group"] == group].copy()
        if part.empty:
            continue
        rows = []
        for w_lgbm, w_xgb, w_extra in weight_grid(step):
            out = part.copy()
            out["ens_pred"] = w_lgbm * out["lgbm"] + w_xgb * out["xgb"] + w_extra * out["extra"]
            group_scores = []
            for pred_year, year_part in out.groupby("pred_year"):
                score, nmae, ficr = score_group(year_part["actual"], year_part["ens_pred"], group)
                group_scores.append({"pred_year": pred_year, "score": score, "nmae": nmae, "ficr": ficr, "n": len(year_part)})
            scores = pd.DataFrame(group_scores)
            row = {
                "mode": "group_specific",
                "group": group,
                "mean_score": scores["score"].mean(),
                "mean_nmae": scores["nmae"].mean(),
                "mean_ficr": scores["ficr"].mean(),
                "worst_fold": scores["score"].min(),
                "std_score": scores["score"].std(ddof=0),
                "w_lgbm": w_lgbm,
                "w_xgb": w_xgb,
                "w_extra": w_extra,
            }
            rows.append(row)
        group_rows = pd.DataFrame(rows).sort_values("mean_score", ascending=False)
        group_detail.append(group_rows)
        group_best.append(group_rows.iloc[0].to_dict())
    return pd.DataFrame(group_best), pd.concat(group_detail, ignore_index=True)


def build_group_specific_predictions(df, group_best):
    parts = []
    for row in group_best.itertuples():
        part = df[df["group"] == row.group].copy()
        part["ens_pred"] = row.w_lgbm * part["lgbm"] + row.w_xgb * part["xgb"] + row.w_extra * part["extra"]
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    scores, summary = score_predictions(out, "ens_pred")
    return out, scores, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lgbm", default="results/power_lgbm_best_v2_l1_predictions.csv")
    parser.add_argument("--xgb", default="results/power_xgb_v1_4_best_predictions.csv")
    parser.add_argument("--extra", default="results/power_extra_v1_8_best_predictions.csv")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--stem", default="tree_family_ensemble_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_predictions(args)
    global_results = evaluate_global(df, args.step)
    group_best, group_detail = evaluate_group_specific(df, args.step)
    group_pred, group_scores, group_summary = build_group_specific_predictions(df, group_best)
    group_summary.insert(0, "mode", "group_specific_best")

    global_results.to_csv(RESULTS_DIR / f"{args.stem}_global_weights.csv", index=False, encoding="utf-8-sig")
    group_detail.to_csv(RESULTS_DIR / f"{args.stem}_group_weight_grid.csv", index=False, encoding="utf-8-sig")
    group_best.to_csv(RESULTS_DIR / f"{args.stem}_group_best_weights.csv", index=False, encoding="utf-8-sig")
    group_scores.to_csv(RESULTS_DIR / f"{args.stem}_group_best_scores.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(RESULTS_DIR / f"{args.stem}_group_best_summary.csv", index=False, encoding="utf-8-sig")
    group_pred.to_csv(RESULTS_DIR / f"{args.stem}_group_best_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n=== global best ===")
    print(global_results.head(10).to_string(index=False))
    print("\n=== group best weights ===")
    print(group_best.to_string(index=False))
    print("\n=== group-specific ensemble summary ===")
    print(group_summary.to_string(index=False))
    return global_results, group_best, group_summary


if __name__ == "__main__":
    main()
