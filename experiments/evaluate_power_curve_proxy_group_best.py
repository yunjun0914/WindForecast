import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr


RESULTS_DIR = Path("results")


def read_power_curve_predictions(paths):
    parts = []
    for path in paths:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if df.empty:
            continue
        df["source_file"] = Path(path).name
        df["variant"] = df["mode"].astype(str) + "::" + df["proxy_col"].astype(str)
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def read_baseline(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["mode"] = "baseline"
    df["proxy_col"] = "gfs_ws100_speed"
    df["variant"] = "baseline::gfs_ws100_speed"
    return df[["forecast_kst_dtm", "pred_year", "train_years", "mode", "proxy_col", "variant", "group", "actual", "pred"]]


def score_group(actual, pred, group):
    cap = GROUP_CAPACITY_KWH[group]
    pred = np.clip(np.asarray(pred, dtype=float), 0, cap)
    nmae, ficr = group_nmae_ficr(actual, pred, cap)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def score_predictions(df):
    rows = []
    for (pred_year, group), part in df.groupby(["pred_year", "group"]):
        score, nmae, ficr = score_group(part["actual"], part["pred"], group)
        rows.append({"pred_year": pred_year, "group": group, "score": score, "nmae": nmae, "ficr": ficr, "n": len(part)})
    scores = pd.DataFrame(rows)
    fold_means = scores.groupby("pred_year", as_index=False)[["score", "nmae", "ficr"]].mean()
    return {
        "mean_score": fold_means["score"].mean(),
        "mean_nmae": fold_means["nmae"].mean(),
        "mean_ficr": fold_means["ficr"].mean(),
        "worst_fold": fold_means["score"].min(),
        "std_score": fold_means["score"].std(),
    }, scores, fold_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-files",
        default="results/power_curve_proxy_v1_replace_predictions.csv,results/power_curve_proxy_v1_add_selected_predictions.csv",
    )
    parser.add_argument("--baseline", default="results/power_lgbm_best_v2_l1_predictions.csv")
    parser.add_argument("--stem", default="power_curve_proxy_v1_group_best")
    args = parser.parse_args()

    paths = [part.strip() for part in args.prediction_files.split(",") if part.strip()]
    all_pred = read_power_curve_predictions(paths)
    all_pred = pd.concat([all_pred, read_baseline(args.baseline)], ignore_index=True)

    variant_rows = []
    for (variant, group), part in all_pred.groupby(["variant", "group"]):
        summary, _, _ = score_predictions(part)
        variant_rows.append({"variant": variant, "group": group, **summary})
    variant_scores = pd.DataFrame(variant_rows)

    best_rows = variant_scores.sort_values(["group", "mean_score"], ascending=[True, False]).groupby("group").head(1)
    selected_parts = []
    for row in best_rows.itertuples(index=False):
        part = all_pred[(all_pred["group"] == row.group) & (all_pred["variant"] == row.variant)].copy()
        selected_parts.append(part)
    selected = pd.concat(selected_parts, ignore_index=True)
    selected_summary, selected_scores, selected_fold_means = score_predictions(selected)

    summary = pd.DataFrame([{ "mode": "group_best_proxy", **selected_summary }])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    variant_scores.to_csv(RESULTS_DIR / f"{args.stem}_variant_group_scores.csv", index=False, encoding="utf-8-sig")
    best_rows.to_csv(RESULTS_DIR / f"{args.stem}_selected_groups.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")
    selected_scores.to_csv(RESULTS_DIR / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    selected_fold_means.to_csv(RESULTS_DIR / f"{args.stem}_fold_means.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")

    print("=== selected group variants ===")
    print(best_rows.to_string(index=False))
    print("\n=== group-best proxy summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
