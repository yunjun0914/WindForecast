import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS


RESULTS_DIR = Path("results")


def safe_corr(x, y, method="pearson"):
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))
    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or x.std(ddof=0) == 0 or y.std(ddof=0) == 0:
        return np.nan
    if method == "spearman":
        x = x.rank()
        y = y.rank()
    return float(x.corr(y))


def sign_agreement(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    if mask.sum() == 0:
        return np.nan
    return float((np.sign(x[mask]) == np.sign(y[mask])).mean())


def slope_true_on_pred(true_resid, pred_resid):
    y = np.asarray(true_resid, dtype=float)
    x = np.asarray(pred_resid, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.var(x) == 0:
        return np.nan
    return float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))


def summarize(part, candidate):
    true_resid = part["true_residual"].to_numpy(float)
    base_error = part["base_error"].to_numpy(float)
    pred = part[candidate].to_numpy(float)
    return {
        "candidate": candidate,
        "n": len(part),
        "pearson_true": safe_corr(true_resid, pred, "pearson"),
        "spearman_true": safe_corr(true_resid, pred, "spearman"),
        "pearson_base_error": safe_corr(base_error, pred, "pearson"),
        "sign_agree_true": sign_agreement(true_resid, pred),
        "sign_agree_base_error": sign_agreement(base_error, pred),
        "slope_true_on_pred": slope_true_on_pred(true_resid, pred),
        "true_resid_mean": float(np.mean(true_resid)),
        "pred_resid_mean": float(np.mean(pred)),
        "true_resid_std": float(np.std(true_resid)),
        "pred_resid_std": float(np.std(pred)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="results/residual_tree_correction_v1_predictions.csv")
    parser.add_argument("--stem", default="residual_correlation_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.predictions, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df["capacity"] = df["group"].map(GROUP_CAPACITY_KWH).astype(float)
    df["true_residual"] = df["actual"].astype(float) - df["base_pred"].astype(float)
    df["base_error"] = -df["true_residual"]
    df["metric_valid"] = df["actual"].astype(float) >= df["capacity"] * 0.10
    df["abs_true_resid_ratio"] = np.abs(df["true_residual"]) / df["capacity"]

    candidates = [col for col in ["resid_xgb", "resid_extra", "resid_mean"] if col in df.columns]
    rows = []
    for scope_name, scope_df in [
        ("all", df),
        ("metric_valid", df[df["metric_valid"]]),
        ("large_residual_ge_6pct", df[df["abs_true_resid_ratio"] >= 0.06]),
        ("large_residual_ge_8pct", df[df["abs_true_resid_ratio"] >= 0.08]),
    ]:
        for candidate in candidates:
            rows.append({"scope": scope_name, "group": "all", "pred_year": "all", **summarize(scope_df, candidate)})
            for group, part in scope_df.groupby("group"):
                rows.append({"scope": scope_name, "group": group, "pred_year": "all", **summarize(part, candidate)})
            for (pred_year, group), part in scope_df.groupby(["pred_year", "group"]):
                rows.append({"scope": scope_name, "group": group, "pred_year": pred_year, **summarize(part, candidate)})

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")

    overall = summary[(summary["scope"].isin(["all", "metric_valid"])) & (summary["group"] == "all") & (summary["pred_year"] == "all")]
    valid_by_group = summary[(summary["scope"] == "metric_valid") & (summary["group"].isin(TARGET_COLS)) & (summary["pred_year"] == "all")]
    valid_by_year_group = summary[(summary["scope"] == "metric_valid") & (summary["group"].isin(TARGET_COLS)) & (summary["pred_year"] != "all")]

    print("=== overall ===")
    print(
        overall[
            [
                "scope",
                "candidate",
                "n",
                "pearson_true",
                "spearman_true",
                "pearson_base_error",
                "sign_agree_true",
                "slope_true_on_pred",
                "true_resid_mean",
                "pred_resid_mean",
            ]
        ].to_string(index=False)
    )
    print("\n=== metric-valid by group ===")
    print(
        valid_by_group[
            [
                "group",
                "candidate",
                "n",
                "pearson_true",
                "spearman_true",
                "pearson_base_error",
                "sign_agree_true",
                "slope_true_on_pred",
            ]
        ].to_string(index=False)
    )
    print("\n=== metric-valid by year/group, resid_mean ===")
    print(
        valid_by_year_group[valid_by_year_group["candidate"] == "resid_mean"][
            [
                "pred_year",
                "group",
                "n",
                "pearson_true",
                "spearman_true",
                "pearson_base_error",
                "sign_agree_true",
                "slope_true_on_pred",
            ]
        ].to_string(index=False)
    )
    print(f"saved results/{args.stem}_summary.csv")


if __name__ == "__main__":
    main()
