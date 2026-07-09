import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr


RESULTS_DIR = Path("results")
KEYS = ["forecast_kst_dtm", "pred_year", "group"]


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_oof(path, pred_col):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    required = [*KEYS, "actual", "pred"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[required].copy()
    out = out.rename(columns={"pred": pred_col})
    return out.drop_duplicates(KEYS)


def merge_oof(base_csv, extra_csv):
    base = load_oof(base_csv, "base_pred")
    extra = load_oof(extra_csv, "extra_pred")
    merged = base.merge(extra, on=KEYS, how="inner", suffixes=("_base", "_extra"))
    if merged.empty:
        raise ValueError("no overlapping OOF rows")
    actual_delta = np.abs(merged["actual_base"].to_numpy(float) - merged["actual_extra"].to_numpy(float)).max()
    if actual_delta > 1e-2:
        raise ValueError(f"actual mismatch after merge: max delta={actual_delta}")
    merged = merged.rename(columns={"actual_base": "actual"}).drop(columns=["actual_extra"])
    merged["capacity"] = merged["group"].map(GROUP_CAPACITY_KWH).astype(float)
    return merged


def score_fold(df, pred_col, variant):
    rows = []
    for pred_year, fold in df.groupby("pred_year"):
        group_rows = []
        for group, part in fold.groupby("group"):
            pred = np.clip(part[pred_col].to_numpy(float), 0.0, GROUP_CAPACITY_KWH[group])
            nmae, ficr = group_nmae_ficr(part["actual"], pred, GROUP_CAPACITY_KWH[group])
            score = 0.5 * (1.0 - nmae) + 0.5 * ficr
            row = {
                "variant": variant,
                "pred_year": int(pred_year),
                "group": group,
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n_rows": len(part),
            }
            rows.append(row)
            group_rows.append(row)
        rows.append(
            {
                "variant": variant,
                "pred_year": int(pred_year),
                "group": "fold_mean",
                "score": float(np.mean([row["score"] for row in group_rows])),
                "nmae": float(np.mean([row["nmae"] for row in group_rows])),
                "ficr": float(np.mean([row["ficr"] for row in group_rows])),
                "n_rows": int(sum(row["n_rows"] for row in group_rows)),
            }
        )
    return rows


def summarize(scores):
    fold = scores[scores["group"].eq("fold_mean")]
    return (
        fold.groupby("variant", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
            n_folds=("score", "count"),
        )
        .sort_values(["mean_score", "mean_nmae"], ascending=[False, True])
    )


def residual_correlations(df):
    rows = []
    work = df.copy()
    work["base_residual"] = work["actual"] - work["base_pred"]
    work["extra_residual"] = work["actual"] - work["extra_pred"]
    metric = work[work["actual"] >= work["capacity"] * 0.10].copy()
    for name, part in [("overall", metric), *list(metric.groupby("group"))]:
        if len(part) < 3:
            corr = np.nan
        else:
            corr = float(np.corrcoef(part["base_residual"], part["extra_residual"])[0, 1])
        rows.append({"scope": name, "rows": len(part), "residual_corr": corr})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", required=True)
    parser.add_argument("--extra-csv", required=True)
    parser.add_argument("--extra-weights", default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--stem", default="oof_branch_comparison")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = merge_oof(args.base_csv, args.extra_csv)

    score_rows = []
    score_rows.extend(score_fold(df.assign(pred=df["base_pred"]), "pred", "base"))
    score_rows.extend(score_fold(df.assign(pred=df["extra_pred"]), "pred", "extra"))
    for weight in parse_float_list(args.extra_weights):
        blend = df.copy()
        blend["pred"] = ((1.0 - weight) * blend["base_pred"] + weight * blend["extra_pred"]).clip(
            lower=0.0, upper=blend["capacity"]
        )
        score_rows.extend(score_fold(blend, "pred", f"blend_extra_w{weight:.2f}"))

    scores = pd.DataFrame(score_rows)
    summary = summarize(scores)
    corr = residual_correlations(df)

    scores_path = RESULTS_DIR / f"{args.stem}_scores.csv"
    summary_path = RESULTS_DIR / f"{args.stem}_summary.csv"
    corr_path = RESULTS_DIR / f"{args.stem}_residual_corr.csv"
    scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    corr.to_csv(corr_path, index=False, encoding="utf-8-sig")
    print("=== summary ===")
    print(summary.to_string(index=False))
    print("\n=== residual correlations ===")
    print(corr.to_string(index=False))
    print(f"saved {scores_path}")
    print(f"saved {summary_path}")
    print(f"saved {corr_path}")
    return scores, summary, corr


if __name__ == "__main__":
    main()
