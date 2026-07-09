import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr


RESULTS_DIR = Path("results")


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def expand_residual_grid(df, residual_weights, base_col, residual_col):
    if not residual_weights:
        return df
    if base_col not in df.columns or residual_col not in df.columns:
        raise ValueError(f"residual grid needs columns: {base_col}, {residual_col}")
    parts = []
    source = df.drop_duplicates(["forecast_kst_dtm", "pred_year", "group"]).copy()
    for weight in residual_weights:
        work = source.copy()
        work["variant"] = f"residual_grid_w{weight:.3f}"
        work["pred"] = (
            work[base_col].to_numpy(float)
            + weight * work[residual_col].to_numpy(float) * work["capacity"].to_numpy(float)
        )
        work["pred"] = work["pred"].clip(lower=0.0, upper=work["capacity"])
        parts.append(work)
    return pd.concat(parts, ignore_index=True)


def score_fold(df, delta):
    rows = []
    for (variant, pred_year), fold in df.groupby(["variant", "pred_year"]):
        group_rows = []
        for group, part in fold.groupby("group"):
            metric = part[part["actual"] >= part["capacity"] * 0.10]
            if len(metric) == 0:
                continue
            pred = np.clip(
                metric["pred"].to_numpy(float) + delta * GROUP_CAPACITY_KWH[group],
                0.0,
                GROUP_CAPACITY_KWH[group],
            )
            nmae, ficr = group_nmae_ficr(metric["actual"], pred, GROUP_CAPACITY_KWH[group])
            score = 0.5 * (1.0 - nmae) + 0.5 * ficr
            row = {
                "variant": variant,
                "delta": delta,
                "pred_year": int(pred_year),
                "group": group,
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n": len(metric),
            }
            rows.append(row)
            group_rows.append(row)
        rows.append(
            {
                "variant": variant,
                "delta": delta,
                "pred_year": int(pred_year),
                "group": "fold_mean",
                "score": float(np.mean([row["score"] for row in group_rows])),
                "nmae": float(np.mean([row["nmae"] for row in group_rows])),
                "ficr": float(np.mean([row["ficr"] for row in group_rows])),
                "n": int(sum(row["n"] for row in group_rows)),
            }
        )
    return rows


def summarize(scores):
    fold = scores[scores["group"].eq("fold_mean")]
    return (
        fold.groupby(["variant", "delta"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
        )
        .sort_values(["mean_score", "mean_nmae"], ascending=[False, True])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument("--deltas", default="0,0.0175")
    parser.add_argument("--residual-weights", default="")
    parser.add_argument("--base-col", default="base_pred")
    parser.add_argument("--residual-col", default="tcn_residual_ratio")
    parser.add_argument("--stem", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions_csv, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    if "capacity" not in df.columns:
        df["capacity"] = df["group"].map(GROUP_CAPACITY_KWH).astype(float)
    residual_weights = parse_float_list(args.residual_weights)
    df = expand_residual_grid(df, residual_weights, args.base_col, args.residual_col)

    rows = []
    for delta in parse_float_list(args.deltas):
        rows.extend(score_fold(df, delta))
    scores = pd.DataFrame(rows)
    summary = summarize(scores)

    stem = args.stem
    if stem is None:
        stem = Path(args.predictions_csv).stem + "_delta_eval"
    scores.to_csv(RESULTS_DIR / f"{stem}_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / f"{stem}_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
