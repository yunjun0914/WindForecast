import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr


RESULTS_DIR = Path("results")
KEYS = ["forecast_kst_dtm", "pred_year", "group"]
DEFAULT_ALIGNED = "results/pinn_lgbmteacher_powerlgbm_v2_l1_blend_aligned_predictions.csv"
DEFAULT_STACKER = "results/oof_stacker_v1_w60_predictions.csv"


def load_base(path, tree_weight):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df["capacity"] = df["group"].map(GROUP_CAPACITY_KWH).astype(float)
    df["base_pred"] = (1.0 - tree_weight) * df["pinn_pred"] + tree_weight * df["tree_pred"]
    df["base_pred"] = df["base_pred"].clip(lower=0.0, upper=df["capacity"])
    return df[KEYS + ["actual", "capacity", "base_pred"]]


def load_stacker(path, variant):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df = df[df["variant"].eq(variant)].copy()
    return df[KEYS + ["pred"]].rename(columns={"pred": "stacker_pred"}).drop_duplicates(KEYS)


def score_fold(df, pred_col, variant, delta):
    rows = []
    pred_year = int(df["pred_year"].iloc[0])
    for group, part in df.groupby("group"):
        metric = part[part["actual"] >= part["capacity"] * 0.10]
        if len(metric) == 0:
            continue
        pred = np.clip(
            metric[pred_col].to_numpy(float) + delta * GROUP_CAPACITY_KWH[group],
            0.0,
            GROUP_CAPACITY_KWH[group],
        )
        nmae, ficr = group_nmae_ficr(metric["actual"], pred, GROUP_CAPACITY_KWH[group])
        score = 0.5 * (1.0 - nmae) + 0.5 * ficr
        rows.append(
            {
                "variant": variant,
                "delta": delta,
                "pred_year": pred_year,
                "group": group,
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n": len(metric),
            }
        )
    group_rows = list(rows)
    rows.append(
        {
            "variant": variant,
            "delta": delta,
            "pred_year": pred_year,
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


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-csv", default=DEFAULT_ALIGNED)
    parser.add_argument("--stacker-csv", default=DEFAULT_STACKER)
    parser.add_argument("--tree-weight", type=float, default=0.6)
    parser.add_argument("--stacker-variants", default="huber_a001_global_metric,huber_a0001_global_metric")
    parser.add_argument("--weights", default="0,0.025,0.05,0.075,0.10,0.125,0.15,0.20")
    parser.add_argument("--deltas", default="0,0.0175")
    parser.add_argument("--stem", default="stacker_blends_v1_w60")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base(args.aligned_csv, args.tree_weight)
    weights = parse_float_list(args.weights)
    deltas = parse_float_list(args.deltas)
    score_rows = []
    pred_parts = []

    for stacker_variant in [item.strip() for item in args.stacker_variants.split(",") if item.strip()]:
        stacker = load_stacker(args.stacker_csv, stacker_variant)
        df = base.merge(stacker, on=KEYS, how="left")
        df["stacker_pred"] = df["stacker_pred"].fillna(df["base_pred"])
        df["stacker_pred"] = df["stacker_pred"].clip(lower=0.0, upper=df["capacity"])

        for weight in weights:
            variant = f"{stacker_variant}_w{weight:.3f}"
            work = df.copy()
            work["blend_pred"] = ((1.0 - weight) * work["base_pred"] + weight * work["stacker_pred"]).clip(
                lower=0.0,
                upper=work["capacity"],
            )
            for delta in deltas:
                for _, fold in work.groupby("pred_year"):
                    score_rows.extend(score_fold(fold, "blend_pred", variant, delta))
            pred_parts.append(work[KEYS + ["actual", "capacity", "base_pred", "stacker_pred", "blend_pred"]].assign(variant=variant))

    scores = pd.DataFrame(score_rows)
    summary = summarize(scores)
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    scores.to_csv(RESULTS_DIR / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")

    print("=== top stacker blends ===")
    print(summary.head(20).to_string(index=False))
    print("\n=== no delta, sorted by nmae with ficr >= base ===")
    base_row = summary[(summary["variant"].str.endswith("_w0.000")) & (summary["delta"].eq(0.0))].head(1)
    min_ficr = float(base_row["mean_ficr"].iloc[0]) if not base_row.empty else -np.inf
    print(summary[(summary["delta"].eq(0.0)) & (summary["mean_ficr"] >= min_ficr)].sort_values(["mean_nmae", "mean_score"]).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
