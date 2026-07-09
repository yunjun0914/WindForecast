import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr


RESULTS_DIR = Path("results")
DEFAULT_ALIGNED = "results/pinn_lgbmteacher_powerlgbm_v2_l1_blend_aligned_predictions.csv"
DEFAULT_XGB = "results/power_xgb_v1_4_best_predictions.csv"
DEFAULT_EXTRA = "results/power_extra_v1_8_best_predictions.csv"
DEFAULT_PRUNED = "results/family_pruned_lgbm_selected_low_v1_predictions.csv"
GROUPS = TARGET_COLS
KEYS = ["forecast_kst_dtm", "pred_year", "group"]


def load_base(path, tree_weight):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df["capacity"] = df["group"].map(GROUP_CAPACITY_KWH).astype(float)
    df["base_pred"] = (1.0 - tree_weight) * df["pinn_pred"] + tree_weight * df["tree_pred"]
    df["base_pred"] = df["base_pred"].clip(lower=0.0, upper=df["capacity"])
    return df[KEYS + ["actual", "capacity", "pinn_pred", "tree_pred", "base_pred"]]


def load_component(path, pred_name, filters=None):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    if filters:
        for col, value in filters.items():
            df = df[df[col].eq(value)]
    if "pred" not in df.columns:
        raise ValueError(f"{path} has no pred column")
    keep = df[KEYS + ["pred"]].copy()
    keep = keep.rename(columns={"pred": pred_name})
    return keep.drop_duplicates(KEYS)


def add_components(base, args):
    out = base.copy()
    components = {
        "xgb_pred": load_component(args.xgb_csv, "xgb_pred"),
        "extra_pred": load_component(args.extra_csv, "extra_pred"),
        "pruned_pred": load_component(args.pruned_csv, "pruned_pred", filters={"variant": "drop_selected_low"}),
    }
    for name, comp in components.items():
        out = out.merge(comp, on=KEYS, how="left")
        out[name] = out[name].fillna(out["base_pred"])
        out[name] = out[name].clip(lower=0.0, upper=out["capacity"])
    return out


def score_fold(df, pred_col, variant, delta=0.0):
    rows = []
    pred_year = int(df["pred_year"].iloc[0])
    for group, part in df.groupby("group"):
        pred = np.clip(part[pred_col].to_numpy(float) + delta * GROUP_CAPACITY_KWH[group], 0.0, GROUP_CAPACITY_KWH[group])
        nmae, ficr = group_nmae_ficr(part["actual"], pred, GROUP_CAPACITY_KWH[group])
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
                "n": len(part),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-csv", default=DEFAULT_ALIGNED)
    parser.add_argument("--xgb-csv", default=DEFAULT_XGB)
    parser.add_argument("--extra-csv", default=DEFAULT_EXTRA)
    parser.add_argument("--pruned-csv", default=DEFAULT_PRUNED)
    parser.add_argument("--tree-weight", type=float, default=0.6)
    parser.add_argument("--max-extra-weight", type=float, default=0.15)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--deltas", default="0,0.0175")
    parser.add_argument("--stem", default="oof_model_blends_v1_w60")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base(args.aligned_csv, args.tree_weight)
    df = add_components(base, args)
    df = df[df["actual"] >= df["capacity"] * 0.10].reset_index(drop=True)
    deltas = [float(item.strip()) for item in args.deltas.split(",") if item.strip()]

    score_rows = []
    pred_rows = []
    grid = np.arange(0.0, args.max_extra_weight + 1e-12, args.step)
    for wx in grid:
        for we in grid:
            for wp in grid:
                total_extra = wx + we + wp
                if total_extra > args.max_extra_weight + 1e-12:
                    continue
                wb = 1.0 - total_extra
                variant = f"base{wb:.3f}_xgb{wx:.3f}_extra{we:.3f}_pruned{wp:.3f}"
                work = df.copy()
                work["blend_pred"] = (
                    wb * work["base_pred"]
                    + wx * work["xgb_pred"]
                    + we * work["extra_pred"]
                    + wp * work["pruned_pred"]
                )
                work["blend_pred"] = work["blend_pred"].clip(lower=0.0, upper=work["capacity"])
                for delta in deltas:
                    for _, fold in work.groupby("pred_year"):
                        score_rows.extend(score_fold(fold, "blend_pred", variant, delta=delta))
                pred_rows.append(
                    {
                        "variant": variant,
                        "base_weight": wb,
                        "xgb_weight": wx,
                        "extra_weight": we,
                        "pruned_weight": wp,
                    }
                )

    scores = pd.DataFrame(score_rows)
    summary = summarize(scores)
    weights = pd.DataFrame(pred_rows).drop_duplicates()
    summary = summary.merge(weights, on="variant", how="left")
    scores.to_csv(RESULTS_DIR / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")

    print("=== top by score ===")
    print(summary.head(20).to_string(index=False))
    print("\n=== top by nmae with ficr >= baseline ===")
    baseline = summary[(summary["variant"].eq("base1.000_xgb0.000_extra0.000_pruned0.000")) & (summary["delta"].eq(0.0))]
    min_ficr = float(baseline["mean_ficr"].iloc[0]) if not baseline.empty else -np.inf
    candidates = summary[summary["mean_ficr"] >= min_ficr].sort_values(["mean_nmae", "mean_score"], ascending=[True, False])
    print(candidates.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
