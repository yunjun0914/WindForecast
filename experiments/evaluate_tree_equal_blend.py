import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr


RESULTS_DIR = Path("results")


def read_tree_pred(path: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df[["forecast_kst_dtm", "pred_year", "group", "actual", "pred"]].rename(columns={"pred": name})


def load_pinn(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.melt(
        id_vars=["forecast_kst_dtm", "pred_year", "train_years"],
        value_vars=TARGET_COLS,
        var_name="group",
        value_name="pinn",
    )[["forecast_kst_dtm", "pred_year", "group", "pinn"]]


def score_group(actual, pred, group):
    capacity = GROUP_CAPACITY_KWH[group]
    pred = np.clip(np.asarray(pred, dtype=float), 0, capacity)
    nmae, ficr = group_nmae_ficr(actual, pred, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def evaluate(df: pd.DataFrame, pred_col: str):
    rows = []
    for (pred_year, group), part in df.groupby(["pred_year", "group"]):
        score, nmae, ficr = score_group(part["actual"], part[pred_col], group)
        rows.append({"pred_year": pred_year, "group": group, "score": score, "nmae": nmae, "ficr": ficr, "n": len(part)})

    scores = pd.DataFrame(rows)
    fold_means = scores.groupby("pred_year", as_index=False)[["score", "nmae", "ficr"]].mean()
    group_means = scores.groupby("group", as_index=False)[["score", "nmae", "ficr"]].mean()
    summary = {
        "mean_score": fold_means["score"].mean(),
        "mean_nmae": fold_means["nmae"].mean(),
        "mean_ficr": fold_means["ficr"].mean(),
        "worst_fold": fold_means["score"].min(),
        "std_score": fold_means["score"].std(),
    }
    return scores, fold_means, group_means, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lgbm", default="results/power_lgbm_best_v2_l1_predictions.csv")
    parser.add_argument("--xgb", default="results/power_xgb_v1_4_best_predictions.csv")
    parser.add_argument("--extra", default="results/power_extra_v1_8_best_predictions.csv")
    parser.add_argument("--pinn", default="results/pinn_effective_grid_g1_year_bagging_lgbm_time_oof_oof_predictions.csv")
    parser.add_argument("--stem", default="tree_equal_blend_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    key = ["forecast_kst_dtm", "pred_year", "group"]
    df = read_tree_pred(args.lgbm, "lgbm")
    df = df.merge(read_tree_pred(args.xgb, "xgb")[key + ["xgb"]], on=key, how="inner")
    df = df.merge(read_tree_pred(args.extra, "extra")[key + ["extra"]], on=key, how="inner")
    df = df.merge(load_pinn(args.pinn), on=key, how="inner")

    variants = {
        "tree_lgbm": df["lgbm"],
        "tree_lgbm_xgb50": 0.5 * df["lgbm"] + 0.5 * df["xgb"],
        "tree_lgbm_extra50": 0.5 * df["lgbm"] + 0.5 * df["extra"],
        "tree_equal3": (df["lgbm"] + df["xgb"] + df["extra"]) / 3.0,
    }

    summary_rows = []
    group_rows = []
    fold_rows = []
    pred_cols = []
    for name, pred in variants.items():
        df[name] = pred
        df[f"blend_{name}"] = 0.5 * df["pinn"] + 0.5 * df[name]
        pred_cols.extend([name, f"blend_{name}"])

        for mode, col in [("TREE", name), ("PINN50_TREE50", f"blend_{name}")]:
            scores, fold_means, group_means, summary = evaluate(df, col)
            summary_rows.append({"mode": mode, "variant": name, **summary})
            group_means.insert(0, "variant", name)
            group_means.insert(0, "mode", mode)
            group_rows.append(group_means)
            fold_means.insert(0, "variant", name)
            fold_means.insert(0, "mode", mode)
            fold_rows.append(fold_means)

    summary_df = pd.DataFrame(summary_rows).sort_values(["mode", "mean_score"], ascending=[True, False])
    group_df = pd.concat(group_rows, ignore_index=True).sort_values(["mode", "variant", "group"])
    fold_df = pd.concat(fold_rows, ignore_index=True).sort_values(["mode", "variant", "pred_year"])
    pred_df = df[key + ["actual", "pinn", "lgbm", "xgb", "extra"] + pred_cols]

    summary_df.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    group_df.to_csv(RESULTS_DIR / f"{args.stem}_group_scores.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(RESULTS_DIR / f"{args.stem}_fold_scores.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")

    print("=== summary ===")
    print(summary_df.to_string(index=False))
    print("\n=== PINN50_TREE50 group detail ===")
    print(group_df[group_df["mode"] == "PINN50_TREE50"].to_string(index=False))


if __name__ == "__main__":
    main()
