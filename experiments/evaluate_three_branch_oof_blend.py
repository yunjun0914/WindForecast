import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr


RESULTS_DIR = Path("results")
KEYS = ["forecast_kst_dtm", "pred_year", "group"]


def load_oof(path, pred_col):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    required = [*KEYS, "actual", "pred"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[required].copy()
    return out.rename(columns={"pred": pred_col}).drop_duplicates(KEYS)


def merge_component(base, path, pred_col):
    comp = load_oof(path, pred_col)
    out = base.merge(comp, on=KEYS, how="inner", suffixes=("", "_new"))
    actual_delta = np.abs(out["actual"].to_numpy(float) - out["actual_new"].to_numpy(float)).max()
    if actual_delta > 1e-2:
        raise ValueError(f"actual mismatch for {path}: max delta={actual_delta}")
    return out.drop(columns=["actual_new"])


def load_three_branch(args):
    df = load_oof(args.pinn_csv, "pinn_pred")
    df = merge_component(df, args.tree_csv, "tree_pred")
    df = merge_component(df, args.tcn24_csv, "tcn24_pred")
    df = merge_component(df, args.tcn72_csv, "tcn72_pred")
    df = merge_component(df, args.tcn168_csv, "tcn168_pred")
    df["capacity"] = df["group"].map(GROUP_CAPACITY_KWH).astype(float)
    total = args.tcn24_weight + args.tcn72_weight + args.tcn168_weight
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"TCN family weights must sum to 1.0, got {total}")
    df["tcn_family_pred"] = (
        args.tcn24_weight * df["tcn24_pred"]
        + args.tcn72_weight * df["tcn72_pred"]
        + args.tcn168_weight * df["tcn168_pred"]
    ).clip(lower=0.0, upper=df["capacity"])
    return df


def score_fold(df, pred_col, variant, weights):
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
                **weights,
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
                **weights,
            }
        )
    return rows


def summarize(scores):
    fold = scores[scores["group"].eq("fold_mean")]
    return (
        fold.groupby(["variant", "pinn_weight", "tree_weight", "tcn_family_weight"], as_index=False)
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


def add_variant(rows, df, pred_col, variant, pinn_w, tree_w, tcn_w):
    rows.extend(
        score_fold(
            df,
            pred_col,
            variant,
            {
                "pinn_weight": pinn_w,
                "tree_weight": tree_w,
                "tcn_family_weight": tcn_w,
            },
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn-csv", required=True)
    parser.add_argument("--tree-csv", required=True)
    parser.add_argument("--tcn24-csv", required=True)
    parser.add_argument("--tcn72-csv", required=True)
    parser.add_argument("--tcn168-csv", required=True)
    parser.add_argument("--tcn24-weight", type=float, default=0.30)
    parser.add_argument("--tcn72-weight", type=float, default=0.40)
    parser.add_argument("--tcn168-weight", type=float, default=0.30)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--stem", default="three_branch_oof_blend")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_three_branch(args)

    score_rows = []
    add_variant(score_rows, df.assign(pred=df["pinn_pred"]), "pred", "pinn_only", 1.0, 0.0, 0.0)
    add_variant(score_rows, df.assign(pred=df["tree_pred"]), "pred", "tree_only", 0.0, 1.0, 0.0)
    add_variant(score_rows, df.assign(pred=df["tcn_family_pred"]), "pred", "tcn_family_only", 0.0, 0.0, 1.0)
    add_variant(score_rows, df.assign(pred=0.5 * df["pinn_pred"] + 0.5 * df["tree_pred"]), "pred", "pinn50_tree50", 0.5, 0.5, 0.0)

    grid = np.arange(0.0, 1.0 + 1e-12, args.step)
    for pinn_w in grid:
        for tree_w in grid:
            tcn_w = 1.0 - pinn_w - tree_w
            if tcn_w < -1e-12:
                continue
            if abs(round(tcn_w / args.step) * args.step - tcn_w) > 1e-9:
                continue
            tcn_w = max(0.0, float(tcn_w))
            pinn_w = float(pinn_w)
            tree_w = float(tree_w)
            pred = (
                pinn_w * df["pinn_pred"]
                + tree_w * df["tree_pred"]
                + tcn_w * df["tcn_family_pred"]
            ).clip(lower=0.0, upper=df["capacity"])
            variant = f"pinn{pinn_w:.2f}_tree{tree_w:.2f}_tcn{tcn_w:.2f}"
            add_variant(score_rows, df.assign(pred=pred), "pred", variant, pinn_w, tree_w, tcn_w)

    scores = pd.DataFrame(score_rows)
    summary = summarize(scores)
    best = summary.iloc[0]
    best_pred = (
        float(best["pinn_weight"]) * df["pinn_pred"]
        + float(best["tree_weight"]) * df["tree_pred"]
        + float(best["tcn_family_weight"]) * df["tcn_family_pred"]
    ).clip(lower=0.0, upper=df["capacity"])
    family = df[KEYS + ["actual", "tcn_family_pred"]].rename(columns={"tcn_family_pred": "pred"})
    family["train_years"] = family["pred_year"].map(lambda y: ",".join(str(year) for year in [2022, 2023, 2024] if year != int(y)))
    family["model_family"] = "seqnn"
    family["model_name"] = f"tcn_family_w24_{args.tcn24_weight:.2f}_w72_{args.tcn72_weight:.2f}_w168_{args.tcn168_weight:.2f}"
    family["is_clipped"] = True
    family = family[["forecast_kst_dtm", "pred_year", "train_years", "model_family", "model_name", "group", "actual", "pred", "is_clipped"]]
    best_oof = df[KEYS + ["actual"]].copy()
    best_oof["pred"] = best_pred
    best_oof["train_years"] = best_oof["pred_year"].map(lambda y: ",".join(str(year) for year in [2022, 2023, 2024] if year != int(y)))
    best_oof["model_family"] = "pinn_tree_seqnn"
    best_oof["model_name"] = str(best["variant"])
    best_oof["is_clipped"] = True
    best_oof = best_oof[["forecast_kst_dtm", "pred_year", "train_years", "model_family", "model_name", "group", "actual", "pred", "is_clipped"]]

    scores_path = RESULTS_DIR / f"{args.stem}_scores.csv"
    summary_path = RESULTS_DIR / f"{args.stem}_summary.csv"
    best_oof_path = RESULTS_DIR / f"{args.stem}_best_oof.csv"
    family_path = RESULTS_DIR / f"oof_tcn_family_w24_{args.tcn24_weight:.2f}_w72_{args.tcn72_weight:.2f}_w168_{args.tcn168_weight:.2f}.csv"
    scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    best_oof.to_csv(best_oof_path, index=False, encoding="utf-8-sig")
    family.to_csv(family_path, index=False, encoding="utf-8-sig")
    print("=== top summary ===")
    print(summary.head(30).to_string(index=False))
    print(f"saved {scores_path}")
    print(f"saved {summary_path}")
    print(f"saved {best_oof_path}")
    print(f"saved {family_path}")
    return scores, summary, best_oof, family


if __name__ == "__main__":
    main()
