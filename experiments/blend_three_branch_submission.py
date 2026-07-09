import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS


def load_submission(path, prefix):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    required = ["forecast_id", "forecast_kst_dtm", *TARGET_COLS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[required].copy()
    return out.rename(columns={group: f"{prefix}_{group}" for group in TARGET_COLS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--tcn24", required=True)
    parser.add_argument("--tcn72", required=True)
    parser.add_argument("--tcn168", required=True)
    parser.add_argument("--pinn-weight", type=float, default=0.25)
    parser.add_argument("--tree-weight", type=float, default=0.40)
    parser.add_argument("--tcn-family-weight", type=float, default=0.35)
    parser.add_argument("--tcn24-weight", type=float, default=0.30)
    parser.add_argument("--tcn72-weight", type=float, default=0.40)
    parser.add_argument("--tcn168-weight", type=float, default=0.30)
    parser.add_argument("--output", default="results/submission_pinn25_tree40_tcnfamily35_w24_030_w72_040_w168_030.csv")
    parser.add_argument("--also-update-submission-csv", action="store_true")
    args = parser.parse_args()

    branch_total = args.pinn_weight + args.tree_weight + args.tcn_family_weight
    if abs(branch_total - 1.0) > 1e-9:
        raise ValueError(f"branch weights must sum to 1.0, got {branch_total}")
    tcn_total = args.tcn24_weight + args.tcn72_weight + args.tcn168_weight
    if abs(tcn_total - 1.0) > 1e-9:
        raise ValueError(f"TCN family weights must sum to 1.0, got {tcn_total}")

    out = load_submission(args.pinn, "pinn")
    for path, prefix in [
        (args.tree, "tree"),
        (args.tcn24, "tcn24"),
        (args.tcn72, "tcn72"),
        (args.tcn168, "tcn168"),
    ]:
        comp = load_submission(path, prefix)
        out = out.merge(comp, on=["forecast_id", "forecast_kst_dtm"], how="inner")

    if len(out) != len(pd.read_csv(args.pinn, encoding="utf-8-sig")):
        raise ValueError("row count changed during merge")

    final = out[["forecast_id", "forecast_kst_dtm"]].copy()
    for group in TARGET_COLS:
        tcn_family = (
            args.tcn24_weight * out[f"tcn24_{group}"]
            + args.tcn72_weight * out[f"tcn72_{group}"]
            + args.tcn168_weight * out[f"tcn168_{group}"]
        )
        pred = args.pinn_weight * out[f"pinn_{group}"] + args.tree_weight * out[f"tree_{group}"] + args.tcn_family_weight * tcn_family
        final[group] = np.clip(pred.to_numpy(float), 0.0, GROUP_CAPACITY_KWH[group])

    if final[TARGET_COLS].isna().any().any():
        raise ValueError("non-finite predictions in blended submission")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"saved {output}: {final.shape}")
    print(final[TARGET_COLS].agg(["min", "max", "mean"]).to_string())

    if args.also_update_submission_csv:
        final.to_csv("results/submission.csv", index=False, encoding="utf-8-sig")
        print("saved results/submission.csv")

    return final


if __name__ == "__main__":
    main()
