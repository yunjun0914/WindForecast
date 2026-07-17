from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr, pooled_oof_summary


KEYS = ["forecast_kst_dtm", "group", "pred_year"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-variant", default="baseline_best")
    parser.add_argument("--old-tcn", type=Path, required=True)
    parser.add_argument("--new-tcn", type=Path, required=True)
    parser.add_argument("--tcn-weight", type=float, default=0.45)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="tcn_scada_cutin_features_replacement_v1")
    return parser.parse_args()


def load_variant(path: Path, variant: str, pred_name: str) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig")
    target = "official_target" if "official_target" in table else "actual"
    selected = table.loc[table["variant"].eq(variant), [*KEYS, target, "pred"]].copy()
    if selected.empty or selected.duplicated(KEYS).any():
        raise ValueError(f"Invalid variant={variant} in {path}")
    selected["forecast_kst_dtm"] = pd.to_datetime(selected["forecast_kst_dtm"])
    return selected.rename(columns={target: f"actual_{pred_name}", "pred": pred_name})


def fold_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, group, pred_year), part in predictions.groupby(
        ["variant", "group", "pred_year"], sort=False
    ):
        nmae, ficr = group_nmae_ficr(
            part["official_target"], part["pred"], GROUP_CAPACITY_KWH[group]
        )
        rows.append(
            {
                "variant": variant,
                "group": group,
                "pred_year": int(pred_year),
                "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                "nmae": nmae,
                "ficr": ficr,
                "n_rows": len(part),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    baseline = load_variant(args.baseline, args.baseline_variant, "baseline")
    old = load_variant(args.old_tcn, "pure_band", "old_tcn")
    new = load_variant(args.new_tcn, "pure_band", "new_tcn")
    table = baseline.merge(old, on=KEYS, how="left", validate="one_to_one").merge(
        new, on=KEYS, how="left", validate="one_to_one"
    )
    if table[["old_tcn", "new_tcn"]].isna().any().any():
        raise ValueError("TCN prediction coverage does not include every baseline row")
    for column in ["actual_old_tcn", "actual_new_tcn"]:
        absolute_delta = np.abs(
            table["actual_baseline"].to_numpy(float) - table[column].to_numpy(float)
        )
        tolerance = 0.02 * table["group"].map(GROUP_CAPACITY_KWH).to_numpy(float)
        if np.any(absolute_delta > tolerance):
            raise ValueError(f"Target mismatch {column}: {np.nanmax(absolute_delta)}")
        if np.nanmax(absolute_delta) > 1e-2:
            print(
                f"warning: {column} official-label max_delta="
                f"{np.nanmax(absolute_delta):.6f}; scoring official target",
                flush=True,
            )

    parts = []
    for alpha in [0.0, 0.25, 0.50, 0.75, 1.0]:
        candidate_tcn = (1.0 - alpha) * table["old_tcn"].to_numpy(float) + alpha * table[
            "new_tcn"
        ].to_numpy(float)
        prediction = table["baseline"].to_numpy(float) + args.tcn_weight * (
            candidate_tcn - table["old_tcn"].to_numpy(float)
        )
        capacity = table["group"].map(GROUP_CAPACITY_KWH).to_numpy(float)
        prediction = np.clip(prediction, 0.10 * capacity, capacity)
        parts.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": table["forecast_kst_dtm"],
                    "group": table["group"],
                    "pred_year": table["pred_year"],
                    "official_target": table["actual_baseline"],
                    "pred": prediction,
                    "variant": f"cutin_tcn_alpha_{alpha:g}".replace(".", "p"),
                }
            )
        )
    predictions = pd.concat(parts, ignore_index=True)
    summary, pooled = pooled_oof_summary(predictions)
    folds = fold_scores(predictions)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        args.results_dir / f"{args.stem}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.results_dir / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig"
    )
    pooled.to_csv(
        args.results_dir / f"{args.stem}_pooled_group_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    folds.to_csv(
        args.results_dir / f"{args.stem}_fold_scores.csv", index=False, encoding="utf-8-sig"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
