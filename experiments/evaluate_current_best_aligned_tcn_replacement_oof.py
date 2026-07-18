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
    parser.add_argument("--current-aligned", type=Path, required=True)
    parser.add_argument("--new-tcn", type=Path, required=True)
    parser.add_argument("--new-tcn-variant", required=True)
    parser.add_argument("--tcn-weight", type=float, default=0.45)
    parser.add_argument("--final-floor", type=float, default=0.10)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", required=True)
    return parser.parse_args()


def score_rows(predictions: pd.DataFrame) -> pd.DataFrame:
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
                "bias_kwh": float(np.mean(part["pred"] - part["official_target"])),
                "n_rows": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    current = pd.read_csv(args.current_aligned, encoding="utf-8-sig")
    required_current = [*KEYS, "official_target", "current_final", "old_tcn"]
    missing = [column for column in required_current if column not in current]
    if missing or current.duplicated(KEYS).any():
        raise ValueError(f"Invalid current-aligned input: missing={missing}")
    new = pd.read_csv(args.new_tcn, encoding="utf-8-sig")
    if "variant" in new:
        new = new.loc[new["variant"].eq(args.new_tcn_variant)].copy()
    required_new = [*KEYS, "official_target", "pred"]
    missing = [column for column in required_new if column not in new]
    if missing or new.empty or new.duplicated(KEYS).any():
        raise ValueError(f"Invalid new-TCN input: missing={missing}")

    for table in (current, new):
        table["forecast_kst_dtm"] = pd.to_datetime(table["forecast_kst_dtm"])
    aligned = current[required_current].merge(
        new[required_new].rename(
            columns={"official_target": "official_target_new", "pred": "new_tcn"}
        ),
        on=KEYS,
        validate="one_to_one",
    )
    if len(aligned) != len(current) or len(aligned) != len(new):
        raise ValueError(
            f"Coverage mismatch current={len(current)} new={len(new)} aligned={len(aligned)}"
        )
    capacity = aligned["group"].map(GROUP_CAPACITY_KWH).to_numpy(float)
    target_delta = np.abs(
        aligned["official_target"].to_numpy(float)
        - aligned["official_target_new"].to_numpy(float)
    )
    if np.any(target_delta > 0.02 * capacity):
        raise ValueError(f"Target mismatch max={float(target_delta.max())}")

    current_prediction = aligned["current_final"].to_numpy(float)
    replacement_raw = current_prediction + args.tcn_weight * (
        aligned["new_tcn"].to_numpy(float)
        - aligned["old_tcn"].to_numpy(float)
    )
    replacement = np.clip(
        replacement_raw, args.final_floor * capacity, capacity
    )
    parts = []
    for variant, values in [
        ("current_best_oof", current_prediction),
        ("foldbest_global_tcn_full_replacement", replacement),
        ("old_tcn_branch", aligned["old_tcn"].to_numpy(float)),
        ("foldbest_global_tcn_branch", aligned["new_tcn"].to_numpy(float)),
    ]:
        part = aligned[[*KEYS, "official_target"]].copy()
        part["pred"] = values
        part["variant"] = variant
        parts.append(part)
    predictions = pd.concat(parts, ignore_index=True)
    summary, pooled = pooled_oof_summary(predictions)
    folds = score_rows(predictions)
    reference = summary.loc[summary["variant"].eq("current_best_oof")].iloc[0]
    summary["delta_score_vs_current"] = summary["mean_score"] - reference["mean_score"]
    summary["delta_nmae_vs_current"] = summary["mean_nmae"] - reference["mean_nmae"]
    summary["delta_ficr_vs_current"] = summary["mean_ficr"] - reference["mean_ficr"]

    diagnostics = aligned[[*KEYS, "official_target"]].copy()
    diagnostics["current_final"] = current_prediction
    diagnostics["old_tcn"] = aligned["old_tcn"]
    diagnostics["new_tcn"] = aligned["new_tcn"]
    diagnostics["replacement_raw"] = replacement_raw
    diagnostics["replacement_final"] = replacement
    diagnostics["delta_final"] = replacement - current_prediction

    args.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.results_dir / args.stem
    summary.to_csv(f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(f"{prefix}_pooled_group_scores.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(f"{prefix}_fold_scores.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(f"{prefix}_aligned_diagnostics.csv", index=False, encoding="utf-8-sig")
    print("=== current-best fixed-weight replacement ===")
    print(summary.to_string(index=False))
    print("\n=== pooled groups ===")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
