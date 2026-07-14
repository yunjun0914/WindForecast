from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS


KEYS = ["forecast_id", "forecast_kst_dtm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--old-tcn", required=True)
    parser.add_argument("--new-tcn", required=True)
    parser.add_argument("--tcn-weight", type=float, default=0.35)
    parser.add_argument("--final-floor", type=float, default=0.10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics", required=True)
    return parser.parse_args()


def load_submission(path: str) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig")
    required = [*KEYS, *TARGET_COLS]
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    table = table[required].copy()
    table["forecast_kst_dtm"] = pd.to_datetime(table["forecast_kst_dtm"])
    if table.duplicated(KEYS).any():
        raise ValueError(f"{path} has duplicate keys")
    return table


def assert_aligned(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
    if len(reference) != len(candidate):
        raise ValueError(f"{label} row mismatch: {len(candidate)} != {len(reference)}")
    if not reference[KEYS].equals(candidate[KEYS]):
        raise ValueError(f"{label} key/order mismatch")


def main() -> None:
    args = parse_args()
    base = load_submission(args.base)
    old_tcn = load_submission(args.old_tcn)
    new_tcn = load_submission(args.new_tcn)
    assert_aligned(base, old_tcn, "old TCN")
    assert_aligned(base, new_tcn, "new TCN")

    output = base.copy()
    diagnostics = []
    for group in TARGET_COLS:
        capacity = GROUP_CAPACITY_KWH[group]
        branch_delta = args.tcn_weight * (
            new_tcn[group].to_numpy(float) - old_tcn[group].to_numpy(float)
        )
        raw = base[group].to_numpy(float) + branch_delta
        final = np.clip(raw, args.final_floor * capacity, capacity)
        output[group] = final
        diagnostics.append(
            {
                "group": group,
                "rows": len(final),
                "changed_rows": int(np.sum(~np.isclose(final, base[group].to_numpy(float)))),
                "final_floor_raised": int(np.sum(raw < args.final_floor * capacity)),
                "capacity_clipped": int(np.sum(raw > capacity)),
                "mean_branch_delta": float(np.mean(branch_delta)),
                "mean_abs_branch_delta": float(np.mean(np.abs(branch_delta))),
                "max_abs_branch_delta": float(np.max(np.abs(branch_delta))),
                "base_mean": float(base[group].mean()),
                "final_mean": float(np.mean(final)),
                "final_min": float(np.min(final)),
                "final_max": float(np.max(final)),
            }
        )

    if len(output) != 8760:
        raise ValueError(f"Expected 8760 rows, got {len(output)}")
    values = output[TARGET_COLS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite predictions")

    output_path = Path(args.output)
    diagnostics_path = Path(args.diagnostics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(f"saved {output_path}: rows={len(output)} sha256={digest}")
    print(pd.DataFrame(diagnostics).to_string(index=False))


if __name__ == "__main__":
    main()
