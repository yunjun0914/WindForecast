import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--delta", type=float, default=0.0175)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.read_csv(args.input, encoding="utf-8-sig")
    required = ["forecast_id", "forecast_kst_dtm", *TARGET_COLS]
    missing = [col for col in required if col not in submission.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    calibrated = submission.copy()
    for group in TARGET_COLS:
        capacity = GROUP_CAPACITY_KWH[group]
        calibrated[group] = np.clip(calibrated[group] + args.delta * capacity, 0.0, capacity)

    calibrated = calibrated[required]
    calibrated.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved {out_path}: {calibrated.shape}, delta={args.delta}")
    print(calibrated[TARGET_COLS].agg(["min", "max", "mean"]).to_string())


if __name__ == "__main__":
    main()
