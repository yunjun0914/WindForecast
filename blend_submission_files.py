import argparse

import numpy as np
import pandas as pd

from utils.metrics import GROUP_CAPACITY_KWH

GROUPS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--extra", required=True)
    parser.add_argument("--extra-weight", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = pd.read_csv(args.base, encoding="utf-8-sig")
    extra = pd.read_csv(args.extra, encoding="utf-8-sig")
    if list(base.columns) != list(extra.columns):
        raise ValueError("column mismatch")
    if not base["forecast_id"].equals(extra["forecast_id"]):
        raise ValueError("forecast_id mismatch")
    if not base["forecast_kst_dtm"].equals(extra["forecast_kst_dtm"]):
        raise ValueError("forecast_kst_dtm mismatch")

    out = base.copy()
    w = args.extra_weight
    for group in GROUPS:
        out[group] = np.clip((1 - w) * base[group] + w * extra[group], 0, GROUP_CAPACITY_KWH[group])
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"saved {args.out}: {out.shape}, extra_weight={w}")
    print(out[GROUPS].agg(["min", "max", "mean"]).to_string())
    return out


if __name__ == "__main__":
    main()
