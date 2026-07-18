from __future__ import annotations

from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from experiments import evaluate_global_scada_wind_to_group_tcn_foldbest_oof_v1 as v1
from utils.metrics import TARGET_COLS


LEGACY_DEFAULT_STEM = "two_stage_scada_cubic_tcn_oof_v1"
DEFAULT_STEM = "global_17wind_3group_foldbest_pure_band_tcn_oof_v2"
_original_parse_args = v1.base.parse_args
_active_args = None


def parse_args_foldbest_v2():
    """Enforce the user-confirmed non-nested fold-best protocol safely."""
    global _active_args
    args = _original_parse_args()
    if args.stem == LEGACY_DEFAULT_STEM:
        args.stem = DEFAULT_STEM

    # The checkpoint update in v1 is `metric < best - min_delta`.  Setting
    # min_delta to zero makes the stored state the literal outer-fold optimum,
    # as required; it is never discarded for a fixed-epoch refit.
    args.wind_min_delta = 0.0

    if args.smoke_test:
        args.wind_epochs = 1
        args.wind_patience = 1
        args.power_epochs = v1.proto.BAND_MIN_EPOCHS
        args.power_patience = 1
        if not args.stem.endswith("_smoke"):
            args.stem = f"{args.stem}_smoke"
    _active_args = args
    return args


def write_ensemble_manifest(args) -> Path:
    checkpoint_dir = args.results_dir / f"{args.stem}_checkpoints"
    rows = []
    requested_years = [2022, 2023, 2024]

    for group in TARGET_COLS:
        stage1_years = [2023, 2024] if group == "kpx_group_3" else requested_years
        for validation_year in stage1_years:
            rows.append(
                {
                    "stage": "TCN1",
                    "output_group": group,
                    "validation_year": validation_year,
                    "checkpoint_path": str(
                        checkpoint_dir
                        / f"stage1_{group}_val{validation_year}.pt"
                    ),
                    "include_in_test_ensemble": True,
                    "reason": "fold has observed SCADA validation target",
                    "hidden_size": args.wind_hidden_size,
                    "num_layers": args.wind_num_layers,
                    "kernel_size": args.wind_kernel_size,
                    "dropout": args.wind_dropout,
                }
            )

    for group in TARGET_COLS:
        for validation_year in requested_years:
            include = not (
                group == "kpx_group_3" and validation_year == 2022
            )
            rows.append(
                {
                    "stage": "TCN2",
                    "output_group": group,
                    "validation_year": validation_year,
                    "checkpoint_path": str(
                        checkpoint_dir / f"stage2_val{validation_year}.pt"
                    ),
                    "include_in_test_ensemble": include,
                    "reason": (
                        "fold has official validation target"
                        if include
                        else "excluded: group3 has no 2022 validation target"
                    ),
                    "hidden_size": args.power_hidden_size,
                    "num_layers": args.power_num_layers,
                    "kernel_size": args.power_kernel_size,
                    "dropout": args.power_dropout,
                }
            )

    path = args.results_dir / f"{args.stem}_ensemble_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> None:
    v1.base.parse_args = parse_args_foldbest_v2
    print(
        "validation protocol: NON-NESTED outer-fold best checkpoint, "
        "literal best state, NO fixed-epoch refit",
        flush=True,
    )
    v1.main()
    if _active_args is None:
        raise RuntimeError("Fold-best v2 argument capture failed")
    manifest_path = write_ensemble_manifest(_active_args)
    print(f"test ensemble manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
