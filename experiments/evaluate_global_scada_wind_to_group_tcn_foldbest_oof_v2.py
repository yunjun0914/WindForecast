from __future__ import annotations

import _bootstrap  # noqa: F401
from experiments import evaluate_global_scada_wind_to_group_tcn_foldbest_oof_v1 as core


LEGACY_DEFAULT_STEM = "two_stage_scada_cubic_tcn_oof_v1"
DEFAULT_STEM = "global_17wind_residual_fixed_epoch_tcn_oof_v1"
_original_parse_args = core.base.parse_args


def parse_args_fixed_epoch():
    args = _original_parse_args()
    if args.stem == LEGACY_DEFAULT_STEM:
        args.stem = DEFAULT_STEM
    args.wind_min_delta = 0.0
    args.residual_rounds = 300
    args.n_jobs = -1
    if args.smoke_test:
        args.wind_epochs = 1
        args.wind_patience = 1
        args.power_epochs = core.proto.BAND_MIN_EPOCHS
        args.power_patience = 1
        args.residual_rounds = 5
        if not args.stem.endswith("_smoke"):
            args.stem = f"{args.stem}_smoke"
    return args


def main() -> None:
    core.base.parse_args = parse_args_fixed_epoch
    print(
        "validation protocol: shared hyperparameters, median fixed epoch, "
        "fresh refit for every outer fold; no fold checkpoint ensemble",
        flush=True,
    )
    core.main()


if __name__ == "__main__":
    main()
