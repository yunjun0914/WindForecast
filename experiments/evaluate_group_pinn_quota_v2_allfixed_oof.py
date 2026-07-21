from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from experiments.evaluate_group_base_recovery_oof import score_predictions
from experiments.evaluate_group_pinn_band_loss_oof import (
    PINNFold,
    discover_epoch,
    fit_fixed_epoch,
)
from experiments.evaluate_group_unified_oof import (
    build_group_scada_targets,
    fit_teacher,
    read_inputs,
)
from utils.group_allfixed_features import (
    ALLFIXED_VARIANTS,
    get_or_build_group_allfixed_panels,
)
from utils.group_quota_v2 import GROUP_QUOTA_V2_CONTRACT_NAME
from utils.metrics import TARGET_COLS
from utils.per_turbine_fixed_grid import FIXED_TURBINE_GRID_CONTRACT_NAME
from utils.preprocessing import TIME_KEY_COLS
from utils.tree_feature_profiles import FEATURE_PROFILE_FULL_V2, build_tree_features


@dataclass(frozen=True)
class VariantPINNFold:
    feature_variant: str
    n_features: int
    fold: PINNFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/group_pinn_quota_v2_allfixed_v1.json"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="group_pinn_quota_v2_allfixed_oof_v1")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_configs(path: Path, smoke_test: bool) -> tuple[dict, dict]:
    experiment = json.loads(path.read_text(encoding="utf-8"))
    base = json.loads(
        Path(experiment["base_config_path"]).read_text(encoding="utf-8")
    )
    if experiment.get("quota_grid_contract") != GROUP_QUOTA_V2_CONTRACT_NAME:
        raise ValueError("PINN config has a different Quota grid contract")
    if (
        experiment.get("turbine_grid_contract")
        != FIXED_TURBINE_GRID_CONTRACT_NAME
    ):
        raise ValueError("PINN config has a different turbine grid contract")
    if experiment.get("loss_mode") != "pure_band_ficr":
        raise ValueError("PINN all-fixed experiment must use FiCR-only loss")
    if smoke_test:
        experiment["teacher_trees"] = min(24, int(experiment["teacher_trees"]))
        experiment["max_epochs"] = 2
        experiment["patience"] = 2
    return experiment, base


def build_variant_folds(
    experiment: dict,
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    labels: pd.DataFrame,
    scada_by_group: dict[str, pd.DataFrame],
    *,
    smoke_test: bool,
) -> tuple[list[VariantPINNFold], list[pd.DataFrame]]:
    groups = list(TARGET_COLS[:1] if smoke_test else TARGET_COLS)
    years = [int(year) for year in experiment["years"]]
    if smoke_test:
        years = years[:1]
    folds = []
    contract_parts = []

    for group_index, group in enumerate(groups):
        print(f"\n=== build all-fixed PINN data: {group} ===", flush=True)
        panels, aux_contract, quota_contract = get_or_build_group_allfixed_panels(
            ldaps,
            gfs,
            group,
            cache_root=Path(experiment["cache_root"]),
        )
        aux_contract = aux_contract.copy()
        aux_contract["selection_kind"] = "panel_fixed_turbine_grid"
        contract_parts.append(aux_contract)
        quota_contract = quota_contract.copy()
        quota_contract["selection_kind"] = "quota_v2_fixed_grid_contract"
        contract_parts.append(quota_contract)

        physics = build_tree_features(
            ldaps,
            gfs,
            group,
            feature_profile=FEATURE_PROFILE_FULL_V2,
        )[[*TIME_KEY_COLS, "phys_gfs_air_density"]]
        for column in TIME_KEY_COLS:
            physics[column] = pd.to_datetime(physics[column])
        teacher_targets = build_group_scada_targets(scada_by_group[group], group)
        label_one = labels[["kst_dtm", group]].rename(
            columns={"kst_dtm": "forecast_kst_dtm", group: "target"}
        )

        for feature_variant in ALLFIXED_VARIANTS:
            panel = panels[feature_variant]
            feature_cols = list(panel.full_feature_cols)
            table = (
                panel.table.merge(
                    physics,
                    on=TIME_KEY_COLS,
                    how="inner",
                    validate="one_to_one",
                )
                .merge(
                    label_one,
                    on="forecast_kst_dtm",
                    how="inner",
                    validate="one_to_one",
                )
                .dropna(subset=["target"])
            )
            table["year"] = pd.to_datetime(table["forecast_kst_dtm"]).dt.year
            for pred_year in years:
                train_years = [
                    int(year)
                    for year in experiment["years"]
                    if int(year) != pred_year
                ]
                train = table.loc[table["year"].isin(train_years)].reset_index(
                    drop=True
                )
                validation = table.loc[table["year"].eq(pred_year)].reset_index(
                    drop=True
                )
                if len(train) < 500 or len(validation) < 200:
                    continue
                teacher_seed = (
                    int(experiment["seed"])
                    + group_index * 10000
                    + pred_year * 10
                )
                teacher_train, teacher_validation = fit_teacher(
                    train,
                    validation,
                    teacher_targets,
                    feature_cols,
                    int(experiment["teacher_trees"]),
                    teacher_seed,
                )
                keep_cols = [
                    "forecast_kst_dtm",
                    "data_available_kst_dtm",
                    "phys_gfs_air_density",
                    "target",
                ]
                fold = PINNFold(
                    group=group,
                    pred_year=pred_year,
                    train=train[keep_cols].copy(),
                    validation=validation[keep_cols].copy(),
                    teacher_train=teacher_train,
                    teacher_validation=teacher_validation,
                )
                folds.append(
                    VariantPINNFold(
                        feature_variant=feature_variant,
                        n_features=len(feature_cols),
                        fold=fold,
                    )
                )
                print(
                    f"teacher {feature_variant} {group} val={pred_year} "
                    f"features={len(feature_cols)} train={len(train)} "
                    f"validation={len(validation)}",
                    flush=True,
                )
        gc.collect()
    return folds, contract_parts


def main() -> None:
    args = parse_args()
    experiment, base_config = load_configs(args.config, args.smoke_test)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_mode = str(experiment["loss_mode"])
    print(
        f"device={device} smoke={args.smoke_test} loss={loss_mode}", flush=True
    )
    ldaps, gfs, labels, scada_by_group = read_inputs()
    variant_folds, contract_parts = build_variant_folds(
        experiment,
        ldaps,
        gfs,
        labels,
        scada_by_group,
        smoke_test=args.smoke_test,
    )
    pinn_config = base_config["pinn"]

    history_rows = []
    discovered = {
        variant: {group: [] for group in TARGET_COLS}
        for variant in ALLFIXED_VARIANTS
    }
    for item in variant_folds:
        fold = item.fold
        seed = (
            int(experiment["seed"])
            + TARGET_COLS.index(fold.group) * 10000
            + fold.pred_year * 100
        )
        print(
            f"\n=== PINN discovery {item.feature_variant} "
            f"{fold.group} val={fold.pred_year} ===",
            flush=True,
        )
        best_epoch, history = discover_epoch(
            fold,
            loss_mode,
            experiment,
            pinn_config,
            device,
            seed,
        )
        discovered[item.feature_variant][fold.group].append(best_epoch)
        for row in history:
            row = dict(row)
            row["loss_mode"] = row.pop("variant")
            row["variant"] = item.feature_variant
            row["n_features"] = item.n_features
            history_rows.append(row)

    fixed_epochs = {
        variant: {
            group: int(np.median(values))
            for group, values in groups.items()
            if values
        }
        for variant, groups in discovered.items()
    }
    print(f"\nPINN fixed epochs={fixed_epochs}", flush=True)

    outputs: dict[tuple[str, int], pd.DataFrame] = {}
    for item in variant_folds:
        fold = item.fold
        key = (fold.group, fold.pred_year)
        if key not in outputs:
            outputs[key] = pd.DataFrame(
                {
                    "forecast_kst_dtm": pd.to_datetime(
                        fold.validation["forecast_kst_dtm"]
                    ),
                    "pred_year": fold.pred_year,
                    "group": fold.group,
                    "actual": fold.validation["target"].to_numpy(float),
                }
            )
        else:
            expected_times = pd.to_datetime(
                outputs[key]["forecast_kst_dtm"]
            ).reset_index(drop=True)
            validation_times = pd.to_datetime(
                fold.validation["forecast_kst_dtm"]
            ).reset_index(drop=True)
            if not validation_times.equals(expected_times):
                raise ValueError(
                    f"PINN validation time order differs for "
                    f"{item.feature_variant}/{fold.group}/{fold.pred_year}"
                )
            expected_actual = outputs[key]["actual"].to_numpy(float)
            validation_actual = fold.validation["target"].to_numpy(float)
            if not np.allclose(
                validation_actual, expected_actual, equal_nan=True
            ):
                raise ValueError(
                    f"PINN validation targets differ for "
                    f"{item.feature_variant}/{fold.group}/{fold.pred_year}"
                )
        fixed_epoch = fixed_epochs[item.feature_variant][fold.group]
        seed = (
            int(experiment["seed"])
            + 10_000_000
            + TARGET_COLS.index(fold.group) * 10000
            + fold.pred_year * 100
        )
        print(
            f"\n=== PINN fixed {item.feature_variant} "
            f"{fold.group} val={fold.pred_year} epoch={fixed_epoch} ===",
            flush=True,
        )
        prediction, stats = fit_fixed_epoch(
            fold,
            loss_mode,
            fixed_epoch,
            pinn_config,
            device,
            seed,
        )
        outputs[key][item.feature_variant] = prediction
        stats = dict(stats)
        stats["loss_mode"] = stats.pop("variant")
        stats["variant"] = item.feature_variant
        stats["n_features"] = item.n_features
        history_rows.append(stats)

    oof = pd.concat(outputs.values(), ignore_index=True).sort_values(
        ["group", "forecast_kst_dtm"]
    )
    summary, diagnostics = score_predictions(oof, list(ALLFIXED_VARIANTS))
    prefix = args.results_dir / args.stem
    oof.to_csv(f"{prefix}_oof.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(
        [pd.DataFrame(history_rows).assign(row_type="training"), diagnostics],
        ignore_index=True,
    ).to_csv(
        f"{prefix}_epoch_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(contract_parts, ignore_index=True).drop_duplicates().to_csv(
        f"{prefix}_grid_contracts.csv", index=False, encoding="utf-8-sig"
    )
    Path(f"{prefix}_fixed_epochs.json").write_text(
        json.dumps(fixed_epochs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== PINN pooled official OOF ===", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
