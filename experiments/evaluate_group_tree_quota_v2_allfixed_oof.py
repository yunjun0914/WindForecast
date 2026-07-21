from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from experiments.evaluate_group_base_recovery_oof import (
    sample_weight,
    score_predictions,
    tree_params,
)
from utils.group_allfixed_features import (
    ALLFIXED_VARIANTS,
    QUOTA_V1_FIXED_AUX_CONTROL,
    QUOTA_V2_ALL_FIXED,
    get_or_build_group_allfixed_panels,
)
from utils.group_quota_v2 import GROUP_QUOTA_V2_CONTRACT_NAME
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr
from utils.per_turbine_fixed_grid import FIXED_TURBINE_GRID_CONTRACT_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/group_tree_quota_v2_allfixed_v1.json"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="group_tree_quota_v2_allfixed_oof_v1")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("quota_grid_contract") != GROUP_QUOTA_V2_CONTRACT_NAME:
        raise ValueError("TREE config has a different Quota grid contract")
    if config.get("turbine_grid_contract") != FIXED_TURBINE_GRID_CONTRACT_NAME:
        raise ValueError("TREE config has a different turbine grid contract")
    return config


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    return ldaps, gfs, labels


def fold_score(actual, prediction, group: str) -> tuple[float, float, float]:
    nmae, ficr = group_nmae_ficr(
        actual, prediction, GROUP_CAPACITY_KWH[group]
    )
    return 0.5 * (1.0 - nmae) + 0.5 * ficr, nmae, ficr


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    years = [int(year) for year in config["years"]]
    groups = list(TARGET_COLS[:1] if args.smoke_test else TARGET_COLS)
    if args.smoke_test:
        years = years[:1]

    ldaps, gfs, labels = read_inputs()
    best = pd.read_csv(config["best_params_path"], encoding="utf-8-sig")
    oof_parts = []
    fold_rows = []
    contract_parts = []

    for group in groups:
        selected = best.loc[best["group"].eq(group)]
        if len(selected) != 1:
            raise ValueError(f"Expected one TREE config row for {group}")
        selected = selected.iloc[0]
        panels, aux_contract, quota_contract = get_or_build_group_allfixed_panels(
            ldaps,
            gfs,
            group,
            cache_root=Path(config["cache_root"]),
        )
        aux_contract = aux_contract.copy()
        aux_contract["selection_kind"] = "panel_fixed_turbine_grid"
        contract_parts.append(aux_contract)
        quota_contract = quota_contract.copy()
        quota_contract["selection_kind"] = "quota_v2_fixed_grid_contract"
        contract_parts.append(quota_contract)

        label_one = labels[["kst_dtm", group]].rename(
            columns={"kst_dtm": "forecast_kst_dtm", group: "target"}
        )
        variant_tables = {}
        for variant, panel in panels.items():
            table = panel.table.merge(
                label_one,
                on="forecast_kst_dtm",
                how="inner",
                validate="one_to_one",
            ).dropna(subset=["target"])
            table["year"] = pd.to_datetime(table["forecast_kst_dtm"]).dt.year
            variant_tables[variant] = (table, list(panel.full_feature_cols))

        for pred_year in years:
            control_table, _ = variant_tables[QUOTA_V1_FIXED_AUX_CONTROL]
            if int(control_table["year"].eq(pred_year).sum()) < 200:
                continue
            output = control_table.loc[
                control_table["year"].eq(pred_year),
                ["forecast_kst_dtm", "target"],
            ].rename(columns={"target": "actual"})
            output["pred_year"] = pred_year
            output["group"] = group

            for variant in ALLFIXED_VARIANTS:
                table, feature_cols = variant_tables[variant]
                train_years = [year for year in years if year != pred_year]
                if args.smoke_test:
                    train_years = [
                        int(year)
                        for year in config["years"]
                        if int(year) != pred_year
                    ]
                train = table.loc[table["year"].isin(train_years)]
                validation = table.loc[table["year"].eq(pred_year)]
                expected_times = pd.to_datetime(
                    output["forecast_kst_dtm"]
                ).reset_index(drop=True)
                validation_times = pd.to_datetime(
                    validation["forecast_kst_dtm"]
                ).reset_index(drop=True)
                if not validation_times.equals(expected_times):
                    raise ValueError(
                        f"TREE validation time order differs for "
                        f"{variant}/{group}/{pred_year}"
                    )
                keep = train["target"].to_numpy(float) >= (
                    GROUP_CAPACITY_KWH[group] * float(selected["min_output_ratio"])
                )
                params = tree_params(
                    selected,
                    str(selected["objective"]),
                    int(config["smoke_estimators"]) if args.smoke_test else 0,
                )
                model = LGBMRegressor(**params)
                model.fit(
                    train.loc[keep, feature_cols],
                    train.loc[keep, "target"],
                    sample_weight=sample_weight(
                        train.loc[keep, "target"].to_numpy(float),
                        group,
                        str(selected["weight_policy"]),
                    ),
                )
                prediction = np.clip(
                    model.predict(validation[feature_cols]),
                    0.0,
                    GROUP_CAPACITY_KWH[group],
                )
                output[variant] = prediction
                score, nmae, ficr = fold_score(
                    validation["target"], prediction, group
                )
                fold_rows.append(
                    {
                        "variant": variant,
                        "group": group,
                        "pred_year": pred_year,
                        "score": score,
                        "nmae": nmae,
                        "ficr": ficr,
                        "n_features": len(feature_cols),
                        "n_train": int(keep.sum()),
                        "n_rows": len(validation),
                    }
                )
                print(
                    f"TREE {variant} {group} val={pred_year} "
                    f"features={len(feature_cols)} score={score:.6f}",
                    flush=True,
                )
            oof_parts.append(output)

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["group", "forecast_kst_dtm"]
    )
    summary, diagnostics = score_predictions(oof, list(ALLFIXED_VARIANTS))
    folds = pd.DataFrame(fold_rows)
    prefix = args.results_dir / args.stem
    oof.to_csv(f"{prefix}_oof.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(
        f"{prefix}_score_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    folds.to_csv(f"{prefix}_fold_scores.csv", index=False, encoding="utf-8-sig")
    pd.concat(contract_parts, ignore_index=True).drop_duplicates().to_csv(
        f"{prefix}_grid_contracts.csv", index=False, encoding="utf-8-sig"
    )
    print("\n=== TREE pooled official OOF ===", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
