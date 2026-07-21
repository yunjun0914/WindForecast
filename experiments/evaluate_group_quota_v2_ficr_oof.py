from __future__ import annotations

import argparse
import gc
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from experiments.evaluate_group_decision_q_tcn_oof import (
    group_score,
    prepare_fold_arrays,
    train_band_fixed_fold,
    train_band_fold,
)
from experiments.evaluate_group_unified_oof import read_inputs
from experiments.evaluate_per_turbine_bin_moe_tcn_oof import build_fold_features
from utils.group_local_panel import build_group_local_panel
from utils.group_quota_v2 import (
    GROUP_FAMILY_QUOTA64_V2_FEATURES,
    get_or_build_group_quota_v2,
)
from utils.issue_block_dataset import make_issue_blocks
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, pooled_oof_summary
from utils.per_turbine_features import get_or_build_group_feature_cache
from utils.per_turbine_optimal_grid_builder import build_wind_candidate_matrix
from utils.per_turbine_scada import build_official_aligned_turbine_targets
from utils.preprocessing import TIME_KEY_COLS


VARIANTS = ("quota_v1_control", "quota_v2_group_local")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/group_quota_v2_ficr_oof_v1.json"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="group_quota_v2_ficr_fixed_oof_v1")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--rebuild-optimal-grid-cache", action="store_true")
    parser.add_argument("--rebuild-group-quota-cache", action="store_true")
    return parser.parse_args()


def load_config(path: Path, smoke_test: bool) -> tuple[dict, Namespace]:
    config = json.loads(path.read_text(encoding="utf-8"))
    training = dict(config["training"])
    if smoke_test:
        training.update(
            {
                "epochs": 2,
                "patience": 2,
                "band_min_epochs": 1,
                "batch_size": 128,
                "eval_batch_size": 512,
            }
        )
    train_args = Namespace(**training)
    train_args.feature_variant = config["feature_variant"]
    train_args.cache_root = Path(config["cache_root"])
    train_args.rebuild_feature_cache = False
    train_args.rebuild_optimal_grid_cache = False
    return config, train_args


def valid_years(labels: pd.DataFrame, group: str, years: list[int]) -> list[int]:
    return [
        year
        for year in years
        if labels.loc[
            labels["kst_dtm"].dt.year.eq(year), group
        ].notna().sum()
        >= 200
    ]


def prediction_frame(
    arrays,
    prediction: np.ndarray,
    group: str,
    pred_year: int,
    variant: str,
) -> pd.DataFrame:
    capacity = float(GROUP_CAPACITY_KWH[group])
    return pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(
                arrays.validation_times.reshape(-1)
            ),
            "variant": variant,
            "group": group,
            "pred_year": int(pred_year),
            "official_target": arrays.y_validation.reshape(-1) * capacity,
            "pred": prediction.reshape(-1) * capacity,
        }
    ).dropna(subset=["official_target", "pred"])


def build_fold_arrays_for_variant(
    *,
    variant: str,
    group: str,
    pred_year: int,
    years: list[int],
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    labels: pd.DataFrame,
    base_features: pd.DataFrame,
    wind_candidates,
    turbine_targets: pd.DataFrame,
    train_args: Namespace,
    rebuild_group_quota_cache: bool,
):
    train_years = [year for year in years if year != pred_year]
    features, _, optimal_selections = build_fold_features(
        base_features,
        wind_candidates,
        turbine_targets,
        group,
        pred_year,
        train_years,
        train_args,
    )
    quota_selections = pd.DataFrame()
    if variant == "quota_v1_control":
        panel = build_group_local_panel(features, group)
    elif variant == "quota_v2_group_local":
        group_quota, quota_selections = get_or_build_group_quota_v2(
            ldaps,
            gfs,
            wind_candidates,
            turbine_targets,
            group,
            pred_year,
            train_years,
            cache_root=train_args.cache_root,
            rebuild=rebuild_group_quota_cache,
        )
        before = len(features)
        features = features.merge(
            group_quota,
            on=TIME_KEY_COLS,
            how="left",
            validate="many_to_one",
        )
        if len(features) != before:
            raise ValueError(
                f"Group quota v2 merge changed rows: {before} -> {len(features)}"
            )
        v2_cols = GROUP_FAMILY_QUOTA64_V2_FEATURES[group]
        missing = [column for column in v2_cols if column not in features]
        if missing:
            raise ValueError(f"Missing v2 quota columns for {group}: {missing}")
        panel = build_group_local_panel(
            features,
            group,
            common_feature_cols=v2_cols,
        )
    else:
        raise ValueError(f"Unknown quota variant: {variant}")

    feature_cols = list(panel.full_feature_cols)
    blocks = make_issue_blocks(
        panel.table,
        labels,
        feature_cols,
        target_cols=[group],
    )
    arrays = prepare_fold_arrays(
        blocks,
        pred_year,
        float(GROUP_CAPACITY_KWH[group]),
        float(train_args.target_min_output_ratio),
    )
    return arrays, feature_cols, optimal_selections, quota_selections


def run_discovery(
    *,
    variant: str,
    config: dict,
    train_args: Namespace,
    groups: list[str],
    years: list[int],
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    labels: pd.DataFrame,
    scada_by_group: dict[str, pd.DataFrame],
    wind_candidates,
    device: torch.device,
    rebuild_group_quota_cache: bool,
) -> tuple[pd.DataFrame, dict[str, list[int]], list[dict], list[pd.DataFrame]]:
    prediction_parts = []
    discovered: dict[str, list[int]] = {group: [] for group in groups}
    history_rows = []
    selection_parts = []
    for group_index, group in enumerate(groups):
        base_features = get_or_build_group_feature_cache(
            ldaps,
            gfs,
            group,
            cache_root=train_args.cache_root,
            rebuild=train_args.rebuild_feature_cache,
        )
        targets = build_official_aligned_turbine_targets(
            scada_by_group[group], labels, group
        )
        for pred_year in valid_years(labels, group, years):
            arrays, feature_cols, optimal_sel, quota_sel = (
                build_fold_arrays_for_variant(
                    variant=variant,
                    group=group,
                    pred_year=pred_year,
                    years=years,
                    ldaps=ldaps,
                    gfs=gfs,
                    labels=labels,
                    base_features=base_features,
                    wind_candidates=wind_candidates,
                    turbine_targets=targets,
                    train_args=train_args,
                    rebuild_group_quota_cache=rebuild_group_quota_cache,
                )
            )
            seed = int(config["seed"]) + group_index * 100000 + pred_year * 100
            print(
                f"\n[discovery] {variant} {group} val={pred_year} "
                f"features={len(feature_cols)}",
                flush=True,
            )
            prediction, stats, history = train_band_fold(
                arrays,
                group,
                train_args,
                device,
                seed,
                loss_mode="pure68",
            )
            discovered[group].append(int(stats["best_epoch"]))
            prediction_parts.append(
                prediction_frame(arrays, prediction, group, pred_year, variant)
            )
            for row in history:
                history_rows.append(
                    {
                        "stage": "epoch_discovery",
                        "variant": variant,
                        "group": group,
                        "pred_year": pred_year,
                        **row,
                    }
                )
            optimal_sel = optimal_sel.copy()
            optimal_sel["pred_year"] = pred_year
            optimal_sel["selection_kind"] = "panel_optimal_grid"
            selection_parts.append(optimal_sel)
            if not quota_sel.empty:
                quota_sel = quota_sel.copy()
                quota_sel["pred_year"] = pred_year
                quota_sel["selection_kind"] = "quota_v2_source_grid"
                selection_parts.append(quota_sel)
            del arrays
            gc.collect()
        del base_features, targets
        gc.collect()
    return (
        pd.concat(prediction_parts, ignore_index=True),
        discovered,
        history_rows,
        selection_parts,
    )


def run_fixed(
    *,
    variant: str,
    fixed_epochs: dict[str, int],
    config: dict,
    train_args: Namespace,
    groups: list[str],
    years: list[int],
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    labels: pd.DataFrame,
    scada_by_group: dict[str, pd.DataFrame],
    wind_candidates,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    prediction_parts = []
    fold_rows = []
    history_rows = []
    for group_index, group in enumerate(groups):
        base_features = get_or_build_group_feature_cache(
            ldaps,
            gfs,
            group,
            cache_root=train_args.cache_root,
            rebuild=False,
        )
        targets = build_official_aligned_turbine_targets(
            scada_by_group[group], labels, group
        )
        for pred_year in valid_years(labels, group, years):
            arrays, feature_cols, _, _ = build_fold_arrays_for_variant(
                variant=variant,
                group=group,
                pred_year=pred_year,
                years=years,
                ldaps=ldaps,
                gfs=gfs,
                labels=labels,
                base_features=base_features,
                wind_candidates=wind_candidates,
                turbine_targets=targets,
                train_args=train_args,
                rebuild_group_quota_cache=False,
            )
            fixed_epoch = int(fixed_epochs[group])
            seed = (
                int(config["seed"])
                + 10_000_000
                + group_index * 100000
                + pred_year * 100
            )
            print(
                f"\n[fixed] {variant} {group} val={pred_year} "
                f"epoch={fixed_epoch} features={len(feature_cols)}",
                flush=True,
            )
            prediction, stats, history = train_band_fixed_fold(
                arrays,
                group,
                fixed_epoch,
                train_args,
                device,
                seed,
            )
            frame = prediction_frame(arrays, prediction, group, pred_year, variant)
            prediction_parts.append(frame)
            score, nmae, ficr = group_score(
                frame["official_target"], frame["pred"], group
            )
            fold_rows.append(
                {
                    "variant": variant,
                    "group": group,
                    "pred_year": pred_year,
                    "score": score,
                    "nmae": nmae,
                    "ficr": ficr,
                    "fixed_epoch": fixed_epoch,
                    "n_features": len(feature_cols),
                    "n_parameters": stats["n_parameters"],
                    "n_rows": len(frame),
                }
            )
            for row in history:
                history_rows.append(
                    {
                        "stage": "fixed_epoch",
                        "variant": variant,
                        "group": group,
                        "pred_year": pred_year,
                        "fixed_epoch": fixed_epoch,
                        **row,
                    }
                )
            del arrays
            gc.collect()
        del base_features, targets
        gc.collect()
    return (
        pd.concat(prediction_parts, ignore_index=True),
        fold_rows,
        history_rows,
    )


def reference_score(config: dict) -> float:
    path = Path(config["reference_oof_path"])
    reference = pd.read_csv(path)
    reference = reference.loc[
        reference["variant"].eq(config["reference_variant"])
    ].copy()
    if reference.empty:
        raise ValueError(f"Reference variant missing from {path}")
    summary, _ = pooled_oof_summary(reference)
    return float(summary.iloc[0]["mean_score"])


def partial_oof_summary(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize smoke runs that intentionally contain only a subset of groups."""
    group_rows = []
    for (variant, group), part in predictions.groupby(["variant", "group"]):
        score, nmae, ficr = group_score(
            part["official_target"], part["pred"], group
        )
        group_rows.append(
            {
                "variant": variant,
                "group": group,
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n_rows": len(part),
            }
        )
    group_scores = pd.DataFrame(group_rows)
    summary = (
        group_scores.groupby("variant", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            n_groups=("group", "nunique"),
        )
        .sort_values("mean_score", ascending=False)
    )
    return summary, group_scores


def main() -> None:
    args = parse_args()
    config, train_args = load_config(args.config, args.smoke_test)
    train_args.rebuild_feature_cache = args.rebuild_feature_cache
    train_args.rebuild_optimal_grid_cache = args.rebuild_optimal_grid_cache
    groups = list(config["groups"])
    years = [int(year) for year in config["years"]]
    if groups != [group for group in TARGET_COLS if group in groups]:
        raise ValueError("Groups must follow TARGET_COLS order")
    if args.smoke_test:
        groups = groups[:1]
        years = years[:2]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} groups={groups} years={years} "
        f"smoke={args.smoke_test}",
        flush=True,
    )
    ldaps, gfs, labels, scada_by_group = read_inputs()
    wind_candidates = build_wind_candidate_matrix(ldaps, gfs)

    discovery_predictions = []
    discoveries = {}
    all_history = []
    all_selections = []
    control_pred, control_epochs, history, selections = run_discovery(
        variant="quota_v1_control",
        config=config,
        train_args=train_args,
        groups=groups,
        years=years,
        ldaps=ldaps,
        gfs=gfs,
        labels=labels,
        scada_by_group=scada_by_group,
        wind_candidates=wind_candidates,
        device=device,
        rebuild_group_quota_cache=False,
    )
    discovery_predictions.append(control_pred)
    discoveries["quota_v1_control"] = control_epochs
    all_history.extend(history)
    all_selections.extend(selections)

    if not args.smoke_test:
        control_summary, _ = pooled_oof_summary(control_pred)
        reproduced = float(control_summary.iloc[0]["mean_score"])
        expected = reference_score(config)
        difference = abs(reproduced - expected)
        print(
            f"\ncontrol reproduction expected={expected:.6f} "
            f"actual={reproduced:.6f} diff={difference:.6f}",
            flush=True,
        )
        if difference > float(config["reference_tolerance"]):
            raise RuntimeError(
                f"Control reproduction gate failed: {difference:.6f} > "
                f"{float(config['reference_tolerance']):.6f}"
            )

    v2_pred, v2_epochs, history, selections = run_discovery(
        variant="quota_v2_group_local",
        config=config,
        train_args=train_args,
        groups=groups,
        years=years,
        ldaps=ldaps,
        gfs=gfs,
        labels=labels,
        scada_by_group=scada_by_group,
        wind_candidates=wind_candidates,
        device=device,
        rebuild_group_quota_cache=args.rebuild_group_quota_cache,
    )
    discovery_predictions.append(v2_pred)
    discoveries["quota_v2_group_local"] = v2_epochs
    all_history.extend(history)
    all_selections.extend(selections)

    fixed_epoch_map = {
        variant: {
            group: int(np.median(epochs))
            for group, epochs in group_epochs.items()
            if epochs
        }
        for variant, group_epochs in discoveries.items()
    }
    print(f"\nfixed epochs={fixed_epoch_map}", flush=True)

    fixed_predictions = []
    fold_rows = []
    for variant in VARIANTS:
        predictions, rows, history = run_fixed(
            variant=variant,
            fixed_epochs=fixed_epoch_map[variant],
            config=config,
            train_args=train_args,
            groups=groups,
            years=years,
            ldaps=ldaps,
            gfs=gfs,
            labels=labels,
            scada_by_group=scada_by_group,
            wind_candidates=wind_candidates,
            device=device,
        )
        fixed_predictions.append(predictions)
        fold_rows.extend(rows)
        all_history.extend(history)

    discovery = pd.concat(discovery_predictions, ignore_index=True)
    fixed = pd.concat(fixed_predictions, ignore_index=True)
    if args.smoke_test:
        discovery_summary, discovery_group_scores = partial_oof_summary(discovery)
        fixed_summary, fixed_group_scores = partial_oof_summary(fixed)
    else:
        discovery_summary, discovery_group_scores = pooled_oof_summary(discovery)
        fixed_summary, fixed_group_scores = pooled_oof_summary(fixed)
    prefix = args.results_dir / args.stem
    discovery.to_csv(
        f"{prefix}_discovery_predictions.csv", index=False, encoding="utf-8-sig"
    )
    discovery_summary.to_csv(
        f"{prefix}_discovery_summary.csv", index=False, encoding="utf-8-sig"
    )
    discovery_group_scores.to_csv(
        f"{prefix}_discovery_group_scores.csv", index=False, encoding="utf-8-sig"
    )
    fixed.to_csv(
        f"{prefix}_fixed_predictions.csv", index=False, encoding="utf-8-sig"
    )
    fixed_summary.to_csv(
        f"{prefix}_fixed_summary.csv", index=False, encoding="utf-8-sig"
    )
    fixed_group_scores.to_csv(
        f"{prefix}_fixed_group_scores.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_rows).to_csv(
        f"{prefix}_fixed_fold_scores.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(all_history).to_csv(
        f"{prefix}_training_history.csv", index=False, encoding="utf-8-sig"
    )
    if all_selections:
        pd.concat(all_selections, ignore_index=True).drop_duplicates().to_csv(
            f"{prefix}_grid_selections.csv", index=False, encoding="utf-8-sig"
        )
    Path(f"{prefix}_fixed_epochs.json").write_text(
        json.dumps(fixed_epoch_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n=== fixed-epoch pooled official OOF ===", flush=True)
    print(fixed_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
