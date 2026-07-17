from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from experiments.evaluate_per_turbine_bin_moe_tcn_oof import (
    YEARS,
    build_fold_features,
    group_score,
    parse_csv,
    train_regressor_checkpoints,
)
from utils.group_local_panel import build_group_local_panel
from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
)
from utils.per_turbine_features import get_or_build_group_feature_cache
from utils.per_turbine_optimal_grid import (
    OPTIMAL_GRID_FEATURES,
    OPTIMAL_GRID_ISSUE_CONTEXT_FEATURES,
    WAKE_FEATURES,
)
from utils.per_turbine_optimal_grid_builder import build_wind_candidate_matrix
from utils.per_turbine_scada import build_official_aligned_turbine_targets
from utils.per_turbine_sequence import (
    SequenceStandardScaler,
    make_per_turbine_sequences,
)
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.two_sided_kneedle import (
    KneedleFitError,
    TwoSidedKneedleResult,
    fit_two_sided_kneedle,
    kneedle_mid_mask,
)


KEYS = ["forecast_kst_dtm", "group", "pred_year"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--years", default=",".join(map(str, YEARS)))
    parser.add_argument("--window", type=int, default=72)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--kneedle-bins", type=int, default=64)
    parser.add_argument("--kneedle-min-bin-count", type=int, default=30)
    parser.add_argument("--kneedle-min-separation", type=int, default=3)
    parser.add_argument("--target-min-output-ratio", type=float, default=0.10)
    parser.add_argument(
        "--feature-variant",
        choices=["optimal_grid_replace_local16", "optimal_grid_issue_context"],
        default="optimal_grid_issue_context",
    )
    parser.add_argument(
        "--baseline-pinn",
        type=Path,
        default=Path(
            "results/per_turbine_pinn_share50_pure_band_oof_v1_predictions.csv"
        ),
    )
    parser.add_argument("--baseline-pinn-variant", default="baseline_retrained_floor20")
    parser.add_argument(
        "--baseline-tree",
        type=Path,
        default=Path("results/tree_scoreboost_group_year_tune_v1_predictions.csv"),
    )
    parser.add_argument("--baseline-tree-variant", default="scoreboost_group_year_tuned")
    parser.add_argument(
        "--baseline-tcn",
        type=Path,
        default=Path("results/group_pure_band_tcn_oof_v1_predictions.csv"),
    )
    parser.add_argument("--baseline-tcn-variant", default="pure_band")
    parser.add_argument("--pinn-weight", type=float, default=0.50)
    parser.add_argument("--tree-weight", type=float, default=0.05)
    parser.add_argument("--tcn-weight", type=float, default=0.45)
    parser.add_argument("--final-floor", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--rebuild-optimal-grid-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--stem", default="group_kneedle_cuberoot_tcn_w72_oof_v2")
    return parser.parse_args()


def load_oof_branch(path: Path, variant: str, prediction_name: str) -> pd.DataFrame:
    table = pd.read_csv(path, encoding="utf-8-sig")
    required = [*KEYS, "variant", "pred"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    actual_col = "official_target" if "official_target" in table.columns else "actual"
    if actual_col not in table.columns:
        raise ValueError(f"{path} has no official_target/actual column")
    work = table.loc[table["variant"].eq(variant), [*KEYS, actual_col, "pred"]].copy()
    if work.empty:
        available = sorted(map(str, table["variant"].dropna().unique()))
        raise ValueError(f"{path} has no variant={variant}; available={available}")
    work["forecast_kst_dtm"] = pd.to_datetime(work["forecast_kst_dtm"])
    if work.duplicated(KEYS).any():
        raise ValueError(f"{path} variant={variant} has duplicate OOF keys")
    return work.rename(
        columns={actual_col: f"actual_{prediction_name}", "pred": prediction_name}
    )


def load_best_baseline(args: argparse.Namespace) -> pd.DataFrame:
    parts = [
        load_oof_branch(args.baseline_pinn, args.baseline_pinn_variant, "pinn_pred"),
        load_oof_branch(args.baseline_tree, args.baseline_tree_variant, "tree_pred"),
        load_oof_branch(args.baseline_tcn, args.baseline_tcn_variant, "tcn_pred"),
    ]
    table = parts[0]
    for part in parts[1:]:
        table = table.merge(part, on=KEYS, how="inner", validate="one_to_one")
    actual_cols = ["actual_pinn_pred", "actual_tree_pred", "actual_tcn_pred"]
    reference = table[actual_cols[0]].to_numpy(float)
    for column in actual_cols[1:]:
        delta = np.nanmax(np.abs(reference - table[column].to_numpy(float)))
        if delta > 0.02 * max(GROUP_CAPACITY_KWH.values()):
            raise ValueError(f"Baseline actual mismatch for {column}: max_delta={delta}")
        if delta > 1e-2:
            print(
                f"warning: baseline {column} target differs from PINN target; "
                f"max_delta={delta:.6f}. Predictions remain aligned by key and "
                "the experiment target is rechecked against train_labels.",
                flush=True,
            )
    table["official_target"] = reference
    table["capacity"] = table["group"].map(GROUP_CAPACITY_KWH).astype(float)
    raw = (
        args.pinn_weight * table["pinn_pred"].to_numpy(float)
        + args.tree_weight * table["tree_pred"].to_numpy(float)
        + args.tcn_weight * table["tcn_pred"].to_numpy(float)
    )
    table["base_pred"] = np.clip(
        raw,
        args.final_floor * table["capacity"].to_numpy(float),
        table["capacity"].to_numpy(float),
    )
    return table[[*KEYS, "official_target", "base_pred"]]


def add_group_effective_wind(panel: pd.DataFrame, group: str) -> pd.DataFrame:
    wind_cols = [
        f"{turbine}__optgrid_ws_calibrated"
        for turbine in GROUP_TURBINE_PREFIXES[group]
    ]
    missing = [column for column in wind_cols if column not in panel.columns]
    if missing:
        raise ValueError(f"Group effective wind missing columns: {missing}")
    wind = np.clip(panel[wind_cols].to_numpy(float), 0.0, None)
    out = panel.copy()
    out["effective_wind"] = np.cbrt(np.mean(np.power(wind, 3), axis=1))
    return out


def curve_rows(
    result: TwoSidedKneedleResult,
    group: str,
    pred_year: int,
) -> list[dict[str, float]]:
    rows = []
    for index in range(len(result.wind_bins)):
        rows.append(
            {
                "group": group,
                "pred_year": pred_year,
                "bin_index": index,
                "wind": result.wind_bins[index],
                "power_ratio_raw": result.raw_power_bins[index],
                "power_ratio_monotone": result.monotone_power_bins[index],
                "difference_y_minus_x": result.difference[index],
                "bin_count": result.bin_counts[index],
                "is_lower_knee": index == result.lower_index,
                "is_upper_knee": index == result.upper_index,
            }
        )
    return rows


def score_regimes(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, group, pred_year, regime), part in predictions.groupby(
        ["variant", "group", "pred_year", "regime"], sort=False
    ):
        capacity = float(GROUP_CAPACITY_KWH[group])
        scored = part.loc[part["official_target"] >= 0.10 * capacity]
        if scored.empty:
            continue
        nmae, ficr = group_nmae_ficr(
            scored["official_target"], scored["pred"], capacity
        )
        rows.append(
            {
                "variant": variant,
                "group": group,
                "pred_year": pred_year,
                "regime": regime,
                "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                "nmae": nmae,
                "ficr": ficr,
                "n_rows": len(scored),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    weights = args.pinn_weight + args.tree_weight + args.tcn_weight
    if not np.isclose(weights, 1.0, atol=1e-12):
        raise ValueError(f"Baseline branch weights must sum to 1, got {weights}")
    if args.epoch <= 0:
        raise ValueError("--epoch must be positive")
    if not 0.0 <= args.target_min_output_ratio < 1.0:
        raise ValueError("--target-min-output-ratio must be in [0, 1)")
    groups = parse_csv(args.groups)
    pred_years = [int(year) for year in parse_csv(args.years)]
    if args.smoke_test:
        groups = groups[:1]
        pred_years = pred_years[:1]
        args.epoch = 1

    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = load_best_baseline(args)
    print(
        f"device={device} groups={groups} years={pred_years} window={args.window} "
        f"h={args.hidden_size} layers={args.num_layers} epoch={args.epoch} "
        f"feature_variant={args.feature_variant} baseline_rows={len(baseline)}",
        flush=True,
    )

    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv(
        "data/train/scada_vestas_train.csv", encoding="utf-8-sig"
    )
    scada_unison = pd.read_csv(
        "data/train/scada_unison_train.csv", encoding="utf-8-sig"
    )
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }
    candidates = build_wind_candidate_matrix(ldaps, gfs)

    prediction_parts = []
    fold_rows = []
    training_rows = []
    knee_rows = []
    curve_bin_rows = []
    selection_parts = []
    local_features = [
        *WAKE_FEATURES,
        *OPTIMAL_GRID_FEATURES,
        *OPTIMAL_GRID_ISSUE_CONTEXT_FEATURES,
    ]

    for group_index, group in enumerate(groups):
        capacity = float(GROUP_CAPACITY_KWH[group])
        base_features = get_or_build_group_feature_cache(
            ldaps,
            gfs,
            group,
            cache_root=args.cache_root,
            rebuild=args.rebuild_feature_cache,
        )
        targets = build_official_aligned_turbine_targets(
            scada_by_group[group], labels, group
        )
        label_one = labels[["kst_dtm", group]].rename(
            columns={"kst_dtm": "forecast_kst_dtm", group: "official_target"}
        )
        label_one["forecast_kst_dtm"] = pd.to_datetime(
            label_one["forecast_kst_dtm"]
        )

        for pred_year in pred_years:
            n_labels = labels.loc[
                labels["kst_dtm"].dt.year.eq(pred_year), group
            ].notna().sum()
            if n_labels < 200:
                continue
            train_years = [year for year in YEARS if year != pred_year]
            features, _, selections = build_fold_features(
                base_features,
                candidates,
                targets,
                group,
                pred_year,
                train_years,
                args,
            )
            selections["pred_year"] = pred_year
            selection_parts.append(selections)
            panel = build_group_local_panel(
                features,
                group,
                local_feature_cols=local_features,
            )
            table = add_group_effective_wind(panel.table, group).merge(
                label_one,
                on="forecast_kst_dtm",
                how="left",
                validate="one_to_one",
            )
            feature_cols = list(panel.full_feature_cols)
            x_all, target_all, wind_all, time_all, year_all = (
                make_per_turbine_sequences(
                    table,
                    feature_cols,
                    window=args.window,
                    target_col="official_target",
                    official_col="effective_wind",
                )
            )
            metric_target = target_all >= capacity * args.target_min_output_ratio
            curve_train = (
                np.isin(year_all, train_years)
                & np.isfinite(target_all)
                & metric_target
            )
            try:
                knee = fit_two_sided_kneedle(
                    wind_all[curve_train],
                    target_all[curve_train] / capacity,
                    n_bins=args.kneedle_bins,
                    min_bin_count=args.kneedle_min_bin_count,
                    min_knee_bins=args.kneedle_min_separation,
                )
                knee_status = "ok"
                curve_bin_rows.extend(curve_rows(knee, group, pred_year))
            except KneedleFitError as error:
                knee = None
                knee_status = f"failed:{error}"

            pred_indices = np.flatnonzero(year_all == pred_year)
            times = pd.to_datetime(np.asarray(time_all)[pred_indices])
            fold_base = baseline.loc[
                baseline["group"].eq(group)
                & baseline["pred_year"].eq(pred_year)
            ]
            aligned = pd.DataFrame(
                {
                    "forecast_kst_dtm": times,
                    "group": group,
                    "pred_year": pred_year,
                    "official_target_model": target_all[pred_indices],
                    "effective_wind": wind_all[pred_indices],
                }
            ).merge(fold_base, on=KEYS, how="inner", validate="one_to_one")
            actual_delta = np.nanmax(
                np.abs(
                    aligned["official_target_model"].to_numpy(float)
                    - aligned["official_target"].to_numpy(float)
                )
            )
            if actual_delta > 1e-2:
                raise ValueError(
                    f"Model/baseline actual mismatch for {group}/{pred_year}: "
                    f"max_delta={actual_delta}"
                )

            if knee is None:
                aligned["mid_pred"] = aligned["base_pred"]
                aligned["regime"] = "unrouted"
                n_train = 0
                train_loss = np.nan
                lower_wind = np.nan
                upper_wind = np.nan
                lower_power = np.nan
                upper_power = np.nan
            else:
                train_keep = (
                    np.isin(year_all, train_years)
                    & np.isfinite(target_all)
                    & metric_target
                    & kneedle_mid_mask(wind_all, knee)
                )
                n_train = int(train_keep.sum())
                if n_train < 500:
                    raise ValueError(
                        f"Too few Kneedle mid rows for {group}/{pred_year}: {n_train}"
                    )
                scaler = SequenceStandardScaler()
                x_train = scaler.fit_transform(x_all[train_keep])
                x_pred = scaler.transform(x_all[pred_indices])
                power_ratio = np.clip(
                    target_all[train_keep] / capacity, 0.0, 1.0
                ).astype(np.float32)
                cube_root_target = np.cbrt(power_ratio).astype(np.float32)
                weights_train = (0.5 + np.sqrt(power_ratio)).astype(np.float32)
                predictions, losses = train_regressor_checkpoints(
                    x_train,
                    cube_root_target,
                    weights_train,
                    x_pred,
                    [args.epoch],
                    args,
                    device,
                    args.seed + group_index * 10000 + pred_year,
                )
                mid_prediction_full = (
                    np.power(np.clip(predictions[args.epoch], 0.0, 1.0), 3)
                    * capacity
                )
                mid_table = pd.DataFrame(
                    {
                        "forecast_kst_dtm": times,
                        "mid_pred_model": mid_prediction_full,
                    }
                )
                aligned = aligned.merge(
                    mid_table,
                    on="forecast_kst_dtm",
                    how="left",
                    validate="one_to_one",
                )
                aligned["mid_pred"] = np.clip(
                    aligned.pop("mid_pred_model"), 0.0, capacity
                )
                mid_mask = kneedle_mid_mask(aligned["effective_wind"], knee)
                aligned["regime"] = np.where(
                    mid_mask,
                    "mid",
                    np.where(
                        aligned["effective_wind"] < knee.lower_wind,
                        "low",
                        "high",
                    ),
                )
                train_loss = losses[args.epoch]
                lower_wind = knee.lower_wind
                upper_wind = knee.upper_wind
                lower_power = knee.lower_power_ratio
                upper_power = knee.upper_power_ratio
                del x_train, x_pred, predictions
                gc.collect()

            aligned["lower_knee_wind"] = lower_wind
            aligned["upper_knee_wind"] = upper_wind
            aligned["lower_knee_power_ratio"] = lower_power
            aligned["upper_knee_power_ratio"] = upper_power
            hard_prediction = np.where(
                aligned["regime"].eq("mid"),
                aligned["mid_pred"],
                aligned["base_pred"],
            )
            variants = {
                "baseline_best": aligned["base_pred"].to_numpy(float),
                "mid_cuberoot_all": aligned["mid_pred"].to_numpy(float),
                "kneedle_hard": hard_prediction,
            }
            for variant, prediction in variants.items():
                part = aligned[
                    [
                        "forecast_kst_dtm",
                        "group",
                        "pred_year",
                        "official_target",
                        "effective_wind",
                        "regime",
                        "base_pred",
                        "mid_pred",
                        "lower_knee_wind",
                        "upper_knee_wind",
                        "lower_knee_power_ratio",
                        "upper_knee_power_ratio",
                    ]
                ].copy()
                part["variant"] = variant
                part["pred"] = np.clip(prediction, 0.0, capacity)
                part = part.dropna(subset=["official_target", "pred"])
                prediction_parts.append(part)
                score, nmae, ficr = group_score(
                    part["official_target"], part["pred"], group
                )
                fold_rows.append(
                    {
                        "variant": variant,
                        "group": group,
                        "pred_year": pred_year,
                        "score": score,
                        "nmae": nmae,
                        "ficr": ficr,
                        "n_rows": len(part),
                    }
                )

            knee_rows.append(
                {
                    "group": group,
                    "pred_year": pred_year,
                    "train_years": ",".join(map(str, train_years)),
                    "status": knee_status,
                    "lower_wind": lower_wind,
                    "upper_wind": upper_wind,
                    "lower_power_ratio": lower_power,
                    "upper_power_ratio": upper_power,
                    "n_curve_train": int(curve_train.sum()),
                    "n_mid_train": n_train,
                    "n_mid_validation": int(aligned["regime"].eq("mid").sum()),
                    "mid_validation_fraction": float(
                        aligned["regime"].eq("mid").mean()
                    ),
                }
            )
            training_rows.append(
                {
                    "group": group,
                    "pred_year": pred_year,
                    "train_years": ",".join(map(str, train_years)),
                    "epoch": args.epoch,
                    "window": args.window,
                    "receptive_field": 1
                    + 2
                    * (args.kernel_size - 1)
                    * sum(2**layer for layer in range(args.num_layers)),
                    "n_features": len(feature_cols),
                    "n_mid_train": n_train,
                    "train_loss": train_loss,
                    "target_min_output_ratio": args.target_min_output_ratio,
                }
            )
            hard_score = fold_rows[-1]["score"]
            base_score = fold_rows[-3]["score"]
            print(
                f"{group} pred_year={pred_year} knee={lower_wind:.3f}~"
                f"{upper_wind:.3f} mid_train={n_train} "
                f"base={base_score:.6f} hard={hard_score:.6f} "
                f"delta={hard_score - base_score:+.6f}",
                flush=True,
            )
            del x_all, target_all, wind_all, time_all, year_all
            gc.collect()
            if args.smoke_test:
                print("smoke test complete after one group-year fold", flush=True)
                return

    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_scores = pd.DataFrame(fold_rows)
    summary, pooled_group_scores = pooled_oof_summary(predictions)
    fold_diagnostics = fold_scores.groupby("variant", as_index=False).agg(
        worst_group_year=("score", "min"),
        std_group_year=("score", lambda values: values.std(ddof=0)),
        n_group_years=("score", "count"),
    )
    summary = summary.merge(fold_diagnostics, on="variant", how="left").sort_values(
        "mean_score", ascending=False
    )
    regime_scores = score_regimes(predictions)
    prefix = args.results_dir / args.stem
    predictions.to_csv(
        f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    fold_scores.to_csv(
        f"{prefix}_fold_scores.csv", index=False, encoding="utf-8-sig"
    )
    pooled_group_scores.to_csv(
        f"{prefix}_pooled_group_scores.csv", index=False, encoding="utf-8-sig"
    )
    regime_scores.to_csv(
        f"{prefix}_regime_scores.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(knee_rows).to_csv(
        f"{prefix}_knees.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(curve_bin_rows).to_csv(
        f"{prefix}_curve_bins.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(training_rows).to_csv(
        f"{prefix}_training.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(selection_parts, ignore_index=True).to_csv(
        f"{prefix}_optimal_grid_selection.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== pooled official OOF summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
