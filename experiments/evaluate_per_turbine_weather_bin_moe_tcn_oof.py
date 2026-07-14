from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from experiments.evaluate_per_turbine_bin_moe_tcn_oof import (
    YEARS,
    build_fold_features,
    group_score,
    parse_checkpoints,
    parse_csv,
    train_regressor_checkpoints,
)
from utils.bin_moe import (
    N_BINS,
    add_centered_weather_regime,
    adjacent_quartile_weights,
    empirical_weather_percentiles,
    fit_weather_quantile_boundaries,
    hard_mix_expert_predictions,
    mix_expert_predictions,
    weather_quantile_bins,
)
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, pooled_oof_summary
from utils.per_turbine_features import get_or_build_group_feature_cache
from utils.per_turbine_optimal_grid_builder import (
    build_wind_candidate_matrix,
)
from utils.per_turbine_scada import (
    apply_turbine_share_shrinkage,
    build_official_aligned_turbine_targets,
    build_static_turbine_share_priors,
    turbine_capacity_kwh,
)
from utils.per_turbine_sequence import SequenceStandardScaler, make_per_turbine_sequences
from utils.power_curve import GROUP_TURBINE_PREFIXES


WEATHER_REGIME_COL = "weather_regime_ws"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--years", default=",".join(map(str, YEARS)))
    parser.add_argument("--checkpoints", default="10,20,40")
    parser.add_argument("--window", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-min-output-ratio", type=float, default=0.10)
    parser.add_argument("--target-share-alpha", type=float, default=1.0)
    parser.add_argument(
        "--feature-variant",
        choices=["optimal_grid_replace_local16", "optimal_grid_issue_context"],
        default="optimal_grid_replace_local16",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--rebuild-optimal-grid-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--stem", default="per_turbine_weather_bin_moe_tcn_w72_oof_v1"
    )
    return parser.parse_args()


def aligned_numeric_column(
    table: pd.DataFrame,
    times,
    column: str,
) -> np.ndarray:
    work = table[["forecast_kst_dtm", column]].copy()
    work["forecast_kst_dtm"] = pd.to_datetime(work["forecast_kst_dtm"])
    if work["forecast_kst_dtm"].duplicated().any():
        raise ValueError(f"Duplicate forecast times while aligning {column}")
    values = pd.to_numeric(
        work.set_index("forecast_kst_dtm")[column].reindex(pd.DatetimeIndex(times)),
        errors="coerce",
    ).to_numpy(dtype=float)
    return values


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.target_share_alpha <= 1.0:
        raise ValueError("--target-share-alpha must be in [0, 1]")
    groups = parse_csv(args.groups)
    pred_years = [int(year) for year in parse_csv(args.years)]
    checkpoints = parse_checkpoints(args.checkpoints)
    if args.smoke_test:
        groups = groups[:1]
        pred_years = pred_years[:1]
        checkpoints = [1]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} groups={groups} years={pred_years} "
        f"checkpoints={checkpoints} window={args.window} h={args.hidden_size} "
        f"layers={args.num_layers} variant={args.feature_variant} "
        f"share={args.target_share_alpha:.2f} weather=median(t-2:t+2)",
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
    print(
        f"candidate_rows={len(candidates.keys)} "
        f"wind_candidates={len(candidates.names)}",
        flush=True,
    )

    prediction_parts = []
    fold_score_rows = []
    router_rows = []
    training_rows = []
    selection_parts = []

    for group in groups:
        group_capacity = float(GROUP_CAPACITY_KWH[group])
        turbine_capacity = float(turbine_capacity_kwh(group))
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
            if labels.loc[
                labels["kst_dtm"].dt.year.eq(pred_year), group
            ].notna().sum() < 200:
                continue
            train_years = [year for year in YEARS if year != pred_year]
            static_shares = build_static_turbine_share_priors(
                targets,
                group,
                train_years,
                target_min_output_ratio=args.target_min_output_ratio,
            )
            fold_targets = apply_turbine_share_shrinkage(
                targets,
                static_shares,
                dynamic_weight=args.target_share_alpha,
            )
            target_one = fold_targets[
                ["forecast_kst_dtm", "turbine_id", "turbine_target"]
            ].copy()
            features, feature_cols, selections = build_fold_features(
                base_features,
                candidates,
                targets,
                group,
                pred_year,
                train_years,
                args,
            )
            features = add_centered_weather_regime(features)
            selections["pred_year"] = pred_year
            selection_parts.append(selections)
            table = (
                features.merge(
                    target_one,
                    on=["forecast_kst_dtm", "turbine_id"],
                    how="left",
                    validate="one_to_one",
                )
                .merge(
                    label_one,
                    on="forecast_kst_dtm",
                    how="left",
                    validate="many_to_one",
                )
            )
            table["forecast_kst_dtm"] = pd.to_datetime(
                table["forecast_kst_dtm"]
            )
            validation = (
                table.loc[
                    table["forecast_kst_dtm"].dt.year.eq(pred_year),
                    ["forecast_kst_dtm", "official_target"],
                ]
                .drop_duplicates("forecast_kst_dtm")
                .sort_values("forecast_kst_dtm")
            )
            group_times = pd.DatetimeIndex(validation["forecast_kst_dtm"])
            actual = validation["official_target"].to_numpy(dtype=float)
            print(
                f"\n{group} pred_year={pred_year} train={train_years} "
                f"features={len(feature_cols)}",
                flush=True,
            )

            baseline_sum = {
                epoch: np.zeros(len(group_times), dtype=float)
                for epoch in checkpoints
            }
            hard_sum = {
                epoch: np.zeros(len(group_times), dtype=float)
                for epoch in checkpoints
            }
            soft_sum = {
                epoch: np.zeros(len(group_times), dtype=float)
                for epoch in checkpoints
            }
            uniform_sum = {
                epoch: np.zeros(len(group_times), dtype=float)
                for epoch in checkpoints
            }

            for turbine_index, turbine in enumerate(GROUP_TURBINE_PREFIXES[group]):
                turbine_table = table.loc[table["turbine_id"].eq(turbine)]
                train_table = turbine_table.loc[
                    turbine_table["forecast_kst_dtm"].dt.year.isin(train_years)
                ]
                val_table = turbine_table.loc[
                    turbine_table["forecast_kst_dtm"].dt.year.eq(pred_year)
                ]
                x_train, y_train, official_train, train_time, _ = (
                    make_per_turbine_sequences(
                        train_table, feature_cols, window=args.window
                    )
                )
                x_val, _, _, val_time, _ = make_per_turbine_sequences(
                    val_table, feature_cols, window=args.window
                )
                val_positions = pd.DatetimeIndex(val_time).get_indexer(group_times)
                if np.any(val_positions < 0):
                    raise ValueError(
                        f"Group/turbine time mismatch for "
                        f"{group}/{turbine}/{pred_year}"
                    )

                train_weather = aligned_numeric_column(
                    train_table, train_time, WEATHER_REGIME_COL
                )
                val_weather = aligned_numeric_column(
                    val_table, pd.DatetimeIndex(val_time)[val_positions], WEATHER_REGIME_COL
                )
                weather_reference = train_weather[np.isfinite(train_weather)]
                boundaries = fit_weather_quantile_boundaries(weather_reference)
                if np.any(~np.isfinite(val_weather)):
                    raise ValueError(
                        f"Nonfinite weather regime for {group}/{turbine}/{pred_year}"
                    )
                all_train_bins = weather_quantile_bins(train_weather, boundaries)
                val_bins = weather_quantile_bins(val_weather, boundaries)
                val_soft_weights = adjacent_quartile_weights(
                    empirical_weather_percentiles(val_weather, weather_reference)
                )

                train_keep = (
                    np.isfinite(y_train)
                    & np.isfinite(official_train)
                    & np.isfinite(train_weather)
                    & (
                        official_train
                        >= group_capacity * args.target_min_output_ratio
                    )
                )
                if int(train_keep.sum()) < 1000:
                    raise ValueError(
                        f"Too few global train rows for "
                        f"{group}/{turbine}/{pred_year}"
                    )
                scaler = SequenceStandardScaler()
                x_train_scaled = scaler.fit_transform(x_train[train_keep])
                x_val_scaled = scaler.transform(x_val[val_positions])
                y_train_norm = np.clip(
                    y_train[train_keep] / turbine_capacity, 0.0, 1.0
                ).astype(np.float32)
                official_train_valid = official_train[train_keep]
                weights = (
                    0.5
                    + np.sqrt(
                        np.clip(official_train_valid / group_capacity, 0.0, 1.0)
                    )
                ).astype(np.float32)
                baseline_predictions, baseline_losses = train_regressor_checkpoints(
                    x_train_scaled,
                    y_train_norm,
                    weights,
                    x_val_scaled,
                    checkpoints,
                    args,
                    device,
                    args.seed
                    + pred_year * 10000
                    + TARGET_COLS.index(group) * 1000
                    + turbine_index,
                )
                for epoch in checkpoints:
                    baseline_sum[epoch] += (
                        baseline_predictions[epoch] * turbine_capacity
                    )
                    training_rows.append(
                        {
                            "group": group,
                            "pred_year": pred_year,
                            "turbine_id": turbine,
                            "model": "global",
                            "bin": -1,
                            "epoch": epoch,
                            "n_train": int(train_keep.sum()),
                            "train_loss": baseline_losses[epoch],
                        }
                    )

                train_bins = all_train_bins[train_keep]
                expert_by_epoch = {
                    epoch: np.zeros((len(group_times), N_BINS), dtype=float)
                    for epoch in checkpoints
                }
                expert_counts = []
                for bin_index in range(N_BINS):
                    bin_keep = train_bins == bin_index
                    expert_counts.append(int(bin_keep.sum()))
                    if int(bin_keep.sum()) < 200:
                        raise ValueError(
                            f"Too few weather-bin rows for "
                            f"{group}/{turbine}/pred{pred_year}/bin{bin_index}"
                        )
                    expert_predictions, expert_losses = train_regressor_checkpoints(
                        x_train_scaled[bin_keep],
                        y_train_norm[bin_keep],
                        weights[bin_keep],
                        x_val_scaled,
                        checkpoints,
                        args,
                        device,
                        args.seed
                        + pred_year * 10000
                        + TARGET_COLS.index(group) * 1000
                        + turbine_index * 10
                        + bin_index
                        + 200000,
                    )
                    for epoch in checkpoints:
                        expert_by_epoch[epoch][:, bin_index] = (
                            expert_predictions[epoch] * turbine_capacity
                        )
                        training_rows.append(
                            {
                                "group": group,
                                "pred_year": pred_year,
                                "turbine_id": turbine,
                                "model": "weather_expert",
                                "bin": bin_index,
                                "epoch": epoch,
                                "n_train": int(bin_keep.sum()),
                                "train_loss": expert_losses[epoch],
                            }
                        )

                router_rows.append(
                    {
                        "group": group,
                        "pred_year": pred_year,
                        "turbine_id": turbine,
                        "q25_ws": boundaries[0],
                        "q50_ws": boundaries[1],
                        "q75_ws": boundaries[2],
                        **{
                            f"train_bin_{index}": count
                            for index, count in enumerate(expert_counts)
                        },
                        **{
                            f"val_bin_{index}": int(np.sum(val_bins == index))
                            for index in range(N_BINS)
                        },
                    }
                )
                for epoch in checkpoints:
                    experts = expert_by_epoch[epoch]
                    hard_sum[epoch] += hard_mix_expert_predictions(
                        experts, np.eye(N_BINS)[val_bins]
                    )
                    soft_sum[epoch] += mix_expert_predictions(
                        experts, val_soft_weights
                    )
                    uniform_sum[epoch] += experts.mean(axis=1)
                print(
                    f"  {turbine}: n={int(train_keep.sum())} "
                    f"weather_bins={expert_counts} "
                    f"q={np.round(boundaries, 3).tolist()}",
                    flush=True,
                )

            for epoch in checkpoints:
                global_prediction = np.clip(
                    baseline_sum[epoch], 0.0, group_capacity
                )
                hard_prediction = np.clip(hard_sum[epoch], 0.0, group_capacity)
                soft_prediction = np.clip(soft_sum[epoch], 0.0, group_capacity)
                variants = {
                    f"global_e{epoch}": global_prediction,
                    f"weather_hard_e{epoch}": hard_prediction,
                    f"weather_soft_e{epoch}": soft_prediction,
                    f"weather_uniform_e{epoch}": np.clip(
                        uniform_sum[epoch], 0.0, group_capacity
                    ),
                    f"global50_weather50_e{epoch}": 0.5
                    * (global_prediction + soft_prediction),
                }
                epoch_scores = {}
                for variant, prediction in variants.items():
                    prediction = np.clip(prediction, 0.0, group_capacity)
                    part = pd.DataFrame(
                        {
                            "forecast_kst_dtm": group_times,
                            "variant": variant,
                            "group": group,
                            "pred_year": pred_year,
                            "official_target": actual,
                            "pred": prediction,
                        }
                    ).dropna(subset=["official_target", "pred"])
                    prediction_parts.append(part)
                    score, nmae, ficr = group_score(
                        part["official_target"], part["pred"], group
                    )
                    epoch_scores[variant] = score
                    fold_score_rows.append(
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
                print(
                    f"  epoch={epoch}: global={epoch_scores[f'global_e{epoch}']:.6f} "
                    f"hard={epoch_scores[f'weather_hard_e{epoch}']:.6f} "
                    f"soft={epoch_scores[f'weather_soft_e{epoch}']:.6f} "
                    f"blend={epoch_scores[f'global50_weather50_e{epoch}']:.6f}",
                    flush=True,
                )
            if args.smoke_test:
                print("smoke test complete after one group-year fold", flush=True)
                return

    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_scores = pd.DataFrame(fold_score_rows)
    summary, pooled_group_scores = pooled_oof_summary(predictions)
    diagnostics = fold_scores.groupby("variant", as_index=False).agg(
        worst_group_year=("score", "min"),
        std_group_year=("score", lambda values: values.std(ddof=0)),
        n_group_years=("score", "count"),
    )
    summary = summary.merge(diagnostics, on="variant", how="left").sort_values(
        "mean_score", ascending=False
    )
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
    pd.DataFrame(router_rows).to_csv(
        f"{prefix}_router.csv", index=False, encoding="utf-8-sig"
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
