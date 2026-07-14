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
    parse_checkpoints,
    parse_csv,
    train_regressor_checkpoints,
)
from utils.group_local_panel import build_group_local_panel
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, pooled_oof_summary
from utils.per_turbine_features import get_or_build_group_feature_cache
from utils.per_turbine_optimal_grid_builder import build_wind_candidate_matrix
from utils.per_turbine_scada import build_official_aligned_turbine_targets
from utils.per_turbine_sequence import SequenceStandardScaler, make_per_turbine_sequences


VARIANTS = ("mean_panel", "full_panel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--years", default=",".join(map(str, YEARS)))
    parser.add_argument("--checkpoints", default="5,10,20,40")
    parser.add_argument("--fallback-epoch", type=int, default=10)
    parser.add_argument("--window", type=int, default=29)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-min-output-ratio", type=float, default=0.10)
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
    parser.add_argument("--stem", default="group_local_panel_tcn_w29_oof_v1")
    return parser.parse_args()


def fit_split_checkpoints(
    features: np.ndarray,
    target: np.ndarray,
    years: np.ndarray,
    train_years: list[int],
    pred_year: int,
    capacity: float,
    checkpoints: list[int],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[int, float], np.ndarray, int]:
    train_keep = (
        np.isin(years, train_years)
        & np.isfinite(target)
        & (target >= capacity * args.target_min_output_ratio)
    )
    pred_indices = np.flatnonzero(years == pred_year)
    if int(train_keep.sum()) < 1000:
        raise ValueError(
            f"Too few direct group train rows: years={train_years} "
            f"n={int(train_keep.sum())}"
        )
    if len(pred_indices) < 200:
        raise ValueError(f"Too few direct group prediction rows for {pred_year}")

    scaler = SequenceStandardScaler()
    x_train = scaler.fit_transform(features[train_keep])
    x_pred = scaler.transform(features[pred_indices])
    y_train = np.clip(target[train_keep] / capacity, 0.0, 1.0).astype(
        np.float32
    )
    weights = (0.5 + np.sqrt(y_train)).astype(np.float32)
    predictions, losses = train_regressor_checkpoints(
        x_train,
        y_train,
        weights,
        x_pred,
        checkpoints,
        args,
        device,
        seed,
    )
    del x_train, x_pred
    gc.collect()
    return predictions, losses, pred_indices, int(train_keep.sum())


def select_epoch_inner_year(
    features: np.ndarray,
    target: np.ndarray,
    years: np.ndarray,
    outer_train_years: list[int],
    group: str,
    checkpoints: list[int],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[int, str, list[dict[str, float]]]:
    capacity = float(GROUP_CAPACITY_KWH[group])
    available = [
        year
        for year in outer_train_years
        if int(np.sum((years == year) & np.isfinite(target))) >= 200
    ]
    if len(available) < 2:
        return args.fallback_epoch, "fixed_single_train_year", []

    rows = []
    for split_index, inner_val_year in enumerate(available):
        inner_train_years = [
            year for year in available if year != inner_val_year
        ]
        predictions, losses, pred_indices, n_train = fit_split_checkpoints(
            features,
            target,
            years,
            inner_train_years,
            inner_val_year,
            capacity,
            checkpoints,
            args,
            device,
            seed + split_index,
        )
        actual = target[pred_indices]
        for epoch in checkpoints:
            prediction = predictions[epoch] * capacity
            score, nmae, ficr = group_score(actual, prediction, group)
            rows.append(
                {
                    "inner_val_year": inner_val_year,
                    "epoch": epoch,
                    "score": score,
                    "nmae": nmae,
                    "ficr": ficr,
                    "train_loss": losses[epoch],
                    "n_train": n_train,
                }
            )

    scores = pd.DataFrame(rows).groupby("epoch", as_index=False)["score"].mean()
    selected = int(
        scores.sort_values(["score", "epoch"], ascending=[False, True]).iloc[0][
            "epoch"
        ]
    )
    return selected, "nested_inner_year_score", rows


def main() -> None:
    args = parse_args()
    groups = parse_csv(args.groups)
    pred_years = [int(year) for year in parse_csv(args.years)]
    checkpoints = parse_checkpoints(args.checkpoints)
    if args.fallback_epoch not in checkpoints:
        raise ValueError("--fallback-epoch must be included in --checkpoints")
    if args.smoke_test:
        groups = groups[:1]
        pred_years = pred_years[:1]
        checkpoints = [1]
        args.fallback_epoch = 1

    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} groups={groups} years={pred_years} "
        f"window={args.window} h={args.hidden_size} layers={args.num_layers} "
        f"checkpoints={checkpoints} variant={args.feature_variant}",
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
    inner_score_rows = []
    training_rows = []
    selection_parts = []

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
            if labels.loc[
                labels["kst_dtm"].dt.year.eq(pred_year), group
            ].notna().sum() < 200:
                continue
            outer_train_years = [year for year in YEARS if year != pred_year]
            features, _, selections = build_fold_features(
                base_features,
                candidates,
                targets,
                group,
                pred_year,
                outer_train_years,
                args,
            )
            selections["pred_year"] = pred_year
            selection_parts.append(selections)
            panel = build_group_local_panel(features, group)
            table = panel.table.merge(
                label_one,
                on="forecast_kst_dtm",
                how="left",
                validate="one_to_one",
            )
            feature_sets = {
                "mean_panel": list(panel.mean_feature_cols),
                "full_panel": list(panel.full_feature_cols),
            }
            print(
                f"\n{group} pred_year={pred_year} train={outer_train_years} "
                f"mean_features={len(panel.mean_feature_cols)} "
                f"full_features={len(panel.full_feature_cols)}",
                flush=True,
            )

            for variant_index, variant in enumerate(VARIANTS):
                feature_cols = feature_sets[variant]
                x_all, target_all, _, time_all, year_all = (
                    make_per_turbine_sequences(
                        table,
                        feature_cols,
                        window=args.window,
                        target_col="official_target",
                        official_col="official_target",
                    )
                )
                seed = (
                    args.seed
                    + group_index * 100000
                    + pred_year * 10
                    + variant_index * 10000
                )
                selected_epoch, selection_method, inner_rows = (
                    select_epoch_inner_year(
                        x_all,
                        target_all,
                        year_all,
                        outer_train_years,
                        group,
                        checkpoints,
                        args,
                        device,
                        seed,
                    )
                )
                for row in inner_rows:
                    inner_score_rows.append(
                        {
                            "variant": variant,
                            "group": group,
                            "pred_year": pred_year,
                            **row,
                        }
                    )

                predictions, losses, pred_indices, n_train = fit_split_checkpoints(
                    x_all,
                    target_all,
                    year_all,
                    outer_train_years,
                    pred_year,
                    capacity,
                    [selected_epoch],
                    args,
                    device,
                    seed + 900000,
                )
                prediction = np.clip(
                    predictions[selected_epoch] * capacity, 0.0, capacity
                )
                actual = target_all[pred_indices]
                times = pd.to_datetime(np.asarray(time_all)[pred_indices])
                part = pd.DataFrame(
                    {
                        "forecast_kst_dtm": times,
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
                training_rows.append(
                    {
                        "variant": variant,
                        "group": group,
                        "pred_year": pred_year,
                        "train_years": ",".join(map(str, outer_train_years)),
                        "selection_method": selection_method,
                        "selected_epoch": selected_epoch,
                        "n_features": len(feature_cols),
                        "n_train": n_train,
                        "train_loss": losses[selected_epoch],
                        "outer_score": score,
                        "outer_nmae": nmae,
                        "outer_ficr": ficr,
                    }
                )
                print(
                    f"  {variant}: epoch={selected_epoch} "
                    f"method={selection_method} score={score:.6f} "
                    f"nmae={nmae:.6f} ficr={ficr:.6f}",
                    flush=True,
                )
                del x_all, target_all, time_all, year_all
                gc.collect()

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
    pd.DataFrame(inner_score_rows).to_csv(
        f"{prefix}_inner_scores.csv", index=False, encoding="utf-8-sig"
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
