from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from models.seqnn import TCNPowerRegressor, TCNRegimeClassifier
from utils.bin_moe import (
    N_BINS,
    capacity_bin_indices,
    hard_mix_expert_predictions,
    mix_expert_predictions,
    oracle_mix_expert_predictions,
)
from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
)
from utils.per_turbine_features import get_or_build_group_feature_cache
from utils.per_turbine_optimal_grid import (
    OPTIMAL_GRID_FEATURES,
    add_optimal_grid_issue_context,
    optimal_grid_input_columns,
)
from utils.per_turbine_optimal_grid_builder import (
    WindCandidateMatrix,
    build_wind_candidate_matrix,
    get_or_build_optimal_grid_fold,
)
from utils.per_turbine_scada import (
    apply_turbine_share_shrinkage,
    build_official_aligned_turbine_targets,
    build_static_turbine_share_priors,
    turbine_capacity_kwh,
)
from utils.per_turbine_sequence import SequenceStandardScaler, make_per_turbine_sequences
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import TIME_KEY_COLS


YEARS = [2022, 2023, 2024]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_checkpoints(value: str) -> list[int]:
    checkpoints = sorted({int(part) for part in parse_csv(value)})
    if not checkpoints or checkpoints[0] <= 0:
        raise ValueError("--checkpoints must contain positive epochs")
    return checkpoints


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
    parser.add_argument("--stem", default="per_turbine_bin_moe_tcn_w72_oof_v1")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_regression(
    model: TCNPowerRegressor,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            parts.append(model(batch).cpu().numpy())
    if not parts:
        return np.empty((0,), dtype=np.float32)
    return np.clip(np.concatenate(parts), 0.0, 1.0).astype(np.float32)


def predict_gate(
    model: TCNRegimeClassifier,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            parts.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    if not parts:
        return np.empty((0, N_BINS), dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def train_regressor_checkpoints(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_val: np.ndarray,
    checkpoints: list[int],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    set_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(weights),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    model = TCNPowerRegressor(
        input_size=x_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    checkpoint_set = set(checkpoints)
    predictions = {}
    losses = {}
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        epoch_losses = []
        for features, target, weight in loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = (torch.abs(prediction - target) * weight).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if epoch in checkpoint_set:
            predictions[epoch] = predict_regression(
                model, x_val, device, args.eval_batch_size
            )
            losses[epoch] = float(np.mean(epoch_losses))

    del model, optimizer, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, losses


def train_gate_checkpoints(
    x_train: np.ndarray,
    bins_train: np.ndarray,
    x_val: np.ndarray,
    checkpoints: list[int],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    set_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(bins_train.astype(np.int64))
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    model = TCNRegimeClassifier(
        input_size=x_train.shape[-1],
        n_classes=N_BINS,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    checkpoint_set = set(checkpoints)
    probabilities = {}
    losses = {}
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        epoch_losses = []
        for features, target_bin in loader:
            features = features.to(device, non_blocking=True)
            target_bin = target_bin.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features), target_bin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if epoch in checkpoint_set:
            probabilities[epoch] = predict_gate(
                model, x_val, device, args.eval_batch_size
            )
            losses[epoch] = float(np.mean(epoch_losses))

    del model, optimizer, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probabilities, losses


def build_fold_features(
    base_features: pd.DataFrame,
    candidates: WindCandidateMatrix,
    targets: pd.DataFrame,
    group: str,
    pred_year: int,
    train_years: list[int],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    optimal, selections = get_or_build_optimal_grid_fold(
        candidates,
        targets,
        group,
        pred_year,
        train_years,
        cache_root=args.cache_root,
        rebuild=args.rebuild_optimal_grid_cache,
    )
    include_issue_context = args.feature_variant == "optimal_grid_issue_context"
    if include_issue_context:
        optimal = add_optimal_grid_issue_context(optimal)
    feature_cols = optimal_grid_input_columns(
        group, include_issue_context=include_issue_context
    )
    optimal_cols = list(OPTIMAL_GRID_FEATURES)
    if include_issue_context:
        optimal_cols.extend(
            [column for column in feature_cols if column.startswith("optgrid_")]
        )
        optimal_cols = list(dict.fromkeys(optimal_cols))
    keys = [*TIME_KEY_COLS, "turbine_id"]
    features = base_features.merge(
        optimal[keys + optimal_cols], on=keys, how="left", validate="one_to_one"
    )
    coverage = float(features["optgrid_ws_raw"].notna().mean())
    if coverage < 0.95:
        raise ValueError(f"Low optimal-grid coverage for {group} pred{pred_year}: {coverage}")
    return features, feature_cols, selections


def build_gate_table(table: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    gate = (
        table.groupby("forecast_kst_dtm", as_index=False)[feature_cols]
        .mean()
        .merge(
            table[["forecast_kst_dtm", "official_target"]].drop_duplicates(
                "forecast_kst_dtm"
            ),
            on="forecast_kst_dtm",
            how="left",
            validate="one_to_one",
        )
    )
    return gate.sort_values("forecast_kst_dtm").reset_index(drop=True)


def gate_diagnostics(
    probabilities: np.ndarray,
    actual_bins: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    selected = np.flatnonzero(valid)
    if len(selected) == 0:
        raise ValueError("No valid gate rows")
    actual = actual_bins[selected]
    forecast = predicted[selected]
    probability = np.clip(
        probabilities[selected, actual], np.finfo(float).eps, 1.0
    )
    return {
        "gate_exact_accuracy": float(np.mean(forecast == actual)),
        "gate_adjacent_accuracy": float(np.mean(np.abs(forecast - actual) <= 1)),
        "gate_nll": float(-np.log(probability).mean()),
        "gate_mean_confidence": float(probabilities[selected].max(axis=1).mean()),
        "n_gate_rows": int(len(selected)),
    }


def group_score(actual, prediction, group: str) -> tuple[float, float, float]:
    nmae, ficr = group_nmae_ficr(
        actual, prediction, GROUP_CAPACITY_KWH[group]
    )
    return 0.5 * (1.0 - nmae) + 0.5 * ficr, nmae, ficr


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
        f"device={device} groups={groups} years={pred_years} checkpoints={checkpoints} "
        f"window={args.window} h={args.hidden_size} layers={args.num_layers} "
        f"variant={args.feature_variant} share={args.target_share_alpha:.2f}",
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
        f"candidate_rows={len(candidates.keys)} wind_candidates={len(candidates.names)}",
        flush=True,
    )

    prediction_parts = []
    fold_score_rows = []
    gate_rows = []
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
            print(
                f"\n{group} pred_year={pred_year} train={train_years} "
                f"features={len(feature_cols)}",
                flush=True,
            )

            gate_table = build_gate_table(table, feature_cols)
            gate_train = gate_table.loc[
                gate_table["forecast_kst_dtm"].dt.year.isin(train_years)
            ]
            gate_val = gate_table.loc[
                gate_table["forecast_kst_dtm"].dt.year.eq(pred_year)
            ]
            x_gate_train, _, official_gate_train, _, _ = make_per_turbine_sequences(
                gate_train,
                feature_cols,
                window=args.window,
                target_col="official_target",
            )
            x_gate_val, _, official_gate_val, gate_val_time, _ = (
                make_per_turbine_sequences(
                    gate_val,
                    feature_cols,
                    window=args.window,
                    target_col="official_target",
                )
            )
            gate_train_keep = (
                np.isfinite(official_gate_train)
                & (
                    official_gate_train
                    >= group_capacity * args.target_min_output_ratio
                )
            )
            gate_scaler = SequenceStandardScaler()
            x_gate_train_scaled = gate_scaler.fit_transform(
                x_gate_train[gate_train_keep]
            )
            x_gate_val_scaled = gate_scaler.transform(x_gate_val)
            gate_train_bins = capacity_bin_indices(
                official_gate_train[gate_train_keep], group_capacity
            )
            gate_probabilities, gate_losses = train_gate_checkpoints(
                x_gate_train_scaled,
                gate_train_bins,
                x_gate_val_scaled,
                checkpoints,
                args,
                device,
                args.seed + pred_year * 1000 + TARGET_COLS.index(group) * 100,
            )
            gate_times = pd.DatetimeIndex(gate_val_time)
            actual = official_gate_val.astype(float)
            actual_bins = capacity_bin_indices(actual, group_capacity)
            gate_valid = (
                np.isfinite(actual)
                & (actual >= group_capacity * args.target_min_output_ratio)
            )
            for epoch in checkpoints:
                gate_rows.append(
                    {
                        "group": group,
                        "pred_year": pred_year,
                        "epoch": epoch,
                        "gate_train_loss": gate_losses[epoch],
                        **gate_diagnostics(
                            gate_probabilities[epoch], actual_bins, gate_valid
                        ),
                    }
                )

            baseline_sum = {
                epoch: np.zeros(len(gate_times), dtype=float)
                for epoch in checkpoints
            }
            expert_sum = {
                epoch: np.zeros((len(gate_times), N_BINS), dtype=float)
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
                x_train, y_train, official_train, _, _ = make_per_turbine_sequences(
                    train_table, feature_cols, window=args.window
                )
                x_val, _, _, val_time, _ = make_per_turbine_sequences(
                    val_table, feature_cols, window=args.window
                )
                val_positions = pd.DatetimeIndex(val_time).get_indexer(gate_times)
                if np.any(val_positions < 0):
                    raise ValueError(
                        f"Gate/turbine time mismatch for {group}/{turbine}/{pred_year}"
                    )
                train_keep = (
                    np.isfinite(y_train)
                    & np.isfinite(official_train)
                    & (
                        official_train
                        >= group_capacity * args.target_min_output_ratio
                    )
                )
                if int(train_keep.sum()) < 1000:
                    raise ValueError(
                        f"Too few global train rows for {group}/{turbine}/{pred_year}"
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

                train_bins = capacity_bin_indices(
                    official_train_valid, group_capacity
                )
                for bin_index in range(N_BINS):
                    bin_keep = train_bins == bin_index
                    if int(bin_keep.sum()) < 200:
                        raise ValueError(
                            f"Too few bin rows for {group}/{turbine}/pred{pred_year}/bin{bin_index}"
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
                        + 100000,
                    )
                    for epoch in checkpoints:
                        expert_sum[epoch][:, bin_index] += (
                            expert_predictions[epoch] * turbine_capacity
                        )
                        training_rows.append(
                            {
                                "group": group,
                                "pred_year": pred_year,
                                "turbine_id": turbine,
                                "model": "expert",
                                "bin": bin_index,
                                "epoch": epoch,
                                "n_train": int(bin_keep.sum()),
                                "train_loss": expert_losses[epoch],
                            }
                        )
                print(
                    f"  {turbine}: n={int(train_keep.sum())} "
                    f"bins={np.bincount(train_bins, minlength=N_BINS).tolist()}",
                    flush=True,
                )

            for epoch in checkpoints:
                experts = np.clip(expert_sum[epoch], 0.0, group_capacity)
                global_prediction = np.clip(
                    baseline_sum[epoch], 0.0, group_capacity
                )
                soft_prediction = mix_expert_predictions(
                    experts, gate_probabilities[epoch]
                )
                variants = {
                    f"global_e{epoch}": global_prediction,
                    f"moe_soft_e{epoch}": soft_prediction,
                    f"moe_hard_e{epoch}": hard_mix_expert_predictions(
                        experts, gate_probabilities[epoch]
                    ),
                    f"moe_uniform_e{epoch}": experts.mean(axis=1),
                    f"moe_oracle_e{epoch}": oracle_mix_expert_predictions(
                        experts, actual_bins
                    ),
                    f"global50_moe50_e{epoch}": 0.5
                    * (global_prediction + soft_prediction),
                }
                for variant, prediction in variants.items():
                    prediction = np.clip(prediction, 0.0, group_capacity)
                    part = pd.DataFrame(
                        {
                            "forecast_kst_dtm": gate_times,
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
                print(
                    f"  epoch={epoch}: global={fold_score_rows[-6]['score']:.6f} "
                    f"soft={fold_score_rows[-5]['score']:.6f} "
                    f"oracle={fold_score_rows[-2]['score']:.6f}",
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
    pd.DataFrame(gate_rows).to_csv(
        f"{prefix}_gate_diagnostics.csv", index=False, encoding="utf-8-sig"
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
