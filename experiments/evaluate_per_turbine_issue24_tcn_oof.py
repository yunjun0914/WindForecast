from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from models.issue_block_tcn import IssueBlockTCN
from utils.issue_block_dataset import (
    PerTurbineIssueBlockData,
    make_per_turbine_issue_blocks,
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
from utils.per_turbine_sequence import SequenceStandardScaler
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import TIME_KEY_COLS


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--years", default="2022,2023,2024")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--target-min-output-ratio", type=float, default=0.10)
    parser.add_argument("--target-share-alpha", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--rebuild-optimal-grid-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--stem", default="per_turbine_issue24_tcn_oof_v1")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def turbine_metric(
    actual: np.ndarray,
    prediction: np.ndarray,
    capacity: float,
) -> tuple[float, float, float]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.clip(np.asarray(prediction, dtype=float), 0.0, capacity)
    error_rate = np.abs(prediction - actual) / capacity
    nmae = float(error_rate.mean())
    unit_price = np.select(
        [error_rate <= 0.06, error_rate <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    denominator = float(np.sum(actual * 4.0))
    ficr = float(np.sum(actual * unit_price) / denominator) if denominator > 0 else 0.0
    return 0.5 * (1.0 - nmae) + 0.5 * ficr, nmae, ficr


def predict_normalized(
    model: IssueBlockTCN,
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
        return np.empty((0, 24), dtype=np.float32)
    return np.clip(np.concatenate(parts, axis=0)[..., 0], 0.0, 1.0).astype(
        np.float32
    )


def train_turbine_fold(
    blocks: PerTurbineIssueBlockData,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    group: str,
    turbine: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    set_seed(seed)
    group_capacity = float(GROUP_CAPACITY_KWH[group])
    turbine_capacity = float(turbine_capacity_kwh(group))
    train_features = blocks.features[train_mask]
    train_targets = blocks.targets[train_mask]
    train_official = blocks.official_targets[train_mask]
    train_keep = (
        np.isfinite(train_targets)
        & np.isfinite(train_official)
        & (train_official >= group_capacity * args.target_min_output_ratio)
    )
    issue_keep = train_keep.any(axis=1)
    train_features = train_features[issue_keep]
    train_targets = train_targets[issue_keep]
    train_official = train_official[issue_keep]
    train_keep = train_keep[issue_keep]

    val_targets = blocks.targets[val_mask]
    val_official = blocks.official_targets[val_mask]
    val_keep = (
        np.isfinite(val_targets)
        & np.isfinite(val_official)
        & (val_official >= group_capacity * args.target_min_output_ratio)
    )
    if int(train_keep.sum()) < 500 or int(val_keep.sum()) < 200:
        raise ValueError(
            f"Insufficient issue targets {group}/{turbine}: "
            f"train={int(train_keep.sum())} val={int(val_keep.sum())}"
        )

    scaler = SequenceStandardScaler()
    scaler.fit(train_features[train_keep])
    x_train = scaler.transform(train_features)
    x_val = scaler.transform(blocks.features[val_mask])
    y_train = np.clip(train_targets / turbine_capacity, 0.0, 1.0)
    y_train = np.nan_to_num(y_train, nan=0.0).astype(np.float32)
    observed = train_keep.astype(np.float32)
    weights = (
        0.5 + np.sqrt(np.clip(train_official / group_capacity, 0.0, 1.0))
    )
    weights = np.nan_to_num(weights, nan=0.0).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(weights),
        torch.from_numpy(observed),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    model = IssueBlockTCN(
        input_size=x_train.shape[-1],
        output_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        full_context=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_score = -np.inf
    best_epoch = 0
    best_state = None
    best_nmae = np.nan
    best_ficr = np.nan
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb, wb, mb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)[..., 0]
            weighted_mask = wb * mb
            loss = (
                torch.abs(prediction - yb) * weighted_mask
            ).sum() / torch.clamp(weighted_mask.sum(), min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        val_prediction = (
            predict_normalized(model, x_val, device, args.eval_batch_size)
            * turbine_capacity
        )
        score, nmae, ficr = turbine_metric(
            val_targets[val_keep], val_prediction[val_keep], turbine_capacity
        )
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_nmae = nmae
            best_ficr = ficr
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"No valid checkpoint for {group}/{turbine}")
    model.load_state_dict(best_state)
    val_prediction = (
        predict_normalized(model, x_val, device, args.eval_batch_size)
        * turbine_capacity
    )
    stats = {
        "group": group,
        "turbine_id": turbine,
        "best_epoch": best_epoch,
        "turbine_val_score": best_score,
        "turbine_val_nmae": best_nmae,
        "turbine_val_ficr": best_ficr,
        "n_train_issues": int(issue_keep.sum()),
        "n_train_targets": int(train_keep.sum()),
        "n_val_issues": int(val_mask.sum()),
        "n_val_targets": int(val_keep.sum()),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    del model, optimizer, loader, dataset, x_train, x_val
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return val_prediction, stats


def prepare_fold_table(
    base_features: pd.DataFrame,
    candidates: WindCandidateMatrix,
    targets: pd.DataFrame,
    labels: pd.DataFrame,
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
    keys = [*TIME_KEY_COLS, "turbine_id"]
    features = base_features.merge(
        optimal[keys + OPTIMAL_GRID_FEATURES],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if float(features["optgrid_ws_raw"].notna().mean()) < 0.95:
        raise ValueError(f"Low rebuilt optimal-grid coverage for {group} pred{pred_year}")
    features = add_optimal_grid_issue_context(features)

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
    label_one = labels[["kst_dtm", group]].rename(
        columns={"kst_dtm": "forecast_kst_dtm", group: "official_target"}
    )
    label_one["forecast_kst_dtm"] = pd.to_datetime(label_one["forecast_kst_dtm"])
    table = features.merge(
        target_one,
        on=["forecast_kst_dtm", "turbine_id"],
        how="left",
        validate="one_to_one",
    ).merge(
        label_one,
        on="forecast_kst_dtm",
        how="left",
        validate="many_to_one",
    )
    feature_cols = optimal_grid_input_columns(group, include_issue_context=True)
    return table, feature_cols, selections


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.target_share_alpha <= 1.0:
        raise ValueError("--target-share-alpha must be in [0, 1]")
    groups = parse_csv(args.groups)
    pred_years = [int(year) for year in parse_csv(args.years)]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} groups={groups} years={pred_years} "
        f"h={args.hidden_size} layers={args.num_layers} share={args.target_share_alpha:.2f}",
        flush=True,
    )

    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
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
    turbine_prediction_parts = []
    score_rows = []
    training_rows = []
    selection_parts = []
    model_counter = 0
    variant = "per_turbine_issue24_full"
    for group in groups:
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
        for pred_year in pred_years:
            if labels.loc[
                labels["kst_dtm"].dt.year.eq(pred_year), group
            ].notna().sum() < 200:
                continue
            train_years = [year for year in [2022, 2023, 2024] if year != pred_year]
            table, feature_cols, selections = prepare_fold_table(
                base_features,
                candidates,
                targets,
                labels,
                group,
                pred_year,
                train_years,
                args,
            )
            selections["pred_year"] = pred_year
            selection_parts.append(selections)
            print(
                f"\n{group} pred_year={pred_year} train={train_years} "
                f"features={len(feature_cols)}",
                flush=True,
            )

            fold_turbines = []
            for turbine_index, turbine in enumerate(GROUP_TURBINE_PREFIXES[group]):
                turbine_table = table.loc[table["turbine_id"].eq(turbine)].copy()
                blocks = make_per_turbine_issue_blocks(turbine_table, feature_cols)
                train_mask = blocks.years != pred_year
                val_mask = blocks.years == pred_year
                prediction, stats = train_turbine_fold(
                    blocks,
                    train_mask,
                    val_mask,
                    group,
                    turbine,
                    args,
                    device,
                    args.seed + pred_year * 100 + turbine_index,
                )
                model_counter += 1
                stats.update(
                    {
                        "pred_year": pred_year,
                        "train_years": ",".join(map(str, train_years)),
                        "n_features": len(feature_cols),
                        "target_share_alpha": args.target_share_alpha,
                    }
                )
                training_rows.append(stats)
                turbine_prediction = pd.DataFrame(
                    {
                        "forecast_kst_dtm": blocks.forecast_times[val_mask].reshape(-1),
                        "group": group,
                        "turbine_id": turbine,
                        "pred_year": pred_year,
                        "pred": prediction.reshape(-1),
                    }
                )
                fold_turbines.append(turbine_prediction)
                turbine_prediction_parts.append(turbine_prediction)
                print(
                    f"  {turbine}: epoch={stats['best_epoch']} "
                    f"turbine_score={stats['turbine_val_score']:.5f}",
                    flush=True,
                )
                if args.smoke_test:
                    print("smoke test complete after one turbine-fold", flush=True)
                    return

            fold_turbines = pd.concat(fold_turbines, ignore_index=True)
            expected_turbines = len(GROUP_TURBINE_PREFIXES[group])
            group_prediction = (
                fold_turbines.groupby("forecast_kst_dtm", as_index=False)
                .agg(pred=("pred", "sum"), n_turbines=("turbine_id", "nunique"))
            )
            if not group_prediction["n_turbines"].eq(expected_turbines).all():
                raise ValueError(f"Incomplete issue turbine sum for {group} pred{pred_year}")
            label_one = labels[["kst_dtm", group]].rename(
                columns={"kst_dtm": "forecast_kst_dtm", group: "official_target"}
            )
            group_prediction = group_prediction.merge(
                label_one, on="forecast_kst_dtm", how="inner", validate="one_to_one"
            ).dropna(subset=["official_target"])
            group_prediction["pred"] = group_prediction["pred"].clip(
                0.0, GROUP_CAPACITY_KWH[group]
            )
            nmae, ficr = group_nmae_ficr(
                group_prediction["official_target"],
                group_prediction["pred"],
                GROUP_CAPACITY_KWH[group],
            )
            score = 0.5 * (1.0 - nmae) + 0.5 * ficr
            score_rows.append(
                {
                    "variant": variant,
                    "group": group,
                    "pred_year": pred_year,
                    "score": score,
                    "nmae": nmae,
                    "ficr": ficr,
                    "n_rows": len(group_prediction),
                    "n_features": len(feature_cols),
                }
            )
            group_prediction["variant"] = variant
            group_prediction["group"] = group
            group_prediction["pred_year"] = pred_year
            prediction_parts.append(group_prediction)
            print(
                f"{group} pred_year={pred_year}: score={score:.6f} "
                f"nMAE={nmae:.6f} FiCR={ficr:.6f}",
                flush=True,
            )
        del base_features, targets
        gc.collect()

    predictions = pd.concat(prediction_parts, ignore_index=True)
    scores = pd.DataFrame(score_rows)
    summary, pooled_group_scores = pooled_oof_summary(predictions)
    fold_means = scores.groupby(["variant", "pred_year"], as_index=False).agg(
        score=("score", "mean")
    )
    diagnostics = fold_means.groupby("variant", as_index=False).agg(
        worst_fold=("score", "min"),
        std_fold_score=("score", lambda values: values.std(ddof=0)),
        n_folds=("score", "count"),
    )
    summary = summary.merge(diagnostics, on="variant", how="left")
    summary["n_models"] = model_counter
    summary["n_features"] = scores["n_features"].max()
    summary["issue_hours"] = 24
    prefix = args.results_dir / args.stem
    predictions.to_csv(f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(f"{prefix}_scores.csv", index=False, encoding="utf-8-sig")
    pooled_group_scores.to_csv(
        f"{prefix}_pooled_group_scores.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(training_rows).to_csv(
        f"{prefix}_training.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(selection_parts, ignore_index=True).to_csv(
        f"{prefix}_optimal_grid_selection.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(turbine_prediction_parts, ignore_index=True).to_csv(
        f"{prefix}_turbine_predictions.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== pooled official OOF summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
