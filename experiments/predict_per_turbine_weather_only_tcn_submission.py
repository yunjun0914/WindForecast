from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from models.seqnn import TCNPowerRegressor
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS
from utils.per_turbine_features import get_or_build_group_feature_cache
from utils.per_turbine_optimal_grid import (
    OPTIMAL_GRID_ISSUE_CONTEXT_TAG,
    OPTIMAL_GRID_REPLACE_TAG,
    load_optimal_grid_full_features,
    optimal_grid_input_columns,
)
from utils.per_turbine_scada import (
    apply_turbine_share_shrinkage,
    build_official_aligned_turbine_targets,
    build_static_turbine_share_priors,
    turbine_capacity_kwh,
)
from utils.per_turbine_sequence import SequenceStandardScaler, make_per_turbine_sequences
from utils.power_curve import GROUP_TURBINE_PREFIXES


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epoch_map(path: str) -> dict[tuple[str, str], int]:
    table = pd.read_csv(path, encoding="utf-8-sig")
    required = ["group", "turbine_id", "best_epoch"]
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise ValueError(f"Epoch source {path} missing columns: {missing}")
    medians = table.groupby(["group", "turbine_id"])["best_epoch"].median()
    return {key: max(1, int(round(value))) for key, value in medians.items()}


def train_tcn_predict(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    feature_cols: list[str],
    group: str,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    x_train, y_train, official_train, _, _ = make_per_turbine_sequences(
        train_table,
        feature_cols,
        window=args.tcn_window,
    )
    x_test, _, _, _, _ = make_per_turbine_sequences(
        test_table,
        feature_cols,
        window=args.tcn_window,
    )
    keep = (
        np.isfinite(y_train)
        & np.isfinite(official_train)
        & (official_train >= GROUP_CAPACITY_KWH[group] * 0.10)
    )
    x_train = x_train[keep]
    y_train = y_train[keep]
    official_train = official_train[keep]
    scaler = SequenceStandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    capacity = turbine_capacity_kwh(group)
    y_norm = np.clip(y_train / capacity, 0, 1).astype(np.float32)
    weights = (
        0.5 + np.sqrt(np.clip(official_train / GROUP_CAPACITY_KWH[group], 0, 1))
    ).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_norm),
        torch.from_numpy(weights),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.tcn_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    model = TCNPowerRegressor(
        input_size=len(feature_cols),
        hidden_size=args.tcn_hidden_size,
        num_layers=args.tcn_num_layers,
        kernel_size=args.tcn_kernel_size,
        dropout=args.tcn_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = (torch.abs(pred - yb) * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x_test), 4096):
            xb = torch.from_numpy(x_test[start : start + 4096]).to(device)
            parts.append(model(xb).cpu().numpy())
    pred = np.concatenate(parts)
    del model, optimizer, loader, dataset, x_train, x_test
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.clip(pred, 0, 1) * capacity


def branch_submission(
    sample: pd.DataFrame,
    turbine_predictions: pd.DataFrame,
    branch: str,
) -> pd.DataFrame:
    grouped = (
        turbine_predictions.groupby(["forecast_kst_dtm", "group"], as_index=False)[branch]
        .sum()
        .pivot(index="forecast_kst_dtm", columns="group", values=branch)
        .reset_index()
    )
    out = sample[["forecast_id", "forecast_kst_dtm"]].merge(
        grouped,
        on="forecast_kst_dtm",
        how="left",
    )
    for group in TARGET_COLS:
        out[group] = out[group].clip(0, GROUP_CAPACITY_KWH[group])
    return out[["forecast_id", "forecast_kst_dtm", *TARGET_COLS]]


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    expected_cols = ["forecast_id", "forecast_kst_dtm", *TARGET_COLS]
    if list(submission.columns) != expected_cols:
        raise ValueError(f"Submission columns differ: {submission.columns.tolist()}")
    if len(submission) != len(sample):
        raise ValueError(f"Submission rows differ: {len(submission)} != {len(sample)}")
    if not submission["forecast_id"].equals(sample["forecast_id"]):
        raise ValueError("forecast_id order differs from sample submission")
    if submission[TARGET_COLS].isna().any().any():
        raise ValueError("Submission contains missing predictions")
    for group in TARGET_COLS:
        if not submission[group].between(0, GROUP_CAPACITY_KWH[group]).all():
            raise ValueError(f"Submission values out of range for {group}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tcn-oof-training",
        default="results/per_turbine_tcn_optgrid_weather_only_oof_v1_training.csv",
    )
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--tcn-window", type=int, default=72)
    parser.add_argument("--tcn-batch-size", type=int, default=512)
    parser.add_argument("--tcn-hidden-size", type=int, default=64)
    parser.add_argument("--tcn-num-layers", type=int, default=1)
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--tcn-dropout", type=float, default=0.10)
    parser.add_argument("--target-share-alpha", type=float, default=1.0)
    parser.add_argument("--include-issue-context", action="store_true")
    parser.add_argument(
        "--turbine-output",
        default="results/per_turbine_optgrid_weather_only_tcn_full_v1_turbine_predictions.csv",
    )
    parser.add_argument(
        "--training-output",
        default="results/per_turbine_optgrid_weather_only_tcn_full_v1_training_epochs.csv",
    )
    parser.add_argument(
        "--submission-output",
        default="results/submission_per_turbine_optgrid_weather_only_tcn_w72_full_v1.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.target_share_alpha <= 1.0:
        raise ValueError("--target-share-alpha must be in [0, 1]")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} feature_variant="
        f"{OPTIMAL_GRID_ISSUE_CONTEXT_TAG if args.include_issue_context else OPTIMAL_GRID_REPLACE_TAG} "
        f"window={args.tcn_window} hidden={args.tcn_hidden_size} "
        f"layers={args.tcn_num_layers} kernel={args.tcn_kernel_size} "
        f"share_alpha={args.target_share_alpha:.2f}",
        flush=True,
    )
    tcn_epochs = epoch_map(args.tcn_oof_training)

    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv("data/test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv("data/test/gfs_test.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    sample = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")
    sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"])
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }

    turbine_parts = []
    training_rows = []
    for group_index, group in enumerate(TARGET_COLS):
        print(f"\n=== prepare optimal-grid full train/test {group} ===", flush=True)
        base_train = get_or_build_group_feature_cache(
            ldaps_train, gfs_train, group, cache_root=args.cache_root
        )
        base_test = get_or_build_group_feature_cache(
            ldaps_test,
            gfs_test,
            group,
            cache_root=args.cache_root,
            cache_tag="test",
        )
        train_features, test_features = load_optimal_grid_full_features(
            base_train,
            base_test,
            args.cache_root,
            group,
            include_issue_context=args.include_issue_context,
        )
        feature_cols = optimal_grid_input_columns(
            group, include_issue_context=args.include_issue_context
        )
        targets = build_official_aligned_turbine_targets(scada_by_group[group], labels, group)
        train_years = sorted(
            targets.loc[targets["official_target"].notna(), "year"].astype(int).unique()
        )
        static_shares = build_static_turbine_share_priors(
            targets, group, train_years, target_min_output_ratio=0.10
        )
        targets = apply_turbine_share_shrinkage(
            targets, static_shares, dynamic_weight=args.target_share_alpha
        )
        keys = ["forecast_kst_dtm", "turbine_id"]
        target_one = targets[["forecast_kst_dtm", "turbine_id", "turbine_target"]]
        label_one = labels[["kst_dtm", group]].rename(
            columns={"kst_dtm": "forecast_kst_dtm", group: "official_target"}
        )
        train_table = train_features.merge(target_one, on=keys, how="left")
        train_table = train_table.merge(label_one, on="forecast_kst_dtm", how="left")
        test_table = test_features.copy()
        test_table["turbine_target"] = np.nan
        test_table["official_target"] = np.nan
        for table in [train_table, test_table]:
            table["forecast_kst_dtm"] = pd.to_datetime(table["forecast_kst_dtm"])
            table["data_available_kst_dtm"] = pd.to_datetime(table["data_available_kst_dtm"])
        print(f"{group}: features={len(feature_cols)}", flush=True)

        for turbine_index, turbine in enumerate(GROUP_TURBINE_PREFIXES[group]):
            train_one = (
                train_table.loc[train_table["turbine_id"].eq(turbine)]
                .sort_values("forecast_kst_dtm")
                .reset_index(drop=True)
            )
            test_one = (
                test_table.loc[test_table["turbine_id"].eq(turbine)]
                .sort_values("forecast_kst_dtm")
                .reset_index(drop=True)
            )
            tcn_epoch = tcn_epochs[(group, turbine)]
            seed_offset = group_index * 100 + turbine_index
            tcn_pred = train_tcn_predict(
                train_one,
                test_one,
                feature_cols,
                group,
                tcn_epoch,
                args,
                device,
                args.seed + 1000 + seed_offset,
            )
            if len(tcn_pred) != len(test_one):
                raise ValueError(f"Prediction length mismatch for {turbine}")
            turbine_parts.append(
                pd.DataFrame(
                    {
                        "forecast_kst_dtm": test_one["forecast_kst_dtm"],
                        "group": group,
                        "turbine_id": turbine,
                        "tcn": tcn_pred,
                    }
                )
            )
            training_rows.append(
                {
                    "group": group,
                    "turbine_id": turbine,
                    "tcn_epochs": tcn_epoch,
                    "n_features": len(feature_cols),
                    "window": args.tcn_window,
                    "hidden_size": args.tcn_hidden_size,
                    "num_layers": args.tcn_num_layers,
                    "kernel_size": args.tcn_kernel_size,
                    "target_share_alpha": args.target_share_alpha,
                    "static_share": float(static_shares.loc[turbine]),
                    "include_issue_context": args.include_issue_context,
                }
            )
            print(
                f"  {turbine}: tcn_epoch={tcn_epoch}",
                flush=True,
            )

    turbine_predictions = pd.concat(turbine_parts, ignore_index=True)
    turbine_path = Path(args.turbine_output)
    training_path = Path(args.training_output)
    turbine_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.parent.mkdir(parents=True, exist_ok=True)
    turbine_predictions.to_csv(turbine_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(training_rows).to_csv(training_path, index=False, encoding="utf-8-sig")
    print(f"saved turbine predictions: {turbine_path} rows={len(turbine_predictions)}", flush=True)

    submission = branch_submission(sample, turbine_predictions, "tcn")
    validate_submission(submission, sample)
    path = Path(args.submission_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved tcn: {path}", flush=True)


if __name__ == "__main__":
    main()
