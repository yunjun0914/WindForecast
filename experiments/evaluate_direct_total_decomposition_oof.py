from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from torch.utils.data import DataLoader, TensorDataset

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from models.issue_block_tcn import IssueBlockTCN
from utils.issue_block_dataset import IssueBlockData, make_issue_blocks
from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
)
from utils.seq_dataset import SequenceStandardScaler, build_seqnn_weather


TARGET_SPECS = {
    "s12": {
        "groups": ["kpx_group_1", "kpx_group_2"],
        "capacity": GROUP_CAPACITY_KWH["kpx_group_1"]
        + GROUP_CAPACITY_KWH["kpx_group_2"],
    },
    "s123": {
        "groups": TARGET_COLS,
        "capacity": sum(GROUP_CAPACITY_KWH.values()),
    },
}


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--tree-estimators", type=int, default=1200)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=Path("results/per_turbine_issue24_tcn_oof_v1_predictions.csv"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="direct_total_decomposition_oof_v1")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def total_metric(
    actual: np.ndarray,
    prediction: np.ndarray,
    capacity: float,
) -> tuple[float, float, float]:
    nmae, ficr = group_nmae_ficr(actual, prediction, capacity)
    return 0.5 * (1.0 - nmae) + 0.5 * ficr, nmae, ficr


def predict_tcn(
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
            parts.append(model(batch).cpu().numpy()[..., 0])
    if not parts:
        return np.empty((0, 24), dtype=np.float32)
    return np.clip(np.concatenate(parts), 0.0, 1.0).astype(np.float32)


def train_tcn_total_fold(
    blocks: IssueBlockData,
    target_index: int,
    pred_year: int,
    capacity: float,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    set_seed(seed)
    train_mask = blocks.years != pred_year
    val_mask = blocks.years == pred_year
    train_targets = blocks.targets[train_mask, :, target_index]
    train_issue_keep = np.isfinite(train_targets).any(axis=1)
    x_train_raw = blocks.features[train_mask][train_issue_keep]
    y_train_raw = train_targets[train_issue_keep]
    x_val_raw = blocks.features[val_mask]
    y_val = blocks.targets[val_mask, :, target_index]
    if int(np.isfinite(y_train_raw).sum()) < 500 or int(np.isfinite(y_val).sum()) < 200:
        raise ValueError(f"Insufficient total targets for pred_year={pred_year}")

    scaler = SequenceStandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    observed = np.isfinite(y_train_raw).astype(np.float32)
    y_train = np.clip(y_train_raw / capacity, 0.0, 1.0)
    y_train = np.nan_to_num(y_train, nan=0.0).astype(np.float32)
    weights = 0.5 + np.sqrt(np.clip(y_train, 0.0, 1.0))
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(weights.astype(np.float32)),
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
    val_keep = np.isfinite(y_val) & (y_val >= capacity * 0.10)
    best_score = -np.inf
    best_epoch = 0
    best_state = None
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

        prediction = predict_tcn(model, x_val, device, args.eval_batch_size) * capacity
        score, _, _ = total_metric(y_val[val_keep], prediction[val_keep], capacity)
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No direct-total TCN checkpoint selected")
    model.load_state_dict(best_state)
    prediction = predict_tcn(model, x_val, device, args.eval_batch_size) * capacity
    score, nmae, ficr = total_metric(y_val[val_keep], prediction[val_keep], capacity)
    return prediction, {
        "best_epoch": best_epoch,
        "score": score,
        "nmae": nmae,
        "ficr": ficr,
        "n_train_issues": int(train_issue_keep.sum()),
        "n_val_issues": int(val_mask.sum()),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def tree_params(args: argparse.Namespace, seed: int) -> dict[str, object]:
    return {
        "objective": "regression_l1",
        "n_estimators": args.tree_estimators,
        "learning_rate": 0.02,
        "num_leaves": 64,
        "max_depth": 6,
        "min_child_samples": 80,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.02,
        "reg_lambda": 2.0,
        "min_split_gain": 0.02,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def train_tree_total_fold(
    blocks: IssueBlockData,
    target_index: int,
    pred_year: int,
    capacity: float,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    train_mask = blocks.years != pred_year
    val_mask = blocks.years == pred_year
    x_train = blocks.features[train_mask].reshape(-1, blocks.features.shape[-1])
    y_train = blocks.targets[train_mask, :, target_index].reshape(-1)
    x_val = blocks.features[val_mask].reshape(-1, blocks.features.shape[-1])
    y_val = blocks.targets[val_mask, :, target_index].reshape(-1)
    train_keep = np.isfinite(y_train) & (y_train >= capacity * 0.10)
    val_keep = np.isfinite(y_val) & (y_val >= capacity * 0.10)
    weights = 0.5 + np.sqrt(np.clip(y_train[train_keep] / capacity, 0.0, 1.0))
    model = LGBMRegressor(**tree_params(args, seed))
    model.fit(
        x_train[train_keep],
        y_train[train_keep] / capacity,
        sample_weight=weights,
    )
    prediction = np.clip(model.predict(x_val), 0.0, 1.0) * capacity
    score, nmae, ficr = total_metric(y_val[val_keep], prediction[val_keep], capacity)
    return prediction.reshape(-1, 24), {
        "best_epoch": args.tree_estimators,
        "score": score,
        "nmae": nmae,
        "ficr": ficr,
        "n_train_rows": int(train_keep.sum()),
        "n_val_rows": int(val_keep.sum()),
        "n_parameters": np.nan,
    }


def add_mean_total_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["forecast_kst_dtm", "pred_year", "target", "actual"]
    pivot = predictions.pivot_table(index=keys, columns="model", values="pred")
    if not {"tcn", "tree"}.issubset(pivot.columns):
        return predictions
    mean = pivot[["tcn", "tree"]].mean(axis=1).rename("pred").reset_index()
    mean["model"] = "mean"
    return pd.concat([predictions, mean[predictions.columns]], ignore_index=True)


def redistribute_total(
    total_prediction: np.ndarray,
    base_prediction: np.ndarray,
    capacities: np.ndarray,
) -> np.ndarray:
    total_prediction = np.asarray(total_prediction, dtype=float)
    base_prediction = np.asarray(base_prediction, dtype=float)
    capacities = np.asarray(capacities, dtype=float)
    denominator = base_prediction.sum(axis=1, keepdims=True)
    fallback = capacities / capacities.sum()
    shares = np.divide(
        base_prediction,
        denominator,
        out=np.broadcast_to(fallback, base_prediction.shape).copy(),
        where=denominator > 1e-8,
    )
    return shares * total_prediction[:, None]


def prepare_base_wide(base_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(base_path, encoding="utf-8-sig")
    base["forecast_kst_dtm"] = pd.to_datetime(base["forecast_kst_dtm"])
    base_pred = base.pivot_table(
        index="forecast_kst_dtm", columns="group", values="pred", aggfunc="first"
    )
    base_actual = base.pivot_table(
        index="forecast_kst_dtm",
        columns="group",
        values="official_target",
        aggfunc="first",
    )
    return base_pred, base_actual


def make_redistributed_variant(
    total: pd.DataFrame,
    base_pred: pd.DataFrame,
    base_actual: pd.DataFrame,
    target: str,
    model: str,
    alpha: float,
) -> pd.DataFrame:
    groups = TARGET_SPECS[target]["groups"]
    direct = total.loc[
        total["target"].eq(target) & total["model"].eq(model)
    ].set_index("forecast_kst_dtm")
    common = direct.index.intersection(base_pred.dropna(subset=groups).index)
    direct = direct.loc[common]
    base_groups = base_pred.loc[common, groups]
    base_total = base_groups.sum(axis=1).to_numpy(float)
    blended_total = (1.0 - alpha) * base_total + alpha * direct["pred"].to_numpy(float)
    capacities = np.asarray([GROUP_CAPACITY_KWH[group] for group in groups], dtype=float)
    redistributed = redistribute_total(
        blended_total,
        base_groups.to_numpy(float),
        capacities,
    )
    rows = []
    for group_index, group in enumerate(groups):
        actual = base_actual.reindex(common)[group]
        finite = actual.notna().to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": common[finite],
                    "group": group,
                    "official_target": actual.to_numpy(float)[finite],
                    "pred": np.clip(
                        redistributed[finite, group_index],
                        0.0,
                        GROUP_CAPACITY_KWH[group],
                    ),
                }
            )
        )
    for group in TARGET_COLS:
        if group in groups or group not in base_pred.columns:
            continue
        actual = base_actual.reindex(common)[group]
        prediction = base_pred.reindex(common)[group]
        finite = actual.notna().to_numpy() & prediction.notna().to_numpy()
        rows.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": common[finite],
                    "group": group,
                    "official_target": actual.to_numpy(float)[finite],
                    "pred": prediction.to_numpy(float)[finite],
                }
            )
        )
    out = pd.concat(rows, ignore_index=True)
    out["pred_year"] = pd.to_datetime(out["forecast_kst_dtm"]).dt.year
    out["target"] = target
    out["model"] = model
    out["alpha"] = alpha
    return out


def build_group_variants(
    total_predictions: pd.DataFrame,
    base_path: Path,
    alphas: list[float],
) -> pd.DataFrame:
    base_pred, base_actual = prepare_base_wide(base_path)
    parts = []
    base_long = pd.read_csv(base_path, encoding="utf-8-sig")
    base_long["forecast_kst_dtm"] = pd.to_datetime(base_long["forecast_kst_dtm"])
    base_long["variant"] = "base"
    parts.append(base_long[["forecast_kst_dtm", "group", "official_target", "pred", "variant"]])

    direct_parts: dict[tuple[str, str, float], pd.DataFrame] = {}
    for model in sorted(total_predictions["model"].unique()):
        for alpha in alphas:
            for target in TARGET_SPECS:
                part = make_redistributed_variant(
                    total_predictions,
                    base_pred,
                    base_actual,
                    target,
                    model,
                    alpha,
                )
                direct_parts[(target, model, alpha)] = part
                named = part.copy()
                named["variant"] = f"{target}_{model}_a{alpha:.2f}"
                parts.append(named)

            s12 = direct_parts[("s12", model, alpha)]
            s123 = direct_parts[("s123", model, alpha)]
            hierarchy = pd.concat(
                [
                    s12.loc[s12["pred_year"].eq(2022)],
                    s123.loc[s123["pred_year"].isin([2023, 2024])],
                ],
                ignore_index=True,
            )
            hierarchy["variant"] = f"hier_{model}_a{alpha:.2f}"
            parts.append(hierarchy)
    return pd.concat(parts, ignore_index=True)


def score_total_predictions(
    total_predictions: pd.DataFrame,
    base_path: Path,
) -> pd.DataFrame:
    base_pred, base_actual = prepare_base_wide(base_path)
    rows = []
    for target, spec in TARGET_SPECS.items():
        groups = spec["groups"]
        capacity = float(spec["capacity"])
        for model in total_predictions["model"].unique():
            part = total_predictions.loc[
                total_predictions["target"].eq(target)
                & total_predictions["model"].eq(model)
            ]
            score, nmae, ficr = total_metric(part["actual"], part["pred"], capacity)
            rows.append(
                {
                    "target": target,
                    "model": model,
                    "score": score,
                    "nmae": nmae,
                    "ficr": ficr,
                    "n_rows": len(part),
                }
            )
        common = base_pred.dropna(subset=groups).index.intersection(
            base_actual.dropna(subset=groups).index
        )
        actual = base_actual.loc[common, groups].sum(axis=1)
        prediction = base_pred.loc[common, groups].sum(axis=1)
        score, nmae, ficr = total_metric(actual, prediction, capacity)
        rows.append(
            {
                "target": target,
                "model": "base_sum",
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n_rows": len(common),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} h={args.hidden_size} layers={args.num_layers} "
        f"tree_estimators={args.tree_estimators}",
        flush=True,
    )
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels["s12"] = labels[["kpx_group_1", "kpx_group_2"]].sum(
        axis=1, min_count=2
    )
    labels["s123"] = labels[TARGET_COLS].sum(axis=1, min_count=3)
    weather = build_seqnn_weather(ldaps, gfs, TARGET_COLS[0])
    feature_cols = [
        column
        for column in weather.columns
        if column not in ["forecast_kst_dtm", "data_available_kst_dtm"]
    ]
    blocks = make_issue_blocks(
        weather,
        labels[["kst_dtm", "s12", "s123"]],
        feature_cols,
        target_cols=["s12", "s123"],
    )
    print(
        f"issues={len(blocks.years)} features={len(feature_cols)} "
        f"years={pd.Series(blocks.years).value_counts().sort_index().to_dict()}",
        flush=True,
    )

    prediction_parts = []
    training_rows = []
    target_items = list(TARGET_SPECS.items())
    if args.smoke_test:
        target_items = target_items[:1]
    for target_index, (target, spec) in enumerate(target_items):
        capacity = float(spec["capacity"])
        valid_years = [
            year
            for year in [2022, 2023, 2024]
            if np.isfinite(blocks.targets[blocks.years == year, :, target_index]).sum()
            >= 200
        ]
        if args.smoke_test:
            valid_years = valid_years[:1]
        for pred_year in valid_years:
            val_mask = blocks.years == pred_year
            times = blocks.forecast_times[val_mask].reshape(-1)
            actual = blocks.targets[val_mask, :, target_index].reshape(-1)
            tcn_prediction, tcn_stats = train_tcn_total_fold(
                blocks,
                target_index,
                pred_year,
                capacity,
                args,
                device,
                args.seed + target_index * 1000 + pred_year,
            )
            tree_prediction, tree_stats = train_tree_total_fold(
                blocks,
                target_index,
                pred_year,
                capacity,
                args,
                args.seed + target_index * 1000 + pred_year,
            )
            for model, prediction, stats in [
                ("tcn", tcn_prediction, tcn_stats),
                ("tree", tree_prediction, tree_stats),
            ]:
                flat_prediction = prediction.reshape(-1)
                finite = np.isfinite(actual)
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "forecast_kst_dtm": times[finite],
                            "pred_year": pred_year,
                            "target": target,
                            "model": model,
                            "actual": actual[finite],
                            "pred": flat_prediction[finite],
                        }
                    )
                )
                training_rows.append(
                    {
                        "target": target,
                        "model": model,
                        "pred_year": pred_year,
                        **stats,
                    }
                )
                print(
                    f"{target} {model} pred_year={pred_year}: "
                    f"score={stats['score']:.6f}",
                    flush=True,
                )

    total_predictions = add_mean_total_predictions(
        pd.concat(prediction_parts, ignore_index=True)
    )
    prefix = args.results_dir / args.stem
    total_predictions.to_csv(
        f"{prefix}_total_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(training_rows).to_csv(
        f"{prefix}_training.csv", index=False, encoding="utf-8-sig"
    )
    if args.smoke_test or not args.base_predictions.exists():
        print("smoke/direct-total training complete; reconstruction skipped", flush=True)
        return

    total_scores = score_total_predictions(total_predictions, args.base_predictions)
    total_scores.to_csv(
        f"{prefix}_total_scores.csv", index=False, encoding="utf-8-sig"
    )
    group_predictions = build_group_variants(
        total_predictions,
        args.base_predictions,
        [float(value) for value in parse_csv(args.alphas)],
    )
    group_predictions.to_csv(
        f"{prefix}_group_predictions.csv", index=False, encoding="utf-8-sig"
    )
    summary, group_scores = pooled_oof_summary(group_predictions)
    summary = summary.sort_values("mean_score", ascending=False).reset_index(drop=True)
    summary.to_csv(f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    group_scores.to_csv(
        f"{prefix}_group_scores.csv", index=False, encoding="utf-8-sig"
    )
    print("\n=== direct total scores ===", flush=True)
    print(total_scores.sort_values(["target", "score"], ascending=[True, False]).to_string(index=False), flush=True)
    print("\n=== redistributed official pooled OOF ===", flush=True)
    print(summary.head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
