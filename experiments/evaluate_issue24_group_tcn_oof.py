from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from models.issue_block_tcn import IssueBlockTCN
from utils.issue_block_dataset import IssueBlockData, make_issue_blocks
from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
)
from utils.seq_dataset import SequenceStandardScaler, build_seqnn_weather


RESULTS_DIR = Path("results")
VARIANTS = {
    "independent_causal": ("independent", False),
    "independent_full": ("independent", True),
    "shared_causal": ("shared", False),
    "shared_full": ("shared", True),
}


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default=",".join(VARIANTS))
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
    parser.add_argument(
        "--weight-policy",
        default="actual_sqrt",
        choices=["none", "actual_sqrt", "metric_x2"],
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--stem", default="issue24_group_tcn_oof_v1")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_targets(targets: np.ndarray) -> np.ndarray:
    capacities = np.asarray(
        [GROUP_CAPACITY_KWH[group] for group in TARGET_COLS], dtype=np.float32
    )
    return targets / capacities.reshape(1, 1, -1)


def sample_weights(targets_norm: np.ndarray, policy: str) -> np.ndarray:
    clipped = np.clip(np.nan_to_num(targets_norm, nan=0.0), 0.0, 1.0)
    if policy == "none":
        return np.ones_like(clipped, dtype=np.float32)
    if policy == "actual_sqrt":
        return (0.5 + np.sqrt(clipped)).astype(np.float32)
    if policy == "metric_x2":
        return (1.0 + 2.0 * (clipped >= 0.10)).astype(np.float32)
    raise ValueError(f"Unknown weight policy: {policy}")


def build_model(
    input_size: int,
    output_size: int,
    full_context: bool,
    args: argparse.Namespace,
) -> IssueBlockTCN:
    return IssueBlockTCN(
        input_size=input_size,
        output_size=output_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        full_context=full_context,
    )


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
        return np.empty((0, 24, len(model.heads)), dtype=np.float32)
    return np.clip(np.concatenate(parts), 0.0, 1.0).astype(np.float32)


def group_score(actual: np.ndarray, forecast: np.ndarray, group: str) -> tuple[float, float, float]:
    finite = np.isfinite(actual) & np.isfinite(forecast)
    if not finite.any():
        return np.nan, np.nan, np.nan
    nmae, ficr = group_nmae_ficr(
        actual[finite], forecast[finite], GROUP_CAPACITY_KWH[group]
    )
    return 0.5 * (1.0 - nmae) + 0.5 * ficr, nmae, ficr


def validation_score(
    actual: np.ndarray,
    forecast_norm: np.ndarray,
    group_indices: list[int],
) -> float:
    scores = []
    for output_index, group_index in enumerate(group_indices):
        group = TARGET_COLS[group_index]
        forecast = forecast_norm[..., output_index] * GROUP_CAPACITY_KWH[group]
        score, _, _ = group_score(actual[..., group_index], forecast, group)
        if np.isfinite(score):
            scores.append(score)
    if not scores:
        raise ValueError("Validation fold has no scored group")
    return float(np.mean(scores))


def equal_group_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    observed: torch.Tensor,
    group_loss_scales: torch.Tensor,
) -> torch.Tensor:
    group_losses = []
    for group_index in range(prediction.shape[-1]):
        mask = observed[..., group_index]
        weighted_mask = weight[..., group_index] * mask
        denominator = weighted_mask.sum()
        if denominator.item() > 0:
            error = torch.abs(prediction[..., group_index] - target[..., group_index])
            group_loss = (error * weighted_mask).sum() / denominator
            group_losses.append(group_loss * group_loss_scales[group_index])
    if not group_losses:
        raise ValueError("Training batch has no observed target")
    return torch.stack(group_losses).sum() / prediction.shape[-1]


def train_fold(
    blocks: IssueBlockData,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    group_indices: list[int],
    full_context: bool,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    set_seed(seed)
    train_features = blocks.features[train_mask]
    train_targets = normalized_targets(blocks.targets[train_mask])[..., group_indices]
    train_has_target = np.isfinite(train_targets).any(axis=(1, 2))
    train_features = train_features[train_has_target]
    train_targets = train_targets[train_has_target]
    if not train_has_target.any():
        raise ValueError(f"Training fold has no targets for group indices {group_indices}")

    scaler = SequenceStandardScaler()
    x_train = scaler.fit_transform(train_features)
    x_val = scaler.transform(blocks.features[val_mask])
    y_train = train_targets
    y_val = blocks.targets[val_mask]
    observed = np.isfinite(y_train).astype(np.float32)
    weights = sample_weights(y_train, args.weight_policy)
    y_train = np.nan_to_num(y_train, nan=0.0).astype(np.float32)
    observed_counts = observed.sum(axis=(0, 1))
    group_loss_scales = observed_counts.max() / observed_counts

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
    model = build_model(
        input_size=x_train.shape[-1],
        output_size=len(group_indices),
        full_context=full_context,
        args=args,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    group_loss_scales_tensor = torch.from_numpy(group_loss_scales.astype(np.float32)).to(
        device
    )

    best_score = -np.inf
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb, wb, mb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = equal_group_l1(
                prediction,
                yb,
                wb,
                mb,
                group_loss_scales_tensor,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_prediction = predict_normalized(
            model, x_val, device=device, batch_size=args.eval_batch_size
        )
        score = validation_score(y_val, val_prediction, group_indices)
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if args.verbose and (
            epoch == 1 or epoch % args.log_every == 0 or bad_epochs == 0
        ):
            print(
                f"  epoch={epoch:03d} loss={np.mean(losses):.6f} "
                f"val_score={score:.6f} best={best_score:.6f}@{best_epoch}",
                flush=True,
            )
        if bad_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("No valid issue-block checkpoint was selected")
    model.load_state_dict(best_state)
    val_prediction = predict_normalized(
        model, x_val, device=device, batch_size=args.eval_batch_size
    )
    return val_prediction, {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "n_train_issues": int(train_has_target.sum()),
        "n_val_issues": int(val_mask.sum()),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "group_loss_scales": ",".join(f"{value:.3f}" for value in group_loss_scales),
    }


def append_group_predictions(
    prediction_rows: list[pd.DataFrame],
    score_rows: list[dict],
    blocks: IssueBlockData,
    val_mask: np.ndarray,
    prediction_norm: np.ndarray,
    group_indices: list[int],
    variant: str,
    pred_year: int,
) -> None:
    times = blocks.forecast_times[val_mask]
    actual = blocks.targets[val_mask]
    for output_index, group_index in enumerate(group_indices):
        group = TARGET_COLS[group_index]
        group_actual = actual[..., group_index]
        group_prediction = prediction_norm[..., output_index] * GROUP_CAPACITY_KWH[group]
        finite = np.isfinite(group_actual)
        if not finite.any():
            continue
        score, nmae, ficr = group_score(
            group_actual[finite], group_prediction[finite], group
        )
        score_rows.append(
            {
                "variant": variant,
                "group": group,
                "pred_year": pred_year,
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n_rows": int(finite.sum()),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": times[finite],
                    "variant": variant,
                    "group": group,
                    "pred_year": pred_year,
                    "actual": group_actual[finite],
                    "pred": group_prediction[finite],
                }
            )
        )


def main() -> None:
    args = parse_args()
    variants = parse_csv(args.variants)
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    pred_years = [int(value) for value in parse_csv(args.years)]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} variants={variants} years={pred_years} "
        f"h={args.hidden_size} layers={args.num_layers}",
        flush=True,
    )

    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    weather = build_seqnn_weather(ldaps, gfs, TARGET_COLS[0])
    feature_cols = [
        col
        for col in weather.columns
        if col not in ["forecast_kst_dtm", "data_available_kst_dtm"]
    ]
    blocks = make_issue_blocks(weather, labels, feature_cols)
    print(
        f"issues={len(blocks.years)} features={len(feature_cols)} "
        f"year_counts={pd.Series(blocks.years).value_counts().sort_index().to_dict()}",
        flush=True,
    )

    prediction_rows = []
    score_rows = []
    training_rows = []
    for variant_index, variant in enumerate(variants):
        training_mode, full_context = VARIANTS[variant]
        print(f"\n=== {variant} ===", flush=True)
        for pred_year in pred_years:
            train_mask = blocks.years != pred_year
            val_mask = blocks.years == pred_year
            if not train_mask.any() or not val_mask.any():
                print(f"skip pred_year={pred_year}: no train or validation issues", flush=True)
                continue

            if training_mode == "shared":
                group_indices = list(range(len(TARGET_COLS)))
                seed = args.seed + variant_index * 1000 + pred_year
                prediction, stats = train_fold(
                    blocks,
                    train_mask,
                    val_mask,
                    group_indices,
                    full_context,
                    args,
                    device,
                    seed,
                )
                append_group_predictions(
                    prediction_rows,
                    score_rows,
                    blocks,
                    val_mask,
                    prediction,
                    group_indices,
                    variant,
                    pred_year,
                )
                training_rows.append(
                    {
                        "variant": variant,
                        "group": "shared",
                        "pred_year": pred_year,
                        **stats,
                    }
                )
                print(
                    f"{variant} pred_year={pred_year}: "
                    f"score={stats['best_score']:.6f} epoch={stats['best_epoch']}",
                    flush=True,
                )
            else:
                for group_index, group in enumerate(TARGET_COLS):
                    if not np.isfinite(blocks.targets[val_mask, :, group_index]).any():
                        continue
                    group_indices = [group_index]
                    seed = (
                        args.seed
                        + variant_index * 1000
                        + pred_year
                        + group_index * 100
                    )
                    prediction, stats = train_fold(
                        blocks,
                        train_mask,
                        val_mask,
                        group_indices,
                        full_context,
                        args,
                        device,
                        seed,
                    )
                    append_group_predictions(
                        prediction_rows,
                        score_rows,
                        blocks,
                        val_mask,
                        prediction,
                        group_indices,
                        variant,
                        pred_year,
                    )
                    training_rows.append(
                        {
                            "variant": variant,
                            "group": group,
                            "pred_year": pred_year,
                            **stats,
                        }
                    )
                    print(
                        f"{variant} {group} pred_year={pred_year}: "
                        f"score={stats['best_score']:.6f} epoch={stats['best_epoch']}",
                        flush=True,
                    )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    scores = pd.DataFrame(score_rows)
    summary, pooled_group_scores = pooled_oof_summary(
        predictions,
        actual_col="actual",
        forecast_col="pred",
    )
    fold_means = (
        scores.groupby(["variant", "pred_year"], as_index=False)
        .agg(score=("score", "mean"))
    )
    fold_diagnostics = (
        fold_means.groupby("variant", as_index=False)
        .agg(
            worst_fold=("score", "min"),
            std_fold_score=("score", lambda values: values.std(ddof=0)),
            n_folds=("score", "count"),
        )
    )
    summary = summary.merge(fold_diagnostics, on="variant", how="left")
    summary["n_features"] = len(feature_cols)
    summary["issue_hours"] = 24
    summary = summary.sort_values("mean_score", ascending=False).reset_index(drop=True)

    prefix = RESULTS_DIR / args.stem
    predictions.to_csv(f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(f"{prefix}_scores.csv", index=False, encoding="utf-8-sig")
    pooled_group_scores.to_csv(
        f"{prefix}_pooled_group_scores.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(training_rows).to_csv(
        f"{prefix}_training.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== pooled official OOF summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
