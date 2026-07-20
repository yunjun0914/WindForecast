from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from experiments.evaluate_group_base_recovery_oof import (
    FoldData,
    build_site_static,
    fold_metric,
    make_loader,
    make_model,
    predict_tcn,
    prepare_folds,
    prepare_tcn_data,
    score_predictions,
    set_seed,
    tcn_prediction_frame,
    train_tcn_epoch,
)
from experiments.evaluate_group_unified_oof import build_group_tables, read_inputs
from utils.metrics import TARGET_COLS


LOSS_MODES = ("pure_band_ficr", "ficr_mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/group_tcn_band_loss_v1.json"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="group_tcn_band_loss_v1")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-outer-folds", type=int, default=0)
    return parser.parse_args()


def load_configs(path: Path, smoke_test: bool) -> tuple[dict, dict]:
    experiment = json.loads(path.read_text(encoding="utf-8"))
    base_path = Path(experiment["base_config_path"])
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if smoke_test:
        experiment["base_epoch"] = 1
        experiment["finetune"]["max_epochs"] = 2
        experiment["finetune"]["patience"] = 2
        base["tcn"]["batch_size"] = 128
    return experiment, base


def soft_band_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    mode: str,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode not in LOSS_MODES:
        raise ValueError(f"Unknown band loss mode: {mode}")
    losses = []
    nmaes = []
    ficrs = []
    for group_index in range(len(TARGET_COLS)):
        mask = observed[:, group_index]
        denominator = mask.sum()
        if float(denominator.detach().cpu()) <= 0.0:
            continue
        actual = target[:, group_index]
        error = torch.abs(prediction[:, group_index] - actual)
        nmae = (error * mask).sum() / denominator.clamp_min(1.0)
        unit_price = 4.0 - torch.sigmoid((error - 0.06) / gamma) - 3.0 * torch.sigmoid(
            (error - 0.08) / gamma
        )
        ficr = (actual * unit_price * mask).sum() / (
            actual * 4.0 * mask
        ).sum().clamp_min(1e-6)
        if mode == "pure_band_ficr":
            losses.append(1.0 - ficr)
        else:
            losses.append(0.5 * nmae + 0.5 * (1.0 - ficr))
        nmaes.append(nmae.detach())
        ficrs.append(ficr.detach())
    if not losses:
        zero = prediction.sum() * 0.0
        return zero, zero.detach(), zero.detach()
    return (
        torch.stack(losses).mean(),
        torch.stack(nmaes).mean(),
        torch.stack(ficrs).mean(),
    )


def train_base(
    fold: FoldData,
    experiment: dict,
    base_config: dict,
    device: torch.device,
    seed: int,
) -> tuple[object, object, dict[str, torch.Tensor], np.ndarray, np.ndarray, dict]:
    set_seed(seed)
    train_dataset, validation_dataset = prepare_tcn_data(fold, base_config)
    tcn_config = base_config["tcn"]
    loader = make_loader(
        train_dataset, int(tcn_config["batch_size"]), True, seed, device
    )
    model = make_model(len(fold.tcn_features), tcn_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcn_config["learning_rate"]),
        weight_decay=float(tcn_config["weight_decay"]),
    )
    train_loss = np.nan
    for epoch in range(1, int(experiment["base_epoch"]) + 1):
        train_loss = train_tcn_epoch(
            model, loader, optimizer, float(tcn_config["grad_clip"]), device
        )
        print(
            f"  base val={fold.pred_year} epoch={epoch:03d}/"
            f"{int(experiment['base_epoch']):03d} loss={train_loss:.5f}",
            flush=True,
        )
    train_prediction = predict_tcn(
        model, train_dataset, int(tcn_config["eval_batch_size"]), device
    )
    validation_prediction = predict_tcn(
        model, validation_dataset, int(tcn_config["eval_batch_size"]), device
    )
    score, group_rows = fold_metric(fold.validation, validation_prediction)
    state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    stats = {
        "stage": "weighted_l1_base",
        "pred_year": fold.pred_year,
        "base_epoch": int(experiment["base_epoch"]),
        "train_loss": train_loss,
        "score": score,
        "group_scores": ";".join(
            f"{row['group']}={row['score']:.6f}" for row in group_rows
        ),
    }
    del model, optimizer, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        train_dataset,
        validation_dataset,
        state,
        train_prediction,
        validation_prediction,
        stats,
    )


def train_finetune_epoch(
    model,
    loader,
    optimizer,
    base_prediction: np.ndarray,
    mode: str,
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    losses = []
    objectives = []
    anchors = []
    nmaes = []
    ficrs = []
    for features, target, _, observed, row_index in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        observed = observed.to(device, non_blocking=True)
        base = torch.from_numpy(base_prediction[row_index.numpy()]).to(
            device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        objective, nmae, ficr = soft_band_objective(
            prediction,
            target,
            observed,
            mode,
            float(config["gamma"]),
        )
        anchor = ((prediction - base).pow(2) * observed).sum() / observed.sum().clamp_min(
            1.0
        )
        loss = objective + float(config["anchor_weight"]) * anchor
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        objectives.append(float(objective.detach().cpu()))
        anchors.append(float(anchor.detach().cpu()))
        nmaes.append(float(nmae.cpu()))
        ficrs.append(float(ficr.cpu()))
    return {
        "train_loss": float(np.mean(losses)),
        "train_objective": float(np.mean(objectives)),
        "train_anchor": float(np.mean(anchors)),
        "train_soft_nmae": float(np.mean(nmaes)),
        "train_soft_ficr": float(np.mean(ficrs)),
    }


def discover_finetune_epoch(
    fold: FoldData,
    train_dataset,
    validation_dataset,
    base_state: dict[str, torch.Tensor],
    base_train_prediction: np.ndarray,
    mode: str,
    experiment: dict,
    base_config: dict,
    device: torch.device,
    seed: int,
) -> tuple[int, list[dict]]:
    set_seed(seed)
    config = experiment["finetune"]
    model = make_model(len(fold.tcn_features), base_config["tcn"], device)
    model.load_state_dict(base_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loader = make_loader(
        train_dataset,
        int(base_config["tcn"]["batch_size"]),
        True,
        seed,
        device,
    )
    best_score = -np.inf
    best_epoch = 0
    bad_epochs = 0
    history = []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        train_stats = train_finetune_epoch(
            model,
            loader,
            optimizer,
            base_train_prediction,
            mode,
            config,
            device,
        )
        prediction = predict_tcn(
            model,
            validation_dataset,
            int(base_config["tcn"]["eval_batch_size"]),
            device,
        )
        score, group_rows = fold_metric(fold.validation, prediction)
        history.append(
            {
                "stage": "finetune_discovery",
                "variant": mode,
                "pred_year": fold.pred_year,
                "epoch": epoch,
                "score": score,
                "group_scores": ";".join(
                    f"{row['group']}={row['score']:.6f}" for row in group_rows
                ),
                **train_stats,
            }
        )
        if score > best_score + float(config["min_delta"]):
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"  {mode} val={fold.pred_year} epoch={epoch:03d} "
            f"score={score:.6f} best={best_score:.6f}@{best_epoch}",
            flush=True,
        )
        if bad_epochs >= int(config["patience"]):
            break
    if best_epoch <= 0:
        raise RuntimeError(f"No fine-tune epoch selected for {mode}/{fold.pred_year}")
    del model, optimizer, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, history


def fit_fixed_finetune(
    fold: FoldData,
    train_dataset,
    validation_dataset,
    base_state: dict[str, torch.Tensor],
    base_train_prediction: np.ndarray,
    mode: str,
    fixed_epoch: int,
    experiment: dict,
    base_config: dict,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    config = experiment["finetune"]
    model = make_model(len(fold.tcn_features), base_config["tcn"], device)
    model.load_state_dict(base_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loader = make_loader(
        train_dataset,
        int(base_config["tcn"]["batch_size"]),
        True,
        seed,
        device,
    )
    train_stats = {}
    for epoch in range(1, fixed_epoch + 1):
        train_stats = train_finetune_epoch(
            model,
            loader,
            optimizer,
            base_train_prediction,
            mode,
            config,
            device,
        )
        print(
            f"  fixed {mode} val={fold.pred_year} "
            f"epoch={epoch:03d}/{fixed_epoch:03d}",
            flush=True,
        )
    prediction = predict_tcn(
        model,
        validation_dataset,
        int(base_config["tcn"]["eval_batch_size"]),
        device,
    )
    score, group_rows = fold_metric(fold.validation, prediction)
    stats = {
        "stage": "finetune_fixed",
        "variant": mode,
        "pred_year": fold.pred_year,
        "fixed_epoch": fixed_epoch,
        "score": score,
        "group_scores": ";".join(
            f"{row['group']}={row['score']:.6f}" for row in group_rows
        ),
        **train_stats,
    }
    del model, optimizer, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, stats


def build_all_folds(
    max_outer_folds: int,
) -> list[FoldData]:
    ldaps, gfs, labels, scada_by_group = read_inputs()
    print("build deduplicated site inputs", flush=True)
    site_static = build_site_static(ldaps, gfs)
    tables_by_group = {}
    for group in TARGET_COLS:
        print(f"build group inputs: {group}", flush=True)
        tables_by_group[group] = build_group_tables(
            ldaps,
            gfs,
            labels,
            scada_by_group[group],
            group,
            include_residual=False,
        )
    folds = prepare_folds(labels, tables_by_group, site_static, max_outer_folds)
    del ldaps, gfs, labels, scada_by_group, site_static, tables_by_group
    gc.collect()
    return folds


def main() -> None:
    args = parse_args()
    experiment, base_config = load_configs(args.config, args.smoke_test)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} smoke={args.smoke_test} "
        f"base_epoch={experiment['base_epoch']}",
        flush=True,
    )
    folds = build_all_folds(args.max_outer_folds)

    epoch_rows = []
    discovered = {mode: [] for mode in LOSS_MODES}
    for fold_index, fold in enumerate(folds):
        print(f"\n=== discovery val={fold.pred_year} ===", flush=True)
        artifacts = train_base(
            fold,
            experiment,
            base_config,
            device,
            int(experiment["seed"]) + fold_index * 1000,
        )
        train_dataset, validation_dataset, state, train_pred, _, base_stats = artifacts
        epoch_rows.append(base_stats)
        for mode in LOSS_MODES:
            best_epoch, history = discover_finetune_epoch(
                fold,
                train_dataset,
                validation_dataset,
                state,
                train_pred,
                mode,
                experiment,
                base_config,
                device,
                int(experiment["seed"]) + 100 + fold_index * 1000,
            )
            discovered[mode].append(best_epoch)
            epoch_rows.extend(history)
        del train_dataset, validation_dataset, state, train_pred
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fixed_epochs = {
        mode: int(np.median(values)) for mode, values in discovered.items()
    }
    print(f"\nfixed fine-tune epochs={fixed_epochs}", flush=True)

    oof_parts = []
    for fold_index, fold in enumerate(folds):
        print(f"\n=== fixed OOF val={fold.pred_year} ===", flush=True)
        artifacts = train_base(
            fold,
            experiment,
            base_config,
            device,
            int(experiment["seed"]) + 10000 + fold_index * 1000,
        )
        train_dataset, validation_dataset, state, train_pred, base_val_pred, base_stats = artifacts
        epoch_rows.append({**base_stats, "stage": "weighted_l1_base_fixed"})
        output = tcn_prediction_frame(fold, base_val_pred).rename(
            columns={"tcn_multihead": "weighted_l1_base"}
        )
        for mode in LOSS_MODES:
            prediction, stats = fit_fixed_finetune(
                fold,
                train_dataset,
                validation_dataset,
                state,
                train_pred,
                mode,
                fixed_epochs[mode],
                experiment,
                base_config,
                device,
                int(experiment["seed"]) + 10100 + fold_index * 1000,
            )
            epoch_rows.append(stats)
            frame = tcn_prediction_frame(fold, prediction).rename(
                columns={"tcn_multihead": mode}
            )
            output = output.merge(
                frame,
                on=["forecast_kst_dtm", "pred_year", "group", "actual"],
                how="inner",
                validate="one_to_one",
            )
            share = float(experiment["base_share"])
            output[f"base25_{mode}75"] = (
                share * output["weighted_l1_base"]
                + (1.0 - share) * output[mode]
            )
        oof_parts.append(output)
        del train_dataset, validation_dataset, state, train_pred, base_val_pred
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["group", "forecast_kst_dtm"]
    )
    prediction_cols = [
        column
        for column in oof.columns
        if column not in {"forecast_kst_dtm", "pred_year", "group", "actual"}
    ]
    summary, score_diagnostics = score_predictions(oof, prediction_cols)
    diagnostics = pd.concat(
        [
            pd.DataFrame(epoch_rows).assign(row_type="training"),
            score_diagnostics,
        ],
        ignore_index=True,
    )
    oof_path = args.results_dir / f"{args.stem}_oof.csv"
    summary_path = args.results_dir / f"{args.stem}_summary.csv"
    diagnostics_path = args.results_dir / f"{args.stem}_epoch_diagnostics.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    print("\n=== pooled official OOF ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"saved {oof_path}", flush=True)
    print(f"saved {summary_path}", flush=True)
    print(f"saved {diagnostics_path}", flush=True)


if __name__ == "__main__":
    main()
