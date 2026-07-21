from __future__ import annotations

import argparse
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
)
from experiments.evaluate_group_unified_oof import build_group_tables, read_inputs
from utils.metrics import TARGET_COLS


LOSS_MODES = ("pure_band_ficr", "ficr_nmae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/group_tcn_band_loss_v1.json"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="group_tcn_direct_band_loss_v1")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-outer-folds", type=int, default=0)
    return parser.parse_args()


def load_configs(path: Path, smoke_test: bool) -> tuple[dict, dict]:
    experiment = json.loads(path.read_text(encoding="utf-8"))
    base = json.loads(Path(experiment["base_config_path"]).read_text(encoding="utf-8"))
    if smoke_test:
        base["tcn"]["max_epochs"] = 2
        base["tcn"]["patience"] = 2
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
        nmae_denominator = mask.sum()
        ficr_denominator = (target[:, group_index] * 4.0 * mask).sum()
        if (
            float(nmae_denominator.detach().cpu()) <= 0.0
            or float(ficr_denominator.detach().cpu()) <= 0.0
        ):
            continue
        actual = target[:, group_index]
        error = torch.abs(prediction[:, group_index] - actual)
        nmae = (error * mask).sum() / nmae_denominator.clamp_min(1.0)
        unit_price = 4.0 - torch.sigmoid((error - 0.06) / gamma) - 3.0 * torch.sigmoid(
            (error - 0.08) / gamma
        )
        ficr = (actual * unit_price * mask).sum() / ficr_denominator.clamp_min(
            1e-6
        )
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


def train_direct_epoch(
    model,
    loader,
    optimizer,
    mode: str,
    gamma: float,
    grad_clip: float,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    losses = []
    nmaes = []
    ficrs = []
    for features, target, _, observed, _ in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        observed = observed.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss, nmae, ficr = soft_band_objective(
            prediction,
            target,
            observed,
            mode,
            gamma,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        nmaes.append(float(nmae.cpu()))
        ficrs.append(float(ficr.cpu()))
    if not losses:
        raise RuntimeError("Direct TCN epoch had no observed targets")
    return {
        "train_loss": float(np.mean(losses)),
        "train_soft_nmae": float(np.mean(nmaes)),
        "train_soft_ficr": float(np.mean(ficrs)),
    }


def discover_epoch(
    fold: FoldData,
    mode: str,
    experiment: dict,
    base_config: dict,
    device: torch.device,
    seed: int,
) -> tuple[int, list[dict]]:
    set_seed(seed)
    train_dataset, validation_dataset = prepare_tcn_data(fold, base_config)
    config = base_config["tcn"]
    loader = make_loader(
        train_dataset, int(config["batch_size"]), True, seed, device
    )
    model = make_model(len(fold.tcn_features), config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    best_score = -np.inf
    best_epoch = 0
    bad_epochs = 0
    history = []
    for epoch in range(1, int(config["max_epochs"]) + 1):
        train_stats = train_direct_epoch(
            model,
            loader,
            optimizer,
            mode,
            float(experiment["gamma"]),
            float(config["grad_clip"]),
            device,
        )
        prediction = predict_tcn(
            model, validation_dataset, int(config["eval_batch_size"]), device
        )
        score, group_rows = fold_metric(fold.validation, prediction)
        history.append(
            {
                "stage": "direct_epoch_discovery",
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
            f"  direct {mode} val={fold.pred_year} epoch={epoch:03d} "
            f"score={score:.6f} best={best_score:.6f}@{best_epoch}",
            flush=True,
        )
        if bad_epochs >= int(config["patience"]):
            break
    if best_epoch <= 0:
        raise RuntimeError(f"No direct epoch selected for {mode}/{fold.pred_year}")
    del model, optimizer, loader, train_dataset, validation_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_epoch, history


def fit_fixed_epoch(
    fold: FoldData,
    mode: str,
    fixed_epoch: int,
    experiment: dict,
    base_config: dict,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    train_dataset, validation_dataset = prepare_tcn_data(fold, base_config)
    config = base_config["tcn"]
    loader = make_loader(
        train_dataset, int(config["batch_size"]), True, seed, device
    )
    model = make_model(len(fold.tcn_features), config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_stats = {}
    for epoch in range(1, fixed_epoch + 1):
        train_stats = train_direct_epoch(
            model,
            loader,
            optimizer,
            mode,
            float(experiment["gamma"]),
            float(config["grad_clip"]),
            device,
        )
        print(
            f"  fixed direct {mode} val={fold.pred_year} "
            f"epoch={epoch:03d}/{fixed_epoch:03d}",
            flush=True,
        )
    prediction = predict_tcn(
        model, validation_dataset, int(config["eval_batch_size"]), device
    )
    score, group_rows = fold_metric(fold.validation, prediction)
    stats = {
        "stage": "direct_fixed_epoch",
        "variant": mode,
        "pred_year": fold.pred_year,
        "fixed_epoch": fixed_epoch,
        "score": score,
        "group_scores": ";".join(
            f"{row['group']}={row['score']:.6f}" for row in group_rows
        ),
        **train_stats,
    }
    del model, optimizer, loader, train_dataset, validation_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, stats


def build_all_folds(max_outer_folds: int) -> list[FoldData]:
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
    print(f"device={device} smoke={args.smoke_test} training=direct", flush=True)
    folds = build_all_folds(args.max_outer_folds)

    epoch_rows = []
    discovered = {mode: [] for mode in LOSS_MODES}
    for mode in LOSS_MODES:
        for fold_index, fold in enumerate(folds):
            print(f"\n=== discovery {mode} val={fold.pred_year} ===", flush=True)
            best_epoch, history = discover_epoch(
                fold,
                mode,
                experiment,
                base_config,
                device,
                int(experiment["seed"]) + fold_index * 1000,
            )
            discovered[mode].append(best_epoch)
            epoch_rows.extend(history)

    fixed_epochs = {
        mode: int(np.median(values)) for mode, values in discovered.items()
    }
    print(f"\nfixed direct epochs={fixed_epochs}", flush=True)

    oof_parts = []
    for fold_index, fold in enumerate(folds):
        print(f"\n=== fixed direct OOF val={fold.pred_year} ===", flush=True)
        output = None
        for mode in LOSS_MODES:
            prediction, stats = fit_fixed_epoch(
                fold,
                mode,
                fixed_epochs[mode],
                experiment,
                base_config,
                device,
                int(experiment["seed"]) + 10000 + fold_index * 1000,
            )
            epoch_rows.append(stats)
            frame = tcn_prediction_frame(fold, prediction).rename(
                columns={"tcn_multihead": mode}
            )
            if output is None:
                output = frame
            else:
                output = output.merge(
                    frame,
                    on=["forecast_kst_dtm", "pred_year", "group", "actual"],
                    how="inner",
                    validate="one_to_one",
                )
        if output is None:
            raise RuntimeError(f"No direct predictions for val={fold.pred_year}")
        oof_parts.append(output)

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["group", "forecast_kst_dtm"]
    )
    summary, score_diagnostics = score_predictions(oof, list(LOSS_MODES))
    diagnostics = pd.concat(
        [pd.DataFrame(epoch_rows).assign(row_type="training"), score_diagnostics],
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
