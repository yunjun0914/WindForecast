import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from models.seqnn import DLinearPowerRegressor, GRUPowerRegressor, TCNPowerRegressor
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr
from utils.seq_dataset import (
    build_group_table,
    build_seqnn_weather,
    make_sequences,
    split_year_fold,
    SequenceStandardScaler,
    YEARS,
)


RESULTS_DIR = Path("results")


def parse_list(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_one(y_true, pred, group):
    capacity = GROUP_CAPACITY_KWH[group]
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, capacity)
    nmae, ficr = group_nmae_ficr(y_true, pred, capacity)
    return 0.5 * (1.0 - nmae) + 0.5 * ficr, nmae, ficr


def sample_weight(y_norm, policy):
    y_norm = np.clip(np.asarray(y_norm, dtype=np.float32), 0.0, 1.0)
    if policy == "none":
        return np.ones_like(y_norm, dtype=np.float32)
    if policy == "actual_sqrt":
        return (0.5 + np.sqrt(y_norm)).astype(np.float32)
    if policy == "metric_x2":
        return (1.0 + 2.0 * (y_norm >= 0.10)).astype(np.float32)
    raise ValueError(f"unknown weight_policy: {policy}")


def predict_numpy(model, x, capacity, device, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            pred_norm = model(xb).detach().cpu().numpy()
            preds.append(pred_norm)
    pred_norm = np.concatenate(preds) if preds else np.empty((0,), dtype=np.float32)
    return np.clip(pred_norm, 0.0, 1.0) * capacity


def train_one_fold(x_train, y_train, x_val, y_val, group, args, device):
    capacity = GROUP_CAPACITY_KWH[group]
    scaler = SequenceStandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    y_train_norm = np.clip(y_train / capacity, 0.0, 1.0).astype(np.float32)

    weights = sample_weight(y_train_norm, args.weight_policy)
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_norm, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    if args.model == "gru":
        model = GRUPowerRegressor(
            input_size=x_train.shape[-1],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
    elif args.model == "dlinear":
        model = DLinearPowerRegressor(
            input_size=x_train.shape[-1],
            window=x_train.shape[1],
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        ).to(device)
    elif args.model == "tcn":
        model = TCNPowerRegressor(
            input_size=x_train.shape[-1],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(f"unknown model: {args.model}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -np.inf
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = (torch.abs(pred - yb) * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        pred_val = predict_numpy(model, x_val, capacity, device, args.eval_batch_size)
        score, nmae, ficr = score_one(y_val, pred_val, group)
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if args.verbose and (epoch == 1 or epoch % args.log_every == 0 or bad_epochs == 0):
            print(
                f"epoch={epoch:03d} train_loss={np.mean(train_losses):.5f} "
                f"val_score={score:.5f} nmae={nmae:.5f} ficr={ficr:.5f} "
                f"best={best_score:.5f}@{best_epoch}"
            )

        if bad_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    pred_val = predict_numpy(model, x_val, capacity, device, args.eval_batch_size)
    score, nmae, ficr = score_one(y_val, pred_val, group)
    return pred_val, {"best_epoch": best_epoch, "score": score, "nmae": nmae, "ficr": ficr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--model", default="gru", choices=["gru", "dlinear", "tcn"])
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-policy", default="actual_sqrt", choices=["none", "actual_sqrt", "metric_x2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stem", default="seqnn_short_gru_w24_v1")
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model} window={args.window} stem={args.stem}")

    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])

    score_rows = []
    pred_rows = []
    fold_count = 0

    for group in parse_list(args.groups):
        print(f"\n=== build SeqNN weather {group} ===")
        weather = build_seqnn_weather(ldaps, gfs, group)
        feature_cols = [col for col in weather.columns if col not in ["forecast_kst_dtm", "data_available_kst_dtm"]]
        print(f"{group}: features={len(feature_cols)}")
        table = build_group_table(weather, labels, group)

        for pred_year in YEARS:
            train, val, train_years = split_year_fold(table, pred_year)
            if len(train) < 1000 or len(val) < 200:
                print(f"{group} pred_year={pred_year}: skip train={len(train)} val={len(val)}")
                continue
            if args.max_folds and fold_count >= args.max_folds:
                break

            x_train, y_train, _ = make_sequences(train, feature_cols, window=args.window)
            x_val, y_val, time_val = make_sequences(val, feature_cols, window=args.window)
            print(
                f"\n=== train {group} pred_year={pred_year} train_years={','.join(map(str, train_years))} "
                f"x_train={x_train.shape} x_val={x_val.shape} ==="
            )
            pred, stats = train_one_fold(x_train, y_train, x_val, y_val, group, args, device)
            score, nmae, ficr = score_one(y_val, pred, group)

            train_years_text = ",".join(map(str, train_years))
            score_rows.append(
                {
                    "pred_year": pred_year,
                    "train_years": train_years_text,
                    "model_family": "seqnn",
                    "model_name": args.stem,
                    "group": group,
                    "score": score,
                    "nmae": nmae,
                    "ficr": ficr,
                    "n_rows": len(y_val),
                    "best_epoch": stats["best_epoch"],
                    "window": args.window,
                    "n_features": len(feature_cols),
                }
            )
            pred_rows.append(
                pd.DataFrame(
                    {
                        "forecast_kst_dtm": pd.to_datetime(time_val).to_numpy(),
                        "pred_year": pred_year,
                        "train_years": train_years_text,
                        "model_family": "seqnn",
                        "model_name": args.stem,
                        "group": group,
                        "actual": y_val.astype(float),
                        "pred": pred.astype(float),
                        "is_clipped": True,
                    }
                )
            )
            print(
                f"{group} pred_year={pred_year}: score={score:.5f}, nmae={nmae:.5f}, "
                f"ficr={ficr:.5f}, best_epoch={stats['best_epoch']}"
            )
            fold_count += 1

        if args.max_folds and fold_count >= args.max_folds:
            break

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    if not scores.empty:
        means = (
            scores.groupby(["pred_year", "model_family", "model_name"], as_index=False)
            .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"), n_rows=("n_rows", "sum"))
        )
        means["train_years"] = ""
        means["group"] = "fold_mean"
        means["best_epoch"] = np.nan
        means["window"] = args.window
        means["n_features"] = scores["n_features"].max()
        scores = pd.concat([scores, means[scores.columns]], ignore_index=True)
        summary = (
            scores[scores["group"] == "fold_mean"]
            .groupby(["model_family", "model_name"], as_index=False)
            .agg(
                mean_score=("score", "mean"),
                mean_nmae=("nmae", "mean"),
                mean_ficr=("ficr", "mean"),
                worst_fold=("score", "min"),
                std_score=("score", "std"),
                n_folds=("score", "count"),
                n_features=("n_features", "max"),
            )
        )
    else:
        summary = pd.DataFrame()

    oof_path = RESULTS_DIR / f"oof_{args.stem}.csv"
    scores_path = RESULTS_DIR / f"scores_{args.stem}.csv"
    summary_path = RESULTS_DIR / f"summary_{args.stem}.csv"
    predictions.to_csv(oof_path, index=False, encoding="utf-8-sig")
    scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print(f"saved {oof_path}")
    print(f"saved {scores_path}")
    print(f"saved {summary_path}")
    return scores, predictions, summary


if __name__ == "__main__":
    main()
