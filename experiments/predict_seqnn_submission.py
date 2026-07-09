import argparse
import copy
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from models.seqnn import DLinearPowerRegressor, GRUPowerRegressor, TCNPowerRegressor
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr
from utils.seq_dataset import (
    YEARS,
    SequenceStandardScaler,
    build_group_table,
    build_seqnn_weather,
    make_sequences,
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


def build_model(args, input_size, window):
    if args.model == "gru":
        return GRUPowerRegressor(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    if args.model == "dlinear":
        return DLinearPowerRegressor(
            input_size=input_size,
            window=window,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )
    if args.model == "tcn":
        return TCNPowerRegressor(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
    raise ValueError(f"unknown model: {args.model}")


def predict_numpy(model, x, capacity, device, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(xb).detach().cpu().numpy())
    pred_norm = np.concatenate(preds) if preds else np.empty((0,), dtype=np.float32)
    return np.clip(pred_norm, 0.0, 1.0) * capacity


def train_predict_fold(train, valid, test_weather, feature_cols, group, train_years, valid_year, args, device):
    capacity = GROUP_CAPACITY_KWH[group]
    x_train, y_train, _ = make_sequences(train, feature_cols, window=args.window)
    x_valid, y_valid, _ = make_sequences(valid, feature_cols, window=args.window)
    x_test, _, time_test = make_sequences(test_weather, feature_cols, window=args.window)

    scaler = SequenceStandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)
    y_train_norm = np.clip(y_train / capacity, 0.0, 1.0).astype(np.float32)

    weights = sample_weight(y_train_norm, args.weight_policy)
    dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_norm, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = build_model(args, input_size=x_train.shape[-1], window=args.window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -np.inf
    best_epoch = -1
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
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
            losses.append(float(loss.detach().cpu()))

        pred_valid = predict_numpy(model, x_valid, capacity, device, args.eval_batch_size)
        score, nmae, ficr = score_one(y_valid, pred_valid, group)
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if args.verbose and (epoch == 1 or epoch % args.log_every == 0 or bad_epochs == 0):
            print(
                f"{group} train={','.join(map(str, train_years))} valid={valid_year} "
                f"epoch={epoch:03d} loss={np.mean(losses):.5f} "
                f"valid_score={score:.5f} nmae={nmae:.5f} ficr={ficr:.5f} "
                f"best={best_score:.5f}@{best_epoch}"
            )
        if bad_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    pred_test = predict_numpy(model, x_test, capacity, device, args.eval_batch_size)
    return pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(time_test), "pred": pred_test}), {
        "train_years": ",".join(map(str, train_years)),
        "valid_year": valid_year,
        "best_epoch": best_epoch,
        "valid_score": best_score,
        "n_train": len(y_train),
        "n_valid": len(y_valid),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--model", default="tcn", choices=["gru", "dlinear", "tcn"])
    parser.add_argument("--window", type=int, default=72)
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
    parser.add_argument("--stem", default="submission_seqnn_tcn_w72_v1")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fold-stats-output", default=None)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} model={args.model} window={args.window} stem={args.stem}")

    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    ldaps_test = pd.read_csv("data/test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv("data/test/gfs_test.csv", encoding="utf-8-sig")
    submission = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"])

    out = submission[["forecast_id", "forecast_kst_dtm"]].copy()
    stat_rows = []

    for group in parse_list(args.groups):
        print(f"\n=== build SeqNN train/test weather {group} ===")
        train_weather = build_seqnn_weather(ldaps_train, gfs_train, group)
        test_weather = build_seqnn_weather(ldaps_test, gfs_test, group)
        feature_cols = [col for col in train_weather.columns if col not in ["forecast_kst_dtm", "data_available_kst_dtm"]]
        test_weather = test_weather.reindex(columns=["forecast_kst_dtm", "data_available_kst_dtm", *feature_cols], fill_value=0)
        table = build_group_table(train_weather, labels, group)
        table["forecast_kst_dtm"] = pd.to_datetime(table["forecast_kst_dtm"])
        test_preds = []

        for train_years in combinations(YEARS, len(YEARS) - 1):
            train_years = list(train_years)
            valid_years = [year for year in YEARS if year not in train_years]
            valid_year = valid_years[0]
            train = table[table["year"].isin(train_years)].copy().reset_index(drop=True)
            valid = table[table["year"] == valid_year].copy().reset_index(drop=True)
            if len(train) < 1000 or len(valid) < 200:
                print(f"{group} train={train_years} valid={valid_year}: skip train={len(train)} valid={len(valid)}")
                continue
            print(
                f"{group} train={','.join(map(str, train_years))} valid={valid_year}: "
                f"train_rows={len(train)} valid_rows={len(valid)} features={len(feature_cols)}"
            )
            pred, stats = train_predict_fold(
                train,
                valid,
                test_weather,
                feature_cols,
                group,
                train_years,
                valid_year,
                args,
                device,
            )
            test_preds.append(pred)
            stat_rows.append({"group": group, "window": args.window, "model": args.model, **stats})

        if not test_preds:
            raise ValueError(f"no SeqNN test predictions generated for {group}")
        base_time = test_preds[0]["forecast_kst_dtm"].reset_index(drop=True)
        for idx, pred in enumerate(test_preds[1:], start=1):
            if not base_time.equals(pred["forecast_kst_dtm"].reset_index(drop=True)):
                raise ValueError(f"test time mismatch for {group} fold {idx}")
        stacked = np.vstack([pred["pred"].to_numpy(float) for pred in test_preds])
        group_pred = np.clip(stacked.mean(axis=0), 0.0, GROUP_CAPACITY_KWH[group])
        group_df = pd.DataFrame({"forecast_kst_dtm": base_time, group: group_pred})
        out = out.merge(group_df, on="forecast_kst_dtm", how="left")
        print(f"{group}: folds={len(test_preds)} min={group_pred.min():.2f} max={group_pred.max():.2f} mean={group_pred.mean():.2f}")

    groups = parse_list(args.groups)
    if out[groups].isna().any().any():
        missing = out[out[groups].isna().any(axis=1)].head()
        raise ValueError(f"missing SeqNN predictions:\n{missing}")
    out = out[["forecast_id", "forecast_kst_dtm", *groups]]
    output = Path(args.output) if args.output else RESULTS_DIR / f"{args.stem}.csv"
    fold_stats_output = Path(args.fold_stats_output) if args.fold_stats_output else RESULTS_DIR / f"{args.stem}_fold_stats.csv"
    out.to_csv(output, index=False, encoding="utf-8-sig")
    pd.DataFrame(stat_rows).to_csv(fold_stats_output, index=False, encoding="utf-8-sig")
    print(f"\nsaved {output}: {out.shape}")
    print(out[groups].agg(["min", "max", "mean"]).to_string())
    print(f"saved {fold_stats_output}")
    return out


if __name__ == "__main__":
    main()
