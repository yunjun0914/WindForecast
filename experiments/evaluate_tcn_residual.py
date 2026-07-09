import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from tune_power_lgbm_hyperparams import parse_list, prepare_fold_cache
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr


RESULTS_DIR = Path("results")
DEFAULT_ALIGNED = "results/pinn_lgbmteacher_powerlgbm_v2_l1_blend_aligned_predictions.csv"
DEFAULT_IMPORTANCE = "results/feature_importance_v1_tree_lgbm_best.csv"
YEARS = [2022, 2023, 2024]
KEYS = ["forecast_kst_dtm", "pred_year", "group"]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_base(path, tree_weight):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["forecast_kst_dtm"] = pd.to_datetime(df["forecast_kst_dtm"])
    df["capacity"] = df["group"].map(GROUP_CAPACITY_KWH).astype(float)
    df["base_pred"] = (1.0 - tree_weight) * df["pinn_pred"] + tree_weight * df["tree_pred"]
    df["base_pred"] = df["base_pred"].clip(lower=0.0, upper=df["capacity"])
    df["target_ratio"] = df["actual"].to_numpy(float) / df["capacity"].to_numpy(float)
    df["base_ratio"] = df["base_pred"].to_numpy(float) / df["capacity"].to_numpy(float)
    df["pinn_ratio"] = df["pinn_pred"].to_numpy(float) / df["capacity"].to_numpy(float)
    df["tree_ratio"] = df["tree_pred"].to_numpy(float) / df["capacity"].to_numpy(float)
    df["tree_minus_pinn_ratio"] = df["tree_ratio"] - df["pinn_ratio"]
    df["residual_ratio"] = df["target_ratio"] - df["base_ratio"]
    return df


def select_importance_features(path, top_n):
    if top_n <= 0 or not Path(path).exists():
        return []
    imp = pd.read_csv(path, encoding="utf-8-sig")
    agg = (
        imp.groupby("feature", as_index=False)
        .agg(gain=("gain", "sum"), split=("split", "sum"))
        .sort_values(["gain", "split"], ascending=False)
    )
    return agg["feature"].head(top_n).tolist()


def build_oof_weather_features(groups):
    cache = prepare_fold_cache(groups)
    parts = []
    for group in groups:
        for fold in cache[group]:
            x = fold["x_val"].copy()
            x["forecast_kst_dtm"] = pd.to_datetime(fold["time_val"]).to_numpy()
            x["pred_year"] = int(fold["pred_year"])
            x["group"] = group
            parts.append(x)
    if not parts:
        raise RuntimeError("no OOF weather features were built")
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(KEYS)


def add_model_features(df):
    out = df.copy()
    dt = pd.to_datetime(out["forecast_kst_dtm"])
    hour = 2.0 * np.pi * dt.dt.hour.to_numpy(float) / 24.0
    doy = 2.0 * np.pi * dt.dt.dayofyear.to_numpy(float) / 365.25
    out["tcn_sin_hod"] = np.sin(hour)
    out["tcn_cos_hod"] = np.cos(hour)
    out["tcn_sin_doy"] = np.sin(doy)
    out["tcn_cos_doy"] = np.cos(doy)
    for group in TARGET_COLS:
        out[f"group_is_{group}"] = (out["group"] == group).astype(float)
    return out


def build_table(args):
    groups = parse_list(args.groups)
    base = load_base(args.aligned_csv, args.tree_weight)
    base = base[base["group"].isin(groups)].copy()
    weather = build_oof_weather_features(groups)
    top_features = select_importance_features(args.importance_csv, args.top_weather_features)
    top_features = [col for col in top_features if col in weather.columns]
    keep_weather = KEYS + top_features
    table = base.merge(weather[keep_weather], on=KEYS, how="left")
    table = add_model_features(table)

    model_features = [
        "base_ratio",
        "pinn_ratio",
        "tree_ratio",
        "tree_minus_pinn_ratio",
        "tcn_sin_hod",
        "tcn_cos_hod",
        "tcn_sin_doy",
        "tcn_cos_doy",
    ]
    model_features.extend([f"group_is_{group}" for group in TARGET_COLS])
    feature_cols = model_features + top_features
    for col in feature_cols:
        table[col] = pd.to_numeric(table[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        table[col] = table[col].fillna(table[col].median())
        table[col] = table[col].fillna(0.0)

    table = table.sort_values(["group", "pred_year", "forecast_kst_dtm"]).reset_index(drop=True)
    return table, feature_cols


def make_sequences(df, feature_cols, window):
    features = df[feature_cols].to_numpy(np.float32)
    n_features = features.shape[1]
    x_seq = np.zeros((len(df), window, n_features), dtype=np.float32)
    for (_, _), idx in df.groupby(["group", "pred_year"]).groups.items():
        idx = np.asarray(sorted(idx), dtype=int)
        arr = features[idx]
        for local_i, global_i in enumerate(idx):
            start = max(0, local_i - window + 1)
            seq = arr[start : local_i + 1]
            if len(seq) < window:
                pad = np.repeat(seq[:1], window - len(seq), axis=0)
                seq = np.vstack([pad, seq])
            x_seq[global_i] = seq
    return x_seq


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.left_padding, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.downsample(x)


class TCNResidual(nn.Module):
    def __init__(self, n_features, hidden, levels, kernel_size, dropout):
        super().__init__()
        blocks = []
        in_channels = n_features
        for level in range(levels):
            out_channels = hidden
            blocks.append(TemporalBlock(in_channels, out_channels, kernel_size, 2**level, dropout))
            in_channels = out_channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.tcn(x)[:, :, -1]
        return self.head(h).squeeze(-1)


def standardize(train_x, all_x):
    mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0, keepdims=True)
    std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((all_x - mean) / std).astype(np.float32)


def train_one_fold(x_all, y_all, train_idx, val_idx, args, device):
    x_scaled = standardize(x_all[train_idx], x_all)
    y_train = y_all[train_idx].astype(np.float32)
    if args.target_clip > 0:
        y_train = np.clip(y_train, -args.target_clip, args.target_clip)

    train_ds = TensorDataset(torch.from_numpy(x_scaled[train_idx]), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = TCNResidual(
        n_features=x_all.shape[-1],
        hidden=args.hidden,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=args.huber_beta)

    model.train()
    for epoch in range(args.epochs):
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"  epoch={epoch + 1:02d}/{args.epochs} train_loss={np.mean(losses):.6f}")

    model.eval()
    preds = np.zeros(len(val_idx), dtype=np.float32)
    val_tensor = torch.from_numpy(x_scaled[val_idx])
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=args.batch_size, shuffle=False)
    offset = 0
    with torch.no_grad():
        for (xb,) in val_loader:
            pred = model(xb.to(device)).detach().cpu().numpy().astype(np.float32)
            preds[offset : offset + len(pred)] = pred
            offset += len(pred)
    if args.pred_clip > 0:
        preds = np.clip(preds, -args.pred_clip, args.pred_clip)
    return preds


def score_prediction(df, pred_col, variant, residual_weight, delta):
    rows = []
    for pred_year, fold in df.groupby("pred_year"):
        group_rows = []
        for group, part in fold.groupby("group"):
            metric = part[part["actual"] >= part["capacity"] * 0.10]
            if len(metric) == 0:
                continue
            pred = np.clip(
                metric[pred_col].to_numpy(float) + delta * GROUP_CAPACITY_KWH[group],
                0.0,
                GROUP_CAPACITY_KWH[group],
            )
            nmae, ficr = group_nmae_ficr(metric["actual"], pred, GROUP_CAPACITY_KWH[group])
            score = 0.5 * (1.0 - nmae) + 0.5 * ficr
            row = {
                "variant": variant,
                "residual_weight": residual_weight,
                "delta": delta,
                "pred_year": int(pred_year),
                "group": group,
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n": len(metric),
            }
            rows.append(row)
            group_rows.append(row)
        rows.append(
            {
                "variant": variant,
                "residual_weight": residual_weight,
                "delta": delta,
                "pred_year": int(pred_year),
                "group": "fold_mean",
                "score": float(np.mean([row["score"] for row in group_rows])),
                "nmae": float(np.mean([row["nmae"] for row in group_rows])),
                "ficr": float(np.mean([row["ficr"] for row in group_rows])),
                "n": int(sum(row["n"] for row in group_rows)),
            }
        )
    return rows


def summarize(scores):
    fold = scores[scores["group"].eq("fold_mean")]
    return (
        fold.groupby(["variant", "residual_weight", "delta"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
        )
        .sort_values(["mean_score", "mean_nmae"], ascending=[False, True])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-csv", default=DEFAULT_ALIGNED)
    parser.add_argument("--importance-csv", default=DEFAULT_IMPORTANCE)
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--tree-weight", type=float, default=0.6)
    parser.add_argument("--top-weather-features", type=int, default=48)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.02)
    parser.add_argument("--target-clip", type=float, default=0.08)
    parser.add_argument("--pred-clip", type=float, default=0.04)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--residual-weights", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--deltas", default="0")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stem", default="tcn_residual_v1_w60")
    args = parser.parse_args()

    set_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    table, feature_cols = build_table(args)
    x_all = make_sequences(table, feature_cols, args.window)
    y_all = table["residual_ratio"].to_numpy(np.float32)
    table["tcn_residual_ratio"] = 0.0

    for pred_year in YEARS:
        val_mask = table["pred_year"].eq(pred_year).to_numpy()
        train_mask = ~val_mask
        if val_mask.sum() == 0 or train_mask.sum() < 1000:
            continue
        train_idx = np.flatnonzero(train_mask)
        val_idx = np.flatnonzero(val_mask)
        print(f"\n=== TCN residual fold pred_year={pred_year} train={len(train_idx)} val={len(val_idx)} ===")
        pred_resid = train_one_fold(x_all, y_all, train_idx, val_idx, args, device)
        table.loc[val_idx, "tcn_residual_ratio"] = pred_resid

    score_rows = []
    pred_parts = []
    residual_weights = parse_float_list(args.residual_weights)
    deltas = parse_float_list(args.deltas)
    for residual_weight in residual_weights:
        variant = f"tcn_residual_w{residual_weight:.3f}"
        pred_col = f"pred_{variant}"
        table[pred_col] = (
            table["base_pred"].to_numpy(float)
            + residual_weight * table["tcn_residual_ratio"].to_numpy(float) * table["capacity"].to_numpy(float)
        )
        table[pred_col] = table[pred_col].clip(lower=0.0, upper=table["capacity"])
        for delta in deltas:
            score_rows.extend(score_prediction(table, pred_col, variant, residual_weight, delta))
        pred_parts.append(
            table[KEYS + ["actual", "capacity", "base_pred", "tcn_residual_ratio", pred_col]]
            .rename(columns={pred_col: "pred"})
            .assign(variant=variant)
        )

    scores = pd.DataFrame(score_rows)
    summary = summarize(scores)
    predictions = pd.concat(pred_parts, ignore_index=True)
    feature_info = pd.DataFrame({"feature": feature_cols})

    scores.to_csv(RESULTS_DIR / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")
    feature_info.to_csv(RESULTS_DIR / f"{args.stem}_features.csv", index=False, encoding="utf-8-sig")

    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print(f"\nfeatures={len(feature_cols)} window={args.window} rows={len(table)}")


if __name__ == "__main__":
    main()
