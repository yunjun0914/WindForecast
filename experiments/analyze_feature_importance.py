import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import _bootstrap  # noqa: F401
from predict_power_lgbm_best import clean_params, sample_weight
from predict_tree_compact_physics_v2 import build_all_meteo_compact_v2
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS
from utils.pinn_effective_pipeline import (
    EXT_TARGETS,
    _make_lgbm_teacher,
    build_extended_pinn_weather,
    build_extended_scada_targets,
    extended_feature_cols,
)
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature_oof
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset


RESULTS_DIR = Path("results")


def _rank_importance(df):
    out = df.copy()
    out["gain_norm"] = out.groupby(["mode", "group"])["gain"].transform(lambda s: s / s.sum() if s.sum() else 0)
    out["split_norm"] = out.groupby(["mode", "group"])["split"].transform(lambda s: s / s.sum() if s.sum() else 0)
    out["gain_rank"] = out.groupby(["mode", "group"])["gain"].rank(method="first", ascending=False).astype(int)
    out["split_rank"] = out.groupby(["mode", "group"])["split"].rank(method="first", ascending=False).astype(int)
    return out.sort_values(["mode", "group", "gain_rank"]).reset_index(drop=True)


def load_train_data():
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
    return ldaps, gfs, labels, scada_by_group


def tree_importance(groups, best_csv):
    ldaps, gfs, labels, scada_by_group = load_train_data()
    best = pd.read_csv(best_csv, encoding="utf-8-sig")
    rows = []

    for group in groups:
        best_row = best[best["group"] == group]
        if best_row.empty:
            print(f"{group}: skip, no best params")
            continue
        best_row = best_row.iloc[0]
        print(f"fit tuned TREE LGBM importance {group}")

        weather = build_all_meteo_compact_v2(ldaps, gfs, group)
        train_weather, _ = add_power_curve_feature_oof(
            weather,
            weather,
            scada_by_group[group],
            group,
            HUB_HEIGHT_PROXY_COL,
            GROUP_N_TURBINES[group],
        )
        x_train, y_train = build_group_dataset(train_weather, labels, group)
        feature_cols = [col for col in train_weather.columns if col not in TIME_KEY_COLS]
        x_train = x_train.reindex(columns=feature_cols, fill_value=0)

        min_output_ratio = float(best_row["min_output_ratio"])
        mask = y_train.to_numpy(float) >= GROUP_CAPACITY_KWH[group] * min_output_ratio
        x_train = x_train.loc[mask].reset_index(drop=True)
        y_used = y_train.loc[mask].reset_index(drop=True)

        model = LGBMRegressor(**clean_params(best_row))
        model.fit(x_train, y_used, sample_weight=sample_weight(y_used, group, best_row["weight_policy"]))
        booster = model.booster_
        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
        for feature, gain_value, split_value in zip(feature_cols, gain, split):
            rows.append(
                {
                    "mode": "tree_lgbm_best",
                    "group": group,
                    "target": "power",
                    "feature": feature,
                    "gain": float(gain_value),
                    "split": int(split_value),
                    "n_train": len(y_used),
                    "n_features": len(feature_cols),
                }
            )

    return _rank_importance(pd.DataFrame(rows)) if rows else pd.DataFrame()


def teacher_importance(groups):
    ldaps, gfs, _, scada_by_group = load_train_data()
    weather = build_extended_pinn_weather(ldaps, gfs)
    feature_cols = extended_feature_cols(weather)
    rows = []

    for group in groups:
        print(f"fit PINN LGBM teacher importance {group}")
        targets = build_extended_scada_targets(scada_by_group[group], group)
        df = weather.merge(targets, on="forecast_kst_dtm", how="inner").dropna()
        model = _make_lgbm_teacher(seed=42)
        model.fit(df[feature_cols], df[EXT_TARGETS])

        for target, estimator in zip(EXT_TARGETS, model.estimators_):
            booster = estimator.booster_
            gain = booster.feature_importance(importance_type="gain")
            split = booster.feature_importance(importance_type="split")
            for feature, gain_value, split_value in zip(feature_cols, gain, split):
                rows.append(
                    {
                        "mode": "teacher_lgbm_time_oof_fullfit",
                        "group": group,
                        "target": target,
                        "feature": feature,
                        "gain": float(gain_value),
                        "split": int(split_value),
                        "n_train": len(df),
                        "n_features": len(feature_cols),
                    }
                )

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, raw

    target_ranked = raw.copy()
    target_ranked["gain_norm_target"] = target_ranked.groupby(["group", "target"])["gain"].transform(
        lambda s: s / s.sum() if s.sum() else 0
    )
    target_ranked["gain_rank_target"] = target_ranked.groupby(["group", "target"])["gain"].rank(
        method="first", ascending=False
    ).astype(int)
    target_ranked = target_ranked.sort_values(["group", "target", "gain_rank_target"]).reset_index(drop=True)

    aggregated = (
        raw.groupby(["mode", "group", "feature"], as_index=False)
        .agg(
            gain=("gain", "sum"),
            split=("split", "sum"),
            mean_gain=("gain", "mean"),
            targets_used=("target", "nunique"),
            n_train=("n_train", "max"),
            n_features=("n_features", "max"),
        )
    )
    aggregated["target"] = "all_ext_targets"
    aggregated = _rank_importance(aggregated)
    return target_ranked, aggregated


def print_top(title, df, n=20):
    if df.empty:
        return
    print(f"\n=== {title} ===")
    cols = [col for col in ["mode", "group", "target", "feature", "gain_norm", "gain", "split", "gain_rank"] if col in df.columns]
    for group, part in df.groupby("group"):
        print(f"\n[{group}]")
        print(part.head(n)[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tree", "teacher", "both"], default="both")
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--best-csv", default="results/power_lgbm_hyperparams_v2_l1_20_best.csv")
    parser.add_argument("--stem", default="feature_importance_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = [part.strip() for part in args.groups.split(",") if part.strip()]

    if args.mode in {"tree", "both"}:
        tree_df = tree_importance(groups, args.best_csv)
        tree_path = RESULTS_DIR / f"{args.stem}_tree_lgbm_best.csv"
        tree_df.to_csv(tree_path, index=False, encoding="utf-8-sig")
        print_top("TREE LGBM best gain top", tree_df)
        print(f"saved {tree_path}")

    if args.mode in {"teacher", "both"}:
        teacher_target_df, teacher_agg_df = teacher_importance(groups)
        teacher_target_path = RESULTS_DIR / f"{args.stem}_teacher_lgbm_by_target.csv"
        teacher_agg_path = RESULTS_DIR / f"{args.stem}_teacher_lgbm_aggregated.csv"
        teacher_target_df.to_csv(teacher_target_path, index=False, encoding="utf-8-sig")
        teacher_agg_df.to_csv(teacher_agg_path, index=False, encoding="utf-8-sig")
        print_top("PINN teacher LGBM aggregated gain top", teacher_agg_df)
        print(f"saved {teacher_target_path}")
        print(f"saved {teacher_agg_path}")


if __name__ == "__main__":
    main()
