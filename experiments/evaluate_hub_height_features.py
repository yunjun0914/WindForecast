import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import _bootstrap  # noqa: F401
from predict_power_lgbm_best import clean_params, sample_weight
from predict_tree_compact_physics_v2 import build_all_meteo_compact_v2
from tune_power_lgbm_hyperparams import filter_scada_years, filter_weather_years, parse_list, score_one
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature_oof
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset
from utils.site_metadata import group_site_summary


RESULTS_DIR = Path("results")
YEARS = [2022, 2023, 2024]
HUB_HEIGHT_M = 117.0
EPS = 0.1


def _clip_alpha(alpha):
    return np.clip(alpha, -0.5, 1.0)


def _shear_alpha(high_speed, low_speed, high_m, low_m):
    ratio = (np.asarray(high_speed, dtype=float).clip(min=0.0) + EPS) / (
        np.asarray(low_speed, dtype=float).clip(min=0.0) + EPS
    )
    return _clip_alpha(np.log(ratio) / np.log(float(high_m) / float(low_m)))


def _project_speed(speed, alpha, from_height_m):
    return np.asarray(speed, dtype=float).clip(min=0.0) * (HUB_HEIGHT_M / float(from_height_m)) ** np.asarray(alpha, dtype=float)


def _safe_mean(cols):
    stacked = np.vstack([np.asarray(col, dtype=float) for col in cols])
    return np.nanmean(stacked, axis=0)


def _safe_spread(cols):
    stacked = np.vstack([np.asarray(col, dtype=float) for col in cols])
    return np.nanmax(stacked, axis=0) - np.nanmin(stacked, axis=0)


def _add_density_power(out, rho_col, speed_col):
    if rho_col in out.columns and speed_col in out.columns:
        out[f"{rho_col}_x_{speed_col}_cube"] = out[rho_col] * out[speed_col].clip(lower=0.0) ** 3


def add_hub_height_features(weather, group):
    out = weather.copy()
    site = group_site_summary()[group]
    axis_x, axis_y = site["axis_x"], site["axis_y"]

    new_cols = []

    if {"gfs_ws10_speed", "gfs_ws80_speed", "gfs_ws100_speed"}.issubset(out.columns):
        a_gfs_100_10 = _shear_alpha(out["gfs_ws100_speed"], out["gfs_ws10_speed"], 100.0, 10.0)
        a_gfs_100_80 = _shear_alpha(out["gfs_ws100_speed"], out["gfs_ws80_speed"], 100.0, 80.0)
        a_gfs_80_10 = _shear_alpha(out["gfs_ws80_speed"], out["gfs_ws10_speed"], 80.0, 10.0)

        out["phys_hub_gfs_alpha_100_10"] = a_gfs_100_10
        out["phys_hub_gfs_alpha_100_80"] = a_gfs_100_80
        out["phys_hub_gfs_alpha_80_10"] = a_gfs_80_10
        out["phys_hub_gfs117_from_100_10_speed"] = _project_speed(out["gfs_ws100_speed"], a_gfs_100_10, 100.0)
        out["phys_hub_gfs117_from_100_80_speed"] = _project_speed(out["gfs_ws100_speed"], a_gfs_100_80, 100.0)
        out["phys_hub_gfs117_from_80_10_speed"] = _project_speed(out["gfs_ws80_speed"], a_gfs_80_10, 80.0)
        out["phys_hub_gfs117_blend_speed"] = _safe_mean(
            [
                out["phys_hub_gfs117_from_100_10_speed"],
                out["phys_hub_gfs117_from_100_80_speed"],
                out["phys_hub_gfs117_from_80_10_speed"],
            ]
        )
        out["phys_hub_gfs117_spread"] = _safe_spread(
            [
                out["phys_hub_gfs117_from_100_10_speed"],
                out["phys_hub_gfs117_from_100_80_speed"],
                out["phys_hub_gfs117_from_80_10_speed"],
            ]
        )
        out["phys_hub_gfs117_minus_ws100"] = out["phys_hub_gfs117_blend_speed"] - out["gfs_ws100_speed"]
        out["phys_hub_gfs117_ratio_ws100"] = out["phys_hub_gfs117_blend_speed"] / (out["gfs_ws100_speed"].clip(lower=0.0) + EPS)
        new_cols.extend(
            [
                "phys_hub_gfs_alpha_100_10",
                "phys_hub_gfs_alpha_100_80",
                "phys_hub_gfs_alpha_80_10",
                "phys_hub_gfs117_from_100_10_speed",
                "phys_hub_gfs117_from_100_80_speed",
                "phys_hub_gfs117_from_80_10_speed",
                "phys_hub_gfs117_blend_speed",
                "phys_hub_gfs117_spread",
                "phys_hub_gfs117_minus_ws100",
                "phys_hub_gfs117_ratio_ws100",
            ]
        )

        u_col, v_col = "gfs_heightAboveGround_100_100u", "gfs_heightAboveGround_100_100v"
        if u_col in out.columns and v_col in out.columns:
            scale = out["phys_hub_gfs117_blend_speed"] / (out["gfs_ws100_speed"].clip(lower=0.0) + EPS)
            out["phys_hub_gfs117_u"] = out[u_col] * scale
            out["phys_hub_gfs117_v"] = out[v_col] * scale
            out["phys_hub_gfs117_axis_along"] = out["phys_hub_gfs117_u"] * axis_x + out["phys_hub_gfs117_v"] * axis_y
            out["phys_hub_gfs117_axis_cross"] = -out["phys_hub_gfs117_u"] * axis_y + out["phys_hub_gfs117_v"] * axis_x
            out["phys_hub_gfs117_southwest"] = -(out["phys_hub_gfs117_u"] + out["phys_hub_gfs117_v"]) / np.sqrt(2.0)
            new_cols.extend(
                [
                    "phys_hub_gfs117_u",
                    "phys_hub_gfs117_v",
                    "phys_hub_gfs117_axis_along",
                    "phys_hub_gfs117_axis_cross",
                    "phys_hub_gfs117_southwest",
                ]
            )

    if {"ldaps_ws10_speed", "ldaps_ws50_max_speed", "ldaps_ws50_min_speed"}.issubset(out.columns):
        ldaps_50_avg = 0.5 * (out["ldaps_ws50_max_speed"] + out["ldaps_ws50_min_speed"])
        a_ldaps_50avg_10 = _shear_alpha(ldaps_50_avg, out["ldaps_ws10_speed"], 50.0, 10.0)
        a_ldaps_50max_10 = _shear_alpha(out["ldaps_ws50_max_speed"], out["ldaps_ws10_speed"], 50.0, 10.0)
        a_ldaps_50min_10 = _shear_alpha(out["ldaps_ws50_min_speed"], out["ldaps_ws10_speed"], 50.0, 10.0)

        out["phys_hub_ldaps_alpha_50avg_10"] = a_ldaps_50avg_10
        out["phys_hub_ldaps_alpha_50max_10"] = a_ldaps_50max_10
        out["phys_hub_ldaps_alpha_50min_10"] = a_ldaps_50min_10
        out["phys_hub_ldaps117_from_10_alpha014_speed"] = out["ldaps_ws10_speed"].clip(lower=0.0) * (HUB_HEIGHT_M / 10.0) ** 0.14
        out["phys_hub_ldaps117_from_50avg_10_speed"] = _project_speed(ldaps_50_avg, a_ldaps_50avg_10, 50.0)
        out["phys_hub_ldaps117_from_50max_10_speed"] = _project_speed(out["ldaps_ws50_max_speed"], a_ldaps_50max_10, 50.0)
        out["phys_hub_ldaps117_from_50min_10_speed"] = _project_speed(out["ldaps_ws50_min_speed"], a_ldaps_50min_10, 50.0)
        out["phys_hub_ldaps117_blend_speed"] = _safe_mean(
            [
                out["phys_hub_ldaps117_from_10_alpha014_speed"],
                out["phys_hub_ldaps117_from_50avg_10_speed"],
            ]
        )
        out["phys_hub_ldaps117_spread"] = _safe_spread(
            [
                out["phys_hub_ldaps117_from_10_alpha014_speed"],
                out["phys_hub_ldaps117_from_50avg_10_speed"],
                out["phys_hub_ldaps117_from_50max_10_speed"],
                out["phys_hub_ldaps117_from_50min_10_speed"],
            ]
        )
        new_cols.extend(
            [
                "phys_hub_ldaps_alpha_50avg_10",
                "phys_hub_ldaps_alpha_50max_10",
                "phys_hub_ldaps_alpha_50min_10",
                "phys_hub_ldaps117_from_10_alpha014_speed",
                "phys_hub_ldaps117_from_50avg_10_speed",
                "phys_hub_ldaps117_from_50max_10_speed",
                "phys_hub_ldaps117_from_50min_10_speed",
                "phys_hub_ldaps117_blend_speed",
                "phys_hub_ldaps117_spread",
            ]
        )

    if {"phys_hub_gfs117_blend_speed", "phys_hub_ldaps117_blend_speed"}.issubset(out.columns):
        out["phys_hub_mix117_blend_speed"] = 0.5 * (
            out["phys_hub_gfs117_blend_speed"] + out["phys_hub_ldaps117_blend_speed"]
        )
        out["phys_hub_mix117_gfs_minus_ldaps"] = out["phys_hub_gfs117_blend_speed"] - out["phys_hub_ldaps117_blend_speed"]
        out["phys_hub_mix117_spread"] = _safe_spread(
            [out["phys_hub_gfs117_blend_speed"], out["phys_hub_ldaps117_blend_speed"]]
        )
        new_cols.extend(
            [
                "phys_hub_mix117_blend_speed",
                "phys_hub_mix117_gfs_minus_ldaps",
                "phys_hub_mix117_spread",
            ]
        )

    for rho_col in ["phys_gfs_air_density", "phys_ldaps_air_density"]:
        for speed_col in [
            "phys_hub_gfs117_blend_speed",
            "phys_hub_ldaps117_blend_speed",
            "phys_hub_mix117_blend_speed",
        ]:
            before = set(out.columns)
            _add_density_power(out, rho_col, speed_col)
            new_cols.extend([col for col in out.columns if col not in before])

    out = out.sort_values("forecast_kst_dtm").reset_index(drop=True)
    for speed_col in [
        "phys_hub_gfs117_blend_speed",
        "phys_hub_ldaps117_blend_speed",
        "phys_hub_mix117_blend_speed",
    ]:
        if speed_col in out.columns:
            out[f"{speed_col}_diff1"] = out[speed_col].diff(1)
            out[f"{speed_col}_diff3"] = out[speed_col].diff(3)
            new_cols.extend([f"{speed_col}_diff1", f"{speed_col}_diff3"])

    new_cols = [col for col in dict.fromkeys(new_cols) if col in out.columns]
    out.attrs["hub_height_feature_cols"] = new_cols
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


def load_data():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    return ldaps, gfs, labels, {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }


def feature_columns(weather):
    return [col for col in weather.columns if col not in TIME_KEY_COLS]


def score_predictions(scores):
    fold_means = (
        scores.groupby(["variant", "pred_year"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"))
    )
    summary = (
        fold_means.groupby("variant", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
        )
        .sort_values("mean_score", ascending=False)
    )
    group_summary = (
        scores.groupby(["variant", "group"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            n_features=("n_features", "max"),
            n_folds=("score", "count"),
        )
    )
    return summary, group_summary, fold_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--best-csv", default="results/power_lgbm_hyperparams_v2_l1_20_best.csv")
    parser.add_argument("--variants", default="baseline,hub_v1,hub_v1_curve")
    parser.add_argument("--stem", default="hub_height_features_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = parse_list(args.groups)
    variants = parse_list(args.variants)
    best = pd.read_csv(args.best_csv, encoding="utf-8-sig")
    ldaps, gfs, labels, scada_by_group = load_data()

    feature_cache = {}
    hub_cols = {}
    for group in groups:
        base = build_all_meteo_compact_v2(ldaps, gfs, group)
        hub = add_hub_height_features(base, group)
        feature_cache[(group, "baseline")] = base
        feature_cache[(group, "hub_v1")] = hub
        feature_cache[(group, "hub_v1_curve")] = hub
        hub_cols[group] = hub.attrs.get("hub_height_feature_cols", [])
        print(f"{group}: added hub features={len(hub_cols[group])}")

    score_rows = []
    pred_parts = []
    feature_rows = []
    for group in groups:
        row = best[best["group"] == group]
        if row.empty:
            continue
        best_row = row.iloc[0]
        params = clean_params(best_row)
        policy = {
            "min_output_ratio": float(best_row["min_output_ratio"]),
            "weight_policy": best_row["weight_policy"],
        }
        for pred_year in YEARS:
            train_years = [year for year in YEARS if year != pred_year]
            scada = filter_scada_years(scada_by_group[group], train_years)
            if len(scada) == 0:
                continue
            for variant in variants:
                base_weather = feature_cache[(group, variant)]
                train_weather_base = filter_weather_years(base_weather, train_years)
                val_weather_base = filter_weather_years(base_weather, [pred_year])

                train_weather, val_weather = add_power_curve_feature_oof(
                    train_weather_base,
                    val_weather_base,
                    scada,
                    group,
                    HUB_HEIGHT_PROXY_COL,
                    GROUP_N_TURBINES[group],
                    out_col="power_curve_est",
                )
                if variant == "hub_v1_curve" and "phys_hub_mix117_blend_speed" in train_weather.columns:
                    train_weather, val_weather = add_power_curve_feature_oof(
                        train_weather,
                        val_weather,
                        scada,
                        group,
                        "phys_hub_mix117_blend_speed",
                        GROUP_N_TURBINES[group],
                        out_col="power_curve_est_hub117",
                    )

                x_train_full, y_train = build_group_dataset(train_weather, labels, group)
                x_val_full, y_val = build_group_dataset(val_weather, labels, group)
                if len(x_train_full) < 1000 or len(x_val_full) < 200:
                    continue
                val_times = (
                    val_weather.merge(labels[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
                    .dropna(subset=[group])["forecast_kst_dtm"]
                    .reset_index(drop=True)
                )
                train_mask = y_train.to_numpy(float) >= GROUP_CAPACITY_KWH[group] * policy["min_output_ratio"]
                x_train = x_train_full.loc[train_mask].reset_index(drop=True)
                y_used = y_train.loc[train_mask].reset_index(drop=True)
                feature_cols = feature_columns(train_weather)
                x_train = x_train.reindex(columns=feature_cols, fill_value=0)
                x_val = x_val_full.reindex(columns=feature_cols, fill_value=0)

                model = LGBMRegressor(**params)
                weights = sample_weight(y_used, group, policy["weight_policy"])
                model.fit(x_train, y_used, sample_weight=weights)
                pred = np.clip(model.predict(x_val), 0.0, GROUP_CAPACITY_KWH[group])
                score, nmae, ficr = score_one(y_val, pred, group)
                score_rows.append(
                    {
                        "variant": variant,
                        "group": group,
                        "pred_year": pred_year,
                        "train_years": ",".join(map(str, train_years)),
                        "score": score,
                        "nmae": nmae,
                        "ficr": ficr,
                        "n_features": len(feature_cols),
                        "n_hub_features": len(hub_cols[group]) + (1 if variant == "hub_v1_curve" else 0),
                        "n_used": len(y_used),
                        "n_val": len(y_val),
                    }
                )
                pred_parts.append(
                    pd.DataFrame(
                        {
                            "forecast_kst_dtm": pd.to_datetime(val_times).to_numpy(),
                            "pred_year": pred_year,
                            "train_years": ",".join(map(str, train_years)),
                            "variant": variant,
                            "group": group,
                            "actual": y_val.to_numpy(float),
                            "pred": pred,
                        }
                    )
                )
                print(f"{group} {variant} pred_year={pred_year}: score={score:.5f} nmae={nmae:.5f} ficr={ficr:.5f} features={len(feature_cols)}")

        for col in hub_cols[group]:
            feature_rows.append({"group": group, "feature": col})

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    features = pd.DataFrame(feature_rows).drop_duplicates()
    summary, group_summary, fold_means = score_predictions(scores)

    scores.to_csv(RESULTS_DIR / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(RESULTS_DIR / f"{args.stem}_group_summary.csv", index=False, encoding="utf-8-sig")
    fold_means.to_csv(RESULTS_DIR / f"{args.stem}_fold_means.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")
    features.to_csv(RESULTS_DIR / f"{args.stem}_features.csv", index=False, encoding="utf-8-sig")

    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print("\n=== group summary ===")
    print(group_summary.sort_values(["group", "mean_score"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
