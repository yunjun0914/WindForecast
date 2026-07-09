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
from utils.power_curve import GROUP_N_TURBINES, GROUP_TURBINE_PREFIXES, add_power_curve_feature_oof, fit_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset


RESULTS_DIR = Path("results")
YEARS = [2022, 2023, 2024]
N_SECTORS = 8
POWER_CURVE_CAP_TOL = 1.02


DIRECTION_SPECS = [
    ("gfs850", "gfs_isobaricInhPa_850_u", "gfs_isobaricInhPa_850_v", "gfs_ws850_speed"),
    ("gfs100", "gfs_heightAboveGround_100_100u", "gfs_heightAboveGround_100_100v", "gfs_ws100_speed"),
    ("gfs10", "gfs_heightAboveGround_10_10u", "gfs_heightAboveGround_10_10v", "gfs_ws10_speed"),
    ("ldaps50max", "ldaps_heightAboveGround_50_50MUmax", "ldaps_heightAboveGround_50_50MVmax", "ldaps_ws50_max_speed"),
    ("ldaps10", "ldaps_heightAboveGround_10_10u", "ldaps_heightAboveGround_10_10v", "ldaps_ws10_speed"),
]


CURVE_CONFIG = {
    "dir_curve_ldaps50max": ("ldaps50max", "ldaps_ws50_max_speed"),
    "dir_curve_gfs850": ("gfs850", "gfs_ws850_speed"),
}


def _met_direction_deg(u, v):
    """Meteorological 'from' direction, degrees clockwise from north."""
    return (np.degrees(np.arctan2(-np.asarray(u, dtype=float), -np.asarray(v, dtype=float))) + 360.0) % 360.0


def _sector_from_deg(deg, n_sectors=N_SECTORS):
    width = 360.0 / n_sectors
    return np.floor(((np.asarray(deg, dtype=float) + width / 2.0) % 360.0) / width).astype(int)


def add_direction_compact_features(weather):
    out = weather.copy()
    new_cols = []
    for name, u_col, v_col, speed_col in DIRECTION_SPECS:
        if u_col not in out.columns or v_col not in out.columns or speed_col not in out.columns:
            continue
        speed = out[speed_col].astype(float).clip(lower=0.0)
        direction = _met_direction_deg(out[u_col], out[v_col])
        angle = np.radians(direction)
        sector = _sector_from_deg(direction)

        out[f"phys_dir_{name}_sin"] = np.sin(angle)
        out[f"phys_dir_{name}_cos"] = np.cos(angle)
        out[f"phys_dir_{name}_sector8"] = sector
        out[f"phys_dir_{name}_abs_east_west"] = np.abs(out[u_col]) / (speed + 0.1)
        out[f"phys_dir_{name}_abs_north_south"] = np.abs(out[v_col]) / (speed + 0.1)
        new_cols.extend(
            [
                f"phys_dir_{name}_sin",
                f"phys_dir_{name}_cos",
                f"phys_dir_{name}_sector8",
                f"phys_dir_{name}_abs_east_west",
                f"phys_dir_{name}_abs_north_south",
            ]
        )

        if name in {"gfs850", "gfs100", "ldaps50max"}:
            for sector_id in range(N_SECTORS):
                flag_col = f"phys_dir_{name}_sector{sector_id}"
                ws_col = f"phys_dir_{name}_sector{sector_id}_ws"
                cube_col = f"phys_dir_{name}_sector{sector_id}_ws_cube"
                flag = (sector == sector_id).astype(float)
                out[flag_col] = flag
                out[ws_col] = flag * speed
                out[cube_col] = flag * speed**3
                new_cols.extend([flag_col, ws_col, cube_col])

    if {"phys_dir_gfs850_sin", "phys_dir_gfs100_sin", "phys_dir_gfs850_cos", "phys_dir_gfs100_cos"}.issubset(out.columns):
        diff = np.arctan2(
            out["phys_dir_gfs850_sin"] * out["phys_dir_gfs100_cos"] - out["phys_dir_gfs850_cos"] * out["phys_dir_gfs100_sin"],
            out["phys_dir_gfs850_cos"] * out["phys_dir_gfs100_cos"] + out["phys_dir_gfs850_sin"] * out["phys_dir_gfs100_sin"],
        )
        out["phys_dir_gfs850_gfs100_turn_abs"] = np.abs(diff)
        new_cols.append("phys_dir_gfs850_gfs100_turn_abs")

    out.attrs["direction_feature_cols"] = [col for col in dict.fromkeys(new_cols) if col in out.columns]
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


def _single_turbine_10m_capacity(group):
    return GROUP_CAPACITY_KWH[group] / GROUP_N_TURBINES[group] / 6.0


def fit_directional_power_curves(scada_df, group, n_sectors=N_SECTORS, clean=True):
    ws_parts, wd_parts, power_parts = [], [], []
    for prefix in GROUP_TURBINE_PREFIXES[group]:
        ws_col = f"{prefix}_ws"
        wd_col = f"{prefix}_wd"
        power_col = f"{prefix}_power_kw10m"
        if ws_col not in scada_df.columns or wd_col not in scada_df.columns or power_col not in scada_df.columns:
            continue
        ws_parts.append(scada_df[ws_col].to_numpy(float))
        wd_parts.append(scada_df[wd_col].to_numpy(float))
        power_parts.append(scada_df[power_col].to_numpy(float))

    ws = np.concatenate(ws_parts)
    wd = np.concatenate(wd_parts)
    power = np.concatenate(power_parts)
    valid = ~(np.isnan(ws) | np.isnan(wd) | np.isnan(power))
    if clean:
        cap_10m = _single_turbine_10m_capacity(group) * POWER_CURVE_CAP_TOL
        valid &= (ws >= 0) & (ws <= 35) & (power >= 0) & (power <= cap_10m)
    ws = ws[valid]
    wd = wd[valid]
    power = power[valid]
    if len(ws) == 0:
        raise ValueError(f"no valid SCADA rows for directional power curve: {group}")

    global_curve = fit_power_curve(ws, power)
    sectors = _sector_from_deg(wd, n_sectors=n_sectors)
    curves = []
    for sector_id in range(n_sectors):
        mask = sectors == sector_id
        if int(mask.sum()) < 200:
            curves.append(global_curve)
        else:
            curves.append(fit_power_curve(ws[mask], power[mask]))
    return global_curve, curves


def add_directional_power_curve_feature(df, speed_col, sector_col, curves, n_turbines, out_col):
    out = df.copy()
    speed = out[speed_col].to_numpy(float)
    sector = out[sector_col].to_numpy(int)
    pred = np.zeros(len(out), dtype=float)
    global_curve, sector_curves = curves
    for sector_id, curve in enumerate(sector_curves):
        mask = sector == sector_id
        if mask.any():
            pred[mask] = curve(speed[mask])
    missing = ~np.isfinite(pred)
    if missing.any():
        pred[missing] = global_curve(speed[missing])
    out[out_col] = pred * n_turbines
    return out


def add_directional_power_curve_oof(train_df, pred_df, scada_df, group, speed_col, sector_col, out_col, n_splits=5):
    curves = fit_directional_power_curves(scada_df, group)
    train_out = add_directional_power_curve_feature(train_df, speed_col, sector_col, curves, GROUP_N_TURBINES[group], out_col)
    pred_out = add_directional_power_curve_feature(pred_df, speed_col, sector_col, curves, GROUP_N_TURBINES[group], out_col)

    if "forecast_kst_dtm" not in train_out.columns or len(train_out) == 0:
        return train_out, pred_out

    times = pd.to_datetime(train_out["forecast_kst_dtm"]).dt.floor("h")
    unique_times = np.array(sorted(times.dropna().unique()))
    n_folds = min(int(n_splits), len(unique_times))
    if n_folds < 2:
        return train_out, pred_out

    scada = scada_df.copy()
    scada["hour"] = pd.to_datetime(scada["kst_dtm"]).dt.floor("h")
    for fold_times in np.array_split(unique_times, n_folds):
        fold_mask = times.isin(fold_times)
        if not bool(fold_mask.any()):
            continue
        fit_scada = scada.loc[~scada["hour"].isin(fold_times)].drop(columns=["hour"])
        if len(fit_scada) == 0:
            continue
        fold_curves = fit_directional_power_curves(fit_scada, group)
        train_out.loc[fold_mask, out_col] = add_directional_power_curve_feature(
            train_out.loc[fold_mask],
            speed_col,
            sector_col,
            fold_curves,
            GROUP_N_TURBINES[group],
            out_col,
        )[out_col].to_numpy()
    return train_out, pred_out


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
    parser.add_argument("--variants", default="baseline,dir_compact,dir_curve_ldaps50max,dir_curve_gfs850")
    parser.add_argument("--stem", default="direction_features_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = parse_list(args.groups)
    variants = parse_list(args.variants)
    best = pd.read_csv(args.best_csv, encoding="utf-8-sig")
    ldaps, gfs, labels, scada_by_group = load_data()

    feature_cache = {}
    direction_cols = {}
    for group in groups:
        base = build_all_meteo_compact_v2(ldaps, gfs, group)
        direction = add_direction_compact_features(base)
        feature_cache[(group, "baseline")] = base
        for variant in ["dir_compact", "dir_curve_ldaps50max", "dir_curve_gfs850"]:
            feature_cache[(group, variant)] = direction
        direction_cols[group] = direction.attrs.get("direction_feature_cols", [])
        print(f"{group}: added direction features={len(direction_cols[group])}")

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

                if variant in CURVE_CONFIG:
                    level_name, speed_col = CURVE_CONFIG[variant]
                    sector_col = f"phys_dir_{level_name}_sector8"
                    if speed_col in train_weather.columns and sector_col in train_weather.columns:
                        train_weather, val_weather = add_directional_power_curve_oof(
                            train_weather,
                            val_weather,
                            scada,
                            group,
                            speed_col,
                            sector_col,
                            out_col=f"power_curve_est_{level_name}_sector",
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
                        "n_direction_features": len(direction_cols[group]),
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

        for col in direction_cols[group]:
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
