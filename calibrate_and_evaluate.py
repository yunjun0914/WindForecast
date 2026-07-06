import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_weather_features

MODELS = {
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=42, n_jobs=-1),
}
VAL_START = "2024-01-01 01:00:00"
CV_SPLITTER = KFold(n_splits=3, shuffle=True, random_state=42)

RESULTS_PATH = "results/calibration_comparison.csv"


def time_split(weather_df, labels_df, group):
    merged = weather_df.merge(
        labels_df[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner"
    )
    merged = merged.dropna(subset=[group]).reset_index(drop=True)
    is_val = merged["forecast_kst_dtm"] >= VAL_START
    feature_cols = [c for c in weather_df.columns if c not in TIME_KEY_COLS]
    X_train, X_val = merged.loc[~is_val, feature_cols], merged.loc[is_val, feature_cols]
    y_train, y_val = merged.loc[~is_val, group], merged.loc[is_val, group]
    return X_train, X_val, y_train, y_val


def ensemble_predict(models, X_train, y_train, X_pred):
    preds = []
    for model in models.values():
        model.fit(X_train, y_train)
        preds.append(model.predict(X_pred))
    return sum(preds) / len(preds)


def main():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather = build_weather_features(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {"kpx_group_1": scada_vestas, "kpx_group_2": scada_vestas, "kpx_group_3": scada_unison}

    # first pass: gather each group's OOF (train-period) and raw val (2024) predictions
    per_group = {}
    pooled_oof_pred_pct, pooled_oof_actual_pct = [], []

    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group])
        X_train, X_val, y_train, y_val = time_split(group_weather, labels, group)

        # calibration curve fit on out-of-fold predictions within the train period only
        # (2022-2023) -- never touches the 2024 holdout, so evaluating on 2024 stays honest
        oof_preds = np.zeros(len(y_train))
        for model in MODELS.values():
            oof_preds += cross_val_predict(clone(model), X_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        oof_preds /= len(MODELS)

        raw_val_pred = ensemble_predict(MODELS, X_train, y_train, X_val)

        per_group[group] = {
            "capacity": capacity,
            "oof_preds": oof_preds,
            "y_train": y_train.to_numpy(),
            "raw_val_pred": raw_val_pred,
            "y_val": y_val,
        }
        pooled_oof_pred_pct.append(oof_preds / capacity)
        pooled_oof_actual_pct.append(y_train.to_numpy() / capacity)

    # one shared calibration curve fit on all 3 groups' OOF data, normalized to % of
    # capacity -- borrows strength across groups (all 3 show the same bias shape:
    # over-predict low output, under-predict high output), which especially helps
    # kpx_group_3 whose own OOF set (2023 only) is too small for a stable per-group fit
    pooled_calibrator = IsotonicRegression(out_of_bounds="clip")
    pooled_calibrator.fit(np.concatenate(pooled_oof_pred_pct), np.concatenate(pooled_oof_actual_pct))

    # group_2 alone: highest capacity factor / consistently the best-fit group across
    # every test so far -- maybe its OOF pairs are the "cleanest" calibration signal,
    # and pooling in group_1/group_3 just adds noise instead of helping
    group2_calibrator = IsotonicRegression(out_of_bounds="clip")
    group2_calibrator.fit(
        per_group["kpx_group_2"]["oof_preds"] / per_group["kpx_group_2"]["capacity"],
        per_group["kpx_group_2"]["y_train"] / per_group["kpx_group_2"]["capacity"],
    )

    rows = []
    for group, data in per_group.items():
        capacity = data["capacity"]

        per_group_calibrator = IsotonicRegression(out_of_bounds="clip")
        per_group_calibrator.fit(data["oof_preds"], data["y_train"])
        per_group_cal_pred = np.clip(per_group_calibrator.predict(data["raw_val_pred"]), 0, capacity)

        pooled_cal_pred_pct = pooled_calibrator.predict(data["raw_val_pred"] / capacity)
        pooled_cal_pred = np.clip(pooled_cal_pred_pct * capacity, 0, capacity)

        group2_cal_pred_pct = group2_calibrator.predict(data["raw_val_pred"] / capacity)
        group2_cal_pred = np.clip(group2_cal_pred_pct * capacity, 0, capacity)

        variants = {
            "raw": data["raw_val_pred"],
            "per_group_isotonic": per_group_cal_pred,
            "pooled_isotonic": pooled_cal_pred,
            "group2_only_isotonic": group2_cal_pred,
        }
        for variant, pred in variants.items():
            nmae, ficr = group_nmae_ficr(data["y_val"], pred, capacity)
            score = 0.5 * (1 - nmae) + 0.5 * ficr
            rows.append({"group": group, "variant": variant, "score": score, "nmae": nmae, "ficr": ficr})
            print(f"{group}/{variant}: score={score:.4f} (nmae={nmae:.4f}, ficr={ficr:.4f})")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print()
    print(results_df.pivot(index="group", columns="variant", values="score").round(4))

    return results_df


if __name__ == "__main__":
    main()
