import numpy as np
import pandas as pd

from evaluate_group3_effective_teacher_mix import blend_weather
from evaluate_pinn_effective_wind_teacher import build_extended_pinn_weather
from evaluate_pinn_meteo_teacher_model_blend import apply_teacher, fit_teacher
from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features
from predict_final_tree_ensemble import GROUP1, GROUP2, GROUP3, GROUPS, predict_pinn, train_full_pinn
from utils.metrics import GROUP_CAPACITY_KWH

SUBMISSION_PATH = "results/submission_pinn_meteo_teacher_blend.csv"
TEACHER_KIND = "rf_hist_gbr_avg"


def teacher_train_test(weather_train, weather_test, scada_df, group, v_mode):
    teacher = fit_teacher(weather_train, scada_df, group, TEACHER_KIND, fit_before=None)
    return apply_teacher(weather_train, teacher, v_mode), apply_teacher(weather_test, teacher, v_mode)


def build_weather_train_test():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv("data/test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv("data/test/gfs_test.csv", encoding="utf-8-sig")

    weather_train = build_extended_pinn_weather(ldaps_train, gfs_train)
    weather_test = build_extended_pinn_weather(ldaps_test, gfs_test)
    weather_train = add_meteo_block(weather_train, build_meteo_features(ldaps_train, gfs_train), "all_meteo")
    weather_test = add_meteo_block(weather_test, build_meteo_features(ldaps_test, gfs_test), "all_meteo")

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")

    g1_train, g1_test = teacher_train_test(weather_train, weather_test, scada_vestas, GROUP1, "cubic")
    g2_train, g2_test = teacher_train_test(weather_train, weather_test, scada_vestas, GROUP2, "p90")
    g3_unison_train, g3_unison_test = teacher_train_test(weather_train, weather_test, scada_unison, GROUP3, "p90")
    g3_vestas_train, g3_vestas_test = teacher_train_test(weather_train, weather_test, scada_vestas, GROUP2, "p90")
    g3_train = blend_weather("group3_meteo_teacher_blend_p90_mix", g3_unison_train, g3_vestas_train, 0.30)
    g3_test = blend_weather("group3_meteo_teacher_blend_p90_mix", g3_unison_test, g3_vestas_test, 0.30)

    return {
        "train": {GROUP1: g1_train, GROUP2: g2_train, GROUP3: g3_train},
        "test": {GROUP1: g1_test, GROUP2: g2_test, GROUP3: g3_test},
    }


def main():
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    weather = build_weather_train_test()

    vestas_model, vestas_bias = train_full_pinn(
        "vestas",
        {GROUP1: weather["train"][GROUP1], GROUP2: weather["train"][GROUP2]},
        labels,
    )
    unison_model, unison_bias = train_full_pinn("unison", {GROUP3: weather["train"][GROUP3]}, labels)

    prediction = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(weather["test"][GROUP1]["forecast_kst_dtm"])})
    prediction[GROUP1] = np.clip(
        predict_pinn(vestas_model, vestas_bias[GROUP1], GROUP1, weather["test"][GROUP1]),
        0,
        GROUP_CAPACITY_KWH[GROUP1],
    )
    prediction[GROUP2] = np.clip(
        predict_pinn(vestas_model, vestas_bias[GROUP2], GROUP2, weather["test"][GROUP2]),
        0,
        GROUP_CAPACITY_KWH[GROUP2],
    )
    prediction[GROUP3] = np.clip(
        predict_pinn(unison_model, unison_bias[GROUP3], GROUP3, weather["test"][GROUP3]),
        0,
        GROUP_CAPACITY_KWH[GROUP3],
    )

    submission = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"])
    merged = submission[["forecast_id", "forecast_kst_dtm"]].merge(prediction, on="forecast_kst_dtm", how="left")
    if merged[GROUPS].isna().any().any():
        raise ValueError("submission has missing predictions")
    merged = merged[["forecast_id", "forecast_kst_dtm", GROUP1, GROUP2, GROUP3]]
    merged.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
    print(f"saved {SUBMISSION_PATH}: {merged.shape}")
    print(merged.head())
    print(merged.tail())
    return merged


if __name__ == "__main__":
    main()
