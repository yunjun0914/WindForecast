import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from evaluate_group3_effective_teacher_mix import blend_weather
from evaluate_pinn_effective_wind_teacher import (
    EXT_TARGETS,
    apply_extended_teacher,
    build_extended_pinn_weather,
    build_extended_scada_targets,
    extended_feature_cols,
)
from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features
from train_pinn import train_manufacturer

RESULTS_PATH = "results/pinn_meteo_teacher_model_blend_scores.csv"
GROUP1 = "kpx_group_1"
GROUP2 = "kpx_group_2"
GROUP3 = "kpx_group_3"
VAL_START = "2024-01-01 01:00:00"


def make_rf():
    return MultiOutputRegressor(RandomForestRegressor(n_estimators=120, min_samples_leaf=10, random_state=42, n_jobs=-1))


def make_hist_gbr():
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, l2_regularization=0.01, random_state=42)
    )


def fit_teacher(weather, scada_df, group, teacher_kind, fit_before=VAL_START):
    targets = build_extended_scada_targets(scada_df, group)
    df = weather.merge(targets, on="forecast_kst_dtm", how="inner").dropna()
    if fit_before is not None:
        df = df[df["forecast_kst_dtm"] < pd.Timestamp(fit_before)]
    feature_cols = extended_feature_cols(weather)

    if teacher_kind == "rf":
        model = make_rf()
        model.fit(df[feature_cols], df[EXT_TARGETS])
        return feature_cols, model
    if teacher_kind == "rf_hist_gbr_avg":
        rf = make_rf()
        hist = make_hist_gbr()
        rf.fit(df[feature_cols], df[EXT_TARGETS])
        hist.fit(df[feature_cols], df[EXT_TARGETS])
        return feature_cols, (rf, hist)
    raise ValueError(teacher_kind)


def apply_teacher(weather, teacher, v_mode):
    feature_cols, model = teacher
    if isinstance(model, tuple):
        pred = sum(m.predict(weather[feature_cols]) for m in model) / len(model)
        # Reuse the production conversion logic by wrapping the averaged predictor.
        class AveragedModel:
            def predict(self, _):
                return pred

        return apply_extended_teacher(weather, (feature_cols, AveragedModel()), v_mode)
    return apply_extended_teacher(weather, teacher, v_mode)


def teacher_weather(weather, scada_df, group, v_mode, teacher_kind, fit_before=VAL_START):
    teacher = fit_teacher(weather, scada_df, group, teacher_kind, fit_before=fit_before)
    return apply_teacher(weather, teacher, v_mode)


def build_recipe_weather(weather, scada_vestas, scada_unison, teacher_kind):
    g1 = teacher_weather(weather, scada_vestas, GROUP1, "cubic", teacher_kind)
    g2 = teacher_weather(weather, scada_vestas, GROUP2, "p90", teacher_kind)
    g3_unison = teacher_weather(weather, scada_unison, GROUP3, "p90", teacher_kind)
    g3_vestas = teacher_weather(weather, scada_vestas, GROUP2, "p90", teacher_kind)
    g3 = blend_weather(f"group3_meteo_{teacher_kind}_p90_mix", g3_unison, g3_vestas, 0.30)
    return {"vestas": {GROUP1: g1, GROUP2: g2}, "unison": {GROUP3: g3}}


def main():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")

    weather = build_extended_pinn_weather(ldaps, gfs)
    weather = add_meteo_block(weather, build_meteo_features(ldaps, gfs), "all_meteo")

    rows = []
    for teacher_kind in ["rf", "rf_hist_gbr_avg"]:
        print(f"\n=== teacher: {teacher_kind} ===")
        weather_by_manufacturer = build_recipe_weather(weather, scada_vestas, scada_unison, teacher_kind)
        for manufacturer, weather_by_group in weather_by_manufacturer.items():
            _, _, stage1, stage2 = train_manufacturer(
                manufacturer,
                weather_by_group,
                labels,
                verbose=False,
                save=False,
            )
            stage1["stage"] = "physics_only"
            stage2["stage"] = "with_bias"
            for frame in [stage1, stage2]:
                frame["teacher"] = teacher_kind
                frame["manufacturer"] = manufacturer
                rows.append(frame)
        current = pd.concat(rows, ignore_index=True)
        print(current[(current["teacher"] == teacher_kind) & (current["stage"] == "with_bias")].to_string(index=False))

    results = pd.concat(rows, ignore_index=True)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    summary = (
        results[results["stage"] == "with_bias"]
        .groupby("teacher", as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"))
        .sort_values("score", ascending=False)
    )
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
