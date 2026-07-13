from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from utils.per_turbine_features import tree_feature_columns
from utils.per_turbine_scada import clean_power_10m, turbine_capacity_kwh
from utils.power_curve import GROUP_TURBINE_PREFIXES, fit_power_curve


TEACHER_CACHE_VERSION = "per_turbine_teacher_v1"
TEACHER_TARGET_COLS = [
    "scada_ws_mean",
    "scada_ws_cubic",
    "scada_wd_sin",
    "scada_wd_cos",
]
TEACHER_FEATURE_COLS = [
    "teacher_ws_mean",
    "teacher_ws_cubic",
    "teacher_wd_sin",
    "teacher_wd_cos",
    "teacher_power_curve_kwh",
]


def _teacher_matrix(
    table: pd.DataFrame,
    group: str,
    input_feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    numeric_cols = [
        *(tree_feature_columns(group) if input_feature_cols is None else input_feature_cols),
        "turbine_x_km",
        "turbine_y_km",
    ]
    numeric = table.reindex(columns=numeric_cols, fill_value=0).astype(float)
    one_hot = pd.get_dummies(table["turbine_id"], prefix="teacher_turbine", dtype=float)
    expected = [f"teacher_turbine_{t}" for t in GROUP_TURBINE_PREFIXES[group]]
    one_hot = one_hot.reindex(columns=expected, fill_value=0.0)
    return pd.concat([numeric.reset_index(drop=True), one_hot.reset_index(drop=True)], axis=1).replace(
        [np.inf, -np.inf], 0
    ).fillna(0)


def _fit_turbine_power_curves(
    scada: pd.DataFrame,
    group: str,
    train_years: list[int],
) -> dict[str, object]:
    source = scada.copy()
    source["kst_dtm"] = pd.to_datetime(source["kst_dtm"])
    source = source.loc[source["kst_dtm"].dt.year.isin(train_years)]
    curves = {}
    for turbine in GROUP_TURBINE_PREFIXES[group]:
        ws = pd.to_numeric(source[f"{turbine}_ws"], errors="coerce")
        power = clean_power_10m(source[f"{turbine}_power_kw10m"], group)
        valid = ws.between(0, 35) & power.notna()
        if int(valid.sum()) < 100:
            raise ValueError(f"Insufficient SCADA curve rows for {turbine}: {int(valid.sum())}")
        curves[turbine] = fit_power_curve(ws[valid].to_numpy(float), power[valid].to_numpy(float))
    return curves


def _add_power_curve_proxy(
    teacher: pd.DataFrame,
    curves: dict[str, object],
    group: str,
) -> pd.DataFrame:
    out = teacher.copy()
    out["teacher_power_curve_kwh"] = np.nan
    cap = turbine_capacity_kwh(group)
    for turbine, curve in curves.items():
        mask = out["turbine_id"].eq(turbine)
        out.loc[mask, "teacher_power_curve_kwh"] = np.clip(
            curve(out.loc[mask, "teacher_ws_cubic"].to_numpy(float)) * 6.0,
            0,
            cap,
        )
    return out


def build_teacher_fold(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    scada: pd.DataFrame,
    group: str,
    train_years: list[int],
    pred_year: int,
    n_estimators: int = 80,
    input_feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    keys = ["forecast_kst_dtm", "turbine_id"]
    feature_table = features.copy()
    feature_table["forecast_kst_dtm"] = pd.to_datetime(feature_table["forecast_kst_dtm"])
    feature_table["year"] = feature_table["forecast_kst_dtm"].dt.year
    target_table = targets[keys + TEACHER_TARGET_COLS].copy()
    target_table["forecast_kst_dtm"] = pd.to_datetime(target_table["forecast_kst_dtm"])

    train_all = feature_table.loc[feature_table["year"].isin(train_years)].copy()
    val_all = feature_table.loc[feature_table["year"].eq(pred_year)].copy()
    train_fit = train_all.merge(target_table, on=keys, how="inner").dropna(subset=TEACHER_TARGET_COLS)
    if len(train_fit) < 5_000 or len(val_all) == 0:
        raise ValueError(
            f"Insufficient teacher rows for {group} fold={pred_year}: train={len(train_fit)} val={len(val_all)}"
        )

    x_fit = _teacher_matrix(train_fit, group, input_feature_cols)
    y_fit = train_fit[TEACHER_TARGET_COLS].to_numpy(float)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=20,
        max_features=0.70,
        max_samples=0.80,
        bootstrap=True,
        oob_score=True,
        random_state=20260711 + pred_year,
        n_jobs=-1,
    )
    model.fit(x_fit, y_fit)

    output_names = TEACHER_FEATURE_COLS[:4]
    train_pred = model.predict(_teacher_matrix(train_all, group, input_feature_cols))
    val_pred = model.predict(_teacher_matrix(val_all, group, input_feature_cols))
    train_out = train_all[keys].copy()
    val_out = val_all[keys].copy()
    train_out[output_names] = train_pred
    val_out[output_names] = val_pred

    oob = np.asarray(model.oob_prediction_, dtype=float)
    oob_table = train_fit[keys].copy()
    oob_table[[f"{col}_oob" for col in output_names]] = oob
    train_out = train_out.merge(oob_table, on=keys, how="left")
    for col in output_names:
        oob_col = f"{col}_oob"
        valid_oob = np.isfinite(train_out[oob_col])
        train_out.loc[valid_oob, col] = train_out.loc[valid_oob, oob_col]
    train_out = train_out.drop(columns=[f"{col}_oob" for col in output_names])

    train_out["split"] = "train_oob"
    val_out["split"] = "validation"
    out = pd.concat([train_out, val_out], ignore_index=True)
    out["group"] = group
    out["pred_year"] = pred_year
    curves = _fit_turbine_power_curves(scada, group, train_years)
    out = _add_power_curve_proxy(out, curves, group)
    return out


def get_or_build_teacher_cache(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    scada: pd.DataFrame,
    group: str,
    train_years: list[int],
    pred_year: int,
    cache_root: Path = Path("cache"),
    rebuild: bool = False,
    n_estimators: int = 80,
    input_feature_cols: list[str] | None = None,
    cache_tag: str | None = None,
) -> pd.DataFrame:
    cache_dir = cache_root / TEACHER_CACHE_VERSION
    suffix = f"_{cache_tag}" if cache_tag else ""
    cache_path = cache_dir / f"{group}_pred{pred_year}{suffix}.pkl"
    if cache_path.exists() and not rebuild:
        return pd.read_pickle(cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher = build_teacher_fold(
        features=features,
        targets=targets,
        scada=scada,
        group=group,
        train_years=train_years,
        pred_year=pred_year,
        n_estimators=n_estimators,
        input_feature_cols=input_feature_cols,
    )
    teacher.to_pickle(cache_path)
    return teacher


def build_full_teacher(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    targets: pd.DataFrame,
    scada: pd.DataFrame,
    group: str,
    n_estimators: int = 80,
    input_feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    keys = ["forecast_kst_dtm", "turbine_id"]
    target_table = targets[keys + TEACHER_TARGET_COLS].copy()
    target_table["forecast_kst_dtm"] = pd.to_datetime(target_table["forecast_kst_dtm"])
    train_all = train_features.copy()
    test_all = test_features.copy()
    train_all["forecast_kst_dtm"] = pd.to_datetime(train_all["forecast_kst_dtm"])
    test_all["forecast_kst_dtm"] = pd.to_datetime(test_all["forecast_kst_dtm"])
    train_fit = train_all.merge(target_table, on=keys, how="inner").dropna(
        subset=TEACHER_TARGET_COLS
    )
    if len(train_fit) < 5_000 or len(test_all) == 0:
        raise ValueError(
            f"Insufficient full teacher rows for {group}: "
            f"train={len(train_fit)} test={len(test_all)}"
        )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=20,
        max_features=0.70,
        max_samples=0.80,
        bootstrap=True,
        oob_score=True,
        random_state=20260711,
        n_jobs=-1,
    )
    model.fit(
        _teacher_matrix(train_fit, group, input_feature_cols),
        train_fit[TEACHER_TARGET_COLS].to_numpy(float),
    )

    output_names = TEACHER_FEATURE_COLS[:4]
    train_out = train_all[keys].copy()
    test_out = test_all[keys].copy()
    train_out[output_names] = model.predict(
        _teacher_matrix(train_all, group, input_feature_cols)
    )
    test_out[output_names] = model.predict(
        _teacher_matrix(test_all, group, input_feature_cols)
    )
    oob = np.asarray(model.oob_prediction_, dtype=float)
    oob_table = train_fit[keys].copy()
    oob_table[[f"{col}_oob" for col in output_names]] = oob
    train_out = train_out.merge(oob_table, on=keys, how="left")
    for col in output_names:
        oob_col = f"{col}_oob"
        valid_oob = np.isfinite(train_out[oob_col])
        train_out.loc[valid_oob, col] = train_out.loc[valid_oob, oob_col]
    train_out = train_out.drop(columns=[f"{col}_oob" for col in output_names])
    train_out["split"] = "train_oob"
    test_out["split"] = "test"
    out = pd.concat([train_out, test_out], ignore_index=True)
    out["group"] = group
    train_years = sorted(pd.to_datetime(scada["kst_dtm"]).dt.year.unique().tolist())
    curves = _fit_turbine_power_curves(scada, group, train_years)
    return _add_power_curve_proxy(out, curves, group)


def get_or_build_full_teacher_cache(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    targets: pd.DataFrame,
    scada: pd.DataFrame,
    group: str,
    cache_root: Path = Path("cache"),
    rebuild: bool = False,
    n_estimators: int = 80,
    input_feature_cols: list[str] | None = None,
    cache_tag: str | None = None,
) -> pd.DataFrame:
    cache_dir = cache_root / TEACHER_CACHE_VERSION
    suffix = f"_{cache_tag}" if cache_tag else ""
    cache_path = cache_dir / f"{group}_full_test{suffix}.pkl"
    if cache_path.exists() and not rebuild:
        return pd.read_pickle(cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher = build_full_teacher(
        train_features=train_features,
        test_features=test_features,
        targets=targets,
        scada=scada,
        group=group,
        n_estimators=n_estimators,
        input_feature_cols=input_feature_cols,
    )
    teacher.to_pickle(cache_path)
    return teacher
