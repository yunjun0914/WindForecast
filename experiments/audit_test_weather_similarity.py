"""test 2025 기상분포가 train 어느 연도와 닮았는지 진단.

모델이 실제로 보는 입력 공간(build_weather_features + all_meteo)에서
피처별 월-매칭 Wasserstein distance를 계산한다. 학습/제출 없음.

출력:
- results/test_weather_similarity_feature_distances.csv
- results/test_weather_similarity_year_summary.csv
- results/test_weather_similarity_month_summary.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import _bootstrap  # noqa: F401
from utils.meteo_features import add_meteo_block, build_meteo_features
from utils.preprocessing import TIME_KEY_COLS, build_weather_features


TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEAR = 2025
YEAR_BAG_FOLDS = [(2022, 2023), (2022, 2024), (2023, 2024)]

# 모델 중요도가 확인된 핵심 신호. all-피처 집계와 순위가 일치하는지가 신뢰도 체크.
CORE_FEATURES = [
    "gfs_ws850_speed",
    "gfs_ws100_speed",  # hub-height proxy
    "gfs_ws80_speed",
    "ldaps_ws10_speed",
    "ldaps_ws50_max_speed",
    "gfs_surface_0_gust",
    "met_ldaps_heightAboveGround_2_t",
    "met_ldaps_surface_0_sp",
]

PINN_OOF_SCORES = "results/pinn_effective_grid_g1_year_bagging_lgbm_time_oof_oof_scores.csv"
TREE_OOF_SCORES = "results/power_lgbm_best_v2_l1_scores.csv"


def build_feature_table(ldaps, gfs):
    weather = build_weather_features(ldaps, gfs)
    meteo = build_meteo_features(ldaps, gfs)
    out = add_meteo_block(weather, meteo, "all_meteo")
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["year"] = out["forecast_kst_dtm"].dt.year
    out["month"] = out["forecast_kst_dtm"].dt.month
    return out


def feature_columns(df):
    skip = set(TIME_KEY_COLS) | {"year", "month"}
    cols = []
    for c in df.columns:
        if c in skip or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].std(skipna=True) > 0:  # 상수 피처는 거리 0으로 무의미
            cols.append(c)
    return cols


def month_matched_distances(target_df, train_df, features, scale):
    """target(연도 1개) vs train 각 연도, 같은 달끼리 정규화 Wasserstein distance."""
    rows = []
    for year in sorted(train_df["year"].unique()):
        year_df = train_df[train_df["year"] == year]
        for month in sorted(target_df["month"].unique()):
            a_month = target_df[target_df["month"] == month]
            b_month = year_df[year_df["month"] == month]
            if len(a_month) == 0 or len(b_month) == 0:
                continue
            for feat in features:
                a = a_month[feat].dropna().to_numpy()
                b = b_month[feat].dropna().to_numpy()
                if len(a) == 0 or len(b) == 0:
                    continue
                dist = wasserstein_distance(a, b) / scale[feat]
                rows.append({"train_year": year, "month": month, "feature": feat, "wasserstein_norm": dist})
    return pd.DataFrame(rows)


def whole_year_distances(target_df, train_df, features, scale):
    rows = []
    for year in sorted(train_df["year"].unique()):
        year_df = train_df[train_df["year"] == year]
        for feat in features:
            a = target_df[feat].dropna().to_numpy()
            b = year_df[feat].dropna().to_numpy()
            dist = wasserstein_distance(a, b) / scale[feat]
            rows.append({"train_year": year, "feature": feat, "wasserstein_norm_whole_year": dist})
    return pd.DataFrame(rows)


def summarize_years(dist_df, whole_df, features_core):
    """연도별 core/all 집계 + 순위."""
    per_year_feat = dist_df.groupby(["train_year", "feature"], as_index=False)["wasserstein_norm"].mean()
    rows = []
    for year, year_df in per_year_feat.groupby("train_year"):
        core = year_df[year_df["feature"].isin(features_core)]["wasserstein_norm"].mean()
        allf = year_df["wasserstein_norm"].mean()
        whole = whole_df[whole_df["train_year"] == year]["wasserstein_norm_whole_year"].mean()
        rows.append(
            {
                "train_year": year,
                "dist_core_month_matched": core,
                "dist_all_month_matched": allf,
                "dist_all_whole_year": whole,
            }
        )
    out = pd.DataFrame(rows)
    out["rank_core"] = out["dist_core_month_matched"].rank().astype(int)
    out["rank_all"] = out["dist_all_month_matched"].rank().astype(int)
    return out.sort_values("dist_all_month_matched").reset_index(drop=True)


def fold_weights_from_distances(year_summary, dist_col="dist_all_month_matched"):
    """year-bagging fold(train {a,b})의 유사도 가중 제안: softmax(-mean distance / T)."""
    dist_by_year = dict(zip(year_summary["train_year"], year_summary[dist_col]))
    fold_dist = {fold: np.mean([dist_by_year[y] for y in fold]) for fold in YEAR_BAG_FOLDS}
    temp = np.std(list(fold_dist.values()))
    if temp <= 0:
        return {fold: 1.0 / len(YEAR_BAG_FOLDS) for fold in YEAR_BAG_FOLDS}, fold_dist
    logits = {fold: -d / temp for fold, d in fold_dist.items()}
    max_logit = max(logits.values())
    exps = {fold: np.exp(v - max_logit) for fold, v in logits.items()}
    total = sum(exps.values())
    return {fold: v / total for fold, v in exps.items()}, fold_dist


def month_summary(dist_df):
    per_month = dist_df.groupby(["month", "train_year"], as_index=False)["wasserstein_norm"].mean()
    rows = []
    for month, month_df in per_month.groupby("month"):
        best = month_df.loc[month_df["wasserstein_norm"].idxmin()]
        pivot = {f"dist_{int(r['train_year'])}": r["wasserstein_norm"] for _, r in month_df.iterrows()}
        rows.append({"month": month, "nearest_train_year": int(best["train_year"]), **pivot})
    return pd.DataFrame(rows)


def load_fold_scores():
    """기존 OOF 점수에서 pred_year별 PINN/TREE fold mean을 뽑아 교차분석."""
    out = {}
    try:
        pinn = pd.read_csv(PINN_OOF_SCORES)
        pinn_fold = pinn[pinn["stage"] == "fold_mean"][["pred_year", "score"]]
        out["pinn"] = dict(zip(pinn_fold["pred_year"].astype(int), pinn_fold["score"]))
    except FileNotFoundError:
        print(f"warning: {PINN_OOF_SCORES} not found, skip PINN cross-check")
    try:
        tree = pd.read_csv(TREE_OOF_SCORES)
        tree_fold = tree[tree["group"] == "fold_mean"][["pred_year", "score"]]
        out["tree"] = dict(zip(tree_fold["pred_year"].astype(int), tree_fold["score"]))
    except FileNotFoundError:
        print(f"warning: {TREE_OOF_SCORES} not found, skip TREE cross-check")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-stem", default="results/test_weather_similarity")
    args = parser.parse_args()

    print("load raw weather data")
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv("data/test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv("data/test/gfs_test.csv", encoding="utf-8-sig")

    print("build model-input feature tables")
    train_feat = build_feature_table(ldaps_train, gfs_train)
    test_feat = build_feature_table(ldaps_test, gfs_test)
    # 연말 경계 행(예: train의 2025-01-01 00:00) 제거 — 기존 filter_forecast_years와 같은 달력 연도 기준
    train_feat = train_feat[train_feat["year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    test_feat = test_feat[test_feat["year"] == TEST_YEAR].reset_index(drop=True)

    features = [c for c in feature_columns(train_feat) if c in test_feat.columns]
    core = [c for c in CORE_FEATURES if c in features]
    missing_core = [c for c in CORE_FEATURES if c not in features]
    if missing_core:
        print(f"warning: core features missing from table: {missing_core}")
    print(f"features: all={len(features)}, core={len(core)}, train_rows={len(train_feat)}, test_rows={len(test_feat)}")

    # 스케일: train 3년 전체 std (피처 간 집계를 가능하게 하는 정규화)
    scale = {f: max(train_feat[f].std(skipna=True), 1e-12) for f in features}

    print("\n=== test 2025 vs train years ===")
    dist_df = month_matched_distances(test_feat, train_feat, features, scale)
    whole_df = whole_year_distances(test_feat, train_feat, features, scale)
    year_sum = summarize_years(dist_df, whole_df, core)
    print(year_sum.to_string(index=False))

    weights, fold_dist = fold_weights_from_distances(year_sum)
    print("\nyear-bagging fold weight 제안 (균등=0.333):")
    for fold in YEAR_BAG_FOLDS:
        print(f"  train {fold}: mean_dist={fold_dist[fold]:.4f}, weight={weights[fold]:.3f}")

    mon_sum = month_summary(dist_df)
    print("\n2025 월별 최근접 train 연도:")
    print(mon_sum.to_string(index=False))

    # sanity check: 2024를 pseudo-test로 두고 2022/2023과의 거리 순위 확인
    print("\n=== sanity check: 2024 pseudo-test vs 2022/2023 ===")
    pseudo_target = train_feat[train_feat["year"] == 2024]
    pseudo_train = train_feat[train_feat["year"].isin([2022, 2023])]
    pseudo_dist = month_matched_distances(pseudo_target, pseudo_train, features, scale)
    pseudo_whole = whole_year_distances(pseudo_target, pseudo_train, features, scale)
    pseudo_sum = summarize_years(pseudo_dist, pseudo_whole, core)
    print(pseudo_sum.to_string(index=False))

    # 교차분석: 2025 최근접 연도를 예측한 fold에서 PINN vs TREE 점수
    fold_scores = load_fold_scores()
    if fold_scores:
        nearest = int(year_sum.iloc[0]["train_year"])
        print(f"\n=== 모델 강도 교차분석 (2025 최근접 연도 = {nearest}) ===")
        print("pred_year별 fold mean score (참고: blend는 운영 원칙상 50:50 고정):")
        for model, scores in fold_scores.items():
            line = ", ".join(f"{y}: {s:.4f}" for y, s in sorted(scores.items()))
            marker = f" <- {nearest} fold가 2025와 가장 유사" if nearest in scores else ""
            print(f"  {model}: {line}{marker}")

    dist_out = f"{args.out_stem}_feature_distances.csv"
    year_out = f"{args.out_stem}_year_summary.csv"
    month_out = f"{args.out_stem}_month_summary.csv"

    per_year_feat = dist_df.groupby(["train_year", "feature"], as_index=False)["wasserstein_norm"].mean()
    per_year_feat = per_year_feat.merge(whole_df, on=["train_year", "feature"], how="left")
    per_year_feat["is_core"] = per_year_feat["feature"].isin(core)
    per_year_feat.to_csv(dist_out, index=False, encoding="utf-8-sig")

    year_sum["fold_weight_suggestion"] = np.nan
    weight_rows = []
    for fold in YEAR_BAG_FOLDS:
        weight_rows.append(
            {
                "train_year": f"fold_{fold[0]}_{fold[1]}",
                "dist_all_month_matched": fold_dist[fold],
                "fold_weight_suggestion": weights[fold],
            }
        )
    pd.concat([year_sum, pd.DataFrame(weight_rows)], ignore_index=True).to_csv(
        year_out, index=False, encoding="utf-8-sig"
    )
    mon_sum.to_csv(month_out, index=False, encoding="utf-8-sig")

    print(f"\nsaved {dist_out}")
    print(f"saved {year_out}")
    print(f"saved {month_out}")


if __name__ == "__main__":
    main()
