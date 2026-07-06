import pandas as pd


def describe_csv(path, time_col):
    df = pd.read_csv(path, encoding="utf-8-sig", nrows=20000)
    full_cols = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns.tolist()
    print(f"\n== {path} ==")
    print(f"n_cols={len(full_cols)}")
    print(full_cols)
    if time_col in df.columns:
        dt = pd.to_datetime(df[time_col])
        print("sample time range:", dt.min(), dt.max())
    for key in ["grid_id", "latitude", "longitude"]:
        if key in df.columns:
            print(key, "unique/sample:", df[key].nunique(), sorted(df[key].dropna().unique())[:10])
    missing = df.isna().mean().sort_values(ascending=False)
    print("top missing cols:")
    print(missing[missing > 0].head(20).to_string())


def main():
    describe_csv("data/train/ldaps_train.csv", "forecast_kst_dtm")
    describe_csv("data/train/gfs_train.csv", "forecast_kst_dtm")
    describe_csv("data/train/scada_vestas_train.csv", "kst_dtm")
    describe_csv("data/train/scada_unison_train.csv", "kst_dtm")
    describe_csv("data/train/train_labels.csv", "kst_dtm")
    describe_csv("data/test/ldaps_test.csv", "forecast_kst_dtm")
    describe_csv("data/test/gfs_test.csv", "forecast_kst_dtm")


if __name__ == "__main__":
    main()
