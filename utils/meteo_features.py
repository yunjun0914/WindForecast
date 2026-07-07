import numpy as np
import pandas as pd

from utils.preprocessing import FARM_CENTROID, TIME_KEY_COLS, haversine_km


LDAPS_WIND_RAW = {
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_50_50MUmax",
    "heightAboveGround_50_50MUmin",
    "heightAboveGround_50_50MVmax",
    "heightAboveGround_50_50MVmin",
    "heightAboveGround_5_XBLWS",
    "heightAboveGround_5_YBLWS",
}

GFS_WIND_RAW = {
    "heightAboveGround_10_10u",
    "heightAboveGround_10_10v",
    "heightAboveGround_80_u",
    "heightAboveGround_80_v",
    "heightAboveGround_100_100u",
    "heightAboveGround_100_100v",
    "planetaryBoundaryLayer_0_u",
    "planetaryBoundaryLayer_0_v",
    "isobaricInhPa_850_u",
    "isobaricInhPa_850_v",
    "isobaricInhPa_700_u",
    "isobaricInhPa_700_v",
    "isobaricInhPa_500_u",
    "isobaricInhPa_500_v",
    "surface_0_gust",
}


def add_lead_feature(out, raw_df):
    keys = raw_df[["forecast_kst_dtm", "data_available_kst_dtm"]].drop_duplicates().copy()
    keys["forecast_kst_dtm"] = pd.to_datetime(keys["forecast_kst_dtm"])
    keys["data_available_kst_dtm"] = pd.to_datetime(keys["data_available_kst_dtm"])
    lead = keys.groupby("forecast_kst_dtm")["data_available_kst_dtm"].min().reset_index()
    lead["lead_hour"] = (lead["forecast_kst_dtm"] - lead["data_available_kst_dtm"]).dt.total_seconds() / 3600.0
    return out.merge(lead[["forecast_kst_dtm", "lead_hour"]], on="forecast_kst_dtm", how="left")


def nearest_gfs_grid_id(df):
    grids = df[["grid_id", "latitude", "longitude"]].drop_duplicates()
    dist = haversine_km(FARM_CENTROID[0], FARM_CENTROID[1], grids["latitude"], grids["longitude"])
    return grids.loc[dist.idxmin(), "grid_id"]


def build_meteo_features(ldaps, gfs):
    ldaps_cols = [
        c
        for c in ldaps.columns
        if c not in TIME_KEY_COLS + ["grid_id", "latitude", "longitude"] and c not in LDAPS_WIND_RAW
    ]
    ldaps_agg = ldaps.groupby(TIME_KEY_COLS, as_index=False)[ldaps_cols].mean()
    ldaps_agg = ldaps_agg.rename(columns={c: f"met_ldaps_{c}" for c in ldaps_cols})

    grid_id = nearest_gfs_grid_id(gfs)
    gfs_sel = gfs[gfs["grid_id"] == grid_id].copy()
    gfs_cols = [
        c
        for c in gfs_sel.columns
        if c not in TIME_KEY_COLS + ["grid_id", "latitude", "longitude"] and c not in GFS_WIND_RAW
    ]
    gfs_agg = gfs_sel[TIME_KEY_COLS + gfs_cols].rename(columns={c: f"met_gfs_{c}" for c in gfs_cols})

    out = ldaps_agg.merge(gfs_agg, on="forecast_kst_dtm", how="inner", suffixes=("", "_gfs_dup"))
    out = out.drop(columns=[c for c in out.columns if c.endswith("_gfs_dup")])
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out = add_lead_feature(out, ldaps)
    return out.sort_values("forecast_kst_dtm").ffill().fillna(0).reset_index(drop=True)


def add_meteo_block(weather, meteo, block):
    out = weather.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    met = meteo.copy()
    met["forecast_kst_dtm"] = pd.to_datetime(met["forecast_kst_dtm"])
    if block == "baseline":
        return out

    cols = ["forecast_kst_dtm"]
    if "lead" in block:
        cols += ["lead_hour"]
    if block == "thermo":
        keys = [
            "_2_t",
            "_2_dpt",
            "_2_r",
            "_2_q",
            "_2_2t",
            "_2_2d",
            "_2_2r",
            "_2_2sh",
            "_sp",
            "_prmsl",
            "_850_t",
            "_850_r",
            "_700_t",
            "_500_t",
            "_blh",
        ]
        cols += [c for c in met.columns if any(k in c for k in keys)]
    elif block == "radiation_cloud":
        keys = ["SW", "LW", "dswrf", "dlwrf", "cloud", "Cloud", "hcc", "mcc", "lcc", "tcc", "CDC", "prate", "tp", "lsrate", "snol", "SNOM"]
        cols += [c for c in met.columns if any(k in c for k in keys)]
    elif "all_meteo" in block:
        cols += [c for c in met.columns if c not in TIME_KEY_COLS]

    cols = [c for c in dict.fromkeys(cols) if c in met.columns]
    return out.merge(met[cols], on="forecast_kst_dtm", how="left").sort_values("forecast_kst_dtm").ffill().fillna(0)


def speed(u, v):
    return np.sqrt(u**2 + v**2)


def aggregate_speed_distribution(df, specs, prefix, group_cols):
    work = df[group_cols].copy()
    for u_col, v_col, name in specs:
        work[f"_{name}"] = speed(df[u_col], df[v_col])

    rows = []
    for name in [spec[2] for spec in specs]:
        grouped = work.groupby("forecast_kst_dtm")[f"_{name}"]
        stat = grouped.agg(["mean", "std", "min", "max"]).reset_index()
        stat[f"{prefix}_{name}_p75"] = grouped.quantile(0.75).to_numpy()
        stat = stat.rename(
            columns={
                "mean": f"{prefix}_{name}_grid_mean",
                "std": f"{prefix}_{name}_grid_std",
                "min": f"{prefix}_{name}_grid_min",
                "max": f"{prefix}_{name}_grid_max",
            }
        )
        rows.append(stat)

    out = rows[0]
    for row in rows[1:]:
        out = out.merge(row, on="forecast_kst_dtm", how="left")
    return out

