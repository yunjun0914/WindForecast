import re

import numpy as np
import pandas as pd


INFO_PATH = "data/info.xlsx"
GROUP_COL = "KPX그룹"
COORD_COL = "좌표(Google)"


def parse_google_dms(value):
    """Parse strings like 37°16'55.61"N 128°57'02.10"E into decimal degrees."""
    if pd.isna(value):
        return np.nan, np.nan
    text = str(value)
    nums = re.findall(r"(\d+(?:\.\d+)?)", text)
    dirs = re.findall(r"([NSEW])", text.upper())
    if len(nums) < 6 or len(dirs) < 2:
        return np.nan, np.nan

    def one(deg, minute, second, direction):
        sign = -1 if direction in {"S", "W"} else 1
        return sign * (float(deg) + float(minute) / 60.0 + float(second) / 3600.0)

    lat = one(nums[0], nums[1], nums[2], dirs[0])
    lon = one(nums[3], nums[4], nums[5], dirs[1])
    return lat, lon


def load_turbine_metadata(path=INFO_PATH):
    raw = pd.read_excel(path, sheet_name="info", header=None)
    header_idx = raw.index[raw.eq(GROUP_COL).any(axis=1)][0]
    header = raw.loc[header_idx].tolist()
    df = raw.loc[header_idx + 1 :].copy()
    df.columns = header
    df = df.dropna(how="all")
    df = df[df[COORD_COL].notna()].copy()
    df[GROUP_COL] = df[GROUP_COL].ffill().astype(int)

    coords = df[COORD_COL].apply(parse_google_dms)
    df["latitude"] = coords.apply(lambda x: x[0])
    df["longitude"] = coords.apply(lambda x: x[1])
    df["group"] = "kpx_group_" + df[GROUP_COL].astype(str)
    df["turbine_id"] = df["제작사"].str.lower() + "_wtg" + df["호기"].astype(int).astype(str).str.zfill(2)

    numeric_cols = ["Hub Height(m)", "Rotor Diameter(m)", "설비용량(MW)", "그룹설비용량(MW)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def _latlon_to_xy_km(lat, lon, ref_lat, ref_lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = (lon - ref_lon) * 111.32 * np.cos(np.radians(ref_lat))
    y = (lat - ref_lat) * 110.57
    return x, y


def group_site_summary(path=INFO_PATH):
    meta = load_turbine_metadata(path)
    rows = {}
    for group, group_df in meta.groupby("group"):
        lat = group_df["latitude"].to_numpy(float)
        lon = group_df["longitude"].to_numpy(float)
        centroid_lat = float(np.nanmean(lat))
        centroid_lon = float(np.nanmean(lon))
        x, y = _latlon_to_xy_km(lat, lon, centroid_lat, centroid_lon)
        points = np.column_stack([x, y])
        if len(points) >= 2:
            _, _, vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
            axis_x, axis_y = vt[0]
        else:
            axis_x, axis_y = 1.0, 0.0
        rows[group] = {
            "group": group,
            "latitude": centroid_lat,
            "longitude": centroid_lon,
            "axis_x": float(axis_x),
            "axis_y": float(axis_y),
            "n_turbines": int(len(group_df)),
            "manufacturer": "+".join(sorted(group_df["제작사"].dropna().unique())),
            "capacity_mw": float(group_df["설비용량(MW)"].sum()),
        }
    return rows
