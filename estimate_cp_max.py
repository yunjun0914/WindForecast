import numpy as np
import pandas as pd

from utils.pinn_physics import MANUFACTURER_AREA, air_density

RESULTS_PATH = "results/cp_max_estimates.csv"

# (turbine_prefix, manufacturer) pairs; scada wind speed columns are at nacelle
# anemometer height, treated here as the same v used in the physics equation
TURBINE_MANUFACTURER = {
    **{f"vestas_wtg{i:02d}": "vestas" for i in range(1, 13)},
    **{f"unison_wtg{i:02d}": "unison" for i in range(1, 6)},
}

# per info.xlsx: unison_wtg01 sits at 태백가덕산 (same site as all vestas turbines),
# unison_wtg02-05 sit at 태백원동 (a different site) -- checked as an alternative
# explanation for unison's inflated cp before settling on the turbulence-bias one
TURBINE_SITE = {
    "unison_wtg01": "taebaek_gadeoksan",
    "unison_wtg02": "taebaek_wondong",
    "unison_wtg03": "taebaek_wondong",
    "unison_wtg04": "taebaek_wondong",
    "unison_wtg05": "taebaek_wondong",
}

MIN_WIND_SPEED = 9.0  # m/s -- clear of the cut-in turbulence-bias zone (see
# estimate_cp_max diagnostics: cp is inflated and falling monotonically from v=5
# up through v=11, only leveling off into a physically plausible range past ~9)


def hourly_mean_temp_pressure(ldaps_df):
    """LDAPS 16-grid mean temperature (K) and surface pressure (Pa), hourly."""
    agg = ldaps_df.groupby("forecast_kst_dtm", as_index=False)[
        ["heightAboveGround_2_t", "surface_0_sp"]
    ].mean()
    agg = agg.rename(columns={"heightAboveGround_2_t": "temp_k", "surface_0_sp": "pressure_pa"})
    agg["forecast_kst_dtm"] = pd.to_datetime(agg["forecast_kst_dtm"])
    return agg.sort_values("forecast_kst_dtm").reset_index(drop=True)


def implied_cp_eff(scada_df, weather_hourly, turbine_prefix, area):
    ws_col, power_col = f"{turbine_prefix}_ws", f"{turbine_prefix}_power_kw10m"
    df = scada_df[["kst_dtm", ws_col, power_col]].dropna().copy()
    df["kst_dtm"] = pd.to_datetime(df["kst_dtm"])
    df["nearest_hour"] = df["kst_dtm"].dt.round("h")

    df = df.merge(weather_hourly, left_on="nearest_hour", right_on="forecast_kst_dtm", how="inner")
    df = df[df[ws_col] >= MIN_WIND_SPEED]

    rho = air_density(df["temp_k"].to_numpy(), df["pressure_pa"].to_numpy())
    # power_kw10m is energy (kWh) accumulated over the 10-min interval, not average
    # kW power -- confirmed against train_labels.csv (sum-of-six-readings matches the
    # hourly label; mean-of-six is off by ~5.6x). Convert to average power: kWh -> kW
    # by dividing by the 10-min duration in hours (x6), then kW -> W (x1000).
    power_w = df[power_col].to_numpy() * 6 * 1000.0
    v = df[ws_col].to_numpy()

    cp_eff = power_w / (0.5 * rho * area * v**3)
    return cp_eff


def main():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    weather_hourly = hourly_mean_temp_pressure(ldaps_train)

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_manufacturer = {"vestas": scada_vestas, "unison": scada_unison}

    rows = []
    cp_by_manufacturer = {}
    for prefix, manufacturer in TURBINE_MANUFACTURER.items():
        scada_df = scada_by_manufacturer[manufacturer]
        cp_eff = implied_cp_eff(scada_df, weather_hourly, prefix, MANUFACTURER_AREA[manufacturer])
        cp_by_manufacturer.setdefault(manufacturer, []).append(cp_eff)
        rows.append({"turbine": prefix, "manufacturer": manufacturer, "n_samples": len(cp_eff),
                     "p50": np.percentile(cp_eff, 50), "p95": np.percentile(cp_eff, 95),
                     "p99": np.percentile(cp_eff, 99), "max": cp_eff.max()})

    per_turbine_df = pd.DataFrame(rows)
    print(per_turbine_df.round(4).to_string(index=False))

    print("\nPer-manufacturer (pooled across turbines):")
    summary_rows = []
    for manufacturer, arrays in cp_by_manufacturer.items():
        pooled = np.concatenate(arrays)
        summary_rows.append({
            "manufacturer": manufacturer,
            "n_samples": len(pooled),
            "p50": np.percentile(pooled, 50),
            "p95": np.percentile(pooled, 95),
            "p99": np.percentile(pooled, 99),
            "max": pooled.max(),
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.round(4).to_string(index=False))

    print("\nUnison split by site:")
    site_df = per_turbine_df[per_turbine_df["manufacturer"] == "unison"].copy()
    site_df["site"] = site_df["turbine"].map(TURBINE_SITE)
    print(site_df[["turbine", "site", "n_samples", "p50", "p95", "p99", "max"]].round(4).to_string(index=False))

    summary_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    return per_turbine_df, summary_df


if __name__ == "__main__":
    main()
