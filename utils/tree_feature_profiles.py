import numpy as np

from utils.compact_physics_features import add_compact_physics_features
from utils.meteo_features import add_meteo_block, build_meteo_features
from utils.preprocessing import TIME_KEY_COLS, build_weather_features
from utils.tree_context_features import add_tree_context_features, tree_context_feature_names
from utils.weather_time_features import add_weather_time_features, weather_time_feature_names


FEATURE_PROFILE_FULL_V2 = "full_v2"
FEATURE_PROFILE_AGGRESSIVE_MINIMAL_V1 = "aggressive_minimal_v1"
FEATURE_PROFILE_AGGRESSIVE_MINIMAL_ROLLING_V1 = "aggressive_minimal_rolling_v1"
FEATURE_PROFILE_AGGRESSIVE_MINIMAL_ROLLMEAN_V1 = "aggressive_minimal_rollmean_v1"
FEATURE_PROFILE_AGGRESSIVE_MINIMAL_CONTEXT_V1 = "aggressive_minimal_context_v1"
FEATURE_PROFILE_WIND_CUBIC_MINIMAL_V1 = "wind_cubic_minimal_v1"
FEATURE_PROFILE_IMPORTANCE_LEAN_V1 = "importance_lean_v1"
FEATURE_PROFILE_GROUP_TOP40_V1 = "group_top40_v1"
FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_V1 = "group_family_quota65_v1"
FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_CAUSALFLOW_V1 = "group_family_quota65_causalflow_v1"
FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_LEUSTAGOS_TPM3_V1 = "group_family_quota65_leustagos_tpm3_v1"
FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_LEUSTAGOS_POWER_TPM3_V1 = "group_family_quota65_leustagos_power_tpm3_v1"
FEATURE_PROFILES = [
    FEATURE_PROFILE_FULL_V2,
    FEATURE_PROFILE_AGGRESSIVE_MINIMAL_V1,
    FEATURE_PROFILE_AGGRESSIVE_MINIMAL_ROLLING_V1,
    FEATURE_PROFILE_AGGRESSIVE_MINIMAL_ROLLMEAN_V1,
    FEATURE_PROFILE_AGGRESSIVE_MINIMAL_CONTEXT_V1,
    FEATURE_PROFILE_WIND_CUBIC_MINIMAL_V1,
    FEATURE_PROFILE_IMPORTANCE_LEAN_V1,
    FEATURE_PROFILE_GROUP_TOP40_V1,
    FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_V1,
    FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_CAUSALFLOW_V1,
    FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_LEUSTAGOS_TPM3_V1,
    FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_LEUSTAGOS_POWER_TPM3_V1,
]


AGGRESSIVE_MINIMAL_V1_FEATURES = [
    # Calendar features that consistently survive importance checks.
    "sin_doy",
    "cos_doy",
    "sin_hod",
    "cos_hod",
    "hour_day_year_sin",
    "hour_day_year_cos",
    "hdw_sin",
    # Core LDAPS wind levels.
    "ldaps_heightAboveGround_10_10u",
    "ldaps_heightAboveGround_10_10v",
    "ldaps_heightAboveGround_50_50MUmax",
    "ldaps_heightAboveGround_50_50MVmax",
    "ldaps_heightAboveGround_50_50MUmin",
    "ldaps_heightAboveGround_50_50MVmin",
    "ldaps_heightAboveGround_5_XBLWS",
    "ldaps_heightAboveGround_5_YBLWS",
    "ldaps_ws10_speed",
    "ldaps_ws50_max_speed",
    "ldaps_ws50_min_speed",
    "ldaps_ws5_bl_speed",
    # Core GFS wind levels.
    "gfs_heightAboveGround_10_10u",
    "gfs_heightAboveGround_10_10v",
    "gfs_heightAboveGround_80_u",
    "gfs_heightAboveGround_80_v",
    "gfs_heightAboveGround_100_100u",
    "gfs_heightAboveGround_100_100v",
    "gfs_isobaricInhPa_850_u",
    "gfs_isobaricInhPa_850_v",
    "gfs_surface_0_gust",
    "gfs_ws10_speed",
    "gfs_ws80_speed",
    "gfs_ws100_speed",
    "gfs_ws850_speed",
    "gfs_ws500_speed",
    # Compact spatial wind statistics: keep only the strongest, interpretable forms.
    "phys_ldaps_ws50max_grid_max",
    "phys_ldaps_ws50max_grid_range",
    "phys_ldaps_ws50max_grid_p90",
    "phys_ldaps_ws10_grid_max",
    "phys_ldaps_ws10_grid_p90",
    "phys_ldaps_ws50min_grid_max",
    "phys_ldaps_ws50min_grid_range",
    "phys_gfs_ws850_grid_max",
    "phys_gfs_ws100_grid_max",
    "phys_gfs_ws10_grid_max",
    "phys_gfs_ws850_grid_p90",
    # Directional/spatial components.
    "phys_gfs_ws10_near_southwest",
    "phys_gfs_ws850_near_axis_cross",
    "phys_gfs_ws500_near_axis_cross",
    "phys_gfs_ws850_east_west_gradient",
    "phys_gfs_ws500_east_west_gradient",
    "phys_ldaps_ws50max_east_west_gradient",
    "phys_gfs_ws850_upwind_minus_downwind",
    "phys_gfs_ws100_upwind_minus_downwind",
    # Small physics block.
    "phys_gfs_air_density_x_gfs_ws850_speed_cube",
    "phys_gfs_air_density_x_gfs_ws100_speed_cube",
    "phys_ldaps_air_density_x_ldaps_ws50_max_speed_cube",
    "phys_shear_gfs_850_100",
    "phys_shear_gfs_100_10",
    "phys_shear_ldaps_50max_10",
    "phys_gfs_gust_factor",
    "phys_gfs_gust_minus_ws10",
    "phys_gfs_lapse_850_500",
    "phys_gfs_pbl_vrate",
    "phys_gfs_surface_pressure",
    "phys_ldaps_surface_pressure",
    "phys_gfs_shortwave",
    "phys_ldaps_shortwave",
    # A few historically strong raw meteo signals, retained explicitly rather than via all_meteo.
    "met_gfs_surface_0_dswrf",
    "met_gfs_isobaricInhPa_850_r",
    "met_gfs_isobaricInhPa_700_t",
    "met_ldaps_surface_0_sp",
]


WIND_CUBIC_MINIMAL_V1_FEATURES = [
    "sin_doy",
    "cos_doy",
    "sin_hod",
    "cos_hod",
    "lead_hour",
    "ldaps_ws10_speed",
    "ldaps_ws50_max_speed",
    "ldaps_ws50_min_speed",
    "gfs_ws100_speed",
    "gfs_ws850_speed",
    "gfs_surface_0_gust",
    "phys_ldaps_ws50max_grid_mean",
    "phys_ldaps_ws50max_grid_max",
    "phys_ldaps_ws50max_grid_p90",
    "phys_ldaps_ws50max_grid_cubic",
    "phys_ldaps_ws50min_grid_cubic",
    "phys_ldaps_ws10_grid_cubic",
    "phys_gfs_ws100_grid_cubic",
    "phys_gfs_ws850_grid_cubic",
    "phys_gfs_ws10_grid_cubic",
    "phys_gfs_air_density_x_gfs_ws100_speed_cube",
    "phys_gfs_air_density_x_gfs_ws850_speed_cube",
    "phys_ldaps_air_density_x_ldaps_ws50_max_speed_cube",
    "phys_shear_gfs_100_10",
    "phys_shear_ldaps_50max_10",
    "phys_gfs_gust_factor",
    "ldaps_ws50_max_speed_lead1",
    "ldaps_ws50_max_speed_lead3",
    "ldaps_ws50_max_speed_roll3_mean",
]


IMPORTANCE_LEAN_V1_FEATURES = [
    "sin_doy",
    "cos_doy",
    "sin_hod",
    "cos_hod",
    "phys_ldaps_ws50max_grid_max",
    "phys_ldaps_ws50max_grid_max_lead1",
    "phys_ldaps_ws50max_grid_max_lead3",
    "phys_ldaps_ws50max_grid_max_roll3_mean",
    "phys_ldaps_ws50max_grid_range",
    "phys_ldaps_ws50max_east_west_gradient",
    "phys_ldaps_ws50min_grid_range",
    "phys_ldaps_ws50min_grid_max",
    "phys_ldaps_ws10_grid_max",
    "phys_gfs_ws10_near_southwest",
    "phys_gfs_ws850_east_west_gradient",
    "phys_gfs_ws850_near_axis_cross",
    "phys_gfs_ws500_near_axis_cross",
    "phys_gfs_ws500_east_west_gradient",
    "gfs_ws100_speed",
    "gfs_ws500_speed",
    "gfs_ws850_speed",
    "gfs_ws850_speed_roll3_mean",
    "gfs_heightAboveGround_10_10v",
    "gfs_isobaricInhPa_850_u",
    "gfs_isobaricInhPa_850_v",
    "phys_gfs_lapse_850_500",
    "met_gfs_isobaricInhPa_700_t",
    "phys_gfs_surface_pressure",
    "phys_ldaps_ws50max_grid_cubic",
    "phys_gfs_ws850_grid_cubic",
]


GROUP_TOP40_V1_FEATURES = {
    "kpx_group_1": [
        "phys_ldaps_ws50max_grid_max_roll3_mean",
        "phys_ldaps_ws50max_grid_max_lead1",
        "phys_ldaps_ws50max_grid_max",
        "phys_ldaps_ws50max_grid_max_lead3",
        "cos_doy",
        "phys_ldaps_ws50max_grid_range",
        "sin_doy",
        "phys_ldaps_ws50max_east_west_gradient",
        "met_gfs_isobaricInhPa_700_t",
        "phys_gfs_ws10_near_southwest",
        "phys_gfs_ws850_east_west_gradient",
        "phys_ldaps_ws50min_grid_range",
        "gfs_isobaricInhPa_850_u",
        "phys_gfs_ws500_near_axis_cross",
        "phys_gfs_lapse_850_500",
        "phys_ldaps_surface_pressure",
        "gfs_ws500_speed",
        "ldaps_heightAboveGround_5_XBLWS",
        "phys_gfs_surface_pressure",
        "met_gfs_isobaricInhPa_850_r",
        "hdw_sin",
        "phys_ldaps_ws50min_grid_max",
        "phys_gfs_ws850_near_axis_cross",
        "phys_gfs_shortwave",
        "gfs_ws10_speed_roll3_mean",
        "gfs_ws850_speed_roll3_mean",
        "gfs_isobaricInhPa_850_v",
        "gfs_heightAboveGround_80_u",
        "phys_gfs_ws500_east_west_gradient",
        "hour_day_year_sin",
        "hour_day_year_cos",
        "phys_ldaps_ws10_grid_max",
        "phys_gfs_ws100_upwind_minus_downwind",
        "phys_gfs_ws850_upwind_minus_downwind",
        "phys_shear_gfs_100_10",
        "phys_gfs_ws850_grid_p90_roll3_mean",
        "gfs_heightAboveGround_10_10u",
        "phys_gfs_ws10_grid_max",
        "phys_gfs_ws100_grid_max",
        "gfs_ws100_speed",
    ],
    "kpx_group_2": [
        "phys_ldaps_ws50max_grid_max_lead1",
        "phys_ldaps_ws50max_grid_max_lead3",
        "phys_ldaps_ws50max_grid_max",
        "phys_ldaps_ws50min_grid_range",
        "phys_ldaps_ws50max_grid_range",
        "phys_ldaps_ws50max_grid_max_roll3_mean",
        "phys_ldaps_ws50max_east_west_gradient",
        "phys_gfs_ws850_near_axis_cross",
        "cos_doy",
        "sin_doy",
        "gfs_ws850_speed_roll3_mean",
        "phys_gfs_ws10_near_southwest",
        "phys_gfs_ws850_east_west_gradient",
        "phys_gfs_ws500_near_axis_cross",
        "met_gfs_isobaricInhPa_700_t",
        "phys_ldaps_ws50min_grid_max",
        "phys_gfs_shortwave",
        "gfs_ws500_speed",
        "phys_gfs_lapse_850_500",
        "ldaps_heightAboveGround_5_XBLWS",
        "phys_gfs_surface_pressure",
        "phys_ldaps_surface_pressure",
        "met_gfs_isobaricInhPa_850_r",
        "phys_gfs_ws850_grid_max_roll3_mean",
        "phys_shear_gfs_850_100",
        "phys_shear_gfs_100_10",
        "gfs_ws850_speed_lead3",
        "gfs_isobaricInhPa_850_u",
        "gfs_heightAboveGround_80_u",
        "gfs_isobaricInhPa_850_v",
        "phys_gfs_air_density_x_gfs_ws850_speed_cube",
        "gfs_heightAboveGround_100_100u",
        "gfs_ws10_speed_roll3_mean",
        "hour_day_year_sin",
        "phys_gfs_ws850_grid_p90_roll3_mean",
        "phys_gfs_ws500_east_west_gradient",
        "hour_day_year_cos",
        "phys_gfs_ws10_grid_max",
        "hdw_sin",
        "gfs_ws100_speed",
    ],
    "kpx_group_3": [
        "phys_ldaps_ws50max_grid_max_lead3",
        "phys_ldaps_ws50max_grid_max_roll3_mean",
        "phys_ldaps_ws50max_grid_max_lead1",
        "sin_doy",
        "gfs_heightAboveGround_10_10v",
        "phys_gfs_ws500_near_axis_cross",
        "phys_ldaps_ws10_grid_max",
        "phys_ldaps_ws50min_grid_range",
        "phys_gfs_ws850_east_west_gradient",
        "cos_doy",
        "phys_ldaps_ws50min_grid_max",
        "phys_gfs_ws850_near_axis_cross",
        "phys_ldaps_ws50max_grid_range",
        "phys_ldaps_ws50max_east_west_gradient",
        "phys_gfs_lapse_850_500",
        "phys_gfs_ws10_near_southwest",
        "gfs_ws500_speed",
        "met_gfs_isobaricInhPa_700_t",
        "phys_gfs_surface_pressure",
        "phys_ldaps_surface_pressure",
        "gfs_ws850_speed",
        "hdw_sin",
        "ldaps_heightAboveGround_5_XBLWS",
        "phys_gfs_ws500_east_west_gradient",
        "gfs_ws850_speed_roll3_mean",
        "gfs_isobaricInhPa_850_u",
        "phys_gfs_ws850_grid_max_roll3_mean",
        "hour_day_year_cos",
        "phys_gfs_ws850_grid_max_lead3",
        "met_gfs_isobaricInhPa_850_r",
        "phys_ldaps_ws50max_grid_max",
        "gfs_isobaricInhPa_850_v",
        "hour_day_year_sin",
        "phys_gfs_ws850_upwind_minus_downwind",
        "phys_gfs_shortwave",
        "phys_shear_gfs_850_100",
        "phys_gfs_ws100_upwind_minus_downwind",
        "gfs_ws10_speed_lead3",
        "phys_gfs_ws100_grid_max",
        "gfs_ws100_speed",
    ],
}


GROUP_FAMILY_QUOTA65_V1_FEATURES = {
    "kpx_group_1": [
        "phys_ldaps_ws50max_grid_max_roll3_mean",
        "phys_ldaps_ws50max_grid_max_lead1",
        "phys_ldaps_ws50max_grid_max",
        "phys_ldaps_ws50max_grid_max_lead3",
        "phys_ldaps_ws50max_grid_range",
        "phys_ldaps_ws50max_east_west_gradient",
        "phys_gfs_ws10_near_southwest",
        "phys_gfs_ws850_east_west_gradient",
        "phys_ldaps_ws50min_grid_range",
        "phys_gfs_ws500_near_axis_cross",
        "phys_ldaps_ws50min_grid_max",
        "phys_gfs_ws850_near_axis_cross",
        "phys_gfs_ws500_east_west_gradient",
        "phys_ldaps_ws10_grid_max",
        "phys_gfs_ws100_upwind_minus_downwind",
        "phys_gfs_ws850_upwind_minus_downwind",
        "phys_gfs_ws850_grid_p90_roll3_mean",
        "phys_gfs_ws10_grid_max",
        "gfs_ws500_speed",
        "ldaps_heightAboveGround_5_XBLWS",
        "gfs_heightAboveGround_10_10u",
        "ldaps_heightAboveGround_50_50MUmin",
        "gfs_heightAboveGround_10_10v",
        "ldaps_ws5_bl_speed",
        "ldaps_heightAboveGround_50_50MVmin",
        "gfs_heightAboveGround_100_100v",
        "ldaps_heightAboveGround_50_50MVmax",
        "gfs_heightAboveGround_100_100u",
        "gfs_ws850_speed",
        "gfs_ws10_speed_roll3_mean",
        "gfs_ws850_speed_roll3_mean",
        "gfs_ws850_speed_lead3",
        "gfs_ws850_speed_lead1",
        "gfs_ws10_speed_lead3",
        "ldaps_ws10_speed_lead3",
        "gfs_surface_0_gust_lead3",
        "gfs_surface_0_gust_roll3_mean",
        "ldaps_ws10_speed_roll3_mean",
        "gfs_ws100_speed_lead3",
        "phys_gfs_lapse_850_500",
        "phys_ldaps_surface_pressure",
        "phys_gfs_surface_pressure",
        "phys_gfs_shortwave",
        "phys_shear_gfs_100_10",
        "phys_gfs_air_density_x_gfs_ws850_speed_cube",
        "phys_shear_gfs_850_100",
        "phys_shear_ldaps_50max_10",
        "phys_gfs_gust_factor",
        "cos_doy",
        "sin_doy",
        "hdw_sin",
        "hour_day_year_sin",
        "hour_day_year_cos",
        "sin_hod",
        "cos_hod",
        "met_gfs_isobaricInhPa_700_t",
        "met_gfs_isobaricInhPa_850_r",
        "met_ldaps_surface_0_sp",
        "met_gfs_surface_0_dswrf",
        "gfs_isobaricInhPa_850_u",
        "gfs_isobaricInhPa_850_v",
        "gfs_heightAboveGround_80_u",
        "gfs_heightAboveGround_80_v",
        "gfs_ws100_speed",
    ],
    "kpx_group_2": [
        "phys_ldaps_ws50max_grid_max_lead1",
        "phys_ldaps_ws50max_grid_max_lead3",
        "phys_ldaps_ws50max_grid_max",
        "phys_ldaps_ws50min_grid_range",
        "phys_ldaps_ws50max_grid_range",
        "phys_ldaps_ws50max_grid_max_roll3_mean",
        "phys_ldaps_ws50max_east_west_gradient",
        "phys_gfs_ws850_near_axis_cross",
        "phys_gfs_ws10_near_southwest",
        "phys_gfs_ws850_east_west_gradient",
        "phys_gfs_ws500_near_axis_cross",
        "phys_ldaps_ws50min_grid_max",
        "phys_gfs_ws850_grid_max_roll3_mean",
        "phys_gfs_ws850_grid_p90_roll3_mean",
        "phys_gfs_ws500_east_west_gradient",
        "phys_gfs_ws10_grid_max",
        "phys_gfs_ws100_grid_max",
        "phys_gfs_ws850_grid_max",
        "gfs_ws500_speed",
        "ldaps_heightAboveGround_5_XBLWS",
        "gfs_heightAboveGround_100_100u",
        "ldaps_heightAboveGround_50_50MVmax",
        "ldaps_heightAboveGround_50_50MVmin",
        "gfs_heightAboveGround_10_10v",
        "ldaps_ws5_bl_speed",
        "gfs_heightAboveGround_100_100v",
        "gfs_heightAboveGround_10_10u",
        "gfs_ws10_speed",
        "ldaps_heightAboveGround_10_10v",
        "gfs_ws850_speed_roll3_mean",
        "gfs_ws850_speed_lead3",
        "gfs_ws10_speed_roll3_mean",
        "gfs_surface_0_gust_lead3",
        "gfs_ws100_speed_roll3_mean",
        "gfs_ws850_speed_lead1",
        "gfs_ws10_speed_lead3",
        "gfs_ws10_speed_lead1",
        "ldaps_ws50_max_speed_lead3",
        "gfs_ws100_speed_lead3",
        "phys_gfs_shortwave",
        "phys_gfs_lapse_850_500",
        "phys_gfs_surface_pressure",
        "phys_ldaps_surface_pressure",
        "phys_shear_gfs_850_100",
        "phys_shear_gfs_100_10",
        "phys_gfs_air_density_x_gfs_ws850_speed_cube",
        "phys_ldaps_shortwave",
        "phys_gfs_gust_factor",
        "cos_doy",
        "sin_doy",
        "hour_day_year_sin",
        "hour_day_year_cos",
        "hdw_sin",
        "sin_hod",
        "cos_hod",
        "met_gfs_isobaricInhPa_700_t",
        "met_gfs_isobaricInhPa_850_r",
        "met_gfs_surface_0_dswrf",
        "met_ldaps_surface_0_sp",
        "gfs_isobaricInhPa_850_u",
        "gfs_heightAboveGround_80_u",
        "gfs_isobaricInhPa_850_v",
        "gfs_heightAboveGround_80_v",
        "gfs_ws100_speed",
    ],
    "kpx_group_3": [
        "phys_ldaps_ws50max_grid_max_lead3",
        "phys_ldaps_ws50max_grid_max_roll3_mean",
        "phys_ldaps_ws50max_grid_max_lead1",
        "phys_gfs_ws500_near_axis_cross",
        "phys_ldaps_ws10_grid_max",
        "phys_ldaps_ws50min_grid_range",
        "phys_gfs_ws850_east_west_gradient",
        "phys_ldaps_ws50min_grid_max",
        "phys_gfs_ws850_near_axis_cross",
        "phys_ldaps_ws50max_grid_range",
        "phys_ldaps_ws50max_east_west_gradient",
        "phys_gfs_ws10_near_southwest",
        "phys_gfs_ws500_east_west_gradient",
        "phys_gfs_ws850_grid_max_roll3_mean",
        "phys_gfs_ws850_grid_max_lead3",
        "phys_ldaps_ws50max_grid_max",
        "phys_gfs_ws850_upwind_minus_downwind",
        "phys_gfs_ws100_upwind_minus_downwind",
        "gfs_heightAboveGround_10_10v",
        "gfs_ws500_speed",
        "gfs_ws850_speed",
        "ldaps_heightAboveGround_5_XBLWS",
        "ldaps_heightAboveGround_50_50MVmin",
        "gfs_heightAboveGround_10_10u",
        "gfs_heightAboveGround_100_100v",
        "ldaps_heightAboveGround_50_50MVmax",
        "ldaps_heightAboveGround_5_YBLWS",
        "ldaps_heightAboveGround_50_50MUmin",
        "ldaps_heightAboveGround_10_10v",
        "gfs_ws850_speed_roll3_mean",
        "gfs_ws10_speed_lead3",
        "gfs_ws850_speed_lead1",
        "ldaps_ws10_speed_lead3",
        "gfs_ws850_speed_lead3",
        "gfs_ws10_speed_roll3_mean",
        "gfs_ws100_speed_lead3",
        "gfs_surface_0_gust_lead3",
        "ldaps_ws10_speed_roll3_mean",
        "ldaps_ws50_max_speed_lead3",
        "phys_gfs_lapse_850_500",
        "phys_gfs_surface_pressure",
        "phys_ldaps_surface_pressure",
        "phys_gfs_shortwave",
        "phys_shear_gfs_850_100",
        "phys_ldaps_shortwave",
        "phys_shear_gfs_100_10",
        "phys_shear_ldaps_50max_10",
        "phys_gfs_gust_minus_ws10",
        "sin_doy",
        "cos_doy",
        "hdw_sin",
        "hour_day_year_cos",
        "hour_day_year_sin",
        "cos_hod",
        "sin_hod",
        "met_gfs_isobaricInhPa_700_t",
        "met_gfs_isobaricInhPa_850_r",
        "met_ldaps_surface_0_sp",
        "met_gfs_surface_0_dswrf",
        "gfs_isobaricInhPa_850_u",
        "gfs_isobaricInhPa_850_v",
        "gfs_heightAboveGround_80_v",
        "gfs_heightAboveGround_80_u",
        "gfs_ws100_speed",
    ],
}


CAUSAL_FLOW_BASE_COLS = [
    "phys_ldaps_ws50max_grid_max",
    "phys_ldaps_ws50max_grid_p90",
    "phys_gfs_ws850_grid_max",
    "phys_gfs_ws850_grid_p90",
    "gfs_ws850_speed",
]

LEUSTAGOS_NEIGHBOR_SPECS = {
    "ldaps50max": (
        "ldaps_ws50_max_speed",
        "ldaps_heightAboveGround_50_50MUmax",
        "ldaps_heightAboveGround_50_50MVmax",
    ),
    "gfs850": (
        "gfs_ws850_speed",
        "gfs_isobaricInhPa_850_u",
        "gfs_isobaricInhPa_850_v",
    ),
    "gfs100": (
        "gfs_ws100_speed",
        "gfs_heightAboveGround_100_100u",
        "gfs_heightAboveGround_100_100v",
    ),
}


def causal_flow_feature_names(base_cols=CAUSAL_FLOW_BASE_COLS):
    names = []
    for col in base_cols:
        names.extend(
            [
                f"{col}_causal_kernel3",
                f"{col}_causal_max3",
                f"{col}_delta1",
                f"{col}_delta2",
            ]
        )
    return names


def add_causal_flow_features(df, base_cols=CAUSAL_FLOW_BASE_COLS):
    out = df.sort_values("forecast_kst_dtm").copy()
    for col in base_cols:
        if col not in out.columns:
            continue
        current = out[col].astype(float)
        lag1 = current.shift(1).fillna(current)
        lag2 = current.shift(2).fillna(current)
        out[f"{col}_causal_kernel3"] = 0.60 * current + 0.30 * lag1 + 0.10 * lag2
        out[f"{col}_causal_max3"] = current.to_frame().assign(_lag1=lag1, _lag2=lag2).max(axis=1)
        out[f"{col}_delta1"] = current - lag1
        out[f"{col}_delta2"] = current - lag2
    return out.replace([float("inf"), float("-inf")], 0).ffill().fillna(0).reset_index(drop=True)


def _met_direction_deg(u, v):
    return (np.degrees(np.arctan2(-np.asarray(u, dtype=float), -np.asarray(v, dtype=float))) + 360.0) % 360.0


def _sector_from_deg(deg, n_sectors=12):
    width = 360.0 / float(n_sectors)
    return np.floor(((np.asarray(deg, dtype=float) + width / 2.0) % 360.0) / width).astype(int)


def leustagos_neighbor_feature_names(specs=LEUSTAGOS_NEIGHBOR_SPECS):
    names = []
    suffixes = [
        "",
        "_p1",
        "_p2",
        "_p3",
        "_n1",
        "_n2",
        "_n3",
        "_win7_mean",
        "_win7_max",
        "_win7_q65",
        "_win7_range",
    ]
    for name in specs:
        base = f"leu_{name}_ws_angle"
        names.extend([f"{base}{suffix}" for suffix in suffixes])
        names.extend([f"leu_{name}_sector12", f"leu_{name}_dir_sin", f"leu_{name}_dir_cos"])
    return names


def leustagos_power_neighbor_feature_names(specs=LEUSTAGOS_NEIGHBOR_SPECS):
    names = []
    suffixes = [
        "",
        "_p1",
        "_p2",
        "_p3",
        "_n1",
        "_n2",
        "_n3",
        "_win7_mean",
        "_win7_max",
        "_win7_q65",
        "_win7_range",
    ]
    for name in specs:
        base = f"leu_{name}_power"
        names.extend([f"{base}{suffix}" for suffix in suffixes])
    return names


def _shift_with_current(grouped, current, periods):
    shifted = grouped.shift(periods)
    return shifted.fillna(current)


def _rolling_transform(grouped, func):
    return grouped.transform(lambda s: func(s.rolling(window=7, center=True, min_periods=1)))


def add_leustagos_neighbor_features(df, specs=LEUSTAGOS_NEIGHBOR_SPECS):
    sort_cols = [col for col in ["data_available_kst_dtm", "forecast_kst_dtm"] if col in df.columns]
    out = df.sort_values(sort_cols).copy() if sort_cols else df.copy()
    group_key = "data_available_kst_dtm" if "data_available_kst_dtm" in out.columns else None

    for name, (speed_col, u_col, v_col) in specs.items():
        if speed_col not in out.columns or u_col not in out.columns or v_col not in out.columns:
            continue
        speed = out[speed_col].astype(float).clip(lower=0.0)
        direction = _met_direction_deg(out[u_col], out[v_col])
        sector = _sector_from_deg(direction, n_sectors=12)
        power = speed + speed**2 + speed**3
        # Leustagos-style scalar: wind polynomial modulated by direction sector.
        # Keep it deterministic; the tree learns the non-linear cuts.
        col = f"leu_{name}_ws_angle"
        out[col] = power * ((sector.astype(float) + 1.0) / 12.0)
        out[f"leu_{name}_sector12"] = sector.astype(float)
        out[f"leu_{name}_dir_sin"] = np.sin(np.radians(direction))
        out[f"leu_{name}_dir_cos"] = np.cos(np.radians(direction))

        current = out[col].astype(float)
        grouped = out.groupby(group_key, sort=False)[col] if group_key else None
        if grouped is not None:
            out[f"{col}_p1"] = _shift_with_current(grouped, current, 1)
            out[f"{col}_p2"] = _shift_with_current(grouped, current, 2)
            out[f"{col}_p3"] = _shift_with_current(grouped, current, 3)
            out[f"{col}_n1"] = _shift_with_current(grouped, current, -1)
            out[f"{col}_n2"] = _shift_with_current(grouped, current, -2)
            out[f"{col}_n3"] = _shift_with_current(grouped, current, -3)
            out[f"{col}_win7_mean"] = _rolling_transform(grouped, lambda r: r.mean())
            out[f"{col}_win7_max"] = _rolling_transform(grouped, lambda r: r.max())
            out[f"{col}_win7_q65"] = _rolling_transform(grouped, lambda r: r.quantile(0.65))
        else:
            out[f"{col}_p1"] = current.shift(1).fillna(current)
            out[f"{col}_p2"] = current.shift(2).fillna(current)
            out[f"{col}_p3"] = current.shift(3).fillna(current)
            out[f"{col}_n1"] = current.shift(-1).fillna(current)
            out[f"{col}_n2"] = current.shift(-2).fillna(current)
            out[f"{col}_n3"] = current.shift(-3).fillna(current)
            roll7 = current.rolling(window=7, center=True, min_periods=1)
            out[f"{col}_win7_mean"] = roll7.mean()
            out[f"{col}_win7_max"] = roll7.max()
            out[f"{col}_win7_q65"] = roll7.quantile(0.65)
        out[f"{col}_win7_range"] = out[f"{col}_win7_max"] - out[f"{col}_win7_mean"]

    return out.replace([float("inf"), float("-inf")], 0).ffill().fillna(0).reset_index(drop=True)


def add_leustagos_power_neighbor_features(df, specs=LEUSTAGOS_NEIGHBOR_SPECS):
    sort_cols = [col for col in ["data_available_kst_dtm", "forecast_kst_dtm"] if col in df.columns]
    out = df.sort_values(sort_cols).copy() if sort_cols else df.copy()
    group_key = "data_available_kst_dtm" if "data_available_kst_dtm" in out.columns else None

    for name, (speed_col, _, _) in specs.items():
        if speed_col not in out.columns:
            continue
        speed = out[speed_col].astype(float).clip(lower=0.0)
        col = f"leu_{name}_power"
        out[col] = speed + speed**2 + speed**3

        current = out[col].astype(float)
        grouped = out.groupby(group_key, sort=False)[col] if group_key else None
        if grouped is not None:
            out[f"{col}_p1"] = _shift_with_current(grouped, current, 1)
            out[f"{col}_p2"] = _shift_with_current(grouped, current, 2)
            out[f"{col}_p3"] = _shift_with_current(grouped, current, 3)
            out[f"{col}_n1"] = _shift_with_current(grouped, current, -1)
            out[f"{col}_n2"] = _shift_with_current(grouped, current, -2)
            out[f"{col}_n3"] = _shift_with_current(grouped, current, -3)
            out[f"{col}_win7_mean"] = _rolling_transform(grouped, lambda r: r.mean())
            out[f"{col}_win7_max"] = _rolling_transform(grouped, lambda r: r.max())
            out[f"{col}_win7_q65"] = _rolling_transform(grouped, lambda r: r.quantile(0.65))
        else:
            out[f"{col}_p1"] = current.shift(1).fillna(current)
            out[f"{col}_p2"] = current.shift(2).fillna(current)
            out[f"{col}_p3"] = current.shift(3).fillna(current)
            out[f"{col}_n1"] = current.shift(-1).fillna(current)
            out[f"{col}_n2"] = current.shift(-2).fillna(current)
            out[f"{col}_n3"] = current.shift(-3).fillna(current)
            roll7 = current.rolling(window=7, center=True, min_periods=1)
            out[f"{col}_win7_mean"] = roll7.mean()
            out[f"{col}_win7_max"] = roll7.max()
            out[f"{col}_win7_q65"] = roll7.quantile(0.65)
        out[f"{col}_win7_range"] = out[f"{col}_win7_max"] - out[f"{col}_win7_mean"]

    return out.replace([float("inf"), float("-inf")], 0).ffill().fillna(0).reset_index(drop=True)


def _build_full_v2(ldaps, gfs, group):
    base = build_weather_features(ldaps, gfs)
    meteo = build_meteo_features(ldaps, gfs)
    all_meteo = add_meteo_block(base, meteo, "all_meteo")
    return add_compact_physics_features(all_meteo, ldaps, gfs, group=group, include_advanced=True)


def select_aggressive_minimal_v1(weather):
    return _select_features(weather, AGGRESSIVE_MINIMAL_V1_FEATURES)


def select_aggressive_minimal_rolling_v1(weather):
    with_time = add_weather_time_features(weather)
    return _select_features(with_time, AGGRESSIVE_MINIMAL_V1_FEATURES + weather_time_feature_names())


def select_aggressive_minimal_rollmean_v1(weather):
    with_time = add_weather_time_features(weather)
    time_features = [
        name
        for name in weather_time_feature_names()
        if name.endswith("_lead1") or name.endswith("_lead3") or name.endswith("_roll3_mean")
    ]
    return _select_features(with_time, AGGRESSIVE_MINIMAL_V1_FEATURES + time_features)


def select_aggressive_minimal_context_v1(weather, ldaps, gfs):
    with_time = add_weather_time_features(weather)
    with_context = add_tree_context_features(with_time, ldaps, gfs)
    feature_names = AGGRESSIVE_MINIMAL_V1_FEATURES + weather_time_feature_names() + tree_context_feature_names()
    return _select_features(with_context, feature_names)


def select_wind_cubic_minimal_v1(weather):
    with_time = add_weather_time_features(weather, base_cols=["ldaps_ws50_max_speed"])
    return _select_features(with_time, WIND_CUBIC_MINIMAL_V1_FEATURES)


def select_importance_lean_v1(weather):
    with_time = add_weather_time_features(weather, base_cols=["phys_ldaps_ws50max_grid_max", "gfs_ws850_speed"])
    return _select_features(with_time, IMPORTANCE_LEAN_V1_FEATURES)


def select_group_top40_v1(weather, group):
    with_time = add_weather_time_features(weather)
    return _select_features(with_time, GROUP_TOP40_V1_FEATURES[group])


def select_group_family_quota65_v1(weather, group):
    with_time = add_weather_time_features(weather)
    return _select_features(with_time, GROUP_FAMILY_QUOTA65_V1_FEATURES[group])


def select_group_family_quota65_causalflow_v1(weather, group):
    with_time = add_weather_time_features(weather)
    with_flow = add_causal_flow_features(with_time)
    feature_names = GROUP_FAMILY_QUOTA65_V1_FEATURES[group] + causal_flow_feature_names()
    return _select_features(with_flow, feature_names)


def select_group_family_quota65_leustagos_tpm3_v1(weather, group):
    with_time = add_weather_time_features(weather)
    with_leustagos = add_leustagos_neighbor_features(with_time)
    feature_names = GROUP_FAMILY_QUOTA65_V1_FEATURES[group] + leustagos_neighbor_feature_names()
    return _select_features(with_leustagos, feature_names)


def select_group_family_quota65_leustagos_power_tpm3_v1(weather, group):
    with_time = add_weather_time_features(weather)
    with_leustagos = add_leustagos_power_neighbor_features(with_time)
    feature_names = GROUP_FAMILY_QUOTA65_V1_FEATURES[group] + leustagos_power_neighbor_feature_names()
    return _select_features(with_leustagos, feature_names)


def _select_features(weather, feature_names):
    keep_cols = [col for col in TIME_KEY_COLS if col in weather.columns]
    keep_cols.extend([col for col in feature_names if col in weather.columns])
    seen = set()
    keep_cols = [col for col in keep_cols if not (col in seen or seen.add(col))]
    return weather[keep_cols].copy()


def build_tree_features(ldaps, gfs, group, feature_profile=FEATURE_PROFILE_FULL_V2):
    if feature_profile not in FEATURE_PROFILES:
        raise ValueError(f"unknown feature_profile: {feature_profile}")
    full = _build_full_v2(ldaps, gfs, group)
    if feature_profile == FEATURE_PROFILE_FULL_V2:
        return full
    if feature_profile == FEATURE_PROFILE_AGGRESSIVE_MINIMAL_V1:
        return select_aggressive_minimal_v1(full)
    if feature_profile == FEATURE_PROFILE_AGGRESSIVE_MINIMAL_ROLLING_V1:
        return select_aggressive_minimal_rolling_v1(full)
    if feature_profile == FEATURE_PROFILE_AGGRESSIVE_MINIMAL_ROLLMEAN_V1:
        return select_aggressive_minimal_rollmean_v1(full)
    if feature_profile == FEATURE_PROFILE_WIND_CUBIC_MINIMAL_V1:
        return select_wind_cubic_minimal_v1(full)
    if feature_profile == FEATURE_PROFILE_IMPORTANCE_LEAN_V1:
        return select_importance_lean_v1(full)
    if feature_profile == FEATURE_PROFILE_GROUP_TOP40_V1:
        return select_group_top40_v1(full, group)
    if feature_profile == FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_V1:
        return select_group_family_quota65_v1(full, group)
    if feature_profile == FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_CAUSALFLOW_V1:
        return select_group_family_quota65_causalflow_v1(full, group)
    if feature_profile == FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_LEUSTAGOS_TPM3_V1:
        return select_group_family_quota65_leustagos_tpm3_v1(full, group)
    if feature_profile == FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_LEUSTAGOS_POWER_TPM3_V1:
        return select_group_family_quota65_leustagos_power_tpm3_v1(full, group)
    return select_aggressive_minimal_context_v1(full, ldaps, gfs)
