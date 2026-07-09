from utils.compact_physics_features import add_compact_physics_features
from utils.meteo_features import add_meteo_block, build_meteo_features
from utils.preprocessing import TIME_KEY_COLS, build_weather_features
from utils.tree_context_features import add_tree_context_features, tree_context_feature_names
from utils.weather_time_features import add_weather_time_features, weather_time_feature_names


FEATURE_PROFILE_FULL_V2 = "full_v2"
FEATURE_PROFILE_AGGRESSIVE_MINIMAL_V1 = "aggressive_minimal_v1"
FEATURE_PROFILE_AGGRESSIVE_MINIMAL_ROLLING_V1 = "aggressive_minimal_rolling_v1"
FEATURE_PROFILE_AGGRESSIVE_MINIMAL_CONTEXT_V1 = "aggressive_minimal_context_v1"
FEATURE_PROFILES = [
    FEATURE_PROFILE_FULL_V2,
    FEATURE_PROFILE_AGGRESSIVE_MINIMAL_V1,
    FEATURE_PROFILE_AGGRESSIVE_MINIMAL_ROLLING_V1,
    FEATURE_PROFILE_AGGRESSIVE_MINIMAL_CONTEXT_V1,
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


def select_aggressive_minimal_context_v1(weather, ldaps, gfs):
    with_time = add_weather_time_features(weather)
    with_context = add_tree_context_features(with_time, ldaps, gfs)
    feature_names = AGGRESSIVE_MINIMAL_V1_FEATURES + weather_time_feature_names() + tree_context_feature_names()
    return _select_features(with_context, feature_names)


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
    return select_aggressive_minimal_context_v1(full, ldaps, gfs)
