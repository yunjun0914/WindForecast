from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utils.source_expert_dataset import (
    EXPECTED_LEADS,
    GEFS_FHOURS,
    GFS_CORE_SPEC,
    GFS_10M_CORE_SPEC,
    GFS_SURFACE_REGIME_CHANNELS,
    GFS_SURFACE_REGIME_SPEC,
    GFS_SURFACE_PRESSURE_SPEC,
    GFS_THERMO_SYNOPTIC_CHANNELS,
    GFS_THERMO_SYNOPTIC_SPEC,
    GFS_VERTICAL_THERMO_SPEC,
    GFS_VERTICAL_WIND_EXTRA_SCALARS,
    GFS_VERTICAL_WIND_EXTRA_VECTORS,
    GFS_VERTICAL_WIND_SPEC,
    GridSourceSpec,
    LDAPS_CORE_SPEC,
    LDAPS_DERIVED_FAMILY_CHANNELS,
    LDAPS_5M_CORE_SPEC,
    LDAPS_BLH_COLUMN,
    LDAPS_BLH_FLOOR_M,
    LDAPS_BLH_RATIO_SPEC,
    LDAPS_HUB_OVER_BLH_CHANNEL,
    LDAPS_MSLP_COLUMN,
    LDAPS_MSLP_SPEC,
    LDAPS_PRESSURE_TENDENCY_CHANNEL,
    LDAPS_PRESSURE_TENDENCY_SPEC,
    LDAPS_SURFACE_PRESSURE_COLUMN,
    LDAPS_SURFACE_PRESSURE_SPEC,
    LDAPS_SURFACE_REGIME_CHANNELS,
    LDAPS_SURFACE_REGIME_SPEC,
    LDAPS_THERMO_PBL_CHANNELS,
    LDAPS_THERMO_PBL_SPEC,
    TURBINE_HUB_HEIGHT_M,
    apply_gefs_publication_fallback,
    build_gefs_mean_core_tensor,
    build_gefs_mean700_core_tensor,
    build_gefs_near_spread_core_tensor,
    build_gefs_spread_core_tensor,
    build_gefs_upper_spread_core_tensor,
    build_grid_source_core_tensor,
    build_ldaps_blh_ratio_tensor,
    build_ldaps_derived_family_tensor,
    build_ldaps_pressure_tendency_tensor,
    fit_source_channel_scaler,
    ldaps_derived_family_required_columns,
    select_gefs_issues,
    transform_source_channels,
)


def make_grid_frame(spec, issue_count=2):
    rows = []
    for issue_index in range(issue_count):
        issue_time = pd.Timestamp("2022-01-01 13:00:00") + pd.Timedelta(
            days=issue_index
        )
        for lead in EXPECTED_LEADS:
            forecast_time = issue_time + pd.Timedelta(hours=int(lead))
            for grid_id in sorted(spec.layout):
                row = {
                    "forecast_kst_dtm": forecast_time,
                    "data_available_kst_dtm": issue_time,
                    "grid_id": grid_id,
                }
                for channel_index, channel in enumerate(spec.raw_channels):
                    row[channel] = (
                        100.0 * issue_index
                        + 10.0 * channel_index
                        + float(lead)
                        + grid_id / 100.0
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def make_gefs_frames(run_count=2):
    pressure_rows = []
    gust_rows = []
    pressure_lats = np.arange(33.0, 41.01, 0.5)
    pressure_lons = np.arange(124.0, 132.01, 0.5)
    gust_lats = np.arange(35.25, 39.251, 0.25)
    gust_lons = np.arange(127.0, 131.001, 0.25)
    pressure_channels = (
        "u10m_mean",
        "v10m_mean",
        "u925_mean",
        "v925_mean",
        "u850_mean",
        "v850_mean",
        "u700_mean",
        "v700_mean",
    )
    pressure_spread_channels = (
        "u10m_sprd",
        "v10m_sprd",
        "u925_sprd",
        "v925_sprd",
        "u850_sprd",
        "v850_sprd",
        "u700_sprd",
        "v700_sprd",
    )
    for run_index in range(run_count):
        run_date = pd.Timestamp("2021-12-30") + pd.Timedelta(days=run_index)
        for fhour in GEFS_FHOURS:
            source_lead = float(fhour - 10)
            for lat in pressure_lats:
                for lon in pressure_lons:
                    row = {
                        "run_date": run_date,
                        "fhour": int(fhour),
                        "lat": float(lat),
                        "lon": float(lon),
                    }
                    for channel_index, channel in enumerate(pressure_channels):
                        row[channel] = (
                            100.0 * run_index
                            + 10.0 * channel_index
                            + source_lead
                            + 0.01 * lat
                            + 0.001 * lon
                        )
                    for channel_index, channel in enumerate(pressure_spread_channels):
                        row[channel] = (
                            10.0
                            + channel_index
                            + 0.1 * run_index
                            + 0.01 * source_lead
                            + 0.0001 * lat
                        )
                    pressure_rows.append(row)
            for lat in gust_lats:
                for lon in gust_lons:
                    gust_rows.append(
                        {
                            "run_date": run_date,
                            "fhour": int(fhour),
                            "lat": float(lat),
                            "lon": float(lon),
                            "gust_mean": (
                                100.0 * run_index
                                + source_lead
                                + 0.01 * lat
                                + 0.001 * lon
                            ),
                            "gust_sprd": (
                                5.0
                                + 0.1 * run_index
                                + 0.01 * source_lead
                                + 0.0001 * lat
                            ),
                        }
                    )
    return pd.DataFrame(pressure_rows), pd.DataFrame(gust_rows)


def make_ldaps_derived_frame(family):
    required = ldaps_derived_family_required_columns(family)
    raw_channels = tuple(
        channel
        for channel in required
        if channel not in {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id"}
    )
    spec = GridSourceSpec(
        name=f"ldaps_{family}_test_input",
        layout=LDAPS_CORE_SPEC.layout,
        vectors=(),
        scalar_channels=raw_channels,
    )
    frame = make_grid_frame(spec)
    replacements = {
        "heightAboveGround_5_XBLWS": 2.0,
        "heightAboveGround_5_YBLWS": 1.0,
        "heightAboveGround_2_t": 280.0,
        "heightAboveGround_2_dpt": 275.0,
        "heightAboveGround_2_r": 70.0,
        "heightAboveGround_2_q": 0.005,
        "surface_0_sp": 90000.0,
        "meanSea_0_prmsl": 101000.0,
        "etc_0_blh": 300.0,
        "surface_0_NDNSW": 100.0,
        "surface_0_NDNLW": -30.0,
        "heightAboveGround_2_SWDIR": 200.0,
        "heightAboveGround_2_SWDIF": 50.0,
        "etc_0_hcc": 20.0,
        "etc_0_mcc": 30.0,
        "etc_0_lcc": 40.0,
        "etc_0_VLCDC": 10.0,
        "surface_0_avg_lsprate": 0.1,
        "surface_0_lssrate": 0.2,
        "surface_0_ncpcp": 0.3,
        "surface_0_snol": 0.1,
        "surface_0_SNOM": 0.2,
    }
    for channel, replacement in replacements.items():
        if channel in frame.columns:
            frame[channel] = replacement
    if "surface_0_h" in frame.columns:
        frame["surface_0_h"] = 850.0 + 5.0 * frame["grid_id"]
    return frame


class SourceExpertDatasetTest(unittest.TestCase):
    def test_ldaps_core_preserves_grid_and_interpolates_within_issue(self):
        frame = make_grid_frame(LDAPS_CORE_SPEC)
        issue_time = pd.Timestamp("2022-01-01 13:00:00")
        missing_time = issue_time + pd.Timedelta(hours=20)
        missing_column = LDAPS_CORE_SPEC.vectors[0].u
        mask = (
            frame["data_available_kst_dtm"].eq(issue_time)
            & frame["forecast_kst_dtm"].eq(missing_time)
            & frame["grid_id"].eq(1)
        )
        frame.loc[mask, missing_column] = np.nan

        tensor = build_grid_source_core_tensor(frame, LDAPS_CORE_SPEC)

        self.assertEqual(tensor.values.shape, (2, 24, 9, 4, 5))
        self.assertEqual(int(tensor.spatial_mask.sum()), 16)
        self.assertEqual(tensor.channel_names, LDAPS_CORE_SPEC.output_channels)
        channel_index = tensor.channel_names.index(missing_column)
        speed_index = tensor.channel_names.index("wind_50m_max_speed")
        lead_index = int(np.where(EXPECTED_LEADS == 20)[0][0])
        row, column = LDAPS_CORE_SPEC.layout[1]
        self.assertTrue(tensor.missing_mask[0, lead_index, channel_index, row, column])
        self.assertTrue(tensor.missing_mask[0, lead_index, speed_index, row, column])
        self.assertAlmostEqual(
            float(tensor.values[0, lead_index, channel_index, row, column]),
            20.01,
            places=5,
        )

    def test_ldaps_5m_core_adds_only_one_vector_and_speed(self):
        core = build_grid_source_core_tensor(
            make_grid_frame(LDAPS_CORE_SPEC),
            LDAPS_CORE_SPEC,
        )
        five_meter = build_grid_source_core_tensor(
            make_grid_frame(LDAPS_5M_CORE_SPEC),
            LDAPS_5M_CORE_SPEC,
        )

        self.assertEqual(five_meter.values.shape, (2, 24, 12, 4, 5))
        self.assertEqual(int(five_meter.spatial_mask.sum()), 16)
        self.assertEqual(
            set(five_meter.channel_names) - set(core.channel_names),
            {
                "heightAboveGround_5_XBLWS",
                "heightAboveGround_5_YBLWS",
                "wind_5m_speed",
            },
        )

    def test_ldaps_blh_ratio_adds_only_the_clipped_hub_height_ratio(self):
        frame = make_grid_frame(LDAPS_CORE_SPEC)
        frame[LDAPS_BLH_COLUMN] = 117.0
        frame.loc[0, LDAPS_BLH_COLUMN] = 58.5
        frame.loc[1, LDAPS_BLH_COLUMN] = 10.0

        tensor = build_ldaps_blh_ratio_tensor(frame)

        self.assertEqual(tensor.values.shape, (2, 24, 10, 4, 5))
        self.assertEqual(tensor.channel_names, LDAPS_BLH_RATIO_SPEC.output_channels)
        self.assertNotIn(LDAPS_BLH_COLUMN, tensor.channel_names)
        self.assertEqual(
            set(tensor.channel_names) - set(LDAPS_CORE_SPEC.output_channels),
            {LDAPS_HUB_OVER_BLH_CHANNEL},
        )
        channel_index = tensor.channel_names.index(LDAPS_HUB_OVER_BLH_CHANNEL)
        first_row, first_column = LDAPS_CORE_SPEC.layout[1]
        second_row, second_column = LDAPS_CORE_SPEC.layout[2]
        self.assertAlmostEqual(
            float(tensor.values[0, 0, channel_index, first_row, first_column]),
            TURBINE_HUB_HEIGHT_M / 58.5,
            places=6,
        )
        self.assertAlmostEqual(
            float(tensor.values[0, 0, channel_index, second_row, second_column]),
            TURBINE_HUB_HEIGHT_M / LDAPS_BLH_FLOOR_M,
            places=6,
        )

    def test_ldaps_pressure_tendency_replaces_raw_pressure_with_one_slope(self):
        tensor = build_ldaps_pressure_tendency_tensor(
            make_grid_frame(LDAPS_SURFACE_PRESSURE_SPEC)
        )

        self.assertEqual(tensor.values.shape, (2, 24, 10, 4, 5))
        self.assertEqual(
            tensor.channel_names,
            LDAPS_PRESSURE_TENDENCY_SPEC.output_channels,
        )
        self.assertNotIn(LDAPS_SURFACE_PRESSURE_COLUMN, tensor.channel_names)
        self.assertEqual(
            set(tensor.channel_names) - set(LDAPS_CORE_SPEC.output_channels),
            {LDAPS_PRESSURE_TENDENCY_CHANNEL},
        )
        tendency_index = tensor.channel_names.index(LDAPS_PRESSURE_TENDENCY_CHANNEL)
        tendency = tensor.values[:, :, tendency_index][:, :, tensor.spatial_mask]
        self.assertTrue(np.allclose(tendency, 1.0))

    def test_ldaps_mslp_adds_exactly_one_raw_scalar_channel(self):
        tensor = build_grid_source_core_tensor(
            make_grid_frame(LDAPS_MSLP_SPEC),
            LDAPS_MSLP_SPEC,
        )

        self.assertEqual(tensor.values.shape, (2, 24, 10, 4, 5))
        self.assertEqual(tensor.channel_names, LDAPS_MSLP_SPEC.output_channels)
        self.assertEqual(
            set(tensor.channel_names) - set(LDAPS_CORE_SPEC.output_channels),
            {LDAPS_MSLP_COLUMN},
        )

    def test_ldaps_raw_families_add_only_the_audited_scalar_channels(self):
        thermo = build_grid_source_core_tensor(
            make_grid_frame(LDAPS_THERMO_PBL_SPEC),
            LDAPS_THERMO_PBL_SPEC,
        )
        surface = build_grid_source_core_tensor(
            make_grid_frame(LDAPS_SURFACE_REGIME_SPEC),
            LDAPS_SURFACE_REGIME_SPEC,
        )

        self.assertEqual(thermo.values.shape, (2, 24, 14, 4, 5))
        self.assertEqual(surface.values.shape, (2, 24, 23, 4, 5))
        self.assertEqual(
            set(thermo.channel_names) - set(LDAPS_CORE_SPEC.output_channels),
            set(LDAPS_THERMO_PBL_CHANNELS),
        )
        self.assertEqual(
            set(surface.channel_names) - set(LDAPS_CORE_SPEC.output_channels),
            set(LDAPS_SURFACE_REGIME_CHANNELS),
        )
        self.assertNotIn("surface_0_lsm", surface.channel_names)

    def test_ldaps_derived_families_append_only_declared_channels(self):
        for family, expected_channels in LDAPS_DERIVED_FAMILY_CHANNELS.items():
            with self.subTest(family=family):
                tensor = build_ldaps_derived_family_tensor(
                    make_ldaps_derived_frame(family),
                    family,
                )
                self.assertEqual(
                    tensor.values.shape,
                    (2, 24, 9 + len(expected_channels), 4, 5),
                )
                self.assertEqual(
                    tensor.channel_names,
                    (*LDAPS_CORE_SPEC.output_channels, *expected_channels),
                )
                self.assertEqual(tensor.source, f"ldaps_{family}_derived_core")
                self.assertTrue(np.isfinite(tensor.values).all())
                self.assertTrue(np.all(tensor.values[..., ~tensor.spatial_mask] == 0.0))
                raw_extras = set(ldaps_derived_family_required_columns(family)) - set(
                    LDAPS_CORE_SPEC.raw_channels
                ) - {"forecast_kst_dtm", "data_available_kst_dtm", "grid_id"}
                self.assertTrue(raw_extras.isdisjoint(tensor.channel_names))

    def test_ldaps_envelope_uses_component_width_not_speed_subtraction(self):
        tensor = build_ldaps_derived_family_tensor(
            make_ldaps_derived_frame("envelope"),
            "envelope",
        )
        norm_index = tensor.channel_names.index("ldaps_50m_envelope_norm")
        relative_index = tensor.channel_names.index("ldaps_50m_relative_envelope")
        envelope_norm = tensor.values[:, :, norm_index][:, :, tensor.spatial_mask]
        relative = tensor.values[:, :, relative_index][:, :, tensor.spatial_mask]
        self.assertTrue(np.all(envelope_norm >= 0.0))
        self.assertTrue(np.isfinite(relative).all())

    def test_ldaps_density_spatial_is_the_exact_24_channel_union(self):
        combined = build_ldaps_derived_family_tensor(
            make_ldaps_derived_frame("density_spatial"),
            "density_spatial",
        )
        expected_additions = (
            *LDAPS_DERIVED_FAMILY_CHANNELS["density_power"],
            *LDAPS_DERIVED_FAMILY_CHANNELS["spatial_flow"],
        )
        self.assertEqual(combined.values.shape, (2, 24, 24, 4, 5))
        self.assertEqual(
            combined.channel_names,
            (*LDAPS_CORE_SPEC.output_channels, *expected_additions),
        )
        self.assertEqual(len(set(combined.channel_names)), 24)

    def test_ldaps_representation_family_channel_counts_and_bounds(self):
        expected_additions = {
            "circular_direction": 4,
            "polynomial_basis": 18,
            "pairwise_basis": 36,
            "spatial_centroid": 4,
            "time_modulation": 8,
        }
        for family, additions in expected_additions.items():
            with self.subTest(family=family):
                tensor = build_ldaps_derived_family_tensor(
                    make_ldaps_derived_frame(family),
                    family,
                )
                self.assertEqual(len(tensor.channel_names), 9 + additions)
                self.assertEqual(len(set(tensor.channel_names)), 9 + additions)

        circular = build_ldaps_derived_family_tensor(
            make_ldaps_derived_frame("circular_direction"),
            "circular_direction",
        )
        circular_values = circular.values[:, :, 9:][..., circular.spatial_mask]
        self.assertLessEqual(float(np.abs(circular_values).max()), 1.0 + 1e-6)

        centroid = build_ldaps_derived_family_tensor(
            make_ldaps_derived_frame("spatial_centroid"),
            "spatial_centroid",
        )
        centroid_values = centroid.values[:, :, 9:][..., centroid.spatial_mask]
        self.assertLessEqual(float(np.abs(centroid_values).max()), 1.0 + 1e-6)

    def test_gfs_core_has_only_approved_channels(self):
        tensor = build_grid_source_core_tensor(make_grid_frame(GFS_CORE_SPEC), GFS_CORE_SPEC)

        self.assertEqual(tensor.values.shape, (2, 24, 7, 3, 3))
        self.assertEqual(
            tensor.channel_names,
            (
                "heightAboveGround_80_u",
                "heightAboveGround_80_v",
                "heightAboveGround_100_100u",
                "heightAboveGround_100_100v",
                "surface_0_gust",
                "wind_80m_speed",
                "wind_100m_speed",
            ),
        )

    def test_gfs_10m_core_adds_only_one_vector_and_speed(self):
        core = build_grid_source_core_tensor(make_grid_frame(GFS_CORE_SPEC), GFS_CORE_SPEC)
        ten_meter = build_grid_source_core_tensor(
            make_grid_frame(GFS_10M_CORE_SPEC),
            GFS_10M_CORE_SPEC,
        )

        self.assertEqual(ten_meter.values.shape, (2, 24, 10, 3, 3))
        self.assertEqual(
            set(ten_meter.channel_names) - set(core.channel_names),
            {
                "heightAboveGround_10_10u",
                "heightAboveGround_10_10v",
                "wind_10m_speed",
            },
        )

    def test_gfs_raw_families_extend_the_accepted_10m_core(self):
        vertical = build_grid_source_core_tensor(
            make_grid_frame(GFS_VERTICAL_WIND_SPEC),
            GFS_VERTICAL_WIND_SPEC,
        )
        thermo = build_grid_source_core_tensor(
            make_grid_frame(GFS_THERMO_SYNOPTIC_SPEC),
            GFS_THERMO_SYNOPTIC_SPEC,
        )
        surface = build_grid_source_core_tensor(
            make_grid_frame(GFS_SURFACE_REGIME_SPEC),
            GFS_SURFACE_REGIME_SPEC,
        )

        self.assertEqual(vertical.values.shape, (2, 24, 23, 3, 3))
        self.assertEqual(thermo.values.shape, (2, 24, 21, 3, 3))
        self.assertEqual(surface.values.shape, (2, 24, 18, 3, 3))
        vertical_channels = {
            channel
            for vector in GFS_VERTICAL_WIND_EXTRA_VECTORS
            for channel in (vector.u, vector.v, f"{vector.name}_speed")
        } | set(GFS_VERTICAL_WIND_EXTRA_SCALARS)
        self.assertEqual(
            set(vertical.channel_names) - set(GFS_10M_CORE_SPEC.output_channels),
            vertical_channels,
        )
        self.assertEqual(
            set(thermo.channel_names) - set(GFS_10M_CORE_SPEC.output_channels),
            set(GFS_THERMO_SYNOPTIC_CHANNELS),
        )
        self.assertEqual(
            set(surface.channel_names) - set(GFS_10M_CORE_SPEC.output_channels),
            set(GFS_SURFACE_REGIME_CHANNELS),
        )

    def test_gfs_vertical_thermo_combined_is_the_exact_family_union(self):
        combined = build_grid_source_core_tensor(
            make_grid_frame(GFS_VERTICAL_THERMO_SPEC),
            GFS_VERTICAL_THERMO_SPEC,
        )

        vertical_channels = {
            channel
            for vector in GFS_VERTICAL_WIND_EXTRA_VECTORS
            for channel in (vector.u, vector.v, f"{vector.name}_speed")
        } | set(GFS_VERTICAL_WIND_EXTRA_SCALARS)
        expected_additions = vertical_channels | set(GFS_THERMO_SYNOPTIC_CHANNELS)
        actual_additions = set(combined.channel_names) - set(
            GFS_10M_CORE_SPEC.output_channels
        )
        self.assertEqual(combined.values.shape, (2, 24, 34, 3, 3))
        self.assertEqual(actual_additions, expected_additions)
        self.assertTrue(actual_additions.isdisjoint(GFS_SURFACE_REGIME_CHANNELS))

    def test_surface_pressure_variants_add_exactly_one_scalar_channel(self):
        ldaps = build_grid_source_core_tensor(
            make_grid_frame(LDAPS_SURFACE_PRESSURE_SPEC),
            LDAPS_SURFACE_PRESSURE_SPEC,
        )
        gfs = build_grid_source_core_tensor(
            make_grid_frame(GFS_SURFACE_PRESSURE_SPEC),
            GFS_SURFACE_PRESSURE_SPEC,
        )
        self.assertEqual(ldaps.values.shape, (2, 24, 10, 4, 5))
        self.assertEqual(gfs.values.shape, (2, 24, 8, 3, 3))
        self.assertEqual(ldaps.channel_names.count("surface_0_sp"), 1)
        self.assertEqual(gfs.channel_names.count("surface_0_sp"), 1)
        self.assertEqual(
            set(ldaps.channel_names) - set(LDAPS_CORE_SPEC.output_channels),
            {"surface_0_sp"},
        )
        self.assertEqual(
            set(gfs.channel_names) - set(GFS_CORE_SPEC.output_channels),
            {"surface_0_sp"},
        )

    def test_scaler_uses_selected_issues_and_shared_channel_scale(self):
        tensor = build_grid_source_core_tensor(make_grid_frame(GFS_CORE_SPEC), GFS_CORE_SPEC)
        scaler = fit_source_channel_scaler(tensor, issue_mask=np.array([True, False]))
        transformed = transform_source_channels(tensor, scaler)

        self.assertEqual(scaler.channel_names, tensor.channel_names)
        self.assertEqual(scaler.mean.shape, (len(tensor.channel_names),))
        self.assertTrue(np.isfinite(transformed).all())
        spatial_values = transformed[0][..., tensor.spatial_mask]
        self.assertAlmostEqual(float(spatial_values[:, 0].mean()), 0.0, places=5)
        self.assertTrue(np.all(transformed[..., ~tensor.spatial_mask] == 0.0))

    def test_gefs_mean_core_keeps_pressure_and_gust_grids_separate(self):
        pressure, gust = make_gefs_frames()
        tensor = build_gefs_mean_core_tensor(pressure, gust)

        self.assertEqual(tensor.pressure.values.shape, (2, 24, 9, 7, 7))
        self.assertEqual(tensor.gust.values.shape, (2, 24, 1, 9, 9))
        self.assertEqual(len(tensor.pressure.latitudes), 7)
        self.assertEqual(len(tensor.gust.latitudes), 9)
        self.assertIn("wind_925_speed", tensor.pressure.channel_names)
        self.assertNotIn("u700_mean", tensor.pressure.channel_names)

        target_index = 0
        channel_index = tensor.pressure.channel_names.index("u10m_mean")
        lat = float(tensor.pressure.latitudes[0])
        lon = float(tensor.pressure.longitudes[0])
        expected = 12.0 + 0.01 * lat + 0.001 * lon
        self.assertAlmostEqual(
            float(tensor.pressure.values[0, target_index, channel_index, 0, 0]),
            expected,
            places=4,
        )

    def test_gefs_spread_core_adds_only_seven_raw_spread_channels(self):
        pressure, gust = make_gefs_frames()
        mean = build_gefs_mean_core_tensor(pressure, gust)
        spread = build_gefs_spread_core_tensor(pressure, gust)

        self.assertEqual(spread.pressure.values.shape, (2, 24, 15, 7, 7))
        self.assertEqual(spread.gust.values.shape, (2, 24, 2, 9, 9))
        self.assertEqual(
            set(spread.pressure.channel_names) - set(mean.pressure.channel_names),
            {
                "u10m_sprd",
                "v10m_sprd",
                "u925_sprd",
                "v925_sprd",
                "u850_sprd",
                "v850_sprd",
            },
        )
        self.assertEqual(
            set(spread.gust.channel_names) - set(mean.gust.channel_names),
            {"gust_sprd"},
        )
        self.assertFalse(any("relative" in name for name in spread.pressure.channel_names))
        self.assertFalse(any("speed" in name for name in spread.pressure.channel_names[6:12]))

    def test_gefs_mean700_adds_only_the_700_mean_vector_and_speed(self):
        pressure, gust = make_gefs_frames()
        mean = build_gefs_mean_core_tensor(pressure, gust)
        mean700 = build_gefs_mean700_core_tensor(pressure, gust)

        self.assertEqual(mean700.pressure.values.shape, (2, 24, 12, 7, 7))
        self.assertEqual(mean700.gust.values.shape, (2, 24, 1, 9, 9))
        self.assertEqual(
            set(mean700.pressure.channel_names) - set(mean.pressure.channel_names),
            {"u700_mean", "v700_mean", "wind_700_speed"},
        )

    def test_gefs_near_spread_keeps_only_surface_uncertainty_family(self):
        pressure, gust = make_gefs_frames()
        mean = build_gefs_mean_core_tensor(pressure, gust)
        near = build_gefs_near_spread_core_tensor(pressure, gust)

        self.assertEqual(near.pressure.values.shape, (2, 24, 15, 7, 7))
        self.assertEqual(near.gust.values.shape, (2, 24, 2, 9, 9))
        self.assertEqual(
            set(near.pressure.channel_names) - set(mean.pressure.channel_names),
            {
                "u10m_sprd",
                "v10m_sprd",
                "u925_sprd",
                "v925_sprd",
                "ensemble_spread_10m_speed",
                "ensemble_spread_925_speed",
            },
        )
        self.assertEqual(
            set(near.gust.channel_names) - set(mean.gust.channel_names),
            {"gust_sprd"},
        )
        self.assertNotIn("u850_sprd", near.pressure.channel_names)
        self.assertNotIn("u700_sprd", near.pressure.channel_names)

    def test_gefs_upper_spread_keeps_only_850_and_700_uncertainty(self):
        pressure, gust = make_gefs_frames()
        mean = build_gefs_mean_core_tensor(pressure, gust)
        upper = build_gefs_upper_spread_core_tensor(pressure, gust)

        self.assertEqual(upper.pressure.values.shape, (2, 24, 15, 7, 7))
        self.assertEqual(upper.gust.values.shape, (2, 24, 1, 9, 9))
        self.assertEqual(
            set(upper.pressure.channel_names) - set(mean.pressure.channel_names),
            {
                "u850_sprd",
                "v850_sprd",
                "u700_sprd",
                "v700_sprd",
                "ensemble_spread_850_speed",
                "ensemble_spread_700_speed",
            },
        )
        self.assertNotIn("u10m_sprd", upper.pressure.channel_names)
        self.assertNotIn("u925_sprd", upper.pressure.channel_names)
        self.assertNotIn("gust_sprd", upper.gust.channel_names)

    def test_gefs_publication_fallback_copies_prior_safe_issue(self):
        pressure, gust = make_gefs_frames()
        tensor = build_gefs_mean_core_tensor(pressure, gust)
        publication = pd.DataFrame(
            {
                "data_available_kst_dtm": pd.to_datetime(tensor.pressure.issue_times),
                "safe": [True, False],
            }
        )

        fallback = apply_gefs_publication_fallback(tensor, publication)

        self.assertFalse(fallback.pressure.fallback_flags[0])
        self.assertTrue(fallback.pressure.fallback_flags[1])
        np.testing.assert_allclose(
            fallback.pressure.values[1], fallback.pressure.values[0]
        )
        np.testing.assert_allclose(fallback.gust.values[1], fallback.gust.values[0])

        selected = select_gefs_issues(fallback, [fallback.pressure.issue_times[1]])
        self.assertEqual(selected.pressure.values.shape[0], 1)
        self.assertTrue(selected.pressure.fallback_flags[0])


if __name__ == "__main__":
    unittest.main()
