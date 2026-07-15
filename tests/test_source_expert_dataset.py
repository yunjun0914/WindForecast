from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utils.source_expert_dataset import (
    EXPECTED_LEADS,
    GEFS_FHOURS,
    GFS_CORE_SPEC,
    GFS_SURFACE_PRESSURE_SPEC,
    LDAPS_CORE_SPEC,
    LDAPS_SURFACE_PRESSURE_SPEC,
    apply_gefs_publication_fallback,
    build_gefs_mean_core_tensor,
    build_grid_source_core_tensor,
    fit_source_channel_scaler,
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
                        }
                    )
    return pd.DataFrame(pressure_rows), pd.DataFrame(gust_rows)


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
