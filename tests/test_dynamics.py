import unittest

import numpy as np
import pandas as pd

from src.evaluate import build_window
from src.features import ARCS
from src.model import feature_columns, multi_horizon_training_features


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2025-12-01", "2026-02-06 23:00", freq="h")
        self.data = pd.MultiIndex.from_product([ARCS, times], names=["arc", "t_1h"]).to_frame(index=False)
        self.data["q"] = 100.0
        self.data["k"] = 10.0
        self.data["is_imputed_q"] = 0
        self.data["is_imputed_k"] = 0
        for date, value in [("2026-01-25 12:00", 140.), ("2026-01-18 12:00", 120.), ("2026-01-11 12:00", 100.)]:
            self.data.loc[(self.data.arc == "Convention") & (self.data.t_1h == date), "q"] = value
        self.data.loc[(self.data.arc == "Champs-Elysées") & (self.data.t_1h >= "2026-01-11"), ["is_imputed_q", "is_imputed_k"]] = 1
        self.data.loc[self.data.arc == "Saint-Pères", ["is_imputed_q", "is_imputed_k"]] = 1

    def test_values_and_stale_sensor(self):
        window = build_window(self.data, pd.Timestamp("2026-02-01"), recent_dynamics=True)
        row = window[(window.arc == "Convention") & (window.t_1h == "2026-02-01 12:00")].iloc[0]
        self.assertAlmostEqual(row.q_recent_vs_usual, 30 / 110)
        self.assertAlmostEqual(row.q_weekly_trend, 20)
        self.assertAlmostEqual(row.q_real_weekly_std, 20)
        self.assertEqual(row.q_real_weekly_count, 3)
        self.assertEqual(row.q_last_real_age_hours, 0)
        stale = window[(window.arc == "Champs-Elysées") & (window.t_1h == "2026-02-01 12:00")].iloc[0]
        self.assertEqual(stale.q_real_weekly_count, 0)
        self.assertTrue(pd.isna(stale.q_weekly_trend))
        self.assertEqual(stale.q_last_real_age_hours, 21 * 24)
        missing = window[window.arc == "Saint-Pères"].iloc[0]
        self.assertTrue(pd.isna(missing.q_last_real_age_hours))

    def test_inputs_unchanged_if_future_measurements_are_altered(self):
        start = pd.Timestamp("2026-02-01")
        changed = self.data.copy()
        changed.loc[changed.t_1h >= start, ["q", "k"]] *= 99
        original = build_window(self.data, start, recent_dynamics=True)
        altered = build_window(changed, start, recent_dynamics=True)
        cols = feature_columns("q", True, recent_dynamics=True) + feature_columns("k", True, recent_dynamics=True)
        pd.testing.assert_frame_equal(original[list(dict.fromkeys(cols))], altered[list(dict.fromkeys(cols))])

    def test_training_features_match_forecast_for_each_day(self):
        start = pd.Timestamp("2026-02-01")
        forecast = build_window(self.data, start, recent_dynamics=True)
        training = multi_horizon_training_features(self.data, pd.Timestamp("2026-02-05 23:00"), recent_dynamics=True)
        cols = list(dict.fromkeys(feature_columns("q", True, recent_dynamics=True) + feature_columns("k", True, recent_dynamics=True)))
        for day in range(1, 6):
            date = start + pd.Timedelta(days=day - 1)
            a = training[(training.t_1h.dt.normalize() == date) & training.horizon_hours.between((day-1)*24+1, day*24)].sort_values(["arc", "t_1h"])
            b = forecast[forecast.t_1h.dt.normalize().eq(date)].sort_values(["arc", "t_1h"])
            pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True), check_dtype=False)


if __name__ == "__main__":
    unittest.main()
