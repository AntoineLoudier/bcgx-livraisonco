import unittest

import numpy as np
import pandas as pd

from main_stability import holiday_windows, score_holidays, stability_by_window


class StabilityTests(unittest.TestCase):
    def test_window_comparison_uses_common_observations(self):
        frame = pd.DataFrame({"window_start": [pd.Timestamp("2026-05-01")] * 2,
            "arc": ["Convention"] * 2, "q": [1., 100.], "k": [1., 1.],
            "is_imputed_q": [0, 0], "is_imputed_k": [0, 0],
            "q_lightgbm": [2., 200.], "k_lightgbm": [2., 2.],
            "q_lag_168h": [1., 1.], "k_lag_168h": [1., 1.],
            "q_weekly_mean_3w": [1., np.nan], "k_weekly_mean_3w": [1., 1.],
            "q_smart_expected": [1., 1.], "k_smart_expected": [1., 1.]})
        row = stability_by_window(frame).query("target == 'q'").iloc[0]
        self.assertEqual(row["count"], 1)
        self.assertEqual(row.lightgbm_mae_same_rows, 1)
        self.assertEqual(row.improvement, -1)

    def test_holidays_exclude_imputed_truth(self):
        day = pd.Timestamp("2026-05-01")
        pred = pd.DataFrame({"window_start": [day] * 2, "t_1h": [day, day + pd.Timedelta(hours=1)],
            "arc": ["Convention"] * 2, "q": [10., 999.], "k": [2., 999.],
            "is_imputed_q": [0, 1], "is_imputed_k": [0, 1],
            "q_lightgbm": [8., 1.], "k_lightgbm": [1., 1.],
            "q_weekly_mean_3w": [7., 1.], "k_weekly_mean_3w": [0., 1.]})
        events = pd.DataFrame({"holiday": [day], "window_start": [day], "window_end": [day + pd.Timedelta(days=4)]})
        row = score_holidays(pred, events).query("period == 'ferie' and target == 'q'").iloc[0]
        self.assertEqual(row["count"], 1)
        self.assertEqual(row.lightgbm_mae, 2)
        self.assertEqual(row.baseline_mae, 3)

    def test_event_windows_stay_within_available_dates(self):
        frame = pd.DataFrame({"t_1h": pd.to_datetime(["2025-10-01 05:00", "2026-11-02 00:00"])})
        events = holiday_windows(frame)
        self.assertTrue((events.window_start - pd.Timedelta(days=21) >= frame.t_1h.min()).all())
        self.assertTrue((events.window_end <= frame.t_1h.max()).all())
        self.assertTrue((events.window_start <= events.holiday).all())
        self.assertTrue((events.window_end >= events.holiday).all())


if __name__ == "__main__":
    unittest.main()
