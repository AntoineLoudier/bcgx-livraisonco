import unittest

import numpy as np
import pandas as pd

from main_holidays import VARIANTS, score
from src.features import add_calendar_and_event_features, add_causal_lags_and_sncf_patterns
from src.model import feature_columns, multi_horizon_training_features


class HolidayTests(unittest.TestCase):
    def test_calendar_distances_types_weekday_and_vacation(self):
        frame = pd.DataFrame({"t_1h": pd.to_datetime(["2026-05-13", "2026-10-31", "2026-11-09", "2026-11-11"]),
                              "arc": ["Convention"] * 4})
        data = add_calendar_and_event_features(frame)
        self.assertEqual((data.loc[0, "days_until_holiday"], data.loc[0, "nearest_holiday_movable"], data.loc[0, "nearest_holiday_weekday"]), (1, 1, 3))
        self.assertEqual(data.loc[1, "near_holiday_in_school_break"], 1)
        self.assertEqual((data.loc[2, "days_until_holiday"], data.loc[2, "nearest_holiday_fixed"], data.loc[2, "nearest_holiday_weekday"]), (2, 1, 2))
        self.assertEqual((data.loc[3, "days_since_holiday"], data.loc[3, "days_until_holiday"]), (0, 0))

    def test_disable_bridge_removes_weights_and_regime_but_keeps_holidays(self):
        times = pd.date_range("2025-10-01", "2026-11-11 23:00", freq="h")
        frame = pd.DataFrame({"arc": "Convention", "t_1h": times, "q": 100., "k": 10.,
                              "is_imputed_q": 0, "is_imputed_k": 0})
        normal = add_calendar_and_event_features(frame)
        neutral = add_calendar_and_event_features(frame, manual_bridge=False)
        day = neutral.t_1h.eq(pd.Timestamp("2026-11-10 12:00"))
        self.assertEqual(normal.loc[day, "bridge_intensity"].iloc[0], 0.5)
        self.assertEqual(neutral.loc[day, "bridge_intensity"].iloc[0], 0)
        self.assertEqual(neutral.loc[day, "day_regime"].iloc[0], 0)
        self.assertTrue(normal.is_public_holiday.equals(neutral.is_public_holiday))
        result = add_causal_lags_and_sncf_patterns(neutral, training_horizon_days=2)
        day = result.t_1h.eq(pd.Timestamp("2026-11-10 12:00"))
        self.assertAlmostEqual(result.loc[day, "q_smart_expected"].iloc[0], result.loc[day, "q_weekly_mean_3w"].iloc[0])

    def test_metrics_use_common_observed_rows(self):
        frame = pd.DataFrame({"arc": ["Convention"] * 3, "q": [10., 20., 30.], "k": [1., 2., 3.],
                              "is_imputed_q": [0, 1, 0], "is_imputed_k": [0, 0, 0]})
        for name in VARIANTS:
            frame[f"q_{name}"] = [9., 0., np.nan if name == "sans_pont" else 0.]
            frame[f"k_{name}"] = 1.
        row = score(frame, ["arc"]).query("target == 'q'").iloc[0]
        self.assertEqual(row.observed, 2)
        self.assertEqual(row.scored, 1)
        self.assertEqual(row.reference_mae, 1)

    def test_no_bridge_variant_never_uses_measurements_after_emission(self):
        times = pd.date_range("2025-10-01", "2025-11-20 23:00", freq="h")
        frame = pd.DataFrame({"arc": "Convention", "t_1h": times,
                              "q": 100. + np.arange(len(times)) % 80,
                              "k": 10. + np.arange(len(times)) % 20,
                              "is_imputed_q": 0, "is_imputed_k": 0})
        cutoff = pd.Timestamp("2025-11-08 23:00")
        changed = frame.copy()
        changed.loc[changed.t_1h > cutoff, ["q", "k"]] *= 7
        original = multi_horizon_training_features(frame, frame.t_1h.max(), manual_bridge=False)
        perturbed = multi_horizon_training_features(changed, frame.t_1h.max(), manual_bridge=False)
        emission = original.t_1h - pd.to_timedelta(original.horizon_hours, unit="h")
        columns = list(dict.fromkeys(feature_columns("q", True, True) + feature_columns("k", True, True)))
        pd.testing.assert_frame_equal(original.loc[emission <= cutoff, columns], perturbed.loc[emission <= cutoff, columns])


if __name__ == "__main__":
    unittest.main()
