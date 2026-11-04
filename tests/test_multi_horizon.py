import unittest

import numpy as np
import pandas as pd

from src.evaluate import build_window
from src.features import ARCS, add_calendar_and_event_features, add_causal_lags_and_sncf_patterns
from src.model import feature_columns, multi_horizon_training_features


class MultiHorizonTests(unittest.TestCase):
    def setUp(self):
        times = pd.date_range("2025-10-01", "2025-12-15 23:00", freq="h")
        self.frame = pd.MultiIndex.from_product([ARCS, times], names=["arc", "t_1h"]).to_frame(index=False)
        self.frame["q"] = 100 + np.arange(len(self.frame)) % 73
        self.frame["k"] = 10 + np.arange(len(self.frame)) % 20
        self.frame["is_imputed_q"] = 0
        self.frame["is_imputed_k"] = 0

    def test_training_features_match_fixed_emission_window(self):
        start = pd.Timestamp("2025-11-09")
        valid = build_window(self.frame, start)
        calendar = add_calendar_and_event_features(self.frame)
        columns = list(dict.fromkeys(feature_columns("q", True) + feature_columns("k", True)))
        for day in range(1, 6):
            historical = add_causal_lags_and_sncf_patterns(calendar, training_horizon_days=day)
            date = start + pd.Timedelta(days=day - 1)
            a = historical[historical.t_1h.dt.normalize().eq(date)].sort_values(["arc", "t_1h"])
            b = valid[valid.t_1h.dt.normalize().eq(date)].sort_values(["arc", "t_1h"])
            pd.testing.assert_frame_equal(a[columns].reset_index(drop=True), b[columns].reset_index(drop=True), check_dtype=False)
            self.assertEqual(a.horizon_hours.min(), (day - 1) * 24 + 1)
            self.assertEqual(a.horizon_hours.max(), day * 24)

    def test_perturbations_after_each_emission_do_not_affect_inputs(self):
        cutoff = pd.Timestamp("2025-11-08 23:00")
        changed = self.frame.copy()
        changed.loc[changed.t_1h > cutoff, ["q", "k"]] *= 13
        original = multi_horizon_training_features(self.frame, pd.Timestamp("2025-11-13 23:00"))
        modified = multi_horizon_training_features(changed, pd.Timestamp("2025-11-13 23:00"))
        emission = original.t_1h - pd.to_timedelta(original.horizon_hours, unit="h")
        mask = emission.le(cutoff)
        cols = list(dict.fromkeys(feature_columns("q", True) + feature_columns("k", True)))
        pd.testing.assert_frame_equal(original.loc[mask, cols], modified.loc[mask, cols])

    def test_invalid_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            add_causal_lags_and_sncf_patterns(add_calendar_and_event_features(self.frame), training_horizon_days=0)


if __name__ == "__main__":
    unittest.main()
