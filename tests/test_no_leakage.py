import unittest

import numpy as np
import pandas as pd

from src.features import add_calendar_and_event_features, add_causal_lags_and_sncf_patterns
from src.preprocessing import impute_missing_values_no_leakage


class LeakageTests(unittest.TestCase):
    def test_imputation_does_not_use_future_values(self):
        dates = pd.date_range("2025-01-01", periods=24 * 23, freq="h")
        frame = pd.DataFrame({"arc": "Convention", "t_1h": dates, "q": 10.0, "k": 5.0})
        frame.loc[100, ["q", "k"]] = np.nan
        frame.loc[24 * 22, ["q", "k"]] = np.nan
        changed = frame.copy()
        changed.loc[101:200, ["q", "k"]] = 10000.0
        original = impute_missing_values_no_leakage(frame)
        modified = impute_missing_values_no_leakage(changed)
        pd.testing.assert_frame_equal(original.loc[:100], modified.loc[:100])
        self.assertTrue(pd.isna(original.loc[100, "q"]))

    def test_future_measurements_cannot_change_historical_features(self):
        times = pd.date_range("2025-10-01", "2025-12-25 23:00", freq="h")
        frame = pd.MultiIndex.from_product(
            [["Champs-Elysées", "Convention", "Saint-Pères"], times], names=["arc", "t_1h"]
        ).to_frame(index=False)
        frame["q"] = 100 + np.arange(len(frame)) % 72
        frame["k"] = 10 + np.arange(len(frame)) % 20
        frame["is_imputed_q"] = 0
        frame["is_imputed_k"] = 0
        cutoff = pd.Timestamp("2025-11-06 12:00")
        changed = frame.copy()
        changed.loc[changed.t_1h > cutoff, ["q", "k"]] *= 5
        original = add_causal_lags_and_sncf_patterns(add_calendar_and_event_features(frame))
        modified = add_causal_lags_and_sncf_patterns(add_calendar_and_event_features(changed))
        selection = original.t_1h <= cutoff + pd.Timedelta(hours=120)
        cols = [col for col in original if "lag_" in col or "weekly_" in col or "hist_mean" in col or "smart_expected" in col]
        pd.testing.assert_frame_equal(original.loc[selection, cols], modified.loc[selection, cols])

    def test_last_weeks_do_not_look_forward(self):
        times = pd.date_range("2025-10-01", "2026-02-01", freq="h")
        frame = pd.DataFrame({"arc": "Champs-Elysées", "t_1h": times})
        frame["q"] = 100.0
        frame["k"] = 10.0
        frame["is_imputed_q"] = 0
        frame["is_imputed_k"] = 0
        changed = frame.copy()
        changed.loc[changed.t_1h >= "2026-01-01", ["q", "k"]] = 999.0
        a = add_causal_lags_and_sncf_patterns(add_calendar_and_event_features(frame))
        b = add_causal_lags_and_sncf_patterns(add_calendar_and_event_features(changed))
        selected = a.t_1h.eq(pd.Timestamp("2025-12-20 12:00"))
        cols = ["q_lag_168h", "q_lag_336h", "q_lag_504h", "q_smart_expected"]
        pd.testing.assert_frame_equal(a.loc[selected, cols], b.loc[selected, cols])


if __name__ == "__main__":
    unittest.main()
