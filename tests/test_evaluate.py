import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import build_window, score_window, weather_availability, window_starts
from src.features import ARCS


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2026-01-01", "2026-03-01 23:00", freq="h")
        self.frame = pd.MultiIndex.from_product([ARCS, dates], names=["arc", "t_1h"]).to_frame(index=False)
        self.frame["q"] = np.arange(len(self.frame)) % 200 + 1.0
        self.frame["k"] = np.arange(len(self.frame)) % 20 + 1.0
        self.frame["is_imputed_q"] = 0
        self.frame["is_imputed_k"] = 0
        self.start = pd.Timestamp("2026-02-01")

    def test_future_truth_never_changes_window_predictions(self):
        changed = self.frame.copy()
        changed.loc[changed.t_1h >= self.start, ["q", "k"]] *= 99
        a, b = build_window(self.frame, self.start), build_window(changed, self.start)
        cols = [c for c in a if "lag_" in c or "weekly_" in c or "hist_mean" in c or "smart_expected" in c]
        pd.testing.assert_frame_equal(a[cols], b[cols])
        self.assertEqual(len(a), 360)
        self.assertTrue(a.forecast_cutoff.eq(self.start - pd.Timedelta(hours=1)).all())

    def test_score_excludes_imputed_and_missing_truth(self):
        result = build_window(self.frame, self.start)
        index = result.index[result.arc == "Convention"]
        result.loc[index[0], "is_imputed_q"] = 1
        result.loc[index[1], "q"] = np.nan
        result.loc[index[2], "q_lag_168h"] = np.nan
        scores = score_window(result)
        row = scores[(scores.arc == "Convention") & (scores.target == "q") & (scores.method == "semaine_precedente")].iloc[0]
        self.assertEqual((row.observed, row.scored, row.missing_prediction), (118, 117, 1))
        result.loc[result.arc == "Champs-Elysées", "is_imputed_q"] = 1
        row = score_window(result).query("arc == 'Champs-Elysées' and target == 'q'").iloc[0]
        self.assertEqual(row.scored, 0)
        self.assertTrue(pd.isna(row.mae))

    def test_weather_requires_timestamp_and_filters_late_forecasts(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weather.csv"
            pd.DataFrame({"t_1h": ["2026-02-01"], "rain": [1]}).to_csv(path, index=False)
            unchecked = weather_availability(path, [self.start]).iloc[0]
            self.assertTrue(pd.isna(unchecked.forecast_hours_with_eligible_record))
            pd.DataFrame({"issued_at": ["2026-01-31T22:00:00+01:00", "2026-02-01T01:00:00+01:00"],
                          "valid_at": ["2026-02-01T00:00:00+01:00", "2026-02-01T01:00:00+01:00"]}).to_csv(path, index=False)
            checked = weather_availability(path, [self.start]).iloc[0]
            self.assertEqual(checked.forecast_hours_with_eligible_record, 1)
            self.assertFalse(checked.used_in_baselines)

    def test_windows_stop_at_last_complete_day(self):
        partial = self.frame[self.frame.t_1h <= "2026-03-01 00:00"]
        self.assertEqual(window_starts(partial, 1, 7), [pd.Timestamp("2026-02-24")])


if __name__ == "__main__":
    unittest.main()
