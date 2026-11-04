import unittest

import numpy as np
import pandas as pd

from main_blend import blended, choose_weight, scores


class BlendTests(unittest.TestCase):
    def history(self):
        rows = []
        for i in range(5):
            start = pd.Timestamp("2026-01-01") + pd.Timedelta(days=7*i)
            rows.append({"window_start": start, "t_1h": start, "arc": "Convention",
                         "q": 100., "k": 10., "is_imputed_q": 0, "is_imputed_k": 0,
                         "q_model": 120., "k_model": 12., "q_weekly_mean_3w": 100., "k_weekly_mean_3w": 10.})
        return pd.DataFrame(rows)

    def test_only_completed_windows_control_calibration(self):
        history = self.history()
        cutoff = pd.Timestamp("2026-01-28 23:00")
        selected = choose_weight(history, "q", cutoff, min_windows=4, min_observations=1)
        self.assertEqual(selected["model_weight"], 0)
        self.assertEqual(selected["calibration_windows"], 4)
        history["k_model"] = history["k"]
        self.assertEqual(choose_weight(history, "k", cutoff, 4, 1)["model_weight"], 1)
        self.assertIsNone(choose_weight(history, "q", pd.Timestamp("2026-01-25 23:00"), min_windows=4, min_observations=1))
        changed = history.copy()
        changed.loc[changed.window_start >= "2026-01-29", "q"] = 9999
        self.assertEqual(selected, choose_weight(changed, "q", cutoff, min_windows=4, min_observations=1))
        history.loc[history.window_start.eq("2026-01-22"), "is_imputed_q"] = 1
        self.assertIsNone(choose_weight(history, "q", cutoff, min_windows=4, min_observations=1))

    def test_unavailable_baseline_uses_model_and_ties_favor_model(self):
        np.testing.assert_allclose(blended([20, 30], [10, np.nan], .25), [12.5, 30])
        history = self.history().iloc[:4].copy()
        history["q_weekly_mean_3w"] = np.nan
        result = choose_weight(history, "q", pd.Timestamp("2026-02-01"), min_windows=4, min_observations=1)
        self.assertEqual(result["model_weight"], 1)
        self.assertEqual(result["baseline_available"], 0)
        with self.assertRaises(ValueError):
            blended([1], [2], 1.1)

    def test_comparison_excludes_imputed_and_reports_fallback(self):
        data = pd.DataFrame({"arc": ["Convention"] * 3, "q": [10., 20., 30.], "k": [1., 2., 3.],
            "is_imputed_q": [0, 1, 0], "is_imputed_k": [0, 0, 0],
            "q_model": [12., 0., 32.], "q_blend": [11., 0., 32.], "q_weekly_mean_3w": [10., 0., np.nan],
            "k_model": [1., 2., 3.], "k_blend": [1., 2., 3.], "k_weekly_mean_3w": [1., 2., 3.]})
        row = scores(data, ["arc"]).query("target == 'q'").iloc[0]
        self.assertEqual((row.observed, row.scored, row.baseline_available, row.fallback_to_model), (2, 2, 1, 1))
        self.assertEqual(row.model_mae, 2)
        self.assertEqual(row.blend_mae, 1.5)
        self.assertEqual(row.blend_on_baseline_mae, 1)


if __name__ == "__main__":
    unittest.main()
