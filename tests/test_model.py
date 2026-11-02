import unittest

import numpy as np
import pandas as pd

from main_train import compare_on_common_rows
from src.evaluate import build_window
from src.features import ARCS
from src.model import feature_columns, fit_model, predict_model, training_features


class ModelTests(unittest.TestCase):
    def test_training_and_predictions_ignore_future_targets(self):
        times = pd.date_range("2026-01-01", "2026-03-01 23:00", freq="h")
        data = pd.MultiIndex.from_product([ARCS, times], names=["arc", "t_1h"]).to_frame(index=False)
        data["q"] = 100.0 + np.arange(len(data)) % 100
        data["k"] = 10.0 + np.arange(len(data)) % 10
        data["is_imputed_q"] = 0
        data["is_imputed_k"] = 0
        start = pd.Timestamp("2026-02-01")
        cutoff = start - pd.Timedelta(hours=1)
        changed = data.copy()
        changed.loc[changed.t_1h >= start, ["q", "k"]] *= 7
        for target in ("q", "k"):
            a, n_a = fit_model(training_features(data, cutoff), target)
            b, n_b = fit_model(training_features(changed, cutoff), target)
            left = predict_model(a, build_window(data, start), target)
            right = predict_model(b, build_window(changed, start), target)
            self.assertEqual(n_a, n_b)
            np.testing.assert_allclose(left, right)
            self.assertNotIn("q", feature_columns(target))
            self.assertNotIn("k", feature_columns(target))

    def test_comparison_uses_same_observations(self):
        data = pd.DataFrame({"arc": ["Convention", "Convention"], "q": [1., 100.], "k": [1., 1.],
            "is_imputed_q": [0, 0], "is_imputed_k": [0, 0], "q_lightgbm": [2., 200.],
            "k_lightgbm": [2., 2.], "q_lag_168h": [1., np.nan], "k_lag_168h": [1., 1.],
            "q_weekly_mean_3w": [1., np.nan], "k_weekly_mean_3w": [1., 1.],
            "q_smart_expected": [1., np.nan], "k_smart_expected": [1., 1.]})
        row = compare_on_common_rows(data).query("target == 'q' and method == 'semaine_precedente'").iloc[0]
        self.assertEqual(row["count"], 1)
        self.assertEqual(row["lightgbm_mae_same_rows"], 1)


if __name__ == "__main__":
    unittest.main()
