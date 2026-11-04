import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from main_recency import fit_strategy, score, select_training


class RecencyTests(unittest.TestCase):
    def test_cutoffs_and_decay_never_use_future(self):
        cutoff = pd.Timestamp("2026-10-31 23:00")
        dates = pd.to_datetime(["2026-04-30 23:00", "2026-05-01 00:00", "2026-07-31 23:00", "2026-08-01 00:00", "2026-10-31 23:00", "2026-11-01 00:00"])
        frame = pd.DataFrame({"t_1h": dates})
        three, _ = select_training(frame, cutoff, "months_3")
        six, _ = select_training(frame, cutoff, "months_6")
        self.assertEqual(three.t_1h.tolist(), dates[3:5].tolist())
        self.assertEqual(six.t_1h.tolist(), dates[1:5].tolist())
        weighted, weights = select_training(frame, cutoff, "decay_180d")
        self.assertEqual(len(weighted), 5)
        self.assertAlmostEqual(weights.iloc[-1], 1.0)
        self.assertGreater(weights.iloc[-1], weights.iloc[0])

    def test_weight_normalization_uses_observed_targets(self):
        cutoff = pd.Timestamp("2026-10-31 23:00")
        frame = pd.DataFrame({"t_1h": [cutoff - pd.Timedelta(days=180), cutoff, cutoff],
                              "arc": ["Convention"] * 3, "q": [1., 2., 999.],
                              "is_imputed_q": [0, 0, 1]})
        with patch("main_recency.fit_model", return_value=("model", 2)) as fit:
            _, n, counts = fit_strategy(frame, cutoff, "decay_180d", "q")
        weights = fit.call_args.kwargs["sample_weight"]
        self.assertAlmostEqual(weights.iloc[:2].mean(), 1.0)
        self.assertAlmostEqual(weights.iloc[1] / weights.iloc[0], 2.0)
        self.assertEqual(n, 2)
        self.assertEqual(next(row for row in counts if row["arc"] == "Convention")["observed_hours"], 2)
        self.assertEqual(next(row for row in counts if row["arc"] == "Champs-Elysées")["observed_hours"], 0)

    def test_scores_only_common_observed_rows(self):
        frame = pd.DataFrame({"arc": ["Convention"] * 3, "q": [1., 100., 200.], "k": [1., 1., 1.],
                              "is_imputed_q": [0, 1, 0], "is_imputed_k": [0, 0, 0]})
        for strategy in ("all", "months_3", "months_6", "decay_180d"):
            frame[f"q_{strategy}"] = [2., 0., np.nan if strategy == "months_3" else 0.]
            frame[f"k_{strategy}"] = [1., 1., 1.]
        row = score(frame, ["arc"]).query("target == 'q'").iloc[0]
        self.assertEqual(row.observed, 2)
        self.assertEqual(row.scored, 1)
        self.assertEqual(row.all_mae, 1)


if __name__ == "__main__":
    unittest.main()
