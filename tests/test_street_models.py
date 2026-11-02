import unittest

import numpy as np
import pandas as pd

from main_compare_streets import evaluate_windows, score_groups
from src.features import ARCS


class StreetModelTests(unittest.TestCase):
    def test_local_model_ignores_other_streets_and_future(self):
        times = pd.date_range("2026-01-01", "2026-02-12 23:00", freq="h")
        frame = pd.MultiIndex.from_product([ARCS, times], names=["arc", "t_1h"]).to_frame(index=False)
        frame["q"] = 30.0 + np.arange(len(frame)) % 50
        frame["k"] = 5.0 + np.arange(len(frame)) % 10
        frame["is_imputed_q"] = 0
        frame["is_imputed_k"] = 0
        start = pd.Timestamp("2026-02-01")
        changed = frame.copy()
        changed.loc[(changed.arc != "Convention") | (changed.t_1h >= start), ["q", "k"]] *= 8
        a, _ = evaluate_windows(frame, [start])
        b, _ = evaluate_windows(changed, [start])
        a = a[a.arc == "Convention"]
        b = b[b.arc == "Convention"]
        for target in ("q", "k"):
            np.testing.assert_allclose(a[f"{target}_street"], b[f"{target}_street"])

    def test_comparisons_use_common_real_measurements(self):
        data = pd.DataFrame({"arc": ["Convention"] * 3, "q": [1., 20., 100.], "k": [1., 1., 1.],
                             "is_imputed_q": [0, 1, 0], "is_imputed_k": [0, 0, 0],
                             "q_global": [2., 0., 100.], "q_street": [3., 0., 1.],
                             "q_weekly_mean_3w": [1., 0., np.nan],
                             "k_global": [1., 1., 1.], "k_street": [1., 1., 1.],
                             "k_weekly_mean_3w": [1., 1., 1.]})
        row = score_groups(data, []).query("target == 'q'").iloc[0]
        self.assertEqual(row["observed"], 2)
        self.assertEqual(row["count"], 1)
        self.assertEqual(row.global_mae, 1)
        self.assertEqual(row.street_mae, 2)


if __name__ == "__main__":
    unittest.main()
