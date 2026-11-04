import unittest
import warnings
import numpy as np
import pandas as pd
from src.features import ARCS
from src.model import multi_horizon_training_features, feature_columns
from src.evaluate import build_window
from src.recent_context import context_table, attach_context, context_columns
from main_recent_context import choose_starts


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        times = pd.date_range('2025-10-01', '2026-01-31 23:00', freq='h')
        cls.frame = pd.MultiIndex.from_product([ARCS, times], names=['arc', 't_1h']).to_frame(index=False)
        cls.frame['q'] = 100 + np.arange(len(cls.frame)) % 73
        cls.frame['k'] = 10 + np.arange(len(cls.frame)) % 20
        cls.frame['is_imputed_q'] = 0
        cls.frame['is_imputed_k'] = 0
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

    def test_causality_and_horizon_alignment_all_columns(self):
        start = pd.Timestamp('2026-01-10')
        cols = list(dict.fromkeys(feature_columns('q', True) + feature_columns('k', True) + context_columns('q')))
        for delay in (0, 24, 72, 120):
            cutoff = start - pd.Timedelta(hours=1 + delay)
            changed = self.frame.copy()
            changed.loc[changed.t_1h > cutoff, ['q', 'k']] *= 100
            valid = attach_context(build_window(self.frame, start, availability_delay_hours=delay), context_table(self.frame), delay)
            altered = attach_context(build_window(changed, start, availability_delay_hours=delay), context_table(changed), delay)
            pd.testing.assert_frame_equal(valid[cols], altered[cols])
            train = multi_horizon_training_features(self.frame, pd.Timestamp('2026-01-31 23:00'), availability_delay_hours=delay)
            train = attach_context(train, context_table(self.frame), delay)
            for day in range(1, 6):
                date = start + pd.Timedelta(days=day-1)
                a = train[(train.t_1h.dt.normalize() == date) & train.horizon_hours.between((day-1)*24+1, day*24)].sort_values(['arc', 't_1h'])
                b = valid[valid.t_1h.dt.normalize().eq(date)].sort_values(['arc', 't_1h'])
                pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True), check_dtype=False)

    def test_excludes_imputed_and_calendar(self):
        data = self.frame.copy()
        mask = (data.arc == 'Convention') & data.t_1h.between('2026-01-09 18:00', '2026-01-09 23:00')
        data.loc[mask, ['q', 'k']] = 1e8
        data.loc[mask, ['is_imputed_q', 'is_imputed_k']] = 1
        table = context_table(data).set_index(['arc', 'available_at'])
        row = table.loc[('Convention', pd.Timestamp('2026-01-09 23:00'))]
        self.assertTrue(pd.isna(row.q_ctx_mean_6))
        self.assertEqual(row.q_ctx_coverage_6, 0)
        self.assertEqual(row.q_ctx_age, 6)
        starts = choose_starts(data, 8)
        self.assertIn(5, [x.dayofweek for x in starts])
        self.assertEqual(len(set(x.dayofweek for x in starts)), 7)
        self.assertGreaterEqual(sum(x.dayofweek == 5 for x in starts), 2)
        self.assertTrue(all((b - a).days >= 5 for a, b in zip(starts, starts[1:])))

    def test_training_labels_respect_latency(self):
        issue = pd.Timestamp('2026-01-20 23:00')
        train = multi_horizon_training_features(self.frame, issue, availability_delay_hours=72)
        self.assertLessEqual(train.t_1h.max(), issue - pd.Timedelta(hours=72))


if __name__ == '__main__':
    unittest.main()
