import unittest
import numpy as np
import pandas as pd
import torch

from main_trajectory import masked_mae, split_inner
from src.trajectory import HORIZON, example, examples_until, make_series


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        times = pd.date_range('2025-10-01', periods=24 * 90, freq='h')
        cls.frame = pd.DataFrame({'arc': 'Convention', 't_1h': times,
                                  'q': np.arange(len(times)) % 100 + 50., 'k': np.arange(len(times)) % 10 + 1.,
                                  'is_imputed_q': 0, 'is_imputed_k': 0})

    def test_input_and_scale_ignore_future_for_each_delay(self):
        start = pd.Timestamp('2025-12-01')
        for delay in (0, 24, 72, 120):
            cutoff = start - pd.Timedelta(hours=1 + delay)
            changed = self.frame.copy()
            changed.loc[changed.t_1h > cutoff, ['q', 'k']] = 1e9
            original = example(make_series(self.frame)['Convention'], start, delay)
            altered = example(make_series(changed)['Convention'], start, delay)
            np.testing.assert_array_equal(original['past'], altered['past'])
            np.testing.assert_array_equal(original['scale'], altered['scale'])
            np.testing.assert_array_equal(original['calendar'], altered['calendar'])

    def test_imputed_values_do_not_affect_scale_input_or_loss(self):
        start = pd.Timestamp('2025-12-01')
        frame = self.frame.copy()
        mask = frame.t_1h.eq(pd.Timestamp('2025-11-30 23:00'))
        frame.loc[mask, ['q', 'is_imputed_q']] = [1e9, 1]
        item = example(make_series(frame)['Convention'], start)
        self.assertEqual(item['past'][0, -1], 0)
        self.assertEqual(item['past'][2, -1], 0)
        self.assertLess(item['scale'][0], 1000)
        truth = torch.tensor([1., 100000.])
        prediction = torch.tensor([2., 0.])
        self.assertEqual(masked_mae(prediction, truth, torch.tensor([1., 0.])).item(), 1.)

    def test_training_and_inner_validation_have_no_overlapping_labels(self):
        cutoff = pd.Timestamp('2025-12-29 18:00')
        samples = examples_until(make_series(self.frame), cutoff, 24)
        self.assertTrue(samples)
        self.assertTrue(all(s['start'] + pd.Timedelta(hours=HORIZON - 1) <= cutoff for s in samples))
        train, valid = split_inner(samples, cutoff, validation_days=14)
        self.assertLess(max(s['start'] + pd.Timedelta(hours=HORIZON - 1) for s in train), min(s['start'] for s in valid))


if __name__ == '__main__':
    unittest.main()
