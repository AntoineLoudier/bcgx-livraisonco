import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from main_tabular_benchmark import ensure_manifest
from src.evaluate import build_window
from src.feature_lab import FeatureStore, columns_for, observed_only, preserve_raw_zeros
from src.features import ARCS
from src.model import multi_horizon_training_features
from src.tabular_benchmark import (FAMILIES, best_blend, inner_starts, observed_records,
                                  paired_reports, predict, target_scale)


class FeatureLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        times = pd.date_range('2026-01-01', '2026-04-15 23:00', freq='h')
        cls.frame = pd.MultiIndex.from_product([ARCS, times], names=['arc', 't_1h']).to_frame(index=False)
        cls.frame['q'] = 100 + 30*np.sin(cls.frame.t_1h.dt.hour/24*2*np.pi) + np.arange(len(cls.frame))%17
        cls.frame['k'] = cls.frame.q/10
        cls.frame['is_imputed_q'] = 0
        cls.frame['is_imputed_k'] = 0
        cls.store = FeatureStore(cls.frame)

    def test_every_feature_causal_and_training_inference_equivalent(self):
        start = pd.Timestamp('2026-04-01')
        for delay in (0, 120):
            changed = self.frame.copy()
            available = start-pd.Timedelta(hours=1+delay)
            future = changed.t_1h > available
            changed.loc[future, ['q', 'k']] = 7777
            changed.loc[future, ['is_imputed_q', 'is_imputed_k']] = 1
            original = self.store.attach(build_window(self.frame, start, availability_delay_hours=delay), delay)
            perturbed = FeatureStore(changed).attach(build_window(changed, start, availability_delay_hours=delay), delay)
            cols = list(dict.fromkeys(columns_for('q', 'all', self.store)+columns_for('k', 'all', self.store)))
            pd.testing.assert_frame_equal(original[cols], perturbed[cols])
            training = multi_horizon_training_features(self.frame, self.frame.t_1h.max(), availability_delay_hours=delay)
            issue = training.t_1h-pd.to_timedelta(training.horizon_hours, unit='h')
            matching = training[issue.eq(start-pd.Timedelta(hours=1))]
            attached = self.store.attach(matching, delay)
            a = attached.sort_values(['arc', 't_1h'])[cols].reset_index(drop=True)
            b = original.sort_values(['arc', 't_1h'])[cols].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_dtype=False)

    def test_imputed_values_do_not_enter_new_features(self):
        data = self.frame.copy()
        mask = data.t_1h.between('2026-03-20', '2026-03-31 23:00')
        data.loc[mask, ['is_imputed_q', 'is_imputed_k']] = 1
        a = FeatureStore(data)
        data.loc[mask, ['q', 'k']] = 1e10
        b = FeatureStore(data)
        pd.testing.assert_frame_equal(a.context, b.context)
        pd.testing.assert_frame_equal(a.profiles, b.profiles)
        self.assertTrue(observed_only(data).loc[mask, 'q'].isna().all())

    def test_analogues_never_use_unfinished_successors(self):
        available = pd.Timestamp('2026-03-20 23:00')
        a, info = self.store.analogues('Convention', available, 120)
        self.assertGreater(info['analog_candidates'], 0)
        changed = self.frame.copy()
        changed.loc[changed.t_1h > available, ['q', 'k']] = 1e9
        b, info2 = FeatureStore(changed).analogues('Convention', available, 120)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(info, info2)
        self.assertLessEqual(info['analog_candidates'], 5)

    def test_quality_flags_and_incomplete_day(self):
        data = self.frame.copy()
        mask = data.arc.eq('Convention') & data.t_1h.between('2026-03-31 17:00', '2026-03-31 23:00')
        data.loc[mask, ['q', 'k']] = 0
        missing = data.arc.eq('Convention') & data.t_1h.eq('2026-03-31 00:00')
        data.loc[missing, 'is_imputed_q'] = 1
        row = FeatureStore(data).context.set_index(['arc', 'available_at']).loc[('Convention', pd.Timestamp('2026-03-31 23:00'))]
        self.assertEqual(row.q_quality_zero_run, 7)
        self.assertEqual(row.q_shape_complete, 0)
        self.assertTrue(pd.isna(row.q_shape_mean))
        self.assertEqual(row.q_quality_age, 0)

    def test_restore_raw_joint_zeros_without_changing_truth(self):
        truth = self.frame.iloc[:2].copy()
        truth.loc[:, ['q', 'k']] = 100
        truth.loc[:, ['is_imputed_q', 'is_imputed_k']] = 1
        before = truth.copy()
        raw = truth[['arc', 't_1h', 'q', 'k']].copy()
        raw['t_1h'] = raw.t_1h.dt.tz_localize('Europe/Paris').dt.tz_convert('UTC').astype(str)
        raw.loc[raw.index[0], ['q', 'k']] = 0
        raw.loc[raw.index[1], 'q'] = -1
        restored, _ = preserve_raw_zeros(raw, truth)
        self.assertEqual(restored.is_imputed_q.iloc[0], 0)
        self.assertEqual(restored.q.iloc[0], 0)
        self.assertEqual(restored.is_imputed_q.iloc[1], 1)
        pd.testing.assert_frame_equal(truth, before)


class ProtocolTests(unittest.TestCase):
    def test_inner_targets_are_available_and_disjoint(self):
        start = pd.Timestamp('2026-10-24')
        for delay in (0, 24, 72, 120):
            starts = inner_starts(start, delay, 3)
            self.assertTrue(all(s+pd.Timedelta(hours=119) <= start-pd.Timedelta(hours=delay+1) for s in starts))
            self.assertTrue(all(b-a >= pd.Timedelta(days=5) for a, b in zip(starts, starts[1:])))

    def test_blend_selects_on_supplied_inner_records(self):
        ref = pd.DataFrame({'arc': ['Convention']*4, 't_1h': pd.date_range('2026-01-01', periods=4, freq='h'),
                            'window_start': pd.Timestamp('2026-01-01'), 'target': 'q', 'truth': 10., 'prediction': 0.})
        other = ref.assign(prediction=20.)
        family, weight, score = best_blend({'lightgbm': ref, 'catboost': other})
        self.assertEqual((family, weight, score), ('catboost', .5, 0.))

    def test_manifest_rejects_stale_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            ensure_manifest(path, {'fingerprint': 'old'})
            with self.assertRaises(ValueError):
                ensure_manifest(path, {'fingerprint': 'new'})
            self.assertEqual(json.loads((path/'manifest.json').read_text())['fingerprint'], 'old')

    def test_scores_exclude_imputed_and_reject_unpaired(self):
        window = pd.DataFrame({'arc': ['Convention']*2, 't_1h': pd.date_range('2026-01-01', periods=2, freq='h'),
                               'window_start': pd.Timestamp('2026-01-01'), 'horizon_hours': [1, 2],
                               'is_public_holiday': 1, 'q': [10., 9999.], 'is_imputed_q': [0, 1]})
        records = observed_records(window, np.array([11., 12.]), 'q')
        self.assertEqual(len(records), 1)
        records['delay_hours'] = 0
        combined = pd.concat([records.assign(method='reference'), records.assign(method='candidate')])
        with tempfile.TemporaryDirectory() as folder:
            paired_reports(combined, Path(folder))
            scores = pd.read_csv(Path(folder)/'summary.csv')
            self.assertTrue(scores.n.eq(1).all())
            bad = pd.concat([combined, records.assign(method='reference', t_1h=pd.Timestamp('2026-01-02'))])
            with self.assertRaises(ValueError):
                paired_reports(bad, Path(folder))

    def test_all_estimators_and_transforms_handle_missing_inputs(self):
        rng = np.random.default_rng(42)
        cols = ['arc_id', 'hour', 'dayofweek', 'month', 'horizon_hours', 'q_regime_hist_mean', 'q_profile_84_median', 'signal']
        x = pd.DataFrame({'arc_id': np.arange(150)%3, 'hour': np.arange(150)%24,
                          'dayofweek': np.arange(150)%7, 'month': 1, 'horizon_hours': np.arange(150)%120+1,
                          'q_regime_hist_mean': 100., 'q_profile_84_median': 90., 'signal': rng.normal(size=150),
                          'q': 100+rng.normal(size=150)*10, 'is_imputed_q': 0})
        x.loc[:10, 'signal'] = np.nan
        x.loc[:5, 'is_imputed_q'] = 1
        with patch('src.tabular_benchmark.columns_for', return_value=cols):
            for family in FAMILIES:
                for variant in ('compact', 'log', 'relative'):
                    p, info = predict(x, x.iloc[:3], 'q', family, 0, variant, None, threads=1)
                    self.assertEqual(info['train_rows'], 144)
                    self.assertTrue(np.isfinite(p).all())
                    self.assertTrue((p >= 0).all())

    def test_occupancy_predictions_respect_physical_bounds(self):
        class Model:
            def fit(self, x, y):
                return self

            def predict(self, x):
                return np.array([-10., 110.])

        frame = pd.DataFrame({'arc_id': np.zeros(120, dtype=int), 'k': 2., 'is_imputed_k': 0})
        with patch('src.tabular_benchmark.columns_for', return_value=['arc_id']), \
             patch('src.tabular_benchmark.estimator', return_value=Model()):
            result, _ = predict(frame, frame.iloc[:2], 'k', 'lightgbm', 0, 'compact', None, threads=1)
        np.testing.assert_array_equal(result, [0., 100.])


if __name__ == '__main__':
    unittest.main()
