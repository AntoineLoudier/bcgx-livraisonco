import unittest
import numpy as np
import pandas as pd

from main_new_experiments import calibrate_intervals, simulate_outage
from main_error_diagnostics import analyse, days
from src.features import add_calendar_and_event_features, add_causal_lags_and_sncf_patterns


class NewExperimentsTest(unittest.TestCase):
    def test_outage_rebuilds_imputations_without_hidden_measurements(self):
        times = pd.date_range('2026-01-01', periods=24 * 30, freq='h')
        frame = pd.DataFrame({'arc': 'Convention', 't_1h': times, 'q': 10., 'k': 2.,
                              'is_imputed_q': 0, 'is_imputed_k': 0})
        cutoff = times[-1]
        frame.loc[frame.t_1h > cutoff - pd.Timedelta(days=7), 'q'] = 1000
        damaged = simulate_outage(frame, 'Convention', cutoff, 7)
        masked = damaged.t_1h > cutoff - pd.Timedelta(days=7)
        self.assertTrue(damaged.loc[masked, 'is_imputed_q'].eq(1).all())
        self.assertTrue(damaged.loc[masked, 'q'].eq(10).all())

    def test_interval_uses_only_completed_prior_windows(self):
        start = pd.Timestamp('2026-01-01')
        entries = []
        for offset, residual in [(0, 1.), (7, 999.), (14, 0.)]:
            window = start + pd.Timedelta(days=offset)
            entries.append({'window_start': window, 't_1h': window, 'arc': 'A', 'target': 'q',
                            'horizon_hours': 1, 'truth': residual + 10, 'prediction': 10., 'variant': 'reference'})
        data = pd.DataFrame(entries)
        # A future residual cannot influence intervals for the second window.
        intervals = calibrate_intervals(data, min_count=1)
        second = intervals[intervals.window_start.eq(start + pd.Timedelta(days=7))].iloc[0]
        self.assertEqual(second['lower'], 11.)
        self.assertEqual(second['upper'], 11.)
        self.assertEqual(second.calibration_n, 1)

    def test_daily_totals_require_24_observed_hours(self):
        frame = pd.DataFrame({'window_start': [pd.Timestamp('2026-01-01')], 'arc': ['A'], 'target': ['q'],
                              'date': [pd.Timestamp('2026-01-01')], 't_1h': [pd.Timestamp('2026-01-01')],
                              'truth': [2.], 'prediction': [1.], 'error': [-1.], 'abs_error': [1.]})
        day = days(frame).iloc[0]
        self.assertTrue(np.isnan(day.truth_total))
        self.assertTrue(np.isnan(day.peak_shift_hours))

    def test_real_lags_exclude_imputation_for_both_targets(self):
        times = pd.date_range('2026-01-01', periods=24 * 25, freq='h')
        frame = pd.DataFrame({'arc': 'Convention', 't_1h': times, 'q': 10., 'k': 2.,
                              'is_imputed_q': 0, 'is_imputed_k': 0})
        frame.loc[frame.t_1h.eq(pd.Timestamp('2026-01-10 12:00')), ['q', 'is_imputed_q']] = [999, 1]
        features = add_causal_lags_and_sncf_patterns(add_calendar_and_event_features(frame), training_horizon_days=1, real_lags_only=True)
        row = features[features.t_1h.eq(pd.Timestamp('2026-01-17 12:00'))].iloc[0]
        self.assertTrue(np.isnan(row.q_lag_168h))
        self.assertEqual(row.q_lag_168h_real, 0)
        self.assertEqual(row.k_lag_168h, 2.)
        self.assertEqual(row.k_lag_168h_real, 1)


if __name__ == '__main__':
    unittest.main()
