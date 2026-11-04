import unittest
import pandas as pd

from main_bias_calibration import calibrate


def sample():
    rows = []
    for i in range(7):
        start = pd.Timestamp('2026-01-01') + pd.Timedelta(days=7*i)
        for hour in range(2):
            rows.append({'window_start': start, 'forecast_cutoff': start - pd.Timedelta(hours=1), 't_1h': start + pd.Timedelta(hours=hour), 'arc': 'A', 'q': 20., 'q_model': 10., 'k': 110., 'k_model': 95., 'is_imputed_q': 0, 'is_imputed_k': 0, 'horizon_day': 1, 'is_public_holiday': 0})
    return pd.DataFrame(rows)


class CalibrationTests(unittest.TestCase):
    def test_constant_bias_and_bounds(self):
        predictions, meta = calibrate(sample(), min_windows=4, min_observations=8)
        last = predictions[(predictions.window_start == sample().window_start.max()) & (predictions.method == 'bias_expanding')]
        self.assertTrue((last[last.target == 'q'].prediction == 20).all())
        self.assertTrue((last[last.target == 'k'].prediction == 100).all())
        self.assertEqual(meta.iloc[0].correction, 0)
        self.assertTrue(meta.iloc[0].method == 'reference')
        self.assertTrue(meta[(meta.method == 'bias_recent') & (meta.window_start == sample().window_start.min())].fallback.all())

    def test_future_and_imputed_excluded(self):
        frame = sample()
        _, original = calibrate(frame, min_windows=1, min_observations=1)
        changed = frame.copy()
        changed.loc[changed.window_start == changed.window_start.max(), 'q'] = 9999
        mask = (changed.window_start == changed.window_start.min()) & (changed.t_1h.dt.hour == 0)
        changed.loc[mask, 'is_imputed_q'] = 1
        changed.loc[mask, 'q'] = 100000
        _, modified = calibrate(changed, min_windows=1, min_observations=1)
        past = original.window_start.lt(frame.window_start.max())
        # Une valeur imputée extrême ne modifie pas la médiane des résidus réels.
        self.assertTrue(original.loc[past, 'correction'].equals(modified.loc[past, 'correction']))
        self.assertTrue((modified.n_observations <= original.n_observations).all())
        row = modified[(modified.window_start == frame.window_start.max()) & (modified.target == 'q') & (modified.method == 'bias_expanding')].iloc[0]
        self.assertEqual(row.correction, 10.)
        self.assertEqual(row.n_observations, 11)

    def test_incomplete_window_and_recent_global(self):
        frame = sample()
        # La fenêtre précédente commence avant l'émission mais ne sera pas finie.
        final = frame.window_start.max()
        frame.loc[frame.window_start == final, 'forecast_cutoff'] = final - pd.Timedelta(days=5)
        _, meta = calibrate(frame, min_windows=1, min_observations=1)
        row = meta[(meta.window_start == final) & (meta.target == 'q') & (meta.method == 'bias_expanding')].iloc[0]
        self.assertEqual(row.n_windows, 5)

    def test_recent_uses_last_eight_calendar_windows(self):
        frame = pd.concat([sample().assign(window_start=lambda d: d.window_start + pd.Timedelta(days=49), forecast_cutoff=lambda d: d.forecast_cutoff + pd.Timedelta(days=49), t_1h=lambda d: d.t_1h + pd.Timedelta(days=49)), sample()], ignore_index=True)
        frame.loc[frame.window_start.gt(frame.window_start.min()) & frame.window_start.lt(frame.window_start.max()), 'is_imputed_q'] = 1
        _, meta = calibrate(frame, min_windows=1, min_observations=1)
        last = meta[(meta.window_start == frame.window_start.max()) & (meta.target == 'q')]
        self.assertFalse(last[last.method == 'bias_expanding'].iloc[0].fallback)
        self.assertTrue(last[last.method == 'bias_recent'].iloc[0].fallback)


if __name__ == '__main__':
    unittest.main()
