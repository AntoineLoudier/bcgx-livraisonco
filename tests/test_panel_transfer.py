import unittest
import numpy as np
import pandas as pd

from main_panel_transfer import normalized, sample_training_features, scale_by_arc
from src.evaluate import build_window
from src.features import ARCS
from src.model import feature_columns
from src.panel_preprocessing import prepare_panel
from src.panel_neighbors import directed_links, neighbor_table


class PanelTests(unittest.TestCase):
    def test_scale_and_features_never_use_future_values(self):
        times = pd.date_range('2026-01-01', periods=24 * 45, freq='h')
        frame = pd.MultiIndex.from_product([ARCS, times], names=['arc', 't_1h']).to_frame(index=False)
        frame['q'], frame['k'] = 100., 5.
        frame['is_imputed_q'], frame['is_imputed_k'] = 0, 0
        frame['is_road_closed'] = 0
        start = pd.Timestamp('2026-02-01')
        cutoff = start - pd.Timedelta(hours=1)
        altered = frame.copy()
        altered.loc[altered.t_1h > cutoff, ['q', 'k']] = 99999
        a = scale_by_arc(frame, cutoff, 'q')
        b = scale_by_arc(altered, cutoff, 'q')
        pd.testing.assert_series_equal(a, b)
        pd.testing.assert_frame_equal(normalized(build_window(frame, start), 'q', a)[feature_columns('q', True)],
                                      normalized(build_window(altered, start), 'q', b)[feature_columns('q', True)])
        training = sample_training_features(frame, cutoff)
        self.assertLessEqual(training.t_1h.max(), cutoff)
        self.assertFalse(training.duplicated(['arc', 't_1h']).any())

    def test_identity_changes_are_excluded_not_mixed(self):
        start = pd.Timestamp('2026-01-01')
        selection = [{'iu_ac': '12', 'libelle_nd_amont': 'A', 'libelle_nd_aval': 'B'}]
        data = pd.DataFrame({'iu_ac': ['12', '12'], 't_1h': ['2025-12-31T23:00:00Z', '2026-01-01T00:00:00Z'],
                             'libelle_nd_amont': ['A', 'B'], 'libelle_nd_aval': ['B', 'A'],
                             'q': [100., 900.], 'k': [5., 9.], 'etat_barre': ['Ouvert', 'Ouvert']})
        panel, audit = prepare_panel(data, selection, start, start + pd.Timedelta(hours=1))
        self.assertEqual(audit.matching_rows.iloc[0], 1)
        self.assertEqual(panel.loc[panel.t_1h.eq(start), 'q'].iloc[0], 100.)
        self.assertEqual(panel.loc[panel.t_1h.eq(start + pd.Timedelta(hours=1)), 'is_imputed_q'].iloc[0], 1)

    def test_neighbors_are_directional_and_causal(self):
        selection = [
            {'iu_ac': '1', 'libelle_nd_amont': 'foo', 'libelle_nd_aval': 'Lecourbe-Convention'},
            {'iu_ac': '2', 'libelle_nd_amont': 'Lecourbe-Convention', 'libelle_nd_aval': 'foo'},
            {'iu_ac': '3', 'libelle_nd_amont': 'Convention-Blomet', 'libelle_nd_aval': 'Lecourbe-Convention'},
        ]
        links = directed_links(selection)
        self.assertEqual(links.loc[links.arc.eq('Convention'), 'neighbor'].tolist(), ['panel_1'])
        times = pd.date_range('2026-01-01', periods=200, freq='h')
        frame = pd.DataFrame({'arc': 'panel_1', 't_1h': times, 'q': 100., 'k': 5.,
                              'is_imputed_q': 0, 'is_imputed_k': 0})
        changed = frame.copy()
        changed.loc[changed.t_1h > times[150], ['q', 'k']] = 9999
        before = neighbor_table(frame, links).set_index('t_1h').loc[times[150]]
        after = neighbor_table(changed, links).set_index('t_1h').loc[times[150]]
        pd.testing.assert_series_equal(before, after)


if __name__ == '__main__':
    unittest.main()
