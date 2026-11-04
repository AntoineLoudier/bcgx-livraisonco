import unittest
import numpy as np
import pandas as pd

from main_event_restrictions import future_scenario, learned_factor
from src.features import ARCS
from src.announced_events import add_announced_features, factors, load_events


class AnnouncedEventsTests(unittest.TestCase):
    def test_publication_time_scope_and_fractional_hours(self):
        event = pd.DataFrame([{'arc': 'A', 'scope': 'direct', 'event_id': 'e', 'availability': 'verified',
                               'start': pd.Timestamp('2026-01-10 10:30'), 'end': pd.Timestamp('2026-01-10 12:15'),
                               'known_at': pd.Timestamp('2026-01-08 10:00')}])
        frame = pd.DataFrame({'arc': ['A', 'A', 'A', 'B'],
                              't_1h': pd.to_datetime(['2026-01-10 10:00', '2026-01-10 11:00', '2026-01-10 12:00', '2026-01-10 11:00'])})
        before = add_announced_features(frame, event, [pd.Timestamp('2026-01-08 09:59')] * 4)
        after = add_announced_features(frame, event, [pd.Timestamp('2026-01-08 10:00')] * 4)
        np.testing.assert_array_equal(before.known_event_direct, [0, 0, 0, 0])
        np.testing.assert_allclose(after.known_event_direct, [.5, 1, .25, 0])
        np.testing.assert_allclose(factors(frame, event, [pd.Timestamp('2026-01-08 10:00')] * 4, 'lower'), [.75, .5, .875, 1])
        event['availability'] = 'inferred'
        unverified = add_announced_features(frame, event, [pd.Timestamp('2026-01-08 10:00')] * 4)
        self.assertEqual(unverified.known_event_direct.sum(), 0)

    def test_calibration_excludes_future_events_and_imputed(self):
        rows = [{'arc': 'A', 'scope': 'direct', 'event_id': x, 'event_end': pd.Timestamp(end),
                 'q': truth, 'q_reference': 100., 'is_imputed_q': imputed}
                for x, end, truth, imputed in [('a', '2026-01-01', 50., 0), ('b', '2026-01-02', 50., 0),
                                                ('c', '2026-02-01', 999., 0), ('d', '2026-01-02', 999., 1)]]
        history = pd.DataFrame(rows)
        factor, events, obs = learned_factor(history, 'A', 'direct', 'q', pd.Timestamp('2026-01-03'), 2, 2)
        self.assertEqual((factor, events, obs), (.5, 2, 2))
        self.assertEqual(learned_factor(history, 'A', 'direct', 'q', pd.Timestamp('2026-01-01'), 2, 2)[0], 1.)

    def test_catalog_has_explicit_offsets_and_conservative_availability(self):
        events = load_events()
        self.assertEqual(len(events), 3)
        self.assertTrue((events.known_at < events.start).all())
        self.assertTrue(events.event_id.is_unique)

    def test_future_scenario_freezes_available_traffic(self):
        times = pd.date_range('2026-08-01', '2026-11-08 23:00', freq='h')
        frame = pd.MultiIndex.from_product([ARCS, times], names=['arc', 't_1h']).to_frame(index=False)
        frame['q'] = 100. + np.arange(len(frame)) % 100
        frame['k'] = 5. + np.arange(len(frame)) % 5
        frame['is_imputed_q'] = 0
        frame['is_imputed_k'] = 0
        cutoff = pd.Timestamp('2026-11-04')
        changed = frame.copy()
        changed.loc[changed.t_1h > cutoff, ['q', 'k']] = 999999
        first = future_scenario(frame, load_events(), cutoff)
        second = future_scenario(changed, load_events(), cutoff)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first[first.arc.eq('Convention')].last_real_q.max(), cutoff)


if __name__ == '__main__':
    unittest.main()
