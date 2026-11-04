"""Restrictions annoncées : validation chronologique et scénarios conditionnels."""
import argparse
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.announced_events import EVENT_FEATURES, add_announced_features, load_events
from src.evaluate import build_window
from src.features import ARCS
from src.model import feature_columns, multi_horizon_training_features

OUT = Path('data/processed/announced_events_analysis')


def issue_times(frame):
    return frame.t_1h - pd.to_timedelta(frame.horizon_hours, unit='h')


def fit(train, target, use_events):
    valid = train[target].notna() & train[f'is_imputed_{target}'].eq(0)
    cols = feature_columns(target, multi_horizon=True) + (EVENT_FEATURES if use_events else [])
    model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=250, learning_rate=.04,
                              num_leaves=15, min_child_samples=80, colsample_bytree=.9,
                              random_state=42, n_jobs=2, verbosity=-1)
    model.fit(train.loc[valid, cols], train.loc[valid, target])
    return model, cols


def window_predictions(frame, start, events, knowledge_cutoff=None):
    cutoff = start - pd.Timedelta(hours=1)
    train = multi_horizon_training_features(frame, cutoff)
    window = build_window(frame, start)
    train_issue = issue_times(train)
    if knowledge_cutoff is not None:
        train_issue = train_issue.clip(upper=knowledge_cutoff)
    train = add_announced_features(train, events, train_issue)
    known_at = min(cutoff, knowledge_cutoff) if knowledge_cutoff is not None else cutoff
    window = add_announced_features(window, events, np.repeat(known_at, len(window)))
    for target in ('q', 'k'):
        for name, enriched in [('reference', False), ('event_model', True)]:
            model, columns = fit(train, target, enriched)
            window[f'{target}_{name}'] = np.clip(model.predict(window[columns]), 0, 100 if target == 'k' else None)
    return window


def weighted_median(values, weights):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2, side='left')])


def learned_factor(history, arc, scope, target, cutoff, min_events=2, min_observations=20):
    """Calibration seulement sur événements terminés, même rue, même portée et vraie cible."""
    if history.empty:
        return 1.0, 0, 0
    valid = (history.arc.eq(arc) & history.scope.eq(scope) & history.event_end.le(cutoff)
             & history[target].notna() & history[f'is_imputed_{target}'].eq(0)
             & history[f'{target}_reference'].gt(0))
    data = history.loc[valid]
    n_events, n = data.event_id.nunique(), len(data)
    if n_events < min_events or n < min_observations:
        return 1.0, n_events, n
    return np.clip(weighted_median((data[target] / data[f'{target}_reference']).to_numpy(),
                                  data[f'{target}_reference'].to_numpy()), 0, 2), n_events, n


def evaluate_events(frame, events):
    predictions, history, calibration = [], pd.DataFrame(), []
    # Fenêtres prédéfinies avant calcul, une émission 23 h la veille.
    starts = {'armistice_2025_ce': '2025-11-07', 'marathon_2026_ce': '2026-04-09',
              'course_15_2026_convention': '2026-05-13'}
    for event in events.itertuples(index=False):
        start = pd.Timestamp(starts[event.event_id])
        cutoff = start - pd.Timedelta(hours=1)
        print(f'{event.event_id} : émission {cutoff}', flush=True)
        window = window_predictions(frame, start, events)
        subset = window[window.arc.eq(event.arc)].copy()
        begin = subset.t_1h
        subset['event_overlap'] = ((begin + pd.Timedelta(hours=1)).clip(upper=event.end) - begin.clip(lower=event.start)).dt.total_seconds().div(3600).clip(lower=0, upper=1)
        subset['confirmed_known_at_issue'] = (event.availability == 'verified' and event.known_at <= cutoff)
        subset['event_id'] = event.event_id
        subset['scope'] = event.scope
        subset['event_end'] = event.end
        for target in ('q', 'k'):
            weight, n_events, n_obs = learned_factor(history, event.arc, event.scope, target, cutoff)
            confirmed = event.availability == 'verified' and event.known_at <= cutoff
            if not confirmed:
                weight = 1.0
            subset[f'{target}_learned'] = subset[f'{target}_reference'] * (1 + (weight - 1) * subset.event_overlap)
            subset[f'{target}_scenario_lower'] = subset[f'{target}_reference']
            subset[f'{target}_scenario_upper'] = subset[f'{target}_reference']
            if target == 'q':
                lower = .5 if event.scope == 'direct' else .8
                upper = 1. if event.scope == 'direct' else 1.2
                subset[f'{target}_scenario_lower'] *= 1 + (lower - 1) * subset.event_overlap
                subset[f'{target}_scenario_upper'] *= 1 + (upper - 1) * subset.event_overlap
            subset[f'{target}_learned'] = subset[f'{target}_learned'].clip(0, 100 if target == 'k' else None)
            calibration.append({'event_id': event.event_id, 'cutoff': cutoff, 'arc': event.arc, 'scope': event.scope,
                                'target': target, 'factor': weight, 'past_events': n_events,
                                'past_observations': n_obs, 'fallback': not confirmed or n_events < 2 or n_obs < 20,
                                'event_availability_verified': confirmed})
        predictions.append(subset)
        history = pd.concat([history, subset.loc[subset.event_overlap.gt(0) & subset.confirmed_known_at_issue]], ignore_index=True)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(calibration)


def score(predictions, keys):
    rows = []
    for key, group in predictions.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for target in ('q', 'k'):
            valid = group[target].notna() & group[f'is_imputed_{target}'].eq(0)
            for method in ('reference', 'event_model', 'learned', 'scenario_lower', 'scenario_upper'):
                valid &= group[f'{target}_{method}'].notna()
            part = group.loc[valid]
            row = dict(zip(keys, key)) | {'target': target, 'n': len(part)}
            for method in ('reference', 'event_model', 'learned', 'scenario_lower', 'scenario_upper'):
                error = part[f'{target}_{method}'] - part[target]
                row[method + '_mae'] = error.abs().mean()
                row[method + '_bias'] = error.mean()
            rows.append(row)
    return pd.DataFrame(rows)


def extend_future(frame, end):
    latest = frame.t_1h.max()
    hours = pd.date_range(latest + pd.Timedelta(hours=1), end, freq='h')
    if hours.empty:
        return frame.copy()
    future = pd.MultiIndex.from_product([ARCS, hours], names=['arc', 't_1h']).to_frame(index=False)
    for target in ('q', 'k'):
        future[target] = np.nan
        future[f'is_imputed_{target}'] = 1
    return pd.concat([frame, future], ignore_index=True).sort_values(['arc', 't_1h']).reset_index(drop=True)


def future_scenario(frame, events, as_of):
    """Émission planifiée du 6 novembre, strictement figée aux données connues à as_of."""
    start, planned_issue = pd.Timestamp('2026-11-07'), pd.Timestamp('2026-11-06 23:00')
    as_of = min(pd.Timestamp(as_of), planned_issue)
    historical = frame.loc[frame.t_1h.le(as_of)].copy()
    grid = extend_future(historical, pd.Timestamp('2026-11-11 23:00'))
    window = window_predictions(grid, start, events, knowledge_cutoff=as_of)
    window['scenario_id'] = 'hypothese_armistice_ce_2026'
    hypothetical = (window.arc.eq('Champs-Elysées') & window.t_1h.between('2026-11-11 07:00', '2026-11-11 13:00'))
    # Aucun avis 2026 n'est inventé : 50 % est une hypothèse purement illustrative.
    window['q_hypothesis_minus50'] = window.q_reference * np.where(hypothetical, .5, 1.)
    window['hypothesis_active'] = hypothetical.astype(int)
    window['planned_issue'] = planned_issue
    window['knowledge_as_of'] = as_of
    for target in ('q', 'k'):
        observed = historical[historical[f'is_imputed_{target}'].eq(0) & historical[target].notna()]
        window[f'last_real_{target}'] = window.arc.map(observed.groupby('arc').t_1h.max())
    columns = ['scenario_id', 'arc', 't_1h', 'planned_issue', 'knowledge_as_of', 'last_real_q', 'last_real_k',
               'known_event_direct', 'known_event_adjacent', 'hypothesis_active',
               'q_reference', 'q_event_model', 'q_hypothesis_minus50', 'k_reference', 'k_event_model']
    return window[columns]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traffic', type=Path, default=Path('data/processed/traffic_clean.csv'))
    parser.add_argument('--events', type=Path, default=Path('configs/evenements_cibles.json'))
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--as-of', type=pd.Timestamp, default=pd.Timestamp('2026-11-04 00:00'))
    args = parser.parse_args()
    frame = pd.read_csv(args.traffic, parse_dates=['t_1h'])
    events = load_events(args.events)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions, calibration = evaluate_events(frame, events)
    predictions.to_csv(args.output / 'event_predictions.csv', index=False)
    calibration.to_csv(args.output / 'calibration.csv', index=False)
    predictions['during_event'] = predictions.event_overlap.gt(0)
    score(predictions, ['event_id', 'arc', 'scope', 'during_event']).to_csv(args.output / 'event_scores.csv', index=False)
    future_scenario(frame, events, args.as_of).to_csv(args.output / 'forecast_scenarios_2026.csv', index=False)
    print(score(predictions, ['event_id', 'arc', 'scope', 'during_event']).to_string(index=False))


if __name__ == '__main__':
    main()
