"""Validation chronologique d'un modèle global élargi et de son adaptation."""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate import build_window
from src.features import ARCS, add_calendar_and_event_features, add_causal_lags_and_sncf_patterns
from src.model import feature_columns, fit_model, multi_horizon_training_features, predict_model
from main_recent_context import choose_starts
from main_new_experiments import simulate_outage

OUT = Path('data/processed/panel_transfer')
METHODS = ('reference', 'targets_sampled', 'panel', 'panel_adapted')


def sample_training_features(frame, cutoff):
    """Un horizon simulé par jour cible, partagé par toutes les variantes normalisées."""
    history = frame.loc[frame.t_1h <= cutoff].copy()
    calendar = add_calendar_and_event_features(history)
    chunks = []
    for day in range(1, 6):
        featured = add_causal_lags_and_sncf_patterns(calendar, training_horizon_days=day)
        # Le feature builder trie les données : recalculer l'index sur sa sortie.
        index = (featured.t_1h.dt.normalize() - pd.Timestamp('1970-01-01')).dt.days
        chunks.append(featured.loc[index.mod(5).eq(day - 1)].copy())
    return pd.concat(chunks, ignore_index=True)


def scale_by_arc(frame, cutoff, target):
    observed = frame.t_1h.le(cutoff) & frame[target].notna() & frame[f'is_imputed_{target}'].eq(0)
    values = frame.loc[observed].groupby('arc')[target].agg(['median', 'count'])
    values = values[values['count'] >= 100]
    return values['median'].abs().clip(lower=10 if target == 'q' else 1)


def normalized(frame, target, scales):
    data = frame.copy()
    data['_scale'] = data.arc.map(scales)
    # Les tronçons auxiliaires partagent une identité distincte des trois cibles.
    data['arc_id'] = data.arc.map({arc: idx for idx, arc in enumerate(ARCS)}).fillna(3)
    cols = [f'{target}_{name}' for name in ('lag_168h', 'lag_336h', 'lag_504h', 'weekly_mean_3w', 'weekly_std_3w', 'regime_hist_mean', 'smart_expected')]
    data[cols] = data[cols].div(data['_scale'], axis=0)
    return data


def fit_normalized(train, target, scales, variant):
    train = normalized(train, target, scales)
    known = train[target].notna() & train[f'is_imputed_{target}'].eq(0) & train['_scale'].notna()
    if variant == 'targets_sampled':
        known &= train.arc.isin(ARCS)
    train = train.loc[known].copy()
    counts = train.groupby('arc')[target].transform('size')
    # Une rue contribue au même poids total, puis les cibles sont multipliées par 4.
    weights = 1 / counts
    if variant == 'panel_adapted':
        weights *= np.where(train.arc.isin(ARCS), 4.0, 1.0)
    weights /= weights.mean()
    model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=250, learning_rate=.04,
                             num_leaves=15, min_child_samples=80, colsample_bytree=.9,
                             random_state=42, n_jobs=2, verbosity=-1)
    model.fit(train[feature_columns(target, multi_horizon=True)], train[target] / train['_scale'], sample_weight=weights)
    return model, len(train)


def run_one(targets, panel, start, outage_arc=None, outage_days=0):
    cutoff = pd.Timestamp(start) - pd.Timedelta(hours=1)
    input_targets = simulate_outage(targets, outage_arc, cutoff, outage_days) if outage_arc else targets
    common = ['arc', 't_1h', 'q', 'k', 'is_imputed_q', 'is_imputed_k', 'is_road_closed']
    frame = pd.concat([input_targets[common], panel[common]], ignore_index=True).sort_values(['arc', 't_1h']).reset_index(drop=True)
    sampled = sample_training_features(frame, cutoff)
    baseline_train = multi_horizon_training_features(input_targets, cutoff)
    validation = build_window(input_targets, start)
    truth = targets[['arc', 't_1h', 'q', 'k', 'is_imputed_q', 'is_imputed_k']]
    validation = validation.drop(columns=['q', 'k', 'is_imputed_q', 'is_imputed_k']).merge(truth, on=['arc', 't_1h'], validate='one_to_one')
    if outage_arc:
        validation = validation[validation.arc.eq(outage_arc)].copy()
    validation['horizon_day'] = (validation.horizon_hours - 1) // 24 + 1
    cols = ['window_start', 'arc', 't_1h', 'horizon_day', 'is_public_holiday', 'q', 'k', 'is_imputed_q', 'is_imputed_k']
    output = validation[cols].copy()
    counts = []
    for target in ('q', 'k'):
        reference, n = fit_model(baseline_train, target, multi_horizon=True)
        output[f'{target}_reference'] = predict_model(reference, validation, target, multi_horizon=True)
        counts.append({'target': target, 'method': 'reference', 'examples': n})
        scales = scale_by_arc(frame, cutoff, target)
        x = normalized(validation, target, scales)
        valid = x['_scale'].notna()
        for method in METHODS[1:]:
            model, n = fit_normalized(sampled, target, scales, method)
            pred = np.full(len(x), np.nan)
            if valid.any():
                pred[valid.to_numpy()] = np.clip(model.predict(x.loc[valid, feature_columns(target, True)]) * x.loc[valid, '_scale'], 0, 100 if target == 'k' else None)
            output[f'{target}_{method}'] = pred
            arcs = len(scales.index.intersection(ARCS)) if method == 'targets_sampled' else scales.index.nunique()
            counts.append({'target': target, 'method': method, 'examples': n, 'arcs': int(arcs)})
    output['outage_arc'] = outage_arc or 'none'
    output['outage_days'] = outage_days
    return output, pd.DataFrame(counts).assign(window_start=start, outage_arc=outage_arc or 'none', outage_days=outage_days)


def scores(frame, keys):
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        for target in ('q', 'k'):
            valid = group[target].notna() & group[f'is_imputed_{target}'].eq(0)
            for method in METHODS:
                valid &= group[f'{target}_{method}'].notna()
            row = dict(zip(keys, key)) | {'target': target, 'n': int(valid.sum())}
            for method in METHODS:
                error = group.loc[valid, f'{target}_{method}'] - group.loc[valid, target]
                row[method + '_mae'] = error.abs().mean()
                row[method + '_bias'] = error.mean()
            rows.append(row)
    return pd.DataFrame(rows)


def window_diagnostics(frame):
    windows = scores(frame, ['outage_arc', 'outage_days', 'arc', 'window_start'])
    windows = windows[windows.n.gt(0)].copy()
    rows = []
    for key, group in windows.groupby(['outage_arc', 'outage_days', 'arc', 'target'], dropna=False):
        for method in METHODS[1:]:
            rows.append(dict(zip(['outage_arc', 'outage_days', 'arc', 'target'], key)) | {
                'method': method, 'windows': len(group),
                'wins': int((group[f'{method}_mae'] < group.reference_mae).sum()),
                'reference_worst_mae': group.reference_mae.max(), 'method_worst_mae': group[f'{method}_mae'].max()})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=int, default=4)
    parser.add_argument('--outages', action='store_true', help='Simule 90, 180 et 330 jours de panne sur une fenêtre')
    parser.add_argument('--event-start', type=pd.Timestamp, help='Fenêtre dédiée, dans un répertoire séparé')
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    targets = pd.read_csv('data/processed/traffic_clean.csv', parse_dates=['t_1h'])
    panel = pd.read_csv('data/processed/panel/traffic_panel_clean.csv', parse_dates=['t_1h'])
    starts = [args.event_start] if args.event_start is not None else choose_starts(targets, args.windows)
    scenarios = [(start, None, 0) for start in starts]
    if args.outages:
        scenarios += [(starts[-1], arc, days) for arc in ('Convention', 'Saint-Pères') for days in (90, 180, 330)]
    args.output.mkdir(parents=True, exist_ok=True)
    parts, counts = [], []
    for start, arc, days in scenarios:
        tag = f'{start:%Y%m%d}_{arc or "normal"}_{days}'
        path = args.output / f'checkpoint_{tag}.csv'
        count_path = args.output / f'counts_{tag}.csv'
        if path.exists() and count_path.exists():
            pred = pd.read_csv(path, parse_dates=['t_1h', 'window_start'])
            count = pd.read_csv(count_path)
        else:
            print(f'Fenêtre {start.date()} : {arc or "normal"}, panne {days} jours', flush=True)
            pred, count = run_one(targets, panel, start, arc, days)
            pred.to_csv(path, index=False)
            count.to_csv(count_path, index=False)
        parts.append(pred)
        counts.append(count)
        # Écrire également les résultats partiels immédiatement.
        combined = pd.concat(parts, ignore_index=True)
        combined.to_csv(args.output / 'predictions.csv', index=False)
        pd.concat(counts, ignore_index=True).to_csv(args.output / 'training_counts.csv', index=False)
        for name, keys in [('summary', ['outage_arc', 'outage_days', 'arc']), ('windows', ['outage_arc', 'outage_days', 'arc', 'window_start']), ('horizons', ['outage_arc', 'outage_days', 'arc', 'horizon_day']), ('holidays', ['outage_arc', 'outage_days', 'arc', 'is_public_holiday'])]:
            scores(combined, keys).to_csv(args.output / f'{name}.csv', index=False)
        window_diagnostics(combined).to_csv(args.output / 'window_diagnostics.csv', index=False)


if __name__ == '__main__':
    main()
