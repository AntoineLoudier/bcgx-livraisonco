"""Ablation des informations de capteurs adjacents, hors entraînement futur."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from main_new_experiments import simulate_outage
from main_panel_transfer import normalized, sample_training_features, scale_by_arc
from main_recent_context import choose_starts
from src.evaluate import build_window
from src.features import ARCS
from src.model import feature_columns
from src.panel_neighbors import attach_neighbors, directed_links, neighbor_columns, neighbor_table


def evaluate(targets, panel, neighbors, links, start, outage_arc=None, outage_days=0):
    cutoff = start - pd.Timedelta(hours=1)
    historical = simulate_outage(targets, outage_arc, cutoff, outage_days) if outage_arc else targets
    cols = ['arc', 't_1h', 'q', 'k', 'is_imputed_q', 'is_imputed_k', 'is_road_closed']
    all_data = pd.concat([historical[cols], panel[cols]], ignore_index=True).sort_values(['arc', 't_1h']).reset_index(drop=True)
    table = neighbor_table(neighbors, links)
    for col in neighbor_columns():
        if col not in table:
            table[col] = np.nan
    train = attach_neighbors(sample_training_features(all_data, cutoff), table)
    validation = attach_neighbors(build_window(historical, start), table)
    truth = targets[['arc', 't_1h', 'q', 'k', 'is_imputed_q', 'is_imputed_k']]
    validation = validation.drop(columns=['q', 'k', 'is_imputed_q', 'is_imputed_k']).merge(truth, on=['arc', 't_1h'], validate='one_to_one')
    if outage_arc:
        validation = validation[validation.arc.eq(outage_arc)].copy()
    output = validation[['window_start', 'arc', 't_1h', 'q', 'k', 'is_imputed_q', 'is_imputed_k']].copy()
    for target in ('q', 'k'):
        scales = scale_by_arc(all_data, cutoff, target)
        x, y = normalized(train, target, scales), normalized(validation, target, scales)
        x = x[x[target].notna() & x[f'is_imputed_{target}'].eq(0) & x._scale.notna()].copy()
        weights = (1 / x.groupby('arc')[target].transform('size')) * np.where(x.arc.isin(ARCS), 4., 1.)
        weights /= weights.mean()
        for method in ('adapted', 'neighbors'):
            cols = feature_columns(target, True) + (neighbor_columns() if method == 'neighbors' else [])
            model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=250, learning_rate=.04,
                                      num_leaves=15, min_child_samples=80, colsample_bytree=.9,
                                      random_state=42, n_jobs=2, verbosity=-1)
            model.fit(x[cols], x[target] / x._scale, sample_weight=weights)
            valid = y._scale.notna()
            prediction = np.full(len(y), np.nan)
            prediction[valid.to_numpy()] = np.clip(model.predict(y.loc[valid, cols]) * y.loc[valid, '_scale'], 0, 100 if target == 'k' else None)
            output[f'{target}_{method}'] = prediction
    output['outage_arc'], output['outage_days'] = outage_arc or 'none', outage_days
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=int, default=2)
    parser.add_argument('--outages', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('data/processed/panel_neighbors'))
    args = parser.parse_args()
    targets = pd.read_csv('data/processed/traffic_clean.csv', parse_dates=['t_1h'])
    panel = pd.read_csv('data/processed/panel/traffic_panel_clean.csv', parse_dates=['t_1h'])
    neighbors = pd.read_csv('data/processed/panel/traffic_neighbors_clean.csv', parse_dates=['t_1h'])
    selection = json.loads(Path('data/raw/panel/neighbors_selection.json').read_text())['selection']
    links = directed_links(selection)
    args.output.mkdir(parents=True, exist_ok=True)
    links.to_csv(args.output / 'directed_links.csv', index=False)
    starts = choose_starts(targets, args.windows)
    cases = [(start, None, 0) for start in starts]
    if args.outages:
        cases += [(starts[-1], arc, 180) for arc in ('Convention', 'Saint-Pères')]
    predictions = []
    for start, arc, days in cases:
        file = args.output / f'checkpoint_{start:%Y%m%d}_{arc or "normal"}_{days}.csv'
        if file.exists():
            result = pd.read_csv(file, parse_dates=['t_1h', 'window_start'])
        else:
            print(f'Voisins : {start.date()}, {arc or "normal"}, panne {days} jours', flush=True)
            result = evaluate(targets, panel, neighbors, links, start, arc, days)
            result.to_csv(file, index=False)
        predictions.append(result)
        data = pd.concat(predictions, ignore_index=True)
        data.to_csv(args.output / 'predictions.csv', index=False)
        rows = []
        for key, group in data.groupby(['outage_arc', 'outage_days', 'arc'], dropna=False):
            for target in ('q', 'k'):
                valid = group[target].notna() & group[f'is_imputed_{target}'].eq(0)
                valid &= group[f'{target}_adapted'].notna() & group[f'{target}_neighbors'].notna()
                row = dict(zip(['outage_arc', 'outage_days', 'arc'], key)) | {'target': target, 'n': int(valid.sum())}
                for method in ('adapted', 'neighbors'):
                    row[method + '_mae'] = (group.loc[valid, f'{target}_{method}'] - group.loc[valid, target]).abs().mean()
                rows.append(row)
        pd.DataFrame(rows).to_csv(args.output / 'summary.csv', index=False)


if __name__ == '__main__':
    main()
