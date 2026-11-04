"""Compare le contexte récent et des modèles par horizon avec délais réalistes."""
import argparse
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.evaluate import build_window
from src.model import feature_columns, multi_horizon_training_features
from src.recent_context import attach_context, context_columns, context_table

METHODS = ('reference', 'context', 'direct')
DELAYS = (0, 24, 72, 120)


def choose_starts(frame, count):
    """Fenêtres espacées d'au moins cinq jours, avec débuts de semaine variés."""
    last = frame.t_1h.max().normalize() - pd.Timedelta(days=5)
    # Choisir le samedi le plus récent puis remonter en variant le jour de début.
    last -= pd.Timedelta(days=(last.dayofweek - 5) % 7)
    return sorted(last - pd.Timedelta(days=8*i) for i in range(count))


def fit(train, target, columns):
    real = train[target].notna() & train[f'is_imputed_{target}'].eq(0)
    train = train.loc[real]
    if len(train) < 100:
        raise ValueError(f'Historique insuffisant pour {target}')
    model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=250, learning_rate=.04,
                              num_leaves=15, min_child_samples=80, colsample_bytree=.9,
                              random_state=42, n_jobs=2, verbosity=-1)
    model.fit(train[columns], train[target])
    return model


def forecast(model, x, columns, target):
    return np.clip(model.predict(x[columns]), 0, 100 if target == 'k' else None)


def run(frame, starts, delays, output):
    contexts = context_table(frame)
    all_predictions = []
    for delay in delays:
        # Toutes les variables sont causales à l'émission propre de chaque exemple.
        training = multi_horizon_training_features(frame, frame.t_1h.max(), availability_delay_hours=delay)
        training = attach_context(training, contexts, delay)
        for start in starts:
            checkpoint = output / 'checkpoints' / f'{delay}h_{start:%Y-%m-%d}.csv'
            if checkpoint.exists():
                print(f'Reprise : {checkpoint}', flush=True)
                all_predictions.append(pd.read_csv(checkpoint, parse_dates=['window_start', 'forecast_cutoff', 't_1h']))
                continue
            cutoff = start - pd.Timedelta(hours=1)
            available = cutoff - pd.Timedelta(hours=delay)
            train = training.loc[training.t_1h <= available]
            validation = attach_context(build_window(frame, start, availability_delay_hours=delay), contexts, delay)
            validation['horizon_day'] = (validation.horizon_hours - 1) // 24 + 1
            validation['delay_hours'] = delay
            print(f'Délai {delay} h, fenêtre {start.date()}', flush=True)
            for target in ('q', 'k'):
                base_cols = feature_columns(target, multi_horizon=True)
                context_cols = base_cols + context_columns(target)
                validation[f'{target}_reference'] = forecast(fit(train, target, base_cols), validation, base_cols, target)
                validation[f'{target}_context'] = forecast(fit(train, target, context_cols), validation, context_cols, target)
                validation[f'{target}_direct'] = np.nan
                days = (train.horizon_hours - 1) // 24 + 1
                for day in range(1, 6):
                    mask = validation.horizon_day.eq(day)
                    validation.loc[mask, f'{target}_direct'] = forecast(fit(train.loc[days.eq(day)], target, context_cols), validation.loc[mask], context_cols, target)
            cols = ['window_start', 'forecast_cutoff', 't_1h', 'arc', 'horizon_day', 'delay_hours', 'is_public_holiday', 'q', 'k', 'is_imputed_q', 'is_imputed_k']
            cols += [f'{target}_{method}' for target in ('q', 'k') for method in METHODS]
            result = validation[cols]
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(checkpoint, index=False)
            all_predictions.append(result)
    return pd.concat(all_predictions, ignore_index=True)


def scores(predictions, keys):
    rows = []
    for values, g in predictions.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        for target in ('q', 'k'):
            valid = g[target].notna() & g[f'is_imputed_{target}'].eq(0)
            for method in METHODS:
                valid &= g[f'{target}_{method}'].notna()
            data = g.loc[valid]
            row = dict(zip(keys, values)) | {'target': target, 'n': len(data)}
            for method in METHODS:
                error = data[f'{target}_{method}'] - data[target]
                row[f'{method}_mae'] = error.abs().mean()
                row[f'{method}_bias'] = error.mean()
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap(predictions, iterations=2000):
    """Intervalle exploratoire apparié en rééchantillonnant les fenêtres."""
    rng = np.random.default_rng(42)
    rows = []
    for (delay, arc), g in predictions.groupby(['delay_hours', 'arc']):
        for target in ('q', 'k'):
            g2 = g[g[target].notna() & g[f'is_imputed_{target}'].eq(0)].copy()
            for method in ('context', 'direct'):
                grouped = []
                for _, window in g2.groupby('window_start'):
                    delta = (window[f'{target}_{method}'] - window[target]).abs() - (window[f'{target}_reference'] - window[target]).abs()
                    grouped.append((delta.sum(), delta.count()))
                if len(grouped) < 2:
                    continue
                a = np.asarray(grouped)
                picks = rng.integers(0, len(a), size=(iterations, len(a)))
                samples = a[picks, 0].sum(axis=1) / a[picks, 1].sum(axis=1)
                lo, hi = np.quantile(samples, [.025, .975])
                rows.append({'delay_hours': delay, 'arc': arc, 'target': target, 'method': method, 'windows': len(a), 'delta_mae_lo': lo, 'delta_mae_hi': hi})
    return pd.DataFrame(rows)


def write_reports(pred, output):
    pred['start_weekday'] = pred.window_start.dt.dayofweek
    output.mkdir(parents=True, exist_ok=True)
    pred.to_csv(output / 'predictions.csv', index=False)
    for name, keys in [('summary', ['delay_hours', 'arc']), ('windows', ['delay_hours', 'arc', 'window_start']), ('horizons', ['delay_hours', 'arc', 'horizon_day']), ('holidays', ['delay_hours', 'arc', 'is_public_holiday']), ('weekdays', ['delay_hours', 'arc', 'start_weekday'])]:
        scores(pred, keys).to_csv(output / f'{name}.csv', index=False)
    window_scores = scores(pred, ['delay_hours', 'arc', 'window_start'])
    diagnostics = []
    for keys, group in window_scores[window_scores.n.gt(0)].groupby(['delay_hours', 'arc', 'target']):
        row = dict(zip(['delay_hours', 'arc', 'target'], keys)) | {'windows': len(group)}
        for method in METHODS:
            row[f'{method}_worst_mae'] = group[f'{method}_mae'].max()
            if method != 'reference':
                row[f'{method}_wins'] = int((group[f'{method}_mae'] < group.reference_mae).sum())
        diagnostics.append(row)
    pd.DataFrame(diagnostics).to_csv(output / 'window_diagnostics.csv', index=False)
    bootstrap(pred).to_csv(output / 'bootstrap.csv', index=False)
    print(scores(pred, ['delay_hours', 'arc']).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traffic', type=Path, default=Path('data/processed/traffic_clean.csv'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/recent_context_analysis'))
    parser.add_argument('--windows', type=int, default=8)
    parser.add_argument('--event-start', type=pd.Timestamp, nargs='+', help='Fenêtres événementielles séparées ; ne pas mélanger au bootstrap ordinaire')
    parser.add_argument('--delays', type=int, nargs='+', choices=DELAYS, default=DELAYS)
    parser.add_argument('--summarize-only', action='store_true', help='Résumer les checkpoints existants, même si le calcul est incomplet')
    args = parser.parse_args()
    if args.summarize_only:
        paths = sorted((args.output / 'checkpoints').glob('*.csv'))
        if not paths:
            raise ValueError('Aucun checkpoint disponible')
        pred = pd.concat([pd.read_csv(path, parse_dates=['window_start', 'forecast_cutoff', 't_1h']) for path in paths], ignore_index=True)
    else:
        frame = pd.read_csv(args.traffic, parse_dates=['t_1h'])
        starts = sorted(args.event_start) if args.event_start else choose_starts(frame, args.windows)
        pred = run(frame, starts, args.delays, args.output)
    write_reports(pred, args.output)


if __name__ == '__main__':
    main()
