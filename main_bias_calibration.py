"""Calibration chronologique du biais des prévisions historiques hors entraînement."""
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path('data/processed/blend_analysis/predictions.csv')
OUTPUT = Path('data/processed/bias_calibration_analysis')
METHODS = ('reference', 'bias_expanding', 'bias_recent')


def calibrate(frame, min_windows=4, min_observations=100):
    frame = frame.copy()
    for col in ('t_1h', 'window_start', 'forecast_cutoff'):
        frame[col] = pd.to_datetime(frame[col])
    starts = sorted(frame.window_start.unique())
    output, metadata = [], []
    for start in starts:
        current = frame[frame.window_start.eq(start)]
        if current.forecast_cutoff.nunique() != 1:
            raise ValueError('Une fenêtre doit avoir une seule date d’émission')
        cutoff = current.forecast_cutoff.iloc[0]
        # Une fenêtre couvre exactement les 120 heures calendaires prévues.
        completed = [s for s in starts if s + pd.Timedelta(hours=119) <= cutoff and s < start]
        for arc in sorted(frame.arc.dropna().unique()):
            for target in ('q', 'k'):
                for method in METHODS:
                    eligible = completed[-8:] if method == 'bias_recent' else completed
                    history = frame[frame.window_start.isin(eligible) & frame.arc.eq(arc)]
                    history = history[history.t_1h.le(cutoff) & history[target].notna() & history[f'{target}_model'].notna() & history[f'is_imputed_{target}'].eq(0)]
                    n_windows = history.window_start.nunique()
                    enough = n_windows >= min_windows and len(history) >= min_observations
                    correction = float((history[target] - history[f'{target}_model']).median()) if enough and method != 'reference' else 0.0
                    rows = current[current.arc.eq(arc) & current[target].notna() & current[f'{target}_model'].notna() & current[f'is_imputed_{target}'].eq(0)].copy()
                    rows['target'] = target
                    rows['method'] = method
                    rows['truth'] = rows[target]
                    # La référence est elle aussi bornée afin de comparer les corrections seules.
                    rows['prediction'] = (rows[f'{target}_model'] + correction).clip(lower=0, upper=100 if target == 'k' else None)
                    rows['correction'] = correction
                    rows['calibrated'] = enough if method != 'reference' else False
                    rows['fallback'] = not enough if method != 'reference' else False
                    output.append(rows)
                    metadata.append({'window_start': start, 'forecast_cutoff': cutoff, 'arc': arc, 'target': target, 'method': method, 'correction': correction, 'n_windows': n_windows, 'n_observations': len(history), 'calibrated': enough if method != 'reference' else False, 'fallback': not enough if method != 'reference' else False, 'n_evaluated': len(rows)})
    return pd.concat(output, ignore_index=True), pd.DataFrame(metadata)


def scores(data, keys):
    d = data.copy()
    d['error'] = d.prediction - d.truth
    d['absolute_error'] = d.error.abs()
    d['squared_error'] = d.error ** 2
    result = d.groupby(keys, dropna=False).agg(n=('error', 'size'), mae=('absolute_error', 'mean'), mse=('squared_error', 'mean'), bias=('error', 'mean')).reset_index()
    result['rmse'] = np.sqrt(result.pop('mse'))
    return result


def compare(data):
    # Chaque méthode est comparée sur les mêmes lignes, y compris pour les sous-ensembles calibrés.
    ref = data[data.method.eq('reference')][['window_start', 'arc', 'target', 't_1h', 'prediction']].rename(columns={'prediction': 'reference_prediction'})
    candidates = data[~data.method.eq('reference')].merge(ref, on=['window_start', 'arc', 'target', 't_1h'], validate='many_to_one')
    pieces = []
    for method in ('bias_expanding', 'bias_recent'):
        subset = candidates[candidates.method.eq(method)]
        for status in ('all', 'calibrated', 'fallback'):
            rows = subset if status == 'all' else subset[subset[status]]
            for label in ('reference', method):
                p = rows.copy()
                if label == 'reference':
                    p['prediction'] = p.reference_prediction
                p['comparison'] = method
                p['method'] = label
                p['status'] = status
                pieces.append(p)
    return pd.concat(pieces, ignore_index=True)


def main():
    frame = pd.read_csv(SOURCE)
    data, metadata = calibrate(frame)
    comparable = compare(data)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT / 'predictions.csv', index=False)
    metadata.to_csv(OUTPUT / 'corrections.csv', index=False)
    base = ['comparison', 'status', 'arc', 'target', 'method']
    for name, extra in {'summary': [], 'windows': ['window_start'], 'horizons': ['horizon_day'], 'holidays': ['is_public_holiday']}.items():
        scores(comparable, base + extra).to_csv(OUTPUT / f'{name}.csv', index=False)
    windows = scores(comparable, base + ['window_start'])
    ref = windows[windows.method.eq('reference')].rename(columns={'mae': 'reference_mae'})
    candidate = windows[~windows.method.eq('reference')].merge(ref[base[:-1] + ['window_start', 'reference_mae']], on=base[:-1] + ['window_start'], validate='one_to_one')
    candidate['won'] = candidate.mae.lt(candidate.reference_mae)
    candidate['lost'] = candidate.mae.gt(candidate.reference_mae)
    candidate.groupby(base[:-1], dropna=False).agg(windows=('mae', 'size'), wins=('won', 'sum'), losses=('lost', 'sum'), worst_mae=('mae', 'max'), worst_reference_mae=('reference_mae', 'max')).reset_index().to_csv(OUTPUT / 'window_diagnostics.csv', index=False)
    metadata[~metadata.method.eq('reference')].groupby(['arc', 'target', 'method']).agg(windows=('fallback', 'size'), fallback_windows=('fallback', 'sum'), evaluated_windows=('n_evaluated', lambda s: s.gt(0).sum())).reset_index().to_csv(OUTPUT / 'fallbacks.csv', index=False)
    print(scores(comparable, base).to_string(index=False))


if __name__ == '__main__':
    main()
