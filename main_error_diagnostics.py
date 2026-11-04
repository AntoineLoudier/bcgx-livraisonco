"""Diagnostic descriptif des prévisions historiques déjà produites hors entraînement."""
from pathlib import Path
import numpy as np
import pandas as pd


def analyse(frame):
    rows = []
    for target in ('q', 'k'):
        valid = frame[frame[target].notna() & frame[f'is_imputed_{target}'].eq(0) & frame[f'{target}_model'].notna()].copy()
        valid['target'] = target
        valid['truth'] = valid[target]
        valid['prediction'] = valid[f'{target}_model']
        valid['error'] = valid.prediction - valid.truth
        valid['abs_error'] = valid.error.abs()
        valid['date'] = valid.t_1h.dt.normalize()
        valid['rush_hour'] = valid.t_1h.dt.hour.isin([7, 8, 9, 10, 17, 18, 19, 20])
        rows.append(valid)
    return pd.concat(rows, ignore_index=True)


def summary(frame, keys):
    return frame.groupby(keys, dropna=False).agg(n=('error', 'size'), mae=('abs_error', 'mean'), bias=('error', 'mean'), rmse=('error', lambda x: np.sqrt(np.mean(x*x)))).reset_index()


def days(frame):
    rows = []
    for keys, group in frame.groupby(['window_start', 'arc', 'target', 'date']):
        # La comparaison des sommes et des heures de pointe exige les 24 heures.
        complete = group.t_1h.nunique() == 24
        rows.append(dict(zip(['window_start', 'arc', 'target', 'date'], keys)) | {
            'observed_hours': len(group), 'mae': group.abs_error.mean(), 'bias': group.error.mean(),
            'truth_total': group.truth.sum() if complete else np.nan,
            'prediction_total': group.prediction.sum() if complete else np.nan,
            'peak_shift_hours': (group.loc[group.prediction.idxmax(), 't_1h'].hour - group.loc[group.truth.idxmax(), 't_1h'].hour) if complete else np.nan})
    return pd.DataFrame(rows)


def main():
    source = Path('data/processed/blend_analysis/predictions.csv')
    output = Path('data/processed/error_diagnostics')
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, parse_dates=['t_1h', 'window_start'])
    data = analyse(frame)
    for name, keys in {'overall': ['arc', 'target'], 'hour': ['arc', 'target', 'hour'], 'horizon': ['arc', 'target', 'horizon_day'], 'rush': ['arc', 'target', 'rush_hour']}.items():
        summary(data, keys).to_csv(output / f'{name}.csv', index=False)
    daily = days(data)
    daily.to_csv(output / 'days.csv', index=False)
    daily.sort_values('mae', ascending=False).groupby(['arc', 'target']).head(10).to_csv(output / 'worst_days.csv', index=False)
    print(summary(data, ['arc', 'target']).to_string(index=False))


if __name__ == '__main__':
    main()
