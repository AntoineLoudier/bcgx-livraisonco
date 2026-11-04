"""Comparable estimators, target transforms and nested chronological selection."""
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.feature_lab import columns_for

FAMILIES = ('lightgbm', 'catboost', 'xgboost', 'extratrees')
VARIANTS = ('compact', 'missing', 'quality', 'profiles', 'shape', 'analogues',
            'joint', 'network', 'regime', 'all', 'log', 'relative', 'preserve_zeros')
CATEGORIES = {'arc_id': [0, 1, 2], 'hour': list(range(24)), 'dayofweek': list(range(7)),
              'month': list(range(1, 13)), 'day_regime': list(range(5))}
SETTINGS = {
    'lightgbm': [dict(n_estimators=250, num_leaves=15, min_child_samples=80),
                 dict(n_estimators=400, num_leaves=7, min_child_samples=160)],
    'catboost': [dict(iterations=250, depth=5, l2_leaf_reg=3),
                dict(iterations=400, depth=4, l2_leaf_reg=10)],
    'xgboost': [dict(n_estimators=250, max_depth=4, min_child_weight=80),
               dict(n_estimators=400, max_depth=3, min_child_weight=160)],
    'extratrees': [dict(n_estimators=120, max_depth=16, min_samples_leaf=20),
                   dict(n_estimators=180, max_depth=24, min_samples_leaf=40)],
}


def matrix(frame, columns, family):
    x = frame[columns].replace([np.inf, -np.inf], np.nan).copy()
    for col, categories in CATEGORIES.items():
        if col not in x:
            continue
        if family == 'catboost':
            x[col] = x[col].fillna(-1).astype(int).astype(str)
        elif family in ('lightgbm', 'xgboost'):
            x[col] = pd.Categorical(x[col], categories=categories)
    return x


def estimator(family, setting, columns, threads=2):
    params = SETTINGS[family][setting]
    if family == 'lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params, objective='regression_l1', learning_rate=.04,
                             colsample_bytree=.9, random_state=42, n_jobs=threads, verbosity=-1)
    if family == 'catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params, loss_function='MAE', learning_rate=.04,
                                 cat_features=[c for c in columns if c in CATEGORIES],
                                 random_seed=42, thread_count=threads, verbose=False,
                                 allow_writing_files=False)
    if family == 'xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(**params, objective='reg:absoluteerror', learning_rate=.04,
                            tree_method='hist', enable_categorical=True, colsample_bytree=.9,
                            random_state=42, n_jobs=threads)
    if family == 'extratrees':
        cats = [c for c in columns if c in CATEGORIES]
        nums = [c for c in columns if c not in cats]
        # Train-fitted imputer, with missing flags. No validation statistics.
        transformer = ColumnTransformer([
            ('cats', OneHotEncoder(categories=[CATEGORIES[c] for c in cats], handle_unknown='ignore', sparse_output=False), cats),
            ('nums', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), nums)])
        # Squared-error splits keep the forest tractable; selection is MAE.
        return make_pipeline(transformer, ExtraTreesRegressor(**params, criterion='squared_error',
                                                              max_features=.8, random_state=42, n_jobs=threads))
    raise ValueError(f'Unknown model: {family}')


def target_scale(frame, target):
    return frame[f'{target}_profile_84_median'].fillna(frame[f'{target}_regime_hist_mean']).fillna(
        10 if target == 'q' else 1).clip(lower=10 if target == 'q' else 1).to_numpy()


def predict(train, valid, target, family, setting, variant, store, threads=2):
    columns = columns_for(target, variant, store)
    known = train[target].notna() & train[f'is_imputed_{target}'].eq(0)
    train = train.loc[known]
    if len(train) < 100:
        raise ValueError(f'Insufficient observed training labels: {target}')
    y = train[target].to_numpy()
    if variant == 'log':
        y = np.log1p(y)
    elif variant == 'relative':
        y = y / target_scale(train, target)
    model = estimator(family, setting, columns, threads)
    started = time.monotonic()
    model.fit(matrix(train, columns, family), y)
    result = model.predict(matrix(valid, columns, family))
    if variant == 'log':
        result = np.expm1(np.minimum(result, 30))
    elif variant == 'relative':
        result = result * target_scale(valid, target)
    result = np.clip(result, 0, 100 if target == 'k' else None)
    if not np.isfinite(result).all():
        raise ValueError(f'Nonfinite predictions from {family}/{variant}')
    return result, {'train_rows': len(train), 'features': len(columns), 'seconds': time.monotonic()-started}


def inner_starts(start, delay, folds=2):
    """All tuning labels available before outer training, disjoint 5-day blocks."""
    available = pd.Timestamp(start) - pd.Timedelta(hours=1+delay)
    last = available.normalize() - pd.Timedelta(days=4 if available.hour == 23 else 5)
    return [last - pd.Timedelta(days=8*i) for i in reversed(range(folds))]


def observed_records(window, predictions, target):
    valid = window[target].notna() & window[f'is_imputed_{target}'].eq(0)
    part = window.loc[valid, ['arc', 't_1h', 'window_start', 'horizon_hours', 'is_public_holiday']].copy()
    part['truth'] = window.loc[valid, target].to_numpy()
    part['prediction'] = predictions[valid.to_numpy()]
    part['target'] = target
    return part


def macro_mae(records):
    """Equal weight to each evaluable street; no imputed labels."""
    error = (records.prediction-records.truth).abs()
    return float(error.groupby(records.arc).mean().mean())


def best_blend(inner_predictions):
    """Choose one pair/weight on *inner* out-of-training predictions only."""
    keys = ['arc', 't_1h', 'window_start', 'target']
    reference = inner_predictions['lightgbm'].set_index(keys)
    best = ('lightgbm', 1., macro_mae(reference.reset_index()))
    for family in FAMILIES[1:]:
        if family not in inner_predictions:
            continue
        other = inner_predictions[family].set_index(keys)
        if not reference.index.equals(other.index):
            raise ValueError('Unpaired inner forecasts')
        for weight in (0., .25, .5, .75):
            part = reference.copy()
            part['prediction'] = weight * reference.prediction + (1-weight) * other.prediction
            score = macro_mae(part.reset_index())
            if score < best[2]:
                best = family, weight, score
    return best


def paired_reports(predictions, output):
    keys = ['delay_hours', 'target', 'window_start', 'arc', 't_1h']
    # An absent variant must fail, not silently shrink a particular score.
    expected = predictions.method.nunique()
    if predictions.duplicated(keys + ['method']).any():
        raise ValueError('Duplicate prediction')
    if not predictions.groupby(keys).method.nunique().eq(expected).all():
        raise ValueError('Methods do not cover identical observed targets')
    data = predictions.copy()
    data['error'] = data.prediction-data.truth
    data['abs_error'] = data.error.abs()
    data['horizon_day'] = (data.horizon_hours-1)//24+1
    data['start_weekday'] = data.window_start.dt.day_of_week
    baseline = data[data.method.eq('reference')][keys+['abs_error']].rename(columns={'abs_error': 'reference_error'})
    data = data.merge(baseline, on=keys, validate='many_to_one')
    data['delta'] = data.abs_error-data.reference_error
    for name, group in {
        'summary': ['delay_hours', 'target', 'arc', 'method'],
        'windows': ['delay_hours', 'target', 'arc', 'method', 'window_start'],
        'horizons': ['delay_hours', 'target', 'arc', 'method', 'horizon_day'],
        'holidays': ['delay_hours', 'target', 'arc', 'method', 'is_public_holiday'],
        'weekdays': ['delay_hours', 'target', 'arc', 'method', 'start_weekday'],
    }.items():
        scores = data.groupby(group).agg(n=('truth', 'size'), mae=('abs_error', 'mean'),
                                        bias=('error', 'mean'), delta_mae=('delta', 'mean')).reset_index()
        scores.to_csv(output / f'{name}.csv', index=False)
    windows = data.groupby(['delay_hours', 'target', 'arc', 'method', 'window_start']).agg(
        mae=('abs_error', 'mean'), delta=('delta', 'mean')).reset_index()
    windows.groupby(['delay_hours', 'target', 'arc', 'method']).agg(
        windows=('mae', 'size'), worst_mae=('mae', 'max'), wins=('delta', lambda x: int((x<0).sum()))
        ).reset_index().to_csv(output / 'worst_windows.csv', index=False)
    correlations = []
    for (delay, target, arc), part in data.groupby(['delay_hours', 'target', 'arc']):
        pivot = part.pivot(index=['window_start', 't_1h'], columns='method', values='error')
        for method in pivot:
            correlations.append({'delay_hours': delay, 'target': target, 'arc': arc, 'method': method,
                                 'error_correlation_reference': pivot[method].corr(pivot.reference)})
    pd.DataFrame(correlations).to_csv(output / 'error_correlations.csv', index=False)
    # Block bootstrap is descriptive on already explored windows.
    rng, rows = np.random.default_rng(42), []
    for key, part in data.groupby(['delay_hours', 'target', 'arc', 'method']):
        blocks = part.groupby('window_start').delta.agg(['sum', 'count']).to_numpy()
        if len(blocks) < 2:
            continue
        draws = rng.integers(len(blocks), size=(2000, len(blocks)))
        samples = blocks[draws, 0].sum(axis=1)/blocks[draws, 1].sum(axis=1)
        lo, hi = np.quantile(samples, [.025, .975])
        rows.append(dict(zip(['delay_hours', 'target', 'arc', 'method'], key)) |
                    {'windows': len(blocks), 'delta_lo': lo, 'delta_hi': hi})
    pd.DataFrame(rows).to_csv(output / 'bootstrap.csv', index=False)
    data.to_csv(output / 'predictions.csv', index=False)
