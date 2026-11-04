"""Comparaisons chronologiques : capteurs, résidus, variables croisées et modèles."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluate import build_window, window_starts, WINDOW_HOURS
from src.model import feature_columns, multi_horizon_training_features
from src.preprocessing import impute_missing_values_no_leakage

TARGETS = ('q', 'k')
OUT = Path('data/processed/new_experiments')


def columns(target, variant):
    cols = feature_columns(target, multi_horizon=True)
    if variant == 'real_lags':
        cols += [f'{target}_lag_{d}h_real' for d in (168, 336, 504)]
    if variant == 'cross':
        other = 'k' if target == 'q' else 'q'
        cols += [f'{other}_lag_{d}h' for d in (168, 336, 504)] + [f'{other}_weekly_mean_3w', f'{other}_regime_hist_mean']
    return cols


def model_predict(train, window, target, variant):
    valid = train[target].notna() & train[f'is_imputed_{target}'].eq(0)
    train = train.loc[valid]
    if len(train) < 100:
        return np.full(len(window), np.nan)
    cols = columns(target, variant)
    y = train[target].to_numpy()
    if variant == 'residual':
        # Profil cumulatif par rue, régime et heure, calculé avant chaque émission.
        y = y - train[f'{target}_regime_hist_mean'].fillna(0).to_numpy()
    if variant == 'additive':
        # Effets additifs d'heure, jour et rue plus quelques termes continus.
        cats = ['arc_id', 'hour', 'dayofweek', 'month', 'is_public_holiday', 'is_school_holiday']
        nums = ['horizon_hours', f'{target}_weekly_mean_3w', f'{target}_regime_hist_mean']
        cols = cats + nums
        transform = ColumnTransformer([('category', OneHotEncoder(handle_unknown='ignore'), cats),
                                       ('numeric', make_pipeline(SimpleImputer(strategy='constant', fill_value=0), StandardScaler()), nums)])
        model = make_pipeline(transform, Ridge(alpha=100.0))
    else:
        settings = dict(objective='huber' if variant == 'huber' else 'regression_l1', n_estimators=250,
                        learning_rate=0.04, num_leaves=7 if variant == 'regularized' else 15,
                        min_child_samples=160 if variant == 'regularized' else 80,
                        subsample=1.0, colsample_bytree=0.9, random_state=42, n_jobs=2, verbosity=-1)
        model = lgb.LGBMRegressor(**settings)
    model.fit(train[cols], y)
    pred = model.predict(window[cols])
    if variant == 'residual':
        pred += window[f'{target}_regime_hist_mean'].fillna(0).to_numpy()
    return np.clip(pred, 0, 100 if target == 'k' else None)


def score(predictions, keys):
    p = predictions.dropna(subset=['truth', 'prediction']).copy()
    p['error'] = p.prediction - p.truth
    p['abs_error'] = p.error.abs()
    return p.groupby(keys, dropna=False).agg(n=('truth', 'size'), mae=('abs_error', 'mean'), bias=('error', 'mean')).reset_index()


def records(window, train, variants):
    rows = []
    for target in TARGETS:
        observed = window[target].notna() & window[f'is_imputed_{target}'].eq(0)
        for variant in variants:
            pred = model_predict(train, window, target, variant)
            part = window.loc[observed, ['arc', 't_1h', 'window_start', 'horizon_hours', 'is_public_holiday']].copy()
            part['truth'] = window.loc[observed, target].to_numpy()
            part['prediction'] = pred[observed.to_numpy()]
            part['target'] = target
            part['variant'] = variant
            rows.append(part)
    return pd.concat(rows, ignore_index=True)


def simulate_outage(frame, arc, cutoff, days):
    """Masque les observations du capteur puis reconstruit toutes ses imputations."""
    result = frame.copy().sort_values(['arc', 't_1h']).reset_index(drop=True)
    for target in TARGETS:
        # Effacer aussi les anciennes valeurs imputées : elles peuvent dépendre de la panne.
        result.loc[result[f'is_imputed_{target}'].eq(1), target] = np.nan
        mask = result.arc.eq(arc) & result.t_1h.between(cutoff - pd.Timedelta(days=days) + pd.Timedelta(hours=1), cutoff)
        result.loc[mask, target] = np.nan
    return impute_missing_values_no_leakage(result)


def run_outages(frame, starts):
    result = []
    for start in starts:
        cutoff = start - pd.Timedelta(hours=1)
        for arc in ('Convention', 'Saint-Pères'):
            for days in (1, 3, 7, 21):
                print(f'Panne simulée {start.date()} {arc} {days} jours', flush=True)
                damaged = simulate_outage(frame, arc, cutoff, days)
                for variant in ('reference', 'real_lags'):
                    real = variant == 'real_lags'
                    train = multi_horizon_training_features(damaged, cutoff, real_lags_only=real)
                    window = build_window(damaged, start, real_lags_only=real)
                    # Vérité issue du fichier intact ; seules les prédictions de l'arc masqué comptent.
                    truth = frame[['arc', 't_1h', 'q', 'k', 'is_imputed_q', 'is_imputed_k']]
                    window = window.drop(columns=['q', 'k', 'is_imputed_q', 'is_imputed_k']).merge(truth, on=['arc', 't_1h'], how='left', validate='one_to_one')
                    part = records(window, train, [variant])
                    part = part[part.arc.eq(arc)].copy()
                    part['outage_days'] = days
                    result.append(part)
    return pd.concat(result, ignore_index=True)


def calibrate_intervals(predictions, min_count=100):
    """Quantiles des erreurs signées des fenêtres entièrement terminées uniquement."""
    baseline = predictions[predictions.variant.eq('reference')].copy()
    baseline['horizon_day'] = (baseline.horizon_hours - 1) // 24 + 1
    baseline['residual'] = baseline.truth - baseline.prediction
    rows = []
    for start in sorted(baseline.window_start.unique()):
        start = pd.Timestamp(start)
        cutoff = start - pd.Timedelta(hours=1)
        history = baseline[(baseline.window_start + pd.Timedelta(hours=WINDOW_HOURS - 1) <= cutoff) & (baseline.t_1h <= cutoff)]
        current = baseline[baseline.window_start.eq(start)]
        for (arc, target, horizon), group in current.groupby(['arc', 'target', 'horizon_day']):
            past = history[history.arc.eq(arc) & history.target.eq(target) & history.horizon_day.eq(horizon)].residual.dropna()
            if len(past) < min_count:
                continue
            lo, hi = past.quantile([.1, .9])
            part = group.copy()
            part['lower'] = (part.prediction + lo).clip(lower=0, upper=100 if target == 'k' else None)
            part['upper'] = (part.prediction + hi).clip(lower=0, upper=100 if target == 'k' else None)
            part['covered'] = part.truth.between(part.lower, part.upper)
            part['width'] = part.upper - part.lower
            part['calibration_n'] = len(past)
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=int, default=8)
    parser.add_argument('--outage-windows', type=int, default=2)
    parser.add_argument('--outage-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv('data/processed/traffic_clean.csv', parse_dates=['t_1h'])
    starts = window_starts(frame, max(args.windows, args.outage_windows), 7)
    if not args.outage_only:
        results = []
        variants = ['reference', 'residual', 'cross', 'additive', 'regularized', 'huber']
        for start in starts[-args.windows:]:
            print(f'Comparaison des modèles {start.date()}', flush=True)
            train = multi_horizon_training_features(frame, start - pd.Timedelta(hours=1))
            window = build_window(frame, start)
            results.append(records(window, train, variants))
        predictions = pd.concat(results, ignore_index=True)
        predictions.to_csv(OUT / 'model_predictions.csv', index=False)
        for name, keys in {'models': ['arc', 'target', 'variant'], 'model_windows': ['window_start', 'arc', 'target', 'variant'], 'model_holidays': ['arc', 'target', 'variant', 'is_public_holiday']}.items():
            score(predictions, keys).to_csv(OUT / f'{name}.csv', index=False)
        # Réutiliser toutes les prévisions de la même référence sur 52 fenêtres.
        past = pd.read_csv('data/processed/blend_analysis/predictions.csv', parse_dates=['t_1h', 'window_start'])
        from main_error_diagnostics import analyse
        calibrated = analyse(past).rename(columns={'prediction': 'prediction', 'truth': 'truth'})
        calibrated['variant'] = 'reference'
        intervals = calibrate_intervals(calibrated)
        intervals.to_csv(OUT / 'intervals.csv', index=False)
        if not intervals.empty:
            intervals.groupby(['arc', 'target', 'horizon_day']).agg(n=('truth', 'size'), coverage=('covered', 'mean'), width=('width', 'mean'), min_calibration=('calibration_n', 'min')).reset_index().to_csv(OUT / 'interval_scores.csv', index=False)
    if args.outage_windows:
        outage = run_outages(frame, starts[-args.outage_windows:])
        outage.to_csv(OUT / 'outage_predictions.csv', index=False)
        score(outage, ['arc', 'target', 'outage_days', 'variant']).to_csv(OUT / 'outages.csv', index=False)
        # Ajouter la référence intacte sur les mêmes fenêtres, si elle existe.
        reference_path = OUT / 'model_predictions.csv'
        if reference_path.exists():
            reference = pd.read_csv(reference_path, parse_dates=['t_1h', 'window_start'])
            reference = reference[reference.variant.eq('reference')]
            keys = ['arc', 'target', 't_1h', 'window_start']
            wanted = outage[keys].drop_duplicates()
            reference = wanted.merge(reference, on=keys, how='inner', validate='one_to_one')
            reference['outage_days'] = 0
            combined = pd.concat([outage, reference], ignore_index=True)
            score(combined, ['arc', 'target', 'outage_days', 'variant']).to_csv(OUT / 'outages_with_intact.csv', index=False)


if __name__ == '__main__':
    main()
