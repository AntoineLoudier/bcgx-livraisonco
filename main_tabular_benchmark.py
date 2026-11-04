"""Nested temporal benchmark of model families, preprocessing and feature blocks."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from main_recent_context import choose_starts, fit as reference_fit, forecast as reference_predict
from src.evaluate import build_window
from src.feature_lab import FeatureStore, columns_for, feature_audit, observed_only, preserve_raw_zeros
from src.model import feature_columns, multi_horizon_training_features
from src.tabular_benchmark import (FAMILIES, SETTINGS, VARIANTS, best_blend, inner_starts,
                                  macro_mae, observed_records, paired_reports, predict)


class Experiment:
    def __init__(self, frame, zeros_frame, delay, threads):
        self.frame, self.delay, self.threads = frame, delay, threads
        self.sources = {'regular': frame, 'missing': observed_only(frame)}
        if zeros_frame is not None:
            self.sources['preserve_zeros'] = zeros_frame
        self.store = FeatureStore(frame)
        self.stores = {'regular': self.store, 'missing': self.store}
        if zeros_frame is not None:
            self.stores['preserve_zeros'] = self.store if zeros_frame is frame else FeatureStore(zeros_frame)
        self.training, self.validation, self.cache = {}, {}, {}

    def source(self, variant):
        if variant == 'preserve_zeros' and self.sources.get(variant) is self.frame:
            return 'regular'
        return variant if variant in ('missing', 'preserve_zeros') else 'regular'

    def examples(self, start, variant):
        source = self.source(variant)
        store = self.stores[source]
        if source not in self.training:
            print(f'Preparing features: delay={self.delay}, source={source}', flush=True)
            raw = multi_horizon_training_features(self.sources[source], self.frame.t_1h.max(),
                                                 real_lags_only=source == 'missing',
                                                 availability_delay_hours=self.delay)
            self.training[source] = store.attach(raw, self.delay)
        key = (start, source)
        if key not in self.validation:
            window = build_window(self.sources[source], start, real_lags_only=source == 'missing',
                                  availability_delay_hours=self.delay)
            # The same evaluation truth for every preprocessing variant.
            truth_cols = ['q', 'k', 'is_imputed_q', 'is_imputed_k']
            window = window.drop(columns=truth_cols).merge(self.frame[['arc', 't_1h']+truth_cols],
                                                           on=['arc', 't_1h'], validate='one_to_one')
            self.validation[key] = store.attach(window, self.delay)
        available = start-pd.Timedelta(hours=1+self.delay)
        train = self.training[source]
        return train[train.t_1h.le(available)], self.validation[key], store

    def prediction(self, start, target, family, setting=0, variant='baseline'):
        key = (start, target, family, setting, variant)
        if key not in self.cache:
            train, valid, store = self.examples(start, variant)
            if family == 'reference':
                cols = feature_columns(target, multi_horizon=True)
                model = reference_fit(train, target, cols)
                pred = reference_predict(model, valid, cols, target)
                stats = {'train_rows': int((train[target].notna() & train[f'is_imputed_{target}'].eq(0)).sum()), 'features': len(cols)}
            else:
                pred, stats = predict(train, valid, target, family, setting, variant, store, self.threads)
            records = observed_records(valid, pred, target)
            self.cache[key] = records, stats
        records, stats = self.cache[key]
        return records.copy(), stats.copy()


def signature(args, starts):
    paths = [args.traffic, args.raw] + sorted(Path('src').glob('*.py')) + [Path(__file__), Path('main_recent_context.py')]
    files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    versions = {p: importlib.metadata.version(p) for p in ('pandas', 'numpy', 'lightgbm', 'catboost', 'xgboost', 'scikit-learn')}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'output'}
    config['starts'] = [str(x) for x in starts]
    manifest = {'files': files, 'config': config, 'settings': SETTINGS, 'versions': versions,
                'python': sys.version, 'test_status': 'exploratory: previously inspected historical windows'}
    manifest['fingerprint'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest


def ensure_manifest(output, manifest):
    path = output / 'manifest.json'
    if path.exists() and json.loads(path.read_text())['fingerprint'] != manifest['fingerprint']:
        raise ValueError('Output contains a different experiment. Choose a new --output directory.')
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')


def run(args):
    frame = pd.read_csv(args.traffic, parse_dates=['t_1h'])
    starts = [pd.Timestamp(x) for x in args.starts] if args.starts else choose_starts(frame, args.windows)
    starts = sorted(starts)
    if any((b-a) < pd.Timedelta(days=5) for a, b in zip(starts, starts[1:])):
        raise ValueError('Outer windows must not overlap')
    manifest = signature(args, starts)
    ensure_manifest(args.output, manifest)
    zeros = None
    if 'preserve_zeros' in args.variants:
        zeros, audit = preserve_raw_zeros(pd.read_csv(args.raw, sep=';'), frame)
        audit.to_csv(args.output / 'raw_zero_audit.csv', index=False)
        if audit.restored_joint_zero.sum() == 0:
            zeros = frame
    availability = []
    for arc, part in frame.groupby('arc'):
        for target in args.targets:
            real = part[target].notna() & part[f'is_imputed_{target}'].eq(0)
            availability.append({'arc': arc, 'target': target, 'observed': int(real.sum()),
                                 'imputed_fraction': 1-real.mean(), 'last_observation': part.loc[real, 't_1h'].max()})
    pd.DataFrame(availability).to_csv(args.output / 'data_audit.csv', index=False)
    all_predictions, all_tuning, all_selections = [], [], []
    for delay in args.delays:
        experiment = Experiment(frame, zeros, delay, args.threads)
        for start in starts:
            stem = f'{delay}h_{start:%Y-%m-%d}'
            checkpoint = args.output / 'checkpoints' / stem
            if checkpoint.with_suffix('.done').exists():
                all_predictions.append(pd.read_csv(checkpoint.with_suffix('.csv'), parse_dates=['window_start', 't_1h']))
                all_tuning.append(pd.read_csv(checkpoint.with_suffix('.tuning.csv')))
                all_selections.append(pd.read_csv(checkpoint.with_suffix('.selection.csv')))
                print(f'Resume completed {stem}', flush=True)
                continue
            print(f'Outer window {start.date()}, delay {delay}h', flush=True)
            inner = inner_starts(start, delay, args.inner_folds)
            rows, tuning, selections = [], [], []
            for target in args.targets:
                reference, _ = experiment.prediction(start, target, 'reference')
                reference['method'] = 'reference'
                rows.append(reference)
                chosen, inner_best, outer_best = {}, {}, {}
                variant_inner, variant_outer = {}, {}
                for family in args.families:
                    candidates = []
                    for setting in range(len(SETTINGS[family])):
                        parts = []
                        for inner_start in inner:
                            p, stats = experiment.prediction(inner_start, target, family, setting)
                            parts.append(p)
                            tuning.append({'target': target, 'family': family, 'variant': 'baseline', 'setting': setting,
                                           'inner_start': inner_start, 'inner_available_at': inner_start-pd.Timedelta(hours=1+delay),
                                           'inner_end': inner_start+pd.Timedelta(hours=119),
                                           'mae_macro': macro_mae(p), **stats})
                        combined = pd.concat(parts, ignore_index=True)
                        candidates.append((macro_mae(combined), setting, combined))
                    score, setting, predictions = min(candidates, key=lambda x: (x[0], x[1]))
                    chosen[family], inner_best[family] = setting, predictions
                    outer, stats = experiment.prediction(start, target, family, setting)
                    outer_best[family] = outer
                    part = outer.copy()
                    part['method'] = f'{family}/baseline'
                    rows.append(part)
                    selections.append({'target': target, 'method': part.method.iloc[0], 'selected': setting,
                                       'inner_mae': score, **stats})
                    variant_inner[(family, 'baseline')] = predictions
                    variant_outer[(family, 'baseline')] = outer
                    print(f'  {target}: {family}, setting {setting}, inner MAE {score:.2f}', flush=True)
                for family in ('lightgbm', 'catboost'):
                    if family not in chosen:
                        continue
                    for variant in args.variants:
                        setting = chosen[family]
                        parts = []
                        for inner_start in inner:
                            p, stats = experiment.prediction(inner_start, target, family, setting, variant)
                            parts.append(p)
                            tuning.append({'target': target, 'family': family, 'variant': variant, 'setting': setting,
                                           'inner_start': inner_start, 'inner_available_at': inner_start-pd.Timedelta(hours=1+delay),
                                           'inner_end': inner_start+pd.Timedelta(hours=119),
                                           'mae_macro': macro_mae(p), **stats})
                        variant_inner[(family, variant)] = pd.concat(parts, ignore_index=True)
                        outer, stats = experiment.prediction(start, target, family, setting, variant)
                        variant_outer[(family, variant)] = outer
                        part = outer.copy()
                        part['method'] = f'{family}/{variant}'
                        rows.append(part)
                        selections.append({'target': target, 'method': part.method.iloc[0], 'selected': setting,
                                           'inner_mae': macro_mae(variant_inner[(family, variant)]), **stats})
                        print(f'  {target}: {family}/{variant} complete', flush=True)
                best = min(variant_inner, key=lambda x: macro_mae(variant_inner[x]))
                selected = variant_outer[best].copy()
                selected['method'] = 'selected_inner'
                rows.append(selected)
                selections.append({'target': target, 'method': 'selected_inner', 'selected': '/'.join(best),
                                   'inner_mae': macro_mae(variant_inner[best])})
                if 'lightgbm' in chosen:
                    family, weight, score = best_blend(inner_best)
                    blend = outer_best['lightgbm'].copy()
                    blend['prediction'] = weight*blend.prediction+(1-weight)*outer_best[family].prediction
                    blend['method'] = 'blend_inner'
                    rows.append(blend)
                    selections.append({'target': target, 'method': 'blend_inner', 'selected': family,
                                       'weight_lightgbm': weight, 'inner_mae': score})
            result, tune, selection = pd.concat(rows, ignore_index=True), pd.DataFrame(tuning), pd.DataFrame(selections)
            for table in (result, tune, selection):
                table['delay_hours'] = delay
                table['outer_start'] = start
                table['outer_available_at'] = start-pd.Timedelta(hours=1+delay)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(checkpoint.with_suffix('.csv'), index=False)
            tune.to_csv(checkpoint.with_suffix('.tuning.csv'), index=False)
            selection.to_csv(checkpoint.with_suffix('.selection.csv'), index=False)
            checkpoint.with_suffix('.done').write_text(manifest['fingerprint'])
            all_predictions.append(result)
            all_tuning.append(tune)
            all_selections.append(selection)
            # An audit describes only history available at this outer issue.
            train, _, store = experiment.examples(start, 'all')
            cols = list(dict.fromkeys(sum([columns_for(t, 'all', store) for t in args.targets], [])))
            audit, redundant = feature_audit(train, cols)
            audit.to_csv(args.output / f'audit_{stem}.csv', index=False)
            redundant.to_csv(args.output / f'redundancy_{stem}.csv', index=False)
            # Useful partial reports survive a deliberate interruption.
            paired_reports(pd.concat(all_predictions, ignore_index=True), args.output)
            pd.concat(all_tuning, ignore_index=True).to_csv(args.output / 'tuning.csv', index=False)
            pd.concat(all_selections, ignore_index=True).to_csv(args.output / 'selection.csv', index=False)
    paired_reports(pd.concat(all_predictions, ignore_index=True), args.output)
    pd.concat(all_tuning, ignore_index=True).to_csv(args.output / 'tuning.csv', index=False)
    pd.concat(all_selections, ignore_index=True).to_csv(args.output / 'selection.csv', index=False)
    print(f'Completed: {args.output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traffic', type=Path, default=Path('data/processed/traffic_clean.csv'))
    parser.add_argument('--raw', type=Path, default=Path('data/raw/traffic_raw.csv'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/tabular_benchmark'))
    parser.add_argument('--windows', type=int, default=4)
    parser.add_argument('--starts', nargs='+', help='Explicit nonoverlapping historical windows')
    parser.add_argument('--delays', type=int, nargs='+', choices=(0, 24, 72, 120), default=[0, 120])
    parser.add_argument('--targets', nargs='+', choices=('q', 'k'), default=['q'])
    parser.add_argument('--families', nargs='+', choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument('--variants', nargs='*', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--inner-folds', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.windows < 1 or args.inner_folds < 2 or args.threads < 1:
        parser.error('Positive window/thread counts and at least two inner folds required')
    run(args)


if __name__ == '__main__':
    main()
