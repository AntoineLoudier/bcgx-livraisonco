"""Causal preprocessing and feature ablations for the tabular benchmark.

Every lookup is bounded by the *example's* availability time. No statistic is
fitted on the evaluation period, and imputed values never enter new summaries.
"""
import warnings

import numpy as np
import pandas as pd

from src.model import feature_columns
from src.preprocessing import impute_missing_values_no_leakage


def observed_only(frame):
    result = frame.copy()
    for target in ('q', 'k'):
        result[target] = result[target].where(result[f'is_imputed_{target}'].eq(0))
    return result


def preserve_raw_zeros(raw, truth):
    """Alternative inputs/labels, same sensor identities as the existing grid.

    Only the old joint-zero heuristic is reversed. Impossible measurements stay
    missing. Evaluation truth is supplied independently by the runner.
    """
    data = raw.copy()
    data['t_1h'] = pd.to_datetime(data.t_1h, utc=True).dt.tz_convert('Europe/Paris').dt.tz_localize(None)
    for target, upper in [('q', 8000), ('k', 100)]:
        data[target] = pd.to_numeric(data[target], errors='coerce')
        data[target] = data[target].where(data[target].between(0, upper))
    # Restore only raw joint zeros at already present arc/time keys. Do not
    # silently merge different variants/tronçons or revise other cleaned data.
    zeros = data.assign(zero=data.q.eq(0) & data.k.eq(0)).groupby(['arc', 't_1h']).zero.all()
    result = observed_only(truth)
    restore = pd.MultiIndex.from_frame(result[['arc', 't_1h']]).map(zeros).fillna(False).to_numpy(dtype=bool)
    for target in ('q', 'k'):
        result.loc[restore & result[f'is_imputed_{target}'].eq(1), target] = 0.
    return impute_missing_values_no_leakage(result), pd.DataFrame({
        'arc': result.arc, 'restored_joint_zero': restore & truth.is_imputed_q.eq(1)
    }).groupby('arc').restored_joint_zero.sum().reset_index()


def _asof(examples, table, keys, columns):
    left = examples[keys + ['available_at']].copy()
    left['_row'] = np.arange(len(left))
    right = table[keys + ['available_at'] + columns]
    result = pd.merge_asof(left.sort_values('available_at'), right.sort_values('available_at'),
                           on='available_at', by=keys, direction='backward')
    return result.sort_values('_row')[columns].reset_index(drop=True)


class FeatureStore:
    """Feature tables calculated forward in time once, then queried as-of."""

    def __init__(self, frame):
        if frame.duplicated(['arc', 't_1h']).any():
            raise ValueError('Duplicate arc/hour')
        self.frame = observed_only(frame).sort_values(['arc', 't_1h'])
        self.context, self.profiles = self._tables()
        self.blocks = {
            'quality': [c for c in self.context if '_quality_' in c],
            'profiles': [c for c in self.profiles if '_profile_' in c] +
                        [c for c in self.context if '_anomaly_' in c],
            'shape': [c for c in self.context if '_shape_' in c],
            'joint': [c for c in self.context if c.startswith('joint_')],
            'network': [c for c in self.context if c.startswith('network_')],
            'regime': [f'{t}_regime_{s}' for t in ('q', 'k') for s in ('median', 'count', 'age')],
            'analogues': [f'{t}_analog_{s}' for t in ('q', 'k') for s in ('median', 'std', 'count')] +
                         ['analog_distance', 'analog_candidates'],
        }
        self._analogue_cache = {}

    def _tables(self):
        contexts, profiles = [], []
        for arc, group in self.frame.groupby('arc', sort=False):
            group = group.set_index('t_1h').asfreq('h')
            out = pd.DataFrame(index=group.index)
            prof = pd.DataFrame(index=group.index)
            anomalies = {}
            for target, floor in [('q', 10.), ('k', 1.)]:
                real = group[target]
                keys = [real.index.day_of_week, real.index.hour]
                usual = real.groupby(keys).transform(lambda x: x.shift().expanding().median())
                anomaly = (real - usual) / usual.abs().clip(lower=floor)
                anomalies[target] = anomaly.clip(-5, 5)
                last = pd.Series(real.index.where(real.notna()), index=real.index).ffill()
                out[f'{target}_quality_age'] = (real.index - last) / pd.Timedelta(hours=1)
                same = real.notna() & real.eq(real.shift())
                runs = same.groupby((~same).cumsum()).cumsum()
                zero = real.eq(0)
                zero_runs = zero.groupby((~zero).cumsum()).cumsum()
                out[f'{target}_quality_constant_run'] = runs
                out[f'{target}_quality_zero_run'] = zero_runs
                for n in (24, 168):
                    out[f'{target}_quality_coverage_{n}'] = real.rolling(n, min_periods=1).count() / n
                    out[f'{target}_quality_zero_fraction_{n}'] = zero.where(real.notna()).rolling(n, min_periods=1).mean()
                    out[f'{target}_anomaly_{n}'] = anomalies[target].rolling(n, min_periods=6).median()
                for days in (28, 84, 180):
                    for stat in ('median', 'std', 'count'):
                        prof[f'{target}_profile_{days}_{stat}'] = real.groupby(keys).transform(
                            lambda x, d=days, s=stat: getattr(x.rolling(f'{d}D', min_periods=1), s)())
                days_index = pd.date_range(real.index.min().normalize(), real.index.max().normalize(), freq='D')
                full_hours = pd.date_range(days_index.min(), days_index.max()+pd.Timedelta(hours=23), freq='h')
                matrix = real.reindex(full_hours).to_numpy().reshape(-1, 24)
                complete = np.isfinite(matrix).all(axis=1)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    total = np.nansum(matrix, axis=1)
                    shape = pd.DataFrame({
                        f'{target}_shape_am_share': np.nansum(matrix[:, 7:11], axis=1) / np.maximum(total, floor),
                        f'{target}_shape_pm_share': np.nansum(matrix[:, 17:21], axis=1) / np.maximum(total, floor),
                        f'{target}_shape_amplitude': (np.nanmax(matrix, axis=1) - np.nanmin(matrix, axis=1)) / np.maximum(total / 24, floor),
                        f'{target}_shape_peak': np.nan_to_num(matrix, nan=-np.inf).argmax(axis=1),
                        f'{target}_shape_mean': total / 24,
                    }, index=days_index + pd.Timedelta(hours=23)).astype(float)
                shape.loc[~complete, :] = np.nan
                shape[f'{target}_shape_complete'] = complete.astype(float)
                # As-of reindex only: an unfinished day is never summarized.
                shape = shape.reindex(group.index, method='ffill')
                out = pd.concat([out, shape], axis=1)
                # A sustained relative level change is a flag, not a diagnosis.
                short = anomalies[target].rolling(168, min_periods=120).mean()
                long = anomalies[target].shift(168).rolling(504, min_periods=360).mean()
                signal = (short - long).abs().gt(.25)
                sustained = signal.rolling(72, min_periods=72).sum().eq(72)
                trigger = sustained & ~sustained.shift(fill_value=False)
                reset = pd.Series(group.index.where(trigger), index=group.index).ffill()
                out[f'{target}_reset_at'] = reset
            out['joint_anomaly_gap'] = (anomalies['k'] - anomalies['q']).rolling(24, min_periods=6).mean()
            out['joint_high_occupancy_low_flow'] = (anomalies['k'].gt(.25) & anomalies['q'].lt(-.25)).where(
                group.q.notna() & group.k.notna()).rolling(168, min_periods=24).mean()
            out['joint_correlation_168'] = group.q.rolling(168, min_periods=24).corr(group.k)
            for table in (out, prof):
                table['arc'] = arc
                table['available_at'] = table.index
            prof['dayofweek'] = prof.index.day_of_week
            prof['hour'] = prof.index.hour
            contexts.append(out.reset_index(drop=True))
            profiles.append(prof.reset_index(drop=True))
        context = pd.concat(contexts, ignore_index=True)
        # Other target streets only, observed up to the same availability time.
        for target in ('q', 'k'):
            key = f'{target}_anomaly_24'
            series = context.pivot(index='available_at', columns='arc', values=key)
            for arc in series:
                other = series.drop(columns=arc)
                mask = context.arc.eq(arc)
                out_index = context.loc[mask, 'available_at']
                context.loc[mask, f'network_{target}_anomaly'] = other.median(axis=1).reindex(out_index).to_numpy()
                context.loc[mask, f'network_{target}_count'] = other.count(axis=1).reindex(out_index).to_numpy()
        return context, pd.concat(profiles, ignore_index=True)

    def attach(self, examples, delay):
        if delay < 0:
            raise ValueError('Negative availability delay')
        result = examples.reset_index(drop=True).copy()
        result['available_at'] = result.t_1h - pd.to_timedelta(result.horizon_hours + delay, unit='h')
        context_cols = [c for c in self.context if c not in ('arc', 'available_at')]
        profile_cols = [c for c in self.profiles if '_profile_' in c]
        result = pd.concat([result, _asof(result, self.context, ['arc'], context_cols),
                            _asof(result, self.profiles, ['arc', 'dayofweek', 'hour'], profile_cols)], axis=1)
        # Counts/ages distinguish no observation from a measured zero.
        for t in ('q', 'k'):
            result[f'{t}_profile_age'] = np.nan
        extra = []
        for (arc, available), idx in result.groupby(['arc', 'available_at']).groups.items():
            rows = result.loc[idx]
            history = self.frame[self.frame.arc.eq(arc) & self.frame.t_1h.le(available)]
            part = pd.DataFrame(index=idx)
            for target in ('q', 'k'):
                observed = history.dropna(subset=[target])
                last = observed.groupby([observed.t_1h.dt.day_of_week, observed.t_1h.dt.hour]).t_1h.max()
                query = pd.MultiIndex.from_arrays([rows.dayofweek, rows.hour])
                latest = last.reindex(query).to_numpy()
                part[f'{target}_profile_age'] = (available - pd.to_datetime(latest)) / pd.Timedelta(hours=1)
                reset = rows[f'{target}_reset_at'].iloc[0]
                segment = observed if pd.isna(reset) else observed[observed.t_1h >= reset]
                grouped = segment.groupby([segment.t_1h.dt.day_of_week, segment.t_1h.dt.hour])[target]
                for name in ('median', 'count'):
                    part[f'{target}_regime_{name}'] = getattr(grouped, name)().reindex(query).to_numpy()
                part[f'{target}_regime_age'] = np.nan if pd.isna(reset) else (available - reset) / pd.Timedelta(hours=1)
            analog, metadata = self.analogues(arc, available, delay)
            offset = ((rows.t_1h - (available + pd.Timedelta(hours=delay))) / pd.Timedelta(hours=1)).astype(int) - 1
            for col in analog:
                part[col] = analog[col].to_numpy()[offset]
            for col, value in metadata.items():
                part[col] = value
            extra.append(part)
        extras = pd.concat(extra).sort_index()
        result = result.drop(columns=[f'{t}_profile_age' for t in ('q', 'k')])
        result = pd.concat([result, extras], axis=1)
        for target in ('q', 'k'):
            col = f'{target}_profile_age'
            if col not in self.blocks['profiles']:
                self.blocks['profiles'].append(col)
        return result.drop(columns=[f'{t}_reset_at' for t in ('q', 'k')])

    def analogues(self, arc, available, delay):
        """Five neighbours; all 120 successor timestamps must already be past.

        Context distances use dimensionless anomalies and daily shape, with
        fixed scales. Candidates share the issue weekday. No future truth is
        used to select neighbours, normalize features or choose their number.
        """
        cache_key = (arc, available, delay)
        if cache_key in self._analogue_cache:
            return self._analogue_cache[cache_key]
        columns = [f'{t}_analog_{s}' for t in ('q', 'k') for s in ('median', 'std', 'count')]
        empty = pd.DataFrame(np.nan, index=range(120), columns=columns)
        empty[[c for c in columns if c.endswith('count')]] = 0.
        meta = {'analog_distance': np.nan, 'analog_candidates': 0.}
        context = self.context[self.context.arc.eq(arc)].set_index('available_at')
        if available not in context.index:
            return empty, meta
        descriptor = [f'{t}_anomaly_{n}' for t in ('q', 'k') for n in (24, 168)] + [
            f'{t}_shape_{s}' for t in ('q', 'k') for s in ('am_share', 'pm_share')]
        current = context.loc[available, descriptor].to_numpy(dtype=float)
        candidates = context.loc[(context.index.hour == available.hour) &
                                 (context.index.day_of_week == available.day_of_week) &
                                 (context.index + pd.Timedelta(hours=delay+120) <= available)]
        if np.isfinite(current).sum() < 4 or candidates.empty:
            return empty, meta
        x = candidates[descriptor].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(current)
        distance = np.sqrt(np.nansum(np.where(valid, (x-current)**2, np.nan), axis=1) / np.maximum(valid.sum(axis=1), 1))
        distance[valid.sum(axis=1) < 4] = np.inf
        observed = self.frame[self.frame.arc.eq(arc)].set_index('t_1h')[['q', 'k']]
        trajectories, distances = [], []
        for position in np.argsort(distance, kind='stable'):
            if not np.isfinite(distance[position]):
                break
            candidate = candidates.index[position]
            dates = pd.date_range(candidate + pd.Timedelta(hours=delay+1), periods=120, freq='h')
            trajectory = observed.reindex(dates).to_numpy()
            if np.isfinite(trajectory[:, 0]).sum() < 96:
                continue
            trajectories.append(trajectory)
            distances.append(distance[position])
            if len(trajectories) == 5:
                break
        if trajectories:
            stack = np.stack(trajectories)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                for i, target in enumerate(('q', 'k')):
                    empty[f'{target}_analog_median'] = np.nanmedian(stack[:, :, i], axis=0)
                    empty[f'{target}_analog_std'] = np.nanstd(stack[:, :, i], axis=0)
                    empty[f'{target}_analog_count'] = np.isfinite(stack[:, :, i]).sum(axis=0)
            meta = {'analog_distance': float(np.mean(distances)), 'analog_candidates': float(len(trajectories))}
        self._analogue_cache[cache_key] = empty, meta
        return empty, meta


def columns_for(target, variant, store):
    if variant == 'baseline':
        return feature_columns(target, multi_horizon=True)
    calendar = ['arc_id', 'hour', 'dayofweek', 'month', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'is_weekend', 'is_public_holiday', 'is_school_holiday', 'horizon_hours']
    cols = calendar + [f'{target}_{s}' for s in ('lag_168h', 'lag_336h', 'lag_504h', 'weekly_mean_3w', 'regime_hist_mean')]
    if variant == 'missing':
        cols += [f'{target}_lag_{d}h_real' for d in (168, 336, 504)]
    blocks = list(store.blocks) if variant == 'all' else [variant] if variant in store.blocks else []
    if variant == 'relative':
        blocks = ['profiles']
    if variant == 'preserve_zeros':
        blocks = ['quality']
    for block in blocks:
        cols += store.blocks[block]
    if variant == 'regime':
        cols.remove(f'{target}_regime_hist_mean')
    return list(dict.fromkeys(cols))


def feature_audit(examples, columns):
    """Diagnostics on training only; never automatic feature selection."""
    numeric = examples[columns].astype(float)
    audit = pd.DataFrame({'feature': columns, 'missing_fraction': numeric.isna().mean().to_numpy(),
                          'unique_values': numeric.nunique().to_numpy()})
    corr = numeric.corr().abs()
    redundant = []
    for i, col in enumerate(columns):
        for other in columns[i+1:]:
            if corr.loc[col, other] >= .98:
                redundant.append({'feature': col, 'other': other, 'abs_correlation': corr.loc[col, other]})
    return audit, pd.DataFrame(redundant, columns=['feature', 'other', 'abs_correlation'])
