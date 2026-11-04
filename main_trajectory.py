"""Prévisions conjointes 120 heures : modèle linéaire et réseau convolutif."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from main_recent_context import choose_starts
from src.evaluate import build_window
from src.features import ARCS
from src.model import fit_model, multi_horizon_training_features, predict_model
from src.trajectory import CALENDAR, HORIZON, LOOKBACK, example, examples_until, make_series

METHODS = ('reference', 'linear', 'conv')
torch.set_num_threads(2)


class LinearTrajectory(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4 * LOOKBACK + HORIZON * len(CALENDAR), HORIZON * 2)

    def forward(self, past, calendar):
        x = torch.cat([past.flatten(1), calendar.flatten(1)], dim=1)
        return self.layer(x).reshape(-1, HORIZON, 2)


class ConvTrajectory(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(4, 16, kernel_size=7, stride=2, padding=3), nn.ReLU(),
                                     nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3), nn.ReLU(),
                                     nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2), nn.ReLU(),
                                     nn.AdaptiveAvgPool1d(24), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(32 * 24 + HORIZON * len(CALENDAR), 128), nn.ReLU(),
                                  nn.Linear(128, HORIZON * 2))

    def forward(self, past, calendar):
        x = torch.cat([self.encoder(past), calendar.flatten(1)], dim=1)
        return self.head(x).reshape(-1, HORIZON, 2)


def masked_mae(prediction, truth, mask):
    return (torch.abs(prediction - truth) * mask).sum() / mask.sum().clamp(min=1)


def tensor_dataset(samples):
    return TensorDataset(*(torch.from_numpy(np.stack([x[key] for x in samples]).astype(np.float32))
                           for key in ('past', 'calendar', 'target', 'mask')))


def sample_loader(samples, dataset, batch_size, seed):
    counts = Counter(s['arc'] for s in samples)
    weights = torch.tensor([1 / counts[s['arc']] for s in samples], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, len(samples), replacement=True, generator=generator)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def train_epochs(model, samples, epochs, seed=42, valid=None, patience=3):
    dataset = tensor_dataset(samples)
    loader = sample_loader(samples, dataset, 64, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.001)
    val_loader = DataLoader(tensor_dataset(valid), batch_size=128) if valid else None
    best, best_epoch, stale = float('inf'), 1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for past, calendar, truth, mask in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mae(model(past, calendar), truth, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        if val_loader is not None:
            model.eval()
            absolute, count = 0., 0.
            with torch.no_grad():
                for past, calendar, truth, mask in val_loader:
                    prediction = model(past, calendar)
                    absolute += (torch.abs(prediction - truth) * mask).sum().item()
                    count += mask.sum().item()
            metric = absolute / max(count, 1)
            if metric < best - 1e-5:
                best, best_epoch, stale = metric, epoch, 0
            else:
                stale += 1
                if stale >= patience:
                    break
    return best_epoch if val_loader else epochs


def split_inner(samples, cutoff, validation_days=21):
    """Sépare les labels par une frontière stricte, sans fenêtres chevauchantes."""
    boundary = pd.Timestamp(cutoff).normalize() - pd.Timedelta(days=validation_days)
    train = [s for s in samples if s['start'] + pd.Timedelta(hours=HORIZON - 1) < boundary]
    valid = [s for s in samples if s['start'] >= boundary]
    if not train or not valid:
        raise ValueError('Historique insuffisant pour la validation chronologique interne')
    return train, valid


def fit_trajectory(samples, cutoff, kind, max_epochs):
    factory = LinearTrajectory if kind == 'linear' else ConvTrajectory
    train, valid = split_inner(samples, cutoff)
    torch.manual_seed(42)
    chosen = train_epochs(factory(), train, max_epochs, valid=valid)
    torch.manual_seed(42)
    model = factory()
    train_epochs(model, samples, chosen)
    model.eval()
    return model, chosen, len(train), len(valid)


def reference(frame, start, delay):
    cutoff = start - pd.Timedelta(hours=1)
    train = multi_horizon_training_features(frame, cutoff, availability_delay_hours=delay)
    window = build_window(frame, start, availability_delay_hours=delay)
    for target in ('q', 'k'):
        model, _ = fit_model(train, target, multi_horizon=True)
        window[f'{target}_reference'] = predict_model(model, window, target, multi_horizon=True)
    return window


def run_window(targets, series, start, delay, max_epochs):
    available = start - pd.Timedelta(hours=1 + delay)
    samples = examples_until(series, available, delay)
    print(f'{start.date()} délai {delay} : {len(samples)} trajectoires', flush=True)
    models, metadata = {}, []
    for kind in ('linear', 'conv'):
        models[kind], epochs, n_train, n_valid = fit_trajectory(samples, available, kind, max_epochs)
        metadata.append({'window_start': start, 'delay_hours': delay, 'model': kind, 'epochs': epochs,
                         'train_examples_inner': n_train, 'validation_examples_inner': n_valid,
                         'train_examples_final': len(samples), 'train_arcs': len({x['arc'] for x in samples})})
    window = reference(targets, start, delay)
    for target in ('q', 'k'):
        for kind in models:
            window[f'{target}_{kind}'] = np.nan
    for arc in ARCS:
        item = example(series[arc], start, delay)
        if item is None:
            continue
        past = torch.from_numpy(item['past'][None].astype(np.float32))
        calendar = torch.from_numpy(item['calendar'][None].astype(np.float32))
        rows = window.arc.eq(arc)
        if rows.sum() != HORIZON:
            raise ValueError(f'Grille incomplète pour {arc}')
        # build_window conserve l'ordre horaire pour chaque arc.
        if not window.loc[rows, 't_1h'].is_monotonic_increasing:
            raise ValueError('Les heures ne sont pas ordonnées')
        for kind, model in models.items():
            with torch.no_grad():
                pred = model(past, calendar).numpy()[0] * item['scale']
            window.loc[rows, f'q_{kind}'] = np.clip(pred[:, 0], 0, None)
            window.loc[rows, f'k_{kind}'] = np.clip(pred[:, 1], 0, 100)
    window['delay_hours'] = delay
    window['horizon_day'] = (window.horizon_hours - 1) // 24 + 1
    cols = ['window_start', 't_1h', 'arc', 'delay_hours', 'horizon_day', 'is_public_holiday', 'q', 'k', 'is_imputed_q', 'is_imputed_k']
    cols += [f'{target}_{method}' for target in ('q', 'k') for method in METHODS]
    return window[cols], pd.DataFrame(metadata)


def scores(data, keys):
    rows = []
    for values, group in data.groupby(keys, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        for target in ('q', 'k'):
            valid = group[target].notna() & group[f'is_imputed_{target}'].eq(0)
            for method in METHODS:
                valid &= group[f'{target}_{method}'].notna()
            row = dict(zip(keys, values)) | {'target': target, 'n': int(valid.sum())}
            for method in METHODS:
                error = group.loc[valid, f'{target}_{method}'] - group.loc[valid, target]
                row[f'{method}_mae'] = error.abs().mean()
                row[f'{method}_rmse'] = np.sqrt((error ** 2).mean())
                row[f'{method}_bias'] = error.mean()
            rows.append(row)
    return pd.DataFrame(rows)


def daily(data):
    data = data.copy()
    data['date'] = data.t_1h.dt.normalize()
    rows = []
    for key, group in data.groupby(['window_start', 'delay_hours', 'arc', 'date']):
        for target in ('q', 'k'):
            valid = group[target].notna() & group[f'is_imputed_{target}'].eq(0)
            for method in METHODS:
                valid &= group[f'{target}_{method}'].notna()
            g = group.loc[valid]
            if len(g) != 24 or g.t_1h.nunique() != 24:
                continue
            row = dict(zip(['window_start', 'delay_hours', 'arc', 'date'], key)) | {'target': target, 'truth_total': g[target].sum()}
            actual_peak = g.loc[g[target].idxmax(), 't_1h'].hour
            for method in METHODS:
                row[f'{method}_total'] = g[f'{target}_{method}'].sum()
                row[f'{method}_peak_shift_hours'] = g.loc[g[f'{target}_{method}'].idxmax(), 't_1h'].hour - actual_peak
            rows.append(row)
    return pd.DataFrame(rows)


def daily_summary(days):
    rows = []
    if days.empty:
        return pd.DataFrame()
    for key, group in days.groupby(['delay_hours', 'arc', 'target']):
        row = dict(zip(['delay_hours', 'arc', 'target'], key)) | {'complete_days': len(group)}
        for method in METHODS:
            row[f'{method}_daily_total_mae'] = (group[f'{method}_total'] - group.truth_total).abs().mean()
            row[f'{method}_peak_absolute_shift_hours'] = group[f'{method}_peak_shift_hours'].abs().mean()
        rows.append(row)
    return pd.DataFrame(rows)


def window_diagnostics(data):
    windows = scores(data, ['delay_hours', 'arc', 'window_start'])
    rows = []
    for key, group in windows[windows.n.gt(0)].groupby(['delay_hours', 'arc', 'target']):
        for method in ('linear', 'conv'):
            rows.append(dict(zip(['delay_hours', 'arc', 'target'], key)) | {
                'method': method, 'windows': len(group), 'wins': int((group[f'{method}_mae'] < group.reference_mae).sum()),
                'reference_worst_mae': group.reference_mae.max(), 'method_worst_mae': group[f'{method}_mae'].max()})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows', type=int, default=4)
    parser.add_argument('--delays', type=int, nargs='+', default=[0, 120])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--event-start', type=pd.Timestamp)
    parser.add_argument('--output', type=Path, default=Path('data/processed/trajectory_analysis'))
    args = parser.parse_args()
    targets = pd.read_csv('data/processed/traffic_clean.csv', parse_dates=['t_1h'])
    panel = pd.read_csv('data/processed/panel/traffic_panel_clean.csv', parse_dates=['t_1h'])
    frame = pd.concat([targets, panel], ignore_index=True).sort_values(['arc', 't_1h'])
    series = make_series(frame)
    starts = [args.event_start] if args.event_start is not None else choose_starts(targets, args.windows)
    args.output.mkdir(parents=True, exist_ok=True)
    parts, metadata = [], []
    for delay in args.delays:
        for start in starts:
            tag = f'{start:%Y%m%d}_{delay}h'
            checkpoint, meta = args.output / f'checkpoint_{tag}.csv', args.output / f'metadata_{tag}.csv'
            if checkpoint.exists() and meta.exists():
                pred = pd.read_csv(checkpoint, parse_dates=['t_1h', 'window_start'])
                record = pd.read_csv(meta)
            else:
                pred, record = run_window(targets, series, start, delay, args.epochs)
                pred.to_csv(checkpoint, index=False)
                record.to_csv(meta, index=False)
            parts.append(pred)
            metadata.append(record)
            result = pd.concat(parts, ignore_index=True)
            result.to_csv(args.output / 'predictions.csv', index=False)
            pd.concat(metadata, ignore_index=True).to_csv(args.output / 'training.csv', index=False)
            for name, keys in [('summary', ['delay_hours', 'arc']), ('windows', ['delay_hours', 'arc', 'window_start']),
                               ('horizons', ['delay_hours', 'arc', 'horizon_day']), ('holidays', ['delay_hours', 'arc', 'is_public_holiday'])]:
                scores(result, keys).to_csv(args.output / f'{name}.csv', index=False)
            days = daily(result)
            days.to_csv(args.output / 'daily.csv', index=False)
            daily_summary(days).to_csv(args.output / 'daily_summary.csv', index=False)
            window_diagnostics(result).to_csv(args.output / 'window_diagnostics.csv', index=False)


if __name__ == '__main__':
    main()
