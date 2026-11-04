"""Exemples causaux de quatre semaines vers cinq jours de trafic."""
import numpy as np
import pandas as pd
from src.features import add_calendar_and_event_features

LOOKBACK = 24 * 28
HORIZON = 120
CALENDAR = ('hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend',
            'is_public_holiday', 'is_school_holiday', 'bridge_intensity',
            'is_paris_respire', 'is_armistice_parade_hours')


def make_series(frame):
    """Prépare les tableaux par arc ; aucune statistique ne voit le futur."""
    output = {}
    for arc, group in frame.groupby('arc', sort=True):
        group = group.sort_values('t_1h').set_index('t_1h').asfreq('h')
        grid = group.reset_index().assign(arc=arc)
        calendar = add_calendar_and_event_features(grid)[list(CALENDAR)].to_numpy(dtype=np.float32)
        values = group[['q', 'k']].to_numpy(dtype=np.float32)
        masks = np.column_stack([(group[t].notna() & group[f'is_imputed_{t}'].eq(0)).to_numpy() for t in ('q', 'k')])
        values = np.where(masks, values, 0).astype(np.float32)
        scales = []
        for idx, floor in enumerate((10., 1.)):
            real = pd.Series(np.where(masks[:, idx], values[:, idx], np.nan), index=group.index)
            # Un capteur absent pendant plus de 28 jours garde sa dernière échelle connue.
            scales.append(real.rolling(LOOKBACK, min_periods=1).median().ffill().abs().clip(lower=floor).fillna(floor).to_numpy(dtype=np.float32))
        output[arc] = {'index': group.index, 'values': values, 'masks': masks, 'scales': np.stack(scales, axis=1), 'calendar': calendar}
    return output


def example(series, start, delay_hours=0):
    """Une émission, avec masques séparés pour l'entrée et la vérité future."""
    if delay_hours < 0:
        raise ValueError('Délai négatif')
    start = pd.Timestamp(start)
    if start != start.normalize():
        raise ValueError('La prévision doit commencer à minuit')
    idx = series['index']
    pos = idx.get_indexer([start])[0]
    if pos < 0 or pos + HORIZON > len(idx) or pos - delay_hours - LOOKBACK < 0:
        return None
    end = pos - delay_hours
    sl = slice(end - LOOKBACK, end)
    scale = series['scales'][end - 1].copy()
    past_values = np.clip(series['values'][sl] / scale, 0, 8)
    past = np.concatenate([past_values, series['masks'][sl].astype(np.float32)], axis=1).T
    future = series['calendar'][pos:pos + HORIZON]
    target = series['values'][pos:pos + HORIZON] / scale
    mask = series['masks'][pos:pos + HORIZON].astype(np.float32)
    return {'past': past, 'calendar': future, 'target': target, 'mask': mask, 'scale': scale,
            'start': start}


def examples_until(all_series, cutoff, delay_hours=0):
    """Tout exemple d'entraînement a ses 120 cibles terminées avant cutoff."""
    cutoff = pd.Timestamp(cutoff)
    result = []
    for arc, series in all_series.items():
        first = (series['index'][0] + pd.Timedelta(hours=LOOKBACK + delay_hours)).ceil('D')
        last = (cutoff - pd.Timedelta(hours=HORIZON - 1)).normalize()
        for start in pd.date_range(first, last, freq='D'):
            if start + pd.Timedelta(hours=HORIZON - 1) > cutoff:
                continue
            item = example(series, start, delay_hours)
            if item is not None and item['mask'].any():
                item['arc'] = arc
                result.append(item)
    return result
