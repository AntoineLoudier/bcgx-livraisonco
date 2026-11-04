"""Contexte des arcs adjacents orientés, gelé à chaque émission."""
import numpy as np
import pandas as pd
from src.data_loader import TARGET_ARCS


def directed_links(selection):
    links = []
    for target in TARGET_ARCS:
        upstream = {s.casefold() for s in target['amont']}
        downstream = {s.casefold() for s in target['aval']}
        for item in selection:
            a = item['libelle_nd_amont'].casefold()
            b = item['libelle_nd_aval'].casefold()
            # Une incidence partagée ne signifie pas que l'arc suit la circulation.
            # Exclure l'inverse exact du tronçon cible, même s'il aboutit à son amont.
            if a in downstream and b in upstream:
                continue
            direction = 'upstream' if b in upstream else 'downstream' if a in downstream else None
            if direction:
                links.append({'arc': target['arc_name'], 'neighbor': 'panel_' + str(item['iu_ac']), 'direction': direction})
    return pd.DataFrame(links, columns=['arc', 'neighbor', 'direction']).drop_duplicates()


def neighbor_table(frame, links):
    parts = []
    for link in links.itertuples(index=False):
        group = frame[frame.arc.eq(link.neighbor)].sort_values('t_1h').copy()
        if group.empty:
            continue
        result = group[['t_1h']].copy()
        for target in ('q', 'k'):
            real = group[target].where(group[f'is_imputed_{target}'].eq(0))
            typical = real.rolling(24 * 28, min_periods=24).median().clip(lower=10 if target == 'q' else 1)
            for hours in (24, 168):
                roll = real.rolling(hours, min_periods=1)
                result[f'{target}_neighbor_{hours}_relative'] = roll.mean() / typical
                result[f'{target}_neighbor_{hours}_coverage'] = roll.count() / hours
        result['arc'], result['direction'] = link.arc, link.direction
        parts.append(result)
    if not parts:
        return pd.DataFrame(columns=['arc', 't_1h'])
    data = pd.concat(parts, ignore_index=True)
    data = data.groupby(['arc', 't_1h', 'direction'], as_index=False).mean(numeric_only=True)
    data = data.pivot(index=['arc', 't_1h'], columns='direction')
    data.columns = [f'{a}_{b}' for a, b in data.columns]
    return data.reset_index()


def attach_neighbors(examples, table):
    data = examples.copy()
    # L'horizon est toujours calculé depuis l'émission à 23 h.
    data['_issued_at'] = data.t_1h - pd.to_timedelta(data.horizon_hours, unit='h')
    data = data.merge(table, left_on=['arc', '_issued_at'], right_on=['arc', 't_1h'], how='left', suffixes=('', '_neighbor_at'), validate='many_to_one')
    return data.drop(columns=['_issued_at', 't_1h_neighbor_at'])


def neighbor_columns():
    return [f'{target}_neighbor_{hours}_{kind}_{direction}' for target in ('q', 'k') for hours in (24, 168)
            for kind in ('relative', 'coverage') for direction in ('upstream', 'downstream')]
