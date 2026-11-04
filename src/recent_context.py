"""Contexte observé à l'émission, sans accès aux mesures ultérieures."""
import numpy as np
import pandas as pd

WINDOWS = (6, 24, 72, 168)


def context_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Calcule des résumés causaux pour chaque heure et chaque capteur.

    Le profil de référence d'une mesure ne voit que les observations du même
    jour de semaine et heure strictement antérieures à cette mesure.
    """
    frame = frame.sort_values(['arc', 't_1h']).copy()
    if frame.duplicated(['arc', 't_1h']).any():
        raise ValueError('Doublons arc/heure')
    pieces = []
    for arc, group in frame.groupby('arc', sort=False):
        group = group.set_index('t_1h').asfreq('h')
        out = pd.DataFrame(index=group.index)
        for target in ('q', 'k'):
            real = group[target].where(group[f'is_imputed_{target}'].eq(0))
            dow, hour = group.index.dayofweek, group.index.hour
            usual = real.groupby([dow, hour]).transform(lambda s: s.shift().expanding(min_periods=1).mean())
            residual = real - usual
            for n in WINDOWS:
                roll = real.rolling(n, min_periods=1)
                out[f'{target}_ctx_mean_{n}'] = roll.mean()
                out[f'{target}_ctx_std_{n}'] = roll.std()
                out[f'{target}_ctx_coverage_{n}'] = roll.count() / n
                out[f'{target}_ctx_anomaly_{n}'] = residual.rolling(n, min_periods=1).mean()
            last = pd.Series(group.index.where(real.notna()), index=group.index).ffill()
            out[f'{target}_ctx_age'] = (group.index - last) / pd.Timedelta(hours=1)
            # Trois derniers jours complets disponibles, profil horaire entier.
            # À 23h le jour courant est complet ; sinon le précédent est complet.
            end = group.index.normalize() - pd.Timedelta(hours=1)
            end = end.where(group.index.hour != 23, group.index)
            for day in range(3):
                for h in range(24):
                    lookup = end - pd.Timedelta(days=day, hours=23-h)
                    out[f'{target}_ctx_day{day+1}_h{h:02}'] = real.reindex(lookup).to_numpy()
        out['arc'] = arc
        out['available_at'] = out.index
        pieces.append(out.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def attach_context(examples: pd.DataFrame, table: pd.DataFrame, delay_hours: int) -> pd.DataFrame:
    """Relie chaque cible au contexte connu à sa propre émission historique."""
    if delay_hours < 0:
        raise ValueError('Délai négatif')
    result = examples.copy()
    emission = result.t_1h - pd.to_timedelta(result.horizon_hours, unit='h')
    result['_context_at'] = emission - pd.Timedelta(hours=delay_hours)
    result = result.merge(table, left_on=['arc', '_context_at'], right_on=['arc', 'available_at'], how='left', validate='many_to_one')
    return result.drop(columns=['_context_at', 'available_at'])


def context_columns(target: str) -> list[str]:
    cols = []
    for metric in (target, 'k' if target == 'q' else 'q'):
        cols += [f'{metric}_ctx_{name}_{n}' for n in WINDOWS for name in ('mean', 'std', 'coverage', 'anomaly')]
        cols += [f'{metric}_ctx_age']
        cols += [f'{metric}_ctx_day{day}_h{hour:02}' for day in range(1, 4) for hour in range(24)]
    return cols
