"""Événements horodatés : seuls les faits connus à l'émission sont utilisables."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

EVENT_FEATURES = ['known_event_direct', 'known_event_adjacent']


def local_time(value):
    time = pd.Timestamp(value)
    if time.tzinfo is None:
        raise ValueError('Les dates de source doivent inclure un fuseau horaire')
    return time.tz_convert('Europe/Paris').tz_localize(None)


def load_events(path=Path('configs/evenements_cibles.json')):
    content = json.loads(path.read_text())
    rows = content['events']
    for row in rows:
        for field in ('event_id', 'arc', 'event_type', 'scope', 'availability', 'start', 'end', 'known_at', 'source'):
            if not row.get(field):
                raise ValueError(f'Champ obligatoire manquant : {field}')
        if row['scope'] not in ('direct', 'adjacent'):
            raise ValueError('Portée non reconnue')
        if row['availability'] not in ('verified', 'inferred'):
            raise ValueError('Statut de disponibilité non reconnu')
    df = pd.DataFrame(rows)
    if df.event_id.duplicated().any():
        raise ValueError('Événement dupliqué : les versions doivent être conservées et traitées explicitement')
    for col in ('start', 'end', 'known_at'):
        df[col] = df[col].map(local_time)
    if (df.end <= df.start).any():
        raise ValueError('Fin avant le début de l’événement')
    return df.sort_values(['start', 'event_id']).reset_index(drop=True)


def add_announced_features(frame, events, issue_times):
    """Recouvrement horaire [début, fin), avec disponibilité par émission.

    Zéro signifie absence d'avis utilisable dans ce catalogue incomplet ;
    il ne prouve pas qu'il n'existe pas d'autre événement.
    """
    out = frame.copy()
    issue = pd.Series(pd.to_datetime(np.asarray(issue_times)), index=out.index)
    if len(issue) != len(out) or issue.isna().any():
        raise ValueError('Une émission valide est requise pour chaque ligne')
    out[EVENT_FEATURES] = 0.0
    for event in events.itertuples(index=False):
        if event.availability != 'verified':
            continue
        valid = out.arc.eq(event.arc) & issue.ge(event.known_at)
        start = out.t_1h
        # Recouvrement mesuré sur l'heure de la cible, y compris les horaires partiels.
        overlap = ((start + pd.Timedelta(hours=1)).clip(upper=event.end) - start.clip(lower=event.start)).dt.total_seconds().div(3600).clip(lower=0, upper=1)
        column = f'known_event_{event.scope}'
        out.loc[valid, column] = np.maximum(out.loc[valid, column], overlap.loc[valid])
    return out


def factors(frame, events, issue_times, scenario):
    """Hypothèses fixes de sensibilité pour q ; elles ne sont pas calibrées."""
    if scenario not in ('none', 'lower', 'upper'):
        raise ValueError('Scénario inconnu')
    flags = add_announced_features(frame, events, issue_times)
    if scenario == 'none':
        return np.ones(len(frame))
    if scenario == 'lower':
        return (1 - .5 * flags.known_event_direct - .2 * flags.known_event_adjacent).clip(lower=0).to_numpy()
    return (1 + .2 * flags.known_event_adjacent).to_numpy()
