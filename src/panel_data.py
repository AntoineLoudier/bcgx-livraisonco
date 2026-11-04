"""Acquisition reproductible d'un panel de capteurs Paris Open Data.

La sélection utilise exclusivement une période précédant les validations.
Les fichiers sources existants ne sont jamais remplacés.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import requests

API = 'https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents'
SELECTION_END = '2026-05-01T00:00:00Z'
SELECTION_START = '2025-10-01T00:00:00Z'
END = '2026-11-03T00:00:00Z'
TARGET_PAIRS = {('lecourbe-convention', 'convention-blomet'),
                ('sts_peres-voltaire', 'sts_peres-universite'),
                ('av_champs_elysees-washington', 'av_champs_elysees-berri')}


def discover(session, limit=20):
    params = {'select': 'iu_ac,libelle,libelle_nd_amont,libelle_nd_aval,count(*) as n',
              'where': f"t_1h >= '{SELECTION_START}' and t_1h < '{SELECTION_END}' and q is not null and k is not null",
              'group_by': 'iu_ac,libelle,libelle_nd_amont,libelle_nd_aval',
              'order_by': 'n desc', 'limit': 100}
    response = session.get(f'{API}/records', params=params, timeout=120)
    response.raise_for_status()
    rows = response.json()['results']
    result = []
    seen = set()
    for row in rows:
        arc_id = str(row['iu_ac'])
        pair = (str(row.get('libelle_nd_amont', '')).casefold(), str(row.get('libelle_nd_aval', '')).casefold())
        if arc_id in seen or pair in TARGET_PAIRS or not all(pair):
            continue
        seen.add(arc_id)
        result.append(row)
        if len(result) == limit:
            break
    if len(result) < limit:
        raise ValueError(f'Panel insuffisant : {len(result)} capteurs')
    return result


def acquire(session, panel, destination):
    ids = ','.join("'" + str(row['iu_ac']).replace("'", "''") + "'" for row in panel)
    params = {'select': 'iu_ac,libelle,t_1h,q,k,etat_barre,libelle_nd_amont,libelle_nd_aval',
              'where': f"iu_ac in ({ids}) and t_1h >= '{SELECTION_START}' and t_1h < '{END}'", 'delimiter': ';'}
    response = session.get(f'{API}/exports/csv', params=params, stream=True, timeout=(30, 300))
    response.raise_for_status()
    temp = destination.with_suffix('.part')
    with temp.open('wb') as f:
        for chunk in response.iter_content(1024 * 1024):
            f.write(chunk)
    temp.replace(destination)


def discover_neighbors(session):
    # Sommets nommés du brief, sélection des arcs orientés incidents sur la seule période de sélection.
    nodes = sorted({x for pair in TARGET_PAIRS for x in pair})
    clauses = [f"lower({field}) = '{node.replace(chr(39), chr(39) * 2)}'" for field in ('libelle_nd_amont', 'libelle_nd_aval') for node in nodes]
    params = {'select': 'iu_ac,libelle,libelle_nd_amont,libelle_nd_aval,count(*) as n',
              'where': f"t_1h >= '{SELECTION_START}' and t_1h < '{SELECTION_END}' and q is not null and k is not null and (" + ' or '.join(clauses) + ')',
              'group_by': 'iu_ac,libelle,libelle_nd_amont,libelle_nd_aval', 'order_by': 'n desc', 'limit': 100}
    response = session.get(f'{API}/records', params=params, timeout=120)
    response.raise_for_status()
    result, seen = [], set()
    for row in response.json()['results']:
        pair = (str(row.get('libelle_nd_amont', '')).casefold(), str(row.get('libelle_nd_aval', '')).casefold())
        if pair not in TARGET_PAIRS and str(row['iu_ac']) not in seen:
            seen.add(str(row['iu_ac']))
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path('data/raw/panel'))
    parser.add_argument('--neighbors-only', action='store_true')
    parser.add_argument('--selection', type=Path, help='Recharger une sélection enregistrée, sans nouvelle sélection')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        if args.neighbors_only:
            neighbors = json.loads(args.selection.read_text())['selection'] if args.selection else discover_neighbors(session)
            (args.output / 'neighbors_selection.json').write_text(json.dumps({'selection_start_utc': SELECTION_START,
                'selection_end_utc_exclusive': SELECTION_END, 'selection': neighbors}, ensure_ascii=False, indent=2) + '\n')
            if neighbors:
                print(f'{len(neighbors)} capteurs adjacents sélectionnés ; téléchargement...', flush=True)
                acquire(session, neighbors, args.output / 'traffic_neighbors.csv')
            else:
                print('Aucun capteur adjacent identifiable par les noms de sommets.')
            return
        panel = json.loads(args.selection.read_text())['selection'] if args.selection else discover(session, args.count)
        (args.output / 'selection.json').write_text(json.dumps({'selection_start_utc': SELECTION_START,
            'selection_end_utc_exclusive': SELECTION_END, 'download_end_utc_exclusive': END,
            'selection': panel}, ensure_ascii=False, indent=2) + '\n')
        print(f'{len(panel)} tronçons sélectionnés ; téléchargement...', flush=True)
        acquire(session, panel, args.output / 'traffic_panel.csv')
    print('Panel sauvegardé dans', args.output)


if __name__ == '__main__':
    main()
