"""Contrôle d'identité et nettoyage du panel acquis, sans toucher aux cibles."""
import json
from pathlib import Path
import pandas as pd
from src.preprocessing import clean_physical_anomalies, impute_missing_values_no_leakage


def prepare_panel(raw, selection, start, end):
    metadata = pd.DataFrame(selection).copy()
    metadata['iu_ac'] = metadata.iu_ac.astype(str)
    raw = raw.copy()
    raw['iu_ac'] = raw.iu_ac.astype(str)
    selected = metadata[['iu_ac', 'libelle_nd_amont', 'libelle_nd_aval']].rename(columns={
        'libelle_nd_amont': 'expected_upstream', 'libelle_nd_aval': 'expected_downstream'})
    raw = raw.merge(selected, on='iu_ac', how='left', validate='many_to_one')
    identity = (raw.libelle_nd_amont.fillna('').str.casefold().eq(raw.expected_upstream.fillna('').str.casefold()) &
                raw.libelle_nd_aval.fillna('').str.casefold().eq(raw.expected_downstream.fillna('').str.casefold()))
    audit = raw.assign(identity_match=identity).groupby('iu_ac').agg(
        rows=('iu_ac', 'size'), matching_rows=('identity_match', 'sum'),
        upstream_variants=('libelle_nd_amont', 'nunique'), downstream_variants=('libelle_nd_aval', 'nunique')).reset_index()
    raw = raw[identity].copy()
    raw['t_1h'] = pd.to_datetime(raw.t_1h, utc=True).dt.tz_convert('Europe/Paris').dt.tz_localize(None)
    raw = raw[raw.t_1h.between(start, end)].copy()
    raw['arc'] = 'panel_' + raw.iu_ac
    raw = clean_physical_anomalies(raw)
    raw = raw.groupby(['arc', 't_1h'], as_index=False).agg({'q': 'mean', 'k': 'mean', 'is_road_closed': 'max'})
    hours = pd.date_range(start, end, freq='h')
    grid = pd.MultiIndex.from_product([('panel_' + metadata.iu_ac), hours], names=['arc', 't_1h']).to_frame(index=False)
    grid = grid.merge(raw, on=['arc', 't_1h'], how='left', validate='one_to_one').sort_values(['arc', 't_1h']).reset_index(drop=True)
    grid['is_road_closed'] = grid.is_road_closed.fillna(0)
    grid = impute_missing_values_no_leakage(grid)
    coverage = grid.groupby('arc').agg(hours=('t_1h', 'size'), imputed_q=('is_imputed_q', 'sum'), imputed_k=('is_imputed_k', 'sum')).reset_index()
    coverage['iu_ac'] = coverage.arc.str.removeprefix('panel_')
    audit = metadata.merge(audit, on='iu_ac', how='left').merge(coverage, on='iu_ac', how='left')
    return grid, audit


def main():
    source = Path('data/raw/panel')
    out = Path('data/processed/panel')
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(source / 'traffic_panel.csv', sep=';', dtype={'iu_ac': str})
    selection = json.loads((source / 'selection.json').read_text())['selection']
    targets = pd.read_csv('data/processed/traffic_clean.csv', parse_dates=['t_1h'])
    panel, audit = prepare_panel(raw, selection, targets.t_1h.min(), targets.t_1h.max())
    panel.to_csv(out / 'traffic_panel_clean.csv', index=False)
    audit.to_csv(out / 'audit.csv', index=False)
    print(audit[['iu_ac', 'rows', 'matching_rows', 'upstream_variants', 'downstream_variants', 'imputed_q', 'imputed_k']].to_string(index=False))
    neighbor_selection = source / 'neighbors_selection.json'
    if neighbor_selection.exists() and (source / 'traffic_neighbors.csv').exists():
        raw = pd.read_csv(source / 'traffic_neighbors.csv', sep=';', dtype={'iu_ac': str})
        selection = json.loads(neighbor_selection.read_text())['selection']
        panel, audit = prepare_panel(raw, selection, targets.t_1h.min(), targets.t_1h.max())
        panel.to_csv(out / 'traffic_neighbors_clean.csv', index=False)
        audit.to_csv(out / 'neighbors_audit.csv', index=False)
        print('Audit des capteurs voisins :')
        print(audit[['iu_ac', 'rows', 'matching_rows', 'upstream_variants', 'downstream_variants', 'imputed_q', 'imputed_k']].to_string(index=False))


if __name__ == '__main__':
    main()
