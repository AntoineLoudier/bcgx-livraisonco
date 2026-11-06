# Fichier de rendu — Les Architectes

`output_Les Architectes.csv` est la copie figée de l’export généré après l’acquisition du **6 novembre 2026**, pour les heures du **7 novembre 00 h au 11 novembre 23 h**, heure de Paris.

- Séparateur : `;`, encodage UTF-8.
- Une colonne d’index initiale, puis `arc`, `datetime`, `debit_horaire`, `taux_occupation`, conformément à l’exporteur du brief.
- 360 lignes : 120 par rue, deux cibles par ligne.
- **Convention et Saint-Pères sont évaluées** ; les Champs-Élysées sont conservés uniquement pour respecter le format initial attendu par le validateur.

La [présentation complète avec annexes](../presentation/LivraisonCo.pdf) est le support à remettre. Les scores portent sur les deux rues évaluées. Les prévisions n’ont pas encore de score sur la période future du 7–11 novembre.

`manifest.json` conserve l’empreinte du CSV, du support et des preuves associées. Le modèle et les données sont identifiés dans [le manifeste des résultats](../presentation/results/manifest.json).

Depuis la racine, vérifier le rendu sans accès réseau :

```bash
python3 scripts/verify_submission.py
```

Les prochaines exécutions de `main_predict.py` écrivent dans `reports/` par défaut et ne remplacent pas ce fichier figé. Pour reproduire exactement les prévisions du rendu, utiliser les données du 6 novembre et les versions de bibliothèques indiquées dans les manifestes.
