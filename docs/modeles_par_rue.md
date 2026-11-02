# Comparaison des modèles par rue

`python main_compare_streets.py` compare deux modèles LightGBM globaux à six modèles LightGBM, un par rue et cible. Ils utilisent la même liste de variables, les mêmes hyperparamètres et les mêmes règles d'imputation et de disponibilité. Le modèle local n'est entraîné que sur les cibles observées de sa rue ; un minimum de 100 observations est requis. Le modèle global utilise les trois rues. Les modèles sont réentraînés avant chaque fenêtre à partir des données connues à la date d'émission, puis évalués sur des observations réelles. Les jeux de validation de trois méthodes (global, local et moyenne des trois semaines) sont restreints à leur intersection.

## Cinquante-deux fenêtres hebdomadaires

Ces fenêtres de cinq jours débutent du 5 novembre 2025 au 28 octobre 2026. Les erreurs ci-dessous sont des MAE sur les mêmes heures ; les fenêtres gagnées sont celles où l'erreur du modèle local est strictement inférieure à celle du global.

| Rue | Cible | MAE global | MAE local | Heures communes | Fenêtres gagnées par le local |
|---|---|---:|---:|---:|---:|
| Convention | q | 46,45 | 46,82 | 5 880 | 24 / 50 |
| Convention | k | 1,1124 | 1,1126 | 5 881 | 28 / 51 |
| Saint-Pères | q | 54,42 | 54,26 | 6 030 | 26 / 51 |
| Saint-Pères | k | 1,3734 | 1,3769 | 6 030 | 26 / 51 |
| Champs-Élysées | q | 123,43 | 126,50 | 247 | 0 / 3 |
| Champs-Élysées | k | 3,1082 | 3,1196 | 246 | 1 / 3 |

Les données ne montrent pas de gain clair des modèles séparés avec cette configuration. La différence entre les MAE globales et locales pour Convention et Saint-Pères est faible et les gains par fenêtre sont partagés. Les trois fenêtres évaluables des Champs-Élysées datent toutes de novembre 2025 : elles ne permettent pas de conclure pour 2026.

## Fériés

Douze fériés ont été examinés sur des fenêtres dédiées de cinq jours. Sur les jours fériés, la MAE du débit est de 74,19 pour le modèle local contre 75,85 pour le global à Convention (275 heures), et de 87,51 contre 84,14 à Saint-Pères (247 heures). Ces valeurs agrégées masquent des variations entre jours et portent sur un petit nombre d'événements.

Le 11 novembre 2025 aux Champs-Élysées, la MAE du débit passe de 258,44 véhicules/h avec le modèle global à 252,47 avec le local, sur 24 observations. Les biais correspondants restent des surestimations moyennes de 207,74 et 199,68 véhicules/h. L'amélioration locale est donc faible au regard de l'erreur ; un seul 11 novembre ne permet pas de valider la prévision 2026.

## Utilisation et limites

Les résultats détaillés sont dans `data/processed/street_models_analysis/` : `weekly_summary.csv`, `weekly_window_scores.csv`, `holiday_scores.csv`, prédictions horaires et nombres de mesures d'entraînement. La météo n'est pas utilisée. Les jours fériés étudiés et le traitement du changement d'heure sont décrits dans l'étude de stabilité.

Les six modèles finaux sont enregistrés dans `models/per_street/` avec leurs variables et leurs dernières observations dans `metadata.json`. Ils ne sont pas intégrés à une chaîne d'inférence. Pour les Champs-Élysées, les dernières mesures utilisées datent du 19 novembre 2025 : le modèle reste particulièrement incertain pour 2026. L'expérience compare une seule configuration de modèle et ne prouve pas qu'aucun modèle local ou mélange de modèles ne pourrait être meilleur. Il faut confirmer un éventuel choix sur d'autres fenêtres avant usage.
