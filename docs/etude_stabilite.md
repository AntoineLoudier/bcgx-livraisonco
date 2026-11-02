# Stabilité du premier modèle LightGBM

Étude sur `data/processed/traffic_clean.csv` disponible jusqu'au 2 novembre 2026. La commande `python main_stability.py` réentraîne le modèle à chaque date d'émission, sans modifier les modèles enregistrés. Les résultats détaillés sont générés dans `data/processed/stability_analysis/`.

## Fenêtres hebdomadaires

Les 52 fenêtres de cinq jours commencent chaque semaine du 5 novembre 2025 au 28 octobre 2026. Pour chaque fenêtre, les données sont arrêtées à 23 h la veille du premier jour. Les erreurs ne sont calculées que sur les observations réelles. La référence est la moyenne pondérée des trois semaines précédentes et les comparaisons portent, pour chaque paire, sur exactement les mêmes heures.

| Arc | Cible | MAE LightGBM | MAE référence | Heures comparées | Fenêtres gagnées par LightGBM |
|---|---|---:|---:|---:|---:|
| Convention | q | 46,45 | 56,10 | 5 880 | 40 / 50 |
| Convention | k | 1,11 | 1,41 | 5 881 | 45 / 51 |
| Saint-Pères | q | 54,42 | 67,47 | 6 030 | 37 / 51 |
| Saint-Pères | k | 1,37 | 1,66 | 6 030 | 46 / 51 |
| Champs-Élysées | q | 123,43 | 125,82 | 247 | 3 / 3 |
| Champs-Élysées | k | 3,11 | 3,11 | 246 | 1 / 3 |

Les effectifs diffèrent parce que les observations ou les retards manquent sur certaines heures. Les trois fenêtres des Champs-Élysées se situent toutes en novembre 2025 : elles ne permettent pas de conclure pour novembre 2026. L'indicateur « fenêtres gagnées » compte chaque fenêtre de façon égale ; la MAE agrégée pondère les heures observées.

## Jours fériés

Douze fériés disponibles entre le 1er novembre 2025 et le 1er novembre 2026 ont chacun une fenêtre dédiée de cinq jours, centrée lorsque les données le permettent. Les scores du jour férié sont calculés séparément de ceux de sa veille et de son lendemain. Les fenêtres peuvent se chevaucher et certaines journées sont incomplètes ; ces cas ne constituent pas douze expériences indépendantes ou représentatives de tous les fériés.

Sur les seuls jours fériés, pour Convention et Saint-Pères regroupés, la MAE du débit est de 79,77 véhicules/h pour LightGBM contre 123,00 pour la référence, sur 522 observations communes. L'erreur d'occupation est de 0,97 contre 2,10 points, sur les mêmes 522 heures. Pour le débit, LightGBM gagne sur 10 des 12 jours évalués à Convention et 10 des 11 jours évalués à Saint-Pères.

Le résultat varie fortement selon le jour. Le 8 mai 2026, l'erreur de débit de LightGBM est de 108,55 contre 48,13 véhicules/h à Convention, et de 83,07 contre 28,35 à Saint-Pères. Le 11 novembre 2025, les Champs-Élysées présentent une erreur de débit de 258,44 contre 355,56 véhicules/h pour la référence, sur 24 observations ; le biais moyen du modèle est une surestimation de 207,74 véhicules/h. Ce seul événement n'établit pas ce qui se passera en 2026.

Les veilles et lendemains des fériés sont aussi détaillés dans `holiday_scores.csv`. Le lendemain du 1er novembre 2026 est hors de la fenêtre exploitable et n'est pas évalué pour cet événement.

## Interprétation et pistes

Le modèle gagne régulièrement sur Convention et Saint-Pères, mais les écarts par fenêtre et les erreurs très variables des fériés appellent une validation plus ciblée avant de le généraliser. Étudier en priorité les erreurs des 8 mai et 11 novembre, comparer aux profils des autres jours fériés et vérifier si l'hypothèse de fermeture est bien connue à la date d'émission. Tester les caractéristiques de pont sur d'autres années ou sources, puis comparer une stratégie par arc et par type de jour.

L'évaluation ne résout pas l'absence de mesures récentes aux Champs-Élysées. Les prévisions météo historiques ne sont pas vérifiables avec les fichiers actuels, et ne sont pas utilisées. L'heure locale répétée au changement d'heure est agrégée. Aucun intervalle d'incertitude statistique n'est calculé ; ces résultats restent descriptifs.
