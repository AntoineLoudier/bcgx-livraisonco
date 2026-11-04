# Influence de l'ancienneté des observations

La commande `python main_recency.py` compare quatre entraînements du modèle global multi-horizon, pour `q` et `k`, avec les mêmes variables et hyperparamètres : tout l'historique observé, les trois derniers mois calendaires, les six derniers mois calendaires et une pondération exponentielle de demi-vie 180 jours. Avec cette dernière, une cible vieille de 180 jours pèse deux fois moins qu'une cible au moment de l'émission. Les poids sont normalisés à une moyenne de 1 sur les cibles observées de chaque apprentissage. Chaque cible peut être représentée par cinq exemples correspondant aux différents horizons ; ces exemples ne sont pas indépendants.

Les fenêtres de trois et six mois limitent les **cibles utilisées pour l'entraînement**. Les variables restent calculées à partir de l'ensemble des mesures antérieures disponibles à l'émission : elles peuvent donc résumer des observations antérieures à la fenêtre d'entraînement. Aucun modèle de cette étude n'utilise la météo. Les paramètres ont été fixés avant cette comparaison ; ils n'ont pas été optimisés.

## Huit fenêtres récentes

Chaque fenêtre couvre cinq jours avec une seule émission à 23 h la veille. Elles débutent du 9 septembre au 28 octobre 2026, avec un modèle réentraîné avant chaque fenêtre. Les MAE ci-dessous sont calculées sur les mêmes 934 heures observées par rue et par cible, à l'intersection des quatre stratégies.

| Rue et cible | Historique complet | 3 mois | 6 mois | Décroissance 180 j |
|---|---:|---:|---:|---:|
| Convention, q (véhicules/h) | 48,42 | 50,90 | 49,90 | 48,50 |
| Convention, k (points) | 1,134 | 1,165 | 1,158 | 1,146 |
| Saint-Pères, q (véhicules/h) | 67,65 | 69,29 | 64,31 | 67,46 |
| Saint-Pères, k (points) | 1,637 | 1,674 | 1,617 | 1,609 |

Pour le débit à Saint-Pères, six mois gagne sur six des huit fenêtres, mais dégrade le débit de Convention sur l'ensemble des fenêtres. La stratégie de décroissance est proche de l'historique complet ; les trois mois font moins bien sur les quatre agrégats. Ce constat porte sur une seule période récente et ne démontre pas la stabilité future.

## Jours fériés

Douze événements du 1er novembre 2025 au 1er novembre 2026 sont évalués dans des fenêtres dédiées. Les valeurs ci-dessous concernent uniquement les jours fériés eux-mêmes ; veilles et lendemains figurent dans `holidays.csv`. Les scores des quatre méthodes portent sur les mêmes observations.

| Rue et cible | Heures | Historique complet | 3 mois | 6 mois | Décroissance 180 j |
|---|---:|---:|---:|---:|---:|
| Convention, q | 275 | 74,52 | 72,87 | 77,11 | 73,73 |
| Convention, k | 275 | 0,80 | 0,78 | 0,79 | 0,80 |
| Saint-Pères, q | 288 | 84,47 | 84,18 | 85,50 | 83,15 |
| Saint-Pères, k | 288 | 1,20 | 1,18 | 1,16 | 1,20 |

Les moyennes masquent des différences : le 8 mai 2026 à Convention, la MAE de débit est de 101,68 avec tout l'historique, 84,49 avec trois mois, 120,45 avec six mois et 98,86 avec décroissance. À d'autres dates, les classements changent. Les premiers fériés ont moins de trois ou six mois de données disponibles : pour certains, les trois fenêtres contiennent exactement le même historique. Les événements ne sont ni nombreux ni indépendants, et le 1er novembre 2026 figure également dans l'une des fenêtres récentes.

## Champs-Élysées et limites

Les dernières mesures des Champs-Élysées datent de novembre 2025. Aux émissions récentes, les fenêtres de trois et six mois ne contiennent donc aucune cible de cette rue. Les prédictions correspondantes ne sont pas considérées évaluables pour cette comparaison, même si le modèle global peut techniquement produire une valeur. Le fichier `training_counts.csv` contient les effectifs, y compris les zéros. Les deux fériés des Champs-Élysées évaluables sont en novembre 2025 : ils ne vérifient pas la situation de 2026.

Les résultats sont descriptifs : les fenêtres ont déjà servi à d'autres analyses et ne constituent pas un test final indépendant. Le choix d'un historique ou d'une demi-vie demanderait une sélection sur le passé et une évaluation sur de nouvelles fenêtres. Aucune stratégie n'est adoptée automatiquement. Les fichiers détaillés, prédictions, effectifs et scores par horizon sont enregistrés dans `data/processed/recency_analysis/`.
