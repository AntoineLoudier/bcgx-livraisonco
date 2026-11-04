# Calibration chronologique du biais

`python main_bias_calibration.py` lit les prévisions hors entraînement de `data/processed/blend_analysis/predictions.csv`, produites par `python main_blend.py`, et écrit les résultats dans `data/processed/bias_calibration_analysis/`. Les sorties sont ignorées par Git. Aucun modèle n'est réentraîné ni promu.

Pour chaque émission, rue et cible, la correction est la médiane des résidus `observation − prévision` sur les fenêtres de 120 heures entièrement terminées à cette émission. `bias_expanding` utilise tout l'historique disponible ; `bias_recent` utilise les huit dernières fenêtres globales terminées, même si certaines ne contiennent aucune mesure de la rue. Il faut quatre fenêtres avec observations et 100 observations réelles ; sinon la correction vaut zéro. Les cibles imputées sont exclues. Les prédictions sont bornées à `q ≥ 0` et `0 ≤ k ≤ 100` ; la référence est bornée également (aucune prédiction initiale ne dépasse ces bornes dans les données). Le biais rapporté est prévision moins observation.

Chaque méthode est comparée à la référence sur les mêmes heures. `summary.csv`, `windows.csv`, `horizons.csv` et `holidays.csv` présentent les scores pour toutes les heures, ainsi que séparément les cas calibrés et les replis ; les comparaisons en statut calibré portent sur les mêmes lignes pour les deux méthodes. `window_diagnostics.csv` donne les fenêtres gagnées, perdues et pires MAE. `corrections.csv` indique pour chaque émission la correction, les effectifs et le repli, y compris en l'absence de cible évaluable ; `fallbacks.csv` synthétise les fréquences. `predictions.csv` ne contient que les cibles effectivement observées.

## Résultats

Sur toutes les fenêtres, pour le débit (MAE en véhicules/h) :

| Rue | Référence | Historique complet | Huit fenêtres récentes |
|---|---:|---:|---:|
| Convention | 46,47 | 46,65 | 46,96 |
| Saint-Pères | 54,63 | 54,50 | 56,95 |

Sur les seules fenêtres calibrées, la variante complète gagne 24 fenêtres sur 46 à Convention et 28 sur 48 à Saint-Pères ; la MAE de Saint-Pères passe de 55,95 à 55,81. Sa pire MAE y passe de 118,67 à 125,83. Aux jours fériés, elle passe de 80,50 à 80,55 à Convention et de 87,38 à 88,32 à Saint-Pères, sur respectivement 166 et 168 heures observées. Ces résultats fériés agrègent seulement quelques événements. Le faible gain global à Saint-Pères ne justifie pas une adoption au vu de la dégradation des pires fenêtres et des fériés. La variante récente dégrade la MAE sur les deux rues.

Les quatre premières fenêtres sont en repli pour Saint-Pères ; Convention compte six émissions en repli sur 52, dont quatre évaluables. Les Champs-Élysées ne disposent que de 247 observations de débit et 246 d'occupation en début de période : elles surviennent avant que la calibration ne soit suffisante. Les corrections appliquées à ces observations sont donc nulles et aucune évaluation récente de ce tronçon n'est possible.

Les fenêtres historiques ont déjà été explorées à plusieurs reprises ; il ne s'agit pas d'un test indépendant. La référence reste retenue. Une décision robuste demanderait de nouvelles fenêtres indépendantes et davantage de fériés.
