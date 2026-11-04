# Mélange du modèle et de la moyenne hebdomadaire

`python main_blend.py` compare le LightGBM multi-horizon à un mélange avec la moyenne pondérée des trois semaines précédentes. Pour chaque cible, le mélange vaut `poids_modele × LightGBM + (1 - poids_modele) × moyenne`. Si la moyenne n'est pas disponible, le mélange reprend la prévision du modèle ; les effectifs concernés sont indiqués dans les scores.

Le poids du modèle est choisi parmi 0, 0,25, 0,5, 0,75 et 1, séparément pour `q` et `k`, en minimisant la MAE sur les **prévisions historiques hors entraînement** des fenêtres entièrement terminées avant l'émission. La calibration regroupe les arcs ; elle utilise uniquement leurs cibles observées et donne le même poids à chaque heure observée. Les quatre premières fenêtres servent au minimum à la calibration et ne sont pas évaluées. Il faut aussi au moins 100 mesures pour choisir un poids ; à égalité d'erreur, le modèle seul est privilégié.

Les modèles LightGBM sont réentraînés à chaque émission sur les données connues jusqu'alors. La pondération est choisie à nouveau pour chaque fenêtre, en utilisant tout l'historique de calibration disponible à cette date. Les fenêtres sont espacées de sept jours et couvrent chacune cinq jours ; les fenêtres de calibration et de validation ne se recouvrent pas. Les paramètres de cette étude sont fixés sans optimiser sur les scores futurs.

Les résultats et les poids sont dans `data/processed/blend_analysis/` : `summary.csv`, `windows.csv`, `horizons.csv`, `holiday_context.csv`, `weights.csv`, `predictions.csv` et `final_weights.json`. Les métriques du modèle et du mélange sont calculées sur les mêmes observations, après les fenêtres de calibration. Les comparaisons avec la moyenne seule utilisent le sous-ensemble où elle est disponible ; les colonnes correspondantes donnent aussi les scores des deux autres méthodes sur ce même sous-ensemble.

`final_weights.json` contient les poids sélectionnés avec les fenêtres terminées au 6 novembre 2026 à 23 h, selon les fichiers de données utilisés. Ils sont expérimentaux et ne sont pas appliqués automatiquement par `main_predict.py`. Les fenêtres historiques ont déjà servi à explorer d'autres variantes ; elles ne constituent pas un test final indépendant. La météo n'est pas utilisée. Le manque de mesures récentes aux Champs-Élysées limite l'évaluation et la calibration de cet arc.

## Résultats de l'expérience

L'expérience utilise 52 fenêtres hebdomadaires du 5 novembre 2025 au 28 octobre 2026 ; les quatre premières servent à initialiser la calibration. Les scores ci-dessous portent sur les mêmes heures réellement observées après cette initialisation. Les poids peuvent changer à chaque émission : sur les 48 fenêtres évaluées, la calibration choisit le modèle seul pour 44 fenêtres pour chaque cible.

| Rue et cible | Heures | MAE LightGBM | MAE mélange | MAE moyenne hebdomadaire |
|---|---:|---:|---:|---:|
| Convention, q | 5 640 | 46,88 | 47,27 | 56,87 |
| Convention, k | 5 640 | 1,095 | 1,104 | 1,404 |
| Saint-Pères, q | 5 672 | 55,95 | 55,97 | 69,49 |
| Saint-Pères, k | 5 672 | 1,380 | 1,380 | 1,678 |

Pour ces observations, la moyenne hebdomadaire était toujours disponible. Les MAE de `q` sont en véhicules/h et celles de `k` en points. Les quatre premières fenêtres ne sont pas incluses dans ces scores ; les mesures des Champs-Élysées disponibles en 2025 se trouvent dans cette période initiale et aucune heure de cet arc n'est évaluable ensuite.

Pour le débit à Convention, la pire MAE hebdomadaire passe de 97,01 à 109,58 véhicules/h avec le mélange ; elle concerne la fenêtre du 24 au 28 décembre 2025. Sur les sept jours fériés présents dans les fenêtres évaluées, la MAE du débit passe de 80,50 à 85,18 véhicules/h à Convention (166 heures) et de 87,38 à 91,34 à Saint-Pères (168 heures). Ces groupes d'heures restent de petits échantillons et ne représentent pas tous les fériés.

Pour l'émission du 6 novembre 2026 à 23 h, la grille sélectionne un poids de **1,0 pour LightGBM** sur chacune des deux cibles à partir des 52 fenêtres historiques terminées. Ce résultat recommande de conserver le modèle seul dans le cadre de cette grille et de ce protocole, sans établir qu'aucun autre mélange ne pourrait être utile. Les prévisions et les modèles de référence ne sont pas modifiés automatiquement.
