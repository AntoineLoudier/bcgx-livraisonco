# Prévision conjointe des trajectoires horaires

`python main_trajectory.py` compare le LightGBM multi-horizon existant à deux modèles produisant chacun les 120 heures de débit et d'occupation en une fois :

* `linear` : régression linéaire à sorties multiples avec régularisation AdamW ;
* `conv` : trois couches convolutives temporelles, suivies de couches denses.

Les deux modèles sont entraînés conjointement sur les trois rues cibles et les 20 arcs du panel décrit dans `docs/panel_transfert.md`. Ils n'ont pas d'entrée identifiant chaque tronçon ; les paramètres sont partagés entre arcs. La référence LightGBM est celle du projet, entraînée uniquement sur les trois cibles. Cette comparaison combine donc l'effet de l'architecture et de l'ensemble d'entraînement. Elle ne permet pas d'attribuer les écarts à une seule cause. PatchTST n'a pas été testé.

## Exemples et masques

Chaque exemple utilise les 672 dernières heures disponibles (28 jours) des deux indicateurs, et les variables calendaires connues des 120 heures futures. Deux masques indiquent quelles valeurs d'entrée sont des mesures réelles. Les valeurs imputées sont remplacées par zéro et le masque reste à zéro ; elles ne sont pas présentées comme des mesures. Si l'émission est retardée de 120 h par rapport aux dernières données accessibles, l'entrée s'arrête cinq jours avant l'émission et l'horizon prédit reste les cinq jours après celle-ci.

Chaque arc et indicateur possède une échelle calculée à l'instant de disponibilité : médiane des seules observations réelles des 28 jours de contexte, soit 672 heures, avec plancher 10 pour `q` et 1 pour `k`. Si ce contexte ne contient aucune mesure réelle, la dernière médiane historique disponible est conservée ; si aucune mesure n'existe, le plancher sert de repli. Les entrées normalisées sont bornées entre 0 et 8, avec les masques fournis au modèle. Les sorties sont ramenées à leur échelle puis bornées à `q >= 0` et `0 <= k <= 100`.

La perte est une MAE sur les sorties normalisées uniquement aux heures et pour les indicateurs dont les cibles sont réellement observées. Les exemples sont échantillonnés de manière à donner le même poids attendu à chaque arc, avec remise. Les deux indicateurs contribuent selon le nombre de leurs cibles observées. Les périodes manquantes ne contribuent pas à la perte.

## Validation

Pour chaque fenêtre extérieure et délai, l'entraînement n'utilise que des trajectoires dont toutes les dates cibles sont terminées au dernier instant disponible. Le nombre d'époques est choisi sur une validation chronologique interne de 21 jours, sans chevauchement de dates cibles avec l'entraînement interne, puis le modèle est réinitialisé et réentraîné sur toutes les données connues. Maximum 15 époques, arrêt après trois époques sans amélioration, graine 42 ; hyperparamètres fixés avant l'expérience. Les exemples internes proches de la frontière dont les cinq jours se chevaucheraient sont exclus de la sélection.

Quatre fenêtres non chevauchantes du 30 septembre au 28 octobre 2026 sont évaluées pour des délais de 0 et 120 heures. Le 1er novembre est étudié dans une fenêtre séparée. Les scores de chaque modèle portent sur les mêmes cibles réellement observées. Les totaux et heures de pointe journaliers exigent 24 heures observées et les prévisions des trois modèles ; les égalités de pointe utilisent la première occurrence. Les heures sont des heures locales naïves et l'heure répétée du passage à l'heure d'hiver est agrégée, comme dans le pipeline existant.

Les fenêtres ont déjà servi aux autres expériences ; elles ne constituent pas un test indépendant et ne couvrent qu'un seul férié. Les Champs-Élysées ne disposent d'aucune mesure récente permettant de calculer ces scores. Le modèle n'est pas intégré à l'inférence finale et aucun poids entraîné n'est promu.

## Résultats

MAE du débit, en véhicules/h ; les méthodes sont évaluées sur les mêmes heures :

| Délai | Rue | Heures | LightGBM | Linéaire | Convolutif |
|---|---|---:|---:|---:|---:|
| 0 h | Convention | 457 | 48,09 | 74,25 | 59,47 |
| 0 h | Saint-Pères | 456 | 60,85 | 96,31 | 76,49 |
| 120 h | Convention | 457 | 50,27 | 82,05 | 62,46 |
| 120 h | Saint-Pères | 456 | 61,82 | 113,92 | 79,91 |

Pour l'occupation à 0 h, les MAE LightGBM / linéaire / convolutif sont de 1,089 / 1,414 / 1,163 points à Convention et 1,541 / 1,913 / 1,623 à Saint-Pères. À 120 h, elles sont de 1,067 / 1,449 / 1,226 et 1,548 / 1,843 / 1,558 respectivement. Le convolutif ne gagne aucune des quatre fenêtres de débit à 0 h ; à 120 h, il en gagne une à Saint-Pères et aucune à Convention. Les résultats détaillés par horizon, fenêtre et cible sont dans `data/processed/trajectory_analysis/`.

Sur les jours complets sans délai, la MAE du total quotidien de débit LightGBM / linéaire / convolutif est de 550 / 745 / 803 véhicules à Convention (19 jours) et 927 / 1 289 / 1 237 à Saint-Pères (18 jours). Le décalage absolu moyen de la pointe de débit est respectivement de 1,42 / 2,11 / 3,37 heures et de 1,11 / 2,39 / 2,22 heures. Ces statistiques sont descriptives et ne doivent pas être interprétées comme des observations indépendantes.

Pour le seul 1er novembre sans délai, la MAE du débit à Convention est de 86,40 / 195,60 / 205,31 et à Saint-Pères de 77,47 / 186,19 / 121,91 sur 24 heures par rue. Les résultats à 120 h et pour l'occupation figurent dans `data/processed/trajectory_holiday/holidays.csv`. Un unique jour férié ne permet pas d'en déduire une performance générale.

## Exécution

Après acquisition et préparation du panel et des données cibles :

```
python -m pip install -r requirements.txt
python main_trajectory.py --windows 4 --delays 0 120
python main_trajectory.py --event-start 2026-10-28 --delays 0 120 --output data/processed/trajectory_holiday
python -m unittest discover -s tests -q
```

Les calculs écrivent un checkpoint par fenêtre et délai, ainsi que `predictions.csv`, `training.csv`, `summary.csv`, `windows.csv`, `horizons.csv`, `holidays.csv`, `daily.csv`, `daily_summary.csv` et `window_diagnostics.csv`. Les poids ne sont pas sauvegardés. Les checkpoints n'enregistrent pas d'empreinte des données, du code ou des paramètres : utiliser un nouveau répertoire de sortie après toute modification. La présente exécution utilise PyTorch 2.15.0+cpu. Les sorties volumineuses sont ignorées par Git.
