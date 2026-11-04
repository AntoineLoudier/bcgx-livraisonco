# Contexte récent, disponibilité et jours d'émission

`python main_recent_context.py` compare le modèle multi-horizon de référence, le même modèle enrichi de contexte et cinq modèles directs enrichis, un par jour de prévision. Les configurations sont fixées avant cette comparaison ; aucune variante n'est promue automatiquement. Sorties : `data/processed/recent_context_analysis/`.

Le protocole utilise huit fenêtres non chevauchantes de cinq jours, espacées de huit jours et couvrant les sept jours de début, dont deux samedis. La cible officielle samedi–mercredi est donc représentée. Les scores concernent les mêmes observations réelles pour chaque méthode ; `windows.csv`, `horizons.csv`, `weekdays.csv`, `holidays.csv` et `window_diagnostics.csv` détaillent les résultats. `bootstrap.csv` donne des intervalles exploratoires à 95 % sur l'écart de MAE candidat moins référence, par rééchantillonnage apparié de fenêtres. Avec huit fenêtres, ces intervalles sont instables et ne constituent pas un test indépendant.

Les fenêtres ordinaires peuvent ne pas contenir de jour férié. Pour isoler l'événement du 1er novembre sans mélanger des fenêtres chevauchantes dans le bootstrap :

```
python main_recent_context.py --event-start 2026-10-28 --output data/processed/recent_context_holiday_analysis
```

Ces cinq jours incluent le 1er novembre, mais un seul événement ne permet pas de conclure sur les fériés. Pour ajouter d'autres événements, fournir leurs dates de début après `--event-start` et vérifier explicitement les chevauchements avant d'interpréter les intervalles bootstrap.

## Méthode et disponibilité

Émission à 23 h avant les cinq jours de prévision. Pour chaque scénario de délai de 0, 24, 72 ou 120 heures, l'instant de dernière donnée accessible est l'émission moins ce délai. Tous les retards et profils de la référence, les nouveaux résumés et les étiquettes d'entraînement respectent ce gel. Le calendrier futur reste connu. Les exemples d'entraînement simulent leur propre émission et le même délai ; l'horizon reste mesuré depuis l'émission, pas depuis la dernière mesure. L'historique fourni au modèle est limité aux étiquettes disponibles à la date d'entraînement.

Pour `q` et `k`, le contexte exploite uniquement les mesures réelles : moyenne, écart-type, couverture et anomalie moyenne sur les 6, 24, 72 et 168 dernières heures, ancienneté de la dernière mesure, et profils horaires des trois derniers jours complets. L'anomalie d'une mesure est sa différence avec la moyenne de ses prédécesseurs de même jour de semaine et heure, construite strictement avant cette mesure. Les valeurs non disponibles restent manquantes et leur couverture est explicitée. Le contexte des deux indicateurs est fourni à chaque modèle. Les prévisions sont bornées aux domaines physiques ; les variables calendaires et corrections manuelles du modèle initial sont conservées.

Les coupures sont en heures locales naïves ; le jeu prétraité agrège l'heure répétée du changement d'heure. Les latences sont des scénarios de simulation, pas une preuve de la fraîcheur réelle de la source. Les données des Champs-Élysées sont très anciennes : l'évaluation récente n'est pas possible. La période a déjà servi à d'autres expériences, aucun résultat ne constitue une confirmation indépendante. Un objectif de décision de 5 % de réduction de MAE par rapport à la référence, avec examen des pires fenêtres et fériés, est indicatif et n'est pas une promesse.

Tests : `python -m unittest tests.test_recent_context` ; ils vérifient la perturbation des mesures postérieures à la disponibilité, toutes les variables utilisées par référence et variantes, l'équivalence des cinq horizons entre entraînement et inférence pour les quatre délais, l'exclusion des valeurs imputées et la répartition des fenêtres.

Le calcul des ratios historiques des événements du 10 et du 11 novembre est fixé à cinq jours avant l'événement, plus le délai simulé. Auparavant, sous les nouveaux scénarios de délai, son profil pouvait varier selon l'horizon de l'exemple d'entraînement. Pour les pipelines existants sans délai, les trois retards hebdomadaires étaient déjà disponibles aux cinq horizons : les ratios et conclusions précédentes ne changent donc pas. Les tests de compatibilité couvrent le scénario sans délai et l'équivalence des horizons avec délai, y compris les événements de novembre.

Les calculs écrivent un fichier par délai et fenêtre dans `checkpoints/` ; relancer la même commande reprend les fenêtres déjà calculées. Supprimer ou changer de répertoire de sortie lorsque les données, le code ou la configuration ont changé : les checkpoints ne portent pas d'empreinte de ces entrées. Pour analyser les délais séparément, `--delays 0`, `--delays 24`, etc., avec un répertoire distinct.

## Première analyse du 1er novembre 2026

Sur les 24 mesures horaires réelles du jour férié et avec un délai de 120 h, la MAE de débit à Convention est de 96,36 pour la référence, 71,50 pour le contexte et 72,51 pour les modèles directs. À Saint-Pères, les chiffres sont respectivement 84,04, 91,67 et 89,71 véhicules/h. Au délai de 72 h, le contexte dégrade Convention de 94,18 à 97,07 mais améliore Saint-Pères de 80,15 à 68,10. Il s'agit d'un seul événement historique, à ne pas généraliser. Les autres délais, l'occupation et les jours ordinaires de cette fenêtre sont dans `data/processed/recent_context_holiday_analysis/holidays.csv`.

## Résultats complets sur les fenêtres ordinaires

Huit fenêtres entre le 29 août et le 28 octobre 2026 ont été évaluées, sans chevauchement ; les 934 heures réelles disponibles par rue et cible sont identiques pour toutes les variantes et les quatre délais. Il n'y a pas de cibles récentes évaluables aux Champs-Élysées. Pour chaque cellule, les chiffres sont les MAE référence / contexte / direct ; unité véhicules/h pour `q`, points de pourcentage pour `k`.

| Délai | Convention `q` | Saint-Pères `q` | Convention `k` | Saint-Pères `k` |
|---|---|---|---|---|
| 0 h | 45,45 / 45,02 / 44,92 | 72,43 / 71,88 / 71,30 | 0,98 / 0,95 / 0,94 | 1,69 / 1,66 / 1,66 |
| 24 h | 45,82 / 45,74 / 45,66 | 73,01 / 73,50 / 74,22 | 0,98 / 0,96 / 0,96 | 1,69 / 1,72 / 1,69 |
| 72 h | 46,60 / 47,03 / 46,00 | 81,76 / 88,47 / 85,82 | 0,96 / 0,95 / 0,96 | 1,77 / 1,79 / 1,76 |
| 120 h | 48,38 / 48,68 / 47,88 | 88,26 / 91,71 / 85,51 | 0,98 / 0,98 / 0,97 | 1,81 / 1,81 / 1,73 |

À 120 h, le modèle direct réduit la MAE de `q` à Saint-Pères de 2,75 véhicules/h, soit 3,1 %, et gagne 7 fenêtres sur 8 ; sa pire fenêtre passe cependant de 121,16 à 124,33. Le contexte gagne seulement 2 fenêtres et sa pire MAE est 135,40. À 72 h, la MAE de `q` de Saint-Pères se dégrade pour les deux variantes et leur pire fenêtre passe de 103,77 à 124,36 et 120,07 respectivement. Le tableau complet des fenêtres gagnées et pires MAE figure dans `window_diagnostics.csv`.

Les deux fenêtres débutant un samedi totalisent 215 heures réelles par rue. À 120 h, pour `q`, Convention obtient 51,69 / 51,73 / 50,26 et Saint-Pères 96,67 / 102,15 / 104,45. Il s'agit de deux fenêtres seulement. Les scores par horizon sont dans `horizons.csv`.

À 120 h pour le modèle direct à Saint-Pères, l'intervalle bootstrap exploratoire à 95 % de l'écart de MAE (`direct − référence`) est [−8,17 ; +4,45] véhicules/h ; il inclut donc une dégradation comme une amélioration. Les intervalles bootstrap de toutes les comparaisons `q` incluent zéro. Avec huit blocs, il ne faut pas les interpréter comme une garantie de couverture.

Le contexte n'atteint pas un gain régulier de 5 % sur les rues et délais. Le modèle direct présente des signaux à étudier, mais ne doit pas remplacer la référence sur cette seule analyse, en particulier vu les dégradations à 72 h et sur les samedis à Saint-Pères. Toutes ces fenêtres ont déjà servi à d'autres travaux ; leur diversité de jours ne les rend pas indépendantes.

Pour reproduire en parallèle ou séparément, par exemple `python main_recent_context.py --delays 120 --output data/processed/recent_context_120h`. Les quatre calculs de cette étude ont été lancés dans des répertoires distincts ; leurs checkpoints peuvent être combinés pour produire le répertoire `recent_context_analysis`. `--summarize-only` résume les checkpoints d'un répertoire même après interruption ; il ne valide pas leur provenance.
