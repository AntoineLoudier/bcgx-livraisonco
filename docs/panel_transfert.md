# Modèle global entraîné sur plusieurs tronçons

## Acquisition et contrôle

Le panel comprend 20 identifiants supplémentaires choisis parmi les plus couverts du 1er octobre 2025 au 30 avril 2026, avant les fenêtres évaluées. La liste, les noms des sommets et l'empreinte du fichier téléchargé sont dans `configs/panel_selection.json`. La sélection favorise les capteurs complets ; elle ne prétend pas être représentative de tout Paris. Les 20 identifiants conservent chacun un seul couple amont–aval dans les données téléchargées. Après nettoyage, le nombre d'heures imputées par cible et par capteur figure dans `data/processed/panel/audit.csv`.

Neuf capteurs incidents aux sommets des rues cibles ont été acquis séparément ; les identités sont dans `configs/panel_neighbors_selection.json` et l'audit dans `data/processed/panel/neighbors_audit.csv`. L'expérience garde uniquement les arcs orientés vers le sommet amont de la cible ou depuis son sommet aval, en excluant son trajet inverse exact. Une correspondance de sommets ne prouve pas que tous les mouvements entre rues sont autorisés : la topologie et les intersections doivent être confirmées avant usage opérationnel.

Les dates sont converties en heure locale de Paris ; l'heure répétée au changement d'heure est agrégée comme pour les cibles. Les lignes dont les noms de sommets ne correspondent pas à la sélection sont exclues et comptées. Les mêmes règles de nettoyage physique et d'imputation causale que les cibles sont appliquées. Un identifiant correspond à un tronçon orienté, pas forcément à un capteur physique unique : les métadonnées disponibles ne permettent pas de confirmer tous les changements matériels.

## Modèles et protocole

Quatre modèles `q` et quatre modèles `k` sont entraînés à chaque émission :

* `reference` : modèle multi-horizon existant, avec cinq exemples par heure cible.
* `targets_sampled` : trois cibles seules, valeurs normalisées et un horizon simulé par jour cible.
* `panel` : 23 arcs, avec cette même représentation normalisée.
* `panel_adapted` : panel global avec poids des rues cibles multiplié par quatre.

Le second modèle est indispensable pour distinguer l'effet du panel de celui de la normalisation et de l'échantillonnage. Le modèle de référence utilise davantage d'exemples par cible : une différence avec lui ne peut donc pas être attribuée au seul panel. Les autres méthodes emploient exactement le même échantillonnage. Pour chaque cible et chaque arc, l'échelle est la médiane absolue des mesures observées avant l'émission, avec un plancher de 10 pour `q` et 1 pour `k` et au moins 100 observations. Les séries sont pondérées pour contribuer également, puis le poids de chaque rue cible est multiplié par quatre dans le modèle adapté. Les paramètres sont fixes et non optimisés.

Les quatre fenêtres ordinaires de cinq jours débutent les 30 septembre, 8, 16 et 24 octobre 2026 ; elles ne se chevauchent pas et la dernière commence un samedi. Les modèles sont réentraînés à chaque émission et les scores comparent les mêmes seules observations réelles. Les mesures sont supposées disponibles jusqu'à l'émission : il s'agit d'un scénario sans délai, et non de la fraîcheur garantie à l'échéance finale. Une fenêtre distincte du 28 octobre au 1er novembre examine le férié. Les périodes ont déjà été explorées et ne constituent pas un test indépendant.

Une ablation séparée ajoute au modèle adapté des niveaux et couvertures observés des voisins sur 24 h et 168 h, normalisés par leur médiane causale sur 28 jours. Les mesures des voisins sont figées à l'émission, jamais à l'heure future de la cible. Les voisins ne sont pas ajoutés aux 20 séries auxiliaires du modèle global. Cette comparaison porte sur deux fenêtres et des pannes simulées de 180 jours ; elle ne suffit pas à établir l'utilité d'un graphe routier complet.

Les pannes artificielles masquent toutes les observations des 90, 180 ou 330 jours précédant l'émission, puis recalculent l'imputation et les variables. La vérité originale est réservée à l'évaluation. Une seule fenêtre par rue et durée est simulée ; même 330 jours ne vérifient pas directement le cas réel des Champs-Élysées. Aucun modèle final n'est remplacé.

## Résultats des fenêtres ordinaires

| Rue et cible | Heures | Référence | 3 cibles normalisées | Panel | Panel adapté |
|---|---:|---:|---:|---:|---:|
| Convention, q | 457 | 48,09 | 48,43 | 48,30 | 47,72 |
| Convention, k | 457 | 1,089 | 1,110 | 1,088 | 1,071 |
| Saint-Pères, q | 456 | 60,85 | 62,58 | 62,19 | 61,47 |
| Saint-Pères, k | 456 | 1,541 | 1,524 | 1,528 | 1,517 |

`q` est en véhicules/h et `k` en points. L'adaptation apporte une légère amélioration à Convention, mais pas pour le débit à Saint-Pères. Les Champs-Élysées n'ont aucune cible récente évaluable. Au 1er novembre seul, la MAE de débit référence / panel adapté est de 86,40 / 77,51 à Convention et 77,47 / 80,69 à Saint-Pères, sur 24 observations par rue. Un événement ne permet pas de conclure sur les fériés.

Sur les deux fenêtres de l'ablation des voisins, la MAE de débit du panel adapté passe de 50,87 à 51,31 à Convention (217 heures) et de 76,09 à 78,74 à Saint-Pères (216 heures). Avec une panne simulée de 180 jours sur une fenêtre, elle passe respectivement de 43,33 à 44,74 (97 heures) et de 91,69 à 94,92 (96 heures). Ces résultats ne montrent pas de gain dans cette configuration ; ils ne permettent pas de conclure pour d'autres voisinages ou pour les Champs-Élysées.

Sur une panne simulée de 330 jours, la MAE de débit référence / 3 cibles normalisées / panel / panel adapté est de 51,57 / 47,03 / 51,73 / 49,25 à Convention (97 heures) et de 75,32 / 76,01 / 71,58 / 71,10 à Saint-Pères (96 heures). Les six scénarios 90, 180 et 330 jours apparaissent séparément dans `summary.csv`. Il s'agit chaque fois de la même fenêtre d'octobre : cela ne constitue pas six validations indépendantes.

## Reproduction

```
python -m src.panel_data --selection configs/panel_selection.json
python -m src.panel_data --neighbors-only --selection configs/panel_neighbors_selection.json
python -m src.panel_preprocessing
python main_panel_transfer.py --windows 4 --outages
python main_panel_transfer.py --event-start 2026-10-28 --output data/processed/panel_transfer_holiday
python main_panel_neighbors.py --windows 2 --outages --output data/processed/panel_neighbors_directed
python -m unittest discover -s tests -q
```

Le service peut corriger ses archives ; comparer les empreintes des fichiers téléchargés aux manifestes. Les sorties de données, modèles et prédictions sont ignorées par Git. Les scripts reprennent les checkpoints présents ; changer de répertoire avant toute modification des données, du code ou des paramètres car les checkpoints ne portent pas leur empreinte. Les résultats détaillés sont dans `data/processed/panel_transfer/`, `panel_transfer_holiday/` et `panel_neighbors_directed/`.
