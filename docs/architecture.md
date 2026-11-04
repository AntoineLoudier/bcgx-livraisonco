# Pipeline de prévision du trafic

Le projet prépare des prévisions horaires pour trois arcs parisiens, du 7 au 11 novembre 2026. Les cibles sont le débit `q` et le taux d'occupation `k`.

## Acquisition et préparation

`src/data_loader.py` télécharge les mesures de trafic et la météo. `src/preprocessing.py` aligne les dates sur l'heure locale, filtre les valeurs aberrantes et construit une grille horaire. Les valeurs manquantes sont remplies avec des observations précédentes : retards hebdomadaires puis médiane historique par arc, jour et heure. Une valeur peut rester manquante si aucun historique n'est disponible. Les indicateurs d'imputation permettent de ne pas utiliser ces valeurs comme cibles d'entraînement.

L'imputation doit être recalculée depuis les données brutes après une modification du code ; les anciens CSV préparés peuvent contenir des informations futures.

## Variables et disponibilité

`src/features.py` simule, pour chaque date historique, une prévision émise cinq jours plus tôt. Pour la fenêtre du 7 au 11 novembre, la date de disponibilité est plafonnée au 6 novembre à 23 h. Les mesures plus récentes ne sont pas utilisées. Les profils des Champs-Élysées sont construits avec les dernières observations réelles disponibles à cette date, et les profils des 10 et 11 novembre 2025 ne sont utilisés qu'une fois leurs mesures disponibles.

Les dates de prévision historiques utilisent un horizon fixe de 120 heures. Pour évaluer une émission unique à un autre horizon ou sur une autre fenêtre, il faut adapter le calcul de la date de disponibilité et reconstruire les variables à chaque fenêtre. Les heures sont représentées en temps local sans fuseau : le passage à l'heure d'hiver est agrégé et cette convention doit être conservée lors de la validation.

Les exports ne contiennent ni météo, ni état de fermeture observé à l'heure cible, ni indicateurs d'imputation de la cible : les archives météo ne sont pas des prévisions historiques et l'état réel futur n'est pas connu. Les variables calendaires et les indicateurs d'événement prévus restent disponibles. Pour réintroduire la météo ou les fermetures, il faut des prévisions ou informations publiées avant chaque date d'émission. Les dates et effets des événements restent des hypothèses à valider.

Le fichier d'entraînement ne conserve que les lignes où les deux cibles sont observées. Les colonnes `q` et `k` sont les étiquettes : **les exclure toutes les deux des entrées du modèle**, même lors de l'entraînement d'un seul indicateur. Le fichier de prévision ne contient pas ces colonnes. Les retards peuvent être manquants lorsque l'historique disponible est insuffisant ; il faut utiliser un modèle ou une stratégie d'imputation ajustée exclusivement sur l'entraînement.

## Exécution

Depuis la racine du dépôt :

```bash
python src/data_loader.py
python src/preprocessing.py
python src/features.py
python src/evaluate.py
python -m unittest discover -s tests -v
```

## Validation chronologique

`src/evaluate.py` évalue par défaut les quatre dernières fenêtres complètes de cinq jours, espacées de sept jours. Chaque fenêtre simule une émission unique à 23 h la veille du premier jour. L'historique est tronqué à cette date, les variables sont recalculées, puis les cibles observées sont jointes uniquement pour le calcul des scores. La MAE et la RMSE sont calculées séparément par arc, cible et fenêtre, avec le nombre d'observations et de prévisions manquantes. Une cellule sans observations n'a pas de score.

Trois références sont évaluées : semaine précédente, moyenne pondérée des trois dernières semaines et profil calendaire. Pour les Champs-Élysées après le 15 décembre 2025, ces retards représentent les dernières observations réelles disponibles, éventuellement anciennes. Ce ne sont pas des modèles entraînés. Les valeurs d'entrée manquantes restent manquantes et les comparaisons entre méthodes doivent tenir compte de leur couverture respective.

Les résultats sont enregistrés dans `data/processed/validation/` : `metrics.csv`, `predictions.csv` et `weather_availability.csv`. Les options `--windows`, `--step-days`, `--traffic`, `--weather` et `--output` permettent de modifier l'exécution. La validation s'appuie sur `traffic_clean.csv`, qui doit avoir été reconstruit avec l'imputation causale actuelle ; les anciens fichiers préparés ne conviennent pas.

Le fichier météo actuel ne contient pas de date d'émission : on ne peut donc pas vérifier rétrospectivement qu'il s'agissait de prévisions disponibles au moment de chaque émission. La météo n'est utilisée par aucune des références. Pour auditer un fichier de prévisions archivées, fournir `--weather` avec les colonnes `issued_at` et `valid_at`, horodatées avec fuseau ; le rapport compte les heures locales pour lesquelles au moins un enregistrement a été émis au plus tard à la date de prévision. Ce compte vérifie les horodatages des enregistrements, pas la présence ou la qualité de chaque variable météo. La grille de trafic agrège l'heure répétée lors du changement d'heure.

## Premier modèle

`python main_train.py` entraîne deux modèles LightGBM globaux, un pour `q` et un pour `k`, avec une perte absolue. Chaque fenêtre est évaluée avec des modèles réentraînés sur les seules mesures observées jusqu'à sa date d'émission ; leurs variables utilisent des retards et des données calendaires. Les prévisions météo ne sont pas utilisées. Les prédictions sont bornées à zéro pour `q` et entre 0 et 100 pour `k`.

Les résultats sont dans `data/processed/model_analysis/` : scores par fenêtre (`metrics.csv`), comparaisons sur les mêmes observations pour chaque paire modèle-référence (`comparison.csv`), erreurs et biais par heure (`errors_by_hour.csv`), importance des variables par gain d'entraînement (`feature_importance.csv`), prédictions et nombres de lignes d'entraînement. Une importance par gain décrit l'usage des variables par les arbres et ne démontre pas leur causalité ni leur importance hors échantillon. Comparer les scores des fenêtres et leurs couvertures avant de conclure.

Après la validation, deux modèles sont entraînés à nouveau jusqu'à la dernière date disponible, plafonnée au gel du 6 novembre 2026. Ils sont enregistrés dans `models/lightgbm_q.txt` et `models/lightgbm_k.txt` ; `lightgbm_metadata.json` contient la date de coupure et les variables. Relancer l'entraînement après actualisation des données. Les modèles enregistrés ne constituent pas encore une chaîne d'inférence opérationnelle.

`python main_stability.py` étend l'analyse à 52 fenêtres hebdomadaires et à des fenêtres autour des jours fériés. Les résultats, les effectifs et les limites de cette étude sont décrits dans [l'étude de stabilité](etude_stabilite.md).

`python main_compare_streets.py` réentraîne et compare des modèles globaux et par rue sur ces fenêtres, puis enregistre les six modèles locaux dans `models/per_street/`. Les résultats figurent dans [la comparaison des modèles par rue](modeles_par_rue.md).

## Entraînement multi-horizon

`python main_multi_horizon.py` entraîne une variante pour une émission quotidienne à 23 h suivie des cinq jours calendaires suivants. Chaque mesure historique devient jusqu'à cinq exemples d'entraînement, correspondant aux cinq émissions quotidiennes possibles ; `horizon_hours` indique la distance, de 1 à 120 heures locales, entre l'émission et la cible. Les retards et profils sont recalculés avec les seules données disponibles pour chaque émission. Les cibles non observées sont exclues. Les heures répétées ou manquantes au changement d'heure suivent la convention de la grille locale agrégée ; il ne s'agit pas d'une durée UTC exacte.

Les deux modèles sont enregistrés dans `models/multi_horizon/` avec leurs variables et leur date de coupure ; les modèles à horizon fixe précédents sont conservés. Après `python src/features.py`, `features_test_nov7_11.csv` contient les 360 lignes et leur `horizon_hours`, avec une émission au 6 novembre 23 h. Ces modèles restent expérimentaux : ils ne sont pas encore raccordés à `main_predict.py`.

La commande compare par défaut les variantes sur huit fenêtres historiques de cinq jours, en réentraînant les deux modèles dans chaque fenêtre. Les fichiers `comparison.csv`, `by_horizon.csv`, `by_window.csv` et `predictions.csv` se trouvent dans `data/processed/multi_horizon_analysis/`. Les métriques ne portent que sur les cibles observées et les mêmes heures pour les deux variantes. Les cinq exemples construits à partir d'une même cible ne sont pas indépendants. Les fenêtres récentes ne permettent pas d'évaluer les Champs-Élysées et ne constituent pas un jeu de test final indépendant de toutes les analyses précédentes.

`python main_recency.py` compare, avec ce modèle multi-horizon, les cibles des trois et six derniers mois à tout l'historique et à une pondération de demi-vie 180 jours. L'[étude d'ancienneté](pondération_recente.md) détaille les résultats et les limites de couverture des rues et des fériés.

`python main_holidays.py` compare quatre variantes croisant un calendrier de fériés enrichi et l'ablation des coefficients et du régime manuels des ponts. La [comparaison des fériés et ponts](feries_et_ponts.md) décrit les variables, les mesures et leurs limites. Ces options restent expérimentales : les modèles existants ne sont pas remplacés.

`python main_dynamics.py` compare la référence à une variante ajoutant tendances hebdomadaires, variabilité, nombre de mesures réelles et âge de la dernière mesure. Le calcul respecte la date d'émission et les indicateurs d'imputation ; les définitions et résultats figurent dans [l'étude de dynamique récente](dynamique_recente.md). Les modèles expérimentaux sont séparés de la référence.

`python main_blend.py` compare le modèle multi-horizon à un mélange avec la moyenne des semaines précédentes. Les poids sont sélectionnés séparément pour chaque cible sur des fenêtres antérieures terminées ; la méthode et les limites sont décrites dans [l'étude du mélange](melange_previsions.md).

Les données téléchargées et les CSV générés ne sont pas versionnés. Les quatre fenêtres récentes ne contiennent pas de valeurs observées aux Champs-Élysées ; leurs résultats ne permettent donc pas d'évaluer ce tronçon, ni de valider spécifiquement le comportement des jours fériés ou des ponts de novembre.
