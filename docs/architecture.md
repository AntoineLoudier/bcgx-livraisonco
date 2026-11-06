# Méthode — version de rendu du 6 novembre 2026

Le modèle et les résultats remis correspondent au code de la révision `c00b661`, après une nouvelle acquisition le 6 novembre. Le nettoyage du dépôt ne change pas les paramètres du modèle ni les prédictions. Les [empreintes et versions](../presentation/results/manifest.json) permettent d’identifier les entrées du calcul.

## 1. Périmètre et livrables

Les cibles sont le débit horaire `q` (véh/h) et le taux d’occupation `k` (%). L’occupation représente la fraction du temps pendant laquelle un véhicule est détecté au-dessus de la boucle ; ce n’est ni une vitesse ni une durée de trajet. Aucun seuil de fluidité universel n’est déduit de ces deux variables.

| Rue évaluée | Tronçon orienté | Identifiant acquis |
|---|---|---:|
| Convention | Lecourbe-Convention → Convention-Blomet | 5671 |
| Saint-Pères | Sts_Peres-Voltaire → Sts_Peres-Universite | 191 |

Les Champs-Élysées ne sont pas évalués. Le pipeline historique traite encore les trois rues et le validateur du brief attend 360 lignes. Le CSV final conserve ce format ; les scores et figures de la présentation portent exclusivement sur Convention et Saint-Pères. Il ne faut pas utiliser le global à trois rues des rapports techniques comme résultat du rendu.

## 2. Acquisition et prétraitement

`src/data_loader.py` télécharge le trafic via l’export CSV de l’API Open Data Paris, avec un filtre sur les nœuds amont et aval, puis la météo Open-Meteo (historique et prévisions). Les coordonnées météo sont celles du centre de Paris. La météo disponible au moment du calcul couvre les 120 heures du 7–11 novembre sans valeur manquante.

`src/preprocessing.py` convertit les horodatages trafic vers `Europe/Paris`, puis retire le fuseau pour construire une grille locale horaire. Les doublons, notamment au changement d’heure, sont agrégés. Les valeurs hors bornes (`q` hors [0, 8000], `k` hors [0, 100]) et les couples q=k=0 en journée sur une voie déclarée ouverte sont traités comme manquants. Cette dernière règle est une heuristique de détection de panne, pas une certitude physique.

Les indicateurs `is_imputed_q` et `is_imputed_k` sont définis avant remplacement des valeurs manquantes. L’imputation utilise successivement la dernière mesure sur trois heures, les valeurs des trois semaines précédentes, une moyenne cumulative par jour/heure, puis un secours par propagation/zéro au début de l’historique. Ces valeurs peuvent alimenter les historiques explicatifs ; les lignes dont q ou k est imputé sont exclues du jeu d’entraînement exporté.

Après prétraitement, Convention dispose de 8 953 heures réelles q/k et Saint-Pères de 8 912. Les dernières mesures des deux rues datent du 6 novembre 2026 à 00 h locale. Les données brutes et préparées ne sont pas versionnées ; leurs SHA-256 sont conservés dans le manifeste.

## 3. Variables et ancrage

`src/features.py` construit le calendrier, les interactions météo et les historiques ; `src/model.py` ajoute les variables croisées. La liste exacte des **49 entrées** est conservée dans le manifeste et dans la dernière annexe du support.

- Calendrier : heure, jour de semaine, mois, encodages sin/cos, week-end, fériés, vacances, ponts et régime du jour. `is_summer_holiday` est calculé mais exclu des entrées du modèle.
- Météo : température, précipitations, pluie, vent, écart de température sur 24 h et interactions avec les heures de pointe.
- Historiques q/k : décalages de 7, 14, 21 et 28 jours, moyenne pondérée, médiane sur quatre semaines, dispersion, profils par heure/régime et quantile 90.
- Interactions : rapport k/q à J−7, différences des lags entre lignes consécutives et vacances × pointe. Si des observations manquent entre deux lignes, cette différence n’est pas nécessairement celle de deux heures adjacentes.

Pour les deux rues évaluées, le lag J−7 est lissé avec les heures voisines (10 % / 80 % / 10 %). Une cascade de remplacement évite de reprendre un jour spécial pour prévoir un jour ordinaire. La moyenne pondérée des trois semaines (60 % / 25 % / 15 %) est mélangée à la médiane de quatre semaines : 10 % de médiane sur Convention et 18 % sur Saint-Pères.

Cet historique forme l’ancrage `smart_expected`. Les jours fériés, l’ancrage est multiplié par 0,80. Les ponts utilisent une intensité calendaire et un ratio historique borné entre 0,75 et 0,95. Ce sont des hypothèses de modèle, construites avec peu d’événements. Le profil du défilé de l’Armistice 2025 est réservé aux Champs-Élysées et ne s’applique pas aux deux rues évaluées.

## 4. Modèle final

Pour chaque rue et cible, deux modèles apprennent le résidu entre la cible et l’ancrage :

```text
prédiction = clip(ancrage + poids × (0,5 × LightGBM + 0,5 × XGBoost))
```

| Paramètre | Convention | Saint-Pères |
|---|---:|---:|
| Arbres par modèle | 400 | 350 |
| Profondeur LightGBM / XGBoost | 5 / 4 | 4 / 3 |
| Régularisation L2 | 4 | 5 |
| Poids résiduel q / k | 1,05 / 1,00 | 0,40 / 0,85 |

Les deux algorithmes utilisent un taux d’apprentissage de 0,025 et une graine de 42. Les résidus d’entraînement sont écrêtés aux quantiles 1 % et 99 %. La pondération croît de 0,85 à 1,20 selon l’ordre chronologique des lignes ; pour q, les heures 8–20 reçoivent un facteur 1,25, les autres un facteur 0,90. La borne haute de prédiction est estimée sur les cibles d’entraînement, avec un plafond de 100 pour k.

`main_train.py` sauvegarde les modèles dans `models/ensemble_models_q_k.joblib`. `main_predict.py` prédit les 120 heures du 7 au 11 novembre pour chaque rue et utilise le validateur de format de `src/evaluate.py`.

## 5. Validation et résultats

`get_temporal_cv_splits` réserve au moins les premières 35 % des observations pour l’historique initial, cherche des fenêtres de 120 heures contenant au moins 85 % de cibles réelles, puis en sélectionne quatre réparties dans le temps, par rue. L’entraînement utilise les dates strictement antérieures au début de la fenêtre correspondante.

Les dates peuvent différer entre rues. Les périodes exactes et effectifs sont disponibles dans [scores_par_fenetre.csv](../presentation/results/scores_par_fenetre.csv). `Start` et `End` sont les premières et dernières cibles **observées**, qui peuvent être en retrait des bornes nominales.

Les [prédictions de validation](../presentation/results/predictions_validation_2_rues.csv) comprennent 464 lignes Convention et 480 Saint-Pères. Le global est calculé sur les 944 erreurs concaténées, séparément pour q et k. La référence principale est `q_lag_168h` / `k_lag_168h`, donc **J−7 prétraité**. Une référence non lissée sur la série nettoyée/imputée est fournie en complément.

La comparaison ancienne/nouvelle version réentraîne le code `e6c9818` sur les mêmes données préparées et les mêmes clés de validation que `c00b661`. Le gain global est de 4,47 % sur q et 2,38 % sur k. Il ne compare pas des scores obtenus sur deux acquisitions différentes.

## 6. Interprétation et limites

- Les quatre fenêtres ont servi au développement et ne constituent pas un test indépendant de la sélection des paramètres.
- Les variables météo historiques sont observées. Les prévisions disponibles aujourd’hui pour novembre ne reconstituent pas les prévisions qui existaient lors des fenêtres passées.
- Les retards historiques de publication du trafic ne sont pas simulés. Les décalages des variables réduisent le risque de fuite, sans certifier à eux seuls le protocole opérationnel complet. La jointure météo utilise notamment des remplissages avant/arrière en cas de trous.
- Le filtrage des observations change la population évaluée ; les périodes de capteur manquant ne sont pas des cibles de validation. Les effectifs restent explicitement publiés.
- La fenêtre d’août sur Saint-Pères reste moins bonne que J−7 prétraité pour q et k. Le gain agrégé ne signifie pas un gain sur toutes les périodes.
- Les SHAP exportés par le pipeline concernent la composante LightGBM sur les dernières lignes d’entraînement. La figure du support applique le facteur 0,5 et le poids résiduel ; elle ne décrit ni XGBoost, ni l’ancrage, ni le clipping final.
- Le script optionnel `src/plots.py` est l’outil exploratoire d’origine. Il peut produire des références de repli lorsqu’un artefact manque ; ses figures et son global à trois rues ne constituent pas les résultats remis. Le support utilise exclusivement `presentation/refresh_evidence.py` et les preuves versionnées.
- Aucun lien empirique entre baisse de RMSE, temps de trajet et économies n’a été établi. La maquette est une illustration statique ; la mesure économique nécessite un pilote opérationnel.

## 7. Reproduction

Les commandes d’installation et d’exécution figurent dans le [README](../README.md). Les dépendances d’entraînement sont verrouillées. Les dépendances de génération du support sont optionnelles (`uv sync --locked --extra presentation`).

`presentation/refresh_evidence.py` vérifie la fraîcheur des artefacts, rejoue la comparaison historique, filtre les deux rues et conserve les scores, prévisions, métadonnées et figures. Il exige l’historique Git contenant `e6c9818`. `presentation/build_presentation.py` reconstruit ensuite le PDF et le PowerPoint à partir des seules preuves versionnées.

Les API étant évolutives, les commandes d’acquisition ne reproduisent pas exactement le jeu du 6 novembre lorsqu’elles sont lancées plus tard. Les preuves remises permettent de vérifier les scores sans accès à ces API. Les calendriers et dates de test du code doivent être adaptés avant une utilisation sur une autre période.
