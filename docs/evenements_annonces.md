# Restrictions annoncées et scénarios d'événements

## Périmètre et provenance

Le catalogue manuel [evenements_cibles.json](../configs/evenements_cibles.json) est **incomplet**. Un zéro indique qu'aucune annonce vérifiée du catalogue ne s'applique ; il ne démontre pas l'absence de perturbation. Pour chaque enregistrement figurent la source, le tronçon orienté concerné, le type, l'intervalle, la portée `direct` ou `adjacent`, la date de disponibilité et la qualité de cette date. Les horodatages sources comprennent leur décalage UTC ; les calculs de trafic utilisent l'heure locale de Paris et la grille horaire existante.

* **11 novembre 2025, Champs-Élysées** : l'[arrêté 2025-01460](https://www.prefectures-regions.gouv.fr/ile-de-france/irecontenu/telechargement/132389/971747/file/recueil-75-2025-678-recueil-des-actes-administratifs-special%20du%2004.11.2025.pdf), dans le recueil du 4 novembre, interdit la circulation de véhicules à moteur de 7 h à 14 h dans un périmètre incluant la portion étudiée. Des exceptions et adaptations sont prévues. L'heure de publication est inconnue ; la disponibilité est fixée prudemment au 5 novembre à 00 h. Le modèle utilise cette annonce vérifiée.
* **12 avril 2026, Champs-Élysées** : l'[arrêté du marathon](https://cdn.paris.fr/paris/2026/04/07/arrete-marathon-2026-joRm.pdf), article 9, interdit la circulation de 00 h 01 à 15 h sur la section contenant le tronçon. La date dans l'URL suggère une mise à disposition le 7 avril, sans prouver sa première publication : l'avis est exclu des entrées automatiquement utilisables. Aucune mesure réelle récente n'est disponible sur ce tronçon pendant l'événement.
* **17 mai 2026, Convention** : le [document prévisionnel municipal](https://cdn.paris.fr/paris/2026/05/07/km-10-tKcl.pdf) annonce des restrictions dans le secteur de Convention pour la course de 6 h 45 à 13 h 30. La portée est `adjacent` : il n'est pas démontré que tout le tronçon Convention-Blomet soit fermé. La date dans l'URL ne prouve pas la première publication ; l'événement est exclu des entrées automatiquement utilisables. La [page municipale](https://mairie15.paris.fr/pages/21e-edition-des-10km-paris-15-dimanche-17-mai-2026-34447) affiche une mise à jour le 7 mai mais lie un arrêté daté du 13 mai : son contenu actuel ne constitue pas une archive de son état au 7 mai.

Les dates `known_at` de précision journalière sont placées au début du lendemain. Les avis `inferred` restent exclus même après cette date tant qu'une preuve d'accessibilité antérieure n'est pas établie. Les doublons d'identifiant sont refusés : les modifications d'un avis demandent un système de versions et ne doivent pas écraser rétroactivement une ancienne annonce. Le jeu [Travaux perturbants la circulation](https://opendata.paris.fr/explore/dataset/chantiers-perturbants/) n'est pas exhaustif et les versions historiques publiées au moment de chaque émission n'ont pas été établies ; il n'est donc pas utilisé pour la validation.

## Méthode

Le modèle de référence multi-horizon est comparé à un modèle identique recevant deux variables : proportion de l'heure affectée par un avis **vérifié et connu à sa propre émission**, d'une part directement et d'autre part sur une voie adjacente. La présence d'un avis sur une voie voisine ne devient jamais une fermeture directe du tronçon. Les valeurs horaires partiellement concernées sont pondérées par la durée de recouvrement, avec des intervalles `[début, fin)`. Les modèles sont réentraînés avant chaque fenêtre et les cibles imputées sont exclues.

Une correction multiplicative distincte du modèle est également prévue. Elle utilise une médiane pondérée des rapports `observation / prévision` de fenêtres hors entraînement, avec au moins deux événements terminés, 20 mesures réelles, la même rue et la même portée. Elle est limitée à [0, 2]. Si ces conditions ou la disponibilité vérifiée manquent, elle vaut 1. Dans cette étude, **toutes les corrections apprises restent à 1** ; il n'y a pas assez d'événements exploitables pour calibrer leur effet.

Les colonnes `scenario_lower` et `scenario_upper` sont des hypothèses de sensibilité prédéfinies pour le débit : -50 % pour une restriction directe ; -20 % et +20 % pour une perturbation adjacente. Elles sont appliquées à la fraction de l'heure concernée, même si la disponibilité n'est pas vérifiée, et sont donc **conditionnelles**. Elles ne représentent pas une probabilité ni des coefficients appris. L'occupation reste inchangée dans ces scénarios. L'existence de dérogations et les mesures de trafic non nulles empêchent de supposer systématiquement un débit nul.

## Résultats historiques

| Événement et heures concernées | Mesures réelles de débit | MAE référence | Modèle avec avis | Scénario bas conditionnel |
|---|---:|---:|---:|---:|
| 11 novembre 2025, Champs-Élysées, 7 h–14 h | 7 | 645,10 | 645,10 | 301,48 |
| 17 mai 2026, Convention, secteur voisin | 8 | 38,69 | 38,69 | 62,59 |
| 12 avril 2026, Champs-Élysées | 0 | — | — | — |

MAE en véhicules/h. Le modèle avec avis est identique à la référence sur ces mesures ; le scénario -50 % réduit l'erreur du seul 11 novembre 2025, sans prouver que ce coefficient se reproduira. La baisse de 20 % aggrave l'erreur lors de la course de mai ; la hausse de 20 % donne une MAE de 60,13. Ces scénarios, décidés avant le calcul, ne sont pas une sélection de modèle. Le groupe de mai inclut des heures partiellement concernées et n'est pas une fermeture confirmée du tronçon. Les cibles imputées et le marathon non observé ne sont pas comptés.

## Scénario 2026

`forecast_scenarios_2026.csv` décrit une émission **planifiée** le 6 novembre à 23 h, avec les données du fichier disponibles au plus tard le 4 novembre à 00 h. Les dernières mesures réelles par rue et cible sont exportées ; dans le fichier actuel, le débit des Champs-Élysées date du 19 novembre 2025 et celui des autres rues du 2 novembre 2026. Les valeurs futures sont des grilles sans vérité observée, et le modèle n'intègre aucune donnée postérieure à l'instant de connaissance fixé.

À la date de recherche du 4 novembre, aucun arrêté 2026 spécifique au tronçon et à ces horaires n'a été vérifié dans les sources consultées. La colonne `q_hypothesis_minus50` est uniquement une hypothèse de baisse de 50 % entre 7 h et 14 h le 11 novembre, par analogie avec 2025 ; elle ne constitue ni une fermeture annoncée ni une prévision calibrée. Les variables `known_event_direct` et `known_event_adjacent` restent à zéro en l'absence d'avis vérifié dans le catalogue. L'indicateur Armistice et la correction historique existants restent toutefois présents dans le modèle de référence. Il faut actualiser les données et rechercher les annonces officielles avant toute émission réelle.

## Reproduction et limites

```
python main_event_restrictions.py --as-of '2026-11-04 00:00'
python -m unittest discover -s tests -q
```

Les résultats sont écrits dans `data/processed/announced_events_analysis/` : prédictions, scores par événement, calibration et scénarios futurs. Les sources ne sont pas collectées automatiquement et aucun modèle final n'est remplacé. Le catalogue ne comprend que trois événements identifiés manuellement ; le problème principal reste le manque d'événements historiques dont les horaires, la localisation, les versions successives et la date exacte de publication soient vérifiables, ainsi que l'absence de mesures récentes aux Champs-Élysées.
