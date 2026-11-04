# Dynamique récente et qualité des mesures

`python main_dynamics.py` compare le modèle global multi-horizon de référence à une variante qui ajoute cinq variables par cible. Les deux modèles ont les mêmes hyperparamètres, les mêmes exemples d'entraînement et les mêmes variables existantes. La comparaison porte sur les mêmes cibles observées, avec réentraînement avant chaque fenêtre. Elle n'intègre pas les variables expérimentales de fériés enrichies ni les variantes de pondération.

## Variables

Pour chaque arc, cible et heure prédite, les observations horaires aux mêmes jour et heure à J-7, J-14 et J-21 sont retenues uniquement si elles étaient effectivement mesurées et disponibles à l'émission :

- `recent_vs_usual` : écart relatif entre J-7 et la moyenne des observations disponibles à J-14 et J-21. Le dénominateur est borné à 10 pour `q` et 1 pour `k` ; le résultat est borné entre -5 et 5.
- `weekly_trend` : différence entre J-7 et J-21 divisée par deux, exprimée par semaine.
- `real_weekly_std` : écart-type des valeurs réellement observées parmi ces trois semaines, défini s'il y en a au moins deux.
- `real_weekly_count` : nombre d'observations réelles disponibles parmi ces trois semaines, de 0 à 3.
- `last_real_age_hours` : temps écoulé entre la dernière mesure réelle de la cible sur cet arc et la date d'émission ; il reste manquant s'il n'existe aucune mesure connue.

Les trous ne sont pas interprétés comme des observations réelles. Les variables restent manquantes quand l'historique est insuffisant ; LightGBM gère ces valeurs manquantes. Le calcul utilise les heures locales naïves de la grille existante : le changement d'heure est agrégé, et l'âge en heures locales peut différer de la durée UTC réelle.

## Fenêtres récentes

Huit fenêtres de cinq jours débutent du 9 septembre au 28 octobre 2026. Les MAE portent sur les mêmes 934 heures observées par rue et cible ; `q` est exprimé en véhicules/h et `k` en points.

| Rue et cible | Référence | Avec dynamique |
|---|---:|---:|
| Convention, q | 48,42 | 48,33 |
| Convention, k | 1,134 | 1,126 |
| Saint-Pères, q | 67,65 | 68,84 |
| Saint-Pères, k | 1,637 | 1,632 |

Les écarts sont faibles sauf pour le débit de Saint-Pères, où la variante est moins bonne sur l'ensemble et ne gagne qu'une des huit fenêtres. Aucun score récent n'est calculable aux Champs-Élysées.

## Jours fériés

Sur les douze fériés observés de novembre 2025 à novembre 2026, les MAE du débit sont de 74,52 contre 77,50 véhicules/h à Convention (275 heures) et 84,47 contre 84,87 à Saint-Pères (288 heures), respectivement pour la référence et la variante. Ces moyennes portent sur peu d'événements et peuvent masquer des différences entre jours. Aux Champs-Élysées, seuls deux fériés de 2025 sont observés ; le 11 novembre 2025, l'erreur de débit passe de 274,16 à 259,13 véhicules/h sur 24 heures, ce qui reste élevé et ne valide pas 2026.

La variante ne justifie pas le remplacement de la référence sur ces résultats. Les huit fenêtres récentes ont déjà été explorées, elles ne constituent pas un test final indépendant ; les fériés sont rares et certaines fenêtres se recoupent. Une importance par gain d'entraînement ne démontre ni effet causal ni performance hors échantillon. Tester ces variables sur d'autres périodes et contrôler l'effet des trous de capteur avant toute sélection.

Le fichier de prévision est régénéré avec `python src/features.py`. Avec les données actuellement téléchargées, l'âge à l'émission prévue du 6 novembre 2026 est de 8 460 heures locales pour le débit aux Champs-Élysées, soit environ 352,5 jours ; les trois semaines utilisées pour l'indicateur de disponibilité ne contiennent aucune mesure réelle de ce capteur. Les autres arcs ont 119 heures d'ancienneté car le fichier local s'arrête au 2 novembre. Ces âges changeront si les données sont actualisées. Ils ne réparent pas l'absence d'observations récentes et le modèle n'a pas pu être évalué sur des mesures récentes de cet arc.

Les résultats détaillés sont dans `data/processed/dynamics_analysis/`, et les deux modèles expérimentaux dans `models/recent_dynamics/`. Ils ne remplacent pas la référence et ne sont pas raccordés à `main_predict.py`.
