# Diagnostic et nouvelles expériences

Les commandes `python main_error_diagnostics.py` et `python main_new_experiments.py` produisent respectivement `data/processed/error_diagnostics/` et `data/processed/new_experiments/`. Elles nécessitent les données préparées ainsi que `data/processed/blend_analysis/predictions.csv` produit par `python main_blend.py`. Les sorties sont ignorées par Git. Les modèles expérimentaux ne remplacent pas la référence et ne sont pas branchés à l'inférence finale.

## 1. Diagnostic des erreurs

Le diagnostic utilise les prévisions multi-horizon historiques de 52 fenêtres, y compris les quatre premières réservées à l'initialisation du mélange. Les biais sont prévision moins observation. Seules les cibles réelles sont évaluées ; les totaux et heures de pointe ne sont calculés que pour les jours de 24 heures observées. Les heures de pointe correspondent aux heures de maxima journaliers ; l'écart est signé et les égalités sont résolues par la première heure rencontrée. Les classements de pires jours indiquent le nombre d'heures observées.

Pour le débit, le biais moyen est de -5,71 véhicules/h à Convention (5 880 observations) et -8,53 à Saint-Pères (6 150). La MAE de l'occupation à Convention est de 1,78 point aux heures de pointe retenues (7–10 h et 17–20 h), contre 0,78 aux autres heures. Les heures de pointe constituent une convention d'analyse. Les chiffres des Champs-Élysées portent uniquement sur 2025.

## 2. Pannes et mesures imputées

Pour chaque panne simulée de 1, 3, 7 ou 21 jours précédant l'émission, toutes les anciennes valeurs imputées sont effacées puis recalculées causalement après masquage de l'arc. La vérité d'évaluation est gardée séparément. On réentraîne le modèle sur l'historique rendu disponible par la panne. La variante `real_lags` ne conserve aux retards hebdomadaires que les mesures effectives, avec un indicateur par retard ; elle ne reprend pas la substitution particulière des Champs-Élysées. La référence conserve l'imputation et la substitution d'origine. Les résumés portent sur deux fenêtres récentes et les deux autres arcs, séparément. Ces pannes courtes ne reproduisent pas la situation des Champs-Élysées.

Pour une panne de 21 jours, la MAE du débit passe de 64,40 à 68,55 véhicules/h à Convention entre la référence et `real_lags`, et de 98,89 à 83,06 à Saint-Pères. Sans panne simulée, la référence atteint respectivement 53,70 et 64,97 sur les mêmes heures. Ces chiffres portent sur 214 et 215 heures observées : ils ne suffisent pas à choisir une stratégie générale.

## 3. Correction d'un profil

La variante `residual` apprend la différence entre la cible et `regime_hist_mean`, profil cumulatif par arc, régime de jour et heure, calculé à chaque émission historique. Si le profil est absent, sa valeur de repli est zéro ; la prédiction est reconstituée en ajoutant le profil et bornée physiquement. La comparaison porte sur les mêmes heures observées et huit fenêtres du 9 septembre au 1er novembre 2026.

Pour le débit, la MAE passe de 48,42 à 47,32 véhicules/h à Convention, mais de 67,65 à 72,32 à Saint-Pères. Le modèle résiduel gagne respectivement 6 et 3 fenêtres sur 8 : ce n'est pas un gain généralisé.

## 4. Historiques croisés

La variante `cross` ajoute les retards et les profils de l'autre indicateur au modèle. Tous sont calculés avec la même date de disponibilité que les variables existantes ; aucune valeur de la cible future n'est utilisée. La MAE de débit passe de 48,42 à 48,11 à Convention et de 67,65 à 68,66 à Saint-Pères. Pour l'occupation, elle passe de 1,134 à 1,131 et de 1,637 à 1,624 respectivement. Les écarts sont limités et demandent confirmation.

Le fichier brut existant contient uniquement les trois tronçons cibles : aucun capteur voisin ni sa topologie n'est disponible pour l'expérience spatiale. Il faudrait acquérir et vérifier les tronçons et leurs sens, sans sélectionner les voisins sur la période d'évaluation future.

## 5. Familles et réglages

Sur les mêmes huit fenêtres, `additive` est une régression Ridge avec effets catégoriels de rue, heure, jour, mois et calendrier, plus horizon et profils historiques. `regularized` réduit le nombre de feuilles LightGBM de 15 à 7 et augmente `min_child_samples` de 80 à 160. `huber` garde les autres paramètres du modèle et utilise l'objectif Huber avec son paramètre par défaut. Aucun réglage n'a été optimisé.

Le modèle additif donne une MAE de débit de 71,40 à Convention et 102,19 à Saint-Pères, contre 48,42 et 67,65 pour la référence. La variante régularisée et Huber n'améliorent pas globalement le débit ; Huber est particulièrement mauvais dans cette configuration. Le fichier `model_holidays.csv` distingue les jours fériés, mais ces huit fenêtres ne comprennent qu'un seul jour férié observé : il ne permet pas de conclusion générale.

## 6. Intervalles empiriques

Pour chaque émission, rue, cible et horizon journalier, les quantiles 10 % et 90 % des résidus signés des fenêtres entièrement terminées avant l'émission sont ajoutés à la prévision. Il faut au moins 100 résidus du groupe ; sinon aucun intervalle n'est produit. Les bornes sont limitées au domaine physique. Cette méthode ne suppose pas que la couverture nominale soit garantie et n'utilise aucune observation du futur.

Sur les fenêtres disposant d'assez de calibration, la couverture horaire agrégée à Convention est de 79,2 % pour `q` et 81,9 % pour `k` ; à Saint-Pères, elle est de 73,6 % et 77,2 %, pour une cible nominale de 80 %. Les scores et largeurs par horizon sont dans `interval_scores.csv`. Les observations horaires sont dépendantes, les fenêtres ont déjà été explorées et la couverture peut changer selon les événements. Il n'y a pas assez de données pour calibrer et évaluer un intervalle récent aux Champs-Élysées.

Les résultats sont exploratoires et les fenêtres récentes ont déjà servi aux études précédentes. Aucune variante n'est sélectionnée automatiquement. Une confirmation exigerait des fenêtres nouvelles, notamment autour de plusieurs fériés, et une analyse des pannes sur davantage de dates.
