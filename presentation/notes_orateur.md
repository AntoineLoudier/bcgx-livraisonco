# LivraisonCo — Notes de présentation

## 1. La recommandation

Ouvrir sur le coût horaire de la livraison et la décision à prendre, sans annoncer d’économies déjà réalisées. Convention est le premier candidat sur la base du backtest recalculé, pas une garantie de rentabilité ou de significativité statistique. Les gains exacts sont 19,97 % sur q et 25,90 % sur k face au J−7 prétraité. Saint-Pères peut être un second site, avec son propre groupe témoin : ce n’est pas un contrôle interchangeable de Convention. 1 h/jour × 10 €/h × 250 jours = 2 500 €/an, avant coût de solution. Le taux 10 €/h vient du cas ; les 250 jours sont une hypothèse. À heures payées constantes, le gain est de la capacité ; ne pas l’appeler économie de trésorerie. Selon la précision de périmètre donnée par l’utilisateur, les Champs-Élysées sont exclus de tous les scores et graphiques de résultat du support. Le pipeline officiel conserve son export de 360 lignes, mais le périmètre évalué correspond à 240 lignes, chacune portant q et k. Équipe : Antoine Loudier, Amin Belfkira, Baptiste Dejardin, Joseph Bouteiller, Louis Jamin.

## 2. L’économie dépend de la récupération des heures

La sensibilité fait varier deux hypothèses indépendamment. Aucune probabilité n’est associée aux cellules. La part conduite est fixée à 40 % uniquement pour la simulation ; elle doit être mesurée. Le scénario repère est 100 × 40 % × 20 % × 70 % = 5,6 h/jour évitées, soit 5,6 % des heures payées et 14 000 €/an au taux du cas. Le coût horaire de 10 € n’est pas nécessairement le coût employeur complet. À coût et effectif constants, parler de capacité, pas de cash. Le plafond de 14 k€ n’est ni un budget recommandé ni un devis : c’est le seuil de rentabilité en année 1 de ce scénario, en supposant le gain sur 250 jours, sans montée en charge, actualisation ni coûts oubliés. Les 6 k€ initiaux et 4 k€/an de la version précédente sont retirés : ils n’étaient étayés par aucun chiffrage technique. Aucun lien empirique entre gain de RMSE et réduction du temps de conduite n’est estimé.

## 3. Résultats actualisés sur le périmètre évalué

Les scores sont recalculés après réacquisition du trafic et de la météo, prétraitement, features et entraînement du code main. Les quatre fenêtres sont sélectionnées par le protocole existant, avec au moins 85 % de cibles observées ; les dates exactes figurent en annexe. Convention : q 71,19 → 56,97 ; k 2,77 → 2,05. Saint-Pères : q 79,61 → 75,85 ; k 2,86 → 2,34. Le J−7 du pipeline est lissé et corrigé des jours spéciaux : ce n’est pas une simple mesure brute de J−7. Une référence non lissée sur la série nettoyée/imputée est fournie séparément dans les résultats, sur les mêmes lignes. Le score global est calculé sur toutes les cibles des deux rues réunies, pas en moyennant les pourcentages. Il s’agit d’un backtest de développement, avec météo historique observée plutôt que prévisions archivées ; ne pas annoncer un test indépendant ni une performance opérationnelle certifiée. La comparaison ancienne/nouvelle version à données identiques est détaillée en annexe.

## 4. Ce qui a changé dans la version main

Le lissage utilise les heures t−169, t−168 et t−167 avec des poids 0,10/0,80/0,10. Les jours spéciaux des lags sont remplacés en cascade J−21, puis J−14, puis J−7. La moyenne 60/25/15 est mélangée à la médiane J−7/J−14/J−21/J−28 : 90/10 sur Convention et 82/18 sur Saint-Pères. Le modèle ajoute les médianes q et k et retire is_summer_holiday de la liste des entrées : 49 variables. Les résidus d’entraînement sont écrêtés aux centiles 1 et 99 sur les deux rues. La moyenne des corrections LightGBM et XGBoost est pondérée : Convention 1,05 pour q et 1,00 pour k ; Saint-Pères 0,40 pour q et 0,85 pour k. L2=4 sur Convention, 5 sur Saint-Pères ; profondeur et nombre d’arbres réduits sur Saint-Pères. Le ratio de pont est borné entre 0,75 et 0,95. Pour les deux rues évaluées, un férié applique 0,80 à l’ancrage ; le profil de défilé de 2025 ne s’applique qu’aux Champs-Élysées, hors évaluation. Ces coefficients restent des hypothèses de modèle. Les variables météo historiques de ce backtest sont observées ; la nouvelle acquisition ne constitue pas une archive des prévisions émises dans le passé.

## 5. Mesurer le maillon manquant

Cette slide explique pourquoi le gain de prédiction n’est pas directement un gain financier. La durée dépend aussi des arrêts, du nombre de colis, des clients et de l’itinéraire. Un faible q ne prouve pas la fluidité : le taux d’occupation k est utile pour distinguer route vide et congestion. L’exemple est un calcul : jour 10 €/h × durée variable ; nuit 14 €/h × 5 h supposées. Égalité à 7 h, soit un surcroît de durée de 40 %. La nuit devient moins chère au-delà, à travail identique et en supposant aucune différence d’autre coût. Le coût employeur complet, les contraintes horaires et les restrictions locales doivent être intégrés avant une recommandation. Le graphe est calculé à partir des hypothèses affichées, pas à partir d’un historique GPS. Aucun moteur de tournée opérationnel n’est revendiqué.

## 6. Maquette validée : du planning à l’exécution

La maquette graphique séparée validée par l’utilisateur est intégrée comme image dans cette slide. Le fichier source n’a pas été modifié : seuls le titre et les marges extérieures de la planche sont masqués dans la mise en page. L’exploitation compare les créneaux et vérifie les contraintes ; le livreur consulte ensuite un planning validé et enregistre arrivée ou retard. Les deux écrans représentent des étapes successives. La courbe, les horaires et la carte sont illustratifs et ne reprennent pas les prévisions recalculées. La ligne Champs-Élysées reste visible dans cette maquette approuvée à titre d’exemple de revue manuelle ; elle ne fait pas partie de l’évaluation. Aucune application opérationnelle ni optimisation de tournée n’est revendiquée. L’image est remplaçable dans PowerPoint mais ses textes internes ne sont pas éditables.

## 7. Une comparaison prospective, pas une démonstration sélectionnée

Séparer validation prédictive et expérimentation opérationnelle. Les fenêtres déjà utilisées pour choisir les modèles sont des données de développement. Prévoir une période chronologique ultérieure gelée ; examiner erreurs par horizon, rue, type de jour et qualité des données. Pour l’impact business, le contrôle est le planning réellement utilisé par l’exploitation. Randomiser des groupes de tournées comparables au sein d’une zone, à une granularité limitant les échanges de livreurs et les interférences. Ne pas comparer simplement Convention traitée à Saint-Pères témoin : les différences de rue seraient confondues avec l’effet. Si la randomisation est impossible, utiliser un appariement explicite et annoncer les limites d’interprétation causale. Définir avant le test le coût complet par colis effectivement livré, une marge de non-dégradation de la ponctualité et le traitement des fériés/incidents. Estimer l’incertitude en tenant compte du groupement des tournées et des jours. Dimensionner l’échantillon selon la variance observée et l’effet minimal rentable. Aucun seuil numérique ou délai de significativité n’est inventé. Ne pas extrapoler une journée fériée à une année.

## 8. Terminer par une demande opérationnelle

Demander l’autorisation de préparer le pilote, les données nécessaires et un interlocuteur responsable des plannings. Ne pas demander un budget au montant inventé ; établir les coûts d’intégration, d’exploitation et d’accompagnement après cadrage. Convention est la première priorité selon les résultats disponibles. Saint-Pères peut être un second terrain avec comparaison interne, utile pour tester un gain prédictif moindre. Les Champs-Élysées sont hors évaluation selon le périmètre précisé. Le solveur et les modèles de durée viennent après l’instrumentation. L’absence de preuve économique ou une dégradation du service doit conduire à adapter ou arrêter le déploiement, sans avoir à défendre le modèle à tout prix. Ne pas promettre un calendrier de six semaines ni une puissance statistique avant de connaître les volumes.

## 9. Isoler le progrès du code

Les versions ancienne et nouvelle sont réentraînées sur les mêmes données fraîchement téléchargées, prétraitées avec le même pipeline. Les dates, cibles et historiques des plis sont identiques. Les valeurs ne comparent donc pas directement les chiffres de l’ancien PDF aux nouvelles fenêtres. Cette comparaison mesure l’effet de la version sur ce backtest ; elle ne constitue pas un test indépendant de la sélection des hyperparamètres. Les sorties détaillées et les empreintes de données sont dans results/. Un gain agrégé peut masquer des pertes sur certains plis.

## 10. Définition exacte des scores

La RMSE globale est calculée en concaténant les erreurs des deux rues, avec pondération implicite par le nombre de mesures réelles. Convention a 464 heures et Saint-Pères 480, soit 944 heures par cible. Aucun score des Champs-Élysées ne contribue aux valeurs. Le gain est 1 − RMSE modèle / RMSE référence, calculé avant arrondi. Le J−7 prétraité reprend les colonnes baseline_q/k de la validation officielle. Le CSV contient aussi la référence J−7 non lissée de la série nettoyée/imputée, pour éviter d’assimiler la référence prétraitée à une observation brute.

## 11. Lire les fenêtres sans confondre les périodes des rues

Les plages affichées sont celles des cibles effectivement observées. Des heures manquantes en bord de fenêtre expliquent qu’une plage paraisse plus courte que les 120 heures nominales. Les fenêtres sont choisies par le protocole existant, au moins 85 % de mesures et entraînement sur le passé. Les horaires ne doivent pas être remplacés par le minimum/maximum commun aux deux rues ou aux Champs-Élysées : ces périodes peuvent différer. Saint-Pères perd face à sa référence lors du pli d’août. Les détails q et k se trouvent dans le CSV ; aucune sélection des plis après le calcul.

## 12. Sources et limites de disponibilité

Les identifiants et les couples de nœuds exacts sont conservés dans le manifeste. Les données brutes ont été retéléchargées plutôt que réutilisées. Les comptes de mesures sont calculés après nettoyage et agrégation horaire ; les doublons de changement d’heure expliquent un éventuel écart au compte brut. La complétude de la météo du 7–11 novembre concerne la prévision actuelle. Elle ne résout pas la question des vintages météo des backtests. L’export officiel reste compatible avec les trois arcs du code ; ce support et son agrégat d’évaluation portent exclusivement sur les deux rues demandées.

## 13. Détail de la correction

La correction est la moyenne 50/50 LightGBM et XGBoost multipliée par le poids de la rue et de la cible. Le lissage horaire à J−7 utilise uniquement des valeurs âgées de plus de six jours et reste disponible à horizon cinq jours hors délai de publication. Les heures spéciales sont remplacées en cascade inverse. Les règles calendrier ne doivent pas être présentées comme des effets causalement estimés. Le profil Armistice 2025 spécifique aux Champs-Élysées ne s’applique pas aux deux rues évaluées.

## 14. Prévisions actualisées du 7 au 11 novembre

Les courbes proviennent des modèles finaux réentraînés après acquisition. Aucun repli sur l’ancrage n’a été utilisé pour dessiner ces résultats. La courbe de la maquette produit reste illustrative et distincte de ces prévisions. Aucun itinéraire ni créneau optimal n’est déduit automatiquement de ces seules courbes.

## 15. Dernière fenêtre : mesures et prédictions

Deux rues et deux cibles, même dernière fenêtre nominale. Les trous éventuels correspondent à des cibles non observées. Il s’agit de prédictions hors entraînement temporel dans ce backtest de développement, pas d’un holdout indépendant du travail de sélection.

## 16. Profils horaires sur les mesures actualisées

Profils descriptifs calculés sur les cibles réelles des deux rues. Ils ne constituent pas une évaluation du modèle et ne tiennent pas lieu de prévision de tournée.

## 17. Débit et occupation apportent des informations différentes

Les nuages de points sont descriptifs. L’occupation n’est ni une vitesse ni directement un temps de parcours. Un seuil de congestion opérationnel doit être calibré avec des mesures terrain.

## 18. Les variables expliquent la correction, pas tout le trafic

Les valeurs SHAP du pipeline portent sur le résidu LightGBM. Ici elles sont multipliées par 0,5 et par le poids résiduel du modèle pour exprimer la contribution de cette composante avant clipping de la prédiction. Elles ne couvrent ni XGBoost, ni l’ancrage, ni le post-traitement non linéaire. Ce graphique décrit les modèles finaux sur un échantillon d’entraînement, pas la causalité des variables ni un effet économique.

## 19. Variables exactes

La liste est extraite du code exécuté, conservée dans le manifeste, puis répartie entre familles. Les variables is_road_closed, is_paris_respire et is_armistice_parade_hours peuvent être constantes sur certaines rues ; leur présence dans la liste ne signifie pas qu’elles contribuent à leurs modèles. Aucun gain causal n’est attribué individuellement à l’ajout des médianes.
