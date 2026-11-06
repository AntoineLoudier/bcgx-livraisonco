# LivraisonCo — Présentation actualisée après acquisition du 6 novembre

Le support contient **8 slides principales et 11 annexes**. Les chiffres et figures sont recalculés avec `main` à la révision `c00b661`, après réacquisition du trafic et de la météo. **Convention et Saint-Pères sont les seules rues évaluées**, conformément à la précision reçue. Les Champs-Élysées sont exclus de tous les tableaux de résultat, agrégats et graphiques du support.

- `LivraisonCo.pdf` : présentation complète avec annexes.
- `LivraisonCo.pptx` : PowerPoint avec notes orateur.
- `notes_orateur.md` : notes et définitions détaillées.

La maquette validée `assets/maquette_exploitation_livreur.png` est intégrée en slide 6. Le fichier image reste inchangé ; la mise en page masque seulement les marges extérieures et le titre de la planche. Ses horaires, courbe et carte sont fictifs. La ligne Champs-Élysées reste visible dans cette illustration approuvée, sans contribuer à l’évaluation.

Les slides principales utilisent des objets PowerPoint éditables, sauf la maquette. Les graphiques scientifiques des annexes sont des images régénérées à partir des données et modèles actuels ; les tableaux d’annexe restent éditables. Le rendu ne conserve qu’une version du support, complète avec ses annexes. Les anciennes versions et planches sources restent accessibles dans l’historique Git.

## Résultats actualisés

La nouvelle acquisition contient des mesures réelles jusqu’au **6 novembre 2026 à 00 h, heure de Paris**, sur les deux rues. La validation utilise quatre fenêtres de cinq jours par rue et uniquement des cibles réellement observées : **464 heures pour Convention, 480 pour Saint-Pères, 944 au total par cible**.

| Rue | RMSE q, référence → modèle (véh/h) | Baisse q | RMSE k, référence → modèle (points de %) | Baisse k |
|---|---:|---:|---:|---:|
| Convention | 71,19 → 56,97 | 19,97 % | 2,77 → 2,05 | 25,90 % |
| Saint-Pères | 79,61 → 75,85 | 4,73 % | 2,86 → 2,34 | 18,17 % |
| Ensemble des deux rues | 75,59 → 67,23 | 11,05 % | 2,81 → 2,20 | 21,75 % |

La référence du pipeline est **J−7 prétraité** : lissage de phase et remplacement des jours spéciaux. Elle ne correspond pas à une mesure brute de la même heure sept jours auparavant. Le fichier `results/scores_2_rues.csv` fournit aussi une référence J−7 non lissée calculée sur la série nettoyée/imputée, sur les mêmes cibles. Tous les gains sont des baisses de RMSE, pas des taux d’accuracy ni des économies.

Le global est calculé sur les erreurs des deux rues concaténées ; il ne résulte pas d’une moyenne de pourcentages ou de RMSE.

## Progrès par rapport à l’ancienne version

L’ancienne implémentation (`e6c9818`) a été réentraînée sur **les mêmes données actualisées, historiques d’entraînement et fenêtres** que la nouvelle (`c00b661`). On ne compare pas directement les anciens chiffres du PDF aux nouvelles fenêtres.

| Rue | Baisse de RMSE q | Baisse de RMSE k |
|---|---:|---:|
| Convention | 0,68 % | 4,02 % |
| Saint-Pères | 6,36 % | 1,10 % |
| Ensemble des deux rues | **4,47 %** | **2,38 %** |

Source : `results/comparaison_versions.csv`. Ces gains sont agrégés ; certaines fenêtres restent moins bonnes que la référence ou l’ancienne version. Le pli d’août sur Saint-Pères perd face à J−7 prétraité sur q et k. Les dates exactes, effectifs et scores sont dans `results/scores_par_fenetre.csv`. `Start` et `End` désignent les premières et dernières **cibles observées** ; les fenêtres nominales durent 120 heures, même si des observations manquent en bordure.

## Portée et limites

- Il s’agit du **backtest de développement** existant, pas d’un nouveau test indépendant de la sélection des modèles et paramètres.
- Le trafic et la météo ont bien été retéléchargés. La météo couvre les 120 heures du 7–11 novembre sans valeur manquante.
- Les fenêtres de validation utilisent néanmoins la **météo historique observée**, et non une archive de prévisions émises à chaque date passée. Les résultats ne certifient donc pas la performance opérationnelle avec prévisions météo et délais réels de publication.
- Le nouveau modèle utilise 49 variables : ajout des médianes hebdomadaires q/k, retrait de `is_summer_holiday`, stabilisation des lags, écrêtage des résidus et poids de correction adaptés à la rue/cible.
- Les règles de pont et de férié restent des hypothèses de modèle. Pour les deux rues évaluées, le férié multiplie l’ancrage par 0,80 ; le profil de défilé des Champs-Élysées ne s’applique pas à ces deux rues.
- L’export technique officiel conserve ses trois rues et ses 360 lignes. Le support analyse uniquement les **240 lignes des deux rues évaluées**, avec q et k dans chaque ligne. Aucun score des Champs-Élysées ne contribue au global affiché.

## Fichiers de preuve

Les fichiers dans `results/` sont conservés avec la présentation pour rendre les chiffres vérifiables, même si `reports/` et les données brutes sont ignorés par Git :

- `manifest.json` : commits, empreintes SHA-256 des entrées et du code, versions des bibliothèques, dates d’acquisition, périmètre, variables et limites.
- `scores_2_rues.csv` : scores complets et les deux définitions de référence hebdomadaire.
- `comparaison_versions.csv` : comparaison contrôlée ancienne/nouvelle version.
- `scores_par_fenetre.csv` : périodes observées, effectifs et scores par pli.
- `predictions_validation_2_rues.csv` : cibles, prédictions des deux versions, références et indicateurs d’observation, pour recalculer les scores.
- `previsions_7_11_novembre_2_rues.csv` : les 240 prévisions horaires actuelles.
- `importance_variables.csv` : importances et SHAP du pipeline courant.

Le graphique SHAP affiche la composante LightGBM après multiplication par 0,5 et par le poids résiduel de la rue/cible, avant clipping final. Il ne couvre ni XGBoost, ni l’ancrage, ni le post-traitement non linéaire. L’échantillon est constitué des dernières lignes d’entraînement, pas d’un jeu de test indépendant.

## Régénération

Pour régénérer seulement le PDF et le PowerPoint à partir des résultats déjà conservés :

```bash
uv sync --locked --extra presentation
.venv/bin/python presentation/build_presentation.py
```

Pour actualiser les données et les résultats à nouveau, depuis la racine :

```bash
.venv/bin/python src/data_loader.py
.venv/bin/python src/preprocessing.py
.venv/bin/python src/features.py
.venv/bin/python main_train.py
.venv/bin/python main_predict.py
.venv/bin/python presentation/refresh_evidence.py
.venv/bin/python presentation/build_presentation.py
```

`refresh_evidence.py` vérifie que les artefacts d’entraînement sont plus récents que les données et le code. Il réentraîne la version de comparaison à partir de Git, sur les mêmes fenêtres, puis recalcule les agrégats à deux rues et les figures. Il limite son calcul à quatre cœurs lorsque l’affinité CPU est disponible. L’historique Git contenant `e6c9818` est nécessaire pour cette comparaison.

La vérification `python3 scripts/verify_submission.py`, depuis la racine, contrôle aussi la concordance avec le CSV final de `rendu/`.

Le rendu lit `scores_source.json` et `results/` ; il ne remplace plus les résultats par les anciens chiffres écrits dans le script. Les paramètres de mise en page et certaines conclusions narratives correspondent au calcul du 6 novembre : les relire après un nouveau rafraîchissement.

Le PDF fourni utilise Arimo (compatible en métriques avec Arial) et le PowerPoint demande Arial. Pour régénérer avec la même police sous Linux, installer les polices Croscore (`fonts-croscore` sous Debian/Ubuntu) ; à défaut, le script utilise DejaVu Sans fourni avec Matplotlib, ce qui peut modifier les retours à la ligne. Les polices ne sont pas intégrées au PowerPoint. Les PDF sont inspectés visuellement ; le rendu dans Microsoft PowerPoint n’est pas contrôlé ici.

## Économie et maquette

Les hypothèses économiques sont inchangées : 100 heures payées/jour, 40 % de conduite, 20 % de conduite en moins, 70 % de ce temps récupérable, 10 €/h et 250 jours/an. Cela donne 5,6 h/jour, soit 14 k€/an bruts **dans le scénario**, pas un gain réalisé par le modèle. Les 14 k€ sont un seuil de coût de solution à l’équilibre dans cette simulation ; aucun devis n’a été produit.

À heures payées constantes, le temps libéré représente de la capacité. Aucun lien empirique entre baisse de RMSE et gain de temps n’a été estimé. La maquette illustre le parcours exploitation → livreur et ne constitue pas une application fonctionnelle.
