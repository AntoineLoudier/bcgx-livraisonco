# Benchmark des modèles, du preprocessing et des variables

`main_tabular_benchmark.py` compare LightGBM, CatBoost, XGBoost et ExtraTrees,
puis des ablations de preprocessing et de variables avec LightGBM et CatBoost.
La cible par défaut est le débit `q` ; `--targets q k` inclut l'occupation.
Le benchmark produit des prévisions historiques et des diagnostics. Aucun
modèle n'est promu dans `main_predict.py`.

## Protocole et périmètre

Chaque fenêtre extérieure couvre cinq jours à partir d'une émission à 23 h.
Les délais de disponibilité de 0 et 120 h sont évalués séparément par défaut.
Le calendrier futur est connu ; ni météo observée future, ni trafic futur des
autres rues ne sont utilisés. Les fenêtres extérieures sont non chevauchantes.
Les dates de fin des deux fenêtres intérieures de réglage précèdent la dernière
date de données accessibles de la fenêtre extérieure. Chaque entraînement
intérieur respecte également son propre délai et exclut ses futures cibles.

Les quatre familles reçoivent les mêmes exemples multi-horizon, les mêmes
variables de référence et deux configurations prédéfinies chacune. Le choix
minimise la moyenne des MAE des rues évaluables, dans les unités originales,
sur les seules fenêtres intérieures. Il s'agit d'un budget égal en nombre de
configurations, pas en secondes de calcul. Temps et effectifs sont exportés.
Les espaces de paramètres sont modestes : ce n'est pas une optimisation exhaustive.

Les variables catégorielles ont des domaines calendaires explicites : traitement
natif pour LightGBM/XGBoost, chaînes pour CatBoost et encodage one-hot pour
ExtraTrees. ExtraTrees utilise une imputation médiane ajustée sur l'entraînement
seulement, avec indicateurs de manque. Les autres modèles traitent les NaN
numériques nativement. Les arbres boostés minimisent une perte absolue ; les
coupures d'ExtraTrees utilisent la perte quadratique pour garder un temps de
calcul raisonnable. Tous sont sélectionnés et évalués avec la MAE.

La méthode `reference` conserve exactement le LightGBM et les colonnes numériques
du pipeline précédent. `lightgbm/baseline` ajoute la sélection intérieure et
le traitement catégoriel : son écart à `reference` ne mesure donc pas une seule
modification. Les comparaisons entre nouvelles familles utilisent le même
ensemble d'information, avec l'encodage adapté à chacune.

Les transformations ne concernent jamais la vérité de validation : les mêmes
cibles réellement observées du fichier préparé sont évaluées pour toutes les
méthodes. Une méthode manquante provoque une erreur plutôt qu'un changement
silencieux de dénominateur. Les Champs-Élysées restent non évaluables sur les
périodes récentes. Les heures locales et l'agrégation de l'heure répétée au
changement d'heure reprennent la convention existante.

Les délais sont simulés à partir de l'archive actuellement disponible. Les
sources peuvent réviser leurs historiques : sans archives datées de chaque
émission, ce protocole ne prouve pas que chaque valeur était déjà publiée
exactement sous cette forme à l'époque.

## Ablations

Les réglages choisis pour chaque famille sur les variables `baseline` restent
fixes entre les ablations : cela isole l'effet de la représentation. Les ablations
sont également prédites hors entraînement sur les fenêtres intérieures, afin de
sélectionner un candidat sans regarder les scores extérieurs.

| Variante | Modification | Comparaison principale |
|---|---|---|
| `compact` | Calendrier simple, horizon, trois retards hebdomadaires, moyenne et profil historique ; retrait des règles manuelles spécifiques | `baseline` |
| `missing` | Effacement des entrées imputées, recalcul des variables et indicateurs de retards réellement observés ; pas de substitution Champs-Élysées | `compact` |
| `quality` | Couverture 24/168 h, ancienneté, suites constantes, suites et proportions de zéros | `compact` |
| `profiles` | Médiane, dispersion et effectifs de mêmes jour/heure sur 28/84/180 jours ; ancienneté et anomalies récentes normalisées | `compact` |
| `shape` | Parts matin/soir, amplitude relative, heure de pointe et moyenne de la dernière journée terminée ; indicateur de complétude | `compact` |
| `analogues` | Médiane et dispersion des suites de cinq jours de cinq situations passées proches ; distance et couverture | `compact` |
| `joint` | Relation débit/occupation : corrélation, écart entre anomalies et fréquence d'occupation haute avec débit bas | `compact` |
| `network` | Anomalies récentes normalisées des deux autres rues cibles, avec nombre de sources disponibles | `compact` |
| `regime` | Remplacement du profil cumulatif par un profil depuis la dernière rupture signalée, avec effectif et ancienneté | `compact` |
| `all` | Union des sept blocs enrichis, incluant les profils de régime ; imputation d'origine | Blocs séparés et `compact` |
| `log` | Entraînement sur `log1p(cible)`, retour aux unités initiales avant sélection et score | `compact` |
| `relative` | Cible divisée par le profil causal de 84 jours ; échelle de repli explicitée dans le code | `profiles` |
| `preserve_zeros` | Rétablissement des seuls zéros conjoints bruts supprimés par l'ancienne heuristique, réimputation causale et variables de qualité | `quality` |

Les valeurs physiquement impossibles restent exclues. Les suites de zéros ou
constantes deviennent des indicateurs ; elles ne sont pas automatiquement
supprimées. Le rétablissement des zéros ne constitue pas une vérification du
capteur : `raw_zero_audit.csv` en indique les effectifs et les vérités extérieures
restent celles du fichier de référence. Les cibles imputées restent exclues de
l'apprentissage, sauf les zéros bruts explicitement rétablis dans cette variante.

Les nouveaux résumés ignorent toujours les valeurs imputées. Un profil de même
jour/heure est mis à jour au passage de ce créneau ; sa fenêtre glissante s'arrête
à cette mise à jour disponible avant l'émission. Une journée incomplète ne fournit
ni total partiel pris pour un total, ni profil horaire présenté comme complet.
Les parts de `k` décrivent sa forme intrajournalière, pas un nombre de véhicules.
La relation entre débit et occupation n'est pas interprétée comme une vitesse.

La rupture est une heuristique fixée avant l'expérience : écart absolu supérieur
à 0,25 entre anomalie relative moyenne des 7 derniers jours et des 21 précédents,
maintenu 72 h, avec au moins 120/360 mesures. On réinitialise le profil au début
de chaque épisode de signal persistant. Cela signale un changement potentiel,
sans distinguer modification réelle de circulation et défaut de capteur.

Les analogues utilisent les anomalies récentes et les parts matin/soir, avec
distance quadratique moyenne sur au moins quatre descripteurs communs. Ils ont
le même jour de semaine et la même heure de disponibilité que la requête.
Leurs 120 heures après émission, délai inclus, doivent être terminées avant la
disponibilité courante et compter au moins 96 observations de débit. Les valeurs
manquantes restantes ne contribuent pas aux statistiques. Le nombre de voisins
est fixé à cinq ; leurs trajectoires futures ne servent jamais à choisir les
plus proches. Cette première représentation ne conditionne pas explicitement
les analogues sur les événements futurs : le calendrier reste fourni au modèle.

## Sélection, mélange et audits

`selected_inner` choisit famille et variante sur les fenêtres intérieures
uniquement. `blend_inner` choisit un mélange de LightGBM et d'une autre famille
sur les prévisions intérieures, avec poids LightGBM dans {0, 0,25, 0,5, 0,75, 1}.
Il ne sélectionne pas son poids sur les scores extérieurs. Les corrélations
d'erreurs exportées sont descriptives et ne changent pas cette sélection.

Les sorties comprennent les scores par rue, horizon, fenêtre, jour férié et
jour de début, les pires fenêtres et le nombre de fenêtres gagnées. Le bootstrap
rééchantillonne des fenêtres complètes et reste exploratoire avec peu de blocs.
Les audits par fenêtre mesurent les valeurs manquantes, les colonnes constantes
et les corrélations supérieures à 0,98 sur l'entraînement seulement. Ils ne
suppriment pas automatiquement des variables. Les tests vérifient l'absence
d'effet d'une perturbation future et l'égalité des variables entre entraînement
et inférence pour les cinq horizons, avec délais de 0 et 120 h.

Un manifeste enregistre les empreintes des données, du code, les versions des
bibliothèques et les paramètres. Une reprise avec un manifeste différent est
refusée. Chaque fenêtre n'est déclarée terminée qu'après écriture des prédictions,
des réglages et des sélections. Changer de répertoire après une modification.

## Exécution

L'exécution complète entraîne plusieurs centaines de modèles ; elle peut prendre
plusieurs dizaines de minutes sur CPU. Chaque fenêtre terminée est réutilisée
en cas de reprise avec le même manifeste.

```bash
python -m pip install -r requirements.txt

# Benchmark complet du débit, quatre fenêtres et deux délais.
python main_tabular_benchmark.py

# Familles seules, sans ablations.
python main_tabular_benchmark.py --variants --output data/processed/tabular_models_only

# Fenêtre du 1er novembre séparée des fenêtres ordinaires.
python main_tabular_benchmark.py --starts 2026-10-28 --output data/processed/tabular_holiday

# Étendre ensuite aux deux indicateurs.
python main_tabular_benchmark.py --targets q k --output data/processed/tabular_both_targets

# Comparaison ciblée de l'occupation avec cinq jours de retard.
python main_tabular_benchmark.py --targets k --delays 120 --families lightgbm extratrees --variants --output data/processed/tabular_benchmark_k_120h
python main_tabular_benchmark.py --targets k --starts 2026-10-28 --delays 120 --families lightgbm extratrees --variants --output data/processed/tabular_benchmark_k_holiday

python -m unittest tests.test_tabular_benchmark -q
```

Les fenêtres disponibles ont déjà été explorées lors d'autres expériences : ce
benchmark ne crée pas rétroactivement un test indépendant. Une promotion exige
une confirmation sur de nouvelles fenêtres et davantage de jours fériés.

## Résultats exploratoires : débit

Quatre fenêtres ordinaires de cinq jours débutent les 30 septembre, 8, 16 et
24 octobre 2026. Elles donnent respectivement 457 et 456 heures observées à
Convention et Saint-Pères. Le tableau indique la MAE en véhicules/h ; les
configurations de chaque famille ont été sélectionnées sur les deux fenêtres
intérieures propres à chaque émission. `Référence` correspond au modèle du
projet, les quatre autres aux familles comparées avec les variables baseline.

| Délai | Rue | Référence | LightGBM | CatBoost | XGBoost | ExtraTrees |
|---|---|---:|---:|---:|---:|---:|
| 0 h | Convention | 48,09 | 47,98 | 48,40 | 47,39 | **46,55** |
| 0 h | Saint-Pères | **60,85** | 63,81 | 66,19 | 64,14 | 62,72 |
| 120 h | Convention | 50,27 | 49,11 | 50,58 | 48,95 | **47,41** |
| 120 h | Saint-Pères | 61,82 | 63,68 | 64,73 | 63,67 | **59,03** |

ExtraTrees gagne les quatre fenêtres de Convention à 0 h et trois sur quatre à
120 h ; il améliore la pire fenêtre de 61,09 à 59,52, puis de 69,09 à 61,67.
À Saint-Pères, il gagne trois fenêtres sur quatre à 0 h mais perd fortement la
dernière : la pire MAE passe de 85,82 à 106,83. À 120 h, il gagne deux fenêtres
sur quatre, mais sa pire MAE passe de 85,02 à 91,28. Les intervalles bootstrap
exploratoires sur l'écart de MAE à 120 h sont [-5,79 ; +0,10] et [-8,41 ; +3,46]
véhicules/h pour Convention et Saint-Pères : quatre blocs sont insuffisants pour
interpréter ces intervalles comme une garantie statistique.

Le candidat `selected_inner` est choisi sur les seules fenêtres antérieures,
mais sa MAE à Saint-Pères atteint 71,69 contre 60,85 pour la référence à 0 h et
67,41 contre 61,82 à 120 h. Cela illustre le risque de sélectionner parmi de
nombreuses variantes avec seulement deux fenêtres intérieures et un choix commun
aux rues. Le mélange intérieur n'apporte pas non plus de gain généralisé : sa
MAE à Saint-Pères est de 63,11 et 62,25 aux deux délais.

Les blocs de variables fournissent des résultats contrastés. Avec CatBoost à
Convention, le bloc `profiles` réduit la MAE du modèle compact de 48,55 à 46,99
à 0 h et de 49,81 à 48,33 à 120 h. À Saint-Pères, la transformation `relative`
détériore la MAE de CatBoost de 68,51 à 88,40 à 0 h et de 70,71 à 188,42 à
120 h, comparée au modèle utilisant les mêmes variables de profil sans
transformation de cible. Les taux de manque et corrélations de chaque bloc
figurent dans les audits. Ces comparaisons, nombreuses et effectuées sur des
fenêtres déjà explorées, sont descriptives ; elles ne suffisent pas à sélectionner
un bloc pour la production.

La fenêtre événementielle du 28 octobre au 1er novembre est étudiée séparément
car elle chevauche la dernière fenêtre ordinaire. Sur les 24 heures observées du
1er novembre :

| Délai | Rue | Référence | ExtraTrees |
|---|---|---:|---:|
| 0 h | Convention | 86,40 | 71,02 |
| 0 h | Saint-Pères | 77,47 | 86,71 |
| 120 h | Convention | 96,36 | 63,04 |
| 120 h | Saint-Pères | 84,04 | 91,74 |

Ce seul jour férié ne permet pas d'estimer la performance sur les autres fériés.
En particulier, les dégradations à Saint-Pères et la dernière fenêtre ordinaire
ne justifient pas de remplacer automatiquement la référence par ExtraTrees.
Les résultats détaillés de cette exécution sont dans
`data/processed/tabular_benchmark_0h/`, `tabular_benchmark_120h/` et
`tabular_benchmark_holiday/`, ignorés par Git.

## Résultats exploratoires : occupation

Une comparaison ciblée inclut la référence, LightGBM et ExtraTrees avec un
délai de 120 h, sans les ablations de variables. Sur les quatre mêmes fenêtres
ordinaires et les mêmes mesures réellement observées :

| Rue | Heures observées | Référence | LightGBM | ExtraTrees |
|---|---:|---:|---:|---:|
| Convention | 457 | 1,067 | 1,052 | 1,163 |
| Saint-Pères | 456 | 1,548 | 1,569 | 1,504 |

MAE en points d'occupation. ExtraTrees gagne une fenêtre sur quatre à
Convention et trois sur quatre à Saint-Pères. Sa pire fenêtre est de 1,974
contre 2,072 pour la référence à Convention, et de 1,600 contre 1,647 à
Saint-Pères. Sur le seul 1er novembre et 24 heures par rue, la MAE
référence / ExtraTrees est de 0,582 / 0,461 à Convention et 1,638 / 1,457
à Saint-Pères. Cet événement isolé ne prouve pas une amélioration sur les
fériés. Les résultats et leurs manifestes sont dans
`data/processed/tabular_benchmark_k_120h/` et `tabular_benchmark_k_holiday/`.

Références des interfaces : [CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit),
[valeurs manquantes CatBoost](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing),
[paramètres XGBoost](https://xgboost.readthedocs.io/en/latest/parameter.html).
