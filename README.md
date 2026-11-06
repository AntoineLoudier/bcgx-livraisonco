# LivraisonCo — Les Architectes

Prévision horaire du trafic parisien pour préparer les tournées de livraison, dans le cadre du projet BCG X. Le modèle estime le débit **q** (véhicules/heure) et le taux d’occupation **k** (pourcentage du temps d’occupation d’une boucle de comptage) du **7 au 11 novembre 2026**.

## Livrables

| Fichier | Contenu |
|---|---|
| [Présentation complète](presentation/LivraisonCo.pdf) | 19 slides : 8 principales et 11 annexes, avec la maquette utilisateur |
| [PowerPoint](presentation/LivraisonCo.pptx) | Même présentation, avec notes orateur |
| [Prédictions à rendre](rendu/output_Les%20Architectes.csv) | Export au format du brief, généré après l’acquisition du 6 novembre |
| [Résultats détaillés](presentation/results/scores_2_rues.csv) | Scores sur les deux rues évaluées |
| [Méthode et limites](docs/architecture.md) | Données, variables, modèles et protocole de validation |

**Périmètre évalué : Convention et Saint-Pères.** Les Champs-Élysées sont exclus des résultats présentés. Le CSV conserve les trois rues et les 360 lignes attendues par le validateur initial du brief ; les 120 lignes Champs-Élysées sont hors évaluation. Les deux rues évaluées représentent 240 lignes horaires, avec deux cibles par ligne.

## Résultats du rendu

Acquisition du **6 novembre 2026**, avec des mesures réelles jusqu’au **6 novembre à 00 h, heure de Paris**. Quatre fenêtres historiques de cinq jours par rue ; 944 heures observées par cible au total.

| Rue | RMSE q : référence → modèle (véh/h) | RMSE k : référence → modèle (points de %) |
|---|---:|---:|
| Convention | 71,19 → **56,97** | 2,77 → **2,05** |
| Saint-Pères | 79,61 → **75,85** | 2,86 → **2,34** |
| Ensemble des deux rues | 75,59 → **67,23** | 2,81 → **2,20** |

La référence est **J−7 prétraité** : lissage et remplacement des jours spéciaux. Le gain global est de **11,05 % sur q** et **21,75 % sur k**, calculé sur les erreurs concaténées. Une référence J−7 non lissée est également fournie dans les résultats. Les gains sont des baisses de RMSE, pas des taux d’accuracy.

Le modèle combine un ancrage hebdomadaire et une correction résiduelle **LightGBM / XGBoost**, avec des paramètres propres à chaque rue et cible et 49 variables. À données et fenêtres identiques, il réduit la RMSE de 4,47 % sur q et de 2,38 % sur k par rapport à la version précédente. Certaines fenêtres restent moins bonnes que la référence.

Ce backtest a servi au développement du modèle : il ne constitue pas un test indépendant. La météo historique utilisée est observée, et non une archive de prévisions disponibles à chaque date passée. Les économies présentées sont un scénario à vérifier en pilote ; elles ne sont pas déduites des gains de RMSE.

## Vérifier le rendu sans entraîner

Depuis la racine, avec Python 3.13 :

```bash
python3 scripts/verify_submission.py
```

Cette vérification utilise uniquement la bibliothèque standard. Elle contrôle le format, les dates et les valeurs du CSV, sa concordance avec les prévisions de la présentation, les scores recalculés sur les cibles observées et les fichiers du support. Les preuves sont versionnées dans [presentation/results/](presentation/results/) ; les données brutes et les modèles entraînés restent des fichiers générés localement.

## Installation

Environnement utilisé : Python 3.13. Les dépendances sont décrites dans `pyproject.toml` et verrouillées dans `uv.lock`.

```bash
uv sync --locked
```

Alternative sans uv, avec les versions exportées du même verrou :

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Sous Windows, remplacer `.venv/bin/python` par `.venv\Scripts\python.exe`. LightGBM peut nécessiter le runtime OpenMP du système (`libomp` sur macOS, `libgomp` sur Linux).

## Exécuter le pipeline

Les commandes suivantes s’exécutent depuis la racine. L’acquisition nécessite Internet ; aucun compte ni secret n’est requis.

```bash
.venv/bin/python src/data_loader.py
.venv/bin/python src/preprocessing.py
.venv/bin/python src/features.py
.venv/bin/python main_train.py
.venv/bin/python main_predict.py --group "Les Architectes" --output-dir reports
```

`main_predict.py --refresh-all` enchaîne également ces étapes. Les modèles sont écrits dans `models/`, les métriques et prédictions générées dans `reports/`. Le fichier final conservé dans `rendu/` n’est pas remplacé par ces commandes.

**Données datées :** les API sont évolutives et le calendrier du projet est fixé à 2025–2026. Un téléchargement ultérieur ne reproduira pas exactement l’acquisition du 6 novembre. Une reproduction exacte de l’entraînement nécessite les fichiers bruts d’origine dont les empreintes sont dans [le manifeste des résultats](presentation/results/manifest.json). Le dépôt permet de vérifier les scores et les prédictions remis sans retélécharger ces données. La commande de rafraîchissement n’est pas un backtest automatique de la période officielle après le 6 novembre.

## Organisation

```text
src/                 Acquisition, prétraitement, variables, modèles, validation
main_train.py        Validation historique, entraînement et rapports
main_predict.py      Génération du CSV au format du brief
scripts/             Vérification du rendu sans dépendances externes
rendu/               CSV final et manifeste de provenance
presentation/        PDF, PowerPoint, maquette, résultats et scripts de génération
docs/                Documentation technique
data/                Données brutes et préparées, générées localement
models/              Modèles entraînés, générés localement
reports/             Rapports et nouveaux exports, générés localement
```

La [documentation du support](presentation/README.md) décrit les fichiers de preuve et les commandes pour régénérer la présentation complète. Les anciennes versions restent accessibles dans l’historique Git.
