# 🚚 LivraisonCo — Prédiction du Trafic Routier Parisien (BCG X)

Projet de Machine Learning développé pour **LivraisonCo** dans le cadre du sprint de deux semaines proposé par **BCG X**.  
L'objectif est d'anticiper le trafic routier parisien sur deux indicateurs critiques (fenêtre de test du **7 au 11 novembre**) :
* **Le débit horaire (`q`)** : nombre de véhicules par heure.
* **Le taux d'occupation (`k`)** : pourcentage de temps où la boucle électromagnétique est occupée (mesure de la congestion : `< 15%` fluide, `15–30%` dense, `30–50%` saturé, `> 50%` bloqué).

**Périmètre (3 artères cibles) :** `Av_des_Champs_Elysees`, `Convention`, `Sts_Peres`.

---

## ⚙️ Installation & Démarrage

```bash
# 1. Créer et activer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\Activate.ps1 sur Windows

# 2. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 3. Lancer l'ingestion des données brutes
python src/data_loader.py

# 4. Lancer le nettoyage et le preprocessing
python src/preprocessing.py

# 5. Construire les variables
python src/features.py

# 6. Évaluer les références sur quatre fenêtres historiques
python src/evaluate.py

# 7. Vérifier les fuites temporelles couvertes par les tests
python -m unittest discover -s tests -v

# 8. Entraîner et évaluer les premiers modèles
python main_train.py

# 9. Comparer l'entraînement à horizons de 1 à 5 jours
python main_multi_horizon.py

# 10. Comparer les fenêtres d'entraînement et la pondération temporelle
python main_recency.py

# 11. Comparer les variables de fériés et la correction des ponts
python main_holidays.py

# 12. Comparer les variables de dynamique récente
python main_dynamics.py

# 13. Comparer un mélange calibré sur les fenêtres passées
python main_blend.py

# 14. Diagnostiquer les erreurs à partir des prévisions historiques du mélange
python main_error_diagnostics.py

# 15. Comparer profils corrigés, variables croisées, modèles, pannes et intervalles
python main_new_experiments.py

# 16. Télécharger le panel public et les voisins de la sélection conservée
python -m src.panel_data --selection configs/panel_selection.json
python -m src.panel_data --neighbors-only --selection configs/panel_neighbors_selection.json
python -m src.panel_preprocessing

# 17. Comparer le transfert et les capteurs adjacents
python main_panel_transfer.py --windows 4 --outages
python main_panel_neighbors.py --windows 2 --outages --output data/processed/panel_neighbors_directed

# 18. Comparer les trajectoires à sorties multiples (PyTorch CPU)
python main_trajectory.py --windows 4 --delays 0 120

# 19. Comparer les avis d'événements vérifiés et les scénarios conditionnels
python main_event_restrictions.py --as-of '2026-11-04 00:00'

# 20. Comparer quatre familles et les ablations de preprocessing/variables
python main_tabular_benchmark.py

```

Le protocole, les variantes et les résultats de ce dernier benchmark sont décrits
dans [la comparaison modèles et variables](docs/benchmark_modeles_features.md).

## 📂 Architecture du Projet

```text
bcgx-livraisonco/
├── data/
│   ├── raw/                  # Données brutes (ignorées par Git)
│   └── processed/            # Données nettoyées et prêtes pour les modèles (ignorées par Git)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Extraction Open Data Paris (ODSQL) & API Open-Meteo
│   ├── preprocessing.py      # Nettoyage physique et imputation causale
│   ├── features.py           # Variables calendaires et retards disponibles
│   ├── model.py              # Entraînement LightGBM sur cibles observées
│   └── evaluate.py           # Validation temporelle et disponibilité météo
├── notebooks/                # Notebooks d'exploration (EDA)
├── models/                   # Modèles sauvegardés
├── main_train.py             # Entraînement et validation temporelle
├── main_predict.py           # Script d'inférence pour la période du 7 au 11 novembre
├── requirements.txt          # Dépendances Python
└── README.md                 # Documentation du projet

Les fichiers préparés doivent être régénérés après modification du pipeline. Les
colonnes `q` et `k` du fichier d'entraînement sont des cibles, à exclure toutes
les deux des entrées du modèle. Voir les conditions de disponibilité et de
validation dans [la documentation technique](docs/architecture.md).
