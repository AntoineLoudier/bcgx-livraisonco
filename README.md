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

```

## 📂 Architecture du Projet

```text
bcgx-livraisonco/
├── data/
│   ├── raw/                  # Données brutes (ignorées par Git)
│   └── processed/            # Données nettoyées et prêtes pour les modèles (ignorées par Git)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Extraction Open Data Paris (ODSQL) & API Open-Meteo
│   ├── preprocessing.py      # Nettoyage physique, grille horaire 1h, imputation sans leakage
│   ├── features.py           # [À FAIRE] Encodage cyclique, calendrier, lags
│   ├── model.py              # [À FAIRE] Entraînement LightGBM/XGBoost
│   └── evaluate.py           # [À FAIRE] Validation croisée temporelle (TimeSeriesSplit)
├── notebooks/                # Notebooks d'exploration (EDA)
├── models/                   # Modèles sauvegardés
├── main_train.py             # Script d'entraînement global
├── main_predict.py           # Script d'inférence pour la période du 7 au 11 novembre
├── requirements.txt          # Dépendances Python
└── README.md                 # Documentation du projet