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
