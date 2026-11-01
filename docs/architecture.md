# 📚 Documentation Technique — Pipeline Data & Feature Engineering

Ce document détaille les choix techniques et l'implémentation du pipeline réalisé pour le cas **LivraisonCo** (BCG X).

## 1. Module d'Acquisition (`src/data_loader.py`)
- **Trafic (`download_paris_traffic`) :** Utilise l'endpoint d'export direct CSV de l'API Open Data Paris v2.1 (`comptages-routiers-permanents`, historique glissant de 13 mois) avec un filtre ODSQL strict sur les 3 couples de nœuds amont/aval du brief officiel :
  - `Champs-Elysées` : `Av_Champs_Elysees-Washington` $\rightarrow$ `Av_Champs_Elysees-Berri`
  - `Convention` : `Lecourbe-Convention` $\rightarrow$ `Convention-Blomet`
  - `Saint-Pères` : `Sts_Peres-Voltaire` $\rightarrow$ `Sts_Peres-Universite`
- **Météo (`download_paris_weather`) :** Combine l'API Archive (`/v1/archive`) jusqu'à $J-5$ et l'API Forecast (`/v1/forecast`) de $J-7$ à $J+16$ afin de couvrir sans trou la période d'entraînement ainsi que la fenêtre de test du **7 au 11 novembre 2026**. Extrait `temperature_2m`, `precipitation`, `rain`, et `wind_speed_10m` au pas horaire.

## 2. Module de Prétraitement (`src/preprocessing.py`)
Le pipeline exécute 5 étapes pour préparer les données de ML :
1. **Standardisation temporelle :** Conversion des horodatages UTC vers le fuseau `Europe/Paris` et traitement des doublons liés au passage à l'heure d'hiver (moyenne des deux heures `02:00`).
2. **Nettoyage physique (Diagramme Fondamental) :** Suppression des valeurs aberrantes ($q \notin [0, 8000]$, $k \notin [0, 100]$) et conversion en `NaN`.
3. **Détection des faux zéros :** En journée (`07h00–22h00`), si une voie ouverte (`is_road_closed == 0`) affiche simultanément $q=0$ et $k=0$, il s'agit d'une panne de capteur $\rightarrow$ conversion en `NaN`. Les fermetures réelles (`etat_barre` actif, comme le matin du 11 novembre aux Champs-Élysées) sont préservées.
4. **Grille horaire continue :** Reconstruction d'un pas strict de `1h` sans trou temporel pour les 3 arcs cibles.
5. **Imputation hybride sans fuite du futur (Zero Data Leakage) :**
   - Micro-coupures ($\le 3\text{ heures}$) : Interpolation linéaire temporelle.
   - Pannes longues ($> 3\text{ heures}$) : Moyenne causale des 3 semaines passées à la même heure (`shift(168)`, `shift(336)`, `shift(504)`).
   - Colonnes indicateurs `is_imputed_q` / `is_imputed_k` ajoutées pour tracer les valeurs reconstruites.
   - **Sortie générée :** `data/processed/traffic_clean.csv`.

## 3. Module de Feature Engineering (`src/features.py`)
1. **Extension à la grille de test (`extend_grid_to_test_period`) :** Prolonge la grille horaire des 3 arcs jusqu'au `2026-11-11 23:00`, joint les prévisions météo horaires, et projette l'état `is_road_closed = 1` sur `Champs-Elysées` le `2026-11-11` de `08h00` à `14h00` (conformément à l'historique observé le `2025-11-11`).
2. **Calendrier & Modélisation fine des Ponts (`1 jour` vs `2 jours`) :**
   - Encodage cyclique ($\sin/\cos$) de l'heure et du jour de la semaine, indicateurs `is_weekend`, `is_public_holiday` et `is_school_holiday` (calendrier Zone C Paris 2025–2026).
   - **Intensité du pont (`bridge_intensity`) :** Distingue un pont de 1 jour (`1.0`, ex. lundi 10 nov 2025 veille d'un mardi férié, ou vendredi 15 mai 2026 lendemain de l'Ascension) d'un pont de 2 jours / viaduc lorsque le férié tombe un mercredi (comme le mercredi 11 nov 2026 : `0.50` pour le mardi 10 nov et `0.25` pour le lundi 9 nov).
3. **Interactions Météo $\times$ Pointe :** Variables `is_rush_hour`, `is_raining`, `rain_in_rush_hour` (effet multiplicateur de la pluie sur la congestion $k$ en heure de pointe) et `temp_diff_24h`.
4. **Lags causaux à horizon 5 jours & Pattern SNCF $A-1$ recalibré (`add_causal_lags_and_sncf_patterns`) :**
   - Lags hebdomadaires garantis disponibles lors du gel du code le 6 novembre : `lag_168h` ($J-7$), `lag_336h` ($J-14$), `lag_504h` ($J-21$), moyenne pondérée sur 3 semaines (`weekly_mean_3w`) et écart-type (`weekly_std_3w`).
   - Moyenne historique causale par régime de jour (`regime_hist_mean`, décalée de 120h).
   - **Indicateur synthétique `q_smart_expected` / `k_smart_expected` :**
     - *Jour normal :* Suit la tendance récente des 3 dernières semaines (`weekly_mean_3w`).
     - *Jour de pont (9–10 nov 2026) :* Applique le coefficient d'abattement du pont de 2025 (`ratio_bridge_1d_a1`) modulé par `bridge_intensity` ($1 - w \cdot (1 - \text{ratio}_{A-1})$) afin de ne pas surestimer la baisse sur un pont de 2 jours.
     - *Armistice (11 nov 2026) :* Applique au niveau de base 2026 le ratio heure par heure observé le `2025-11-11` (`ratio_armistice_a1`), capturant à la fois le flux résiduel des cortèges officiels aux Champs-Élysées (`08h00–13h00`, $q \in [12, 81]$) et le pic de réouverture de `14h00` ($q \approx 600, k \approx 16\%$).
5. **Sorties générées :**
   - `data/processed/features_train.csv` : Jeu d'entraînement complet sans `NaN`.
   - `data/processed/features_test_nov7_11.csv` : Les 360 lignes cibles (`2026-11-07 00:00` au `2026-11-11 23:00`).