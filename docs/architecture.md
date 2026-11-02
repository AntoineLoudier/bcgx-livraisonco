# 📚 Documentation Technique Exhaustive — Projet LivraisonCo (BCG X)

Ce document constitue la référence technique complète du pipeline de Machine Learning développé pour **LivraisonCo**. Il détaille l'architecture logicielle, les règles physiques, les formules mathématiques et les garanties strictes d'absence de fuite du futur (*Zero Data Leakage*) de chaque script du dépôt.

---

## 📂 1. Vue d'Ensemble de l'Architecture

```text
bcgx-livraisonco/
├── data/
│   ├── raw/
│   │   ├── traffic_raw.csv                  # Export brut Open Data Paris (3 arcs cibles)
│   │   └── weather_raw.csv                  # Météo horaire Paris (Archive + Forecast J+16)
│   └── processed/
│       ├── traffic_clean.csv                # Grille horaire continue nettoyée et imputée causalement
│       ├── features_train.csv               # Jeu d'entraînement (100% mesures réelles, 0 NaN)
│       ├── features_test_nov7_11.csv        # Jeu de test officiel (360 lignes : 07/11 au 11/11/2026)
│       └── cv_predictions_oof.csv           # Prédictions Out-Of-Fold de validation croisée
├── src/
│   ├── __init__.py
│   ├── data_loader.py                       # Acquisition API Open Data Paris (ODSQL) & Open-Meteo
│   ├── preprocessing.py                     # Nettoyage physique, grille 1h et imputation causale
│   ├── features.py                          # Feature Engineering temporel, ponts 1j/2j et ancrage SNCF A-1
│   ├── evaluate.py                          # Validation croisée multi-époques (120h) et export officiel BCG X
│   └── model.py                             # Architecture hybride Résiduelle (50% LightGBM + 50% XGBoost)
├── models/
│   └── ensemble_models_q_k.joblib           # Dictionnaire sérialisé des 6 modèles finaux entraînés
├── reports/
│   ├── RAPPORT_INTERPRETABILITE.md          # Rapport automatique (RMSE, MAE, Gain %, Top Features & SHAP)
│   ├── rmse_by_arc_and_outcome.csv          # Synthèse des performances par arc et par cible
│   ├── rmse_by_fold_detail.csv              # Détail des scores par pli de validation de 5 jours
│   └── feature_importances_by_arc_outcome.csv # Importances LightGBM, XGBoost et valeurs SHAP
├── main_train.py                            # Orchestrateur d'évaluation CV, d'entraînement et de reporting
├── main_predict.py                          # Orchestrateur d'inférence finale (génère output_<groupe>.csv)
├── requirements.txt                         # Dépendances Python verrouillées
└── README.md                                # Présentation générale du projet
```

---

## 🌐 2. Module d'Acquisition des Données (`src/data_loader.py`)

### 2.1. Acquisition du Trafic Routier (`download_paris_traffic`)
* **Source :** API Open Data Paris Explore v2.1 (`comptages-routiers-permanents`), qui expose une fenêtre glissante des 13 derniers mois de comptages issus des boucles électromagnétiques de la Ville de Paris.
* **Contournement de la limite d'API :** Utilise l'endpoint de streaming `/exports/csv` couplé à une clause de filtrage **ODSQL (`where`)** exécutée côté serveur pour récupérer l'intégralité des enregistrements sans subir la pagination standard limitée à $10\,000$ lignes.
* **Ciblage exact des 3 arcs du PoC (Slide 8 du Brief) :** Le filtrage ne s'effectue pas sur la rue entière mais sur un **couple orienté `(libelle_nd_amont, libelle_nd_aval)`** définissant un tronçon unique et un sens de circulation précis :
  1. **`Champs-Elysées` :** Nœud amont `Av_Champs_Elysees-Washington` (ou variante de casse `AV_Champs_Elysees-Washington`) $\rightarrow$ Nœud aval `Av_Champs_Elysees-Berri` (`AV_Champs_Elysees-Berri`).
  2. **`Convention` :** Nœud amont `Lecourbe-Convention` $\rightarrow$ Nœud aval `Convention-Blomet`.
  3. **`Saint-Pères` :** Nœud amont `Sts_Peres-Voltaire` $\rightarrow$ Nœud aval `Sts_Peres-Universite`.
* **Variables extraites :**
  * `t_1h` : Horodatage de la mesure (pas horaire).
  * `q` : **Débit horaire** (nombre de véhicules ayant franchi la boucle sur l'heure écoulée, en $\text{véh/h}$).
  * `k` : **Taux d'occupation** (pourcentage du temps, sur une heure, pendant lequel la boucle magnétique a détecté la présence d'un véhicule au-dessus d'elle, en $\%$).
  * `etat_barre` : Indicateur de fermeture administrative ou physique de la voie.
* **Diagnostic intégré :** À la fin du téléchargement, le script audite le nombre de mesures valides par arc. Cet audit a mis en évidence que le capteur `Champs-Elysées` (`Washington -> Berri`) a cessé de télétransmettre ses mesures après le **`2025-11-19 11:00`** ($1\,143$ mesures réelles d'automne), tandis que `Convention` ($8\,859$ mesures) et `Saint-Pères` ($8\,816$ mesures) transmettent en continu jusqu'en novembre 2026.

### 2.2. Acquisition Météorologique Hybride (`download_paris_weather`)
* **Problème résolu :** L'API historique d'Open-Meteo (`/v1/archive`) possède un délai de consolidation de 5 jours, ce qui créerait un trou de données sur la semaine précédant le rendu du 6 novembre et ne fournirait aucune prévision pour la période cible du **7 au 11 novembre 2026**.
* **Solution implémentée :** Le script interroge successivement deux endpoints aux coordonnées exactes du centre de Paris ($\text{lat} = 48.8566, \text{lon} = 2.3522$, fuseau `Europe/Paris`) :
  1. **`/v1/archive` :** De `start_date` jusqu'à $J-5$.
  2. **`/v1/forecast` :** Avec `past_days=7` et `forecast_days=16`, couvrant la jonction récente ainsi que l'intégralité de la fenêtre de test du **7 au 11 novembre 2026**.
* **Variables extraites :** `temperature_2m` ($^\circ\text{C}$), `precipitation` ($\text{mm}$), `rain` ($\text{mm}$), `wind_speed_10m` ($\text{km/h}$).
* **Déduplication :** Concaténation et suppression des doublons (`drop_duplicates(subset=["t_1h"], keep="last")`) pour privilégier les observations consolidées les plus récentes.

---

## 🧹 3. Module de Prétraitement Causal (`src/preprocessing.py`)

Ce module transforme les données brutes en une série temporelle continue au pas strict de `1h` (`data/processed/traffic_clean.csv`) tout en respectant une **flèche du temps stricte** (aucune opération n'utilise une information de $t+1$ pour corriger l'instant $t$).

### 3.1. Standardisation Temporelle (`load_and_standardize_dates`)
* Convertit tous les horodatages UTC d'Open Data Paris vers le fuseau local `Europe/Paris` (`.dt.tz_convert("Europe/Paris").dt.tz_localize(None)`).
* Gère le changement d'heure d'hiver (fin octobre, où l'heure `02:00` apparaît deux fois) par agrégation moyenne sur `(arc, t_1h)`.

### 3.2. Nettoyage Physique et Détection des Pannes (`clean_physical_anomalies`)
1. **Bornes du diagramme fondamental :** Toute valeur de débit $q \notin [0, 8000]\text{ véh/h}$ ou de taux d'occupation $k \notin [0, 100]\%$ est physiquement impossible et convertie en `NaN`.
2. **Décodage des fermetures de voie (`is_road_closed`) :** Convertit la colonne `etat_barre` en indicateur binaire (`1` si la voie est barrée/fermée/invalide, `0` sinon). Permet notamment d'identifier la fermeture officielle des Champs-Élysées lors de la cérémonie du 11 novembre (`08h00–14h00`).
3. **Élimination des « Faux Zéros » diurnes :** En plein jour (`07h00` $\le \text{heure} \le$ `22h00`), si une artère majeure ouverte à la circulation (`is_road_closed == 0`) affiche simultanément $q = 0$ et $k = 0$, il est physiquement exclu qu'aucun véhicule ne soit passé en 60 minutes. Il s'agit d'un décrochage électrique du capteur : le couple $(q, k)$ est converti en `NaN`.

### 3.3. Grille Horaire Continue (`build_complete_hourly_grid`)
* Génère le produit cartésien complet `MultiIndex.from_product([arcs, full_time_range])` au pas de `1h`.
* Garantit qu'un décalage `.shift(168)` dans Pandas recule toujours exactement de $168\text{ heures}$ ($7\text{ jours}$), même si le fichier brut comportait des heures sautées.

### 3.4. Imputation Hybride 100 % Causale (`impute_missing_values_no_leakage`)
> **Pourquoi imputer si l'entraînement filtre les valeurs imputées ?**  
> L'imputation ne sert **pas** à créer de fausses cibles d'entraînement : elle sert à garantir que les **variables explicatives passées ($X$, comme les lags $J-7, J-14, J-21$)** ne contiennent pas de `NaN` lorsqu'une micro-panne a eu lieu une ou deux semaines avant la date à prédire.

1. **Traçabilité stricte :** Avant toute modification, création des drapeaux binaires `is_imputed_q` et `is_imputed_k` (`1` si la mesure originale était absente ou invalide, `0` si c'est une vraie mesure certifiée).
2. **Niveau 1 — Micro-coupures ($\le 3\text{ heures}$) :** Application d'un *Forward-Fill* limité à 3 heures (`s.ffill(limit=3)`). Contrairement à une interpolation linéaire qui utiliserait $t+1$ (fuite du futur), le `ffill` propage uniquement la dernière mesure connue en $t-1$.
3. **Niveau 2 — Pannes moyennes ($> 3\text{ heures}$) :** Remplacement par la moyenne causale des 3 semaines précédentes exactement à la même heure et au même jour de la semaine :
   $$\hat{y}_{\text{impute}}(t) = \frac{y(t - 168\text{h}) + y(t - 336\text{h}) + y(t - 504\text{h})}{3}$$
4. **Niveau 3 — Secours causal (`expanding().mean()`) :** Si les 3 semaines précédentes étaient également manquantes, utilise la moyenne cumulative passée (`expanding`) par `(arc, dayofweek, hour)`, qui ne regarde que les dates antérieures ou égales à $t$.

---

## ⚙️ 4. Module de Feature Engineering (`src/features.py`)

Ce script construit l'ensemble de la matrice de variables explicatives et sépare l'historique d'entraînement (`features_train.csv`) des 360 lignes cibles (`features_test_nov7_11.csv`).

### 4.1. Extension à la Grille de Test (`extend_grid_to_test_period`)
* Prolonge la grille horaire des 3 arcs jusqu'au **`2026-11-11 23:00:00`**.
* **Projection protocolaire du 11 Novembre :** Conformément à l'observation empirique du `2025-11-11` sur les Champs-Élysées, force `is_road_closed = 1` le `2026-11-11` entre `08h00` et `14h00` sur l'arc `Champs-Elysées`.
* Fusionne la météo complète (historique + prévisions horaires du 7 au 11 novembre).

### 4.2. Calendrier, Vacances et Modélisation des Ponts (`add_calendar_and_event_features`)
1. **Encodage cyclique trigonométrique :** Préserve la continuité entre `23h00` et `00h00` ainsi qu'entre le dimanche et le lundi :
   $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
   $$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{dayofweek}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{dayofweek}}{7}\right)$$
2. **Séparation Petites Vacances vs Vacances d'Été (Zone C — Paris) :**
   * En août, le trafic parisien chute de près de $-47\%$ sur `Saint-Pères`, contre seulement $-7\%$ à $-15\%$ pendant les petites vacances scolaires (Toussaint, Noël, Hiver, Printemps).
   * Le code crée donc deux variables distinctes : `is_school_holiday` (petites vacances scolaires hors juillet–août) et `is_summer_holiday` (juillet–août).
3. **Modélisation continue des Ponts : Pont de 1 jour vs Pont de 2 jours / Viaduc (`bridge_intensity`) :**
   * **Constat métier :** En 2025, le 11 novembre tombait un mardi, faisant du lundi 10 novembre 2025 un **pont de 1 jour** (très forte baisse de trafic). En 2026, le 11 novembre tombe un **mercredi** : faire le pont (« viaduc ») nécessite de poser 2 jours de congés (lundi 9 et mardi 10 novembre), ce qui dilue l'effet.
   * **Échelle d'intensité implémentée :**
     * `bridge_intensity = 1.00` : Vrai pont de 1 jour (Lundi veille d'un Mardi férié ou Vendredi lendemain d'un Jeudi férié, ex. `2025-11-10`, Ascension `2026-05-15`, `2026-07-13`).
     * `bridge_intensity = 0.50` : Mardi veille ou Jeudi lendemain d'un Mercredi férié (ex. **Mardi 10 novembre 2026**).
     * `bridge_intensity = 0.25` : Lundi avant-veille ou Vendredi surlendemain d'un Mercredi férié (ex. **Lundi 9 novembre 2026**, captant le léger surplus de télétravail).
4. **Événements spécifiques aux Champs-Élysées :**
   * `is_paris_respire` : Piétonnisation du 1er dimanche de chaque mois de `10h00` à `18h00` sur `Champs-Elysées`.
   * `is_armistice` & `is_armistice_parade_hours` : Cible le 11 novembre et le créneau `08h00–13h00` du défilé militaire aux Champs-Élysées.
5. **Régime de jour (`day_regime`) :** Variable catégorielle synthétique (`0` = Ouvré normal, `1` = Samedi, `2` = Dimanche, `3` = Pont, `4` = Férié).

### 4.3. Interactions Météo $\times$ Heure de Pointe (`add_weather_interaction_features`)
* `is_rush_hour` : `1` sur les créneaux `[07h–10h]` et `[17h–20h]`, `0` sinon.
* `is_raining` : Indicateur de pluie significative (`rain > 0.2 mm`).
* `rain_in_rush_hour = rain * is_rush_hour` : Capte l'effet non-linéaire d'une averse tombant pendant l'heure de pointe sur la saturation $k$.
* `temp_diff_24h` : Écart de température sur 24 heures ($\Delta T = T_t - T_{t-24\text{h}}$), responsable du report modal vélo/deux-roues vers l'automobile.

### 4.4. Clean Lags, Profils P90 et Ancrage SNCF $A-1$ (`add_causal_lags_and_sncf_patterns`)
Pour chaque cible $y \in \{q, k\}$ :
1. **Lags hebdomadaires en cascade causale ($J-7, J-14, J-21, J-28$) :**
   * Calcul de `lag_168h` ($J-7$), `lag_336h` ($J-14$), `lag_504h` ($J-21$) et `lag_672h` ($J-28$).
   * **Cascade anti-NaN :** Si le code est exécuté quelques jours avant le 6 novembre et que les dernières heures de début novembre ($J-7$ des 9–11 novembre) ne sont pas encore en base, `lag_168h` bascule automatiquement sur `lag_336h` ($J-14$) sans jamais utiliser `.bfill()`.
2. **Mécanisme des « Clean Lags » (Dépollution des jours fériés passés) :**
   * Si le jour à prédire $t$ est un jour normal (ni férié, ni pont, ni fermé), mais que $t - 7\text{j}$ tombait sur un jour anormal (ex. le dimanche 1er novembre férié + Paris Respire qui polluerait la prédiction du dimanche 8 novembre !), le code substitue automatiquement `lag_336h` ($J-14$) à `lag_168h`.
3. **Ancrage Automne 2025 pour le capteur arrêté des Champs-Élysées :**
   * Pour toutes les dates postérieures au `2025-12-15` sur `Champs-Elysées`, remplace les lags (qui seraient figés sur la médiane d'imputation) par les 3 dernières semaines réelles d'octobre–novembre 2025 (`t_1h <= 2025-11-19`) alignées par `(dayofweek, hour)`.
4. **Moyenne pondérée réactive (`weekly_mean_3w`) et Volatilité (`weekly_std_3w`) :**
   $$y_{\text{weekly\_mean\_3w}}(t) = 0,60 \cdot y_{\text{clean}}(t-168) + 0,25 \cdot y_{\text{clean}}(t-336) + 0,15 \cdot y_{\text{clean}}(t-504)$$
5. **Profil historique causal (`regime_hist_mean`) et Centile de congestion (`hour_p90`) :**
   * Calculés uniquement sur les mesures réelles (`is_imputed == 0`) préalablement décalées de 5 jours (`shift(120)`) via une fenêtre cumulative `expanding()`.
   * `hour_p90` donne le 90e centile historique de chaque heure, indiquant aux arbres de décision l'amplitude maximale des embouteillages sur chaque artère.
6. **Construction de l'ancrage métier `smart_expected` (Zéro fuite sur $A-1$) :**
   * Extraction des profils horaires d'abattement observés lors du pont du `2025-11-10` ($r_{\text{bridge}, A-1}(h)$) et de l'Armistice du `2025-11-11` ($r_{\text{armistice}, A-1}(h)$).
   * **Garantie anti-leakage :** Ces deux ratios ne sont appliqués **qu'aux dates strictement postérieures au 12 novembre 2025** (`t_1h >= 2025-11-12`, donc en 2026) :
     * *Jour normal :* $y_{\text{smart}}(t) = y_{\text{weekly\_mean\_3w}}(t)$
     * *Pont en 2026 ($w = \text{bridge\_intensity} \in \{0.25, 0.50, 1.00\}$) :*
       $$y_{\text{smart}}(t) = y_{\text{weekly\_mean\_3w}}(t) \times \Big(1 - w \cdot \big(1 - r_{\text{bridge}, A-1}(h)\big)\Big)$$
     * *Armistice 2026 (`2026-11-11`) :*
       $$y_{\text{smart}}(t) = y_{\text{weekly\_mean\_3w}}(t) \times r_{\text{armistice}, A-1}(h)$$
       Reproduit fidèlement le flux résiduel des cortèges officiels aux Champs-Élysées (`08h–13h`, $q \in [12, 81]\text{ véh/h}$) puis l'appel d'air de la réouverture à `14h00` ($q \approx 601\text{ véh/h}, k \approx 16\%$).

### 4.5. Filtrage Strict de `features_train.csv` (`run_feature_pipeline`)
* Le fichier `features_train.csv` exporté ne conserve **que les lignes où `is_imputed_q == 0` et `is_imputed_k == 0`** (après les 7 premiers jours d'initialisation). Aucune donnée reconstruite n'entre jamais dans l'entraînement ni dans l'évaluation.

---

## 📏 5. Module d'Évaluation & d'Export Officiel (`src/evaluate.py`)

1. **Métrique officielle (`compute_rmse`) :** Calcule la *Root Mean Squared Error* :
   $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2}$$
2. **Validation Croisée Temporelle Multi-Époques (`get_temporal_cv_splits`) :**
   * Au lieu de ne valider que sur la dernière semaine d'octobre, l'algorithme parcourt l'historique réel de chaque arc et sélectionne `n_splits = 4` fenêtres disjointes de **5 jours consécutifs ($120\text{ heures}$)** comportant au moins $85\%$ de mesures réelles ($\ge 102\text{h}$ sur $120\text{h}$), réparties sur différentes saisons (automne, hiver, printemps, été, rentrée).
   * Pour chaque pli, l'entraînement est restreint au passé strict de la fenêtre (`t_1h < val_start`).
3. **Formatage et Assertions BCG X (`format_predictions_for_bcgx` & `test_format_and_export_output`) :**
   * Applique un bornage physique de sécurité (`np.clip`) et force les types exacts requis par Pandas (`object` pour `arc` et `datetime`, `float64` pour `debit_horaire` et `taux_occupation`).
   * Exécute à l'identique les **6 assertions officielles de la Slide 13 du brief BCG X** (noms des colonnes, types, format de date `"%Y-%m-%d %H:%M"`, orthographe exacte de `["Champs-Elysées", "Convention", "Saint-Pères"]`, dimension exacte de `360` lignes) et exporte `output_<nom_du_groupe>.csv` avec le séparateur `;`.

---

## 🧠 6. Module de Modélisation Hybride (`src/model.py`)

### 6.1. Variables Physiques Croisées (`add_cross_physical_features`)
Ajoute à la volée 4 indicateurs dynamiques reliant le débit $q$ et le taux d'occupation $k$ :
* **`congestion_index_168h`** $= \frac{k_{t-168}}{q_{t-168} + 10}$ : Ratio de saturation issu du diagramme fondamental. Permet de distinguer immédiatement un faible débit fluide de nuit ($k/q \approx 0$) d'un faible débit paralysé par un embouteillage ($k/q$ élevé).
* **`q_hour_slope_168h` & `k_hour_slope_168h`** $= y_{t-168} - y_{t-169}$ : Pente horaire mesurant la vitesse de montée ou de résorption de l'heure de pointe.
* **`holiday_x_rush`** $= \text{is\_school\_holiday} \times \text{is\_rush\_hour}$ : Interaction ciblant l'évaporation du trafic spécifiquement aux heures de pointe pendant les vacances scolaires.

### 6.2. Stratégie des 6 Modèles Spécialisés (`3 Arcs × 2 Cibles`)
* Entraîne **6 modèles indépendants** : un pour chaque couple `(arc, target)`.
* **Justification mathématique :** La RMSE étant quadratique, le débit moyen des Champs-Élysées ($\sim 741\text{ véh/h}$, écart-type élevé) génère des erreurs au carré 4 à 6 fois plus grandes que la rue de la Convention ($\sim 378\text{ véh/h}$). Séparer les modèles garantit que chaque artère optimise à 100 % sa propre fonction de perte.

### 6.3. Apprentissage Résiduel (`ResidualEnsembleModel`)
Au lieu de prédire directement $y$, chaque modèle d'arbre apprend le **résidu** par rapport à l'ancrage métier :
$$r_i = y_i - y_{\text{smart\_expected}, i}$$
* **Avantage décisif sur les événements rares :** Les hyperparamètres de régularisation des arbres (`min_child_samples = 20`) interdisent d'isoler un événement qui ne dure que 6 heures par an (comme le défilé du 11 novembre aux Champs-Élysées). Grâce à l'architecture résiduelle, `smart_expected` porte le choc événementiel exact, tandis que les arbres apprennent la correction météo et conjoncturelle $\hat{r}_i$ :
  $$\hat{y}_i = \text{clip}\Big(y_{\text{smart\_expected}, i} + \big(0,5 \cdot \hat{r}_{\text{LGBM}, i} + 0,5 \cdot \hat{r}_{\text{XGB}, i}\big), \; 0, \; y_{\max}\Big)$$

### 6.4. Régularisation Adaptative & Pondération Temporelle
* **Adaptation à la taille d'échantillon :** Sur `Champs-Elysées` ($1\,143$ heures réelles), la complexité des arbres est automatiquement bridée (`n_estimators = 120`, `max_depth = 3`, `num_leaves = 9`, `min_child_samples = 35`) pour interdire tout sur-apprentissage, contre `n_estimators = 400` et `max_depth = 4–5` sur `Convention` et `Saint-Pères` ($\sim 8\,800$ heures réelles).
* **Pondération des observations (`compute_sample_weights`) :** Croissance linéaire du poids avec la récence (`0.85` $\rightarrow$ `1.20`) et sur-pondération des heures diurnes (`x1.25` entre `08h00` et `20h00` sur le débit $q$), là où se concentre plus de $70\%$ de la MSE.

---

## 📊 7. Orchestration & Reporting d'Interprétabilité (`main_train.py` & `main_predict.py`)

### 7.1. Entraînement et Explicabilité SHAP (`main_train.py`)
L'exécution de `python main_train.py` réalise séquentiellement :
1. Le benchmark de validation croisée sur 4 fenêtres de 5 jours, comparant la **Baseline $J-7$**, l'**Ancrage SNCF seul** et le **Modèle Hybride Final** (en RMSE et MAE).
2. L'entraînement des 6 modèles finaux sur 100 % des mesures réelles et leur sauvegarde dans `models/ensemble_models_q_k.joblib`.
3. Le calcul des importances relatives (LightGBM et XGBoost en $\%$) et des **valeurs SHAP exactes (`TreeExplainer`)** exprimées en unités physiques ($\text{véh/h}$ pour $q$ et points de $\%$ pour $k$).
4. La génération automatique du rapport `reports/RAPPORT_INTERPRETABILITE.md` et des tableaux CSV associés.

### 7.2. Commande Unique pour le Rendu Final du 6 Novembre (`main_predict.py`)
Le jour du gel des données (avant le **6 novembre 18h00**), une seule commande exécute le rafraîchissement complet de bout en bout :

```bash
python main_predict.py --refresh-all --group <nom_de_votre_groupe>
```

Cette commande re-télécharge les dernières heures de trafic et les dernières prévisions météo du 7 au 11 novembre, reconstruit les features sans aucun `NaN`, ré-entraîne les 6 modèles finaux et génère le fichier **`output_<nom_de_votre_groupe>.csv`** certifié par la fonction de test officielle du BCG X.