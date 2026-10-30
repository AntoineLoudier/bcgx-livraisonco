# 📚 Documentation Technique — Socle de Données 

Ce document détaille les choix techniques et l'implémentation du pipeline de données réalisé pour le cas LivraisonCo.

## 1. Module d'Acquisition (`src/data_loader.py`)
* **Trafic (`download_paris_traffic`) :** Utilise l'endpoint d'export direct CSV de l'API Open Data Paris v2.1 (`comptages-routiers-permanents`) avec un filtre ODSQL (`libelle IN (...)`) pour contourner la limite de 10 000 lignes. Récupère l'historique depuis le 1er janvier 2025 pour les 3 rues cibles. L'identifiant `iu_ac` est systématiquement aliasé en `id_arc_trafic`.
* **Météo (`download_paris_weather`) :** Interroge l'API historique gratuite **Open-Meteo** (`/v1/archive`) aux coordonnées de Paris pour extraire `temperature_2m`, `precipitation`, `rain`, et `wind_speed_10m` au pas horaire.

## 2. Module de Prétraitement (`src/preprocessing.py`)
Le pipeline exécute 6 étapes rigoureuses pour préparer les données de ML :
1. **Standardisation temporelle :** Conversion des horodatages UTC vers le fuseau `Europe/Paris`, gestion des doublons liés au passage à l'heure d'hiver (moyenne des deux heures `02:00`).
2. **Nettoyage physique (Diagramme Fondamental) :** Suppression des valeurs aberrantes ($q \notin [0, 8000]$, $k \notin [0, 100]$) et conversion en `NaN`.
3. **Détection des faux zéros :** En journée (7h-22h), si une voie ouverte affiche simultanément $q=0$ et $k=0$, c'est une panne de capteur $\rightarrow$ conversion en `NaN`.
4. **Grille horaire continue :** Reconstruction d'un pas strict de `1h` sans trou temporel par capteur pour permettre l'application correcte des fonctions de décalage (`lags`).
5. **Imputation hybride sans fuite du futur (Zero Data Leakage) :**
   * Micro-coupures ($\le 3$ heures) : Interpolation linéaire.
   * Pannes longues ($> 3$ heures) : Moyenne causale des 3 semaines passées à la même heure (`shift(168)`, `shift(336)`, `shift(504)`).
   * Colonnes indicateurs `is_imputed_q` / `is_imputed_k` ajoutées pour tracer les valeurs reconstruites.
6. **Double sortie générée dans `data/processed/` :**
   * `traffic_clean_by_arc.csv` (maille capteur fin).
   * `traffic_clean_by_street.csv` (maille rue agrégée avec débit moyen, débit total, taux moyen et goulot d'étranglement $k_{\max}$).