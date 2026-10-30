# 📚 Documentation Technique — Socle de Données

Ce document détaille les choix techniques et l'implémentation du pipeline de données réalisé pour le cas LivraisonCo.

## 1. Module d'Acquisition (`src/data_loader.py`)
- **Trafic (`download_paris_traffic`) :** Utilise l'endpoint d'export direct CSV de l'API Open Data Paris v2.1 (`comptages-routiers-permanents`) avec un filtre ODSQL strict sur 3 couples de nœuds amont/aval spécifiques : `Av_Champs_Elysees-Washington` / `Av_Champs_Elysees-Berri` ; `Lecourbe-Convention` / `Convention-Blomet` ; `Sts_Peres-Voltaire` / `Sts_Peres-Universite`. Récupère l'historique depuis le 1er janvier 2023. La colonne cible finale est formattée pour correspondre au test BCG X (`arc` : `Champs-Elysées`, `Convention`, `Saint-Pères`).
- **Météo (`download_paris_weather`) :** Interroge l'API historique gratuite **Open-Meteo** (`/v1/archive`) aux coordonnées de Paris pour l'historique jusqu'à J-5, combinée à l'API Forecast (`/v1/forecast`) pour les 5 derniers jours et les 16 jours futurs, garantissant l'absence de trou temporel. Extrait `temperature_2m`, `precipitation`, `rain`, et `wind_speed_10m` au pas horaire.

## 2. Module de Prétraitement (`src/preprocessing.py`)
Le pipeline exécute 5 étapes rigoureuses pour préparer les données de ML :
1. **Standardisation temporelle :** Conversion des horodatages UTC vers le fuseau `Europe/Paris`, gestion des doublons liés au passage à l'heure d'hiver (moyenne des deux heures `02:00`).
2. **Nettoyage physique (Diagramme Fondamental) :** Suppression des valeurs aberrantes ($q \notin [0, 8000]$, $k \notin [0, 100]$) et conversion en `NaN`.
3. **Détection des faux zéros :** En journée (7h-22h), si une voie ouverte affiche simultanément $q=0$ et $k=0$, c'est une panne de capteur $\rightarrow$ conversion en `NaN`.
4. **Grille horaire continue :** Reconstruction d'un pas strict de `1h` sans trou temporel pour les 3 arcs cibles afin de permettre l'application correcte des fonctions de décalage (`lags`).
5. **Imputation hybride sans fuite du futur (Zero Data Leakage) :**
   * Micro-coupures ($\le 3$ heures) : Interpolation linéaire.
   * Pannes longues ($> 3$ heures) : Moyenne causale des 3 semaines passées à la même heure (`shift(168)`, `shift(336)`, `shift(504)`).
   * Colonnes indicateurs `is_imputed_q` / `is_imputed_k` ajoutées pour tracer les valeurs reconstruites.
   * **Sortie unique :** Génère le fichier final `data/processed/traffic_clean.csv` fusionné avec la météo.