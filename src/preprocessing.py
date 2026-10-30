import os
import numpy as np
import pandas as pd

# Configuration des chemins
RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")


def load_and_standardize_dates(traffic_path: str, weather_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les fichiers bruts et aligne les horodatages sur le fuseau 'Europe/Paris'."""
    print("1/6 - Chargement et alignement temporel (Europe/Paris)...")
    df_traffic = pd.read_csv(traffic_path, sep=";")
    df_weather = pd.read_csv(weather_path)

    # Sécurité : s'assurer que id_arc_trafic est bien présent à partir de iu_ac
    if "id_arc_trafic" not in df_traffic.columns and "iu_ac" in df_traffic.columns:
        df_traffic["id_arc_trafic"] = df_traffic["iu_ac"]

    # Trafic : lecture UTC -> conversion heure locale Paris -> suppression du tz pour jointure propre
    df_traffic["t_1h"] = (
        pd.to_datetime(df_traffic["t_1h"], utc=True)
        .dt.tz_convert("Europe/Paris")
        .dt.tz_localize(None)
    )

    # Météo : déjà en Europe/Paris, moyenne en cas de doublon (passage à l'heure d'hiver)
    df_weather["t_1h"] = pd.to_datetime(df_weather["t_1h"])
    df_weather = df_weather.groupby("t_1h", as_index=False).mean(numeric_only=True)

    return df_traffic, df_weather


def clean_physical_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les valeurs physiquement impossibles et convertit les faux zéros diurnes en NaN."""
    print("2/6 - Nettoyage des anomalies physiques et détection des faux zéros...")
    df = df.copy()

    # Bornes physiques du diagramme fondamental : q in [0, 8000] véh/h et k in [0, 100] %
    df.loc[(df["q"] < 0) | (df["q"] > 8000), "q"] = np.nan
    df.loc[(df["k"] < 0) | (df["k"] > 100), "k"] = np.nan

    # Identification des vraies fermetures de voies (travaux, événements)
    closed_states = ["barré", "barre", "fermé", "ferme", "invalide", "1", "true"]
    is_closed = df["etat_barre"].astype(str).str.lower().isin(closed_states)
    df["is_road_closed"] = is_closed.astype(int)

    # Entre 7h et 22h, si la voie est ouverte mais q=0 et k=0, c'est une panne capteur -> NaN
    hour = df["t_1h"].dt.hour
    is_daytime = (hour >= 7) & (hour <= 22)
    is_fake_zero = is_daytime & (~is_closed) & (df["q"] == 0) & (df["k"] == 0)

    df.loc[is_fake_zero, ["q", "k"]] = np.nan
    print(f"   -> {is_fake_zero.sum()} faux zéros diurnes convertis en NaN.")

    return df


def build_complete_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit une grille horaire continue (1h) sans trou temporel pour chaque capteur."""
    print("3/6 - Reconstruction de la grille horaire complète par capteur...")
    # Agrégation en cas de doublon horaire (ex. 02h00 au changement d'heure)
    df_agg = (
        df.groupby(["id_arc_trafic", "libelle", "t_1h"], as_index=False)
        .agg({
            "q": "mean",
            "k": "mean",
            "is_road_closed": "max",
            "libelle_nd_amont": "first",
            "libelle_nd_aval": "first",
        })
    )

    full_time_range = pd.date_range(start=df_agg["t_1h"].min(), end=df_agg["t_1h"].max(), freq="1h")
    arcs_meta = df_agg[["id_arc_trafic", "libelle", "libelle_nd_amont", "libelle_nd_aval"]].drop_duplicates("id_arc_trafic")

    # Produit cartésien (chaque capteur x chaque heure)
    grid_list = [
        pd.DataFrame({
            "id_arc_trafic": row["id_arc_trafic"],
            "libelle": row["libelle"],
            "libelle_nd_amont": row["libelle_nd_amont"],
            "libelle_nd_aval": row["libelle_nd_aval"],
            "t_1h": full_time_range,
        })
        for _, row in arcs_meta.iterrows()
    ]
    full_grid = pd.concat(grid_list, ignore_index=True)

    df_complete = pd.merge(
        full_grid,
        df_agg.drop(columns=["libelle_nd_amont", "libelle_nd_aval"]),
        on=["id_arc_trafic", "libelle", "t_1h"],
        how="left",
    )
    df_complete["is_road_closed"] = df_complete["is_road_closed"].fillna(0).astype(int)

    return df_complete.sort_values(["id_arc_trafic", "t_1h"]).reset_index(drop=True)


def impute_missing_values_no_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Impute les NaN sans fuite du futur (interpolation <=3h, puis moyenne J-7/J-14/J-21)."""
    print("4/6 - Imputation hybride des valeurs manquantes (zéro Data Leakage)...")
    df = df.copy()

    df["is_imputed_q"] = df["q"].isna().astype(int)
    df["is_imputed_k"] = df["k"].isna().astype(int)

    for col in ["q", "k"]:
        # 1. Micro-coupures (<= 3h) : interpolation linéaire
        df[col] = df.groupby("id_arc_trafic")[col].transform(
            lambda s: s.interpolate(method="linear", limit=3)
        )

        # 2. Pannes longues (> 3h) : moyenne des 3 semaines passées à la même heure
        lag_1w = df.groupby("id_arc_trafic")[col].shift(24 * 7)
        lag_2w = df.groupby("id_arc_trafic")[col].shift(24 * 14)
        lag_3w = df.groupby("id_arc_trafic")[col].shift(24 * 21)
        historical_mean = pd.concat([lag_1w, lag_2w, lag_3w], axis=1).mean(axis=1)
        df[col] = df[col].fillna(historical_mean)

        # 3. Fallback pour les 3 premières semaines de janvier 2025
        dow = df["t_1h"].dt.dayofweek
        hour = df["t_1h"].dt.hour
        fallback_median = df.groupby(["id_arc_trafic", dow, hour])[col].transform("median")
        df[col] = df[col].fillna(fallback_median)

    return df


def aggregate_by_street(df_arcs: pd.DataFrame) -> pd.DataFrame:
    """Agrège les données nettoyées à la maille de la rue ('libelle')."""
    print("5/6 - Agrégation à la maille Rue (libelle)...")
    return (
        df_arcs.groupby(["libelle", "t_1h"], as_index=False)
        .agg(
            q=("q", "mean"),
            q_total=("q", "sum"),
            k=("k", "mean"),
            k_max=("k", "max"),
            is_road_closed=("is_road_closed", "max"),
            ratio_imputed_q=("is_imputed_q", "mean"),
            n_active_sensors=("id_arc_trafic", "count"),
        )
    )


def merge_weather_safely(df: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """Fusionne la météo et gère le décalage de 5 jours de l'API Archive Open-Meteo."""
    merged = pd.merge(df, df_weather, on="t_1h", how="left")
    weather_cols = ["temperature_2m", "precipitation", "rain", "wind_speed_10m"]
    for col in weather_cols:
        if col in merged.columns:
            merged[col] = merged[col].ffill().bfill()
    return merged


def run_preprocessing_pipeline() -> None:
    """Exécute le pipeline complet et sauvegarde les jeux de données prêts pour le Feature Engineering."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    traffic_path = os.path.join(RAW_DATA_DIR, "traffic_raw.csv")
    weather_path = os.path.join(RAW_DATA_DIR, "weather_raw.csv")

    df_traffic, df_weather = load_and_standardize_dates(traffic_path, weather_path)
    df_clean = clean_physical_anomalies(df_traffic)
    df_grid = build_complete_hourly_grid(df_clean)
    df_imputed = impute_missing_values_no_leakage(df_grid)

    print("6/6 - Fusion météo et sauvegarde des fichiers finaux...")
    # Sauvegarde maille Capteur (Arc)
    df_arcs_final = merge_weather_safely(df_imputed, df_weather)
    df_arcs_final.to_csv(os.path.join(PROCESSED_DATA_DIR, "traffic_clean_by_arc.csv"), index=False)

    # Sauvegarde maille Rue (Street)
    df_street = aggregate_by_street(df_imputed)
    df_street_final = merge_weather_safely(df_street, df_weather)
    df_street_final.to_csv(os.path.join(PROCESSED_DATA_DIR, "traffic_clean_by_street.csv"), index=False)

    print(f"Pipeline terminé ! Fichiers disponibles dans : {PROCESSED_DATA_DIR}/")


if __name__ == "__main__":
    run_preprocessing_pipeline()