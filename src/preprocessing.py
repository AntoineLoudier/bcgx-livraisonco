import os
import numpy as np
import pandas as pd

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")


def load_and_standardize_dates(traffic_path: str, weather_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les fichiers bruts et aligne les horodatages sur le fuseau 'Europe/Paris'."""
    print("1/5 - Chargement et alignement temporel (Europe/Paris)...")
    df_traffic = pd.read_csv(traffic_path, sep=";")
    df_weather = pd.read_csv(weather_path)

    df_traffic["t_1h"] = (
        pd.to_datetime(df_traffic["t_1h"], utc=True)
        .dt.tz_convert("Europe/Paris")
        .dt.tz_localize(None)
    )
    df_weather["t_1h"] = pd.to_datetime(df_weather["t_1h"])
    df_weather = df_weather.groupby("t_1h", as_index=False).mean(numeric_only=True)

    return df_traffic, df_weather


def clean_physical_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les valeurs physiquement impossibles et convertit les faux zéros diurnes en NaN."""
    print("2/5 - Nettoyage des anomalies physiques et détection des faux zéros...")
    df = df.copy()

    df.loc[(df["q"] < 0) | (df["q"] > 8000), "q"] = np.nan
    df.loc[(df["k"] < 0) | (df["k"] > 100), "k"] = np.nan

    closed_states = ["barré", "barre", "fermé", "ferme", "invalide", "1", "true"]
    is_closed = df["etat_barre"].astype(str).str.lower().isin(closed_states)
    df["is_road_closed"] = is_closed.astype(int)

    hour = df["t_1h"].dt.hour
    is_daytime = (hour >= 7) & (hour <= 22)
    is_fake_zero = is_daytime & (~is_closed) & (df["q"] == 0) & (df["k"] == 0)

    df.loc[is_fake_zero, ["q", "k"]] = np.nan
    print(f"   -> {is_fake_zero.sum()} faux zéros diurnes convertis en NaN.")
    return df


def build_complete_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruit une grille horaire continue (1h) pour chacun des 3 arcs officiels."""
    print("3/5 - Reconstruction de la grille horaire complète pour les 3 arcs...")
    df_agg = (
        df.groupby(["arc", "t_1h"], as_index=False)
        .agg({"q": "mean", "k": "mean", "is_road_closed": "max"})
    )

    full_time_range = pd.date_range(start=df_agg["t_1h"].min(), end=df_agg["t_1h"].max(), freq="1h")
    arcs = ["Champs-Elysées", "Convention", "Saint-Pères"]

    full_grid = pd.MultiIndex.from_product([arcs, full_time_range], names=["arc", "t_1h"]).to_frame().reset_index(drop=True)
    df_complete = pd.merge(full_grid, df_agg, on=["arc", "t_1h"], how="left")
    df_complete["is_road_closed"] = df_complete["is_road_closed"].fillna(0).astype(int)

    return df_complete.sort_values(["arc", "t_1h"]).reset_index(drop=True)


def impute_missing_values_no_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Impute les NaN sans fuite du futur (interpolation <=3h, puis moyenne J-7/J-14/J-21)."""
    print("4/5 - Imputation hybride des valeurs manquantes (zéro Data Leakage)...")
    df = df.copy()
    df["is_imputed_q"] = df["q"].isna().astype(int)
    df["is_imputed_k"] = df["k"].isna().astype(int)

    for col in ["q", "k"]:
        df[col] = df.groupby("arc")[col].transform(lambda s: s.interpolate(method="linear", limit=3))

        lag_1w = df.groupby("arc")[col].shift(24 * 7)
        lag_2w = df.groupby("arc")[col].shift(24 * 14)
        lag_3w = df.groupby("arc")[col].shift(24 * 21)
        historical_mean = pd.concat([lag_1w, lag_2w, lag_3w], axis=1).mean(axis=1)
        df[col] = df[col].fillna(historical_mean)

        dow = df["t_1h"].dt.dayofweek
        hour = df["t_1h"].dt.hour
        fallback_median = df.groupby(["arc", dow, hour])[col].transform("median")
        df[col] = df[col].fillna(fallback_median)

    return df


def run_preprocessing_pipeline() -> None:
    """Exécute le pipeline complet et sauvegarde le dataset propre des 3 arcs."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    traffic_path = os.path.join(RAW_DATA_DIR, "traffic_raw.csv")
    weather_path = os.path.join(RAW_DATA_DIR, "weather_raw.csv")

    df_traffic, df_weather = load_and_standardize_dates(traffic_path, weather_path)
    df_clean = clean_physical_anomalies(df_traffic)
    df_grid = build_complete_hourly_grid(df_clean)
    df_imputed = impute_missing_values_no_leakage(df_grid)

    print("5/5 - Fusion météo et sauvegarde finale...")
    df_final = pd.merge(df_imputed, df_weather, on="t_1h", how="left")
    output_file = os.path.join(PROCESSED_DATA_DIR, "traffic_clean.csv")
    df_final.to_csv(output_file, index=False)
    print(f"Terminé ! Dataset prêt ({len(df_final)} lignes) -> {output_file}")


if __name__ == "__main__":
    run_preprocessing_pipeline()