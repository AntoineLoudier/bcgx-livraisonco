import os
from datetime import datetime, timedelta
import requests
import pandas as pd

# Configuration des chemins et constantes
RAW_DATA_DIR = os.path.join("data", "raw")
TARGET_STREETS = ["Convention", "Av_des_Champs_Elysees", "Sts_Peres"]
PARIS_COORDS = {"latitude": 48.8566, "longitude": 2.3522}


def download_paris_traffic(streets: list[str], start_date: str = "2025-01-01") -> pd.DataFrame:
    """Télécharge l'historique horaire de trafic depuis Open Data Paris pour les rues cibles."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    output_path = os.path.join(RAW_DATA_DIR, "traffic_raw.csv")

    streets_query = ", ".join(f"'{s}'" for s in streets)
    params = {
        "where": f"libelle IN ({streets_query}) AND t_1h >= '{start_date}T00:00:00'",
        "delimiter": ";",
        "select": "iu_ac,libelle,t_1h,q,k,etat_barre,libelle_nd_amont,libelle_nd_aval",
    }

    url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents/exports/csv"
    print(f"Téléchargement du trafic ({', '.join(streets)}) depuis le {start_date}...")

    with requests.get(url, params=params, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    df = pd.read_csv(output_path, sep=";")
    df["id_arc_trafic"] = df["iu_ac"]
    df.to_csv(output_path, sep=";", index=False)

    print(f"Trafic sauvegardé : {len(df)} lignes -> {output_path}")
    return df


def download_paris_weather(start_date: str = "2025-01-01", end_date: str | None = None) -> pd.DataFrame:
    """Télécharge l'historique météo horaire de Paris via Open-Meteo."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    output_path = os.path.join(RAW_DATA_DIR, "weather_raw.csv")

    if end_date is None:
        end_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    params = {
        **PARIS_COORDS,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "precipitation", "rain", "wind_speed_10m"],
        "timezone": "Europe/Paris",
    }

    url = "https://archive-api.open-meteo.com/v1/archive"
    print(f"Téléchargement de la météo du {start_date} au {end_date}...")

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    df_weather = pd.DataFrame(response.json()["hourly"]).rename(columns={"time": "t_1h"})
    df_weather.to_csv(output_path, index=False)

    print(f"Météo sauvegardée : {len(df_weather)} lignes -> {output_path}")
    return df_weather


if __name__ == "__main__":
    download_paris_traffic(TARGET_STREETS, start_date="2025-01-01")
    download_paris_weather(start_date="2025-01-01")