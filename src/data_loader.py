import os
from datetime import datetime, timedelta
import requests
import pandas as pd

RAW_DATA_DIR = os.path.join("data", "raw")
PARIS_COORDS = {"latitude": 48.8566, "longitude": 2.3522}

TARGET_ARCS = [
    {
        "arc_name": "Champs-Elysées",
        "amont": "Av_Champs_Elysees-Washington",
        "aval": "Av_Champs_Elysees-Berri",
    },
    {
        "arc_name": "Convention",
        "amont": "Lecourbe-Convention",
        "aval": "Convention-Blomet",
    },
    {
        "arc_name": "Saint-Pères",
        "amont": "Sts_Peres-Voltaire",
        "aval": "Sts_Peres-Universite",
    },
]


def download_paris_traffic(start_date: str = "2023-01-01") -> pd.DataFrame:
    """Télécharge les données des 3 arcs exacts (filtrés par noeud amont et aval)."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    output_path = os.path.join(RAW_DATA_DIR, "traffic_raw.csv")

    # Construction du filtre ODSQL exact sur les couples (noeud amont, noeud aval)
    arc_conditions = [
        f"(libelle_nd_amont = '{a['amont']}' AND libelle_nd_aval = '{a['aval']}')"
        for a in TARGET_ARCS
    ]
    where_clause = f"({' OR '.join(arc_conditions)}) AND t_1h >= '{start_date}T00:00:00'"

    params = {
        "where": where_clause,
        "delimiter": ";",
        "select": "iu_ac,libelle,t_1h,q,k,etat_barre,libelle_nd_amont,libelle_nd_aval",
    }

    url = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents/exports/csv"
    print(f"Téléchargement des 3 arcs cibles depuis le {start_date}...")

    with requests.get(url, params=params, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    df = pd.read_csv(output_path, sep=";")

    # Mapping vers le nom exact attendu par la fonction de test BCG X 
    amont_to_name = {a["amont"]: a["arc_name"] for a in TARGET_ARCS}
    df["arc"] = df["libelle_nd_amont"].map(amont_to_name)
    df.to_csv(output_path, sep=";", index=False)

    print(f"Trafic sauvegardé : {len(df)} lignes -> {output_path}")
    print("Répartition par arc :\n", df["arc"].value_counts())
    return df


def download_paris_weather(start_date: str = "2023-01-01") -> pd.DataFrame:
    """Combine l'historique météo (Archive) et les prévisions (Forecast) jusqu'à J+14."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    output_path = os.path.join(RAW_DATA_DIR, "weather_raw.csv")
    hourly_vars = ["temperature_2m", "precipitation", "rain", "wind_speed_10m"]

    # 1. Météo historique (Archive jusqu'à J-5)
    archive_end = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    params_hist = {
        **PARIS_COORDS,
        "start_date": start_date,
        "end_date": archive_end,
        "hourly": hourly_vars,
        "timezone": "Europe/Paris",
    }
    r_hist = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params_hist, timeout=30)
    r_hist.raise_for_status()
    df_hist = pd.DataFrame(r_hist.json()["hourly"]).rename(columns={"time": "t_1h"})

    # 2. Météo récente (5 derniers jours) + Prévisions futures (jusqu'à 16 jours, couvre 7-11 nov)
    params_fcst = {
        **PARIS_COORDS,
        "past_days": 7,
        "forecast_days": 16,
        "hourly": hourly_vars,
        "timezone": "Europe/Paris",
    }
    r_fcst = requests.get("https://api.open-meteo.com/v1/forecast", params=params_fcst, timeout=30)
    r_fcst.raise_for_status()
    df_fcst = pd.DataFrame(r_fcst.json()["hourly"]).rename(columns={"time": "t_1h"})

    # 3. Concaténation sans trou ni doublon (donne la priorité aux observations récentes)
    df_weather = pd.concat([df_hist, df_fcst], ignore_index=True)
    df_weather = df_weather.drop_duplicates(subset=["t_1h"], keep="last").sort_values("t_1h")
    df_weather.to_csv(output_path, index=False)

    print(f"Météo complète (Archive + Forecast) sauvegardée : {len(df_weather)} lignes -> {output_path}")
    return df_weather


if __name__ == "__main__":
    download_paris_traffic(start_date="2023-01-01")
    download_paris_weather(start_date="2023-01-01")