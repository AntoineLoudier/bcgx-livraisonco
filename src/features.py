import os
import numpy as np
import pandas as pd

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

# Fenêtre officielle à prédire pour le Livrable 1 (Slide 11 & 13 du Brief BCG X)
TEST_START = "2026-11-07 00:00:00"
TEST_END = "2026-11-11 23:00:00"
ARCS = ["Champs-Elysées", "Convention", "Saint-Pères"]

# Jours fériés sur la fenêtre glissante de 13 mois (Octobre 2025 - Novembre 2026)
FRENCH_HOLIDAYS = set(pd.to_datetime([
    "2025-11-01", "2025-11-11", "2025-12-25",
    "2026-01-01", "2026-04-06", "2026-05-01", "2026-05-08",
    "2026-05-14", "2026-05-25", "2026-07-14", "2026-08-15",
    "2026-11-01", "2026-11-11",
]).date)

# Vacances scolaires Zone C (Paris) sur la fenêtre de 13 mois (Octobre 2025 - Novembre 2026)
ZONE_C_VACATIONS = [
    ("2025-10-18", "2025-11-03"),  # Toussaint 2025
    ("2025-12-20", "2026-01-05"),  # Noël 2025
    ("2026-02-21", "2026-03-09"),  # Hiver 2026
    ("2026-04-18", "2026-05-04"),  # Printemps 2026
    ("2026-07-04", "2026-09-01"),  # Été 2026
    ("2026-10-17", "2026-11-02"),  # Toussaint 2026
]


def extend_grid_to_test_period(df_clean: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    """Prolonge la grille temporelle jusqu'au 11 novembre 2026 23h00 et y joint la météo."""
    print("1/5 - Extension de la grille horaire jusqu'au 11 novembre 2026 23:00...")
    min_date = df_clean["t_1h"].min()
    max_date = max(df_clean["t_1h"].max(), pd.to_datetime(TEST_END))

    full_range = pd.date_range(start=min_date, end=max_date, freq="1h")
    full_grid = (
        pd.MultiIndex.from_product([ARCS, full_range], names=["arc", "t_1h"])
        .to_frame()
        .reset_index(drop=True)
    )

    traffic_cols = ["arc", "t_1h", "q", "k", "is_road_closed", "is_imputed_q", "is_imputed_k"]
    df_merged = pd.merge(full_grid, df_clean[traffic_cols], on=["arc", "t_1h"], how="left")
    for col in ["is_road_closed", "is_imputed_q", "is_imputed_k"]:
        df_merged[col] = df_merged[col].fillna(0).astype(int)

    # Projection de l'état 'is_road_closed = 1' observé le 11 nov 2025 (08h-14h) sur le 11 nov 2026 aux Champs-Élysées
    mask_armistice_closure_future = (
        (df_merged["arc"] == "Champs-Elysées")
        & (df_merged["t_1h"].dt.month == 11)
        & (df_merged["t_1h"].dt.day == 11)
        & (df_merged["t_1h"].dt.hour.between(8, 14))
    )
    df_merged.loc[mask_armistice_closure_future, "is_road_closed"] = 1

    df_merged = pd.merge(df_merged, df_weather, on="t_1h", how="left")
    for w_col in ["temperature_2m", "precipitation", "rain", "wind_speed_10m"]:
        if w_col in df_merged.columns:
            df_merged[w_col] = df_merged.groupby("arc")[w_col].transform(lambda s: s.ffill().bfill())

    return df_merged.sort_values(["arc", "t_1h"]).reset_index(drop=True)


def add_calendar_and_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables calendaires, distingue les ponts de 1 jour vs 2 jours (viaduc) et cible les événements."""
    print("2/5 - Création des variables calendaires, intensité des ponts (1j vs 2j) et événements...")
    df = df.copy()
    dt = df["t_1h"]

    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear

    # Encodage cyclique (sin/cos)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # Week-ends et Jours fériés
    dates_only = dt.dt.date
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_public_holiday"] = dates_only.isin(FRENCH_HOLIDAYS).astype(int)

    # Distinction Pont de 1 jour (intensity = 1.0) vs Pont de 2 jours / Mercredi férié (intensity = 0.5 / 0.25)
    next_1d = (dt + pd.Timedelta(days=1)).dt.date
    next_2d = (dt + pd.Timedelta(days=2)).dt.date
    prev_1d = (dt - pd.Timedelta(days=1)).dt.date
    prev_2d = (dt - pd.Timedelta(days=2)).dt.date

    # 1. Vrai pont de 1 jour : Lundi avant Mardi férié (ex: 10 nov 2025) ou Vendredi après Jeudi férié (ex: 15 mai 2026)
    is_1d_bridge = (
        ((df["dayofweek"] == 0) & next_1d.isin(FRENCH_HOLIDAYS))
        | ((df["dayofweek"] == 4) & prev_1d.isin(FRENCH_HOLIDAYS))
    ) & (df["is_public_holiday"] == 0)

    # 2. Pont de 2 jours (Mercredi férié comme le 11 nov 2026) :
    # - Mardi veille (10 nov 2026) ou Jeudi lendemain : intensité modérée (0.5)
    is_2d_bridge_adjacent = (
        ((df["dayofweek"] == 1) & next_1d.isin(FRENCH_HOLIDAYS))
        | ((df["dayofweek"] == 3) & prev_1d.isin(FRENCH_HOLIDAYS))
    ) & (df["is_public_holiday"] == 0)

    # - Lundi avant-veille (9 nov 2026) ou Vendredi surlendemain : léger effet télétravail/viaduc (0.25)
    is_2d_bridge_outer = (
        ((df["dayofweek"] == 0) & next_2d.isin(FRENCH_HOLIDAYS))
        | ((df["dayofweek"] == 4) & prev_2d.isin(FRENCH_HOLIDAYS))
    ) & (df["is_public_holiday"] == 0)

    df["bridge_intensity"] = 0.0
    df.loc[is_2d_bridge_outer, "bridge_intensity"] = 0.25
    df.loc[is_2d_bridge_adjacent, "bridge_intensity"] = 0.50
    df.loc[is_1d_bridge, "bridge_intensity"] = 1.00
    df["is_bridge_day"] = (df["bridge_intensity"] >= 0.5).astype(int)

    # Vacances scolaires Zone C (Paris)
    is_vacation = pd.Series(False, index=df.index)
    for start_v, end_v in ZONE_C_VACATIONS:
        is_vacation |= (dt >= start_v) & (dt < end_v)
    df["is_school_holiday"] = is_vacation.astype(int)

    # Événements Champs-Élysées : Paris Respire (1er dimanche) & Cérémonie 11 Novembre (creux 08h-13h)
    is_first_sunday = (df["dayofweek"] == 6) & (df["day"] <= 7)
    df["is_paris_respire"] = (
        (df["arc"] == "Champs-Elysées") & is_first_sunday & (df["hour"].between(10, 18))
    ).astype(int)

    df["is_armistice"] = ((df["month"] == 11) & (df["day"] == 11)).astype(int)
    df["is_armistice_champs_closure"] = (
        (df["arc"] == "Champs-Elysées") & (df["is_armistice"] == 1) & (df["hour"].between(8, 13))
    ).astype(int)

    # Régime de jour (0: Ouvré normal, 1: Samedi, 2: Dimanche, 3: Pont, 4: Férié)
    df["day_regime"] = 0
    df.loc[df["dayofweek"] == 5, "day_regime"] = 1
    df.loc[df["dayofweek"] == 6, "day_regime"] = 2
    df.loc[df["is_bridge_day"] == 1, "day_regime"] = 3
    df.loc[df["is_public_holiday"] == 1, "day_regime"] = 4

    return df


def add_weather_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les variables d'interaction entre la météo et les heures de pointe."""
    print("3/5 - Création des interactions météo x heure de pointe...")
    df = df.copy()

    is_rush_hour = df["hour"].isin([7, 8, 9, 10, 17, 18, 19, 20]).astype(int)
    df["is_rush_hour"] = is_rush_hour
    df["is_raining"] = (df["rain"] > 0.2).astype(int)
    df["rain_in_rush_hour"] = df["rain"] * is_rush_hour
    df["temp_diff_24h"] = df.groupby("arc")["temperature_2m"].diff(24).fillna(0)

    return df


def add_causal_lags_and_sncf_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule :
    1. Les lags hebdomadaires causaux (J-7, J-14, J-21) et la tendance récente (weekly_mean_3w)
    2. Le profil historique par régime de jour
    3. Le pattern A-1 recalibré (11 nov 2025 direct + 10 nov 2025 modulé par bridge_intensity)
    """
    print("4/5 - Calcul des lags hebdomadaires et du pattern A-1 modulé par l'intensité du pont...")
    df = df.copy()
    df["date_str"] = df["t_1h"].dt.strftime("%Y-%m-%d")

    for target in ["q", "k"]:
        grp = df.groupby("arc")[target]

        # 1. Lags hebdomadaires (168h = J-7, 336h = J-14, 504h = J-21, 672h = J-28)
        lag_7d = grp.shift(24 * 7)
        lag_14d = grp.shift(24 * 14)
        lag_21d = grp.shift(24 * 21)
        lag_28d = grp.shift(24 * 28)

        df[f"{target}_lag_168h"] = lag_7d.fillna(lag_14d)
        df[f"{target}_lag_336h"] = lag_14d.fillna(lag_21d)
        df[f"{target}_lag_504h"] = lag_21d.fillna(lag_28d)

        # Niveau de base récent de l'arc à cette heure (moyenne pondérée sur 3 semaines)
        df[f"{target}_weekly_mean_3w"] = (
            0.5 * df[f"{target}_lag_168h"]
            + 0.3 * df[f"{target}_lag_336h"]
            + 0.2 * df[f"{target}_lag_504h"]
        )
        df[f"{target}_weekly_std_3w"] = pd.concat(
            [df[f"{target}_lag_168h"], df[f"{target}_lag_336h"], df[f"{target}_lag_504h"]], axis=1
        ).std(axis=1)

        # 2. Profil causal par 'day_regime' (décalé de 5 jours pour zéro Data Leakage)
        shifted_target = grp.shift(24 * 5)
        df[f"{target}_regime_hist_mean"] = (
            shifted_target.groupby([df["arc"], df["day_regime"], df["hour"]])
            .transform(lambda s: s.expanding(min_periods=1).mean())
        ).fillna(df[f"{target}_weekly_mean_3w"])

        # 3. APPROCHE SNCF A-1 : Pattern du Pont de 1 jour (10 nov 2025) et de l'Armistice (11 nov 2025)
        df[f"{target}_ratio_to_base"] = (df[target] / (df[f"{target}_weekly_mean_3w"] + 1e-3)).clip(0.0, 3.0)

        armistice_2025 = (
            df[df["date_str"] == "2025-11-11"]
            .groupby(["arc", "hour"])[f"{target}_ratio_to_base"]
            .mean()
            .rename("ratio_armistice_a1")
        )
        bridge_1d_2025 = (
            df[df["date_str"] == "2025-11-10"]
            .groupby(["arc", "hour"])[f"{target}_ratio_to_base"]
            .mean()
            .rename("ratio_bridge_1d_a1")
        )

        df = df.merge(armistice_2025, on=["arc", "hour"], how="left")
        df = df.merge(bridge_1d_2025, on=["arc", "hour"], how="left")

        # Construction de 'smart_expected' :
        # - Jour normal : weekly_mean_3w
        # - Pont (1j ou 2j) : modulation proportionnelle à bridge_intensity !
        #   Si bridge_intensity = 1.0 (pont 1j), on applique 100% du ratio_bridge_1d_a1.
        #   Si bridge_intensity = 0.5 (mardi 10 nov 2026), on applique seulement 50% de la baisse du pont 1j !
        smart_val = df[f"{target}_weekly_mean_3w"].copy()

        mask_any_bridge = df["bridge_intensity"] > 0
        w = df.loc[mask_any_bridge, "bridge_intensity"]
        r_1d = df.loc[mask_any_bridge, "ratio_bridge_1d_a1"].fillna(0.85)
        effective_ratio = 1.0 - w * (1.0 - r_1d)
        smart_val.loc[mask_any_bridge] = df.loc[mask_any_bridge, f"{target}_weekly_mean_3w"] * effective_ratio

        # - Autres jours fériés hors 11 novembre : moyenne historique des fériés
        mask_holidays_other = (df["is_public_holiday"] == 1) & (df["is_armistice"] == 0)
        smart_val.loc[mask_holidays_other] = df.loc[mask_holidays_other, f"{target}_regime_hist_mean"]

        # - 11 Novembre (Armistice) : application du profil A-1 du 11 nov 2025 (capte les cortèges 8h-13h et le rebond de 14h)
        mask_armistice = df["is_armistice"] == 1
        smart_val.loc[mask_armistice] = (
            df.loc[mask_armistice, f"{target}_weekly_mean_3w"]
            * df.loc[mask_armistice, "ratio_armistice_a1"].fillna(0.5)
        )

        df[f"{target}_smart_expected"] = smart_val
        df.drop(columns=[f"{target}_ratio_to_base", "ratio_armistice_a1", "ratio_bridge_1d_a1"], inplace=True)

    df.drop(columns=["date_str"], inplace=True)

    arc_mapping = {"Champs-Elysées": 0, "Convention": 1, "Saint-Pères": 2}
    df["arc_id"] = df["arc"].map(arc_mapping)

    return df


def run_feature_pipeline() -> None:
    """Exécute le pipeline de Feature Engineering et sauvegarde train et test (7-11 nov)."""
    clean_path = os.path.join(PROCESSED_DATA_DIR, "traffic_clean.csv")
    weather_path = os.path.join(RAW_DATA_DIR, "weather_raw.csv")

    df_clean = pd.read_csv(clean_path, parse_dates=["t_1h"])
    df_weather = pd.read_csv(weather_path, parse_dates=["t_1h"])
    df_weather = df_weather.groupby("t_1h", as_index=False).mean(numeric_only=True)

    df_full = extend_grid_to_test_period(df_clean, df_weather)
    df_feat = add_calendar_and_event_features(df_full)
    df_feat = add_weather_interaction_features(df_feat)
    df_feat = add_causal_lags_and_sncf_patterns(df_feat)

    print("5/5 - Séparation et sauvegarde de features_train.csv et features_test_nov7_11.csv...")
    mask_test = (df_feat["t_1h"] >= TEST_START) & (df_feat["t_1h"] <= TEST_END)
    df_test = df_feat.loc[mask_test].copy()

    mask_train = (df_feat["t_1h"] < TEST_START) & df_feat["q"].notna() & df_feat["k"].notna()
    df_train = df_feat.loc[mask_train].dropna(subset=["q_lag_168h", "k_lag_168h"]).copy()

    train_out = os.path.join(PROCESSED_DATA_DIR, "features_train.csv")
    test_out = os.path.join(PROCESSED_DATA_DIR, "features_test_nov7_11.csv")

    df_train.to_csv(train_out, index=False)
    df_test.to_csv(test_out, index=False)

    print(f"   -> Train set sauvegardé : {len(df_train)} lignes ({df_train['t_1h'].min()} -> {df_train['t_1h'].max()})")
    print(f"   -> Test set (7-11 nov) sauvegardé : {len(df_test)} lignes")


if __name__ == "__main__":
    run_feature_pipeline()