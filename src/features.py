import os
import numpy as np
import pandas as pd

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

# Fenêtre officielle à prédire pour le Livrable 1 (Slide 11 & 13 du Brief BCG X)
TEST_START = "2026-11-07 00:00:00"
TEST_END = "2026-11-11 23:00:00"
FORECAST_CUTOFF = pd.Timestamp("2026-11-06 23:00:00")
FORECAST_HORIZON = pd.Timedelta(hours=120)
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
    """Prolonge la grille temporelle jusqu'au 11 novembre 2026 23h00 et projette la fermeture du 11 nov."""
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
        df_merged[col] = df_merged[col].fillna(1 if "imputed" in col else 0).astype(int)

    # Projection de 'is_road_closed = 1' observé le 11 nov 2025 (08h00-14h00) sur le 11 nov 2026 aux Champs-Élysées
    mask_armistice_closure = (
        (df_merged["arc"] == "Champs-Elysées")
        & (df_merged["t_1h"].dt.month == 11)
        & (df_merged["t_1h"].dt.day == 11)
        & (df_merged["t_1h"].dt.hour.between(8, 14))
    )
    df_merged.loc[mask_armistice_closure, "is_road_closed"] = 1

    # Jointure météo complète (Archive + Forecast)
    df_merged = pd.merge(df_merged, df_weather, on="t_1h", how="left")
    return df_merged.sort_values(["arc", "t_1h"]).reset_index(drop=True)


def add_calendar_and_event_features(df: pd.DataFrame, manual_bridge: bool = True) -> pd.DataFrame:
    """Crée les variables cycliques, calendaires, l'intensité des ponts (1j vs 2j) et les événements."""
    print("2/5 - Création des variables calendaires et distinction Pont 1j vs Pont 2j (Viaduc)...")
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

    # Qualification des ponts (1 jour vs 2 jours autour d'un mercredi férié)
    next_1d = (dt + pd.Timedelta(days=1)).dt.date
    next_2d = (dt + pd.Timedelta(days=2)).dt.date
    prev_1d = (dt - pd.Timedelta(days=1)).dt.date
    prev_2d = (dt - pd.Timedelta(days=2)).dt.date

    is_1d_bridge = (
        ((df["dayofweek"] == 0) & next_1d.isin(FRENCH_HOLIDAYS))
        | ((df["dayofweek"] == 4) & prev_1d.isin(FRENCH_HOLIDAYS))
    ) & (df["is_public_holiday"] == 0)

    is_2d_bridge_adjacent = (
        ((df["dayofweek"] == 1) & next_1d.isin(FRENCH_HOLIDAYS))
        | ((df["dayofweek"] == 3) & prev_1d.isin(FRENCH_HOLIDAYS))
    ) & (df["is_public_holiday"] == 0)

    is_2d_bridge_outer = (
        ((df["dayofweek"] == 0) & next_2d.isin(FRENCH_HOLIDAYS))
        | ((df["dayofweek"] == 4) & prev_2d.isin(FRENCH_HOLIDAYS))
    ) & (df["is_public_holiday"] == 0)

    df["bridge_intensity"] = 0.0
    df.loc[is_2d_bridge_outer, "bridge_intensity"] = 0.25
    df.loc[is_2d_bridge_adjacent, "bridge_intensity"] = 0.50
    df.loc[is_1d_bridge, "bridge_intensity"] = 1.00
    if not manual_bridge:
        # Ablation : ni coefficients de pont ni régime « pont » construit à partir d'eux.
        df["bridge_intensity"] = 0.0
    df["is_bridge_day"] = (df["bridge_intensity"] >= 0.50).astype(int)

    # Vacances scolaires Zone C (Paris)
    is_vacation = pd.Series(False, index=df.index)
    for start_v, end_v in ZONE_C_VACATIONS:
        is_vacation |= (dt >= start_v) & (dt < end_v)
    df["is_school_holiday"] = is_vacation.astype(int)

    # Calendrier connu à l'avance : distances bornées et type du férié le plus
    # proche. En cas d'égalité, choisir le prochain férié.
    holidays = pd.DatetimeIndex(sorted(FRENCH_HOLIDAYS))
    dates = dt.dt.normalize().to_numpy(dtype="datetime64[D]")
    calendar = holidays.to_numpy(dtype="datetime64[D]")
    position = np.searchsorted(calendar, dates)
    before = np.clip(position - 1, 0, len(calendar) - 1)
    after = np.clip(position, 0, len(calendar) - 1)
    previous_days = (dates - calendar[before]).astype(int)
    next_days = (calendar[after] - dates).astype(int)
    # La recherche place la date même parmi les prochains fériés.
    previous_days = np.where(next_days == 0, 0, previous_days)
    previous_days = np.where((position > 0) | (next_days == 0), previous_days, 8)
    next_days = np.where(position < len(calendar), next_days, 8)
    df["days_since_holiday"] = np.minimum(previous_days, 8)
    df["days_until_holiday"] = np.minimum(next_days, 8)
    choose_next = next_days <= previous_days
    nearest = np.where(choose_next, after, before)
    near = np.minimum(previous_days, next_days) <= 7
    nearest_dates = holidays[nearest]
    fixed_days = {(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)}
    fixed = np.array([(date.month, date.day) in fixed_days for date in nearest_dates])
    df["nearest_holiday_fixed"] = (near & fixed).astype(int)
    df["nearest_holiday_movable"] = (near & ~fixed).astype(int)
    df["nearest_holiday_weekday"] = np.where(near, nearest_dates.dayofweek, -1)
    df["near_holiday_in_school_break"] = ((np.minimum(previous_days, next_days) <= 2) & is_vacation.to_numpy()).astype(int)

    # Événements Champs-Élysées
    is_first_sunday = (df["dayofweek"] == 6) & (df["day"] <= 7)
    df["is_paris_respire"] = (
        (df["arc"] == "Champs-Elysées") & is_first_sunday & (df["hour"].between(10, 18))
    ).astype(int)

    df["is_armistice"] = ((df["month"] == 11) & (df["day"] == 11)).astype(int)
    df["is_armistice_parade_hours"] = (
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
    df["temp_diff_24h"] = df.groupby("arc")["temperature_2m"].diff(24).fillna(0.0)

    return df


def add_causal_lags_and_sncf_patterns(
    df: pd.DataFrame,
    forecast_start: pd.Timestamp | None = None,
    forecast_cutoff: pd.Timestamp | None = None,
    training_horizon_days: int | None = None,
    recent_dynamics: bool = False,
    real_lags_only: bool = False,
    availability_delay_hours: int = 0,
) -> pd.DataFrame:
    """Construit des variables à partir des seules mesures disponibles à l'émission.

    Chaque ligne historique simule une prévision à cinq jours. Pour la fenêtre
    finale, aucune mesure postérieure au gel du 6 novembre n'est utilisée.
    """
    df = df.sort_values(["arc", "t_1h"]).reset_index(drop=True).copy()
    if training_horizon_days is not None and (forecast_start is not None or forecast_cutoff is not None):
        raise ValueError("Un horizon d'entraînement ne peut pas être combiné à une émission fixe")
    if training_horizon_days is not None:
        if training_horizon_days not in range(1, 6):
            raise ValueError("L'horizon d'entraînement doit être compris entre 1 et 5 jours")
        # Une émission à 23 h précède le premier jour de chaque fenêtre.
        available_at = df["t_1h"].dt.normalize() - pd.Timedelta(days=training_horizon_days - 1, hours=1)
        df["horizon_hours"] = ((df["t_1h"] - available_at) / pd.Timedelta(hours=1)).astype(int)
    else:
        available_at = df["t_1h"] - FORECAST_HORIZON
    if (forecast_start is None) != (forecast_cutoff is None):
        raise ValueError("forecast_start et forecast_cutoff doivent être fournis ensemble")
    if forecast_start is not None:
        forecast_start, forecast_cutoff = pd.Timestamp(forecast_start), pd.Timestamp(forecast_cutoff)
        if forecast_cutoff >= forecast_start:
            raise ValueError("La date d'émission doit précéder la fenêtre de prévision")
        available_at = available_at.where(df["t_1h"] < forecast_start, forecast_cutoff)
    elif training_horizon_days is None:
        test = df["t_1h"] >= pd.Timestamp(TEST_START)
        available_at = available_at.where(~test, available_at.clip(upper=FORECAST_CUTOFF))
    if availability_delay_hours < 0:
        raise ValueError("Le délai de disponibilité doit être positif")
    available_at = available_at - pd.Timedelta(hours=availability_delay_hours)
    df["available_at"] = available_at

    def historical_lookup(source, keys, columns):
        left = df[keys + ["available_at"]].copy()
        left["_row"] = np.arange(len(left))
        right = source[keys + ["t_1h"] + columns].rename(columns={"t_1h": "observed_at"})
        if right.empty:
            return pd.DataFrame(np.nan, index=df.index, columns=columns)
        result = pd.merge_asof(
            left.sort_values("available_at"), right.sort_values("observed_at"),
            left_on="available_at", right_on="observed_at", by=keys,
            direction="backward", allow_exact_matches=True,
        ).sort_values("_row")
        return result[columns].reset_index(drop=True)

    for target in ["q", "k"]:
        group = df.groupby("arc")[target]
        if recent_dynamics:
            real_lags = []
            imputed = df.groupby("arc")[f"is_imputed_{target}"]
            for days in (7, 14, 21):
                lag = group.shift(24 * days)
                valid = (imputed.shift(24 * days).eq(0) &
                         (df["t_1h"] - pd.Timedelta(days=days) <= df["available_at"]))
                real_lags.append(lag.where(valid))
            last_week, two_weeks, three_weeks = real_lags
            samples = pd.concat(real_lags, axis=1)
            typical = pd.concat([two_weeks, three_weeks], axis=1).mean(axis=1)
            scale = typical.abs().clip(lower=10.0 if target == "q" else 1.0)
            df[f"{target}_recent_vs_usual"] = ((last_week - typical) / scale).clip(-5, 5)
            df[f"{target}_weekly_trend"] = (last_week - three_weeks) / 2
            df[f"{target}_real_weekly_std"] = samples.std(axis=1)
            df[f"{target}_real_weekly_count"] = samples.count(axis=1)

            observed = df[(df[f"is_imputed_{target}"] == 0) & df[target].notna()].copy()
            observed["last_real_time"] = observed["t_1h"]
            last_time = pd.to_datetime(historical_lookup(observed, ["arc"], ["last_real_time"])["last_real_time"])
            df[f"{target}_last_real_age_hours"] = (df["available_at"] - last_time) / pd.Timedelta(hours=1)
        for days in (7, 14, 21):
            lag = group.shift(24 * days)
            known = df["t_1h"] - pd.Timedelta(days=days) <= df["available_at"]
            real = df.groupby("arc")[f"is_imputed_{target}"].shift(24 * days).eq(0)
            df[f"{target}_lag_{days * 24}h_real"] = (known & real & lag.notna()).astype(int)
            df[f"{target}_lag_{days * 24}h"] = lag.where(known & real if real_lags_only else known)

        # En cas de capteur indisponible, utiliser seulement les dernières
        # observations effectivement connues au moment de la prévision.
        real = df[(df["arc"] == "Champs-Elysées") & (df[f"is_imputed_{target}"] == 0) & df[target].notna()].copy()
        keys = ["arc", "dayofweek", "hour"]
        for i in range(3):
            real[f"real_week_{i}"] = real.groupby(keys)[target].shift(i)
        past = historical_lookup(real, keys, [f"real_week_{i}" for i in range(3)])
        dead = (df["arc"] == "Champs-Elysées") & (df["t_1h"] >= "2025-12-15") & (not real_lags_only)
        for i, days in enumerate((7, 14, 21)):
            df.loc[dead, f"{target}_lag_{days * 24}h"] = past.loc[dead, f"real_week_{i}"]

        lags = [f"{target}_lag_{days * 24}h" for days in (7, 14, 21)]
        df[f"{target}_weekly_mean_3w"] = sum(df[col] * weight for col, weight in zip(lags, (0.5, 0.3, 0.2)))
        df[f"{target}_weekly_std_3w"] = df[lags].std(axis=1)

        # Le régime appartient à l'observation passée, pas à la ligne décalée.
        real = df[(df[f"is_imputed_{target}"] == 0) & df[target].notna()].copy()
        regime_keys = ["arc", "day_regime", "hour"]
        real["history_mean"] = real.groupby(regime_keys)[target].transform(lambda s: s.expanding().mean())
        history = historical_lookup(real, regime_keys, ["history_mean"])["history_mean"]
        df[f"{target}_regime_hist_mean"] = history.fillna(df[f"{target}_weekly_mean_3w"])

        # L'anomalie de l'événement historique doit être identique quel que
        # soit l'horizon de l'exemple qui la consulte. Son profil est figé
        # cinq jours (et le délai simulé) avant cet événement.
        event_lags = [group.shift(24 * days) if 24 * days >= 120 + availability_delay_hours else pd.Series(np.nan, index=df.index)
                      for days in (7, 14, 21)]
        event_profile = sum(lag * weight for lag, weight in zip(event_lags, (0.5, 0.3, 0.2)))
        ratio = (df[target] / (event_profile + 1e-3)).clip(0, 3)
        smart = df[f"{target}_weekly_mean_3w"].copy()
        for event_date, mask in [
            ("2025-11-10", df["bridge_intensity"] > 0),
            ("2025-11-11", df["is_armistice"] == 1),
        ]:
            # Les ratios ne peuvent servir qu'après observation des événements.
            reference = df.loc[(df["t_1h"].dt.strftime("%Y-%m-%d") == event_date) & (df[f"is_imputed_{target}"] == 0), ["arc", "hour", "t_1h"]].copy()
            reference["event_ratio"] = ratio.loc[reference.index]
            reference = reference.dropna(subset=["event_ratio"])
            event_ratio = historical_lookup(reference, ["arc", "hour"], ["event_ratio"])["event_ratio"]
            eligible = mask & event_ratio.notna()
            if event_date.endswith("10"):
                weight = df.loc[eligible, "bridge_intensity"]
                smart.loc[eligible] *= 1 - weight * (1 - event_ratio.loc[eligible])
            else:
                smart.loc[eligible] *= event_ratio.loc[eligible]
        other_holidays = (df["is_public_holiday"] == 1) & (df["is_armistice"] == 0)
        smart.loc[other_holidays] = df.loc[other_holidays, f"{target}_regime_hist_mean"]
        df[f"{target}_smart_expected"] = smart

    df["arc_id"] = df["arc"].map({"Champs-Elysées": 0, "Convention": 1, "Saint-Pères": 2})
    return df.drop(columns="available_at")


def run_feature_pipeline() -> None:
    """Exécute le pipeline de Feature Engineering et sauvegarde train et test (7-11 nov)."""
    clean_path = os.path.join(PROCESSED_DATA_DIR, "traffic_clean.csv")
    weather_path = os.path.join(RAW_DATA_DIR, "weather_raw.csv")

    df_clean = pd.read_csv(clean_path, parse_dates=["t_1h"])
    df_weather = pd.read_csv(weather_path, parse_dates=["t_1h"])
    df_weather = df_weather.groupby("t_1h", as_index=False).mean(numeric_only=True)

    df_full = extend_grid_to_test_period(df_clean, df_weather)
    df_feat = add_calendar_and_event_features(df_full)
    df_feat = add_causal_lags_and_sncf_patterns(
        df_feat, forecast_start=pd.Timestamp(TEST_START), forecast_cutoff=FORECAST_CUTOFF,
        recent_dynamics=True,
    )

    print("5/5 - Séparation et sauvegarde de features_train.csv et features_test_nov7_11.csv...")
    mask_test = (df_feat["t_1h"] >= TEST_START) & (df_feat["t_1h"] <= TEST_END)
    df_test = df_feat.loc[mask_test].copy()
    df_test["horizon_hours"] = ((df_test["t_1h"] - FORECAST_CUTOFF) / pd.Timedelta(hours=1)).astype(int)

    # On retire les 7 premiers jours (initialisation du lag 168h)
    min_train_date = df_feat["t_1h"].min() + pd.Timedelta(days=7)
    mask_train = (df_feat["t_1h"] >= min_train_date) & (df_feat["t_1h"] < TEST_START) & df_feat["q"].notna() & df_feat["k"].notna()
    mask_train &= (df_feat["is_imputed_q"] == 0) & (df_feat["is_imputed_k"] == 0)
    df_train = df_feat.loc[mask_train].copy()

    # Ces champs décrivent la cible ou des observations météo futures, et ne
    # sont pas des variables connues lors de l’émission de la prévision.
    unavailable = ["is_road_closed", "is_imputed_q", "is_imputed_k",
                   "temperature_2m", "precipitation", "rain", "wind_speed_10m"]
    df_train = df_train.drop(columns=unavailable, errors="ignore")
    df_test = df_test.drop(columns=unavailable, errors="ignore")
    df_test = df_test.drop(columns=["q", "k"])

    train_out = os.path.join(PROCESSED_DATA_DIR, "features_train.csv")
    test_out = os.path.join(PROCESSED_DATA_DIR, "features_test_nov7_11.csv")

    df_train.to_csv(train_out, index=False)
    df_test.to_csv(test_out, index=False)

    print(f"   -> Train set sauvegardé : {len(df_train)} lignes (cibles observées uniquement)")
    print(f"   -> Test set (7-11 nov) sauvegardé : {len(df_test)} lignes")


if __name__ == "__main__":
    run_feature_pipeline()
