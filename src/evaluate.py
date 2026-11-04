"""Validation chronologique de références à émission unique sur cinq jours."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .features import ARCS, add_calendar_and_event_features, add_causal_lags_and_sncf_patterns
except ImportError:
    from features import ARCS, add_calendar_and_event_features, add_causal_lags_and_sncf_patterns

WINDOW_HOURS = 120
METHODS = {"semaine_precedente": "lag_168h", "moyenne_3_semaines": "weekly_mean_3w", "profil_calendrier": "smart_expected"}


def window_starts(frame: pd.DataFrame, count: int, step_days: int) -> list[pd.Timestamp]:
    """Dernières fenêtres complètes et non chevauchantes par défaut."""
    if count < 1 or step_days < 5:
        raise ValueError("Il faut au moins une fenêtre et un pas d'au moins cinq jours")
    measured = ((frame["q"].notna() & frame["is_imputed_q"].eq(0)) |
                (frame["k"].notna() & frame["is_imputed_k"].eq(0)))
    observed = frame.loc[measured, "t_1h"]
    if observed.empty:
        raise ValueError("Aucune observation de trafic disponible")
    last = observed.max()
    last_complete = last.normalize() if last.hour == 23 else last.normalize() - pd.Timedelta(days=1)
    latest = last_complete - pd.Timedelta(days=4)
    starts = [latest - pd.Timedelta(days=step_days * i) for i in reversed(range(count))]
    if starts[0] - pd.Timedelta(days=21) < frame.t_1h.min():
        raise ValueError("Historique insuffisant pour les fenêtres et leurs trois semaines de retard")
    return starts


def build_window(frame: pd.DataFrame, start: pd.Timestamp, manual_bridge: bool = True, recent_dynamics: bool = False, real_lags_only: bool = False, availability_delay_hours: int = 0) -> pd.DataFrame:
    """Masque toutes les cibles futures avant de recalculer les variables."""
    start = pd.Timestamp(start)
    if start != start.normalize():
        raise ValueError("Le début d'une fenêtre doit être à minuit")
    cutoff = start - pd.Timedelta(hours=1)
    end = start + pd.Timedelta(hours=WINDOW_HOURS - 1)
    if end > frame.t_1h.max():
        raise ValueError(f"Fenêtre incomplète : {start.date()} à {end.date()}")
    historical = frame[frame.t_1h <= cutoff - pd.Timedelta(hours=availability_delay_hours)].copy()
    if historical.empty:
        raise ValueError("Aucun historique disponible avant la fenêtre")
    if historical.duplicated(["arc", "t_1h"]).any():
        raise ValueError("Doublons arc/heure dans les données préparées")
    expected_hours = pd.date_range(historical.t_1h.min(), end, freq="h")
    grid = pd.MultiIndex.from_product([ARCS, expected_hours], names=["arc", "t_1h"]).to_frame(index=False)
    data = grid.merge(historical, on=["arc", "t_1h"], how="left", validate="one_to_one")
    for col in ("is_imputed_q", "is_imputed_k"):
        data[col] = data[col].fillna(1)
    features = add_causal_lags_and_sncf_patterns(
        add_calendar_and_event_features(data, manual_bridge=manual_bridge), forecast_start=start,
        forecast_cutoff=cutoff, recent_dynamics=recent_dynamics, real_lags_only=real_lags_only, availability_delay_hours=availability_delay_hours
    )
    features = features[features.t_1h.between(start, end)].copy()
    truth = frame.loc[frame.t_1h.between(start, end), ["arc", "t_1h", "q", "k", "is_imputed_q", "is_imputed_k"]]
    features = features.drop(columns=["q", "k", "is_imputed_q", "is_imputed_k"])
    result = features.merge(truth, on=["arc", "t_1h"], how="left", validate="one_to_one")
    result["window_start"] = start
    result["forecast_cutoff"] = cutoff
    result["horizon_hours"] = ((result["t_1h"] - cutoff) / pd.Timedelta(hours=1)).astype(int)
    return result


def score_window(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calcule MAE et RMSE sur les couples prévision/mesure observée uniquement."""
    rows = []
    for arc, group in predictions.groupby("arc", sort=True):
        for target in ("q", "k"):
            observed = group[target].notna() & group[f"is_imputed_{target}"].eq(0)
            for method, feature in METHODS.items():
                predicted = group[f"{target}_{feature}"].notna()
                valid = observed & predicted
                errors = group.loc[valid, f"{target}_{feature}"] - group.loc[valid, target]
                rows.append({"window_start": group.window_start.iloc[0], "forecast_cutoff": group.forecast_cutoff.iloc[0],
                             "arc": arc, "target": target, "method": method, "hours": len(group),
                             "observed": int(observed.sum()), "scored": int(valid.sum()),
                             "missing_prediction": int((observed & ~predicted).sum()),
                             "mae": errors.abs().mean() if len(errors) else np.nan,
                             "rmse": np.sqrt((errors ** 2).mean()) if len(errors) else np.nan})
    return pd.DataFrame(rows)


def weather_availability(path: Path, starts: list[pd.Timestamp]) -> pd.DataFrame:
    """Un fichier sans date d'émission ne prouve pas une prévision historique disponible."""
    rows = []
    if not path.exists():
        status = "fichier absent"
        weather = None
    else:
        weather = pd.read_csv(path)
        status = "date d'émission absente : prévisions historiques non vérifiables"
        if {"issued_at", "valid_at"}.issubset(weather.columns):
            for col in ("issued_at", "valid_at"):
                if weather[col].isna().any() or weather[col].map(lambda value: pd.Timestamp(value).tzinfo is None).any():
                    raise ValueError(f"{col} doit inclure un fuseau horaire")
                weather[col] = pd.to_datetime(weather[col], errors="raise", utc=True).dt.tz_convert("Europe/Paris")
            status = "dates d'émission présentes"
    for start in starts:
        cutoff = (start - pd.Timedelta(hours=1)).tz_localize("Europe/Paris", ambiguous="raise", nonexistent="raise")
        # La grille de trafic utilise 120 heures locales, avec l'heure répétée
        # du changement d'heure agrégée en une seule ligne.
        valid = pd.date_range(start, periods=WINDOW_HOURS, freq="h")
        coverage = np.nan
        if weather is not None and {"issued_at", "valid_at"}.issubset(weather.columns):
            local_valid = weather.valid_at.dt.tz_localize(None)
            eligible = (weather.issued_at <= cutoff) & local_valid.isin(valid)
            coverage = local_valid[eligible].nunique()
        rows.append({"window_start": start, "forecast_cutoff": cutoff.isoformat(), "status": status,
                     "forecast_hours_with_eligible_record": coverage, "expected_hours": WINDOW_HOURS,
                     "used_in_baselines": False})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", type=Path, default=Path("data/processed/traffic_clean.csv"))
    parser.add_argument("--weather", type=Path, default=Path("data/raw/weather_raw.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/validation"))
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--step-days", type=int, default=7)
    args = parser.parse_args()
    frame = pd.read_csv(args.traffic, parse_dates=["t_1h"])
    starts = window_starts(frame, args.windows, args.step_days)
    all_predictions = pd.concat([build_window(frame, start) for start in starts], ignore_index=True)
    metrics = pd.concat([score_window(group) for _, group in all_predictions.groupby("window_start")], ignore_index=True)
    weather = weather_availability(args.weather, starts)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    all_predictions.to_csv(args.output / "predictions.csv", index=False)
    weather.to_csv(args.output / "weather_availability.csv", index=False)
    print("\nRésultats par fenêtre, arc, cible et référence :", args.output / "metrics.csv")
    print(metrics.to_string(index=False))
    print("\nDisponibilité météo :")
    print(weather.to_string(index=False))


if __name__ == "__main__":
    main()
