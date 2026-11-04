"""Premier modèle LightGBM et préparation temporelle de son entraînement."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import add_calendar_and_event_features, add_causal_lags_and_sncf_patterns

CALENDAR_FEATURES = [
    "arc_id", "hour", "dayofweek", "month", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_public_holiday", "is_school_holiday", "bridge_intensity", "day_regime",
    "is_paris_respire", "is_armistice", "is_armistice_parade_hours",
]
LAG_FEATURES = ["lag_168h", "lag_336h", "lag_504h", "weekly_mean_3w", "weekly_std_3w", "regime_hist_mean", "smart_expected"]
HOLIDAY_FEATURES = ["days_since_holiday", "days_until_holiday", "nearest_holiday_fixed",
                    "nearest_holiday_movable", "nearest_holiday_weekday", "near_holiday_in_school_break"]
DYNAMIC_FEATURES = ["recent_vs_usual", "weekly_trend", "real_weekly_std", "real_weekly_count", "last_real_age_hours"]


def feature_columns(target: str, multi_horizon: bool = False, holiday_enriched: bool = False, recent_dynamics: bool = False) -> list[str]:
    if target not in ("q", "k"):
        raise ValueError("La cible doit être q ou k")
    return (CALENDAR_FEATURES + [f"{target}_{name}" for name in LAG_FEATURES]
            + (["horizon_hours"] if multi_horizon else []) + (HOLIDAY_FEATURES if holiday_enriched else [])
            + ([f"{target}_{name}" for name in DYNAMIC_FEATURES] if recent_dynamics else []))


def training_features(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Ne lire que l'historique disponible et recalculer ses variables causales."""
    history = frame.loc[frame.t_1h <= cutoff].copy()
    if history.empty:
        raise ValueError("Aucune observation avant la date d'émission")
    return add_causal_lags_and_sncf_patterns(add_calendar_and_event_features(history))


def multi_horizon_training_features(frame: pd.DataFrame, cutoff: pd.Timestamp, manual_bridge: bool = True, recent_dynamics: bool = False, real_lags_only: bool = False, availability_delay_hours: int = 0) -> pd.DataFrame:
    """Cinq exemples par cible : émissions quotidiennes et horizons de 1 à 120 h."""
    history = frame.loc[frame.t_1h <= cutoff - pd.Timedelta(hours=availability_delay_hours)].copy()
    if history.empty:
        raise ValueError("Aucune observation avant la date d'émission")
    calendar = add_calendar_and_event_features(history, manual_bridge=manual_bridge)
    examples = [add_causal_lags_and_sncf_patterns(calendar, training_horizon_days=day, recent_dynamics=recent_dynamics, real_lags_only=real_lags_only, availability_delay_hours=availability_delay_hours)
                for day in range(1, 6)]
    return pd.concat(examples, ignore_index=True)


def fit_model(
    features: pd.DataFrame,
    target: str,
    multi_horizon: bool = False,
    sample_weight: pd.Series | None = None,
    holiday_enriched: bool = False,
    recent_dynamics: bool = False,
) -> tuple[lgb.LGBMRegressor, int]:
    """Un modèle global par cible, entraîné sur les seules cibles observées."""
    known = features[target].notna() & features[f"is_imputed_{target}"].eq(0)
    train = features.loc[known]
    if len(train) < 100:
        raise ValueError(f"Historique observé insuffisant pour {target} : {len(train)} lignes")
    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=250, learning_rate=0.04,
        num_leaves=15, min_child_samples=80, subsample=1.0,
        colsample_bytree=0.9, random_state=42, n_jobs=2, verbosity=-1,
    )
    weights = None
    if sample_weight is not None:
        weights = sample_weight.reindex(train.index)
        if weights.isna().any() or not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Les poids des observations doivent être positifs et finis")
    model.fit(train[feature_columns(target, multi_horizon, holiday_enriched, recent_dynamics)], train[target], sample_weight=weights)
    return model, len(train)


def predict_model(model: lgb.LGBMRegressor, frame: pd.DataFrame, target: str, multi_horizon: bool = False, holiday_enriched: bool = False, recent_dynamics: bool = False) -> np.ndarray:
    """Respecte les bornes physiques de chaque cible."""
    return np.clip(model.predict(frame[feature_columns(target, multi_horizon, holiday_enriched, recent_dynamics)]), 0, 100 if target == "k" else None)
