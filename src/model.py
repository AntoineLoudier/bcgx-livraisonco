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


def feature_columns(target: str) -> list[str]:
    if target not in ("q", "k"):
        raise ValueError("La cible doit être q ou k")
    return CALENDAR_FEATURES + [f"{target}_{name}" for name in LAG_FEATURES]


def training_features(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Ne lire que l'historique disponible et recalculer ses variables causales."""
    history = frame.loc[frame.t_1h <= cutoff].copy()
    if history.empty:
        raise ValueError("Aucune observation avant la date d'émission")
    return add_causal_lags_and_sncf_patterns(add_calendar_and_event_features(history))


def fit_model(features: pd.DataFrame, target: str) -> tuple[lgb.LGBMRegressor, int]:
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
    model.fit(train[feature_columns(target)], train[target])
    return model, len(train)


def predict_model(model: lgb.LGBMRegressor, frame: pd.DataFrame, target: str) -> np.ndarray:
    """Respecte les bornes physiques de chaque cible."""
    return np.clip(model.predict(frame[feature_columns(target)]), 0, 100 if target == "k" else None)
