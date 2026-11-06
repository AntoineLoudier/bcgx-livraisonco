import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

MODELS_DIR = "models"

FEATURE_COLS = [
    # Calendrier & Cyclique (is_summer_holiday exclu pour éviter le sur-ajustement sur l'unique été 2026)
    "hour", "dayofweek", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_public_holiday", "bridge_intensity", "is_bridge_day",
    "is_school_holiday", "day_regime",
    # Fermetures & Événements
    "is_road_closed", "is_paris_respire", "is_armistice", "is_armistice_parade_hours",
    # Météo & Interactions
    "temperature_2m", "precipitation", "rain", "wind_speed_10m",
    "is_rush_hour", "is_raining", "rain_in_rush_hour", "temp_diff_24h",
    # Lags Débit (q)
    "q_lag_168h", "q_lag_336h", "q_lag_504h", "q_lag_672h",
    "q_weekly_mean_3w", "q_weekly_median_4w", "q_weekly_std_3w", "q_regime_hist_mean", "q_smart_expected",
    # Lags Taux d'occupation (k)
    "k_lag_168h", "k_lag_336h", "k_lag_504h", "k_lag_672h",
    "k_weekly_mean_3w", "k_weekly_median_4w", "k_weekly_std_3w", "k_regime_hist_mean", "k_smart_expected",
    # Variables physiques croisées et pentes horaires (ajoutées à la volée)
    "congestion_index_168h", "q_hour_slope_168h", "k_hour_slope_168h",
    "holiday_x_rush",
    "q_hour_p90", "k_hour_p90",
]


def add_cross_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute :
    1. L'indice de congestion k/q
    2. La pente horaire (accélération du trafic entre h-1 et h à J-7)
    3. La médiane robuste sur 4 semaines (J-7 à J-28), insensible aux anomalies isolées sur Saint-Pères
    4. L'interaction Vacances scolaires x Heure de pointe
    """
    df = df.copy()
    df["congestion_index_168h"] = df["k_lag_168h"] / (df["q_lag_168h"] + 10.0)
    df["q_hour_slope_168h"] = df.groupby("arc")["q_lag_168h"].diff(1).fillna(0.0)
    df["k_hour_slope_168h"] = df.groupby("arc")["k_lag_168h"].diff(1).fillna(0.0)
    df["q_weekly_median_4w"] = (
        pd.concat([df["q_lag_168h"], df["q_lag_336h"], df["q_lag_504h"], df["q_lag_672h"]], axis=1)
        .median(axis=1)
    )
    df["k_weekly_median_4w"] = (
        pd.concat([df["k_lag_168h"], df["k_lag_336h"], df["k_lag_504h"], df["k_lag_672h"]], axis=1)
        .median(axis=1)
    )
    df["holiday_x_rush"] = df["is_school_holiday"] * df["is_rush_hour"]
    return df


def compute_sample_weights(df_arc: pd.DataFrame, target: str = "q") -> np.ndarray:
    """
    1. Donne plus de poids aux observations récentes (0.85 -> 1.20)
    2. Sur le débit (q), sur-pondère les heures de journée 08h-20h (x1.25)
    """
    n = len(df_arc)
    if n <= 1:
        return np.ones(n)
    recency = np.linspace(0.85, 1.20, n)
    if target == "q":
        daytime_boost = np.where(df_arc["hour"].between(8, 20), 1.25, 0.90)
        return recency * daytime_boost
    return recency


class ResidualEnsembleModel:
    """
    Modèle hybride entraîné EXCLUSIVEMENT sur les heures réelles (is_imputed == 0) :
    - Ancrage métier/historique ('smart_expected')
    - Ensemble 50% LightGBM + 50% XGBoost sur le résidu (y - smart_expected)
    """

    def __init__(self, target: str, arc: str):
        self.target = target
        self.arc = arc
        self.base_col = f"{target}_smart_expected"
        self.q_min = 0.0
        self.q_max = 8000.0 if target == "q" else 100.0

        # Paramétrage différencié et régularisation adaptée par typologie d'axe (Configuration Antigravité) :
        # - Champs-Élysées : faible historique réel (~1143h), modèle fortement contraint.
        # - Saint-Pères : axe urbain étroit et bruité ; profondeur réduite et shrinkage résiduel (0.40 / 0.85).
        # - Convention : grand axe pendulaire structuré ; capacité nominale complète (1.05 / 1.00).
        if arc == "Champs-Elysées":
            n_est = 120
            depth_lgb = 3
            leaves_lgb = 9
            depth_xgb = 3
            min_child = 35
            reg_l2 = 4.0
            self.residual_weight = 0.80
        elif arc == "Saint-Pères":
            n_est = 350
            depth_lgb = 4
            leaves_lgb = 14
            depth_xgb = 3
            min_child = 25
            reg_l2 = 5.0
            self.residual_weight = 0.40 if target == "q" else 0.85
        else:  # Convention
            n_est = 400
            depth_lgb = 5
            leaves_lgb = 23
            depth_xgb = 4
            min_child = 20
            reg_l2 = 4.0
            self.residual_weight = 1.05 if target == "q" else 1.00

        self.lgbm = LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=n_est,
            learning_rate=0.025,
            num_leaves=leaves_lgb,
            max_depth=depth_lgb,
            min_child_samples=min_child,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=reg_l2,
            random_state=42,
            verbose=-1,
        )

        self.xgb = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=n_est,
            learning_rate=0.025,
            max_depth=depth_xgb,
            min_child_weight=min_child,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=2.0,
            reg_lambda=reg_l2,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, df_train_arc: pd.DataFrame) -> "ResidualEnsembleModel":
        df_prep = add_cross_physical_features(df_train_arc)

        # FILTRE CRUCIAL : On n'entraîne les arbres QUE sur les vraies mesures capteurs (is_imputed == 0)
        imputed_col = f"is_imputed_{self.target}"
        if imputed_col in df_prep.columns:
            df_prep = df_prep[df_prep[imputed_col] == 0].copy()

        X = df_prep[FEATURE_COLS]
        y = df_prep[self.target].to_numpy()
        base = df_prep[self.base_col].to_numpy()

        residual = y - base
        weights = compute_sample_weights(df_prep, self.target)

        self.q_max = float(np.nanpercentile(y, 99.9) * 1.10)
        if self.target == "k":
            self.q_max = min(100.0, self.q_max)

        valid_mask = ~(
            np.isnan(residual) | np.isinf(residual) |
            np.isnan(weights) | np.isinf(weights) |
            X.isna().any(axis=1) | np.isinf(X.select_dtypes(include=[np.number])).any(axis=1)
        )

        X_clean = X.loc[valid_mask]
        res_clean = residual[valid_mask]
        w_clean = weights[valid_mask]

        # Écrêtage robuste (Winsorizing 1% - 99%) des résidus d'entraînement sur Saint-Pères et Convention
        if self.arc in ["Saint-Pères", "Convention"] and len(res_clean) > 100:
            low_q, high_q = np.percentile(res_clean, [1.0, 99.0])
            res_clean = np.clip(res_clean, low_q, high_q)

        self.lgbm.fit(X_clean, res_clean, sample_weight=w_clean)
        self.xgb.fit(X_clean, res_clean, sample_weight=w_clean)
        return self

    def predict(self, df_test_arc: pd.DataFrame) -> np.ndarray:
        df_prep = add_cross_physical_features(df_test_arc)
        X = df_prep[FEATURE_COLS].fillna(0.0)
        base = df_prep[self.base_col].fillna(df_prep[f"{self.target}_lag_168h"]).to_numpy()

        pred_res_lgb = self.lgbm.predict(X)
        pred_res_xgb = self.xgb.predict(X)
        pred_residual = (0.5 * pred_res_lgb + 0.5 * pred_res_xgb) * self.residual_weight

        is_parade = df_prep["is_armistice_parade_hours"].to_numpy() == 1
        pred_residual = np.where(is_parade, 0.15 * pred_residual, pred_residual)

        y_pred = base + pred_residual
        return np.clip(y_pred, self.q_min, self.q_max)

    def get_feature_importance(self) -> pd.Series:
        imp_lgb = self.lgbm.feature_importances_ / (self.lgbm.feature_importances_.sum() + 1e-9)
        imp_xgb = self.xgb.feature_importances_ / (self.xgb.feature_importances_.sum() + 1e-9)
        avg_imp = 0.5 * (imp_lgb + imp_xgb) * 100.0
        return pd.Series(avg_imp, index=FEATURE_COLS).sort_values(ascending=False)


def train_all_models(df_train: pd.DataFrame) -> dict[tuple[str, str], ResidualEnsembleModel]:
    models = {}
    for arc in ["Champs-Elysées", "Convention", "Saint-Pères"]:
        df_arc = df_train[df_train["arc"] == arc].sort_values("t_1h").reset_index(drop=True)
        for target in ["q", "k"]:
            model = ResidualEnsembleModel(target=target, arc=arc)
            model.fit(df_arc)
            models[(arc, target)] = model
    return models


def predict_with_models(
    models: dict[tuple[str, str], ResidualEnsembleModel], df_data: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    pred_q = np.zeros(len(df_data), dtype="float64")
    pred_k = np.zeros(len(df_data), dtype="float64")

    for arc in ["Champs-Elysées", "Convention", "Saint-Pères"]:
        mask = (df_data["arc"] == arc).to_numpy()
        if not mask.any():
            continue
        df_arc = df_data.loc[mask]
        pred_q[mask] = models[(arc, "q")].predict(df_arc)
        pred_k[mask] = models[(arc, "k")].predict(df_arc)

    return pred_q, pred_k


def save_models(models: dict[tuple[str, str], ResidualEnsembleModel], path: str = None) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(MODELS_DIR, "ensemble_models_q_k.joblib")
    joblib.dump(models, path)
    print(f"Modèles sauvegardés avec succès dans : {path}")
    return path


def load_models(path: str = None) -> dict[tuple[str, str], ResidualEnsembleModel]:
    if path is None:
        path = os.path.join(MODELS_DIR, "ensemble_models_q_k.joblib")
    return joblib.load(path)