import os
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

MODELS_DIR = "models"

# Liste des variables explicatives communes aux modèles q et k
FEATURE_COLS = [
    # Calendrier & Cyclique
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
    "q_lag_168h", "q_lag_336h", "q_lag_504h",
    "q_weekly_mean_3w", "q_weekly_std_3w", "q_regime_hist_mean", "q_smart_expected",
    # Lags Taux d'occupation (k)
    "k_lag_168h", "k_lag_336h", "k_lag_504h",
    "k_weekly_mean_3w", "k_weekly_std_3w", "k_regime_hist_mean", "k_smart_expected",
    # Variable physique croisée (ajoutée à la volée)
    "congestion_index_168h",
]


def add_cross_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute le ratio de congestion historique k/q pour relier les deux cibles."""
    df = df.copy()
    df["congestion_index_168h"] = df["k_lag_168h"] / (df["q_lag_168h"] + 10.0)
    return df


def compute_sample_weights(df_arc: pd.DataFrame, target: str) -> np.ndarray:
    """
    Calcule le poids de chaque heure d'entraînement :
    1. Pénalise les lignes où le capteur était en panne (is_imputed == 1 -> poids x0.3)
    2. Donne progressivement plus d'importance aux mois récents (de 0.7 à 1.3)
    """
    n = len(df_arc)
    recency_weights = np.linspace(0.7, 1.3, n)
    imputed_col = f"is_imputed_{target}"
    quality_weights = np.where(df_arc[imputed_col] == 1, 0.3, 1.0)
    return recency_weights * quality_weights


class ResidualEnsembleModel:
    """
    Modèle hybride combinant :
    - L'ancrage métier/historique ('smart_expected')
    - Un ensemble 50% LightGBM + 50% XGBoost entraîné sur le résidu (y - smart_expected)
    """

    def __init__(self, target: str, arc: str):
        self.target = target
        self.arc = arc
        self.base_col = f"{target}_smart_expected"
        self.q_min = 0.0
        self.q_max = 8000.0 if target == "q" else 100.0

        self.lgbm = LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            verbose=-1,
        )

        self.xgb = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=15,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, df_train_arc: pd.DataFrame) -> "ResidualEnsembleModel":
        df_prep = add_cross_physical_features(df_train_arc)
        X = df_prep[FEATURE_COLS]
        y = df_prep[self.target].to_numpy()
        base = df_prep[self.base_col].to_numpy()

        # Cible de l'arbre = Résidu entre la vraie valeur et l'attendu intelligent
        residual = y - base
        weights = compute_sample_weights(df_prep, self.target)

        # Mémorisation du plafond historique observé sur cet arc (+10% de marge)
        self.q_max = float(np.nanpercentile(y, 99.9) * 1.10)
        if self.target == "k":
            self.q_max = min(100.0, self.q_max)

        # SÉCURITÉ ANTI-NAN / INF : Supprime toute ligne où X, residual ou weights ont un NaN ou un Inf
        valid_mask = ~(
            np.isnan(residual) | np.isinf(residual) |
            np.isnan(weights) | np.isinf(weights) |
            X.isna().any(axis=1) | np.isinf(X.select_dtypes(include=[np.number])).any(axis=1)
        )

        X_clean = X.loc[valid_mask]
        res_clean = residual[valid_mask]
        w_clean = weights[valid_mask]

        # Entraînement des deux modèles de l'ensemble sur les données nettoyées
        self.lgbm.fit(X_clean, res_clean, sample_weight=w_clean)
        self.xgb.fit(X_clean, res_clean, sample_weight=w_clean)
        return self

    def predict(self, df_test_arc: pd.DataFrame) -> np.ndarray:
        df_prep = add_cross_physical_features(df_test_arc)
        X = df_prep[FEATURE_COLS]
        base = df_prep[self.base_col].to_numpy()

        # Blending 50% LightGBM + 50% XGBoost sur le résidu
        pred_res_lgb = self.lgbm.predict(X)
        pred_res_xgb = self.xgb.predict(X)
        pred_residual = 0.5 * pred_res_lgb + 0.5 * pred_res_xgb

        # Sur les heures exactes du défilé du 11 novembre aux Champs-Élysées (08h-13h),
        # on réduit le résidu météo/arbre à 15% pour respecter le filtrage policier A-1
        is_parade = df_prep["is_armistice_parade_hours"].to_numpy() == 1
        pred_residual = np.where(is_parade, 0.15 * pred_residual, pred_residual)

        y_pred = base + pred_residual
        return np.clip(y_pred, self.q_min, self.q_max)

    def get_feature_importance(self) -> pd.Series:
        """Retourne l'importance moyenne des variables (normalisée en %)."""
        imp_lgb = self.lgbm.feature_importances_ / (self.lgbm.feature_importances_.sum() + 1e-9)
        imp_xgb = self.xgb.feature_importances_ / (self.xgb.feature_importances_.sum() + 1e-9)
        avg_imp = 0.5 * (imp_lgb + imp_xgb) * 100.0
        return pd.Series(avg_imp, index=FEATURE_COLS).sort_values(ascending=False)


def train_all_models(df_train: pd.DataFrame) -> dict[tuple[str, str], ResidualEnsembleModel]:
    """Entraîne les 6 modèles spécialisés (3 arcs x 2 cibles : q et k)."""
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
    """Génère les vecteurs pred_q et pred_k alignés sur l'ordre exact de df_data."""
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
    """Sauvegarde le dictionnaire des 6 modèles entraînés sur le disque."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(MODELS_DIR, "ensemble_models_q_k.joblib")
    joblib.dump(models, path)
    print(f"Modèles sauvegardés avec succès dans : {path}")
    return path


def load_models(path: str = None) -> dict[tuple[str, str], ResidualEnsembleModel]:
    """Charge le dictionnaire des 6 modèles depuis le disque."""
    if path is None:
        path = os.path.join(MODELS_DIR, "ensemble_models_q_k.joblib")
    return joblib.load(path)