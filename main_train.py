import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from src.evaluate import compute_rmse, evaluate_predictions_detailed, get_temporal_cv_splits
from src.model import (
    FEATURE_COLS,
    add_cross_physical_features,
    predict_with_models,
    save_models,
    train_all_models,
)

PROCESSED_DATA_DIR = os.path.join("data", "processed")
REPORTS_DIR = "reports"


def run_cross_validation_and_generate_metrics(df_train: pd.DataFrame, n_splits: int = 4) -> pd.DataFrame:
    """
    1. Exécute la validation croisée sur 4 fenêtres de 5 jours (120h).
    2. Sauvegarde les métriques détaillées par pli, par arc et par variable (q et k).
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print(f"1/4 - Validation Croisée Temporelle ({n_splits} fenêtres de 5 jours / 120h)...")
    splits = get_temporal_cv_splits(df_train, n_splits=n_splits, val_days=5)

    all_val_dfs = []
    fold_rows = []

    for fold_idx, (train_idx, val_idx, val_start, val_end) in enumerate(splits, start=1):
        fold_train = df_train.loc[train_idx].copy()
        fold_val = df_train.loc[val_idx].copy()

        models = train_all_models(fold_train)
        pred_q, pred_k = predict_with_models(models, fold_val)

        fold_val["fold"] = fold_idx
        fold_val["pred_q"] = pred_q
        fold_val["pred_k"] = pred_k
        fold_val["baseline_q"] = fold_val["q_lag_168h"]
        fold_val["baseline_k"] = fold_val["k_lag_168h"]

        # Enregistrement des scores par arc sur ce pli de 5 jours
        period_str = f"{val_start.strftime('%d/%m/%Y')} -> {val_end.strftime('%d/%m/%Y')}"
        for arc in ["Champs-Elysées", "Convention", "Saint-Pères"]:
            sub = fold_val[fold_val["arc"] == arc]
            fold_rows.append({
                "Fold": fold_idx,
                "Periode_5_jours": period_str,
                "Arc": arc,
                "RMSE_q_Model": round(compute_rmse(sub["q"], sub["pred_q"]), 2),
                "RMSE_q_Baseline_J7": round(compute_rmse(sub["q"], sub["baseline_q"]), 2),
                "RMSE_k_Model": round(compute_rmse(sub["k"], sub["pred_k"]), 2),
                "RMSE_k_Baseline_J7": round(compute_rmse(sub["k"], sub["baseline_k"]), 2),
            })

        all_val_dfs.append(fold_val)

    df_cv_all = pd.concat(all_val_dfs, ignore_index=True)

    # Export 1 : Détail par pli de validation
    df_folds = pd.DataFrame(fold_rows)
    df_folds.to_csv(os.path.join(REPORTS_DIR, "rmse_by_fold_detail.csv"), index=False)

    # Export 2 : Tableau de synthèse par Arc et par Outcome (q et k)
    summary_rows = []
    arcs_list = ["Champs-Elysées", "Convention", "Saint-Pères", "GLOBAL (3 arcs)"]

    for arc in arcs_list:
        sub = df_cv_all if arc == "GLOBAL (3 arcs)" else df_cv_all[df_cv_all["arc"] == arc]
        for target, label, unit in [
            ("q", "Débit horaire (q)", "véh/h"),
            ("k", "Taux d'occupation (k)", "%"),
        ]:
            y_true = sub[target]
            y_base_j7 = sub[f"baseline_{target}"]
            y_smart = sub[f"{target}_smart_expected"]
            y_pred = sub[f"pred_{target}"]

            rmse_j7 = compute_rmse(y_true, y_base_j7)
            rmse_smart = compute_rmse(y_true, y_smart)
            rmse_mod = compute_rmse(y_true, y_pred)
            mae_mod = float(mean_absolute_error(y_true, y_pred))
            gain_pct = (1.0 - rmse_mod / rmse_j7) * 100.0 if rmse_j7 > 1e-6 else 0.0

            summary_rows.append({
                "Arc": arc,
                "Outcome": label,
                "Unite": unit,
                "Moyenne_Reelle": round(float(y_true.mean()), 2),
                "RMSE_Baseline_J7": round(rmse_j7, 2),
                "RMSE_Ancrage_SNCF_Seul": round(rmse_smart, 2),
                "RMSE_Modele_Final": round(rmse_mod, 2),
                "MAE_Modele_Final": round(mae_mod, 2),
                "Gain_RMSE_vs_J7_Pct": round(gain_pct, 2),
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(REPORTS_DIR, "rmse_by_arc_and_outcome.csv"), index=False)

    # Sauvegarde des prédictions Out-Of-Fold pour les graphiques Réel vs Prédit
    df_cv_all.to_csv(os.path.join(PROCESSED_DATA_DIR, "cv_predictions_oof.csv"), index=False)
    evaluate_predictions_detailed(df_cv_all)

    return df_summary


def generate_feature_importance_and_shap(models: dict, df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule pour chacun des 6 modèles (3 arcs x 2 cibles) :
    - L'importance dans LightGBM (%)
    - L'importance dans XGBoost (%)
    - L'importance moyenne (%)
    - L'impact SHAP moyen en valeur absolue (exprimé directement en véh/h pour q et en % pour k)
    """
    print("3/4 - Calcul de l'importance des variables et des valeurs SHAP par arc et par cible...")
    try:
        import shap
        has_shap = True
    except ImportError:
        has_shap = False

    rows = []
    for (arc, target), model in models.items():
        df_arc = df_train[df_train["arc"] == arc].tail(2000).copy()
        df_prep = add_cross_physical_features(df_arc)
        X_sample = df_prep[FEATURE_COLS]

        # Importances arbres (LightGBM & XGBoost)
        imp_lgb = model.lgbm.feature_importances_ / (model.lgbm.feature_importances_.sum() + 1e-9) * 100.0
        imp_xgb = model.xgb.feature_importances_ / (model.xgb.feature_importances_.sum() + 1e-9) * 100.0
        imp_avg = 0.5 * (imp_lgb + imp_xgb)

        # Valeurs SHAP (impact moyen en unités physiques sur le résidu)
        if has_shap:
            explainer_lgb = shap.TreeExplainer(model.lgbm)
            shap_vals_lgb = explainer_lgb.shap_values(X_sample)
            mean_abs_shap = np.abs(shap_vals_lgb).mean(axis=0)
        else:
            mean_abs_shap = np.zeros(len(FEATURE_COLS))

        df_mod_imp = pd.DataFrame({
            "Arc": arc,
            "Outcome": "Débit horaire (q)" if target == "q" else "Taux d'occupation (k)",
            "Target_Code": target,
            "Feature": FEATURE_COLS,
            "Importance_Moyenne_Pct": np.round(imp_avg, 2),
            "Importance_LGBM_Pct": np.round(imp_lgb, 2),
            "Importance_XGB_Pct": np.round(imp_xgb, 2),
            "Impact_SHAP_Moyen_Unite": np.round(mean_abs_shap, 3),
        }).sort_values("Importance_Moyenne_Pct", ascending=False)

        df_mod_imp["Rang"] = range(1, len(df_mod_imp) + 1)
        rows.append(df_mod_imp)

    df_all_imp = pd.concat(rows, ignore_index=True)
    df_all_imp.to_csv(os.path.join(REPORTS_DIR, "feature_importances_by_arc_outcome.csv"), index=False)
    return df_all_imp


def write_markdown_interpretability_report(df_summary: pd.DataFrame, df_all_imp: pd.DataFrame) -> None:
    """Génère un rapport Markdown lisible dans VS Code (Cmd+K V) résumant RMSE et Top Features."""
    print("4/4 - Rédaction automatique du rapport reports/RAPPORT_INTERPRETABILITE.md...")
    report_path = os.path.join(REPORTS_DIR, "RAPPORT_INTERPRETABILITE.md")

    lines = [
        "# 📊 Rapport d'Interprétabilité & Performance (LivraisonCo — BCG X)\n",
        "Ce rapport est généré automatiquement par `main_train.py`. Il détaille la performance prédictive (RMSE & MAE) et les variables explicatives majeures pour chacun des **3 arcs** et chacune des **2 variables cibles**.\n",
        "---\n",
        "## 1. Synthèse des Performances (Validation Croisée sur fenêtres de 5 jours)\n",
        "- **`RMSE_Baseline_J7`** : Prédiction naïve utilisant uniquement la valeur d'il y a 7 jours (`lag_168h`).",
        "- **`RMSE_Ancrage_SNCF_Seul`** : Prédiction utilisant notre variable `smart_expected` (moyenne 3 semaines + modulation pont 1j/2j + ratio A-1 Armistice).",
        "- **`RMSE_Modele_Final`** : Prédiction finale après correction résiduelle par l'ensemble **50% LightGBM + 50% XGBoost**.",
        "- **`MAE_Modele_Final`** : Erreur moyenne absolue (interprétation directe pour le client LivraisonCo).\n",
        df_summary.to_markdown(index=False),
        "\n---\n",
        "## 2. Top 8 des Variables Explicatives par Arc et par Variable Cible\n",
        "> **Comment lire ce tableau :** Notre modèle prédit la cible en partant de `smart_expected` puis en ajustant le résidu grâce aux arbres. `Importance_Moyenne_Pct` indique le poids de la variable dans les décisions des arbres (météo, heure, congestion passée, volatilité), et `Impact_SHAP_Moyen_Unite` mesure l'amplitude moyenne de la correction en **véh/h** (pour $q$) ou en **points de %** (pour $k$).\n",
    ]

    for arc in ["Champs-Elysées", "Convention", "Saint-Pères"]:
        lines.append(f"### 📍 Arc : `{arc}`\n")
        for target_code, label in [("q", "Débit Horaire (q - véh/h)"), ("k", "Taux d'Occupation (k - %)")]:
            lines.append(f"#### Cible : **{label}**\n")
            sub_imp = df_all_imp[
                (df_all_imp["Arc"] == arc) & (df_all_imp["Target_Code"] == target_code)
            ].head(8)[
                ["Rang", "Feature", "Importance_Moyenne_Pct", "Importance_LGBM_Pct", "Importance_XGB_Pct", "Impact_SHAP_Moyen_Unite"]
            ]
            lines.append(sub_imp.to_markdown(index=False))
            lines.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Rapport complet généré -> {report_path} (Ouvre-le avec Cmd+K puis V dans VS Code !)")


def main() -> None:
    train_path = os.path.join(PROCESSED_DATA_DIR, "features_train.csv")
    df_train = pd.read_csv(train_path, parse_dates=["t_1h"])

    # 1. Validation croisée et génération des tableaux RMSE par arc et par cible
    df_summary = run_cross_validation_and_generate_metrics(df_train, n_splits=4)

    # 2. Entraînement des 6 modèles finaux sur 100% de l'historique
    print("2/4 - Entraînement des 6 modèles finaux sur 100% des données historiques...")
    final_models = train_all_models(df_train)
    save_models(final_models)

    # 3. Calcul des importances et des valeurs SHAP par arc et par cible
    df_all_imp = generate_feature_importance_and_shap(final_models, df_train)

    # 4. Génération du rapport Markdown d'interprétabilité
    write_markdown_interpretability_report(df_summary, df_all_imp)


if __name__ == "__main__":
    main()