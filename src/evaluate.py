from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error


def compute_rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Calcule la RMSE (Root Mean Squared Error), métrique officielle du datathon BCG X."""
    return float(root_mean_squared_error(y_true, y_pred))


def evaluate_predictions_detailed(df_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule et affiche la RMSE par arc et au global UNIQUEMENT sur les vraies mesures capteurs
    (is_imputed_q == 0) pour éviter tout biais lié aux périodes de panne.
    """
    df_real = df_eval[df_eval["is_imputed_q"] == 0].copy()
    rows = []
    for arc in ["Champs-Elysées", "Convention", "Saint-Pères"]:
        sub = df_real[df_real["arc"] == arc]
        if len(sub) == 0:
            continue
        rmse_q = compute_rmse(sub["q"], sub["pred_q"])
        rmse_k = compute_rmse(sub["k"], sub["pred_k"])
        rows.append({
            "Arc": arc,
            "Heures_Testees_Reelles": len(sub),
            "RMSE_Debit (q)": round(rmse_q, 2),
            "RMSE_Taux (k)": round(rmse_k, 2),
        })

    rmse_q_all = compute_rmse(df_real["q"], df_real["pred_q"])
    rmse_k_all = compute_rmse(df_real["k"], df_real["pred_k"])
    rows.append({
        "Arc": "GLOBAL (3 arcs pondérés)",
        "Heures_Testees_Reelles": len(df_real),
        "RMSE_Debit (q)": round(rmse_q_all, 2),
        "RMSE_Taux (k)": round(rmse_k_all, 2),
    })

    report_df = pd.DataFrame(rows)
    print("\n======== RAPPORT D'ÉVALUATION RMSE (SUR MESURES 100% RÉELLES) ========")
    print(report_df.to_string(index=False))
    print("======================================================================\n")
    return report_df


def get_temporal_cv_splits(df_train: pd.DataFrame, n_splits: int = 5, val_days: int = 5) -> list:
    """
    Balaye plusieurs fenêtres de test de 5 jours (120h) réparties sur toute la durée de vie
    réelle de chaque capteur (is_imputed_q == 0), et non uniquement sur octobre 2026.
    Pour chaque Fold k (de 1 à n_splits) :
    - Sélectionne pour chaque arc la k-ième fenêtre de 5 jours ayant >= 85% de données réelles
      (après avoir conservé au moins les 35% premières données réelles pour l'entraînement initial).
    - Entraîne uniquement sur le passé strict de cet arc (t_1h < val_start).
    """
    val_hours = val_days * 24
    arc_candidate_windows = {}

    for arc in ["Champs-Elysées", "Convention", "Saint-Pères"]:
        sub = df_train[df_train["arc"] == arc].sort_values("t_1h")
        real_times = sub[sub["is_imputed_q"] == 0]["t_1h"]
        if len(real_times) == 0:
            continue

        # On réserve au moins les 35% premières heures réelles pour le premier entraînement
        min_train_time = real_times.iloc[int(len(real_times) * 0.35)]
        max_real_time = real_times.max()

        # Recherche de toutes les fenêtres disjointes de 5 jours ayant >= 85% de mesures réelles
        valid_windows = []
        curr_end = max_real_time
        while curr_end - pd.Timedelta(hours=val_hours - 1) >= min_train_time:
            curr_start = curr_end - pd.Timedelta(hours=val_hours - 1)
            win_sub = sub[(sub["t_1h"] >= curr_start) & (sub["t_1h"] <= curr_end)]
            real_ratio = (win_sub["is_imputed_q"] == 0).mean()

            if len(win_sub) == val_hours and real_ratio >= 0.85:
                valid_windows.append((curr_start, curr_end))
                curr_end = curr_start - pd.Timedelta(hours=1)
            else:
                curr_end -= pd.Timedelta(hours=24)

        valid_windows.reverse()  # Ordre chronologique

        # Échantillonnage régulier de n_splits fenêtres sur toute la période disponible de l'arc
        if len(valid_windows) >= n_splits:
            indices = np.linspace(0, len(valid_windows) - 1, n_splits, dtype=int)
            arc_candidate_windows[arc] = [valid_windows[i] for i in indices]
        else:
            arc_candidate_windows[arc] = valid_windows

    # Assemblage des n_splits plis multi-époques
    splits = []
    for fold_i in range(n_splits):
        train_indices = []
        val_indices = []
        fold_dates_info = []

        for arc, win_list in arc_candidate_windows.items():
            idx = min(fold_i, len(win_list) - 1)
            v_start, v_end = win_list[idx]

            arc_mask = df_train["arc"] == arc
            t_mask = arc_mask & (df_train["t_1h"] < v_start) & (df_train["is_imputed_q"] == 0)
            v_mask = arc_mask & (df_train["t_1h"] >= v_start) & (df_train["t_1h"] <= v_end) & (df_train["is_imputed_q"] == 0)

            train_indices.extend(df_train.index[t_mask].tolist())
            val_indices.extend(df_train.index[v_mask].tolist())
            fold_dates_info.append(f"{arc[:6]}:{v_start.strftime('%d/%m/%y')}")

        label_start = pd.to_datetime(min(w[min(fold_i, len(w)-1)][0] for w in arc_candidate_windows.values()))
        label_end = pd.to_datetime(max(w[min(fold_i, len(w)-1)][1] for w in arc_candidate_windows.values()))
        splits.append((np.array(train_indices), np.array(val_indices), label_start, label_end))

    return splits


def format_predictions_for_bcgx(df_test: pd.DataFrame, pred_q: np.ndarray, pred_k: np.ndarray) -> pd.DataFrame:
    """Formate les prédictions selon le cahier des charges strict de la Slide 13 du Brief BCG X."""
    output_df = pd.DataFrame()
    output_df["arc"] = df_test["arc"].astype(str)
    output_df["datetime"] = pd.to_datetime(df_test["t_1h"]).dt.strftime("%Y-%m-%d %H:%M")
    output_df["debit_horaire"] = np.clip(pred_q, 0.0, 8000.0).astype("float64")
    output_df["taux_occupation"] = np.clip(pred_k, 0.0, 100.0).astype("float64")

    output_df["arc"] = output_df["arc"].astype(object)
    output_df["datetime"] = output_df["datetime"].astype(object)
    return output_df


def test_format_and_export_output(output_df: pd.DataFrame, output_directory: str, name_of_the_group: str) -> Path:
    """Fonction officielle de vérification et d'export du BCG X (Slide 13 du Brief)."""
    output_columns = {"arc": object, "datetime": object, "debit_horaire": float, "taux_occupation": float}

    assert sorted(list(output_df.columns)) == list(
        output_columns.keys()
    ), "Some columns are missing or unnecessary columns are in output"

    for col, col_type in output_columns.items():
        assert output_df[col].dtype == col_type, f"Column {col} does not have type {col_type}"

    try:
        output_df.datetime.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M"))
    except ValueError as e:
        raise e

    assert sorted(list(output_df["arc"].unique())) == [
        "Champs-Elysées",
        "Convention",
        "Saint-Pères",
    ], "Output does not have expected unique values for column `arc`"

    assert output_df.shape[0] == 360, f"Expected number of rows is 360, output has {output_df.shape[0]}"

    output_path = Path(output_directory) / f"output_{name_of_the_group}.csv"
    print(f"[SAVE OUTPUT] Saving output here: {output_path}")
    output_df[list(output_columns.keys())].to_csv(output_path, sep=";")
    return output_path