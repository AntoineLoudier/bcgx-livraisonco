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
    Calcule et affiche la RMSE par arc et au global pour le débit (q) et le taux d'occupation (k).
    Attend un DataFrame contenant : ['arc', 'q', 'k', 'pred_q', 'pred_k'].
    """
    rows = []
    for arc in ["Champs-Elysées", "Convention", "Saint-Pères"]:
        sub = df_eval[df_eval["arc"] == arc]
        if len(sub) == 0:
            continue
        rmse_q = compute_rmse(sub["q"], sub["pred_q"])
        rmse_k = compute_rmse(sub["k"], sub["pred_k"])
        rows.append({
            "Arc": arc,
            "RMSE_Debit (q)": round(rmse_q, 2),
            "RMSE_Taux (k)": round(rmse_k, 2),
        })

    # Score global sur les 3 arcs confondus
    rmse_q_all = compute_rmse(df_eval["q"], df_eval["pred_q"])
    rmse_k_all = compute_rmse(df_eval["k"], df_eval["pred_k"])
    rows.append({
        "Arc": "GLOBAL (3 arcs)",
        "RMSE_Debit (q)": round(rmse_q_all, 2),
        "RMSE_Taux (k)": round(rmse_k_all, 2),
    })

    report_df = pd.DataFrame(rows)
    print("\n================ RAPPORT D'ÉVALUATION RMSE ================")
    print(report_df.to_string(index=False))
    print("===========================================================\n")
    return report_df


def get_temporal_cv_splits(df_train: pd.DataFrame, n_splits: int = 4, val_days: int = 5) -> list:
    """
    Découpe l'historique en fenêtres de validation glissantes de 5 jours consécutifs (120h),
    reproduisant exactement l'horizon de test du 7 au 11 novembre.
    Garantit que l'entraînement (train_idx) est strictement antérieur à la validation (val_idx).
    """
    max_date = df_train["t_1h"].max()
    val_delta = pd.Timedelta(days=val_days)

    splits = []
    for i in range(n_splits, 0, -1):
        val_end = max_date - pd.Timedelta(days=(i - 1) * val_days)
        val_start = val_end - val_delta + pd.Timedelta(hours=1)

        train_mask = df_train["t_1h"] < val_start
        val_mask = (df_train["t_1h"] >= val_start) & (df_train["t_1h"] <= val_end)

        train_idx = df_train.index[train_mask].to_numpy()
        val_idx = df_train.index[val_mask].to_numpy()
        splits.append((train_idx, val_idx, val_start, val_end))

    return splits


def format_predictions_for_bcgx(df_test: pd.DataFrame, pred_q: np.ndarray, pred_k: np.ndarray) -> pd.DataFrame:
    """
    Formate les prédictions selon le cahier des charges de la Slide 13 du Brief BCG X :
    - Colonnes : ['arc', 'datetime', 'debit_horaire', 'taux_occupation']
    - Format datetime : '%Y-%m-%d %H:%M'
    - Bornes physiques : debit_horaire in [0, 8000] et taux_occupation in [0, 100]
    """
    output_df = pd.DataFrame()
    output_df["arc"] = df_test["arc"].astype(str)
    output_df["datetime"] = pd.to_datetime(df_test["t_1h"]).dt.strftime("%Y-%m-%d %H:%M")
    output_df["debit_horaire"] = np.clip(pred_q, 0.0, 8000.0).astype("float64")
    output_df["taux_occupation"] = np.clip(pred_k, 0.0, 100.0).astype("float64")

    # Forçage explicite du dtype 'object' pour passer le test d'assertion de la Slide 13
    output_df["arc"] = output_df["arc"].astype(object)
    output_df["datetime"] = output_df["datetime"].astype(object)

    return output_df


def test_format_and_export_output(output_df: pd.DataFrame, output_directory: str, name_of_the_group: str) -> Path:
    """
    Fonction officielle de vérification et d'export fournie par le BCG X (Slide 13 du Brief).
    """
    output_columns = {"arc": object, "datetime": object, "debit_horaire": float, "taux_occupation": float}

    # 1. Check relevant columns are in output dataframe
    assert sorted(list(output_df.columns)) == list(
        output_columns.keys()
    ), "Some columns are missing or unnecessary columns are in output"

    # 2. Check types
    for col, col_type in output_columns.items():
        assert output_df[col].dtype == col_type, f"Column {col} does not have type {col_type}"

    # 3. Check datetime string has right format
    try:
        output_df.datetime.apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M"))
    except ValueError as e:
        raise e

    # 4. Check `arc` columns has right values
    assert sorted(list(output_df["arc"].unique())) == [
        "Champs-Elysées",
        "Convention",
        "Saint-Pères",
    ], "Output does not have expected unique values for column `arc`"

    # 5. Check dataframe has right number of rows
    assert output_df.shape[0] == 360, f"Expected number of rows is 360, output has {output_df.shape[0]}"

    # 6. Export output
    output_path = Path(output_directory) / f"output_{name_of_the_group}.csv"
    print(f"[SAVE OUTPUT] Saving output here: {output_path}")
    output_df[list(output_columns.keys())].to_csv(output_path, sep=";")
    return output_path