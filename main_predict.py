import argparse
import os
import pandas as pd
from src.data_loader import download_paris_traffic, download_paris_weather
from src.evaluate import format_predictions_for_bcgx, test_format_and_export_output
from src.features import run_feature_pipeline
from src.model import load_models, predict_with_models
from src.preprocessing import run_preprocessing_pipeline
from main_train import main as run_training

PROCESSED_DATA_DIR = os.path.join("data", "processed")

DEFAULT_GROUP_NAME = "Les Architectes"


def generate_final_submission(group_name: str, output_dir: str = ".") -> None:
    """
    Charge les modèles entraînés, prédit les 360 lignes du 07/11 au 11/11,
    et valide/exporte le CSV via la fonction officielle du BCG X (Slide 13).
    """
    test_path = os.path.join(PROCESSED_DATA_DIR, "features_test_nov7_11.csv")
    df_test = pd.read_csv(test_path, parse_dates=["t_1h"])

    # Tri canonique par arc puis par date
    df_test = df_test.sort_values(["arc", "t_1h"]).reset_index(drop=True)

    print("Chargement des modèles depuis models/ensemble_models_q_k.joblib...")
    models = load_models()

    print("Génération des prédictions horaires du 07/11/2026 00:00 au 11/11/2026 23:00 (360 lignes)...")
    pred_q, pred_k = predict_with_models(models, df_test)

    output_df = format_predictions_for_bcgx(df_test, pred_q, pred_k)

    print("\n--- Aperçu des 5 premières lignes de la soumission ---")
    print(output_df.head())
    print("\n--- Moyennes prédites par Arc sur la période du 7 au 11 novembre ---")
    print(output_df.groupby("arc")[["debit_horaire", "taux_occupation"]].mean().round(2))

    # Validation stricte et export via la fonction officielle BCG X (Slide 13)
    test_format_and_export_output(
        output_df=output_df,
        output_directory=output_dir,
        name_of_the_group=group_name,
    )
    print("\n[SUCCÈS] Toutes les assertions du BCG X sont validées à 100% !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génération du livrable CSV officiel BCG X (7-11 nov)")
    parser.add_argument(
        "--group",
        type=str,
        default=DEFAULT_GROUP_NAME,
        help="Nom du groupe pour le fichier output_<nom_du_groupe>.csv",
    )
    parser.add_argument(
        "--refresh-all",
        action="store_true",
        help="Re-télécharge les dernières données, relance le preprocessing, les features et l'entraînement avant de prédire (à utiliser le 5/6 novembre !)",
    )
    args = parser.parse_args()

    if args.refresh_all:
        print("=== MISE À JOUR COMPLÈTE DU PIPELINE (DONNÉES FRAÎCHES + RÉ-ENTRAÎNEMENT) ===")
        download_paris_traffic(start_date="2025-09-01")
        download_paris_weather(start_date="2025-09-01")
        run_preprocessing_pipeline()
        run_feature_pipeline()
        run_training()

    generate_final_submission(group_name=args.group, output_dir=".")