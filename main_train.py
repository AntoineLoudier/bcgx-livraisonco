"""Évalue un premier LightGBM sur des fenêtres temporelles puis entraîne les modèles finaux."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import build_window, score_window, weather_availability, window_starts
from src.features import FORECAST_CUTOFF
from src.model import feature_columns, fit_model, predict_model, training_features


def evaluate_models(frame: pd.DataFrame, starts: list[pd.Timestamp]):
    predictions, metrics, importances, training = [], [], [], []
    for start in starts:
        cutoff = start - pd.Timedelta(hours=1)
        print(f"Validation {start.date()} — émission {cutoff}...")
        history = training_features(frame, cutoff)
        window = build_window(frame, start)
        for target in ("q", "k"):
            model, size = fit_model(history, target)
            window[f"{target}_lightgbm"] = predict_model(model, window, target)
            training.append({"window_start": start, "cutoff": cutoff, "target": target, "observed_training_rows": size})
            imp = model.booster_.feature_importance(importance_type="gain")
            importances.extend({"window_start": start, "target": target, "feature": feature, "gain": float(gain)}
                               for feature, gain in zip(feature_columns(target), imp))
        base = score_window(window)
        model_scores = []
        for arc, group in window.groupby("arc"):
            for target in ("q", "k"):
                observed = group[target].notna() & group[f"is_imputed_{target}"].eq(0)
                valid = observed & group[f"{target}_lightgbm"].notna()
                error = group.loc[valid, f"{target}_lightgbm"] - group.loc[valid, target]
                model_scores.append({"window_start": start, "forecast_cutoff": cutoff, "arc": arc,
                                     "target": target, "method": "lightgbm", "hours": len(group),
                                     "observed": int(observed.sum()), "scored": int(valid.sum()),
                                     "missing_prediction": int((observed & ~valid).sum()),
                                     "mae": error.abs().mean() if len(error) else np.nan,
                                     "rmse": np.sqrt((error ** 2).mean()) if len(error) else np.nan})
        metrics.append(pd.concat([base, pd.DataFrame(model_scores)], ignore_index=True))
        predictions.append(window)
    return (pd.concat(predictions, ignore_index=True), pd.concat(metrics, ignore_index=True),
            pd.DataFrame(importances), pd.DataFrame(training))


def compare_on_common_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Comparaison strictement sur les mêmes observations pour chaque référence."""
    rows = []
    methods = {"lightgbm": "lightgbm", "semaine_precedente": "lag_168h", "moyenne_3_semaines": "weekly_mean_3w", "profil_calendrier": "smart_expected"}
    grouped = [(arc, target, group) for target in ("q", "k") for arc, group in predictions.groupby("arc")]
    for arc, target, group in grouped:
        observed = group[target].notna() & group[f"is_imputed_{target}"].eq(0)
        for method, suffix in methods.items():
            col = f"{target}_{suffix}"
            common = observed & group[col].notna() & group[f"{target}_lightgbm"].notna()
            error = group.loc[common, col] - group.loc[common, target]
            model_error = group.loc[common, f"{target}_lightgbm"] - group.loc[common, target]
            rows.append({"arc": arc, "target": target, "method": method, "count": int(common.sum()),
                         "mae": error.abs().mean() if len(error) else np.nan,
                         "rmse": np.sqrt((error ** 2).mean()) if len(error) else np.nan,
                         "lightgbm_mae_same_rows": model_error.abs().mean() if len(error) else np.nan})
    return pd.DataFrame(rows)


def error_by_hour(predictions: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for target in ("q", "k"):
        valid = predictions[target].notna() & predictions[f"is_imputed_{target}"].eq(0)
        part = predictions.loc[valid, ["arc", "hour", target, f"{target}_lightgbm"]].copy()
        part["target"] = target
        part["error"] = part[f"{target}_lightgbm"] - part[target]
        part["absolute_error"] = part.error.abs()
        parts.append(part[["arc", "hour", "target", "error", "absolute_error"]])
    data = pd.concat(parts, ignore_index=True)
    return data.groupby(["arc", "target", "hour"], as_index=False).agg(
        count=("error", "size"), bias=("error", "mean"), mae=("absolute_error", "mean"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", type=Path, default=Path("data/processed/traffic_clean.csv"))
    parser.add_argument("--weather", type=Path, default=Path("data/raw/weather_raw.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/model_analysis"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--step-days", type=int, default=7)
    args = parser.parse_args()
    frame = pd.read_csv(args.traffic, parse_dates=["t_1h"])
    starts = window_starts(frame, args.windows, args.step_days)
    pred, metrics, importance, train_counts = evaluate_models(frame, starts)
    comparison = compare_on_common_rows(pred)
    args.output.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.output / "predictions.csv", index=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    comparison.to_csv(args.output / "comparison.csv", index=False)
    error_by_hour(pred).to_csv(args.output / "errors_by_hour.csv", index=False)
    importance.to_csv(args.output / "feature_importance.csv", index=False)
    train_counts.to_csv(args.output / "training_counts.csv", index=False)
    weather_availability(args.weather, starts).to_csv(args.output / "weather_availability.csv", index=False)

    # Dernier modèle : pas de cible ni de mesure de trafic postérieure au gel officiel.
    cutoff = min(pd.Timestamp(FORECAST_CUTOFF), frame.t_1h.max())
    history = training_features(frame, cutoff)
    args.models.mkdir(parents=True, exist_ok=True)
    metadata = {"cutoff": str(cutoff), "method": "LightGBM regression_l1", "targets": {}}
    for target in ("q", "k"):
        model, size = fit_model(history, target)
        model.booster_.save_model(str(args.models / f"lightgbm_{target}.txt"))
        metadata["targets"][target] = {"features": feature_columns(target), "observed_training_rows": size}
    (args.models / "lightgbm_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print("\nComparaison sur les mêmes observations :")
    print(comparison.to_string(index=False))
    print(f"\nAnalyses : {args.output} ; modèles : {args.models} ; données jusqu'au {cutoff}.")


if __name__ == "__main__":
    main()
