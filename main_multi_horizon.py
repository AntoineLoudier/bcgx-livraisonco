"""Entraîne et compare un modèle à horizons de 1 à 120 heures."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import build_window, window_starts
from src.features import FORECAST_CUTOFF
from src.model import (feature_columns, fit_model, multi_horizon_training_features,
                       predict_model, training_features)


def compare(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """MAE, RMSE et couverture sur des cibles observées communes."""
    output = []
    for values, group in frame.groupby(keys, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        for target in ("q", "k"):
            observed = group[target].notna() & group[f"is_imputed_{target}"].eq(0)
            valid = observed & group[f"{target}_fixed"].notna() & group[f"{target}_multi"].notna()
            row = {**dict(zip(keys, values)), "target": target,
                   "observed": int(observed.sum()), "scored": int(valid.sum())}
            for name in ("fixed", "multi"):
                errors = group.loc[valid, f"{target}_{name}"] - group.loc[valid, target]
                row[f"{name}_mae"] = errors.abs().mean() if len(errors) else np.nan
                row[f"{name}_rmse"] = np.sqrt((errors ** 2).mean()) if len(errors) else np.nan
            output.append(row)
    return pd.DataFrame(output)


def evaluate(frame: pd.DataFrame, starts: list[pd.Timestamp]):
    predictions, counts = [], []
    for start in starts:
        cutoff = start - pd.Timedelta(hours=1)
        print(f"Fenêtre {start.date()} : comparaison des deux entraînements", flush=True)
        fixed = training_features(frame, cutoff)
        multi = multi_horizon_training_features(frame, cutoff)
        window = build_window(frame, start)
        for target in ("q", "k"):
            baseline, n_fixed = fit_model(fixed, target)
            candidate, n_multi = fit_model(multi, target, multi_horizon=True)
            window[f"{target}_fixed"] = predict_model(baseline, window, target)
            window[f"{target}_multi"] = predict_model(candidate, window, target, multi_horizon=True)
            counts.append({"window_start": start, "cutoff": cutoff, "target": target,
                           "fixed_examples": n_fixed, "multi_examples": n_multi})
        window["horizon_day"] = (window.horizon_hours - 1) // 24 + 1
        predictions.append(window)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", type=Path, default=Path("data/processed/traffic_clean.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/multi_horizon_analysis"))
    parser.add_argument("--models", type=Path, default=Path("models/multi_horizon"))
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--step-days", type=int, default=7)
    args = parser.parse_args()
    frame = pd.read_csv(args.traffic, parse_dates=["t_1h"])
    starts = window_starts(frame, args.windows, args.step_days)
    predictions, counts = evaluate(frame, starts)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    compare(predictions, ["arc"]).to_csv(args.output / "comparison.csv", index=False)
    compare(predictions, ["arc", "horizon_day"]).to_csv(args.output / "by_horizon.csv", index=False)
    compare(predictions, ["window_start", "arc"]).to_csv(args.output / "by_window.csv", index=False)
    counts.to_csv(args.output / "training_counts.csv", index=False)

    cutoff = min(pd.Timestamp(FORECAST_CUTOFF), frame.t_1h.max())
    history = multi_horizon_training_features(frame, cutoff)
    args.models.mkdir(parents=True, exist_ok=True)
    metadata = {"cutoff": str(cutoff), "method": "LightGBM regression_l1",
                "emission": "23:00 heure locale, prévision des cinq jours suivants",
                "horizon_hours": "1-120 en heures locales", "targets": {}}
    for target in ("q", "k"):
        model, size = fit_model(history, target, multi_horizon=True)
        model.booster_.save_model(str(args.models / f"lightgbm_{target}.txt"))
        metadata["targets"][target] = {"features": feature_columns(target, True), "training_examples": size}
    (args.models / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print("\nComparaison sur les mêmes observations :")
    print(compare(predictions, ["arc"]).to_string(index=False))
    print(f"\nAnalyses : {args.output} ; modèles : {args.models} ; coupure : {cutoff}.")


if __name__ == "__main__":
    main()
