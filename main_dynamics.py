"""Compare un modèle multi-horizon avec et sans variables de dynamique réelle."""

import argparse
import json
from pathlib import Path

import pandas as pd

from main_multi_horizon import compare
from main_stability import holiday_windows
from src.evaluate import build_window, window_starts
from src.features import FORECAST_CUTOFF
from src.model import feature_columns, fit_model, multi_horizon_training_features, predict_model


def evaluate(frame, starts):
    predictions, counts, importance = [], [], []
    for start in starts:
        cutoff = start - pd.Timedelta(hours=1)
        print(f"Fenêtre {start.date()} : référence et dynamique récente", flush=True)
        history = multi_horizon_training_features(frame, cutoff, recent_dynamics=True)
        window = build_window(frame, start, recent_dynamics=True)
        for target in ("q", "k"):
            baseline, n_fixed = fit_model(history, target, multi_horizon=True)
            candidate, n_multi = fit_model(history, target, multi_horizon=True, recent_dynamics=True)
            window[f"{target}_fixed"] = predict_model(baseline, window, target, multi_horizon=True)
            window[f"{target}_multi"] = predict_model(candidate, window, target, multi_horizon=True, recent_dynamics=True)
            counts.append({"window_start": start, "cutoff": cutoff, "target": target,
                           "reference_examples": n_fixed, "dynamics_examples": n_multi})
            gains = candidate.booster_.feature_importance(importance_type="gain")
            importance.extend({"window_start": start, "target": target, "feature": feature, "gain": float(gain)}
                              for feature, gain in zip(feature_columns(target, True, recent_dynamics=True), gains))
        window["horizon_day"] = (window.horizon_hours - 1) // 24 + 1
        predictions.append(window)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(counts), pd.DataFrame(importance)


def event_scores(predictions, events):
    pieces = []
    for event in events.itertuples(index=False):
        window = predictions[predictions.window_start == event.window_start]
        dates = window.t_1h.dt.normalize()
        for period, offset in (("avant_veille", -2), ("veille", -1), ("ferie", 0),
                               ("lendemain", 1), ("surlendemain", 2)):
            part = window.loc[dates.eq(event.holiday + pd.Timedelta(days=offset))].copy()
            if not part.empty:
                part["holiday"] = event.holiday
                part["period"] = period
                pieces.append(part)
    if not pieces:
        return pd.DataFrame()
    return compare(pd.concat(pieces, ignore_index=True), ["holiday", "period", "arc"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", type=Path, default=Path("data/processed/traffic_clean.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/dynamics_analysis"))
    parser.add_argument("--models", type=Path, default=Path("models/recent_dynamics"))
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--step-days", type=int, default=7)
    args = parser.parse_args()
    frame = pd.read_csv(args.traffic, parse_dates=["t_1h"])
    starts = window_starts(frame, args.windows, args.step_days)
    events = holiday_windows(frame)
    all_starts = sorted(set(starts) | set(events.window_start))
    predictions, counts, importance = evaluate(frame, all_starts)
    recent = predictions[predictions.window_start.isin(starts)]
    holiday = predictions[predictions.window_start.isin(events.window_start)]
    args.output.mkdir(parents=True, exist_ok=True)
    compare(recent, ["arc"]).to_csv(args.output / "recent_summary.csv", index=False)
    compare(recent, ["window_start", "arc"]).to_csv(args.output / "recent_windows.csv", index=False)
    compare(recent, ["arc", "horizon_day"]).to_csv(args.output / "recent_horizons.csv", index=False)
    event_scores(predictions, events).to_csv(args.output / "holidays.csv", index=False)
    events.to_csv(args.output / "holiday_windows.csv", index=False)
    counts.to_csv(args.output / "training_counts.csv", index=False)
    importance.to_csv(args.output / "feature_importance.csv", index=False)
    recent.to_csv(args.output / "recent_predictions.csv", index=False)
    holiday.to_csv(args.output / "holiday_predictions.csv", index=False)

    cutoff = min(pd.Timestamp(FORECAST_CUTOFF), frame.t_1h.max())
    history = multi_horizon_training_features(frame, cutoff, recent_dynamics=True)
    args.models.mkdir(parents=True, exist_ok=True)
    metadata = {"cutoff": str(cutoff), "method": "LightGBM regression_l1", "experimental": True, "targets": {}}
    for target in ("q", "k"):
        model, size = fit_model(history, target, multi_horizon=True, recent_dynamics=True)
        model.booster_.save_model(str(args.models / f"lightgbm_{target}.txt"))
        metadata["targets"][target] = {"features": feature_columns(target, True, recent_dynamics=True), "training_examples": size}
    (args.models / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print("\nComparaison sur les mêmes observations :")
    print(compare(recent, ["arc"]).to_string(index=False))
    print(f"\nRésultats : {args.output} ; modèles expérimentaux : {args.models}")


if __name__ == "__main__":
    main()
