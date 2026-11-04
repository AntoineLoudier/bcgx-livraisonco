"""Compare variables de fériés et ablation de la correction manuelle des ponts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from main_stability import holiday_windows
from src.evaluate import build_window, window_starts
from src.model import fit_model, multi_horizon_training_features, predict_model

# Deux facteurs croisés pour distinguer les effets du calendrier et des ponts.
VARIANTS = {
    "reference": (True, False),
    "sans_pont": (False, False),
    "calendrier": (True, True),
    "calendrier_sans_pont": (False, True),
}


def evaluate(frame, starts):
    predictions, training_counts = [], []
    for start in starts:
        cutoff = start - pd.Timedelta(hours=1)
        print(f"Fenêtre {start.date()} : comparaison calendrier et ponts", flush=True)
        histories = {bridge: multi_horizon_training_features(frame, cutoff, manual_bridge=bridge)
                     for bridge in (True, False)}
        windows = {bridge: build_window(frame, start, manual_bridge=bridge) for bridge in (True, False)}
        result = windows[True].copy()
        for target in ("q", "k"):
            for name, (bridge, enriched) in VARIANTS.items():
                model, n = fit_model(histories[bridge], target, multi_horizon=True, holiday_enriched=enriched)
                result[f"{target}_{name}"] = predict_model(model, windows[bridge], target,
                                                              multi_horizon=True, holiday_enriched=enriched)
                training_counts.append({"window_start": start, "cutoff": cutoff, "target": target,
                                        "variant": name, "training_examples": n})
        result["horizon_day"] = (result.horizon_hours - 1) // 24 + 1
        predictions.append(result)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(training_counts)


def score(frame, keys):
    """Compare les quatre variantes sur les mêmes cibles observées."""
    result = []
    for values, group in frame.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        for target in ("q", "k"):
            observed = group[target].notna() & group[f"is_imputed_{target}"].eq(0)
            valid = observed & group[[f"{target}_{v}" for v in VARIANTS]].notna().all(axis=1)
            row = {**dict(zip(keys, values)), "target": target,
                   "observed": int(observed.sum()), "scored": int(valid.sum())}
            for variant in VARIANTS:
                error = group.loc[valid, f"{target}_{variant}"] - group.loc[valid, target]
                row[f"{variant}_mae"] = error.abs().mean() if len(error) else np.nan
                row[f"{variant}_rmse"] = np.sqrt((error ** 2).mean()) if len(error) else np.nan
                row[f"{variant}_bias"] = error.mean() if len(error) else np.nan
            result.append(row)
    return pd.DataFrame(result)


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
    return score(pd.concat(pieces, ignore_index=True), ["holiday", "period", "arc"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", type=Path, default=Path("data/processed/traffic_clean.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/holiday_analysis"))
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--step-days", type=int, default=7)
    args = parser.parse_args()
    frame = pd.read_csv(args.traffic, parse_dates=["t_1h"])
    starts = window_starts(frame, args.windows, args.step_days)
    events = holiday_windows(frame)
    all_starts = sorted(set(starts) | set(events.window_start))
    predictions, training_counts = evaluate(frame, all_starts)
    recent = predictions[predictions.window_start.isin(starts)]
    holiday = predictions[predictions.window_start.isin(events.window_start)]
    args.output.mkdir(parents=True, exist_ok=True)
    score(recent, ["arc"]).to_csv(args.output / "recent_summary.csv", index=False)
    score(recent, ["window_start", "arc"]).to_csv(args.output / "recent_windows.csv", index=False)
    score(recent, ["arc", "horizon_day"]).to_csv(args.output / "recent_horizons.csv", index=False)
    event_scores(predictions, events).to_csv(args.output / "holiday_scores.csv", index=False)
    events.to_csv(args.output / "holiday_windows.csv", index=False)
    training_counts.to_csv(args.output / "training_counts.csv", index=False)
    recent.to_csv(args.output / "recent_predictions.csv", index=False)
    holiday.to_csv(args.output / "holiday_predictions.csv", index=False)
    print("\nFenêtres récentes :")
    print(score(recent, ["arc"]).to_string(index=False))
    print(f"\nRésultats : {args.output}")


if __name__ == "__main__":
    main()
