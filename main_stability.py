"""Étude de stabilité par fenêtres hebdomadaires et autour des jours fériés."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from main_train import compare_on_common_rows, evaluate_models
from src.evaluate import WINDOW_HOURS, window_starts
from src.features import FRENCH_HOLIDAYS


def holiday_windows(frame):
    """Cinq jours centrés sur chaque férié quand l'historique le permet."""
    first, last = frame.t_1h.min(), frame.t_1h.max()
    events = []
    for holiday in sorted(FRENCH_HOLIDAYS):
        day = pd.Timestamp(holiday)
        start = min(day - pd.Timedelta(days=2), last.normalize() - pd.Timedelta(days=5))
        end = start + pd.Timedelta(hours=WINDOW_HOURS - 1)
        if start - pd.Timedelta(days=21) >= first and start <= day <= end and end <= last:
            events.append({"holiday": day, "window_start": start, "window_end": end})
    return pd.DataFrame(events)


def stability_by_window(predictions):
    """Compare chaque fenêtre sur l'intersection des observations disponibles."""
    rows = []
    for start, group in predictions.groupby("window_start"):
        comparison = compare_on_common_rows(group)
        comparison = comparison[(comparison.method == "moyenne_3_semaines") & (comparison["count"] > 0)].copy()
        comparison["window_start"] = start
        comparison["improvement"] = comparison.mae - comparison.lightgbm_mae_same_rows
        rows.append(comparison)
    return pd.concat(rows, ignore_index=True)


def score_holidays(predictions, events):
    """J-1, jour J, J+1 et cinq jours, avec couples modèle/référence identiques."""
    rows = []
    for event in events.itertuples(index=False):
        window = predictions[predictions.window_start == event.window_start]
        date = window.t_1h.dt.normalize()
        for period, mask in [("veille", date == event.holiday - pd.Timedelta(days=1)),
                             ("ferie", date == event.holiday),
                             ("lendemain", date == event.holiday + pd.Timedelta(days=1)),
                             ("fenetre_5j", date.notna())]:
            part = window.loc[mask]
            for arc, group in part.groupby("arc"):
                for target in ("q", "k"):
                    valid = (group[target].notna() & group[f"is_imputed_{target}"].eq(0)
                             & group[f"{target}_lightgbm"].notna()
                             & group[f"{target}_weekly_mean_3w"].notna())
                    selected = group.loc[valid]
                    model_error = selected[f"{target}_lightgbm"] - selected[target]
                    baseline_error = selected[f"{target}_weekly_mean_3w"] - selected[target]
                    rows.append({"holiday": event.holiday, "window_start": event.window_start, "period": period,
                                 "arc": arc, "target": target, "hours": len(group), "count": len(selected),
                                 "lightgbm_mae": model_error.abs().mean() if len(selected) else np.nan,
                                 "baseline_mae": baseline_error.abs().mean() if len(selected) else np.nan,
                                 "lightgbm_bias": model_error.mean() if len(selected) else np.nan})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", type=Path, default=Path("data/processed/traffic_clean.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/stability_analysis"))
    parser.add_argument("--windows", type=int, default=52)
    args = parser.parse_args()
    frame = pd.read_csv(args.traffic, parse_dates=["t_1h"])
    starts = window_starts(frame, args.windows, 7)
    events = holiday_windows(frame)
    all_starts = sorted(set(starts) | set(events.window_start))
    predictions, metrics, importance, training = evaluate_models(frame, all_starts)
    weekly = predictions[predictions.window_start.isin(starts)]
    weekly_metrics = metrics[metrics.window_start.isin(starts)]
    summary = compare_on_common_rows(weekly)
    windows = stability_by_window(weekly)
    holiday_scores = score_holidays(predictions, events)
    args.output.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.output / "weekly_predictions.csv", index=False)
    weekly_metrics.to_csv(args.output / "weekly_metrics.csv", index=False)
    summary.to_csv(args.output / "weekly_comparison.csv", index=False)
    windows.to_csv(args.output / "weekly_window_comparison.csv", index=False)
    events.to_csv(args.output / "holiday_windows.csv", index=False)
    holiday_scores.to_csv(args.output / "holiday_scores.csv", index=False)
    predictions[predictions.window_start.isin(events.window_start)].to_csv(args.output / "holiday_predictions.csv", index=False)
    training.to_csv(args.output / "training_counts.csv", index=False)
    print("\nComparaison hebdomadaire :")
    print(summary.to_string(index=False))
    print("\nJours fériés analysés :", events.holiday.dt.strftime("%Y-%m-%d").tolist())
    print("\nSorties :", args.output)


if __name__ == "__main__":
    main()
