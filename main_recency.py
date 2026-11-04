"""Compare l'historique complet, deux fenêtres glissantes et une pondération temporelle."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from main_stability import holiday_windows
from src.evaluate import build_window, window_starts
from src.features import ARCS
from src.model import fit_model, multi_horizon_training_features, predict_model

STRATEGIES = ("all", "months_3", "months_6", "decay_180d")
HALF_LIFE_DAYS = 180.0


def select_training(features: pd.DataFrame, cutoff: pd.Timestamp, strategy: str):
    """Sélectionne les cibles connues ; les caractéristiques peuvent utiliser l'historique antérieur."""
    cutoff = pd.Timestamp(cutoff)
    if strategy not in STRATEGIES:
        raise ValueError(f"Stratégie inconnue : {strategy}")
    eligible = features.t_1h <= cutoff
    if strategy in ("months_3", "months_6"):
        months = 3 if strategy == "months_3" else 6
        eligible &= features.t_1h > cutoff - pd.DateOffset(months=months)
    train = features.loc[eligible].copy()
    if strategy != "decay_180d":
        return train, None
    age_days = (cutoff - train.t_1h).dt.total_seconds() / 86400
    weights = np.exp(-np.log(2) * age_days / HALF_LIFE_DAYS)
    # La moyenne est normalisée dans fit_strategy après filtrage des cibles réelles.
    return train, weights


def fit_strategy(features: pd.DataFrame, cutoff: pd.Timestamp, strategy: str, target: str):
    train, weights = select_training(features, cutoff, strategy)
    observed = train[target].notna() & train[f"is_imputed_{target}"].eq(0)
    if weights is not None:
        weights = weights / weights.loc[observed].mean()
    model, n = fit_model(train, target, multi_horizon=True, sample_weight=weights)
    rows = []
    for arc in ARCS:
        group = train.loc[observed & train.arc.eq(arc)]
        rows.append({"strategy": strategy, "target": target, "arc": arc,
                     "examples": len(group), "observed_hours": group.t_1h.nunique(),
                     "first_observation": group.t_1h.min(), "last_observation": group.t_1h.max()})
    return model, n, rows


def evaluate(frame: pd.DataFrame, starts: list[pd.Timestamp]):
    predictions, all_counts = [], []
    for start in starts:
        cutoff = start - pd.Timedelta(hours=1)
        print(f"Fenêtre {start.date()} : quatre stratégies d'ancienneté", flush=True)
        history = multi_horizon_training_features(frame, cutoff)
        window = build_window(frame, start)
        for target in ("q", "k"):
            for strategy in STRATEGIES:
                model, _, counts = fit_strategy(history, cutoff, strategy, target)
                col = f"{target}_{strategy}"
                window[col] = predict_model(model, window, target, multi_horizon=True)
                arcs_with_labels = {row["arc"] for row in counts if row["observed_hours"] > 0}
                # Un modèle global n'a pas appris un arc absent des cibles sélectionnées.
                window.loc[~window.arc.isin(arcs_with_labels), col] = np.nan
                for row in counts:
                    all_counts.append({"window_start": start, "cutoff": cutoff, **row})
        window["horizon_day"] = (window.horizon_hours - 1) // 24 + 1
        predictions.append(window)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(all_counts)


def score(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Compare toutes les stratégies sur une intersection commune d'observations."""
    rows = []
    for values, group in frame.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        identifiers = dict(zip(keys, values))
        for target in ("q", "k"):
            observed = group[target].notna() & group[f"is_imputed_{target}"].eq(0)
            cols = [f"{target}_{strategy}" for strategy in STRATEGIES]
            valid = observed & group[cols].notna().all(axis=1)
            row = {**identifiers, "target": target, "observed": int(observed.sum()),
                   "scored": int(valid.sum())}
            for strategy, col in zip(STRATEGIES, cols):
                error = group.loc[valid, col] - group.loc[valid, target]
                row[f"{strategy}_mae"] = error.abs().mean() if len(error) else np.nan
                row[f"{strategy}_rmse"] = np.sqrt((error ** 2).mean()) if len(error) else np.nan
                row[f"{strategy}_bias"] = error.mean() if len(error) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def event_scores(predictions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for event in events.itertuples(index=False):
        window = predictions[predictions.window_start == event.window_start]
        dates = window.t_1h.dt.normalize()
        for period, date in (("veille", event.holiday - pd.Timedelta(days=1)),
                             ("ferie", event.holiday), ("lendemain", event.holiday + pd.Timedelta(days=1))):
            part = window.loc[dates == date].copy()
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
    parser.add_argument("--output", type=Path, default=Path("data/processed/recency_analysis"))
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--step-days", type=int, default=7)
    args = parser.parse_args()
    frame = pd.read_csv(args.traffic, parse_dates=["t_1h"])
    starts = window_starts(frame, args.windows, args.step_days)
    events = holiday_windows(frame)
    all_starts = sorted(set(starts) | set(events.window_start))
    predictions, counts = evaluate(frame, all_starts)
    recent = predictions[predictions.window_start.isin(starts)]
    holiday = predictions[predictions.window_start.isin(events.window_start)]
    args.output.mkdir(parents=True, exist_ok=True)
    score(recent, ["arc"]).to_csv(args.output / "recent_summary.csv", index=False)
    score(recent, ["window_start", "arc"]).to_csv(args.output / "recent_windows.csv", index=False)
    score(recent, ["arc", "horizon_day"]).to_csv(args.output / "recent_horizons.csv", index=False)
    event_scores(predictions, events).to_csv(args.output / "holidays.csv", index=False)
    events.to_csv(args.output / "holiday_windows.csv", index=False)
    counts.to_csv(args.output / "training_counts.csv", index=False)
    recent.to_csv(args.output / "recent_predictions.csv", index=False)
    holiday.to_csv(args.output / "holiday_predictions.csv", index=False)
    print("\nComparaison sur les mêmes observations :")
    print(score(recent, ["arc"]).to_string(index=False))
    print(f"\nRésultats : {args.output}")


if __name__ == "__main__":
    main()
