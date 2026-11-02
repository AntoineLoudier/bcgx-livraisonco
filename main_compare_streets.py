"""Compare des modèles LightGBM par rue au modèle global sur des fenêtres temporelles."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from main_stability import holiday_windows
from src.evaluate import build_window, window_starts
from src.features import ARCS, FORECAST_CUTOFF
from src.model import feature_columns, fit_model, predict_model, training_features


def evaluate_windows(frame: pd.DataFrame, starts: list[pd.Timestamp]):
    results, counts = [], []
    for start in starts:
        cutoff = start - pd.Timedelta(hours=1)
        print(f"Fenêtre {start.date()} : modèles globaux et par rue", flush=True)
        history = training_features(frame, cutoff)
        window = build_window(frame, start)
        for target in ("q", "k"):
            global_model, size = fit_model(history, target)
            window[f"{target}_global"] = predict_model(global_model, window, target)
            window[f"{target}_street"] = np.nan
            counts.append({"window_start": start, "arc": "global", "target": target, "training_rows": size})
            for arc in ARCS:
                local_history = history[history.arc == arc]
                n = int((local_history[target].notna() & local_history[f"is_imputed_{target}"].eq(0)).sum())
                counts.append({"window_start": start, "arc": arc, "target": target, "training_rows": n})
                if n < 100:
                    continue
                model, _ = fit_model(local_history, target)
                selected = window.arc == arc
                window.loc[selected, f"{target}_street"] = predict_model(model, window.loc[selected], target)
        results.append(window)
    return pd.concat(results, ignore_index=True), pd.DataFrame(counts)


def score_groups(predictions: pd.DataFrame, extra_keys: list[str]) -> pd.DataFrame:
    """Scores des modèles et de la référence, sur les mêmes cibles observées."""
    rows = []
    keys = extra_keys + ["arc"]
    for values, group in predictions.groupby(keys, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        identifiers = dict(zip(keys, values))
        for target in ("q", "k"):
            columns = [f"{target}_global", f"{target}_street", f"{target}_weekly_mean_3w"]
            observed = group[target].notna() & group[f"is_imputed_{target}"].eq(0)
            valid = observed & group[columns].notna().all(axis=1)
            row = {**identifiers, "target": target, "hours": len(group), "observed": int(observed.sum()),
                   "count": int(valid.sum())}
            for name, col in zip(("global", "street", "baseline"), columns):
                errors = group.loc[valid, col] - group.loc[valid, target]
                row[f"{name}_mae"] = errors.abs().mean() if len(errors) else np.nan
                row[f"{name}_rmse"] = np.sqrt((errors ** 2).mean()) if len(errors) else np.nan
                row[f"{name}_bias"] = errors.mean() if len(errors) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def event_scores(predictions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    data = []
    for event in events.itertuples(index=False):
        group = predictions[predictions.window_start == event.window_start].copy()
        dates = group.t_1h.dt.normalize()
        for period, day in (("veille", event.holiday - pd.Timedelta(days=1)),
                            ("ferie", event.holiday), ("lendemain", event.holiday + pd.Timedelta(days=1))):
            part = group.loc[dates.eq(day)].copy()
            if part.empty:
                continue
            part["holiday"] = event.holiday
            part["period"] = period
            data.append(part)
    if not data:
        return pd.DataFrame()
    return score_groups(pd.concat(data, ignore_index=True), ["holiday", "period"])


def save_final_street_models(frame: pd.DataFrame, output: Path):
    cutoff = min(pd.Timestamp(FORECAST_CUTOFF), frame.t_1h.max())
    history = training_features(frame, cutoff)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"cutoff": str(cutoff), "targets": {}}
    for arc in ARCS:
        metadata["targets"][arc] = {}
        for target in ("q", "k"):
            local = history[history.arc == arc]
            observed = local[target].notna() & local[f"is_imputed_{target}"].eq(0)
            last = local.loc[observed, "t_1h"].max()
            n = int(observed.sum())
            if n < 100:
                metadata["targets"][arc][target] = {"training_rows": n, "model": None}
                continue
            model, n = fit_model(local, target)
            arc_id = int(local.arc_id.iloc[0])
            name = f"lightgbm_arc_{arc_id}_{target}.txt"
            model.booster_.save_model(str(output / name))
            metadata["targets"][arc][target] = {"training_rows": n, "last_observation": str(last),
                                                "model": name, "features": feature_columns(target)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", type=Path, default=Path("data/processed/traffic_clean.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/street_models_analysis"))
    parser.add_argument("--models", type=Path, default=Path("models/per_street"))
    parser.add_argument("--windows", type=int, default=52)
    args = parser.parse_args()
    frame = pd.read_csv(args.traffic, parse_dates=["t_1h"])
    starts = window_starts(frame, args.windows, 7)
    events = holiday_windows(frame)
    all_starts = sorted(set(starts) | set(events.window_start))
    prediction, counts = evaluate_windows(frame, all_starts)
    weekly = prediction[prediction.window_start.isin(starts)]
    summary = score_groups(weekly, [])
    per_window = score_groups(weekly, ["window_start"])
    holidays = event_scores(prediction, events)
    args.output.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.output / "weekly_predictions.csv", index=False)
    summary.to_csv(args.output / "weekly_summary.csv", index=False)
    per_window.to_csv(args.output / "weekly_window_scores.csv", index=False)
    holidays.to_csv(args.output / "holiday_scores.csv", index=False)
    prediction[prediction.window_start.isin(events.window_start)].to_csv(args.output / "holiday_predictions.csv", index=False)
    counts.to_csv(args.output / "training_counts.csv", index=False)
    save_final_street_models(frame, args.models)
    print("\nScores hebdomadaires sur les mêmes observations :")
    print(summary.to_string(index=False))
    print(f"\nRésultats : {args.output} ; modèles par rue : {args.models}")


if __name__ == "__main__":
    main()
