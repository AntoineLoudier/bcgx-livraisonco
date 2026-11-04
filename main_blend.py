"""Mélange LightGBM/moyenne hebdomadaire avec poids choisis sur fenêtres passées."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import WINDOW_HOURS, build_window, window_starts
from src.features import FORECAST_CUTOFF
from src.model import fit_model, multi_horizon_training_features, predict_model

# Priorité au modèle en cas d'égalité des erreurs de calibration.
MODEL_WEIGHTS = (1.0, 0.75, 0.5, 0.25, 0.0)


def blended(model, baseline, model_weight):
    """Poids du modèle ; utiliser ce dernier seul si la référence manque."""
    if not 0 <= model_weight <= 1:
        raise ValueError("Le poids doit être entre 0 et 1")
    model = np.asarray(model, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    return np.where(np.isfinite(baseline), model_weight * model + (1 - model_weight) * baseline, model)


def choose_weight(history, target, cutoff, min_windows=4, min_observations=100):
    """Choix sur des fenêtres entièrement terminées et des cibles observées."""
    if target not in ("q", "k"):
        raise ValueError("La cible doit être q ou k")
    if min_windows < 1 or min_observations < 1:
        raise ValueError("Les minima de calibration doivent être positifs")
    cutoff = pd.Timestamp(cutoff)
    if history.empty:
        return None
    completed = (history.window_start + pd.Timedelta(hours=WINDOW_HOURS - 1) <= cutoff)
    observed = history[target].notna() & history[f"is_imputed_{target}"].eq(0)
    valid = completed & (history.t_1h <= cutoff) & observed & np.isfinite(history[f"{target}_model"])
    past = history.loc[valid]
    windows = past.window_start.nunique()
    if windows < min_windows or len(past) < min_observations:
        return None
    model = past[f"{target}_model"].to_numpy()
    baseline = past[f"{target}_weekly_mean_3w"].to_numpy()
    truth = past[target].to_numpy()
    losses = [float(np.mean(np.abs(blended(model, baseline, weight) - truth))) for weight in MODEL_WEIGHTS]
    index = int(np.argmin(losses))
    return {"model_weight": MODEL_WEIGHTS[index], "baseline_weight": 1 - MODEL_WEIGHTS[index],
            "calibration_windows": int(windows), "calibration_observations": int(len(past)),
            "baseline_available": int(np.isfinite(baseline).sum()), "calibration_mae": losses[index]}


def walk_forward(frame, starts, min_windows=4, min_observations=100):
    """Réentraîne les modèles à chaque émission et calibre le mélange en ligne."""
    history, weights = [], []
    for start in sorted(starts):
        cutoff = start - pd.Timedelta(hours=1)
        print(f"Fenêtre {start.date()} : entraînement et calibration passée", flush=True)
        previous = pd.concat(history, ignore_index=True) if history else pd.DataFrame()
        train = multi_horizon_training_features(frame, cutoff)
        window = build_window(frame, start)
        window["horizon_day"] = (window.horizon_hours - 1) // 24 + 1
        for target in ("q", "k"):
            selected = choose_weight(previous, target, cutoff, min_windows, min_observations)
            model, n = fit_model(train, target, multi_horizon=True)
            window[f"{target}_model"] = predict_model(model, window, target, multi_horizon=True)
            window[f"{target}_blend"] = np.nan
            weight_row = {"window_start": start, "cutoff": cutoff, "target": target,
                          "training_examples": n, "model_weight": np.nan, "baseline_weight": np.nan,
                          "calibration_windows": 0, "calibration_observations": 0,
                          "baseline_available": 0, "calibration_mae": np.nan}
            if selected is not None:
                weight_row.update(selected)
                window[f"{target}_blend"] = blended(window[f"{target}_model"],
                                                      window[f"{target}_weekly_mean_3w"], selected["model_weight"])
            weights.append(weight_row)
        history.append(window)
    return pd.concat(history, ignore_index=True), pd.DataFrame(weights)


def scores(frame, keys):
    """Le modèle et le mélange utilisent les mêmes cibles ; référence évaluée sur ses lignes disponibles."""
    rows = []
    for values, group in frame.groupby(keys, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        for target in ("q", "k"):
            observed = group[target].notna() & group[f"is_imputed_{target}"].eq(0)
            valid = observed & group[f"{target}_model"].notna() & group[f"{target}_blend"].notna()
            baseline_valid = valid & group[f"{target}_weekly_mean_3w"].notna()
            row = {**dict(zip(keys, values)), "target": target, "observed": int(observed.sum()),
                   "scored": int(valid.sum()), "baseline_available": int(baseline_valid.sum()),
                   "fallback_to_model": int((valid & ~baseline_valid).sum())}
            for name, col, mask in (("model", f"{target}_model", valid),
                                    ("blend", f"{target}_blend", valid),
                                    ("baseline", f"{target}_weekly_mean_3w", baseline_valid),
                                    ("model_on_baseline", f"{target}_model", baseline_valid),
                                    ("blend_on_baseline", f"{target}_blend", baseline_valid)):
                error = group.loc[mask, col] - group.loc[mask, target]
                row[f"{name}_mae"] = error.abs().mean() if len(error) else np.nan
                row[f"{name}_rmse"] = np.sqrt((error ** 2).mean()) if len(error) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def window_diagnostics(window_scores):
    """Résumé des erreurs par fenêtre, sans confondre leurs effectifs."""
    available = window_scores[window_scores.scored > 0].copy()
    available["blend_wins"] = available.blend_mae < available.model_mae
    return available.groupby(["arc", "target"], as_index=False).agg(
        windows=("scored", "size"), blend_wins=("blend_wins", "sum"),
        model_window_mae_median=("model_mae", "median"), blend_window_mae_median=("blend_mae", "median"),
        model_window_mae_max=("model_mae", "max"), blend_window_mae_max=("blend_mae", "max"),
        model_window_mae_p90=("model_mae", lambda s: s.quantile(0.9)),
        blend_window_mae_p90=("blend_mae", lambda s: s.quantile(0.9)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic", type=Path, default=Path("data/processed/traffic_clean.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/blend_analysis"))
    parser.add_argument("--windows", type=int, default=52)
    parser.add_argument("--step-days", type=int, default=7)
    parser.add_argument("--min-calibration-windows", type=int, default=4)
    parser.add_argument("--min-calibration-observations", type=int, default=100)
    args = parser.parse_args()
    frame = pd.read_csv(args.traffic, parse_dates=["t_1h"])
    starts = window_starts(frame, args.windows, args.step_days)
    predictions, weights = walk_forward(frame, starts, args.min_calibration_windows, args.min_calibration_observations)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    weights.to_csv(args.output / "weights.csv", index=False)
    scores(predictions, ["arc"]).to_csv(args.output / "summary.csv", index=False)
    window_scores = scores(predictions, ["window_start", "arc"])
    window_scores.to_csv(args.output / "windows.csv", index=False)
    window_diagnostics(window_scores).to_csv(args.output / "window_diagnostics.csv", index=False)
    scores(predictions, ["arc", "horizon_day"]).to_csv(args.output / "horizons.csv", index=False)
    scores(predictions, ["arc", "is_public_holiday"]).to_csv(args.output / "holiday_context.csv", index=False)
    final = {"forecast_cutoff": str(FORECAST_CUTOFF), "grid_model_weight": MODEL_WEIGHTS,
             "min_calibration_windows": args.min_calibration_windows,
             "min_calibration_observations": args.min_calibration_observations,
             "targets": {target: choose_weight(predictions, target, FORECAST_CUTOFF,
                                                args.min_calibration_windows, args.min_calibration_observations)
                         for target in ("q", "k")}}
    (args.output / "final_weights.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    print("\nRésultats sur les mêmes observations :")
    print(scores(predictions, ["arc"]).to_string(index=False))
    print(f"\nRésultats : {args.output}")


if __name__ == "__main__":
    main()
