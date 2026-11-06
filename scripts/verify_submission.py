"""Check the frozen submission and recalculate its scores, offline, using stdlib only."""

import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "presentation/results"
EVALUATED = {"Convention", "Saint-Pères"}
EXPORTED = EVALUATED | {"Champs-Elysées"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path, delimiter=","):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9), label)


def rmse(rows, target, prediction):
    errors = [(float(row[target]) - float(row[prediction])) ** 2 for row in rows]
    return math.sqrt(math.fsum(errors) / len(errors))


def main():
    manifest = json.loads((ROOT / "rendu/manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    require(set(evidence["evaluated_streets"]) == EVALUATED, "Périmètre d’évaluation incorrect")
    require(manifest["model_revision"] == evidence["model_revision"], "Révisions incohérentes")
    for name, digest in manifest["files_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"Fichier modifié : {name}")
    # The original acquisition may be absent in a clean clone; model source must match.
    for name, digest in evidence["inputs_sha256"].items():
        if name.startswith("src/") or name == "main_train.py":
            require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"Code modèle modifié : {name}")

    submitted = read_csv(ROOT / "rendu/output_Les Architectes.csv", delimiter=";")
    require(len(submitted) == 360, "Le CSV du brief doit contenir 360 lignes")
    require(list(submitted[0]) == ["", "arc", "datetime", "debit_horaire", "taux_occupation"], "Colonnes de soumission incorrectes")
    require(set(row["arc"] for row in submitted) == EXPORTED, "Rues de l’export incorrectes")
    start = datetime(2026, 11, 7)
    hours = {start + timedelta(hours=i) for i in range(120)}
    lookup = {}
    for i, row in enumerate(submitted):
        require(int(row[""]) == i, "Index CSV incohérent")
        date = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M")
        key = row["arc"], date
        require(key not in lookup, f"Prédiction dupliquée : {key}")
        lookup[key] = row
        for col, upper in [("debit_horaire", 8000), ("taux_occupation", 100)]:
            value = float(row[col])
            require(math.isfinite(value) and 0 <= value <= upper, f"Valeur hors bornes : {key}, {col}")
    for arc in EXPORTED:
        require({date for name, date in lookup if name == arc} == hours, f"Grille horaire incorrecte : {arc}")
    forecast = read_csv(RESULTS / "previsions_7_11_novembre_2_rues.csv")
    forecast_keys = {(row["arc"], datetime.fromisoformat(row["t_1h"])) for row in forecast}
    require(len(forecast) == 240 and forecast_keys == {(arc, hour) for arc in EVALUATED for hour in hours}, "Prévisions du support incomplètes ou dupliquées")
    for row in forecast:
        out = lookup[row["arc"], datetime.fromisoformat(row["t_1h"])]
        close(row["pred_q"], out["debit_horaire"], "Écart de débit entre support et CSV")
        close(row["pred_k"], out["taux_occupation"], "Écart d’occupation entre support et CSV")

    observed = read_csv(RESULTS / "predictions_validation_2_rues.csv")
    require(len(observed) == 944, "Nombre de cibles observées incorrect")
    require({row["arc"] for row in observed} == EVALUATED, "Rue hors évaluation dans les scores")
    require(len({(row["arc"], row["t_1h"]) for row in observed}) == len(observed), "Cibles dupliquées")
    require(all(row["is_imputed_q"] == row["is_imputed_k"] == "0" for row in observed), "Cibles imputées dans les scores")
    scores = read_csv(RESULTS / "scores_2_rues.csv")
    previous = {(row["Arc"], row["Target"]): row for row in read_csv(RESULTS / "comparaison_versions.csv")}
    score_keys = {(arc, target) for arc in EVALUATED | {"GLOBAL (2 rues évaluées)"} for target in ["q", "k"]}
    require(len(scores) == 6 and {(row["Arc"], row["Target"]) for row in scores} == score_keys, "Table de scores incomplète")
    require(set(previous) == score_keys, "Comparaison de versions incomplète")
    for score in scores:
        arc, target = score["Arc"], score["Target"]
        rows = observed if arc.startswith("GLOBAL") else [r for r in observed if r["arc"] == arc]
        require(len(rows) == int(score["N"]), f"Effectif incorrect : {arc}")
        for field, column in [("RMSE_Modele", "pred_" + target), ("RMSE_J7_pretraite", "baseline_" + target), ("RMSE_J7_non_lisse", "unadjusted_j7_" + target), ("RMSE_Ancrage", target + "_smart_expected")]:
            close(rmse(rows, target, column), score[field], f"Score incorrect : {arc} {target} {field}")
        current = rmse(rows, target, "pred_" + target)
        baseline = rmse(rows, target, "baseline_" + target)
        old = rmse(rows, target, "previous_" + target)
        close(100 * (1 - current / baseline), score["Gain_vs_J7_pretraite_pct"], "Gain vs référence incorrect")
        old_score = previous[arc, target]
        close(old, old_score["RMSE_ancienne_version"], "Ancienne RMSE incorrecte")
        close(current, old_score["RMSE_nouvelle_version"], "Nouvelle RMSE incorrecte")
        close(100 * (1 - current / old), old_score["Gain_nouvelle_vs_ancienne_pct"], "Gain de version incorrect")

    folds = read_csv(RESULTS / "scores_par_fenetre.csv")
    require(len(folds) == 16, "Nombre de scores par fenêtre incorrect")
    for fold in folds:
        rows = [r for r in observed if r["arc"] == fold["Arc"] and r["fold"] == fold["Fold"]]
        require(len(rows) == int(fold["N"]), "Effectif par fenêtre incorrect")
        dates = [datetime.fromisoformat(r["t_1h"]) for r in rows]
        require(min(dates).isoformat() == fold["Start"] and max(dates).isoformat() == fold["End"], "Dates de fenêtre incorrectes")
        for field, prefix in [("RMSE_Modele", "pred_"), ("RMSE_J7_pretraite", "baseline_"), ("RMSE_ancienne_version", "previous_")]:
            close(rmse(rows, fold["Target"], prefix + fold["Target"]), fold[field], "Score par fenêtre incorrect")

    pdf = ROOT / "presentation/LivraisonCo.pdf"
    require(pdf.read_bytes().startswith(b"%PDF-"), "Présentation PDF absente ou invalide")
    with zipfile.ZipFile(ROOT / "presentation/LivraisonCo.pptx") as deck:
        slides = [n for n in deck.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        notes = [n for n in deck.namelist() if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", n)]
        require(len(slides) == len(notes) == 19, "PowerPoint incomplet : 19 slides et notes attendues")
        mockup = (ROOT / "presentation/assets/maquette_exploitation_livreur.png").read_bytes()
        require(any(deck.read(n) == mockup for n in deck.namelist() if n.startswith("ppt/media/")), "Maquette manquante ou modifiée")
    print("Rendu vérifié : 360 lignes au format du brief, dont 240 évaluées ; 944 cibles observées ; scores concordants ; présentation avec 19 slides et maquette.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        print(f"ÉCHEC : {error}", file=sys.stderr)
        sys.exit(1)
