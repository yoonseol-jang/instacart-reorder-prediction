"""
Validate Phase 4 Slurm outputs before downloading artifacts for Phase 5.

Run on OOD after tune/final jobs finish:
    python src/validate_phase4_artifacts.py --tag final
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS = ["logreg", "rf", "adaboost", "xgboost"]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file() and path.stat().st_size == 0:
        raise ValueError(f"Empty artifact: {path}")
    return path


def check_thresholds(root: Path, low: float, high: float,
                     warn_only: bool) -> dict:
    path = require(root / "results" / "optimal_thresholds.json")
    data = json.loads(path.read_text())
    missing = [model for model in MODELS if model not in data]
    if missing:
        raise ValueError(f"optimal_thresholds.json missing models: {missing}")
    bad = {
        model: float(data[model])
        for model in MODELS
        if not low <= float(data[model]) <= high
    }
    if bad and not warn_only:
        raise ValueError(
            f"Optimal thresholds outside expected [{low}, {high}]: {bad}"
        )
    if bad:
        log.warning("Threshold warning: %s outside [%.2f, %.2f]", bad, low, high)
    return {model: float(data[model]) for model in MODELS}


def check_model_artifacts(root: Path, model: str, tag: str) -> dict:
    results_dir = root / "results" / model
    models_dir = root / "models" / model
    required = {
        "metrics": require(results_dir / f"{tag}_metrics.json"),
        "predictions": require(results_dir / f"{tag}_predictions.csv"),
        "threshold_sweep": require(results_dir / f"{tag}_threshold_sweep.csv"),
        "validation_threshold_sweep": require(
            results_dir / f"{tag}_validation_threshold_sweep.csv"
        ),
        "best_model": require(models_dir / f"{model}_best.pkl"),
        "feature_importance": require(
            results_dir / f"feature_importance_{model}.csv"
        ),
    }
    tuning_path = results_dir / f"tuning_history_{model}.csv"
    if tag in {"tune", "final"}:
        require(tuning_path)

    metrics = json.loads(required["metrics"].read_text())
    if metrics["config"].get("threshold_source") != "grouped_validation_split":
        raise ValueError(
            f"{required['metrics']} did not record grouped validation thresholding"
        )
    predictions = pd.read_csv(required["predictions"], nrows=5)
    expected_pred_cols = {"user_id", "product_id", "label", "prob", "pred"}
    if not expected_pred_cols.issubset(predictions.columns):
        raise ValueError(
            f"{required['predictions']} missing prediction columns: "
            f"{sorted(expected_pred_cols.difference(predictions.columns))}"
        )
    importances = pd.read_csv(required["feature_importance"])
    if list(importances.columns) != ["feature_name", "importance_score"]:
        raise ValueError(
            f"{required['feature_importance']} must have feature_name,importance_score"
        )
    joblib.load(required["best_model"])
    return {name: str(path) for name, path in required.items()}


def check_baseline(root: Path, tag: str) -> float:
    path = require(root / "results" / "last_basket" / f"{tag}_metrics.json")
    payload = json.loads(path.read_text())
    return float(payload["metrics"]["f1"])


def check_model_vs_baseline(root: Path, tag: str) -> dict:
    baseline_f1 = check_baseline(root, tag)
    report = {"last_basket": baseline_f1}
    failures = {}
    for model in MODELS:
        path = require(root / "results" / model / f"{tag}_metrics.json")
        f1 = float(json.loads(path.read_text())["metrics"]["f1"])
        report[model] = f1
        if f1 <= baseline_f1:
            failures[model] = f1
    if failures:
        raise ValueError(
            f"Models failed to beat last-basket F1={baseline_f1:.4f}: {failures}"
        )
    return report


def run(args) -> dict:
    report = {
        "thresholds": check_thresholds(
            args.root, args.threshold_low, args.threshold_high,
            args.warn_threshold_only
        ),
        "models": {
            model: check_model_artifacts(args.root, model, args.tag)
            for model in MODELS
        },
        "model_vs_last_basket": check_model_vs_baseline(args.root, args.tag),
    }
    out_path = args.root / "results" / "phase4_artifact_validation.json"
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Phase 4 artifact validation passed → %s", out_path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--tag", default="final")
    parser.add_argument("--threshold-low", type=float, default=0.15)
    parser.add_argument("--threshold-high", type=float, default=0.25)
    parser.add_argument("--warn-threshold-only", action="store_true")
    run(parser.parse_args())
