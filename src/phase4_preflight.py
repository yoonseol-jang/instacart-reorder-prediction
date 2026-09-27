"""
Phase 4 preflight checks for OOD interactive sessions.

Run after Phase 2/3 checks and smoke jobs:
    python src/phase4_preflight.py

The checks are intentionally strict enough to stop suspicious training runs
before Slurm tuning jobs spend hours on bad inputs.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from train_utils import (
    best_threshold,
    group_train_valid_split,
    load_data,
    make_preprocessor,
    threshold_sweep,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
MODEL_NAMES = ["logreg", "rf", "adaboost", "xgboost"]


def check_imports(require_optuna: bool = True) -> dict:
    import numpy
    import pandas
    import sklearn
    import xgboost

    versions = {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
    }
    if require_optuna:
        import optuna
        versions["optuna"] = optuna.__version__
    log.info("Import check passed: %s", versions)
    return versions


def check_parquet_reads(data_dir: Path) -> dict:
    train = pd.read_parquet(data_dir / "train.parquet", columns=["user_id", "label"])
    test = pd.read_parquet(data_dir / "test.parquet", columns=["user_id", "label"])
    feature_count = len((data_dir / "feature_cols.txt").read_text().splitlines())
    report = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive_rate": float(train["label"].mean()),
        "test_positive_rate": float(test["label"].mean()),
        "feature_count": int(feature_count),
    }
    log.info("Parquet read check passed: %s", report)
    return report


def leakage_fold1_logreg_check(
    data_dir: Path,
    max_train_rows: int | None,
    seed: int,
    fail_f1: float,
) -> dict:
    train, _, feat_cols = load_data(data_dir, max_train_rows=max_train_rows, seed=seed)
    X = train[feat_cols]
    y = train["label"].to_numpy()
    groups = train["user_id"].to_numpy()
    fold1_train_idx, fold1_valid_idx = next(GroupKFold(n_splits=5).split(X, y, groups))

    pipe = Pipeline([
        ("pre", make_preprocessor("logreg", feat_cols)),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="saga",
            max_iter=1000,
            random_state=seed,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X.iloc[fold1_train_idx], y[fold1_train_idx])
    valid_prob = pipe.predict_proba(X.iloc[fold1_valid_idx])[:, 1]
    sweep_df = threshold_sweep(y[fold1_valid_idx], valid_prob)
    threshold = best_threshold(sweep_df, "f1")
    valid_f1 = f1_score(
        y[fold1_valid_idx],
        (valid_prob >= threshold).astype(int),
        zero_division=0,
    )
    if valid_f1 > fail_f1:
        raise ValueError(
            f"Fold-1 Logistic Regression validation F1={valid_f1:.4f} "
            f"> {fail_f1:.2f}; stop and inspect leakage."
        )
    report = {"fold1_valid_f1": float(valid_f1), "threshold": float(threshold)}
    log.info("Leakage F1 check passed: %s", report)
    return report


def xgboost_inner_validation_source_check(src_dir: Path) -> dict:
    path = src_dir / "train_xgboost.py"
    text = path.read_text()
    ast.parse(text)
    required = [
        "Xi_tr, Xi_va, yi_tr, yi_va = tts(",
        "eval_set=[(Xi_va, yi_va)]",
        "y_prob_va  = clf.predict_proba(Xf_va)[:, 1]",
        "_f1(yf_va, (y_prob_va >= t).astype(int)",
    ]
    missing = [pattern for pattern in required if pattern not in text]
    if missing:
        raise ValueError(
            "XGBoost inner early-stopping independence check failed. "
            f"Missing source patterns: {missing}"
        )
    forbidden = ["eval_set=[(Xf_va", "eval_set=[(X_train", "eval_set=[(X_test"]
    present_forbidden = [pattern for pattern in forbidden if pattern in text]
    if present_forbidden:
        raise ValueError(
            "XGBoost early stopping appears to use outer/test data: "
            f"{present_forbidden}"
        )
    report = {"source": str(path), "inner_eval_set": "Xi_va", "outer_score": "Xf_va"}
    log.info("XGBoost inner validation source check passed")
    return report


def load_metrics(results_dir: Path, model: str, tag: str) -> dict:
    path = results_dir / model / f"{tag}_metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the interactive smoke job for {model} first."
        )
    return json.loads(path.read_text())


def threshold_sanity_check(
    results_dir: Path,
    tag: str,
    low: float,
    high: float,
    warn_only: bool,
) -> dict:
    report = {}
    bad = {}
    for model in MODEL_NAMES:
        payload = load_metrics(results_dir, model, tag)
        threshold = float(payload["metrics"]["threshold"])
        report[model] = threshold
        if not low <= threshold <= high:
            bad[model] = threshold
    if bad and not warn_only:
        raise ValueError(
            f"Threshold sanity failed; expected ML thresholds in [{low}, {high}], "
            f"got {bad}."
        )
    if bad:
        log.warning("Threshold sanity warning: %s outside [%.2f, %.2f]",
                    bad, low, high)
    else:
        log.info("Threshold sanity check passed: %s", report)
    return report


def smoke_artifact_check(results_dir: Path, tag: str) -> dict:
    report = {}
    for model in ["last_basket", *MODEL_NAMES]:
        model_dir = results_dir / model
        required = [
            model_dir / f"{tag}_metrics.json",
            model_dir / f"{tag}_predictions.csv",
            model_dir / f"{tag}_threshold_sweep.csv",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing smoke artifacts for {model}: {missing}"
            )
        report[model] = [path.name for path in required]
    log.info("Smoke artifact check passed")
    return report


def model_vs_last_basket_check(results_dir: Path, tag: str) -> dict:
    baseline_payload = load_metrics(results_dir, "last_basket", tag)
    baseline_f1 = float(baseline_payload["metrics"]["f1"])
    report = {"last_basket": baseline_f1}
    failures = {}
    for model in MODEL_NAMES:
        payload = load_metrics(results_dir, model, tag)
        f1 = float(payload["metrics"]["f1"])
        report[model] = f1
        if f1 <= baseline_f1:
            failures[model] = f1
    if failures:
        raise ValueError(
            f"ML smoke models did not beat last-basket F1={baseline_f1:.4f}: "
            f"{failures}"
        )
    log.info("ML-vs-last-basket check passed: %s", report)
    return report


def validation_split_smoke(data_dir: Path, max_train_rows: int | None,
                           seed: int) -> dict:
    train, _, _ = load_data(data_dir, max_train_rows=max_train_rows, seed=seed)
    fit_df, valid_df = group_train_valid_split(train, seed=seed)
    return {
        "fit_users": int(fit_df["user_id"].nunique()),
        "valid_users": int(valid_df["user_id"].nunique()),
        "user_overlap": int(
            len(set(fit_df["user_id"]).intersection(set(valid_df["user_id"])))
        ),
    }


def run(args) -> dict:
    report = {
        "imports": check_imports(require_optuna=not args.skip_optuna_check),
        "parquet": check_parquet_reads(args.data_dir),
        "validation_split": validation_split_smoke(
            args.data_dir, args.max_train_rows, args.seed
        ),
        "fold1_logreg_leakage": leakage_fold1_logreg_check(
            args.data_dir, args.max_train_rows, args.seed, args.fail_f1
        ),
        "xgboost_early_stopping": xgboost_inner_validation_source_check(
            Path(__file__).resolve().parent
        ),
        "smoke_artifacts": smoke_artifact_check(args.results_dir, args.tag),
        "thresholds": threshold_sanity_check(
            args.results_dir,
            args.tag,
            args.threshold_low,
            args.threshold_high,
            args.warn_threshold_only,
        ),
        "model_vs_last_basket": model_vs_last_basket_check(
            args.results_dir, args.tag
        ),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "phase4_preflight.json"
    out_path.write_text(json.dumps(report, indent=2))
    log.info("Phase 4 preflight passed → %s", out_path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--max-train-rows", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-f1", type=float, default=0.60)
    parser.add_argument("--threshold-low", type=float, default=0.15)
    parser.add_argument("--threshold-high", type=float, default=0.25)
    parser.add_argument(
        "--warn-threshold-only",
        action="store_true",
        help="Log out-of-range thresholds instead of failing preflight.",
    )
    parser.add_argument(
        "--skip-optuna-check",
        action="store_true",
        help="Skip optuna import check. Use only for local code-path tests.",
    )
    run(parser.parse_args())
