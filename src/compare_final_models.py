#!/usr/bin/env python3
"""
Compare final model artifacts and run paired significance checks.

Expected inputs:
    results/<model>/<tag>_predictions.csv
    results/<model>/<tag>_metrics.json

Default models: logreg, rf, adaboost, xgboost.

Examples:
    python src/compare_final_models.py --tag final
    python src/compare_final_models.py --tag final --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


LABELS = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "adaboost": "AdaBoost",
    "xgboost": "XGBoost",
}


def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks for DeLong's ROC-AUC covariance estimate."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def fast_delong(preds_sorted: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """
    DeLong covariance for one or more classifiers.

    preds_sorted shape is (n_classifiers, n_examples), with positives first.
    """
    n_neg = preds_sorted.shape[1] - n_pos
    pos = preds_sorted[:, :n_pos]
    neg = preds_sorted[:, n_pos:]
    k = preds_sorted.shape[0]

    tx = np.empty((k, n_pos), dtype=float)
    ty = np.empty((k, n_neg), dtype=float)
    tz = np.empty((k, n_pos + n_neg), dtype=float)

    for r in range(k):
        tx[r] = compute_midrank(pos[r])
        ty[r] = compute_midrank(neg[r])
        tz[r] = compute_midrank(preds_sorted[r])

    aucs = tz[:, :n_pos].sum(axis=1) / n_pos / n_neg - (n_pos + 1) / (2 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / n_pos + sy / n_neg
    cov = np.atleast_2d(cov)
    return aucs, cov


def delong_roc_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    """Two-sided DeLong test for paired ROC-AUC difference."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {"auc_a": np.nan, "auc_b": np.nan, "auc_diff": np.nan, "z": np.nan, "p": np.nan}

    preds = np.vstack([prob_a, prob_b])[:, order]
    aucs, cov = fast_delong(preds, n_pos)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if var <= 0:
        z = np.nan
        p = np.nan
    else:
        z = diff / np.sqrt(var)
        p = 2 * stats.norm.sf(abs(z))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": diff,
        "z": float(z) if np.isfinite(z) else np.nan,
        "p": float(p) if np.isfinite(p) else np.nan,
    }


def per_user_f1(df: pd.DataFrame) -> pd.Series:
    rows = []
    for user_id, grp in df.groupby("user_id", sort=True):
        rows.append((user_id, f1_score(grp["label"], grp["pred"], zero_division=0)))
    return pd.Series(dict(rows), name="per_order_f1").sort_index()


def bootstrap_user_f1_diff(
    a: pd.Series,
    b: pd.Series,
    n_boot: int,
    seed: int,
) -> dict:
    """Paired bootstrap CI for mean per-user F1 difference, A - B."""
    users = a.index.intersection(b.index)
    diff = (a.loc[users] - b.loc[users]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    n = len(diff)
    for i in range(n_boot):
        boot[i] = diff[rng.integers(0, n, size=n)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant_95": bool(lo > 0 or hi < 0),
    }


def load_predictions(root: Path, models: list[str], tag: str) -> dict[str, pd.DataFrame]:
    preds = {}
    base_keys = None
    for model in models:
        path = root / "results" / model / f"{tag}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = (
            pd.read_csv(path)
            .sort_values(["user_id", "product_id"])
            .reset_index(drop=True)
        )
        required = {"user_id", "product_id", "label", "prob", "pred"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        keys = df[["user_id", "product_id", "label"]]
        if base_keys is None:
            base_keys = keys
        elif not base_keys.equals(keys):
            raise ValueError(f"{model} predictions do not align with previous models")
        preds[model] = df
    return preds


def load_metric_threshold(root: Path, model: str, tag: str) -> float | None:
    path = root / "results" / model / f"{tag}_metrics.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload.get("metrics", {}).get("threshold")


def build_metric_table(root: Path, preds: dict[str, pd.DataFrame], tag: str) -> pd.DataFrame:
    rows = []
    for model, df in preds.items():
        y = df["label"].to_numpy()
        p = df["prob"].to_numpy()
        pred = df["pred"].to_numpy()
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        rows.append({
            "model": model,
            "label": LABELS.get(model, model),
            "tag": tag,
            "threshold": load_metric_threshold(root, model, tag),
            "n_test": int(len(df)),
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "per_order_f1": per_user_f1(df).mean(),
            "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "miss_rate": fn / n_pos if n_pos else np.nan,
            "false_alarm_rate": fp / n_neg if n_neg else np.nan,
        })
    return pd.DataFrame(rows).sort_values("f1", ascending=False)


def build_mcnemar_table(preds: dict[str, pd.DataFrame], alpha: float) -> pd.DataFrame:
    rows = []
    n_pairs = len(list(combinations(preds, 2)))
    bonf_alpha = alpha / n_pairs
    for a, b in combinations(preds, 2):
        da = preds[a]
        db = preds[b]
        y = da["label"].to_numpy()
        a_correct = da["pred"].to_numpy() == y
        b_correct = db["pred"].to_numpy() == y
        a_right_b_wrong = int((a_correct & ~b_correct).sum())
        a_wrong_b_right = int((~a_correct & b_correct).sum())
        n = a_right_b_wrong + a_wrong_b_right
        if n == 0:
            stat = np.nan
            p = np.nan
        else:
            stat = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / n
            p = stats.chi2.sf(stat, df=1)
        rows.append({
            "model_a": a,
            "model_b": b,
            "a_right_b_wrong": a_right_b_wrong,
            "a_wrong_b_right": a_wrong_b_right,
            "mcnemar_chi2": stat,
            "p": p,
            "bonferroni_alpha": bonf_alpha,
            "significant_bonferroni": bool(np.isfinite(p) and p < bonf_alpha),
        })
    return pd.DataFrame(rows)


def build_bootstrap_table(
    preds: dict[str, pd.DataFrame],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    per_user = {model: per_user_f1(df) for model, df in preds.items()}
    rows = []
    for i, (a, b) in enumerate(combinations(preds, 2)):
        result = bootstrap_user_f1_diff(
            per_user[a],
            per_user[b],
            n_boot=n_boot,
            seed=seed + i,
        )
        rows.append({
            "model_a": a,
            "model_b": b,
            "metric": "mean_per_order_f1_diff_a_minus_b",
            **result,
        })
    return pd.DataFrame(rows)


def build_delong_table(preds: dict[str, pd.DataFrame], alpha: float) -> pd.DataFrame:
    rows = []
    n_pairs = len(list(combinations(preds, 2)))
    bonf_alpha = alpha / n_pairs
    for a, b in combinations(preds, 2):
        da = preds[a]
        db = preds[b]
        result = delong_roc_test(
            da["label"].to_numpy(),
            da["prob"].to_numpy(),
            db["prob"].to_numpy(),
        )
        rows.append({
            "model_a": a,
            "model_b": b,
            **result,
            "bonferroni_alpha": bonf_alpha,
            "significant_bonferroni": bool(
                np.isfinite(result["p"]) and result["p"] < bonf_alpha
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root containing results/ and models/.",
    )
    parser.add_argument("--tag", default="final")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "rf", "adaboost", "xgboost"],
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    preds = load_predictions(args.root, args.models, args.tag)
    out_dir = args.root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = build_metric_table(args.root, preds, args.tag)
    mcnemar = build_mcnemar_table(preds, args.alpha)
    bootstrap = build_bootstrap_table(preds, args.n_boot, args.seed)
    delong = build_delong_table(preds, args.alpha)

    metric_path = out_dir / f"{args.tag}_model_comparison.csv"
    mcnemar_path = out_dir / f"{args.tag}_mcnemar_bonferroni.csv"
    bootstrap_path = out_dir / f"{args.tag}_bootstrap_f1_diff.csv"
    delong_path = out_dir / f"{args.tag}_delong_roc_auc.csv"

    metrics.to_csv(metric_path, index=False)
    mcnemar.to_csv(mcnemar_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    delong.to_csv(delong_path, index=False)

    print("\nModel ranking by thresholded F1:")
    cols = [
        "label", "threshold", "f1", "per_order_f1", "precision", "recall",
        "roc_auc", "pr_auc", "miss_rate", "false_alarm_rate",
    ]
    print(metrics[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best = metrics.iloc[0]
    print(f"\nBest by F1: {best['label']} ({best['f1']:.4f})")
    print("\nSaved:")
    for path in [metric_path, mcnemar_path, bootstrap_path, delong_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
