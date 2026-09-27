#!/usr/bin/env python3
"""
Generate full-test predictions for a saved final model.

This is meant for the "train on the controlled/capped setting, then evaluate on
the full held-out test parquet" workflow. It streams test.parquet in batches so
the full test set does not need to be loaded at once.

Examples:
    python src/predict_full_test.py --model rf --source-tag final --out-tag final_fulltest
    python src/predict_full_test.py --model xgboost --source-tag final --out-tag final_fulltest
    python src/predict_full_test.py --compare-xgboost-scale \
        --base-tag final_fulltest --scaled-tag final_fulltrain_fulltest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from compare_final_models import (
    bootstrap_user_f1_diff,
    delong_roc_test,
    per_user_f1,
)


def load_threshold(root: Path, model: str, source_tag: str) -> float:
    """Prefer the threshold actually recorded in the source run metrics."""
    metrics_path = root / "results" / model / f"{source_tag}_metrics.json"
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text())
        threshold = payload.get("metrics", {}).get("threshold")
        if threshold is not None:
            return float(threshold)

    threshold_path = root / "results" / "optimal_thresholds.json"
    if threshold_path.exists():
        payload = json.loads(threshold_path.read_text())
        if model in payload:
            return float(payload[model])

    raise FileNotFoundError(
        f"No threshold found in {metrics_path} or {threshold_path} for {model}"
    )


def default_model_path(root: Path, model: str, source_tag: str) -> Path:
    best_path = root / "models" / model / f"{model}_best.pkl"
    tag_path = root / "models" / model / f"{source_tag}_model.pkl"
    if best_path.exists():
        return best_path
    return tag_path


def run(args: argparse.Namespace) -> dict:
    root = args.root.resolve()
    data_dir = args.data_dir or (root / "data" / "processed")
    out_dir = root / "results" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path or default_model_path(root, args.model, args.source_tag)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    model = joblib.load(model_path)
    threshold = args.threshold
    if threshold is None:
        threshold = load_threshold(root, args.model, args.source_tag)

    feature_cols = (data_dir / "feature_cols.txt").read_text().splitlines()
    cols = ["user_id", "product_id", "label"] + feature_cols

    pred_path = out_dir / f"{args.out_tag}_predictions.csv"
    metrics_path = out_dir / f"{args.out_tag}_metrics.json"

    ys = []
    ps = []
    first = True

    parquet = pq.ParquetFile(data_dir / "test.parquet")
    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=cols):
        df = batch.to_pandas()
        prob = model.predict_proba(df[feature_cols])[:, 1]
        pred = (prob >= threshold).astype(int)

        out = df[["user_id", "product_id", "label"]].copy()
        out["prob"] = np.round(prob, 6)
        out["pred"] = pred
        out.to_csv(pred_path, mode="w" if first else "a", header=first, index=False)
        first = False

        ys.append(df["label"].to_numpy())
        ps.append(prob)

    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = (p >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "n_pos_pred": int(pred.sum()),
        "n_true_pos": int(y.sum()),
        "n_test": int(len(y)),
        "model": args.model,
        "source_tag": args.source_tag,
        "tag": args.out_tag,
        "prediction_source": "full_test_streamed",
        "model_path": str(model_path),
    }

    metrics_path.write_text(json.dumps({"metrics": metrics}, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


def load_xgboost_prediction(root: Path, tag: str) -> pd.DataFrame:
    path = root / "results" / "xgboost" / f"{tag}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    required = {"user_id", "product_id", "label", "prob", "pred"}
    df = pd.read_csv(path)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df.sort_values(["user_id", "product_id"]).reset_index(drop=True)


def summarize_prediction(tag: str, df: pd.DataFrame) -> dict:
    y = df["label"].to_numpy()
    prob = df["prob"].to_numpy()
    pred = df["pred"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    return {
        "tag": tag,
        "model": "xgboost",
        "n_test": int(len(df)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "per_order_f1": float(per_user_f1(df).mean()),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "miss_rate": float(fn / n_pos) if n_pos else np.nan,
        "false_alarm_rate": float(fp / n_neg) if n_neg else np.nan,
    }


def run_xgboost_scale_compare(args: argparse.Namespace) -> None:
    """
    Compare capped-trained XGBoost vs full-trained XGBoost on aligned full-test rows.

    Convention:
      base_tag   = capped/control XGBoost evaluated on full test
      scaled_tag = full-trained XGBoost evaluated on full test

    The reported bootstrap and DeLong differences are scaled - base.
    """
    root = args.root.resolve()
    out_dir = root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_xgboost_prediction(root, args.base_tag)
    scaled = load_xgboost_prediction(root, args.scaled_tag)
    base_keys = base[["user_id", "product_id", "label"]]
    scaled_keys = scaled[["user_id", "product_id", "label"]]
    if not base_keys.equals(scaled_keys):
        raise ValueError(
            "XGBoost scale comparison rows do not align. "
            f"Check results/xgboost/{args.base_tag}_predictions.csv and "
            f"results/xgboost/{args.scaled_tag}_predictions.csv."
        )

    comparison = pd.DataFrame([
        summarize_prediction(args.base_tag, base),
        summarize_prediction(args.scaled_tag, scaled),
    ])

    y = base["label"].to_numpy()
    base_correct = base["pred"].to_numpy() == y
    scaled_correct = scaled["pred"].to_numpy() == y
    scaled_right_base_wrong = int((scaled_correct & ~base_correct).sum())
    scaled_wrong_base_right = int((~scaled_correct & base_correct).sum())
    n_discordant = scaled_right_base_wrong + scaled_wrong_base_right
    if n_discordant:
        mcnemar_chi2 = (
            (abs(scaled_right_base_wrong - scaled_wrong_base_right) - 1) ** 2
            / n_discordant
        )
        from scipy import stats
        mcnemar_p = float(stats.chi2.sf(mcnemar_chi2, df=1))
    else:
        mcnemar_chi2 = np.nan
        mcnemar_p = np.nan
    mcnemar = pd.DataFrame([{
        "base_tag": args.base_tag,
        "scaled_tag": args.scaled_tag,
        "scaled_right_base_wrong": scaled_right_base_wrong,
        "scaled_wrong_base_right": scaled_wrong_base_right,
        "mcnemar_chi2": mcnemar_chi2,
        "p": mcnemar_p,
        "significant_0.05": bool(np.isfinite(mcnemar_p) and mcnemar_p < 0.05),
    }])

    base_user_f1 = per_user_f1(base)
    scaled_user_f1 = per_user_f1(scaled)
    boot = bootstrap_user_f1_diff(
        scaled_user_f1,
        base_user_f1,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    bootstrap = pd.DataFrame([{
        "base_tag": args.base_tag,
        "scaled_tag": args.scaled_tag,
        "metric": "mean_per_order_f1_diff_scaled_minus_base",
        **boot,
    }])

    delong_result = delong_roc_test(
        y,
        scaled["prob"].to_numpy(),
        base["prob"].to_numpy(),
    )
    delong = pd.DataFrame([{
        "base_tag": args.base_tag,
        "scaled_tag": args.scaled_tag,
        "metric": "roc_auc_diff_scaled_minus_base",
        **delong_result,
        "significant_0.05": bool(
            np.isfinite(delong_result["p"]) and delong_result["p"] < 0.05
        ),
    }])

    prefix = args.scale_out_prefix
    comparison_path = out_dir / f"{prefix}_model_comparison.csv"
    mcnemar_path = out_dir / f"{prefix}_mcnemar.csv"
    bootstrap_path = out_dir / f"{prefix}_bootstrap_f1_diff.csv"
    delong_path = out_dir / f"{prefix}_delong_roc_auc.csv"

    comparison.to_csv(comparison_path, index=False)
    mcnemar.to_csv(mcnemar_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    delong.to_csv(delong_path, index=False)

    print("\nXGBoost scale comparison:")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nScaled - base per-order F1 bootstrap:")
    print(bootstrap.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nScaled - base DeLong ROC-AUC:")
    print(delong.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSaved:")
    for path in [comparison_path, mcnemar_path, bootstrap_path, delong_path]:
        print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--model",
        default=None,
        choices=["logreg", "rf", "adaboost", "xgboost"],
    )
    parser.add_argument(
        "--source-tag",
        default="final",
        help="Tag of the run that trained the model and selected the threshold.",
    )
    parser.add_argument(
        "--out-tag",
        default="final_fulltest",
        help="Tag to use for the generated full-test prediction artifacts.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--compare-xgboost-scale",
        action="store_true",
        help=(
            "Skip prediction generation and compare two XGBoost full-test "
            "prediction tags: base/capped vs scaled/full-trained."
        ),
    )
    parser.add_argument(
        "--base-tag",
        default="final_fulltest",
        help="Capped/control XGBoost full-test prediction tag.",
    )
    parser.add_argument(
        "--scaled-tag",
        default="final_fulltrain_fulltest",
        help="Full-trained XGBoost full-test prediction tag.",
    )
    parser.add_argument(
        "--scale-out-prefix",
        default="xgboost_scale",
        help="Prefix for XGBoost scale-comparison tables under results/tables.",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.compare_xgboost_scale:
        run_xgboost_scale_compare(args)
        return
    if args.model is None:
        parser.error("--model is required unless --compare-xgboost-scale is set")
    run(args)


if __name__ == "__main__":
    main()
