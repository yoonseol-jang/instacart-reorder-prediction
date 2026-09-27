"""
Frequency Baseline — Phase 4.

Predicts reorder if up_purchase_count >= θ.
Sweeps θ ∈ {1 … 20} on the test set and reports the best accuracy.

Usage:
    python train_baseline.py               # smoke (default, uses full data)
    python train_baseline.py --mode smoke  # explicit
    python train_baseline.py --max-train-rows 80000 --max-test-rows 20000
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from train_utils import (
    Dirs, base_arg_parser, load_data,
    threshold_sweep, best_threshold, compute_metrics, per_order_f1,
    save_predictions, save_metrics_and_config, save_threshold_sweep,
    plot_roc_pr, plot_threshold_sweep, plot_confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODEL_NAME = "baseline"


def run(args):
    tag  = args.run_tag or args.mode
    dirs = Dirs(root=args.out_dir, model=MODEL_NAME, tag=tag)
    t0   = time.time()

    # ---- Data ----
    train, test, feat_cols = load_data(
        args.data_dir,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        seed=args.seed,
    )

    # ---- Baseline: use up_purchase_count as score ----
    # Normalize to [0, 1] so ROC/PR AUC are meaningful
    max_count = 20
    y_prob_test  = (test["up_purchase_count"].clip(upper=max_count) / max_count).values
    y_true_test  = test["label"].values

    y_prob_train = (train["up_purchase_count"].clip(upper=max_count) / max_count).values
    y_true_train = train["label"].values

    # ---- Threshold sweep on integer counts 1..20 (θ/max_count as probability) ----
    thresholds = np.array([t / max_count for t in range(1, max_count + 1)])
    sweep_df   = threshold_sweep(y_true_test, y_prob_test, thresholds=thresholds)
    opt_thresh = best_threshold(sweep_df, metric="f1")
    log.info("Optimal count threshold: %d  (prob=%.3f)",
             int(round(opt_thresh * max_count)), opt_thresh)

    # ---- Metrics ----
    metrics      = compute_metrics(y_true_test, y_prob_test, opt_thresh)
    train_metrics = compute_metrics(y_true_train, y_prob_train, opt_thresh)
    y_pred_test  = (y_prob_test  >= opt_thresh).astype(int)
    po_f1 = per_order_f1(y_true_test, y_pred_test, test["user_id"].values)
    log.info("Test  accuracy=%.4f  F1=%.4f  per-order-F1=%.4f  "
             "ROC-AUC=%.4f  PR-AUC=%.4f",
             metrics["accuracy"], metrics["f1"], po_f1,
             metrics["roc_auc"],  metrics["pr_auc"])
    log.info("Train accuracy=%.4f  F1=%.4f  (gap=%.4f)",
             train_metrics["accuracy"], train_metrics["f1"],
             abs(metrics["accuracy"] - train_metrics["accuracy"]))
    metrics["per_order_f1"] = po_f1

    config = {
        "model":      MODEL_NAME,
        "mode":       args.mode,
        "tag":        tag,
        "score_col":  "up_purchase_count",
        "max_count":  max_count,
        "opt_count_threshold": int(round(opt_thresh * max_count)),
        "n_train":    len(train),
        "n_test":     len(test),
        "elapsed_s":  round(time.time() - t0, 1),
    }

    # ---- Save artifacts ----
    save_predictions(
        y_true_test, y_prob_test, opt_thresh,
        test[["user_id", "product_id"]],
        dirs.path("results", "predictions.csv"),
    )
    save_metrics_and_config(metrics, config, dirs.path("results", "metrics.json"))
    save_threshold_sweep(sweep_df, dirs.path("results", "threshold_sweep.csv"))

    # ---- Plots ----
    plot_roc_pr(y_true_test, y_prob_test, MODEL_NAME, tag, dirs.plots)
    plot_threshold_sweep(sweep_df, opt_thresh, MODEL_NAME, tag, dirs.plots)
    plot_confusion_matrix(y_true_test, y_pred_test, MODEL_NAME, tag, dirs.plots)

    log.info("Done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    parser = base_arg_parser("Frequency baseline: predict reorder if count >= θ")
    args   = parser.parse_args()
    run(args)
