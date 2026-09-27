"""
Last-Basket Baseline — Phase 4.

Predicts reorder for a (user, product) pair iff the product appeared in the
user's most recent prior order (up_orders_since_last == 0).

Score for ROC/PR curves: 1 / (1 + up_orders_since_last), so the most-recently-
bought products rank highest.  Products bought ≥1 order ago score < 1.

Usage:
    python train_last_basket.py               # uses full processed data
    python train_last_basket.py --max-train-rows 80000 --max-test-rows 20000
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_utils import (
    Dirs, base_arg_parser, load_data, group_train_valid_split,
    threshold_sweep, best_threshold, compute_metrics, per_order_f1,
    save_predictions, save_metrics_and_config, save_threshold_sweep,
    plot_roc_pr, plot_threshold_sweep, plot_confusion_matrix,
    update_optimal_thresholds,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODEL_NAME = "last_basket"


def run(args):
    tag  = args.run_tag or args.mode
    dirs = Dirs(root=args.out_dir, model=MODEL_NAME, tag=tag)
    t0   = time.time()

    train, test, feat_cols = load_data(
        args.data_dir,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        seed=args.seed,
    )

    if "up_orders_since_last" not in test.columns:
        raise ValueError(
            "Feature 'up_orders_since_last' not found in processed data. "
            "Re-run features.py to regenerate the parquet files."
        )
    _, valid_df = group_train_valid_split(train, seed=args.seed)

    # Score = 1 / (1 + orders_since_last).  Last-basket products (0 orders ago)
    # score 1.0; products bought 1 order ago score 0.5; etc.
    y_prob_test  = 1.0 / (1.0 + test["up_orders_since_last"].values.astype(float))
    y_true_test  = test["label"].values
    y_prob_train = 1.0 / (1.0 + train["up_orders_since_last"].values.astype(float))
    y_true_train = train["label"].values
    y_prob_valid = 1.0 / (
        1.0 + valid_df["up_orders_since_last"].values.astype(float)
    )
    y_true_valid = valid_df["label"].values

    # Sweep thresholds matching the validation score values (1, 0.5, 0.333, ...).
    thresholds = np.unique(y_prob_valid)[::-1]
    thresholds = thresholds[thresholds > 0]
    valid_sweep_df = threshold_sweep(
        y_true_valid, y_prob_valid, thresholds=np.clip(thresholds, 0.01, 1.0)
    )
    opt_thresh = best_threshold(valid_sweep_df, "f1")
    log.info("Optimal threshold: %.4f  (last-basket score = 1.0)", opt_thresh)

    sweep_df = threshold_sweep(
        y_true_test, y_prob_test, thresholds=np.clip(thresholds, 0.01, 1.0)
    )
    metrics       = compute_metrics(y_true_test,  y_prob_test,  opt_thresh)
    train_metrics = compute_metrics(y_true_train, y_prob_train, opt_thresh)
    valid_metrics = compute_metrics(y_true_valid, y_prob_valid, opt_thresh)
    y_pred_test   = (y_prob_test >= opt_thresh).astype(int)
    po_f1 = per_order_f1(y_true_test, y_pred_test, test["user_id"].values)
    metrics["per_order_f1"] = po_f1
    log.info("Test  acc=%.4f  F1=%.4f  per-order-F1=%.4f  "
             "ROC-AUC=%.4f  PR-AUC=%.4f",
             metrics["accuracy"], metrics["f1"], po_f1,
             metrics["roc_auc"],  metrics["pr_auc"])
    log.info("Valid acc=%.4f  F1=%.4f  threshold=%.4f",
             valid_metrics["accuracy"], valid_metrics["f1"], opt_thresh)
    log.info("Train acc=%.4f  F1=%.4f  (gap=%.4f)",
             train_metrics["accuracy"], train_metrics["f1"],
             abs(metrics["accuracy"] - train_metrics["accuracy"]))

    config = {
        "model":     MODEL_NAME,
        "mode":      args.mode,
        "tag":       tag,
        "rule":      "predict reorder iff up_orders_since_last == 0",
        "opt_thresh": opt_thresh,
        "threshold_source": "grouped_validation_split",
        "validation_metrics": valid_metrics,
        "n_train":   len(train),
        "n_test":    len(test),
        "elapsed_s": round(time.time() - t0, 1),
    }

    save_predictions(
        y_true_test, y_prob_test, opt_thresh,
        test[["user_id", "product_id"]],
        dirs.path("results", "predictions.csv"),
    )
    save_metrics_and_config(metrics, config, dirs.path("results", "metrics.json"))
    save_threshold_sweep(sweep_df, dirs.path("results", "threshold_sweep.csv"))
    save_threshold_sweep(
        valid_sweep_df, dirs.path("results", "validation_threshold_sweep.csv")
    )
    update_optimal_thresholds(dirs.root, MODEL_NAME, opt_thresh)

    plot_roc_pr(y_true_test, y_prob_test, MODEL_NAME, tag, dirs.plots)
    plot_threshold_sweep(sweep_df, opt_thresh, MODEL_NAME, tag, dirs.plots)
    plot_confusion_matrix(y_true_test, y_pred_test, MODEL_NAME, tag, dirs.plots)

    log.info("Done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    parser = base_arg_parser("Last-basket baseline: predict reorder iff in most recent prior order")
    args   = parser.parse_args()
    run(args)
