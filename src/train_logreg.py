"""
Logistic Regression — Phase 4.

Modes:
  smoke  — one fit with reasonable defaults on small data (~2 min local)
  tune   — GridSearchCV (24 combos × 5 folds) on 20% user subsample
  final  — refit best params on full training data

Usage:
    python train_logreg.py --mode smoke --max-train-rows 80000 --max-test-rows 20000
    python train_logreg.py --mode tune  --n-jobs 4
    python train_logreg.py --mode final --run-tag final
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from train_utils import (
    Dirs, base_arg_parser, load_data, make_preprocessor,
    group_train_valid_split,
    preprocessed_feature_names, leakage_check,
    threshold_sweep, best_threshold, compute_metrics, per_order_f1,
    save_predictions, save_metrics_and_config, save_model, save_best_model,
    save_threshold_sweep, plot_roc_pr, plot_threshold_sweep,
    plot_confusion_matrix, plot_feature_importance,
    assert_group_cv_no_user_overlap, update_optimal_thresholds,
    save_tuning_history, save_feature_importance_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODEL_NAME = "logreg"

SMOKE_PARAMS = dict(
    penalty="l2", C=1.0, solver="saga", max_iter=1000,
    class_weight=None, random_state=42, n_jobs=-1,
)

TUNE_GRID = {
    "clf__penalty":      ["l1", "l2"],
    "clf__C":            [0.001, 0.01, 0.1, 1, 10, 100],
    "clf__class_weight": [None, "balanced"],
}


def build_pipeline(feat_cols: list, **model_kwargs) -> Pipeline:
    pre = make_preprocessor("logreg", feat_cols)
    clf = LogisticRegression(solver="saga", max_iter=1000,
                             random_state=42, n_jobs=-1, **model_kwargs)
    return Pipeline([("pre", pre), ("clf", clf)])


def run(args):
    tag  = args.run_tag or args.mode
    dirs = Dirs(root=args.out_dir, model=MODEL_NAME, tag=tag)
    t0   = time.time()

    # ---- Data ----
    max_tr = args.max_train_rows if args.max_train_rows else (
        80_000 if args.mode == "smoke" else None
    )
    max_te = args.max_test_rows if args.max_test_rows else (
        20_000 if args.mode == "smoke" else None
    )
    train, test, feat_cols = load_data(args.data_dir, max_tr, max_te, args.seed)

    X_train = train[feat_cols]
    y_train = train["label"].values
    X_test  = test[feat_cols]
    y_test  = test["label"].values
    fit_df, valid_df = group_train_valid_split(train, seed=args.seed)
    X_fit = fit_df[feat_cols]
    y_fit = fit_df["label"].values
    X_valid = valid_df[feat_cols]
    y_valid = valid_df["label"].values

    # ---- Leakage check (fit LTE with y, then check each feature's correlation) ----
    pre_check  = make_preprocessor("logreg", feat_cols)
    Xc         = pre_check.fit_transform(X_train, y_train)
    feat_names = preprocessed_feature_names("logreg", feat_cols)
    leakage_check(Xc, y_train, feat_names)

    # ---- Train ----
    if args.mode == "smoke":
        log.info("Smoke mode — single fit with default params")
        pipe = Pipeline([
            ("pre", make_preprocessor("logreg", feat_cols)),
            ("clf", LogisticRegression(**SMOKE_PARAMS)),
        ])
        pipe.fit(X_fit, y_fit)
        best_params = SMOKE_PARAMS.copy()

    elif args.mode == "tune":
        log.info("Tune mode — GridSearchCV over %d combos × 5 folds",
                 2 * 6 * 2)
        groups = train["user_id"].values
        cv     = GroupKFold(n_splits=5)
        assert_group_cv_no_user_overlap(cv, X_train, y_train, groups)
        pipe   = Pipeline([
            ("pre", make_preprocessor("logreg", feat_cols)),
            ("clf", LogisticRegression(solver="saga", max_iter=1000,
                                       random_state=args.seed)),
        ])
        search = GridSearchCV(
            pipe, TUNE_GRID, cv=cv,
            scoring="f1", n_jobs=args.n_jobs,
            refit=True, verbose=1,
        )
        search.fit(X_train, y_train, groups=groups)
        pipe        = search.best_estimator_
        best_params = search.best_params_
        cv_results  = search.cv_results_
        # Save CV results
        import pandas as pd
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv(dirs.path("results", "cv_results.csv"), index=False)
        save_tuning_history(cv_df, MODEL_NAME, dirs)
        log.info("Best params: %s  CV accuracy=%.4f",
                 best_params, search.best_score_)

    elif args.mode == "final":
        best_params_path = dirs.root / "results" / MODEL_NAME / "tune_metrics.json"
        if best_params_path.exists():
            best_params = json.loads(best_params_path.read_text())["config"]["best_params"]
            log.info("Loaded best params: %s", best_params)
        else:
            log.warning("No tune_metrics.json found — using smoke defaults")
            best_params = SMOKE_PARAMS.copy()
        kw = {k.replace("clf__", ""): v for k, v in best_params.items()}
        clf_kwargs = {
            "solver": "saga",
            "max_iter": 1000,
            "random_state": args.seed,
            "n_jobs": -1,
        }
        clf_kwargs.update(kw)
        pipe = Pipeline([
            ("pre", make_preprocessor("logreg", feat_cols)),
            ("clf", LogisticRegression(**clf_kwargs)),
        ])
        pipe.fit(X_train, y_train)

    # ---- Validation threshold + held-out test evaluation ----
    threshold_kw = {k.replace("clf__", ""): v for k, v in best_params.items()}
    threshold_kw.update({
        "solver": threshold_kw.get("solver", "saga"),
        "max_iter": threshold_kw.get("max_iter", 1000),
        "random_state": args.seed,
        "n_jobs": -1,
    })
    threshold_pipe = Pipeline([
        ("pre", make_preprocessor("logreg", feat_cols)),
        ("clf", LogisticRegression(**threshold_kw)),
    ])
    threshold_pipe.fit(X_fit, y_fit)
    y_prob_valid = threshold_pipe.predict_proba(X_valid)[:, 1]
    valid_sweep_df = threshold_sweep(y_valid, y_prob_valid)
    opt_thresh = best_threshold(valid_sweep_df, "f1")
    valid_metrics = compute_metrics(y_valid, y_prob_valid, opt_thresh)

    y_prob_test  = pipe.predict_proba(X_test)[:, 1]
    y_prob_train = pipe.predict_proba(X_train)[:, 1]

    sweep_df = threshold_sweep(y_test, y_prob_test)
    metrics       = compute_metrics(y_test,  y_prob_test,  opt_thresh)
    train_metrics = compute_metrics(y_train, y_prob_train, opt_thresh)
    y_pred_test   = (y_prob_test >= opt_thresh).astype(int)
    po_f1 = per_order_f1(y_test, y_pred_test, test["user_id"].values)
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
        "model": MODEL_NAME, "mode": args.mode, "tag": tag,
        "best_params": best_params,
        "threshold_source": "grouped_validation_split",
        "validation_metrics": valid_metrics,
        "n_train": len(train), "n_test": len(test),
        "elapsed_s": round(time.time() - t0, 1),
    }

    # ---- Save artifacts ----
    save_predictions(y_test, y_prob_test, opt_thresh,
                     test[["user_id", "product_id"]],
                     dirs.path("results", "predictions.csv"))
    save_metrics_and_config(metrics, config, dirs.path("results", "metrics.json"))
    save_threshold_sweep(sweep_df, dirs.path("results", "threshold_sweep.csv"))
    save_threshold_sweep(
        valid_sweep_df, dirs.path("results", "validation_threshold_sweep.csv")
    )
    save_model(pipe, dirs.path("models", "model.pkl"))
    save_best_model(pipe, MODEL_NAME, dirs)
    update_optimal_thresholds(dirs.root, MODEL_NAME, opt_thresh)

    # ---- Plots ----
    plot_roc_pr(y_test, y_prob_test, MODEL_NAME, tag, dirs.plots)
    plot_threshold_sweep(sweep_df, opt_thresh, MODEL_NAME, tag, dirs.plots)
    plot_confusion_matrix(y_test, y_pred_test, MODEL_NAME, tag, dirs.plots)

    # Feature importance: standardized LR coefficients
    clf        = pipe.named_steps["clf"]
    feat_names = preprocessed_feature_names("logreg", feat_cols)
    coefs      = np.abs(clf.coef_[0])
    if len(coefs) == len(feat_names):
        plot_feature_importance(coefs, feat_names, MODEL_NAME, tag, dirs.plots)
        save_feature_importance_csv(coefs, feat_names, MODEL_NAME, dirs)

    log.info("Done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    parser = base_arg_parser("Logistic Regression: smoke / tune / final")
    args   = parser.parse_args()
    run(args)
