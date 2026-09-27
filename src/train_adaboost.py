"""
AdaBoost — Phase 4.

Uses sklearn AdaBoostClassifier (decision-tree stumps by default).
Treated like a tree model for preprocessing (OrdinalEncode cats, median impute).

Modes:
  smoke  — 100 estimators, quick fit
  tune   — RandomizedSearchCV (20 samples × 5 folds, GroupKFold)
  final  — refit best params

Usage:
    python train_adaboost.py --mode smoke --max-train-rows 80000 --max-test-rows 20000
    python train_adaboost.py --mode tune  --n-iter 20 --n-jobs 4
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
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

MODEL_NAME = "adaboost"

SMOKE_PARAMS = dict(
    n_estimators=100,
    learning_rate=1.0,
    random_state=42,
)

TUNE_DIST = {
    "clf__n_estimators":  [50, 100, 200, 300],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
    # max_depth of the base stump (1 = stump, 2 = depth-2 tree)
    "clf__estimator__max_depth": [1, 2, 3],
}


def build_pipeline(feat_cols: list, **model_kwargs) -> Pipeline:
    stump_depth = model_kwargs.pop("estimator__max_depth", 1)
    base = DecisionTreeClassifier(max_depth=stump_depth)
    model_kwargs.setdefault("random_state", 42)
    return Pipeline([
      ("pre", make_preprocessor("adaboost", feat_cols)),
      ("clf", AdaBoostClassifier(estimator=base, **model_kwargs)),
    ])


def run(args):
    tag  = args.run_tag or args.mode
    dirs = Dirs(root=args.out_dir, model=MODEL_NAME, tag=tag)
    t0   = time.time()

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

    # ---- Train ----
    if args.mode == "smoke":
        log.info("Smoke mode — single fit, n_estimators=%d",
                 SMOKE_PARAMS["n_estimators"])
        pipe = build_pipeline(feat_cols, **SMOKE_PARAMS)
        pipe.fit(X_fit, y_fit)
        best_params = SMOKE_PARAMS.copy()

    elif args.mode == "tune":
        n_combos = 4 * 5 * 3
        log.info("Tune mode — RandomizedSearchCV %d iter (of ~%d combos) × 5 folds",
                 args.n_iter, n_combos)
        groups = train["user_id"].values
        cv     = GroupKFold(n_splits=5)
        assert_group_cv_no_user_overlap(cv, X_train, y_train, groups)
        # Use a pipeline with a tunable base estimator depth
        pipe = Pipeline([
            ("pre", make_preprocessor("adaboost", feat_cols)),
            ("clf", AdaBoostClassifier(estimator=DecisionTreeClassifier(),
                                       random_state=args.seed,)),
        ])
        search = RandomizedSearchCV(
            pipe, TUNE_DIST, n_iter=args.n_iter,
            cv=cv, scoring="f1",
            n_jobs=args.n_jobs, random_state=args.seed,
            refit=True, verbose=1,
        )
        search.fit(X_train, y_train, groups=groups)
        pipe        = search.best_estimator_
        best_params = search.best_params_
        import pandas as pd
        cv_df = pd.DataFrame(search.cv_results_)
        cv_df.to_csv(dirs.path("results", "cv_results.csv"), index=False)
        save_tuning_history(cv_df, MODEL_NAME, dirs)
        log.info("Best params: %s  CV accuracy=%.4f",
                 best_params, search.best_score_)

    elif args.mode == "final":
        p = dirs.root / "results" / MODEL_NAME / "tune_metrics.json"
        best_params = (json.loads(p.read_text())["config"]["best_params"]
                       if p.exists() else SMOKE_PARAMS.copy())
        kw = {k.replace("clf__", ""): v for k, v in best_params.items()}
        pipe = build_pipeline(feat_cols, **kw)
        pipe.fit(X_train, y_train)

    # ---- Validation threshold + held-out test evaluation ----
    threshold_kw = {k.replace("clf__", ""): v for k, v in best_params.items()}
    threshold_pipe = build_pipeline(feat_cols, **threshold_kw)
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

    # ---- Artifacts ----
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

    plot_roc_pr(y_test, y_prob_test, MODEL_NAME, tag, dirs.plots)
    plot_threshold_sweep(sweep_df, opt_thresh, MODEL_NAME, tag, dirs.plots)
    plot_confusion_matrix(y_test, y_pred_test, MODEL_NAME, tag, dirs.plots)

    clf = pipe.named_steps["clf"]
    feat_names = preprocessed_feature_names("adaboost", feat_cols)
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        if len(imp) == len(feat_names):
            plot_feature_importance(imp, feat_names, MODEL_NAME, tag, dirs.plots)
            save_feature_importance_csv(imp, feat_names, MODEL_NAME, dirs)

    log.info("Done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    parser = base_arg_parser("AdaBoost: smoke / tune / final")
    args   = parser.parse_args()
    run(args)
