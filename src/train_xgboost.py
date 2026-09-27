"""
XGBoost — Phase 4.

Modes:
  smoke  — 200 trees, quick fit to verify pipeline
  tune   — RandomizedSearchCV 30 samples × 5 folds with early stopping
  final  — refit best params on full training data

Usage:
    python train_xgboost.py --mode smoke --max-train-rows 80000 --max-test-rows 20000
    python train_xgboost.py --mode tune  --n-iter 30 --n-jobs 8
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit("xgboost is not installed. Run: conda install -c conda-forge xgboost")

from sklearn.model_selection import GroupKFold
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

MODEL_NAME = "xgboost"

SMOKE_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, gamma=0,
    min_child_weight=5,
    eval_metric="logloss",
    random_state=42, n_jobs=-1,
)

TUNE_DIST = {
    "clf__learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "clf__max_depth":        [4, 6, 8, 10],
    "clf__min_child_weight": [1, 5, 10],
    "clf__subsample":        [0.7, 0.8, 0.9, 1.0],
    "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "clf__gamma":            [0, 0.1, 0.5],
}


def build_pipeline(feat_cols: list, **model_kwargs) -> Pipeline:
    model_kwargs.pop("use_label_encoder", None)
    clf_kwargs = {
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    clf_kwargs.update(model_kwargs)
    return Pipeline([
        ("pre", make_preprocessor("xgboost", feat_cols)),
        ("clf", XGBClassifier(**clf_kwargs)),
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

    # ---- Scale positives by class ratio for imbalance (scale_pos_weight) ----
    neg_pos_ratio = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    log.info("scale_pos_weight = %.2f", neg_pos_ratio)

    # ---- Train ----
    if args.mode == "smoke":
        log.info("Smoke mode — single fit, n_estimators=%d",
                 SMOKE_PARAMS["n_estimators"])
        kw = {**SMOKE_PARAMS, "scale_pos_weight": neg_pos_ratio}
        # Remove keys not accepted by XGBClassifier constructor
        kw.pop("use_label_encoder", None)
        pipe = build_pipeline(feat_cols, scale_pos_weight=neg_pos_ratio,
                              n_estimators=SMOKE_PARAMS["n_estimators"],
                              max_depth=SMOKE_PARAMS["max_depth"],
                              learning_rate=SMOKE_PARAMS["learning_rate"],
                              subsample=SMOKE_PARAMS["subsample"],
                              colsample_bytree=SMOKE_PARAMS["colsample_bytree"],
                              gamma=SMOKE_PARAMS["gamma"],
                              min_child_weight=SMOKE_PARAMS["min_child_weight"])
        pipe.fit(X_fit, y_fit)
        best_params = {**SMOKE_PARAMS, "scale_pos_weight": neg_pos_ratio}

    elif args.mode == "tune":
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise SystemExit("optuna not installed — run: pip install optuna")

        import pandas as pd
        log.info(
            "Tune mode — Optuna TPE %d trials × 5 GroupKFold folds "
            "(per-fold early stopping, no eval_set leakage). §5.2.6",
            args.n_iter,
        )
        groups = train["user_id"].values
        gkf    = GroupKFold(n_splits=5)
        assert_group_cv_no_user_overlap(gkf, X_train, y_train, groups)

        def objective(trial):
            params = {
                "learning_rate":    trial.suggest_float(
                    "learning_rate", 0.01, 0.2, log=True),
                "max_depth":        trial.suggest_int("max_depth", 4, 10),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample":        trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "gamma":            trial.suggest_float("gamma", 0.0, 0.5),
            }
            fold_f1s = []
            fold_best_rounds = []
            for tr_idx, va_idx in gkf.split(X_train, y_train, groups):
                Xf_tr_raw = X_train.iloc[tr_idx]
                yf_tr     = y_train[tr_idx]
                Xf_va_raw = X_train.iloc[va_idx]
                yf_va     = y_train[va_idx]

                # Per-fold preprocessor (correct leakage handling)
                pre_f  = make_preprocessor("xgboost", feat_cols)
                Xf_tr  = pre_f.fit_transform(Xf_tr_raw)
                Xf_va  = pre_f.transform(Xf_va_raw)

                # Inner split (10% of fold-train users) for early stopping
                from sklearn.model_selection import train_test_split as tts
                Xi_tr, Xi_va, yi_tr, yi_va = tts(
                    Xf_tr, yf_tr,
                    test_size=0.10,
                    random_state=args.seed,
                    stratify=yf_tr,
                )

                clf = XGBClassifier(
                    n_estimators=2000,
                    eval_metric="logloss",
                    early_stopping_rounds=50,
                    scale_pos_weight=neg_pos_ratio,
                    random_state=args.seed,
                    n_jobs=args.n_jobs,
                    **params,
                )
                clf.fit(
                    Xi_tr, yi_tr,
                    eval_set=[(Xi_va, yi_va)],
                    verbose=False,
                )
                if getattr(clf, "best_iteration", None) is not None:
                    fold_best_rounds.append(int(clf.best_iteration) + 1)

                y_prob_va  = clf.predict_proba(Xf_va)[:, 1]
                # Threshold chosen on inner val to avoid using outer val labels
                y_prob_iva = clf.predict_proba(Xi_va)[:, 1]
                sw         = threshold_sweep(yi_va, y_prob_iva)
                t          = best_threshold(sw, "f1")
                from sklearn.metrics import f1_score as _f1
                fold_f1s.append(
                    _f1(yf_va, (y_prob_va >= t).astype(int), zero_division=0))

            if fold_best_rounds:
                trial.set_user_attr(
                    "mean_best_iteration",
                    int(np.ceil(np.mean(fold_best_rounds))),
                )
            return float(np.mean(fold_f1s))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        study.optimize(objective, n_trials=args.n_iter, show_progress_bar=True)

        best_n_estimators = study.best_trial.user_attrs.get(
            "mean_best_iteration", 2000
        )
        best_params = {
            **study.best_params,
            "n_estimators": int(best_n_estimators),
            "scale_pos_weight": neg_pos_ratio,
        }
        log.info("Best params: %s  CV F1=%.4f", best_params, study.best_value)

        trials_df = pd.DataFrame([
            t.params | t.user_attrs | {"value": t.value}
            for t in study.trials
        ])
        trials_df.to_csv(dirs.path("results", "optuna_trials.csv"), index=False)
        save_tuning_history(trials_df, MODEL_NAME, dirs)

        # Refit on full training data with best params
        kw = {k: v for k, v in best_params.items() if k != "scale_pos_weight"}
        pipe = build_pipeline(feat_cols, scale_pos_weight=neg_pos_ratio, **kw)
        pipe.fit(X_train, y_train)

    elif args.mode == "final":
        p = dirs.root / "results" / MODEL_NAME / "tune_metrics.json"
        best_params = (json.loads(p.read_text())["config"]["best_params"]
                       if p.exists() else
                       {**SMOKE_PARAMS, "scale_pos_weight": neg_pos_ratio})
        kw = {k.replace("clf__", ""): v for k, v in best_params.items()}
        kw.pop("use_label_encoder", None)
        pipe = build_pipeline(feat_cols, **kw)
        pipe.fit(X_train, y_train)

    # ---- Validation threshold + held-out test evaluation ----
    threshold_kw = {k.replace("clf__", ""): v for k, v in best_params.items()}
    threshold_kw.pop("use_label_encoder", None)
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
    feat_names = preprocessed_feature_names("xgboost", feat_cols)
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        if len(imp) == len(feat_names):
            plot_feature_importance(imp, feat_names, MODEL_NAME, tag, dirs.plots)
            save_feature_importance_csv(imp, feat_names, MODEL_NAME, dirs)

    log.info("Done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    parser = base_arg_parser("XGBoost: smoke / tune / final")
    args   = parser.parse_args()
    run(args)
