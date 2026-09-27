"""
Shared utilities for all model training scripts.

Covers: output-directory management, data loading, sklearn preprocessing,
threshold sweep, metric computation, leakage sanity check, and all standard
plots + artifact-saving functions used by train_*.py scripts.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

plt.rcParams["figure.dpi"] = 110
log = logging.getLogger(__name__)

# Features that are categorical (need encoding)
CAT_COLS = ["aisle_id", "department_id"]

# Precomputed global target-rate features that LaplaceTargetEncoder replaces for logreg
PRECOMPUTED_RATIO_COLS = ["aisle_reorder_ratio", "dept_reorder_ratio"]

DROPPED_FEATURES = {
    "user_std_days_between_orders",
    "user_unique_aisles",
    "user_unique_departments",
    "prod_total_users",
    "up_purchase_ratio_n5",
    "dow_match_user_pref",
    "hour_diff_from_user_pref",
}

EXPECTED_NUMERIC_FEATURE_RANGE = (33, 35)


def validate_feature_columns(feature_cols: list[str]) -> None:
    present_dropped = sorted(DROPPED_FEATURES.intersection(feature_cols))
    if present_dropped:
        raise ValueError(f"Dropped features are present: {present_dropped}")

    numeric_count = len([c for c in feature_cols if c not in CAT_COLS])
    lo, hi = EXPECTED_NUMERIC_FEATURE_RANGE
    if not lo <= numeric_count <= hi:
        raise ValueError(
            f"Expected {lo}-{hi} numerical features, found {numeric_count}."
        )


# ---------------------------------------------------------------------------
# LaplaceTargetEncoder  (plan §4.3 — per-fold target encoding with smoothing)
# ---------------------------------------------------------------------------

class LaplaceTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Replaces cat_cols with per-fold Laplace-smoothed mean(y) encodings.

    Fit on training fold X, y → stores category → smoothed mean map.
    Transform overwrites the original columns in-place (still named aisle_id,
    department_id) so the downstream ColumnTransformer sees consistent names.

    Also drops also_drop columns (the globally precomputed ratio features that
    this encoder supersedes for logistic regression).

    Smoothing:  ŷ_c = (n_c · ȳ_c + m · ȳ_global) / (n_c + m)
    where m = smooth (default 10, as in plan §4.3).
    """

    def __init__(self, cat_cols: list, smooth: float = 10.0,
                 also_drop: list | None = None):
        self.cat_cols  = cat_cols
        self.smooth    = smooth
        self.also_drop = also_drop or []

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("LaplaceTargetEncoder requires y during fit.")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(y, dtype=float)
        self.global_mean_ = float(y_arr.mean())
        self.maps_: dict[str, dict] = {}
        m = self.smooth
        for col in self.cat_cols:
            if col not in X.columns:
                continue
            tmp = pd.DataFrame({"cat": X[col].values, "y": y_arr})
            agg = tmp.groupby("cat")["y"].agg(["sum", "count"])
            smoothed = (agg["sum"] + m * self.global_mean_) / (agg["count"] + m)
            self.maps_[col] = smoothed.to_dict()
        return self

    def transform(self, X):
        X = (pd.DataFrame(X).copy()
             if not isinstance(X, pd.DataFrame) else X.copy())
        for col in self.cat_cols:
            if col in X.columns and col in self.maps_:
                X[col] = X[col].map(self.maps_[col]).fillna(self.global_mean_)
        for col in self.also_drop:
            if col in X.columns:
                X = X.drop(columns=[col])
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([])
        drop = set(self.also_drop)
        return np.array([f for f in input_features if f not in drop])


# ---------------------------------------------------------------------------
# Output directory management
# ---------------------------------------------------------------------------

@dataclass
class Dirs:
    root: Path
    model: str
    tag: str
    results: Path = field(init=False)
    plots:   Path = field(init=False)
    models:  Path = field(init=False)
    logs:    Path = field(init=False)

    def __post_init__(self):
        for attr, sub in [("results", "results"), ("plots", "plots"),
                          ("models", "models"), ("logs", "logs")]:
            p = self.root / sub / self.model
            p.mkdir(parents=True, exist_ok=True)
            object.__setattr__(self, attr, p)

    def path(self, subdir: str, suffix: str) -> Path:
        return getattr(self, subdir) / f"{self.tag}_{suffix}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: Path,
              max_train_rows: int | None = None,
              max_test_rows:  int | None = None,
              seed: int = 42):
    """Load processed parquet files; optionally subsample rows (stratified)."""
    log.info("Loading data from %s …", data_dir)
    train = pd.read_parquet(data_dir / "train.parquet")
    test  = pd.read_parquet(data_dir / "test.parquet")
    feat_cols = (data_dir / "feature_cols.txt").read_text().splitlines()
    validate_feature_columns(feat_cols)
    
    def _read_parquet_limited(path, max_rows):
        id_cols = ["user_id", "product_id", "label"]
        cols = list(dict.fromkeys(id_cols + feat_cols))
        if not max_rows:
            return pd.read_parquet(path, columns=cols)
        
        import pyarrow.parquet as pq
        batches = []
        rows = 0
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=100_000, columns=cols):
            table = batch.to_pandas()
            need = max_rows - rows
            batches.append(table.iloc[:need])
            rows += len(batches[-1])
            if rows >= max_rows:
                break
        
        return pd.concat(batches, axis=0, ignore_index=True)
    
    train = _read_parquet_limited(data_dir / "train.parquet", max_train_rows)
    test = _read_parquet_limited(data_dir / "test.parquet", max_test_rows)

    def _stratified_sample(df, max_rows):
        if not max_rows or len(df) <= max_rows:
            return df
        pieces = []
        for _, group in df.groupby("label", sort=False):
            n = min(len(group), max(1, int(max_rows * len(group) / len(df))))
            pieces.append(group.sample(n=n, random_state=seed))
        return pd.concat(pieces, axis=0).sample(frac=1.0, random_state=seed)
    
    if max_train_rows and len(train) > max_train_rows:
        train = _stratified_sample(train, max_train_rows)
        log.info("Train subsampled to %d rows", len(train))
    
    if max_test_rows and len(test) > max_test_rows:
        test = _stratified_sample(test, max_test_rows)
        log.info("Test  subsampled to %d rows", len(test))

    log.info("Train: %d rows  pos=%.4f  |  Test: %d rows  pos=%.4f",
             len(train), train["label"].mean(),
             len(test),  test["label"].mean())
    return train, test, feat_cols


# ---------------------------------------------------------------------------
# Split validation
# ---------------------------------------------------------------------------

def assert_group_cv_no_user_overlap(cv, X, y, groups: np.ndarray) -> None:
    """Assert every grouped CV fold has disjoint train/validation users."""
    groups = np.asarray(groups)
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), start=1):
        train_users = set(groups[tr_idx])
        valid_users = set(groups[va_idx])
        overlap = train_users.intersection(valid_users)
        if overlap:
            raise ValueError(
                f"Group CV fold {fold} has user overlap: {len(overlap)} users"
            )
    log.info("Group CV user-overlap check passed for %d folds", cv.get_n_splits())


def group_train_valid_split(
    df: pd.DataFrame,
    group_col: str = "user_id",
    valid_size: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/validation rows with no group overlap."""
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=valid_size, random_state=seed
    )
    groups = df[group_col].to_numpy()
    train_idx, valid_idx = next(splitter.split(df, df["label"].to_numpy(), groups))
    fit_df = df.iloc[train_idx].copy()
    valid_df = df.iloc[valid_idx].copy()
    overlap = set(fit_df[group_col].unique()).intersection(valid_df[group_col].unique())
    if overlap:
        raise ValueError(f"Train/validation group overlap: {len(overlap)} users")
    log.info(
        "Validation split: fit=%d rows (%d users), valid=%d rows (%d users)",
        len(fit_df), fit_df[group_col].nunique(),
        len(valid_df), valid_df[group_col].nunique(),
    )
    return fit_df, valid_df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def make_preprocessor(model_type: str, feature_cols: list):
    """
    Returns a fitted-ready sklearn preprocessor (ColumnTransformer or Pipeline).

    logreg  — LaplaceTargetEncoder on aisle_id/dept_id (per-fold, plan §4.3),
              drops globally precomputed ratio cols, then MinMaxScale + impute all.
    rf/adaboost — OrdinalEncode cats, median-impute numerics.
    xgboost — OrdinalEncode cats, pass-through numerics (NaN handled natively).
    """
    cat_present = [c for c in CAT_COLS if c in feature_cols]
    num_cols    = [c for c in feature_cols if c not in CAT_COLS]

    if model_type == "logreg":
        # After LaplaceTargetEncoder:
        #   • aisle_id / dept_id overwritten with smoothed mean(y) → continuous
        #   • aisle_reorder_ratio / dept_reorder_ratio dropped (superseded)
        # Downstream ColumnTransformer sees: (num_cols − dropped ratios) + cat_present
        also_drop = [c for c in PRECOMPUTED_RATIO_COLS if c in feature_cols]
        num_after = [c for c in num_cols if c not in PRECOMPUTED_RATIO_COLS]
        ct_cols   = num_after + cat_present   # cat_present now float after LTE

        ct = make_column_transformer(
            (Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale",  MinMaxScaler()),
            ]), ct_cols),
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return Pipeline([
            ("lte", LaplaceTargetEncoder(
                cat_cols=cat_present,
                smooth=10.0,
                also_drop=also_drop,
            )),
            ("ct", ct),
        ])

    elif model_type in ("rf", "adaboost"):
        return make_column_transformer(
            (SimpleImputer(strategy="median"), num_cols),
            (OrdinalEncoder(handle_unknown="use_encoded_value",
                            unknown_value=-1), cat_present),
            remainder="drop",
            verbose_feature_names_out=False,
        )

    elif model_type == "xgboost":
        # Passthrough numerics (XGBoost handles NaN natively); encode cats.
        return make_column_transformer(
            ("passthrough", num_cols),
            (OrdinalEncoder(handle_unknown="use_encoded_value",
                            unknown_value=-1), cat_present),
            remainder="drop",
            verbose_feature_names_out=False,
        )

    raise ValueError(f"Unknown model_type '{model_type}'")


def preprocessed_feature_names(model_type: str, feature_cols: list) -> list:
    """Return feature names after preprocessing (same order as transformer output)."""
    cat_present = [c for c in CAT_COLS if c in feature_cols]
    num_cols    = [c for c in feature_cols if c not in CAT_COLS]
    if model_type == "logreg":
        # LTE drops PRECOMPUTED_RATIO_COLS and overwrites cat_present in-place.
        # ColumnTransformer then processes: (num_cols − precomputed) + cat_present.
        num_after = [c for c in num_cols if c not in PRECOMPUTED_RATIO_COLS]
        return num_after + cat_present
    return num_cols + cat_present


# ---------------------------------------------------------------------------
# Threshold sweep + metrics
# ---------------------------------------------------------------------------

def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray,
                    thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Return a DataFrame of per-threshold accuracy, F1, precision, recall."""
    if thresholds is None:
        thresholds = np.arange(0.01, 0.51, 0.01)
    rows = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 4),
            "accuracy":  accuracy_score(y_true, pred),
            "f1":        f1_score(y_true, pred, zero_division=0),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall":    recall_score(y_true, pred, zero_division=0),
        })
    return pd.DataFrame(rows)


def best_threshold(sweep_df: pd.DataFrame, metric: str = "f1") -> float:
    return float(sweep_df.loc[sweep_df[metric].idxmax(), "threshold"])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold":  threshold,
        "accuracy":   accuracy_score(y_true, y_pred),
        "f1":         f1_score(y_true, y_pred, zero_division=0),
        "precision":  precision_score(y_true, y_pred, zero_division=0),
        "recall":     recall_score(y_true, y_pred, zero_division=0),
        "roc_auc":    roc_auc_score(y_true, y_prob),
        "pr_auc":     average_precision_score(y_true, y_prob),
        "n_pos_pred": int((y_pred == 1).sum()),
        "n_true_pos": int(y_true.sum()),
    }


def per_order_f1(y_true: np.ndarray, y_pred: np.ndarray,
                 user_ids: np.ndarray) -> float:
    """
    Per-order (basket-level) F1: compute F1 for each user's predictions then
    average across users.  This is the headline metric used by Onodera 2017
    and the Kaggle competition.  (plan §6.1)
    """
    df = pd.DataFrame({
        "user": user_ids, "yt": y_true.astype(int), "yp": y_pred.astype(int)
    })
    scores = []
    for _, grp in df.groupby("user"):
        scores.append(f1_score(grp["yt"], grp["yp"], zero_division=0))
    return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Leakage sanity check (Phase 4.4)
# ---------------------------------------------------------------------------

def leakage_check(X_train, y_train, feat_names: list,
                  corr_threshold: float = 0.95) -> dict:
    """
    Check for obvious leakage signals.
    Returns a dict with any suspicious features flagged.
    """
    flags = {}
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns=feat_names)
    for col in X_train.columns:
        try:
            corr = float(X_train[col].corr(pd.Series(y_train.astype(float),
                                                       index=X_train.index)))
            if abs(corr) > corr_threshold:
                flags[col] = corr
        except Exception:
            pass
    if flags:
        log.warning("LEAKAGE CHECK — high-correlation features: %s", flags)
    else:
        log.info("Leakage check passed (no feature with |r| > %.2f)", corr_threshold)
    return flags


# ---------------------------------------------------------------------------
# Saving artifacts
# ---------------------------------------------------------------------------

def save_predictions(y_true: np.ndarray, y_prob: np.ndarray,
                     threshold: float, ids_df: pd.DataFrame,
                     path: Path):
    out = ids_df[["user_id", "product_id"]].copy()
    out["label"] = y_true.astype(int)
    out["prob"]  = y_prob.round(6)
    out["pred"]  = (y_prob >= threshold).astype(int)
    out.to_csv(path, index=False)
    log.info("Predictions → %s", path)


def save_metrics_and_config(metrics: dict, config: dict, path: Path):
    out = {"metrics": metrics, "config": config}
    path.write_text(json.dumps(out, indent=2))
    log.info("Metrics/config → %s", path)


def save_model(model, path: Path):
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=5)
    log.info("Model → %s", path)


def save_best_model(model, model_name: str, dirs: Dirs):
    """Save stable Phase 4 model alias required by the HPC artifact checklist."""
    path = dirs.models / f"{model_name}_best.pkl"
    joblib.dump(model, path)
    log.info("Best model alias → %s", path)


def save_threshold_sweep(sweep_df: pd.DataFrame, path: Path):
    sweep_df.to_csv(path, index=False)


def update_optimal_thresholds(root: Path, model_name: str, threshold: float):
    """Update root-level optimal threshold dictionary for Phase 4/5 handoff."""
    path = root / "results" / "optimal_thresholds.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        data = json.loads(path.read_text())
    data[model_name] = round(float(threshold), 6)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)
    log.info("Optimal thresholds → %s", path)


def save_tuning_history(df: pd.DataFrame, model_name: str, dirs: Dirs):
    path = dirs.results / f"tuning_history_{model_name}.csv"
    df.to_csv(path, index=False)
    log.info("Tuning history → %s", path)


def save_feature_importance_csv(
    importances: np.ndarray,
    feat_names: list,
    model_name: str,
    dirs: Dirs,
):
    if len(importances) != len(feat_names):
        log.warning(
            "Skipping feature importance CSV: %d scores for %d feature names",
            len(importances), len(feat_names),
        )
        return
    out = (
        pd.DataFrame({
            "feature_name": feat_names,
            "importance_score": np.asarray(importances, dtype=float),
        })
        .sort_values("importance_score", ascending=False)
    )
    path = dirs.results / f"feature_importance_{model_name}.csv"
    out.to_csv(path, index=False)
    log.info("Feature importance CSV → %s", path)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_pr(y_true, y_prob, model_name: str, tag: str, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, lw=1.8, label=f"AUC = {auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title=f"{model_name} — ROC Curve ({tag})",
                xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[0].legend()

    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    base = y_true.mean()
    axes[1].plot(rec, prec, lw=1.8, label=f"AP = {pr_auc:.4f}")
    axes[1].axhline(base, color="gray", linestyle="--", lw=0.8,
                    label=f"Baseline = {base:.4f}")
    axes[1].set(title=f"{model_name} — PR Curve ({tag})",
                xlabel="Recall", ylabel="Precision", ylim=(0, 1))
    axes[1].legend()

    plt.tight_layout()
    path = out_dir / f"{tag}_roc_pr.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info("Plot → %s", path)


def plot_threshold_sweep(sweep_df: pd.DataFrame, opt_thresh: float,
                         model_name: str, tag: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sweep_df["threshold"], sweep_df["accuracy"], label="Accuracy")
    ax.plot(sweep_df["threshold"], sweep_df["f1"],       label="F1")
    ax.plot(sweep_df["threshold"], sweep_df["precision"],label="Precision", alpha=0.7)
    ax.plot(sweep_df["threshold"], sweep_df["recall"],   label="Recall",    alpha=0.7)
    ax.axvline(opt_thresh, color="red", linestyle="--",
               label=f"Opt threshold = {opt_thresh:.3f}")
    ax.set(title=f"{model_name} — Threshold Sweep ({tag})",
           xlabel="Decision threshold", ylabel="Score", ylim=(0, 1))
    ax.legend(ncol=2)
    plt.tight_layout()
    path = out_dir / f"{tag}_threshold_sweep.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info("Plot → %s", path)


def plot_confusion_matrix(y_true, y_pred, model_name: str, tag: str, out_dir: Path):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Pred 0", "Pred 1"],
           yticklabels=["True 0", "True 1"],
           title=f"{model_name} — Confusion Matrix ({tag})")
    plt.tight_layout()
    path = out_dir / f"{tag}_confusion_matrix.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info("Plot → %s", path)


def plot_feature_importance(importances: np.ndarray, feat_names: list,
                             model_name: str, tag: str, out_dir: Path,
                             top_n: int = 20):
    idx   = np.argsort(importances)[-top_n:]
    names = [feat_names[i] for i in idx]
    vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(7, max(4, top_n * 0.3)))
    ax.barh(names, vals)
    ax.set(title=f"{model_name} — Feature Importance ({tag})",
           xlabel="Importance")
    plt.tight_layout()
    path = out_dir / f"{tag}_feature_importance.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info("Plot → %s", path)


# ---------------------------------------------------------------------------
# Shared argument parser base
# ---------------------------------------------------------------------------

def base_arg_parser(description: str):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mode", choices=["smoke", "tune", "final"], default="smoke",
                   help="smoke=sanity-check, tune=full search, final=refit best params")
    p.add_argument("--data-dir", type=Path,
                   default=Path(__file__).parent.parent / "data" / "processed",
                   help="Directory with train.parquet / test.parquet")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).parent.parent,
                   help="Project root for results/, plots/, models/ subdirs")
    p.add_argument("--max-train-rows", type=int, default=None,
                   help="Cap training rows (stratified sample). Default: all.")
    p.add_argument("--max-test-rows", type=int, default=None,
                   help="Cap test rows (stratified sample). Default: all.")
    p.add_argument("--n-iter", type=int, default=30,
                   help="RandomizedSearchCV iterations (tune mode only)")
    p.add_argument("--n-jobs", type=int, default=4,
                   help="Parallel jobs for search / fitting")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-tag", type=str, default=None,
                   help="Override output file prefix (defaults to mode name)")
    return p
