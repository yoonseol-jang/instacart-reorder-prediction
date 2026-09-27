"""
Phase 2.4 — Correlation matrix + VIF analysis on the training feature set.

Flags pairs with |r| > 0.9 (plan §3.4) and checks VIF > 5 for those features.
Saves a heatmap and a CSV report to results/.

Usage:
    python correlation_check.py
    python correlation_check.py --threshold 0.85  # looser flag
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_DIR    = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR   = Path(__file__).parent.parent / "plots"

# Columns that are categorical IDs — excluded from numeric correlation check
CAT_COLS = ["aisle_id", "department_id"]


def compute_vif(df: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor for every column in df (must be all-numeric)."""
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        log.warning("statsmodels not installed — VIF skipped. pip install statsmodels")
        return pd.Series(dtype=float)

    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(df.fillna(df.median()))
    vifs = {col: variance_inflation_factor(X, i)
            for i, col in enumerate(df.columns)}
    return pd.Series(vifs).sort_values(ascending=False)


def run(data_dir: Path, corr_threshold: float, vif_threshold: float):
    log.info("Loading train.parquet …")
    train     = pd.read_parquet(data_dir / "train.parquet")
    feat_cols = (data_dir / "feature_cols.txt").read_text().splitlines()

    num_cols = [c for c in feat_cols if c not in CAT_COLS]
    X = train[num_cols].copy()
    log.info("%d numeric features × %d rows", len(num_cols), len(X))

    # ---- Pearson correlation matrix ----
    log.info("Computing correlation matrix …")
    corr = X.corr(method="pearson")

    # ---- Flag pairs above threshold ----
    flagged = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            r = corr.iloc[i, j]
            if abs(r) > corr_threshold:
                flagged.append({
                    "feature_a": num_cols[i],
                    "feature_b": num_cols[j],
                    "pearson_r": round(float(r), 4),
                    "abs_r":     round(abs(float(r)), 4),
                })
    flagged.sort(key=lambda d: d["abs_r"], reverse=True)

    log.info("Pairs with |r| > %.2f: %d", corr_threshold, len(flagged))
    for d in flagged:
        log.info("  %s vs %s  r = %.4f", d["feature_a"], d["feature_b"],
                 d["pearson_r"])

    # ---- VIF for flagged features ----
    flagged_feats = list(
        {f for d in flagged for f in (d["feature_a"], d["feature_b"])})
    vif_results = pd.Series(dtype=float)
    if flagged_feats:
        log.info("Computing VIF for %d flagged features (threshold=%.1f) …",
                 len(flagged_feats), vif_threshold)
        vif_results = compute_vif(X[flagged_feats])
        for feat, v in vif_results.items():
            flag = "  ← investigate (VIF > %.0f)" % vif_threshold if v > vif_threshold else ""
            log.info("  VIF %-40s = %.2f%s", feat, v, flag)

    # ---- Save flagged pairs CSV ----
    RESULTS_DIR.mkdir(exist_ok=True)
    pairs_path = RESULTS_DIR / "correlation_flagged_pairs.csv"
    pd.DataFrame(flagged).to_csv(pairs_path, index=False)
    log.info("Flagged pairs → %s", pairs_path)

    corr_path = RESULTS_DIR / "correlation_matrix_train.csv"
    corr.to_csv(corr_path)
    scope_path = RESULTS_DIR / "correlation_scope.json"
    scope_path.write_text(json.dumps({
        "source_table": str(data_dir / "train.parquet"),
        "scope": "local_train_only",
        "n_rows": int(len(train)),
        "n_features": int(len(num_cols)),
        "categorical_excluded": CAT_COLS,
        "corr_threshold": float(corr_threshold),
        "vif_threshold": float(vif_threshold),
    }, indent=2))
    log.info("Correlation matrix → %s", corr_path)
    log.info("Scope metadata → %s", scope_path)

    if not vif_results.empty:
        vif_path = RESULTS_DIR / "vif_scores.csv"
        vif_results.rename("vif").reset_index().rename(
            columns={"index": "feature"}).to_csv(vif_path, index=False)
        log.info("VIF scores → %s", vif_path)

    # ---- Heatmap ----
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(12, len(num_cols) * 0.35),
                                    max(10, len(num_cols) * 0.35)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", center=0, vmin=-1, vmax=1,
        ax=ax, square=True, linewidths=0.3,
        cbar_kws={"shrink": 0.7},
        xticklabels=True, yticklabels=True,
    )
    ax.set_title(
        f"Feature Correlation Matrix (Pearson) — "
        f"{len(flagged)} pairs with |r| > {corr_threshold}",
        fontsize=12,
    )
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()

    hm_path = PLOTS_DIR / "correlation_heatmap.png"
    plt.savefig(hm_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Heatmap → %s", hm_path)

    return corr, flagged, vif_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Flag pairs with |r| above this (default 0.9)")
    parser.add_argument("--vif-threshold", type=float, default=5.0,
                        help="VIF threshold for investigation flag (default 5)")
    args = parser.parse_args()
    run(args.data_dir, args.threshold, args.vif_threshold)
