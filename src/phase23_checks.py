"""
Executable Phase 2/3 checklist checks.

Run after feature generation:
    python phase23_checks.py

The script validates the persisted local train/test split, feature pre-filter,
target-encoding fold behavior, raw eval_set boundaries, and correlation-check
scope metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from train_utils import (
    CAT_COLS,
    LaplaceTargetEncoder,
    assert_group_cv_no_user_overlap,
    validate_feature_columns,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def check_split(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    train_users = set(train["user_id"].unique())
    test_users = set(test["user_id"].unique())
    overlap = train_users.intersection(test_users)
    if overlap:
        raise ValueError(f"Local train/test user overlap: {len(overlap)} users")

    train_user_orders = train[["user_id", "user_total_orders"]].drop_duplicates()
    test_user_orders = test[["user_id", "user_total_orders"]].drop_duplicates()
    train_mean = float(train_user_orders["user_total_orders"].mean())
    test_mean = float(test_user_orders["user_total_orders"].mean())
    delta = abs(train_mean - test_mean)
    if delta > 0.5:
        raise ValueError(
            f"Stratification failed: train mean={train_mean:.3f}, "
            f"test mean={test_mean:.3f}, delta={delta:.3f}"
        )

    log.info("User overlap: 0")
    log.info("Mean user_total_orders: train=%.3f  test=%.3f  delta=%.3f",
             train_mean, test_mean, delta)
    return {
        "train_users": len(train_users),
        "test_users": len(test_users),
        "user_overlap": 0,
        "train_mean_user_total_orders": train_mean,
        "test_mean_user_total_orders": test_mean,
        "mean_user_total_orders_delta": delta,
    }


def check_raw_eval_sets(raw_dir: Path) -> dict:
    orders = pd.read_csv(raw_dir / "orders.csv", usecols=["order_id", "eval_set"])
    order_eval = orders.set_index("order_id")["eval_set"]

    prior_order_ids = pd.read_csv(
        raw_dir / "order_products__prior.csv", usecols=["order_id"]
    )["order_id"].drop_duplicates()
    train_order_ids = pd.read_csv(
        raw_dir / "order_products__train.csv", usecols=["order_id"]
    )["order_id"].drop_duplicates()

    prior_eval_values = set(order_eval.loc[prior_order_ids].unique())
    train_eval_values = set(order_eval.loc[train_order_ids].unique())
    if prior_eval_values != {"prior"}:
        raise ValueError(f"Prior product rows map to eval_set={prior_eval_values}")
    if train_eval_values != {"train"}:
        raise ValueError(f"Train label rows map to eval_set={train_eval_values}")

    log.info("Raw eval_set check passed: feature rows=prior, labels=train")
    return {
        "prior_product_eval_sets": sorted(prior_eval_values),
        "train_label_eval_sets": sorted(train_eval_values),
    }


def check_feature_matrix(train: pd.DataFrame, test: pd.DataFrame,
                         feature_cols: list[str]) -> dict:
    validate_feature_columns(feature_cols)

    required_cols = {"user_id", "product_id", "label", *feature_cols}
    for name, df in [("train", train), ("test", test)]:
        missing = sorted(required_cols.difference(df.columns))
        if missing:
            raise ValueError(f"{name}.parquet missing columns: {missing}")
        dupes = df.duplicated(["user_id", "product_id"]).sum()
        if dupes:
            raise ValueError(f"{name}.parquet has duplicated user/product rows: {dupes}")

    numeric_count = len([c for c in feature_cols if c not in CAT_COLS])
    log.info("Feature pre-filter passed: %d total, %d numeric",
             len(feature_cols), numeric_count)
    return {
        "total_features": len(feature_cols),
        "numeric_features": numeric_count,
    }


def check_group_kfold_and_target_encoding(
    train: pd.DataFrame,
    feature_cols: list[str],
    results_dir: Path,
) -> dict:
    X = train[feature_cols]
    y = train["label"].to_numpy()
    groups = train["user_id"].to_numpy()

    gkf = GroupKFold(n_splits=5)
    assert_group_cv_no_user_overlap(gkf, X, y, groups)

    aisle_to_check = int(train["aisle_id"].value_counts().idxmax())
    rows = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        enc = LaplaceTargetEncoder(cat_cols=["aisle_id"], smooth=10.0)
        enc.fit(X.iloc[tr_idx], y[tr_idx])
        value = float(enc.maps_["aisle_id"].get(aisle_to_check, enc.global_mean_))
        rows.append({
            "fold": fold,
            "aisle_id": aisle_to_check,
            "aisle_reorder_ratio_fold_value": value,
            "train_users": int(pd.Series(groups[tr_idx]).nunique()),
            "valid_users": int(pd.Series(groups[va_idx]).nunique()),
        })

    out = pd.DataFrame(rows)
    if out["aisle_reorder_ratio_fold_value"].nunique() <= 1:
        raise ValueError(
            "Fold target encoding did not fluctuate; possible global target leakage."
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "target_encoding_fold_check.csv"
    out.to_csv(path, index=False)
    log.info("Target encoding fold check → %s", path)
    return {
        "target_encoding_checked_aisle_id": aisle_to_check,
        "target_encoding_unique_values": int(
            out["aisle_reorder_ratio_fold_value"].nunique()
        ),
    }


def check_correlation_scope(results_dir: Path) -> dict:
    scope_path = results_dir / "correlation_scope.json"
    if not scope_path.exists():
        raise ValueError(
            "Missing correlation_scope.json. Run correlation_check.py after features.py."
        )

    scope = json.loads(scope_path.read_text())
    if scope.get("scope") != "local_train_only":
        raise ValueError(f"Correlation scope is not local_train_only: {scope}")
    if not str(scope.get("source_table", "")).endswith("train.parquet"):
        raise ValueError(f"Correlation source is not train.parquet: {scope}")

    log.info("Correlation scope check passed: local_train_only")
    return scope


def run(data_dir: Path, raw_dir: Path, results_dir: Path) -> dict:
    feature_cols = (data_dir / "feature_cols.txt").read_text().splitlines()
    train = pd.read_parquet(data_dir / "train.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")

    report = {
        "split": check_split(train, test),
        "raw_eval_sets": check_raw_eval_sets(raw_dir),
        "feature_matrix": check_feature_matrix(train, test, feature_cols),
        "target_encoding": check_group_kfold_and_target_encoding(
            train, feature_cols, results_dir
        ),
        "correlation_scope": check_correlation_scope(results_dir),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "phase23_checks.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Phase 2/3 checks passed → %s", report_path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    run(args.data_dir, args.raw_dir, args.results_dir)
