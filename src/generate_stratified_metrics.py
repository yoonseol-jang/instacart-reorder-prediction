"""
Generate cohort-stratified performance metrics for §8 Table 5.

Three slicing axes:
    Product reorder rate   - high (>=0.60) / mid (0.20-0.60) / occasional (<0.20)
    User activity tier     - cold-start (<5 orders) / regular (5-29) / power (>=30)
    Aisle purchase variance - high (top-20 std) / low (bottom-20 std)

Inputs:
    results/test_predictions.parquet      - aligned 5-model preds + features
    data/order_products__prior.csv        - prior-only reorder labels (for aisle variance)
    data/products.csv                     - product -> aisle_id

Output:
    results/report_tables/report_stratified_performance.csv

Hygiene:
    1. Product reorder rate uses prod_reorder_ratio from features.py:216, which
       is computed as prior.groupby('product_id')['reordered'].mean(). Prior-only.
    2. Aisle variance reuses EDA cell exec=17 logic verbatim:
       std of per-product reorder rates within aisle, filtered to n_products>=5.
    3. All metrics are rolled up via a single melt + groupby aggregation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRED_PATH = ROOT / "results" / "test_predictions.parquet"
PRIOR_PATH = ROOT / "data" / "order_products__prior.csv"
PRODUCTS_PATH = ROOT / "data" / "products.csv"
OUT_PATH = ROOT / "results" / "report_tables" / "report_stratified_performance.csv"

MODELS = ["last_basket", "logreg", "rf", "adaboost", "xgboost"]
MODEL_LABELS = {
    "last_basket": "Last Basket",
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "adaboost": "AdaBoost",
    "xgboost": "XGBoost",
}


def load_predictions() -> pd.DataFrame:
    df = pd.read_parquet(PRED_PATH)
    needed = {"user_id", "product_id", "label", "aisle_id",
              "prod_reorder_ratio", "user_total_orders"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"test_predictions missing columns: {missing}")
    for m in MODELS:
        if f"{m}_pred" not in df.columns:
            raise ValueError(f"test_predictions missing {m}_pred column")
    return df


def compute_aisle_variance_partition() -> tuple[set[int], set[int]]:
    """Reuse EDA cell exec=17 verbatim: std of product reorder rates within aisle,
    filtered to aisles with n_products>=5. Top-20 are 'high variance'; bottom-20
    are 'low variance'."""
    prior = pd.read_csv(PRIOR_PATH, usecols=["product_id", "reordered"])
    products = pd.read_csv(PRODUCTS_PATH, usecols=["product_id", "aisle_id"])

    product_reorder_rate = prior.groupby("product_id")["reordered"].mean()
    prod_aisle = products.merge(
        product_reorder_rate.rename("reorder_rate"), on="product_id"
    )
    aisle_var = (
        prod_aisle.groupby("aisle_id")["reorder_rate"]
        .agg(["std", "count"])
        .rename(columns={"std": "rate_std", "count": "n_products"})
        .dropna()
        .query("n_products >= 5")
        .sort_values("rate_std", ascending=False)
    )
    high = set(aisle_var.head(20).index.astype(int).tolist())
    low = set(aisle_var.tail(20).index.astype(int).tolist())
    return high, low


def add_cohort_columns(
    df: pd.DataFrame,
    high_var_aisles: set[int],
    low_var_aisles: set[int],
) -> pd.DataFrame:
    out = df.copy()

    # Product reorder rate cohort
    rr = out["prod_reorder_ratio"].astype(float)
    out["product_cohort"] = np.select(
        [rr >= 0.60, (rr >= 0.20) & (rr < 0.60), rr < 0.20],
        ["high_reorder", "mid_reorder", "occasional"],
        default="other",
    )

    # User activity cohort
    uto = out["user_total_orders"].astype(int)
    out["user_cohort"] = np.select(
        [uto < 5, (uto >= 5) & (uto < 30), uto >= 30],
        ["cold_start", "regular", "power_user"],
        default="other",
    )

    # Aisle variance cohort
    out["aisle_cohort"] = np.select(
        [
            out["aisle_id"].astype(int).isin(high_var_aisles),
            out["aisle_id"].astype(int).isin(low_var_aisles),
        ],
        ["high_variance", "low_variance"],
        default="other",
    )
    return out


def metrics_table(df: pd.DataFrame, axis: str, cohort_col: str) -> pd.DataFrame:
    """For each (model, cohort_value) compute precision, recall, f1, miss, fa."""
    base = df[[cohort_col, "label"]].rename(columns={cohort_col: "cohort_value"})
    rows = []
    for m in MODELS:
        sub = base.copy()
        sub["pred"] = df[f"{m}_pred"].astype(int).values
        sub["model"] = m
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    long = long[long["cohort_value"] != "other"]

    grp = long.groupby(["model", "cohort_value"], observed=True)
    agg = grp.apply(
        lambda g: pd.Series(
            _confusion_metrics(g["label"].to_numpy(), g["pred"].to_numpy())
        ),
        include_groups=False,
    ).reset_index()
    agg.insert(0, "axis", axis)
    agg["model_label"] = agg["model"].map(MODEL_LABELS)
    return agg


def _confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    n_rows = tp + fp + fn + tn
    pos = tp + fn
    neg = fp + tn
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / pos if pos else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and not np.isnan(precision) and not np.isnan(recall)
        else (0.0 if (tp + fp + fn) > 0 else np.nan)
    )
    miss = fn / pos if pos else np.nan
    fa = fp / neg if neg else np.nan
    return {
        "n_rows": n_rows,
        "n_positives": pos,
        "positive_rate": pos / n_rows if n_rows else np.nan,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "miss_rate": miss,
        "false_alarm_rate": fa,
    }


def main() -> None:
    print(f"Loading {PRED_PATH} ...")
    preds = load_predictions()
    print(f"  rows={len(preds):,}  users={preds['user_id'].nunique():,}")

    print("Computing aisle-variance partition (EDA cell exec=17 logic) ...")
    high_aisles, low_aisles = compute_aisle_variance_partition()
    print(f"  high-variance aisles: {len(high_aisles)}; low-variance aisles: {len(low_aisles)}")

    print("Tagging cohorts ...")
    tagged = add_cohort_columns(preds, high_aisles, low_aisles)

    for axis_col, axis_name in [
        ("product_cohort", "Product reorder rate"),
        ("user_cohort", "User activity"),
        ("aisle_cohort", "Aisle variance"),
    ]:
        counts = tagged[axis_col].value_counts(dropna=False)
        print(f"  {axis_name}: {counts.to_dict()}")

    print("Rolling up per-cohort metrics ...")
    parts = [
        metrics_table(tagged, "Product reorder rate", "product_cohort"),
        metrics_table(tagged, "User activity",        "user_cohort"),
        metrics_table(tagged, "Aisle variance",       "aisle_cohort"),
    ]
    out = pd.concat(parts, ignore_index=True)
    out = out[[
        "axis", "cohort_value", "model", "model_label",
        "n_rows", "n_positives", "positive_rate",
        "tp", "fp", "fn", "tn",
        "precision", "recall", "f1", "miss_rate", "false_alarm_rate",
    ]]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}  rows={len(out)}")


if __name__ == "__main__":
    main()
