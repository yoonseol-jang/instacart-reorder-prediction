"""
Phase 2 + 3 — Feature Engineering and Train/Test Split.

Builds all selected features for every labeled user's historical
(user, product) pair, attaches the binary reorder label, and writes
leakage-safe local train/test parquet files to data/processed/.

Usage:
    python features.py                  # full labeled data (~8.5M rows, needs ~8GB RAM)
    python features.py --sample 0.1     # 10% of users, for local dev/testing
    python features.py --sample 0.02 --output-dir /tmp/instacart_processed_sample

Phase 3 split: 80/20 at user level, stratified by activity tier,
so no user appears in both train and test.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA      = Path(__file__).parent.parent / "data"
PROCESSED = DATA / "processed"
PROCESSED.mkdir(exist_ok=True)

RANDOM_STATE = 42

DROPPED_FEATURES = {
    "user_std_days_between_orders",
    "user_unique_aisles",
    "user_unique_departments",
    "prod_total_users",
    "up_purchase_ratio_n5",
    "dow_match_user_pref",
    "hour_diff_from_user_pref",
}

CAT_COLS = {"aisle_id", "department_id"}
EXPECTED_NUMERIC_FEATURE_RANGE = (33, 35)


def assert_eval_set(orders_df: pd.DataFrame, expected: str, name: str) -> None:
    """Fail fast if an order subset contains rows from the wrong eval_set."""
    if "eval_set" not in orders_df.columns:
        raise ValueError(f"{name} is missing eval_set; cannot validate leakage.")
    bad = orders_df.loc[orders_df["eval_set"] != expected, "eval_set"].unique()
    if len(bad):
        raise ValueError(f"{name} contains eval_set values {bad}, expected {expected!r}.")


def merge_preserve_rows(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on,
    how: str,
    name: str,
) -> pd.DataFrame:
    """Merge and assert the merge did not duplicate or drop candidate rows."""
    before = len(left)
    out = left.merge(right, on=on, how=how)
    after = len(out)
    if after != before:
        raise ValueError(
            f"{name} changed row count during merge: before={before:,}, after={after:,}"
        )
    return out


def validate_feature_columns(feature_cols: list[str]) -> None:
    """Validate logical prefilter and final numerical feature count."""
    present_dropped = sorted(DROPPED_FEATURES.intersection(feature_cols))
    if present_dropped:
        raise ValueError(f"Logical pre-filter failed; dropped features present: {present_dropped}")

    numeric_count = len([c for c in feature_cols if c not in CAT_COLS])
    lo, hi = EXPECTED_NUMERIC_FEATURE_RANGE
    if not lo <= numeric_count <= hi:
        raise ValueError(
            f"Expected {lo}-{hi} numerical features, found {numeric_count}."
        )


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_raw(sample_frac: float | None = None, seed: int = RANDOM_STATE):
    log.info("Loading CSVs …")
    orders      = pd.read_csv(DATA / "orders.csv")
    prior       = pd.read_csv(DATA / "order_products__prior.csv")
    train_prod  = pd.read_csv(DATA / "order_products__train.csv")
    products    = pd.read_csv(DATA / "products.csv")
    aisles      = pd.read_csv(DATA / "aisles.csv")
    departments = pd.read_csv(DATA / "departments.csv")

    if sample_frac is not None:
        rng = np.random.default_rng(seed)
        all_users = orders["user_id"].unique()
        kept = rng.choice(all_users,
                          size=max(1, int(len(all_users) * sample_frac)),
                          replace=False)
        kept_set = set(kept.tolist())
        orders     = orders[orders["user_id"].isin(kept_set)].copy()
        kept_oids  = set(orders["order_id"].tolist())
        prior      = prior[prior["order_id"].isin(kept_oids)].copy()
        train_prod = train_prod[train_prod["order_id"].isin(kept_oids)].copy()
        log.info("Sampled %.0f%% → %d users", sample_frac * 100, len(kept_set))

    prior_orders = orders[orders["eval_set"] == "prior"].copy()
    train_orders = orders[orders["eval_set"] == "train"].copy()
    assert_eval_set(prior_orders, "prior", "prior_orders")
    assert_eval_set(train_orders, "train", "train_orders")
    prod_info    = (products
                    .merge(aisles, on="aisle_id")
                    .merge(departments, on="department_id"))

    log.info("prior rows=%d  train rows=%d  users=%d",
             len(prior), len(train_prod), orders["user_id"].nunique())
    return prior, train_prod, orders, prior_orders, train_orders, prod_info


# ---------------------------------------------------------------------------
# 2. Labels
# ---------------------------------------------------------------------------

def build_labels(prior, prior_orders, train_prod, train_orders):
    log.info("Building label table …")
    assert_eval_set(prior_orders, "prior", "prior_orders")
    assert_eval_set(train_orders, "train", "train_orders")
    # Candidate (user, product) pairs: products bought in prior by users whose
    # next order is labeled in the official Kaggle train split.
    train_user_ids = set(train_orders["user_id"])
    prior_up = (prior
                .merge(prior_orders[["order_id", "user_id"]], on="order_id")
                [["user_id", "product_id"]]
                .drop_duplicates())
    prior_up = prior_up[prior_up["user_id"].isin(train_user_ids)]

    # Positive labels: products that appear in the user's train order
    pos = (train_prod
           .merge(train_orders[["order_id", "user_id"]], on="order_id")
           [["user_id", "product_id"]]
           .assign(label=np.int8(1)))

    labels = prior_up.merge(pos, on=["user_id", "product_id"], how="left")
    labels["label"] = labels["label"].fillna(np.int8(0))
    if not labels["user_id"].isin(train_user_ids).all():
        raise ValueError("Label table contains users without Kaggle train labels.")
    label_rate = labels["label"].mean()
    log.info("Labels: %d rows  pos=%.3f", len(labels), label_rate)
    return labels


# ---------------------------------------------------------------------------
# 3. Group 1 — User-level features (9)
# ---------------------------------------------------------------------------

def compute_user_features(prior, prior_orders, orders):
    log.info("Group 1: user-level features …")

    # prior_orders is already a subset of orders and contains all needed columns
    po_meta = prior_orders.copy()

    agg = (po_meta.groupby("user_id").agg(
        user_total_orders         = ("order_id", "nunique"),
        user_avg_days_between_orders = ("days_since_prior_order", "mean"),
    ).reset_index())

    # Mode day-of-week and hour (most common in prior orders)
    dow_mode  = (po_meta.groupby("user_id")["order_dow"]
                 .agg(lambda x: x.mode().iat[0])
                 .rename("user_pref_dow"))
    hour_mode = (po_meta.groupby("user_id")["order_hour_of_day"]
                 .agg(lambda x: x.mode().iat[0])
                 .rename("user_pref_hour"))
    agg = agg.join(dow_mode,  on="user_id").join(hour_mode, on="user_id")

    # user_total_products, user_total_distinct_products, user_avg/std_basket_size,
    # user_reorder_ratio — need product rows
    pp = prior.merge(prior_orders[["order_id", "user_id"]], on="order_id")

    basket_sz = (pp.groupby(["user_id", "order_id"])["product_id"]
                 .count().rename("bs").reset_index()
                 .groupby("user_id")["bs"]
                 .agg(user_avg_basket_size="mean", user_std_basket_size="std")
                 .reset_index())

    prod_stats = (pp.groupby("user_id").agg(
        user_total_products          = ("product_id", "count"),
        user_total_distinct_products = ("product_id", "nunique"),
        user_reorder_ratio           = ("reordered", "mean"),
    ).reset_index())

    agg = agg.merge(basket_sz,   on="user_id").merge(prod_stats, on="user_id")
    return agg


# ---------------------------------------------------------------------------
# 4. Group 2 — Product-level features (9)
# ---------------------------------------------------------------------------

def compute_product_features(prior, prior_orders, prod_info):
    log.info("Group 2: product-level features …")

    pf = (prior.groupby("product_id").agg(
        prod_total_purchases   = ("order_id", "count"),
        prod_reorder_ratio     = ("reordered", "mean"),
        prod_avg_cart_position = ("add_to_cart_order", "mean"),
    ).reset_index())

    # prod_reorder_probability: among users who bought product ≥1 time,
    # fraction who bought it ≥2 times
    pp_user = prior.merge(prior_orders[["order_id", "user_id"]], on="order_id")
    up_cnt = (pp_user.groupby(["user_id", "product_id"])["order_id"]
              .count().rename("cnt").reset_index())
    up_cnt["bought_twice"] = (up_cnt["cnt"] >= 2).astype(np.int8)
    reorder_prob = (up_cnt.groupby("product_id").agg(
        n_users=("user_id", "count"),
        n_twice=("bought_twice", "sum"),
    ).reset_index())
    reorder_prob["prod_reorder_probability"] = (
        reorder_prob["n_twice"] / reorder_prob["n_users"]
    )
    reorder_prob = reorder_prob[["product_id", "prod_reorder_probability"]]
    pf = pf.merge(reorder_prob, on="product_id", how="left")

    # prod_avg_orders_between — derived from up features; filled in assemble()

    # aisle_id, department_id (for encoding), aisle_reorder_ratio, dept_reorder_ratio
    pf = pf.merge(prod_info[["product_id", "aisle_id", "department_id"]], on="product_id")

    aisle_dr = (prior.merge(prod_info[["product_id", "aisle_id"]], on="product_id")
                .groupby("aisle_id")["reordered"].mean()
                .rename("aisle_reorder_ratio").reset_index())
    dept_dr  = (prior.merge(prod_info[["product_id", "department_id"]], on="product_id")
                .groupby("department_id")["reordered"].mean()
                .rename("dept_reorder_ratio").reset_index())

    pf = pf.merge(aisle_dr, on="aisle_id").merge(dept_dr, on="department_id")
    return pf


# ---------------------------------------------------------------------------
# 5. Groups 3 + 3b — User × Product features (11 + 3 = 14)
# ---------------------------------------------------------------------------

def compute_up_features(prior, prior_orders, orders):
    log.info("Group 3/3b: user×product features …")

    # Attach user_id and order_number to every prior product row
    pp = prior.merge(
        prior_orders[["order_id", "user_id", "order_number"]],
        on="order_id"
    )

    # --- Base aggregates ---
    base = (pp.groupby(["user_id", "product_id"]).agg(
        up_purchase_count      = ("order_id", "count"),
        up_first_order_number  = ("order_number", "min"),
        up_last_order_number   = ("order_number", "max"),
        up_avg_cart_position   = ("add_to_cart_order", "mean"),
    ).reset_index())

    user_max_order = (prior_orders.groupby("user_id")["order_number"]
                      .max().rename("max_order_number").reset_index())
    base = base.merge(user_max_order, on="user_id")

    # up_orders_since_last
    base["up_orders_since_last"] = (base["max_order_number"]
                                    - base["up_last_order_number"])

    # up_order_ratio_by_chance
    span = (base["up_last_order_number"] - base["up_first_order_number"] + 1).clip(lower=1)
    base["up_order_ratio_by_chance"] = base["up_purchase_count"] / span

    # up_reorder_rate_after_first
    base["up_reorder_rate_after_first"] = np.where(
        base["up_purchase_count"] <= 1,
        0.0,
        (base["up_purchase_count"] - 1) /
        base["up_orders_since_last"].clip(lower=1)
    )

    # --- Days-based features ---
    log.info("  … building cumulative-days timeline")
    po_days = prior_orders[["user_id", "order_number", "days_since_prior_order"]].copy()
    po_days["days_since_prior_order"] = po_days["days_since_prior_order"].fillna(0.0)
    po_days = po_days.sort_values(["user_id", "order_number"])
    po_days["cum_days"] = (po_days.groupby("user_id")["days_since_prior_order"]
                           .cumsum())

    # Merge cum_days onto product occurrences
    pp_days = pp.merge(
        po_days[["user_id", "order_number", "cum_days"]],
        on=["user_id", "order_number"],
        how="left"
    )
    pp_days = pp_days.sort_values(["user_id", "product_id", "order_number"])

    # Gap between consecutive purchases of the same (user, product)
    pp_days["prev_cum"] = (pp_days.groupby(["user_id", "product_id"])["cum_days"]
                           .shift(1))
    pp_days["gap"] = pp_days["cum_days"] - pp_days["prev_cum"]

    gap_stats = (pp_days.groupby(["user_id", "product_id"])["gap"].agg(
        up_avg_days_between = "mean",
        up_max_days_between = "max",
    ).reset_index())
    base = base.merge(gap_stats, on=["user_id", "product_id"], how="left")

    # up_days_since_last
    last_prod_cum = (pp_days.groupby(["user_id", "product_id"])["cum_days"]
                     .max().rename("last_prod_cum").reset_index())
    user_last_cum = (po_days.groupby("user_id")["cum_days"]
                     .max().rename("user_last_cum").reset_index())
    dsl = last_prod_cum.merge(user_last_cum, on="user_id")
    dsl["up_days_since_last"] = dsl["user_last_cum"] - dsl["last_prod_cum"]
    base = base.merge(dsl[["user_id", "product_id", "up_days_since_last"]],
                      on=["user_id", "product_id"], how="left")

    # --- up_streak (vectorized cumprod approach) ---
    log.info("  … computing up_streak")
    pp_s = pp[["user_id", "product_id", "order_number"]].copy()
    pp_s = pp_s.merge(user_max_order, on="user_id")
    pp_s = pp_s.sort_values(["user_id", "product_id", "order_number"],
                             ascending=[True, True, False])
    pp_s["rnk"] = pp_s.groupby(["user_id", "product_id"]).cumcount() + 1
    pp_s["expected"] = pp_s["max_order_number"] - (pp_s["rnk"] - 1)
    pp_s["is_consec"] = (pp_s["order_number"] == pp_s["expected"]).astype(np.int8)
    pp_s["consec_run"] = pp_s.groupby(["user_id", "product_id"])["is_consec"].cumprod()
    streak = (pp_s.groupby(["user_id", "product_id"])["consec_run"]
              .sum().rename("up_streak").reset_index())
    base = base.merge(streak, on=["user_id", "product_id"], how="left")

    # prod_avg_orders_between (product-level average of up_avg_days_between)
    prod_avg_ob = (base.groupby("product_id")["up_avg_days_between"]
                   .mean().rename("prod_avg_orders_between").reset_index())

    # --- Group 3b: recency-windowed (_n5) ---
    log.info("  … computing _n5 recency features")
    N5 = 5
    pp_n5 = pp.merge(user_max_order, on="user_id")
    pp_n5 = pp_n5[pp_n5["order_number"] > pp_n5["max_order_number"] - N5]

    # up_purchase_count_n5
    n5_agg = (pp_n5.groupby(["user_id", "product_id"])["order_id"]
              .count().rename("up_purchase_count_n5").reset_index())

    # up_orders_since_last_n5
    n5_last = (pp_n5.groupby(["user_id", "product_id"])["order_number"]
               .max().rename("last_n5_ord").reset_index()
               .merge(user_max_order, on="user_id"))
    n5_last["up_orders_since_last_n5"] = (
        (n5_last["max_order_number"] - n5_last["last_n5_ord"]).clip(upper=N5)
    )
    n5_agg = n5_agg.merge(
        n5_last[["user_id", "product_id", "up_orders_since_last_n5"]],
        on=["user_id", "product_id"], how="left"
    )

    # up_avg_cart_position_n5
    n5_cart = (pp_n5.groupby(["user_id", "product_id"])["add_to_cart_order"]
               .mean().rename("up_avg_cart_position_n5").reset_index())
    n5_agg = n5_agg.merge(n5_cart, on=["user_id", "product_id"], how="left")

    # up_streak_n5: consecutive most-recent orders within the last-5 window
    pp_n5_s = pp_n5[["user_id", "product_id", "order_number"]].copy()
    pp_n5_s = pp_n5_s.merge(user_max_order, on="user_id")
    pp_n5_s = pp_n5_s.sort_values(["user_id", "product_id", "order_number"],
                                   ascending=[True, True, False])
    pp_n5_s["rnk_n5"] = pp_n5_s.groupby(["user_id", "product_id"]).cumcount() + 1
    pp_n5_s["expected_n5"] = pp_n5_s["max_order_number"] - (pp_n5_s["rnk_n5"] - 1)
    pp_n5_s["is_consec_n5"] = (pp_n5_s["order_number"] == pp_n5_s["expected_n5"]).astype(np.int8)
    pp_n5_s["consec_run_n5"] = (pp_n5_s.groupby(["user_id", "product_id"])["is_consec_n5"]
                                .cumprod())
    streak_n5 = (pp_n5_s.groupby(["user_id", "product_id"])["consec_run_n5"]
                 .sum().rename("up_streak_n5").reset_index())
    n5_agg = n5_agg.merge(streak_n5, on=["user_id", "product_id"], how="left")

    base = base.merge(n5_agg, on=["user_id", "product_id"], how="left")
    base["up_purchase_count_n5"]    = base["up_purchase_count_n5"].fillna(0).astype(np.int16)
    base["up_orders_since_last_n5"] = base["up_orders_since_last_n5"].fillna(N5).astype(np.int8)
    base["up_avg_cart_position_n5"] = base["up_avg_cart_position_n5"].fillna(
        base["up_avg_cart_position"])
    base["up_streak_n5"]            = base["up_streak_n5"].fillna(0).astype(np.int8)

    base = base.drop(columns=["max_order_number"])
    return base, prod_avg_ob


# ---------------------------------------------------------------------------
# 6. Group 4 — Datetime features (4)
# ---------------------------------------------------------------------------

def compute_datetime_features(train_orders, prior, prior_orders):
    log.info("Group 4: datetime features …")

    dt = (train_orders[["user_id", "order_dow", "order_hour_of_day",
                         "days_since_prior_order"]]
          .rename(columns={
              "order_dow": "target_dow",
              "order_hour_of_day": "target_hour",
              "days_since_prior_order": "target_days_since_prior",
          }))

    # up_dow_purchase_count: times user bought this product on the target DOW
    # prior_orders already has order_dow (it's a subset of orders)
    pp = prior.merge(prior_orders[["order_id", "user_id", "order_dow"]], on="order_id")
    user_target_dow = train_orders[["user_id", "order_dow"]].rename(
        columns={"order_dow": "target_dow"})
    dow_cnt = (pp.merge(user_target_dow, on="user_id")
               .query("order_dow == target_dow")
               .groupby(["user_id", "product_id"])["order_id"]
               .count().rename("up_dow_purchase_count").reset_index())

    return dt, dow_cnt


# ---------------------------------------------------------------------------
# 7. Assemble full feature matrix
# ---------------------------------------------------------------------------

def assemble(prior, train_prod, orders, prior_orders, train_orders, prod_info):
    t0 = time.time()

    labels  = build_labels(prior, prior_orders, train_prod, train_orders)
    user_df = compute_user_features(prior, prior_orders, orders)
    prod_df = compute_product_features(prior, prior_orders, prod_info)
    up_df, prod_avg_ob = compute_up_features(prior, prior_orders, orders)
    prod_df = prod_df.merge(prod_avg_ob, on="product_id", how="left")
    dt_df, dow_cnt = compute_datetime_features(train_orders, prior, prior_orders)

    log.info("Merging all feature groups …")
    df = labels.copy()
    df = merge_preserve_rows(
        df, user_df, on="user_id", how="left", name="user features"
    )
    df = merge_preserve_rows(
        df, prod_df, on="product_id", how="left", name="product features"
    )
    df = merge_preserve_rows(
        df, up_df, on=["user_id", "product_id"], how="left", name="user-product features"
    )
    df = merge_preserve_rows(
        df, dt_df, on="user_id", how="left", name="target datetime features"
    )
    df = merge_preserve_rows(
        df, dow_cnt, on=["user_id", "product_id"], how="left", name="target-DOW count"
    )

    df["up_dow_purchase_count"]   = df["up_dow_purchase_count"].fillna(0).astype(np.int16)
    df["target_days_since_prior"] = df["target_days_since_prior"].fillna(
        df["target_days_since_prior"].median())

    log.info("Feature matrix: %d rows × %d cols  (%.0f s)",
             len(df), df.shape[1], time.time() - t0)
    feature_cols = [c for c in df.columns if c not in {"user_id", "product_id", "label"}]
    validate_feature_columns(feature_cols)
    return df


# ---------------------------------------------------------------------------
# 8. Phase 3 — Train/test split (user-level, stratified by activity tier)
# ---------------------------------------------------------------------------

def split_and_save(df, prior_orders, output_dir: Path = PROCESSED):
    log.info("Phase 3: user-level 80/20 split …")
    output_dir.mkdir(parents=True, exist_ok=True)

    user_n_orders = (prior_orders.groupby("user_id")["order_id"]
                     .count().rename("n"))
    users = (df[["user_id"]].drop_duplicates()
             .merge(user_n_orders, on="user_id"))
    users["tier"] = pd.cut(users["n"],
                           bins=[0, 5, 15, 30, 9999],
                           labels=[0, 1, 2, 3]).astype(int)

    local_train_uids, local_test_uids = train_test_split(
        users["user_id"],
        test_size=0.2,
        stratify=users["tier"],
        random_state=RANDOM_STATE,
    )
    train_set = set(local_train_uids.tolist())
    test_set  = set(local_test_uids.tolist())
    overlap = train_set.intersection(test_set)
    if overlap:
        raise ValueError(f"User overlap between local train/test splits: {len(overlap)}")
    assert train_set.intersection(test_set) == set()

    df_train = df[df["user_id"].isin(train_set)].reset_index(drop=True)
    df_test  = df[df["user_id"].isin(test_set)].reset_index(drop=True)
    train_mean_orders = df_train[["user_id", "user_total_orders"]].drop_duplicates()[
        "user_total_orders"].mean()
    test_mean_orders = df_test[["user_id", "user_total_orders"]].drop_duplicates()[
        "user_total_orders"].mean()
    mean_order_delta = abs(train_mean_orders - test_mean_orders)
    if mean_order_delta > 0.5:
        raise ValueError(
            "Stratification check failed: mean user_total_orders differs by "
            f"{mean_order_delta:.2f} (train={train_mean_orders:.2f}, "
            f"test={test_mean_orders:.2f})."
        )

    log.info("Train: %d rows  pos=%.4f", len(df_train), df_train["label"].mean())
    log.info("Test : %d rows  pos=%.4f", len(df_test),  df_test["label"].mean())
    log.info("User overlap: %d", len(overlap))
    log.info("Mean user_total_orders: train=%.2f  test=%.2f",
             train_mean_orders, test_mean_orders)

    df_train.to_parquet(output_dir / "train.parquet", index=False)
    df_test.to_parquet(output_dir / "test.parquet",   index=False)

    feature_cols = [c for c in df.columns if c not in {"user_id", "product_id", "label"}]
    validate_feature_columns(feature_cols)
    (output_dir / "feature_cols.txt").write_text("\n".join(feature_cols))
    log.info("Saved → %s  (%d features)", output_dir, len(feature_cols))
    log.info("Features: %s", feature_cols)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=float, default=None,
                        help="Fraction of users to keep (e.g. 0.1)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED,
                        help="Directory for train/test parquet and feature_cols.txt")
    args = parser.parse_args()

    raw = load_raw(sample_frac=args.sample, seed=args.seed)
    df  = assemble(*raw)
    split_and_save(df, raw[3], args.output_dir)   # raw[3] = prior_orders
