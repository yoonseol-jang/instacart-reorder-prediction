#!/usr/bin/env python3
"""
Generate Phase 5 paper artifacts from final prediction files.

Outputs:
    results/test_predictions.parquet
    results/metrics_summary.csv
    results/model_disagreement.csv
    results/basket_metrics.csv
    results/behavioral_error_tags.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from subgroups import get_user_subgroups


DEFAULT_MODELS = ["last_basket", "logreg", "rf", "adaboost", "xgboost"]
MODEL_LABELS = {
    "last_basket": "Last Basket",
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "adaboost": "AdaBoost",
    "xgboost": "XGBoost",
}
RAW_PRIOR_COLS = ["order_id", "product_id", "add_to_cart_order", "reordered"]


def load_predictions(root: Path, models: list[str], tag: str) -> dict[str, pd.DataFrame]:
    preds: dict[str, pd.DataFrame] = {}
    base_keys: pd.DataFrame | None = None
    for model in models:
        path = root / "results" / model / f"{tag}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path).sort_values(["user_id", "product_id"]).reset_index(drop=True)
        missing = {"user_id", "product_id", "label", "prob", "pred"}.difference(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        keys = df[["user_id", "product_id", "label"]].astype({"label": "int64"})
        if base_keys is None:
            base_keys = keys
        elif not base_keys.equals(keys):
            raise ValueError(f"{model} predictions do not align with previous models")
        preds[model] = df
    return preds


def load_metric_threshold(root: Path, model: str, tag: str) -> float | None:
    path = root / "results" / model / f"{tag}_metrics.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload.get("metrics", {}).get("threshold")


def build_master_predictions(
    preds: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    first = next(iter(preds.values()))
    out = first[["user_id", "product_id", "label"]].copy()
    out = out.merge(metadata, on=["user_id", "product_id"], how="left", validate="one_to_one")
    for model, df in preds.items():
        out[f"{model}_prob"] = df["prob"].to_numpy(dtype=float)
        out[f"{model}_pred"] = df["pred"].to_numpy(dtype=int)
        out[f"{model}_error_type"] = error_type(
            out["label"].to_numpy(dtype=int),
            out[f"{model}_pred"].to_numpy(dtype=int),
        )
    return out


def error_type(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    out = np.full(len(y_true), "TN", dtype=object)
    out[(y_true == 1) & (y_pred == 1)] = "TP"
    out[(y_true == 1) & (y_pred == 0)] = "FN"
    out[(y_true == 0) & (y_pred == 1)] = "FP"
    return out


def read_relevant_prior_products(
    prior_path: Path,
    selected_order_ids: set[int],
    selected_product_ids: set[int],
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_chunks = []
    product_stat_chunks = []
    for chunk in pd.read_csv(prior_path, usecols=RAW_PRIOR_COLS, chunksize=chunksize):
        order_mask = chunk["order_id"].isin(selected_order_ids)
        if order_mask.any():
            user_chunks.append(chunk.loc[order_mask].copy())

        product_mask = chunk["product_id"].isin(selected_product_ids)
        if product_mask.any():
            part = (
                chunk.loc[product_mask]
                .groupby("product_id")["reordered"]
                .agg(prod_total_purchases="count", prod_reorder_sum="sum")
                .reset_index()
            )
            product_stat_chunks.append(part)

    if not user_chunks:
        raise ValueError("No prior product rows found for selected users.")

    user_prior = pd.concat(user_chunks, ignore_index=True)
    product_stats = (
        pd.concat(product_stat_chunks, ignore_index=True)
        .groupby("product_id", as_index=False)
        .agg(prod_total_purchases=("prod_total_purchases", "sum"), prod_reorder_sum=("prod_reorder_sum", "sum"))
    )
    product_stats["prod_reorder_ratio"] = (
        product_stats["prod_reorder_sum"] / product_stats["prod_total_purchases"]
    )
    return user_prior, product_stats.drop(columns=["prod_reorder_sum"])


def count_terminal_streak(order_numbers: list[int], max_order_number: int) -> int:
    present = set(int(x) for x in order_numbers)
    streak = 0
    current = int(max_order_number)
    while current in present:
        streak += 1
        current -= 1
    return streak


def reconstruct_metadata_from_raw(root: Path, keys: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    data = root / "data"
    users = set(keys["user_id"].astype(int))
    products = set(keys["product_id"].astype(int))

    orders = pd.read_csv(data / "orders.csv")
    prior_orders = orders[(orders["eval_set"] == "prior") & orders["user_id"].isin(users)].copy()
    target_orders = orders[(orders["eval_set"] != "prior") & orders["user_id"].isin(users)].copy()
    if prior_orders.empty:
        raise ValueError("No prior orders found for prediction users.")

    order_user = prior_orders[["order_id", "user_id", "order_number"]]
    prior_raw, product_stats = read_relevant_prior_products(
        data / "order_products__prior.csv",
        set(order_user["order_id"].astype(int)),
        products,
        chunksize,
    )
    prior = prior_raw.merge(order_user, on="order_id", how="left", validate="many_to_one")

    max_order = prior_orders.groupby("user_id")["order_number"].max().rename("max_order_number")
    user_features = (
        prior_orders.groupby("user_id")
        .agg(user_total_orders=("order_id", "nunique"))
        .join(max_order)
        .reset_index()
    )
    basket_size = (
        prior.groupby(["user_id", "order_id"])["product_id"]
        .count()
        .groupby("user_id")
        .mean()
        .rename("user_avg_basket_size")
    )
    user_reorder = prior.groupby("user_id")["reordered"].mean().rename("user_reorder_ratio")
    user_features = user_features.join(basket_size, on="user_id").join(user_reorder, on="user_id")

    pair = (
        prior.groupby(["user_id", "product_id"])
        .agg(
            up_purchase_count=("order_id", "count"),
            up_first_order_number=("order_number", "min"),
            up_last_order_number=("order_number", "max"),
            up_avg_cart_position=("add_to_cart_order", "mean"),
            order_numbers=("order_number", list),
        )
        .reset_index()
        .merge(user_features[["user_id", "max_order_number"]], on="user_id", how="left")
    )
    pair["up_orders_since_last"] = pair["max_order_number"] - pair["up_last_order_number"]
    pair["up_streak"] = [
        count_terminal_streak(order_numbers, max_order_number)
        for order_numbers, max_order_number in zip(pair["order_numbers"], pair["max_order_number"])
    ]

    prior_n5 = prior.merge(max_order.reset_index(), on="user_id", how="left")
    prior_n5 = prior_n5[prior_n5["order_number"] > prior_n5["max_order_number"] - 5]
    n5 = (
        prior_n5.groupby(["user_id", "product_id"])
        .agg(
            up_purchase_count_n5=("order_id", "count"),
            last_n5_order_number=("order_number", "max"),
        )
        .reset_index()
        .merge(max_order.reset_index(), on="user_id", how="left")
    )
    n5["up_orders_since_last_n5"] = (n5["max_order_number"] - n5["last_n5_order_number"]).clip(upper=5)
    pair = pair.merge(
        n5[["user_id", "product_id", "up_purchase_count_n5", "up_orders_since_last_n5"]],
        on=["user_id", "product_id"],
        how="left",
    )
    pair["up_purchase_count_n5"] = pair["up_purchase_count_n5"].fillna(0).astype(int)
    pair["up_orders_since_last_n5"] = pair["up_orders_since_last_n5"].fillna(5).astype(int)

    product_info = pd.read_csv(data / "products.csv", usecols=["product_id", "aisle_id", "department_id"])
    target_timing = target_orders[["user_id", "days_since_prior_order"]].rename(
        columns={"days_since_prior_order": "target_days_since_prior"}
    )

    metadata = (
        keys[["user_id", "product_id"]]
        .merge(product_info, on="product_id", how="left", validate="many_to_one")
        .merge(product_stats, on="product_id", how="left", validate="many_to_one")
        .merge(user_features.drop(columns=["max_order_number"]), on="user_id", how="left", validate="many_to_one")
        .merge(target_timing, on="user_id", how="left", validate="many_to_one")
        .merge(
            pair.drop(columns=["order_numbers", "max_order_number"]),
            on=["user_id", "product_id"],
            how="left",
            validate="one_to_one",
        )
    )

    required = ["aisle_id", "up_purchase_count", "user_total_orders", "up_orders_since_last"]
    missing = metadata[required].isna().sum()
    if missing.any():
        raise ValueError(f"Missing reconstructed metadata: {missing[missing > 0].to_dict()}")
    return metadata


def add_subgroups(master: pd.DataFrame) -> pd.DataFrame:
    tagged = get_user_subgroups(master)
    subgroup_cols = [
        "is_cold_start_user",
        "is_none_user",
        "is_high_variance_aisle",
        "is_single_purchase_pair",
        "primary_user_subgroup",
    ]
    if tagged[subgroup_cols].isna().any().any():
        raise ValueError("Subgroup tagging produced NaN values.")
    return tagged


def per_user_basket_metrics(master: pd.DataFrame, model: str) -> pd.DataFrame:
    pred_col = f"{model}_pred"
    grouped = (
        master.groupby("user_id")
        .agg(
            true_basket_size=("label", "sum"),
            pred_basket_size=(pred_col, "sum"),
            tp=(pred_col, lambda x: int(((x == 1) & (master.loc[x.index, "label"] == 1)).sum())),
            fp=(pred_col, lambda x: int(((x == 1) & (master.loc[x.index, "label"] == 0)).sum())),
            fn=(pred_col, lambda x: int(((x == 0) & (master.loc[x.index, "label"] == 1)).sum())),
            is_cold_start_user=("is_cold_start_user", "max"),
            is_none_user=("is_none_user", "max"),
            primary_user_subgroup=("primary_user_subgroup", "first"),
        )
        .reset_index()
    )
    denom_precision = grouped["tp"] + grouped["fp"]
    denom_recall = grouped["tp"] + grouped["fn"]
    grouped["precision"] = np.where(denom_precision > 0, grouped["tp"] / denom_precision, 0.0)
    grouped["recall"] = np.where(denom_recall > 0, grouped["tp"] / denom_recall, 0.0)
    denom_f1 = grouped["precision"] + grouped["recall"]
    grouped["per_order_f1"] = np.where(
        denom_f1 > 0,
        2 * grouped["precision"] * grouped["recall"] / denom_f1,
        0.0,
    )
    grouped["basket_size_error"] = grouped["pred_basket_size"] - grouped["true_basket_size"]
    grouped["basket_size_abs_error"] = grouped["basket_size_error"].abs()
    grouped["basket_size_inflation"] = grouped["basket_size_error"] > 0
    grouped["basket_size_inflation_large"] = grouped["basket_size_error"] >= np.maximum(
        3, np.ceil(0.5 * grouped["true_basket_size"])
    )
    grouped["none_collapse_case"] = (grouped["true_basket_size"] == 0) & (grouped["pred_basket_size"] == 0)
    grouped.insert(1, "model", model)
    grouped.insert(2, "model_label", MODEL_LABELS.get(model, model))
    return grouped


def build_basket_metrics(master: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    return pd.concat([per_user_basket_metrics(master, model) for model in models], ignore_index=True)


def build_metrics_summary(
    root: Path,
    master: pd.DataFrame,
    basket_metrics: pd.DataFrame,
    models: list[str],
    tag: str,
) -> pd.DataFrame:
    rows = []
    y = master["label"].to_numpy(dtype=int)
    for model in models:
        pred = master[f"{model}_pred"].to_numpy(dtype=int)
        prob = master[f"{model}_prob"].to_numpy(dtype=float)
        bm = basket_metrics[basket_metrics["model"] == model]
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "tag": tag,
                "threshold": load_metric_threshold(root, model, tag),
                "n_rows": int(len(master)),
                "n_users": int(master["user_id"].nunique()),
                "positive_rate": float(y.mean()),
                "accuracy": accuracy_score(y, pred),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "per_order_f1": float(bm["per_order_f1"].mean()),
                "roc_auc": roc_auc_score(y, prob),
                "pr_auc": average_precision_score(y, prob),
                "avg_true_basket_size": float(bm["true_basket_size"].mean()),
                "avg_pred_basket_size": float(bm["pred_basket_size"].mean()),
                "basket_size_mae": float(bm["basket_size_abs_error"].mean()),
                "basket_inflation_rate": float(bm["basket_size_inflation"].mean()),
                "large_basket_inflation_rate": float(bm["basket_size_inflation_large"].mean()),
                "none_collapse_rate": float(bm["none_collapse_case"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("f1", ascending=False)


def build_model_disagreement(master: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    pred_cols = [f"{m}_pred" for m in models]
    prob_cols = [f"{m}_prob" for m in models]
    out = master[["user_id", "product_id", "label"] + pred_cols + prob_cols].copy()
    out["n_models_pred_positive"] = out[pred_cols].sum(axis=1)
    out["n_models_pred_negative"] = len(models) - out["n_models_pred_positive"]
    out["prob_range"] = out[prob_cols].max(axis=1) - out[prob_cols].min(axis=1)
    out["models_pred_positive"] = [
        ";".join([model for model in models if row[f"{model}_pred"] == 1])
        for _, row in out.iterrows()
    ]
    out["models_pred_negative"] = [
        ";".join([model for model in models if row[f"{model}_pred"] == 0])
        for _, row in out.iterrows()
    ]
    out["disagreement_type"] = np.select(
        [
            out["n_models_pred_positive"] == 0,
            out["n_models_pred_positive"] == len(models),
        ],
        ["unanimous_negative", "unanimous_positive"],
        default="split_vote",
    )
    return out[out["disagreement_type"] == "split_vote"].sort_values(
        ["n_models_pred_positive", "prob_range"], ascending=[False, False]
    )


def build_behavioral_error_tags(master: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    basket_lookup = {
        model: per_user_basket_metrics(master, model).set_index("user_id")[
            ["basket_size_inflation_large", "none_collapse_case"]
        ]
        for model in models
    }

    for model in models:
        df = master[
            [
                "user_id",
                "product_id",
                "label",
                "aisle_id",
                "department_id",
                "up_purchase_count",
                "up_orders_since_last",
                "up_streak",
                "up_purchase_count_n5",
                "up_orders_since_last_n5",
                "prod_reorder_ratio",
                "user_avg_basket_size",
                "target_days_since_prior",
                f"{model}_prob",
                f"{model}_pred",
                f"{model}_error_type",
            ]
        ].copy()
        df.insert(0, "model", model)
        df.insert(1, "model_label", MODEL_LABELS.get(model, model))
        df = df.rename(
            columns={
                f"{model}_prob": "prob",
                f"{model}_pred": "pred",
                f"{model}_error_type": "error_type",
            }
        )
        df = df.join(basket_lookup[model], on="user_id")

        fp = df["error_type"] == "FP"
        fn = df["error_type"] == "FN"
        df["fp_stockpile_error"] = fp & (df["up_purchase_count"] >= 3) & (df["up_orders_since_last"] <= 1) & (
            df["target_days_since_prior"].fillna(999) <= 7
        )
        df["fp_broken_streak_error"] = fp & (df["up_streak"] >= 2)
        df["fp_recipe_novelty_error"] = fp & (df["up_purchase_count"] == 1) & (
            df["prod_reorder_ratio"].fillna(0) < 0.45
        )
        df["fn_sleeper_staple_error"] = fn & (df["up_purchase_count"] >= 3) & (df["up_orders_since_last"] >= 3) & (
            df["prod_reorder_ratio"].fillna(0) >= 0.45
        )
        df["fn_new_habit_error"] = fn & (df["up_purchase_count_n5"] >= 2) & (df["up_orders_since_last_n5"] <= 1)
        df["basket_size_inflation_error"] = df["basket_size_inflation_large"].fillna(False)
        df["none_collapse_case"] = df["none_collapse_case"].fillna(False)

        tag_cols = [
            "fp_stockpile_error",
            "fp_broken_streak_error",
            "fp_recipe_novelty_error",
            "fn_sleeper_staple_error",
            "fn_new_habit_error",
            "basket_size_inflation_error",
            "none_collapse_case",
        ]
        df["behavioral_error_tag"] = "untagged"
        for col in tag_cols:
            df.loc[df[col] & (df["behavioral_error_tag"] == "untagged"), "behavioral_error_tag"] = col
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def print_effect_sizes(metrics: pd.DataFrame) -> None:
    rf = metrics[metrics["model"] == "rf"].iloc[0]
    print("\nRaw effect sizes for narrative checks:")
    for _, row in metrics.iterrows():
        if row["model"] == "rf":
            continue
        print(
            f"  rf - {row['model']}: "
            f"row_f1={rf['f1'] - row['f1']:+.6f}, "
            f"per_order_f1={rf['per_order_f1'] - row['per_order_f1']:+.6f}, "
            f"roc_auc={rf['roc_auc'] - row['roc_auc']:+.6f}, "
            f"pr_auc={rf['pr_auc'] - row['pr_auc']:+.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--tag", default="final")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--chunksize", type=int, default=2_000_000)
    args = parser.parse_args()

    preds = load_predictions(args.root, args.models, args.tag)
    keys = next(iter(preds.values()))[["user_id", "product_id", "label"]].copy()
    metadata = reconstruct_metadata_from_raw(args.root, keys, args.chunksize)
    master = build_master_predictions(preds, metadata)
    master = add_subgroups(master)

    out_dir = args.root / "results"
    basket_metrics = build_basket_metrics(master, args.models)
    metrics = build_metrics_summary(args.root, master, basket_metrics, args.models, args.tag)
    disagreement = build_model_disagreement(master, args.models)
    behavioral_tags = build_behavioral_error_tags(master, args.models)

    paths = {
        "test_predictions": out_dir / "test_predictions.parquet",
        "metrics_summary": out_dir / "metrics_summary.csv",
        "model_disagreement": out_dir / "model_disagreement.csv",
        "basket_metrics": out_dir / "basket_metrics.csv",
        "behavioral_error_tags": out_dir / "behavioral_error_tags.csv",
    }
    master.to_parquet(paths["test_predictions"], index=False)
    metrics.to_csv(paths["metrics_summary"], index=False)
    disagreement.to_csv(paths["model_disagreement"], index=False)
    basket_metrics.to_csv(paths["basket_metrics"], index=False)
    behavioral_tags.to_csv(paths["behavioral_error_tags"], index=False)

    print("\nMetrics summary:")
    cols = ["model_label", "f1", "per_order_f1", "roc_auc", "pr_auc", "avg_pred_basket_size"]
    print(metrics[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print_effect_sizes(metrics)
    print("\nSaved:")
    for path in paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
