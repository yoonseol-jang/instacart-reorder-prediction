"""
Reusable subgroup tagging for Phase 5 error analysis.

The function is intentionally dataframe-first: pass the evaluation dataframe
and it returns the same rows with stable subgroup columns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEFINITIONS_PATH = PROJECT_ROOT / "results" / "subgroup_definitions.json"


def load_subgroup_definitions(path: str | Path | None = None) -> dict:
    """Load EDA-exported subgroup definitions, returning empty sets if absent."""
    path = Path(path) if path is not None else DEFAULT_DEFINITIONS_PATH
    if not path.exists():
        return {
            "cold_start_users": set(),
            "high_variance_aisle_ids": set(),
        }

    raw = json.loads(path.read_text())
    return {
        "cold_start_users": set(raw.get("cold_start_users", [])),
        "high_variance_aisle_ids": set(raw.get("high_variance_aisle_ids", [])),
    }


def get_user_subgroups(
    df: pd.DataFrame,
    definitions_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Add Phase 1 subgroup tags to an evaluation dataframe.

    Required columns:
        user_id
        label             -> derives users with no actual reorders in this df
        aisle_id          -> flags high-variance aisle rows
        up_purchase_count -> flags single-purchase user-product rows

    Optional column:
        user_total_orders -> derives cold-start users as <5 prior orders

    Returns a copy with no-null subgroup columns:
        is_cold_start_user, is_none_user, is_high_variance_aisle,
        is_single_purchase_pair, primary_user_subgroup
    """
    required_cols = ["user_id", "label", "aisle_id", "up_purchase_count"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"get_user_subgroups is missing required columns: {missing}")

    out = df.copy()
    defs = load_subgroup_definitions(definitions_path)

    if "user_total_orders" in out.columns:
        out["is_cold_start_user"] = out["user_total_orders"] < 5
    else:
        out["is_cold_start_user"] = out["user_id"].isin(defs["cold_start_users"])

    user_positive_counts = out.groupby("user_id")["label"].transform("sum")
    out["is_none_user"] = user_positive_counts == 0

    out["is_high_variance_aisle"] = out["aisle_id"].isin(
        defs["high_variance_aisle_ids"]
    )
    out["is_single_purchase_pair"] = out["up_purchase_count"] == 1

    cold = out["is_cold_start_user"].to_numpy(dtype=bool)
    none = out["is_none_user"].to_numpy(dtype=bool)
    out["primary_user_subgroup"] = np.select(
        [
            cold & none,
            cold,
            none,
        ],
        [
            "cold_start_none_user",
            "cold_start_user",
            "none_user",
        ],
        default="regular_user",
    )

    subgroup_cols = [
        "is_cold_start_user",
        "is_none_user",
        "is_high_variance_aisle",
        "is_single_purchase_pair",
        "primary_user_subgroup",
    ]
    if out[subgroup_cols].isna().any().any():
        raise ValueError("Subgroup tagging produced NaN values.")

    return out
