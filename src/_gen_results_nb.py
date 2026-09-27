#!/usr/bin/env python3
"""Generate src/results.ipynb — paper-ready results notebook."""
import json
from pathlib import Path

NB = Path(__file__).parent / "results.ipynb"

def code(src):
    lines = src.lstrip("\n").splitlines()
    source = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] != "" else [])
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

def md(src):
    lines = src.lstrip("\n").splitlines()
    source = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] != "" else [])
    return {"cell_type": "markdown", "metadata": {}, "source": source}


# ── Cell sources (keep each one self-contained) ──────────────────────────────

TITLE = """\
# Grocery Reorder Prediction — Results

> **Smoke-test outputs** (80 K train / 20 K test rows, single seed, no hyperparameter tuning).
> To load full-experiment results, change `TAG = "smoke"` → `"final"` in the Setup cell.

**Models compared:** Frequency Baseline · Logistic Regression · Random Forest · AdaBoost · XGBoost
**Task:** Binary classification — did the user reorder this product in their most recent order?
**Positive-class base rate:** ≈ 9.8 %
"""

SETUP = """\
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from IPython.display import display
from subgroups import get_user_subgroups
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)

TAG   = "smoke"          # ← change to "final" for full-run results
ROOT  = Path("..").resolve()
DATA  = ROOT / "data"
RES   = ROOT / "results"
TBL   = RES  / "tables"
PLOTS = ROOT / "plots"
TBL.mkdir(parents=True, exist_ok=True)

MODELS = ["baseline", "logreg", "rf", "adaboost", "xgboost"]
LABELS = {
    "baseline":  "Freq. Baseline",
    "logreg":    "Logistic Regression",
    "rf":        "Random Forest",
    "adaboost":  "AdaBoost",
    "xgboost":   "XGBoost",
}
COLORS = {
    "baseline":  "#999999",
    "logreg":    "#4e79a7",
    "rf":        "#f28e2b",
    "adaboost":  "#e15759",
    "xgboost":   "#76b7b2",
}
MIN_POS = 3   # minimum actual reorders for a product to appear in FN tables
print("Setup complete. TAG =", TAG)
"""

LOAD_META = """\
products    = pd.read_csv(DATA / "products.csv")
aisles      = pd.read_csv(DATA / "aisles.csv")
departments = pd.read_csv(DATA / "departments.csv")
prod_meta = (products
    .merge(aisles,      on="aisle_id",      how="left")
    .merge(departments, on="department_id", how="left")
    [["product_id", "product_name", "aisle_id", "aisle", "department_id", "department"]])
print(f"Products: {len(products):,}  |  Aisles: {len(aisles)}  |  Departments: {len(departments)}")
"""

LOAD_PREDS = """\
feature_context = pd.read_parquet(
    DATA / "processed" / "test.parquet",
    columns=[
        "user_id",
        "product_id",
        "user_total_orders",
        "aisle_id",
        "up_purchase_count",
    ],
)
preds_store = {}
for m in MODELS:
    df = pd.read_csv(RES / m / f"{TAG}_predictions.csv")
    df = df.merge(
        feature_context,
        on=["user_id", "product_id"],
        how="left",
        validate="many_to_one",
    )
    df["error_type"] = "TN"
    df.loc[(df.label == 1) & (df.pred == 1), "error_type"] = "TP"
    df.loc[(df.label == 1) & (df.pred == 0), "error_type"] = "FN"
    df.loc[(df.label == 0) & (df.pred == 1), "error_type"] = "FP"
    preds_store[m] = df

rows = []
for m in MODELS:
    d  = preds_store[m]
    et = d["error_type"].value_counts()
    rows.append({
        "Model":             LABELS[m],
        "Total rows":        len(d),
        "Actual reorders":   int(d.label.sum()),
        "TP": int(et.get("TP", 0)),
        "FP": int(et.get("FP", 0)),
        "FN": int(et.get("FN", 0)),
        "TN": int(et.get("TN", 0)),
    })
pd.DataFrame(rows).set_index("Model")
"""

H1 = "## Table 1 — Model Performance"

TABLE1 = """\
perf_rows = []
for m in MODELS:
    d   = json.loads((RES / m / f"{TAG}_metrics.json").read_text())
    met = d["metrics"]
    cfg = d["config"]
    perf_rows.append({
        "Model":      LABELS[m],
        "Threshold":  met["threshold"],
        "Accuracy":   met["accuracy"],
        "Precision":  met["precision"],
        "Recall":     met["recall"],
        "F1":         met["f1"],
        "ROC-AUC":    met["roc_auc"],
        "PR-AUC":     met["pr_auc"],
        "N train":    cfg.get("n_train", ""),
        "N test":     cfg.get("n_test", ""),
    })

t1 = pd.DataFrame(perf_rows).set_index("Model")
t1.to_csv(TBL / f"t1_performance_{TAG}.csv")

num_cols = ["Threshold","Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"]
rank_cols = ["Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"]
display(
    t1.style
      .format({c: "{:.4f}" for c in num_cols})
      .highlight_max(subset=rank_cols, color="#d4edda")
      .highlight_min(subset=rank_cols, color="#f8d7da")
      .set_caption(f"Table 1 — Test-set performance ({TAG}). Green = best, red = worst per column.")
)
print("Saved:", TBL / f"t1_performance_{TAG}.csv")
"""

H2 = "## Table 2 — Error Type Counts"

TABLE2 = """\
ec_rows = []
for m in MODELS:
    d  = preds_store[m]
    et = d["error_type"].value_counts()
    n_pos = int(d.label.sum())
    n_neg = int((d.label == 0).sum())
    tp = int(et.get("TP", 0))
    fn = int(et.get("FN", 0))
    fp = int(et.get("FP", 0))
    tn = int(et.get("TN", 0))
    ec_rows.append({
        "Model":                   LABELS[m],
        "Actual reorders":         n_pos,
        "Caught (TP)":             tp,
        "Missed (FN)":             fn,
        "Miss rate":               fn / n_pos if n_pos else 0,
        "False alarms (FP)":       fp,
        "FA rate":                 fp / n_neg if n_neg else 0,
        "Correct non-reorder (TN)": tn,
        "Total rows":              len(d),
    })

t2 = pd.DataFrame(ec_rows).set_index("Model")
t2.to_csv(TBL / f"t2_error_counts_{TAG}.csv")

display(
    t2.style
      .format({"Miss rate": "{:.3f}", "FA rate": "{:.3f}"})
      .background_gradient(subset=["Miss rate"], cmap="Reds")
      .background_gradient(subset=["FA rate"],   cmap="Oranges")
      .set_caption(f"Table 2 — Error type counts ({TAG}). "
                   "Miss rate = FN / actual reorders; FA rate = FP / actual non-reorders.")
)
print("Saved:", TBL / f"t2_error_counts_{TAG}.csv")
"""

H3 = """\
## Table 3 — False Negative Analysis (Missed Reorders)

A **false negative** is a product the user *did* reorder but the model failed to predict.
For a recommendation system this is the primary error: missed opportunities to surface relevant items.
"""

TABLE3_PER = """\
def fn_table(m, sort_by="missed", top_n=10):
    df   = preds_store[m]
    pos  = df[df.label == 1].copy()
    grp  = (pos.groupby("product_id")
               .agg(actual_reorders=("label", "count"),
                    missed=("error_type", lambda x: (x == "FN").sum()))
               .reset_index())
    grp  = grp[grp.actual_reorders >= MIN_POS].copy()
    grp["fn_rate"] = grp.missed / grp.actual_reorders
    grp  = grp.merge(prod_meta[["product_id","product_name","aisle","department"]],
                     on="product_id", how="left")
    grp  = grp.sort_values(sort_by, ascending=False).head(top_n).reset_index(drop=True)
    grp.index += 1
    return grp[["product_name", "department", "aisle",
                "actual_reorders", "missed", "fn_rate"]]

for m in MODELS:
    tbl = fn_table(m, sort_by="missed")
    print(f"\\n{'─'*60}")
    print(f"  {LABELS[m]} — top 10 most-missed products (min {MIN_POS} actual reorders)")
    print(f"{'─'*60}")
    display(
        tbl.style
           .format({"fn_rate": "{:.3f}"})
           .bar(subset=["fn_rate"], color="#f4a582", vmin=0, vmax=1)
           .set_caption(f"{LABELS[m]}")
    )
"""

TABLE3_COMBINED = """\
fn_parts = []
for m in MODELS:
    df  = preds_store[m]
    pos = df[df.label == 1].copy()
    grp = (pos.groupby("product_id")
              .agg(actual_reorders=("label", "count"),
                   missed=("error_type", lambda x: (x == "FN").sum()))
              .reset_index())
    grp = grp[grp.actual_reorders >= MIN_POS].copy()
    grp["fn_rate"] = grp.missed / grp.actual_reorders
    grp["model"]   = m
    fn_parts.append(grp)

all_fn = pd.concat(fn_parts, ignore_index=True)

# actual_reorders is the same across models (same test set)
actual_lookup = (all_fn.groupby("product_id")["actual_reorders"]
                        .first().reset_index())
# total missed across all models — primary sort key
total_missed = (all_fn.groupby("product_id")["missed"]
                       .sum().reset_index()
                       .rename(columns={"missed": "total_missed"}))
# per-model FN rate pivot
fn_rate_pivot = (all_fn
    .pivot_table(index="product_id", columns="model", values="fn_rate", aggfunc="first")
    .reset_index())
fn_rate_pivot.columns.name = None

combined = (actual_lookup
    .merge(total_missed,   on="product_id")
    .merge(fn_rate_pivot,  on="product_id")
    .merge(prod_meta[["product_id","product_name","aisle","department"]],
           on="product_id", how="left"))

model_cols = [c for c in MODELS if c in combined.columns]
combined["mean_fn_rate"] = combined[model_cols].mean(axis=1, skipna=True)
combined = (combined.sort_values("total_missed", ascending=False)
                    .head(15).reset_index(drop=True))
combined.index += 1

t3 = combined[["product_name","department","aisle",
               "actual_reorders","total_missed"] + model_cols + ["mean_fn_rate"]].rename(
    columns={m: LABELS[m] for m in model_cols})
t3.to_csv(TBL / f"t3_top_fn_products_{TAG}.csv")

label_cols = [LABELS[m] for m in model_cols]
fmt3 = {c: "{:.3f}" for c in label_cols + ["mean_fn_rate"]}
display(
    t3.style
      .format(fmt3, na_rep="—")
      .background_gradient(subset=["total_missed"], cmap="Reds")
      .set_caption(f"Table 3 — Top 15 most-missed products by total missed count across models ({TAG}). "
                   "actual_reorders = test-set positives; total_missed = sum of FN counts across all models; "
                   "FN rate = missed / actual reorders.")
)
print("Saved:", TBL / f"t3_top_fn_products_{TAG}.csv")
"""

H4 = """\
## Table 4 — False Positive Analysis (False Alarms)

A **false positive** is a product the model predicted as a reorder but the user did not actually reorder.
"""

TABLE4_PER = """\
def fp_table(m, top_n=10):
    df  = preds_store[m]
    grp = (df.groupby("product_id")
             .agg(total=("label", "count"),
                  actual_reorders=("label", "sum"),
                  fp_count=("error_type", lambda x: (x == "FP").sum()))
             .reset_index())
    grp["non_reorders"] = grp.total - grp.actual_reorders
    grp = grp[grp.non_reorders >= MIN_POS].copy()
    grp["fp_rate"] = grp.fp_count / grp.non_reorders
    grp = grp.merge(prod_meta[["product_id","product_name","aisle","department"]],
                    on="product_id", how="left")
    grp = grp.sort_values("fp_count", ascending=False).head(top_n).reset_index(drop=True)
    grp.index += 1
    return grp[["product_name", "department", "aisle",
                "non_reorders", "fp_count", "fp_rate"]]

for m in MODELS:
    tbl = fp_table(m)
    print(f"\\n{'─'*60}")
    print(f"  {LABELS[m]} — top 10 most false-alarmed products")
    print(f"{'─'*60}")
    display(
        tbl.style
           .format({"fp_rate": "{:.3f}"})
           .bar(subset=["fp_rate"], color="#92c5de", vmin=0, vmax=1)
           .set_caption(f"{LABELS[m]}")
    )
"""

TABLE4_COMBINED = """\
fp_parts = []
for m in MODELS:
    df  = preds_store[m]
    grp = (df.groupby("product_id")
             .agg(total=("label", "count"),
                  actual_reorders=("label", "sum"),
                  fp_count=("error_type", lambda x: (x == "FP").sum()))
             .reset_index())
    grp["non_reorders"] = grp.total - grp.actual_reorders
    grp = grp[grp.non_reorders >= MIN_POS].copy()
    grp["fp_rate"] = grp.fp_count / grp.non_reorders
    grp["model"]   = m
    fp_parts.append(grp)

all_fp   = pd.concat(fp_parts, ignore_index=True)
fp_pivot = (all_fp
    .pivot_table(index="product_id", columns="model", values="fp_rate", aggfunc="first")
    .reset_index())
fp_pivot.columns.name = None
fp_pivot = fp_pivot.merge(prod_meta[["product_id","product_name","aisle","department"]],
                          on="product_id", how="left")
fp_model_cols = [c for c in MODELS if c in fp_pivot.columns]
fp_pivot["mean_fp_rate"] = fp_pivot[fp_model_cols].mean(axis=1, skipna=True)
fp_pivot = (fp_pivot.sort_values("mean_fp_rate", ascending=False)
                    .head(15).reset_index(drop=True))
fp_pivot.index += 1

t4 = fp_pivot[["product_name","department","aisle"] + fp_model_cols + ["mean_fp_rate"]].rename(
    columns={m: LABELS[m] for m in fp_model_cols})
t4.to_csv(TBL / f"t4_top_fp_products_{TAG}.csv")

fp_label_cols = [LABELS[m] for m in fp_model_cols]
fmt4 = {c: "{:.3f}" for c in fp_label_cols + ["mean_fp_rate"]}
display(
    t4.style
      .format(fmt4, na_rep="—")
      .background_gradient(subset=["mean_fp_rate"], cmap="Blues")
      .set_caption(f"Table 4 — Top 15 products by mean FP rate across models ({TAG}).")
)
print("Saved:", TBL / f"t4_top_fp_products_{TAG}.csv")
"""

H5 = "## Table 5 — FN Rates by Aisle and Department"

TABLE5A = """\
aisle_rows = []
for aid in sorted(prod_meta["aisle_id"].unique()):
    prods = prod_meta[prod_meta.aisle_id == aid]["product_id"].values
    aisle_name = prod_meta[prod_meta.aisle_id == aid]["aisle"].iloc[0]
    row = {"aisle_id": int(aid), "aisle": aisle_name}
    fn_vals = []
    for m in MODELS:
        df  = preds_store[m]
        sub = df[df.product_id.isin(prods) & (df.label == 1)]
        if len(sub) == 0:
            row[m] = float("nan")
        else:
            v = float((sub.error_type == "FN").sum() / len(sub))
            row[m] = v
            fn_vals.append(v)
    row["mean_fn_rate"] = float(np.mean(fn_vals)) if fn_vals else float("nan")
    aisle_rows.append(row)

t5a = (pd.DataFrame(aisle_rows)
         .dropna(subset=["mean_fn_rate"])
         .sort_values("mean_fn_rate", ascending=False)
         .head(20)
         .reset_index(drop=True))
t5a.index += 1
t5a = t5a.rename(columns={m: LABELS[m] for m in MODELS if m in t5a.columns})
t5a.to_csv(TBL / f"t5a_aisle_fn_rates_{TAG}.csv", index=False)

lc = [LABELS[m] for m in MODELS if LABELS[m] in t5a.columns]
display(
    t5a[["aisle"] + lc + ["mean_fn_rate"]].style
      .format({c: "{:.3f}" for c in lc + ["mean_fn_rate"]}, na_rep="—")
      .background_gradient(subset=["mean_fn_rate"], cmap="Reds")
      .set_caption(f"Table 5a — Top 20 aisles by mean FN rate ({TAG}).")
)
print("Saved:", TBL / f"t5a_aisle_fn_rates_{TAG}.csv")
"""

TABLE5B = """\
dept_rows = []
for did in sorted(prod_meta["department_id"].unique()):
    prods = prod_meta[prod_meta.department_id == did]["product_id"].values
    dept_name = prod_meta[prod_meta.department_id == did]["department"].iloc[0]
    row = {"dept_id": int(did), "department": dept_name}
    fn_vals = []
    for m in MODELS:
        df  = preds_store[m]
        sub = df[df.product_id.isin(prods) & (df.label == 1)]
        if len(sub) == 0:
            row[m] = float("nan")
        else:
            v = float((sub.error_type == "FN").sum() / len(sub))
            row[m] = v
            fn_vals.append(v)
    row["mean_fn_rate"] = float(np.mean(fn_vals)) if fn_vals else float("nan")
    dept_rows.append(row)

t5b = (pd.DataFrame(dept_rows)
         .dropna(subset=["mean_fn_rate"])
         .sort_values("mean_fn_rate", ascending=False)
         .reset_index(drop=True))
t5b.index += 1
t5b = t5b.rename(columns={m: LABELS[m] for m in MODELS if m in t5b.columns})
t5b.to_csv(TBL / f"t5b_dept_fn_rates_{TAG}.csv", index=False)

lc5b = [LABELS[m] for m in MODELS if LABELS[m] in t5b.columns]
display(
    t5b[["department"] + lc5b + ["mean_fn_rate"]].style
      .format({c: "{:.3f}" for c in lc5b + ["mean_fn_rate"]}, na_rep="—")
      .background_gradient(subset=["mean_fn_rate"], cmap="Reds")
      .set_caption(f"Table 5b — All departments, FN rate by model ({TAG}).")
)
print("Saved:", TBL / f"t5b_dept_fn_rates_{TAG}.csv")
"""

H6 = """\
## Table 6 — Subgroup Analysis

Subgroups from `results/subgroup_definitions.json`:
- **Cold-start users** — users with few prior orders (limited purchase history)
- **None-users** — users who reordered nothing in prior training orders
- **High-variance aisles** — aisles with inconsistent reorder behavior across users
"""

TABLE6 = """\
sg_rows = []
for m in MODELS:
    df = preds_store[m]
    df = get_user_subgroups(df, RES / "subgroup_definitions.json")
    assert not df[[
        "is_cold_start_user",
        "is_none_user",
        "is_high_variance_aisle",
        "primary_user_subgroup",
    ]].isna().any().any()
    subgroups = [
        ("All",                pd.Series([True]  * len(df), index=df.index)),
        ("Cold-start users",   df["is_cold_start_user"]),
        ("None-users",         df["is_none_user"]),
        ("High-var. aisles",   df["is_high_variance_aisle"]),
    ]
    for sg_name, mask in subgroups:
        sub   = df[mask]
        n_pos = int(sub.label.sum())
        n_neg = int((sub.label == 0).sum())
        fn    = int((sub.error_type == "FN").sum())
        fp    = int((sub.error_type == "FP").sum())
        tot   = len(sub)
        sg_rows.append({
            "Model":             LABELS[m],
            "Subgroup":          sg_name,
            "Rows":              tot,
            "Actual reorders":   n_pos,
            "FN (missed)":       fn,
            "FN rate":           fn / n_pos if n_pos else float("nan"),
            "FP (false alarm)":  fp,
            "Error rate":        (fn + fp) / tot if tot else float("nan"),
        })

t6 = pd.DataFrame(sg_rows).set_index(["Model", "Subgroup"])
t6.to_csv(TBL / f"t6_subgroup_errors_{TAG}.csv")
display(
    t6.style
      .format({"FN rate": "{:.3f}", "Error rate": "{:.3f}"}, na_rep="—")
      .background_gradient(subset=["FN rate", "Error rate"], cmap="Reds")
      .set_caption(f"Table 6 — Subgroup error analysis ({TAG}).")
)
print("Saved:", TBL / f"t6_subgroup_errors_{TAG}.csv")
"""

H7 = "## Figures"

FIG1 = """\
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for m in MODELS:
    df    = preds_store[m]
    y_t   = df["label"].values
    y_p   = df["prob"].values
    lbl   = LABELS[m]
    color = COLORS[m]

    fpr, tpr, _ = roc_curve(y_t, y_p)
    axes[0].plot(fpr, tpr, color=color, lw=2,
                 label=f"{lbl} ({auc(fpr,tpr):.3f})")

    prec, rec, _ = precision_recall_curve(y_t, y_p)
    axes[1].plot(rec, prec, color=color, lw=2,
                 label=f"{lbl} ({auc(rec,prec):.3f})")

axes[0].plot([0,1],[0,1],"k--",lw=0.8,alpha=0.5)
axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
axes[0].legend(fontsize=8, loc="lower right")

axes[1].axhline(0.062, color="k", linestyle="--", lw=0.8, alpha=0.5, label="Baseline (random)")
axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
axes[1].legend(fontsize=8, loc="upper right")

for ax in axes:
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

fig.suptitle(f"Figure 1 — ROC & Precision-Recall ({TAG.upper()} run, AUC in legend)",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(PLOTS / f"paper_roc_pr_{TAG}.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", PLOTS / f"paper_roc_pr_{TAG}.png")
"""

FIG2 = """\
TOP_N = 10

fn_all_parts = []
for m in MODELS:
    df  = preds_store[m]
    pos = df[df.label == 1]
    grp = (pos.groupby("product_id")
              .agg(actual=("label","count"),
                   missed=("error_type", lambda x: (x=="FN").sum()))
              .reset_index())
    grp = grp[grp.actual >= MIN_POS].copy()
    grp["model"] = m
    fn_all_parts.append(grp)

fn_all_df = pd.concat(fn_all_parts, ignore_index=True)
top_pids   = (fn_all_df.groupby("product_id")["missed"]
                        .sum().nlargest(TOP_N).index.tolist())

fn_top  = fn_all_df[fn_all_df.product_id.isin(top_pids)].copy()
fn_top  = fn_top.merge(prod_meta[["product_id","product_name"]], on="product_id", how="left")
fn_top["short"] = fn_top.product_name.str[:30]

pivot = fn_top.pivot_table(index="short", columns="model",
                            values="missed", aggfunc="first", fill_value=0)
pivot = pivot.reindex(columns=MODELS, fill_value=0)
pivot = pivot.sort_values(MODELS[-1], ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
left = np.zeros(len(pivot))
for m in MODELS:
    ax.barh(range(len(pivot)), pivot[m].values.astype(float),
            left=left, color=COLORS[m], label=LABELS[m])
    left += pivot[m].values.astype(float)

ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_xlabel("Missed reorders (FN count)")
ax.set_title(f"Figure 2 — Top {TOP_N} Most-Missed Products by Model ({TAG.upper()} run)",
             fontsize=11, fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(PLOTS / f"paper_top_fn_products_{TAG}.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved:", PLOTS / f"paper_top_fn_products_{TAG}.png")
"""

FIG3 = """\
if sg_rows:
    t6_df = pd.DataFrame(sg_rows)
    sg_order = ["All", "Cold-start users", "None-users", "High-var. aisles"]
    sg_avail  = [s for s in sg_order if s in t6_df["Subgroup"].values]

    x     = np.arange(len(sg_avail))
    width = 0.15
    n_m   = len(MODELS)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(MODELS):
        vals = []
        for sg in sg_avail:
            rows = t6_df[(t6_df.Model == LABELS[m]) & (t6_df.Subgroup == sg)]
            vals.append(float(rows["FN rate"].values[0])
                        if len(rows) and not np.isnan(rows["FN rate"].values[0])
                        else 0.0)
        offset = (i - n_m / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=LABELS[m], color=COLORS[m])

    ax.set_xticks(x)
    ax.set_xticklabels(sg_avail, fontsize=10)
    ax.set_ylabel("FN rate (missed / actual reorders)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Figure 3 — FN Rate by Subgroup ({TAG.upper()} run)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / f"paper_subgroup_fn_rates_{TAG}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved:", PLOTS / f"paper_subgroup_fn_rates_{TAG}.png")
else:
    print("No subgroup rows — Figure 3 skipped.")
"""

RUN_CFG = """\
print(f"{'='*65}")
print(f"  Run configurations ({TAG})")
print(f"{'='*65}")
for m in MODELS:
    d   = json.loads((RES / m / f"{TAG}_metrics.json").read_text())
    cfg = d["config"]
    print(f"\\n{LABELS[m]}")
    print(f"  n_train={cfg.get('n_train','?'):,}  n_test={cfg.get('n_test','?'):,}"
          f"  elapsed={cfg.get('elapsed_s','?')}s")
    bp = cfg.get("best_params") or cfg.get("score_col", "")
    if bp:
        print(f"  params: {bp}")
"""

H8 = """\
## Cross-model Feature Importance Analysis (plan §6.2.4)

Load per-model importances from saved pkl files, rank features, compute
top-10 pairwise overlap matrix, and identify disagreement features.
"""

FEAT_IMP_XMODEL = """\
import pickle
from sklearn.metrics import f1_score as _f1

# ── Load importances ─────────────────────────────────────────────────────────
imp_store = {}
FEAT_COLS_PATH = ROOT / "data" / "processed" / "feature_cols.txt"
feat_cols = FEAT_COLS_PATH.read_text().splitlines() if FEAT_COLS_PATH.exists() else []

for m in MODELS:
    pkl_path = ROOT / "models" / m / f"{TAG}_model.pkl"
    if not pkl_path.exists():
        print(f"  {LABELS[m]}: no pkl at {pkl_path} — skipping")
        continue
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    clf = obj.named_steps.get("clf") if hasattr(obj, "named_steps") else obj
    if clf is None:
        continue
    if m == "baseline":
        continue  # no feature importances for the counting baseline

    # Determine feature names for this model
    if m == "logreg":
        # After LTE: drop raw cat IDs and precomputed ratios; append cat cols
        cat_cols   = ["aisle_id", "department_id"]
        drop_cols  = {"aisle_id", "department_id", "aisle_reorder_ratio", "dept_reorder_ratio"}
        num_after  = [c for c in feat_cols if c not in drop_cols and c not in ["aisle_id","department_id"]]
        fnames     = num_after + [c for c in cat_cols if c in feat_cols]
        coefs      = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else np.array([])
        if len(coefs) == len(fnames):
            imp_store[m] = dict(zip(fnames, coefs))
    else:
        cat_cols  = [c for c in ["aisle_id", "department_id"] if c in feat_cols]
        num_cols  = [c for c in feat_cols if c not in ["aisle_id", "department_id"]]
        fnames    = num_cols + cat_cols
        imp       = getattr(clf, "feature_importances_", None)
        if imp is not None and len(imp) == len(fnames):
            imp_store[m] = dict(zip(fnames, imp))
        else:
            print(f"  {LABELS[m]}: importance length mismatch ({len(imp) if imp is not None else 'None'} vs {len(fnames)})")

print(f"Loaded importances for: {list(imp_store.keys())}")

# ── Top-10 per model ─────────────────────────────────────────────────────────
top10 = {}
for m, imp_d in imp_store.items():
    ranked = sorted(imp_d.items(), key=lambda x: x[1], reverse=True)
    top10[m] = [f for f, _ in ranked[:10]]
    print(f"\\n{LABELS[m]} top-10: {top10[m]}")

# ── Pairwise overlap matrix ───────────────────────────────────────────────────
ml_models = [m for m in MODELS if m in imp_store]
overlap = pd.DataFrame(index=[LABELS[m] for m in ml_models],
                        columns=[LABELS[m] for m in ml_models], dtype=float)
for a in ml_models:
    for b in ml_models:
        shared = len(set(top10.get(a,[])) & set(top10.get(b,[])))
        overlap.loc[LABELS[a], LABELS[b]] = shared

print("\\nTop-10 overlap matrix (shared features out of 10):")
display(overlap)
overlap.to_csv(TBL / f"feat_importance_overlap_{TAG}.csv")

# ── Plot comparison bar chart ─────────────────────────────────────────────────
if imp_store:
    # Collect all features that appear in top-10 of ANY model
    all_top = list({f for fs in top10.values() for f in fs})
    imp_df  = pd.DataFrame({LABELS[m]: [imp_store[m].get(f, 0) for f in all_top]
                            for m in ml_models}, index=all_top)
    # Normalise per model to [0,1] for visual comparison
    imp_df  = imp_df / imp_df.max().replace(0, 1)
    imp_df  = imp_df.sort_values(by=LABELS[ml_models[-1]], ascending=False)

    fig, ax = plt.subplots(figsize=(12, max(5, len(all_top) * 0.4)))
    x   = np.arange(len(all_top))
    w   = 0.8 / max(1, len(ml_models))
    for i, m in enumerate(ml_models):
        ax.barh(x + (i - len(ml_models)/2 + 0.5)*w,
                imp_df[LABELS[m]].values, w,
                label=LABELS[m], color=COLORS[m])
    ax.set_yticks(x)
    ax.set_yticklabels(imp_df.index, fontsize=8)
    ax.set_xlabel("Normalised importance")
    ax.set_title(f"Cross-model feature importance (top-10 union, {TAG.upper()})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / f"paper_feat_importance_xmodel_{TAG}.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved cross-model importance plot")

# ── Disagreement features (plan Step 3) ──────────────────────────────────────
ada_top  = set(top10.get("adaboost", []))
xgb_top  = set(top10.get("xgboost",  []))
both     = ada_top & xgb_top
xgb_only = xgb_top - ada_top
ada_only = ada_top - xgb_top
print(f"\\nAdaBoost vs XGBoost (plan §6.2.4 Step 3):")
print(f"  Both top-10     : {sorted(both)}")
print(f"  XGBoost-only    : {sorted(xgb_only)}  ← likely interaction-dependent features")
print(f"  AdaBoost-only   : {sorted(ada_only)}")
"""

H9 = """\
## Statistical Significance (plan §6.3)

- **McNemar's test** on paired binary predictions (with Bonferroni correction)
- **Bootstrap F1 confidence intervals** (1000 resamples of users)
- **Bootstrap AUC confidence intervals**
"""

STAT_SIG = """\
from scipy.stats import chi2

def mcnemar_stat(preds_a, preds_b, y_true):
    \"\"\"McNemar's test: are the disagreement counts symmetric?\"\"\"\n    a_right_b_wrong = ((preds_a == y_true) & (preds_b != y_true)).sum()
    a_wrong_b_right = ((preds_a != y_true) & (preds_b == y_true)).sum()
    n = a_right_b_wrong + a_wrong_b_right
    if n == 0:
        return float("nan"), float("nan")
    # With continuity correction (recommended when n < 25)
    stat = (abs(a_right_b_wrong - a_wrong_b_right) - 1) ** 2 / n
    p    = 1 - chi2.cdf(stat, df=1)
    return float(stat), float(p)

def bootstrap_f1(y_true, y_pred, user_ids, n_boot=1000, seed=42):
    \"\"\"Bootstrap CI for per-order F1 by resampling users.\"\"\"\n    rng   = np.random.default_rng(seed)
    users = np.unique(user_ids)
    df    = pd.DataFrame({"u": user_ids, "yt": y_true, "yp": y_pred})
    scores = []
    for _ in range(n_boot):
        samp  = rng.choice(users, size=len(users), replace=True)
        sub   = df[df.u.isin(samp)]
        per_u = [float(sub[sub.u==u]["yt"].values.__class__(sub[sub.u==u]["yt"]) ==
                        sub[sub.u==u]["yp"].values.__class__(sub[sub.u==u]["yp"]))
                 for u in samp]
        # per-user f1
        from sklearn.metrics import f1_score as _f1s
        fu = []
        for u in samp:
            g = sub[sub.u == u]
            fu.append(_f1s(g["yt"], g["yp"], zero_division=0))
        scores.append(np.mean(fu))
    return np.percentile(scores, [2.5, 97.5])

def bootstrap_auc(y_true, y_prob, user_ids, n_boot=1000, seed=42):
    \"\"\"Bootstrap CI for ROC-AUC by resampling users.\"\"\"\n    from sklearn.metrics import roc_auc_score as _auc
    rng   = np.random.default_rng(seed)
    users = np.unique(user_ids)
    df    = pd.DataFrame({"u": user_ids, "yt": y_true.astype(float), "yp": y_prob})
    scores = []
    for _ in range(n_boot):
        samp = rng.choice(users, size=len(users), replace=True)
        sub  = df[df.u.isin(samp)]
        if sub["yt"].nunique() < 2:
            continue
        try:
            scores.append(_auc(sub["yt"], sub["yp"]))
        except Exception:
            pass
    return np.percentile(scores, [2.5, 97.5]) if scores else [float("nan")]*2

# ── McNemar's test (all pairs) ───────────────────────────────────────────────
BONFERRONI_ALPHA = 0.05 / 6   # 6 pairs from 4 ML models
mcnemar_rows = []
ml_models_all = [m for m in MODELS if m != "baseline"]
for i, ma in enumerate(ml_models_all):
    for j, mb in enumerate(ml_models_all):
        if j <= i:
            continue
        da  = preds_store[ma]
        db  = preds_store[mb]
        # Align on same test rows (same order guaranteed by same test parquet)
        stat, p = mcnemar_stat(da["pred"].values, db["pred"].values,
                               da["label"].values)
        mcnemar_rows.append({
            "Model A":   LABELS[ma],
            "Model B":   LABELS[mb],
            "McNemar χ²": round(stat, 4) if not np.isnan(stat) else "n/a",
            "p-value":   f"{p:.4g}" if not np.isnan(p) else "n/a",
            "Significant (Bonferroni)": "Yes" if (not np.isnan(p) and p < BONFERRONI_ALPHA) else "No",
        })

t7 = pd.DataFrame(mcnemar_rows)
t7.to_csv(TBL / f"t7_mcnemar_{TAG}.csv", index=False)
display(t7.set_index(["Model A","Model B"])
         .style.set_caption(
             f"Table 7 — McNemar's test on paired predictions ({TAG}). "
             f"Bonferroni α = {BONFERRONI_ALPHA:.4f} ({len(mcnemar_rows)} pairs)."))
print(f"Saved: {TBL / f't7_mcnemar_{TAG}.csv'}")

# ── Bootstrap F1 + AUC CI ────────────────────────────────────────────────────
print("\\nBootstrap F1 and ROC-AUC CIs (1000 resamples of users) …")
boot_rows = []
for m in MODELS:
    df  = preds_store[m]
    f1_ci  = bootstrap_f1( df["label"].values, df["pred"].values,
                            df["user_id"].values)
    auc_ci = bootstrap_auc(df["label"].values, df["prob"].values,
                            df["user_id"].values)
    d = json.loads((RES / m / f"{TAG}_metrics.json").read_text())["metrics"]
    boot_rows.append({
        "Model":          LABELS[m],
        "Per-row F1":     round(d["f1"],      4),
        "F1 95% CI low":  round(f1_ci[0],     4),
        "F1 95% CI high": round(f1_ci[1],     4),
        "ROC-AUC":        round(d["roc_auc"], 4),
        "AUC 95% CI low": round(auc_ci[0],    4),
        "AUC 95% CI high":round(auc_ci[1],    4),
    })

t8 = pd.DataFrame(boot_rows).set_index("Model")
t8.to_csv(TBL / f"t8_bootstrap_ci_{TAG}.csv")
display(t8.style.set_caption(
    f"Table 8 — Bootstrap 95% CIs for F1 and AUC ({TAG}, 1000 user resamples). "
    "Per-order F1 is the basket-level metric; AUC is per-row."))
print(f"Saved: {TBL / f't8_bootstrap_ci_{TAG}.csv'}")
"""


# ── Assemble cells ────────────────────────────────────────────────────────────

cells = [
    md(TITLE),
    code(SETUP),
    code(LOAD_META),
    code(LOAD_PREDS),
    md(H1),
    code(TABLE1),
    md(H2),
    code(TABLE2),
    md(H3),
    code(TABLE3_PER),
    code(TABLE3_COMBINED),
    md(H4),
    code(TABLE4_PER),
    code(TABLE4_COMBINED),
    md(H5),
    code(TABLE5A),
    code(TABLE5B),
    md(H6),
    code(TABLE6),
    md(H7),
    code(FIG1),
    code(FIG2),
    code(FIG3),
    md(H8),
    code(FEAT_IMP_XMODEL),
    md(H9),
    code(STAT_SIG),
    code(RUN_CFG),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {NB} ({len(cells)} cells)")
