#!/usr/bin/env python3
"""Generate the final report results notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NB_PATH = Path(__file__).parent / "final_report_results.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    md(
        """
        # Final Report Results Notebook

        This notebook turns the completed Phase 4 and Phase 5 artifacts into report-ready tables and figures.

        It follows `project_plan.md` Section 6:

        - report both per-row F1 and per-order F1
        - compare precision, recall, ROC-AUC, PR-AUC, and accuracy as supplementary metrics
        - decompose FP/FN errors
        - analyze model disagreement and error overlap
        - slice errors by Phase 1 subgroups
        - compare built-in feature importance across models

        Note: the project plan also describes permutation importance and retraining ablations. Those artifacts are not present in the current repository, so this notebook does not claim those analyses. It limits feature analysis to saved built-in importance files.
        """
    ),
    code(
        """
        from pathlib import Path
        from itertools import combinations
        import json
        import warnings

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.metrics import f1_score, precision_score, recall_score

        warnings.filterwarnings("ignore")

        if Path.cwd().name == "src":
            ROOT = Path.cwd().parent
        else:
            ROOT = Path.cwd()

        RES = ROOT / "results"
        TBL = RES / "report_tables"
        PLOT = ROOT / "plots" / "report"
        TBL.mkdir(parents=True, exist_ok=True)
        PLOT.mkdir(parents=True, exist_ok=True)

        MODELS = ["last_basket", "logreg", "rf", "adaboost", "xgboost"]
        ML_MODELS = ["logreg", "rf", "adaboost", "xgboost"]
        LABELS = {
            "last_basket": "Last Basket",
            "logreg": "Logistic Regression",
            "rf": "Random Forest",
            "adaboost": "AdaBoost",
            "xgboost": "XGBoost",
        }
        COLORS = {
            "last_basket": "#7f7f7f",
            "logreg": "#4e79a7",
            "rf": "#f28e2b",
            "adaboost": "#e15759",
            "xgboost": "#59a14f",
        }

        plt.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        })

        print("Root:", ROOT)
        print("Tables:", TBL)
        print("Plots:", PLOT)
        """
    ),
    md("## Load Final Artifacts"),
    code(
        """
        required = [
            RES / "test_predictions.parquet",
            RES / "metrics_summary.csv",
            RES / "model_disagreement.csv",
            RES / "basket_metrics.csv",
            RES / "behavioral_error_tags.csv",
            RES / "tables" / "final_model_comparison.csv",
            RES / "tables" / "final_mcnemar_bonferroni.csv",
            RES / "tables" / "final_bootstrap_f1_diff.csv",
            RES / "tables" / "final_delong_roc_auc.csv",
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing required artifacts:\\n" + "\\n".join(str(p) for p in missing))

        master = pd.read_parquet(RES / "test_predictions.parquet")
        metrics = pd.read_csv(RES / "metrics_summary.csv")
        disagreement = pd.read_csv(RES / "model_disagreement.csv")
        basket = pd.read_csv(RES / "basket_metrics.csv")
        tags = pd.read_csv(RES / "behavioral_error_tags.csv", low_memory=False)

        final_comparison = pd.read_csv(RES / "tables" / "final_model_comparison.csv")
        mcnemar = pd.read_csv(RES / "tables" / "final_mcnemar_bonferroni.csv")
        bootstrap = pd.read_csv(RES / "tables" / "final_bootstrap_f1_diff.csv")
        delong = pd.read_csv(RES / "tables" / "final_delong_roc_auc.csv")

        subgroup_cols = [
            "is_cold_start_user",
            "is_none_user",
            "is_high_variance_aisle",
            "is_single_purchase_pair",
            "primary_user_subgroup",
        ]
        assert master[subgroup_cols].isna().sum().sum() == 0
        assert set(metrics["model"]) == set(MODELS)

        print("master:", master.shape)
        print("metrics:", metrics.shape)
        print("disagreement:", disagreement.shape)
        print("basket:", basket.shape)
        print("behavioral tags:", tags.shape)
        """
    ),
    md(
        """
        ## Table 1: Model Performance Summary

        The plan asks for both global per-row F1 and Kaggle-style per-order F1. The table below keeps both visible. In these controlled final results, Random Forest is best by row-level F1 and ROC-AUC, XGBoost is best by PR-AUC, and Logistic Regression is best by per-order F1.
        """
    ),
    code(
        """
        metric_cols = [
            "model", "model_label", "threshold", "n_rows", "n_users", "positive_rate",
            "accuracy", "precision", "recall", "f1", "per_order_f1", "roc_auc", "pr_auc",
            "avg_true_basket_size", "avg_pred_basket_size", "basket_size_mae",
            "basket_inflation_rate", "large_basket_inflation_rate", "none_collapse_rate",
        ]
        report_metrics = metrics[metric_cols].copy().sort_values("f1", ascending=False)
        report_metrics.to_csv(TBL / "report_model_metrics.csv", index=False)

        display(
            report_metrics.style
                .format({
                    "threshold": "{:.2f}",
                    "positive_rate": "{:.3f}",
                    "accuracy": "{:.4f}",
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1": "{:.4f}",
                    "per_order_f1": "{:.4f}",
                    "roc_auc": "{:.4f}",
                    "pr_auc": "{:.4f}",
                    "avg_true_basket_size": "{:.2f}",
                    "avg_pred_basket_size": "{:.2f}",
                    "basket_size_mae": "{:.2f}",
                    "basket_inflation_rate": "{:.3f}",
                    "large_basket_inflation_rate": "{:.3f}",
                    "none_collapse_rate": "{:.3f}",
                })
                .highlight_max(subset=["f1", "per_order_f1", "roc_auc", "pr_auc"], color="#d8f3dc")
                .set_caption("Table 1. Controlled final model performance.")
        )
        print("Saved:", TBL / "report_model_metrics.csv")
        """
    ),
    code(
        """
        plot_df = report_metrics.set_index("model_label")[["f1", "per_order_f1", "roc_auc", "pr_auc"]]
        ax = plot_df.plot(kind="bar", figsize=(9.5, 4.8), color=["#4e79a7", "#f28e2b", "#59a14f", "#e15759"])
        ax.set_title("Final model metrics")
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.legend(["Row F1", "Per-order F1", "ROC-AUC", "PR-AUC"], ncol=2, frameon=False)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(PLOT / "01_model_metrics.png")
        plt.show()
        print("Saved:", PLOT / "01_model_metrics.png")
        """
    ),
    md("## Table 2: Confusion-Matrix Decomposition"),
    code(
        """
        rows = []
        for model in MODELS:
            err = master[f"{model}_error_type"]
            y = master["label"].astype(int)
            tp = int((err == "TP").sum())
            fp = int((err == "FP").sum())
            fn = int((err == "FN").sum())
            tn = int((err == "TN").sum())
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            rows.append({
                "model": model,
                "model_label": LABELS[model],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "miss_rate": fn / n_pos,
                "false_alarm_rate": fp / n_neg,
                "predicted_positive_rate": int(master[f"{model}_pred"].sum()) / len(master),
            })
        confusion_report = pd.DataFrame(rows)
        confusion_report.to_csv(TBL / "report_confusion_decomposition.csv", index=False)
        display(
            confusion_report.style
                .format({
                    "miss_rate": "{:.3f}",
                    "false_alarm_rate": "{:.3f}",
                    "predicted_positive_rate": "{:.3f}",
                })
                .background_gradient(subset=["miss_rate"], cmap="Reds")
                .background_gradient(subset=["false_alarm_rate"], cmap="Oranges")
                .set_caption("Table 2. Error decomposition. Miss rate = FN / positives; false alarm rate = FP / negatives.")
        )
        print("Saved:", TBL / "report_confusion_decomposition.csv")
        """
    ),
    code(
        """
        plot_df = confusion_report.set_index("model_label")[["miss_rate", "false_alarm_rate", "predicted_positive_rate"]]
        ax = plot_df.plot(kind="bar", figsize=(9, 4.5), color=["#e15759", "#f28e2b", "#4e79a7"])
        ax.set_title("Misses, false alarms, and prediction volume")
        ax.set_xlabel("")
        ax.set_ylabel("Rate")
        ax.legend(["Miss rate", "False alarm rate", "Predicted positive rate"], frameon=False)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(PLOT / "02_error_rates.png")
        plt.show()
        print("Saved:", PLOT / "02_error_rates.png")
        """
    ),
    md("## Table 3: Effect Sizes and Significance"),
    code(
        """
        rf = report_metrics.set_index("model").loc["rf"]
        effect_rows = []
        for _, row in report_metrics.iterrows():
            if row["model"] == "rf":
                continue
            effect_rows.append({
                "comparison": f"rf - {row['model']}",
                "row_f1_diff": rf["f1"] - row["f1"],
                "per_order_f1_diff": rf["per_order_f1"] - row["per_order_f1"],
                "roc_auc_diff": rf["roc_auc"] - row["roc_auc"],
                "pr_auc_diff": rf["pr_auc"] - row["pr_auc"],
            })
        effect_sizes = pd.DataFrame(effect_rows)
        effect_sizes.to_csv(TBL / "report_effect_sizes_vs_rf.csv", index=False)

        mcnemar.to_csv(TBL / "report_mcnemar_bonferroni.csv", index=False)
        bootstrap.to_csv(TBL / "report_bootstrap_f1_diff.csv", index=False)
        delong.to_csv(TBL / "report_delong_roc_auc.csv", index=False)

        display(effect_sizes.style.format({
            "row_f1_diff": "{:+.6f}",
            "per_order_f1_diff": "{:+.6f}",
            "roc_auc_diff": "{:+.6f}",
            "pr_auc_diff": "{:+.6f}",
        }).set_caption("Table 3A. Raw RF effect sizes versus each model."))
        display(bootstrap.style.format({"mean_diff": "{:+.6f}", "ci_low": "{:+.6f}", "ci_high": "{:+.6f}"}).set_caption("Table 3B. Bootstrap per-order F1 differences from existing controlled comparison."))
        display(delong.style.format({"auc_a": "{:.6f}", "auc_b": "{:.6f}", "auc_diff": "{:+.6f}", "p": "{:.3e}"}).set_caption("Table 3C. DeLong ROC-AUC tests from existing controlled comparison."))
        print("Saved significance/effect-size tables under:", TBL)
        """
    ),
    code(
        """
        ax = effect_sizes.set_index("comparison")[["row_f1_diff", "per_order_f1_diff", "roc_auc_diff", "pr_auc_diff"]].plot(
            kind="bar", figsize=(9, 4.5), color=["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("RF effect sizes versus alternatives")
        ax.set_xlabel("")
        ax.set_ylabel("RF minus comparison")
        ax.legend(["Row F1", "Per-order F1", "ROC-AUC", "PR-AUC"], ncol=2, frameon=False)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(PLOT / "03_effect_sizes_vs_rf.png")
        plt.show()
        print("Saved:", PLOT / "03_effect_sizes_vs_rf.png")
        """
    ),
    md("## Table 4: Basket-Level Metrics"),
    code(
        """
        basket_summary = (
            basket.groupby(["model", "model_label"])
            .agg(
                n_users=("user_id", "nunique"),
                mean_true_basket_size=("true_basket_size", "mean"),
                mean_pred_basket_size=("pred_basket_size", "mean"),
                mean_per_order_f1=("per_order_f1", "mean"),
                median_per_order_f1=("per_order_f1", "median"),
                basket_size_mae=("basket_size_abs_error", "mean"),
                basket_inflation_rate=("basket_size_inflation", "mean"),
                large_basket_inflation_rate=("basket_size_inflation_large", "mean"),
                none_collapse_rate=("none_collapse_case", "mean"),
            )
            .reset_index()
            .sort_values("mean_per_order_f1", ascending=False)
        )
        basket_summary.to_csv(TBL / "report_basket_summary.csv", index=False)
        display(
            basket_summary.style
                .format({
                    "mean_true_basket_size": "{:.2f}",
                    "mean_pred_basket_size": "{:.2f}",
                    "mean_per_order_f1": "{:.4f}",
                    "median_per_order_f1": "{:.4f}",
                    "basket_size_mae": "{:.2f}",
                    "basket_inflation_rate": "{:.3f}",
                    "large_basket_inflation_rate": "{:.3f}",
                    "none_collapse_rate": "{:.3f}",
                })
                .highlight_max(subset=["mean_per_order_f1"], color="#d8f3dc")
                .set_caption("Table 4. Basket-level metrics by model.")
        )
        print("Saved:", TBL / "report_basket_summary.csv")
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        order = basket_summary.sort_values("mean_per_order_f1", ascending=False)["model"].tolist()
        data = [basket.loc[basket["model"] == m, "per_order_f1"].to_numpy() for m in order]
        axes[0].boxplot(data, labels=[LABELS[m] for m in order], showfliers=False)
        axes[0].set_title("Per-order F1 distribution")
        axes[0].set_ylabel("Per-order F1")
        axes[0].tick_params(axis="x", rotation=25)

        width = 0.38
        x = np.arange(len(order))
        true_vals = [basket_summary.set_index("model").loc[m, "mean_true_basket_size"] for m in order]
        pred_vals = [basket_summary.set_index("model").loc[m, "mean_pred_basket_size"] for m in order]
        axes[1].bar(x - width/2, true_vals, width, label="True", color="#bab0ac")
        axes[1].bar(x + width/2, pred_vals, width, label="Predicted", color="#4e79a7")
        axes[1].set_title("Average basket size")
        axes[1].set_ylabel("Products per user")
        axes[1].set_xticks(x, [LABELS[m] for m in order], rotation=25, ha="right")
        axes[1].legend(frameon=False)

        plt.tight_layout()
        plt.savefig(PLOT / "04_basket_metrics.png")
        plt.show()
        print("Saved:", PLOT / "04_basket_metrics.png")
        """
    ),
    md("## Table 5: Model Disagreement and Error Overlap"),
    code(
        """
        pred_cols = {m: f"{m}_pred" for m in MODELS}
        err_cols = {m: f"{m}_error_type" for m in MODELS}

        disagree_mat = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)
        overlap_rows = []
        for a, b in combinations(MODELS, 2):
            a_pred = master[pred_cols[a]].to_numpy()
            b_pred = master[pred_cols[b]].to_numpy()
            a_err = ~master[err_cols[a]].isin(["TP", "TN"]).to_numpy()
            b_err = ~master[err_cols[b]].isin(["TP", "TN"]).to_numpy()

            disagree = float((a_pred != b_pred).mean())
            disagree_mat.loc[a, b] = disagree
            disagree_mat.loc[b, a] = disagree

            both_error = float((a_err & b_err).mean())
            a_only = float((a_err & ~b_err).mean())
            b_only = float((~a_err & b_err).mean())
            both_correct = float((~a_err & ~b_err).mean())
            union_error = float((a_err | b_err).mean())
            overlap_rows.append({
                "model_a": a,
                "model_b": b,
                "prediction_disagreement_rate": disagree,
                "both_error_rate": both_error,
                "model_a_only_error_rate": a_only,
                "model_b_only_error_rate": b_only,
                "both_correct_rate": both_correct,
                "error_jaccard": both_error / union_error if union_error else np.nan,
            })
        np.fill_diagonal(disagree_mat.values, 0)
        disagree_mat.index = [LABELS[m] for m in MODELS]
        disagree_mat.columns = [LABELS[m] for m in MODELS]
        disagreement_matrix = disagree_mat.reset_index().rename(columns={"index": "model"})
        disagreement_matrix.to_csv(TBL / "report_prediction_disagreement_matrix.csv", index=False)

        error_overlap = pd.DataFrame(overlap_rows)
        error_overlap.to_csv(TBL / "report_pairwise_error_overlap.csv", index=False)
        display(error_overlap.style.format({
            "prediction_disagreement_rate": "{:.3f}",
            "both_error_rate": "{:.3f}",
            "model_a_only_error_rate": "{:.3f}",
            "model_b_only_error_rate": "{:.3f}",
            "both_correct_rate": "{:.3f}",
            "error_jaccard": "{:.3f}",
        }).set_caption("Table 5. Pairwise prediction disagreement and error overlap."))
        print("Saved:", TBL / "report_pairwise_error_overlap.csv")
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

        im0 = axes[0].imshow(disagree_mat.values.astype(float), cmap="Blues", vmin=0, vmax=np.nanmax(disagree_mat.values.astype(float)))
        axes[0].set_title("Prediction disagreement rate")
        axes[0].set_xticks(range(len(MODELS)), [LABELS[m] for m in MODELS], rotation=35, ha="right")
        axes[0].set_yticks(range(len(MODELS)), [LABELS[m] for m in MODELS])
        for i in range(len(MODELS)):
            for j in range(len(MODELS)):
                axes[0].text(j, i, f"{disagree_mat.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        jaccard = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)
        for _, row in error_overlap.iterrows():
            jaccard.loc[row["model_a"], row["model_b"]] = row["error_jaccard"]
            jaccard.loc[row["model_b"], row["model_a"]] = row["error_jaccard"]
        np.fill_diagonal(jaccard.values, 1)
        im1 = axes[1].imshow(jaccard.values.astype(float), cmap="Oranges", vmin=0, vmax=1)
        axes[1].set_title("Error-overlap Jaccard")
        axes[1].set_xticks(range(len(MODELS)), [LABELS[m] for m in MODELS], rotation=35, ha="right")
        axes[1].set_yticks(range(len(MODELS)), [LABELS[m] for m in MODELS])
        for i in range(len(MODELS)):
            for j in range(len(MODELS)):
                axes[1].text(j, i, f"{jaccard.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(PLOT / "05_model_disagreement_overlap.png")
        plt.show()
        print("Saved:", PLOT / "05_model_disagreement_overlap.png")
        """
    ),
    code(
        """
        disagreement_examples = (
            disagreement.sort_values(["n_models_pred_positive", "prob_range"], ascending=[False, False])
            .head(30)
            .copy()
        )
        disagreement_examples.to_csv(TBL / "report_disagreement_examples.csv", index=False)
        display(disagreement_examples.head(10))
        print("Saved:", TBL / "report_disagreement_examples.csv")
        """
    ),
    md("## Table 6: Subgroup Error Analysis"),
    code(
        """
        subgroup_rows = []
        for model in MODELS:
            for subgroup, g in master.groupby("primary_user_subgroup"):
                y = g["label"].astype(int).to_numpy()
                pred = g[f"{model}_pred"].astype(int).to_numpy()
                err = g[f"{model}_error_type"]
                n_pos = int((y == 1).sum())
                n_neg = int((y == 0).sum())
                subgroup_rows.append({
                    "model": model,
                    "model_label": LABELS[model],
                    "subgroup": subgroup,
                    "n_rows": len(g),
                    "n_users": g["user_id"].nunique(),
                    "positive_rate": y.mean() if len(y) else np.nan,
                    "precision": precision_score(y, pred, zero_division=0),
                    "recall": recall_score(y, pred, zero_division=0),
                    "f1": f1_score(y, pred, zero_division=0),
                    "miss_rate": int((err == "FN").sum()) / n_pos if n_pos else np.nan,
                    "false_alarm_rate": int((err == "FP").sum()) / n_neg if n_neg else np.nan,
                })
        subgroup_metrics = pd.DataFrame(subgroup_rows)
        subgroup_metrics.to_csv(TBL / "report_subgroup_metrics.csv", index=False)
        display(
            subgroup_metrics.sort_values(["subgroup", "f1"], ascending=[True, False])
            .style
            .format({
                "positive_rate": "{:.3f}",
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1": "{:.3f}",
                "miss_rate": "{:.3f}",
                "false_alarm_rate": "{:.3f}",
            })
            .background_gradient(subset=["f1"], cmap="Greens")
            .set_caption("Table 6. Row-level subgroup metrics by primary user subgroup.")
        )
        print("Saved:", TBL / "report_subgroup_metrics.csv")
        """
    ),
    code(
        """
        pivot = subgroup_metrics.pivot(index="subgroup", columns="model_label", values="f1")
        ax = pivot.plot(kind="bar", figsize=(10, 4.8), color=[COLORS[m] for m in MODELS if LABELS[m] in pivot.columns])
        ax.set_title("Subgroup F1 by model")
        ax.set_xlabel("")
        ax.set_ylabel("Row-level F1")
        ax.legend(frameon=False, ncol=2)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(PLOT / "06_subgroup_f1.png")
        plt.show()
        print("Saved:", PLOT / "06_subgroup_f1.png")
        """
    ),
    md("## Table 7: Behavioral Error Tags"),
    code(
        """
        tag_counts = (
            tags.groupby(["model", "model_label", "behavioral_error_tag"])
            .size()
            .rename("n_rows")
            .reset_index()
        )
        totals = tags.groupby("model").size().rename("model_rows")
        tag_counts = tag_counts.merge(totals, on="model", how="left")
        tag_counts["share_of_model_rows"] = tag_counts["n_rows"] / tag_counts["model_rows"]
        tag_counts = tag_counts.sort_values(["model", "n_rows"], ascending=[True, False])
        tag_counts.to_csv(TBL / "report_behavioral_error_tags.csv", index=False)

        display(
            tag_counts.style
            .format({"share_of_model_rows": "{:.3f}"})
            .set_caption("Table 7. Behavioral error-tag counts. Basket-level tags are repeated across the model's user-product rows.")
        )
        print("Saved:", TBL / "report_behavioral_error_tags.csv")
        """
    ),
    code(
        """
        tag_pivot = tag_counts.pivot(index="behavioral_error_tag", columns="model_label", values="share_of_model_rows").fillna(0)
        tag_pivot = tag_pivot.loc[tag_pivot.mean(axis=1).sort_values(ascending=True).index]
        ax = tag_pivot.plot(kind="barh", figsize=(9.5, 5.5), color=[COLORS[m] for m in MODELS if LABELS[m] in tag_pivot.columns])
        ax.set_title("Behavioral error-tag share by model")
        ax.set_xlabel("Share of model rows")
        ax.set_ylabel("")
        ax.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(PLOT / "07_behavioral_error_tags.png")
        plt.show()
        print("Saved:", PLOT / "07_behavioral_error_tags.png")
        """
    ),
    md("## Table 8: Built-In Feature Importance Comparison"),
    code(
        """
        importance_frames = []
        for model in ML_MODELS:
            path = RES / model / f"feature_importance_{model}.csv"
            if not path.exists():
                print("Missing:", path)
                continue
            imp = pd.read_csv(path)
            if "feature_name" not in imp.columns:
                imp = imp.rename(columns={imp.columns[0]: "feature_name", imp.columns[1]: "importance_score"})
            imp = imp[["feature_name", "importance_score"]].copy()
            imp["model"] = model
            imp["model_label"] = LABELS[model]
            imp["rank"] = imp["importance_score"].rank(method="first", ascending=False).astype(int)
            importance_frames.append(imp)

        feature_importance = pd.concat(importance_frames, ignore_index=True)
        top15_long = feature_importance[feature_importance["rank"] <= 15].sort_values(["model", "rank"])
        top15_long.to_csv(TBL / "report_feature_top15_long.csv", index=False)
        display(top15_long)
        print("Saved:", TBL / "report_feature_top15_long.csv")
        """
    ),
    code(
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        for ax, model in zip(axes, ML_MODELS):
            sub = feature_importance[feature_importance["model"] == model].sort_values("rank").head(15)
            sub = sub.sort_values("importance_score", ascending=True)
            ax.barh(sub["feature_name"], sub["importance_score"], color=COLORS[model])
            ax.set_title(LABELS[model])
            ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig(PLOT / "08_feature_importance_top15.png")
        plt.show()
        print("Saved:", PLOT / "08_feature_importance_top15.png")
        """
    ),
    code(
        """
        top10 = {
            model: set(feature_importance[(feature_importance["model"] == model) & (feature_importance["rank"] <= 10)]["feature_name"])
            for model in ML_MODELS
        }
        overlap = pd.DataFrame(index=ML_MODELS, columns=ML_MODELS, dtype=float)
        shared_counts = pd.DataFrame(index=ML_MODELS, columns=ML_MODELS, dtype=int)
        for a in ML_MODELS:
            for b in ML_MODELS:
                inter = len(top10[a] & top10[b])
                union = len(top10[a] | top10[b])
                overlap.loc[a, b] = inter / union if union else np.nan
                shared_counts.loc[a, b] = inter

        overlap_labeled = overlap.rename(index=LABELS, columns=LABELS)
        shared_counts_labeled = shared_counts.rename(index=LABELS, columns=LABELS)
        overlap_labeled.to_csv(TBL / "report_feature_top10_overlap_jaccard.csv", index_label="model")
        shared_counts_labeled.to_csv(TBL / "report_feature_top10_shared_counts.csv", index_label="model")

        display(overlap_labeled.style.format("{:.3f}").set_caption("Table 8A. Jaccard overlap of each model's top-10 built-in importance features."))
        display(shared_counts_labeled.style.set_caption("Table 8B. Count of shared top-10 built-in importance features."))

        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        im = ax.imshow(overlap.values.astype(float), cmap="Greens", vmin=0, vmax=1)
        ax.set_xticks(range(len(ML_MODELS)), [LABELS[m] for m in ML_MODELS], rotation=35, ha="right")
        ax.set_yticks(range(len(ML_MODELS)), [LABELS[m] for m in ML_MODELS])
        ax.set_title("Top-10 feature importance overlap")
        for i in range(len(ML_MODELS)):
            for j in range(len(ML_MODELS)):
                ax.text(j, i, f"{overlap.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(PLOT / "09_feature_top10_overlap.png")
        plt.show()
        print("Saved feature-overlap tables and plot.")
        """
    ),
    md("## Report Figure Inventory"),
    code(
        """
        inventory = pd.DataFrame({
            "figure": [
                "01_model_metrics.png",
                "02_error_rates.png",
                "03_effect_sizes_vs_rf.png",
                "04_basket_metrics.png",
                "05_model_disagreement_overlap.png",
                "06_subgroup_f1.png",
                "07_behavioral_error_tags.png",
                "08_feature_importance_top15.png",
                "09_feature_top10_overlap.png",
            ],
            "report_use": [
                "Main performance comparison",
                "FP/FN interpretation",
                "Practical effect-size discussion",
                "Per-order/basket-level interpretation",
                "Error-overlap and ensemble potential",
                "Subgroup robustness",
                "Behavioral error taxonomy",
                "Feature reliance by model",
                "Option A shared-feature design payoff",
            ],
        })
        inventory["path"] = inventory["figure"].map(lambda x: str(PLOT / x))
        inventory.to_csv(TBL / "report_figure_inventory.csv", index=False)
        display(inventory)
        print("Saved:", TBL / "report_figure_inventory.csv")
        """
    ),
    md(
        """
        ## Writing Notes

        Suggested final-report structure using this notebook:

        1. Start with Table 1 and `01_model_metrics.png`.
        2. State the metric nuance clearly: RF wins row-level F1 and ROC-AUC; Logistic Regression wins per-order F1; XGBoost wins PR-AUC.
        3. Use `02_error_rates.png` and Table 2 to explain why Logistic Regression has stronger basket-level F1 despite lower row-level precision.
        4. Use Table 3 to separate statistical significance from practical effect size.
        5. Use Table 5 and `05_model_disagreement_overlap.png` to describe shared blind spots and possible ensembling.
        6. Use Table 6 and `06_subgroup_f1.png` for subgroup conclusions.
        7. Use Table 8 and feature plots as the feature-importance analysis that is supported by available artifacts.
        """
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    nbf.write(nb, NB_PATH)
    print(f"Wrote {NB_PATH}")


if __name__ == "__main__":
    main()
