# Final Report File Guide

This guide maps each section of `report/final_report.tex` to the tables, plots,
and source artifacts used to write it.

## Main Report Files

- LaTeX source: `report/final_report.tex`
- Bibliography: `report/references.bib`
- Compiled PDF: `report/final_report.pdf`
- Report notebook: `src/final_report_results.ipynb`
- Notebook generator: `src/_gen_final_report_nb.py`

## Abstract

Uses summary values from:

- `results/report_tables/report_model_metrics.csv`
- `results/report_tables/report_confusion_decomposition.csv`
- `results/report_tables/report_basket_summary.csv`

Key claims supported:

- RF has best row-level F1 and ROC-AUC.
- XGBoost has best PR-AUC.
- Logistic Regression has best per-order F1.
- Logistic Regression predicts larger baskets.

## 1. Introduction

Uses project framing from:

- `project_plan.md`
- `left.md`

Uses citations from:

- `report/references.bib`
  - `instacart2017dataset`
  - `kaggle2017instacart`

No plot/table is directly inserted in this section.

## 2. Data and Problem Setup

Uses implementation/data artifacts:

- `src/eda.ipynb`
- `src/features.py`
- `data/processed/train.parquet`
- `data/processed/test.parquet`
- `results/report_tables/report_model_metrics.csv`

Relevant report values:

- `n_rows`
- `n_users`
- `positive_rate`
- `avg_true_basket_size`

No plot/table is directly inserted in this section.

## 3. Feature Engineering and Selection

Uses project plan and feature pipeline:

- `project_plan.md`
- `src/features.py`
- `src/train_utils.py`
- `data/processed/feature_cols.txt`
- `results/correlation_scope.json`

Uses citations from:

- `report/references.bib`
  - `onodera2017instacart`
  - `belsley1980regression`
  - `micci2001target`

Related report outputs used later:

- `results/report_tables/report_feature_top15_long.csv`
- `results/report_tables/report_feature_top10_overlap_jaccard.csv`
- `results/report_tables/report_feature_top10_shared_counts.csv`
- `plots/report/08_feature_importance_top15.png`
- `plots/report/09_feature_top10_overlap.png`

## 4. Models and Training

Uses training scripts:

- `src/train_last_basket.py`
- `src/train_logreg.py`
- `src/train_rf.py`
- `src/train_adaboost.py`
- `src/train_xgboost.py`
- `src/train_utils.py`

Uses final model artifacts:

- `models/logreg/logreg_best.pkl`
- `models/rf/rf_best.pkl`
- `models/adaboost/adaboost_best.pkl`
- `models/xgboost/xgboost_best.pkl`
- `results/<model>/final_metrics.json`
- `results/<model>/final_predictions.csv`
- `results/<model>/final_threshold_sweep.csv`
- `results/<model>/final_validation_threshold_sweep.csv`

Uses citations from:

- `report/references.bib`
  - `breiman2001random`
  - `freund1997adaboost`
  - `chen2016xgboost`
  - `bergstra2012random`

No plot/table is directly inserted in this section.

## 5. Evaluation Metrics

Uses metric-generation code:

- `src/train_utils.py`
- `src/compare_final_models.py`
- `src/generate_phase5_artifacts.py`

Uses statistical comparison tables:

- `results/tables/final_mcnemar_bonferroni.csv`
- `results/tables/final_bootstrap_f1_diff.csv`
- `results/tables/final_delong_roc_auc.csv`

Report-ready copies:

- `results/report_tables/report_mcnemar_bonferroni.csv`
- `results/report_tables/report_bootstrap_f1_diff.csv`
- `results/report_tables/report_delong_roc_auc.csv`

Uses citations from:

- `report/references.bib`
  - `mcnemar1947`
  - `delong1988`

No plot/table is directly inserted in this section, but the methodology supports
Tables 1 and 3.

## 6. Results

### Table 1: Final Controlled Model Metrics

Inserted in report as:

- Table: `Table 1`

Uses:

- `results/report_tables/report_model_metrics.csv`

Supporting plot:

- `plots/report/01_model_metrics.png`

Original source artifacts:

- `results/metrics_summary.csv`
- `results/tables/final_model_comparison.csv`

### Figure 1: Main Metric Comparison

Inserted in report as:

- `plots/report/01_model_metrics.png`

Generated from:

- `results/report_tables/report_model_metrics.csv`

## 6.1 Confusion Matrix Decomposition

### Table 2: Error Decomposition

Inserted in report as:

- Table: `Table 2`

Uses:

- `results/report_tables/report_confusion_decomposition.csv`

Supporting plot:

- `plots/report/02_error_rates.png`

Original source artifacts:

- `results/test_predictions.parquet`
- `results/metrics_summary.csv`

### Figure 2: Miss and False-Alarm Rates

Inserted in report as:

- `plots/report/02_error_rates.png`

Generated from:

- `results/report_tables/report_confusion_decomposition.csv`

Optional model-specific plots:

- `plots/last_basket/final_confusion_matrix.png`
- `plots/logreg/final_confusion_matrix.png`
- `plots/rf/final_confusion_matrix.png`
- `plots/adaboost/final_confusion_matrix.png`
- `plots/xgboost/final_confusion_matrix.png`

## 6.2 Basket-Level Behavior

Uses:

- `results/report_tables/report_basket_summary.csv`
- `results/basket_metrics.csv`

Inserted plot:

- `plots/report/04_basket_metrics.png`

Generated from:

- `results/report_tables/report_basket_summary.csv`
- `results/basket_metrics.csv`

Key values used in prose:

- `mean_true_basket_size`
- `mean_pred_basket_size`
- `mean_per_order_f1`
- `basket_size_mae`
- `basket_inflation_rate`

## 6.3 Effect Sizes and Statistical Tests

### Table 3: RF Effect Sizes

Inserted in report as:

- Table: `Table 3`

Uses:

- `results/report_tables/report_effect_sizes_vs_rf.csv`

Supporting plot:

- `plots/report/03_effect_sizes_vs_rf.png`

Supporting statistical tables:

- `results/report_tables/report_mcnemar_bonferroni.csv`
- `results/report_tables/report_bootstrap_f1_diff.csv`
- `results/report_tables/report_delong_roc_auc.csv`

Original source artifacts:

- `results/tables/final_mcnemar_bonferroni.csv`
- `results/tables/final_bootstrap_f1_diff.csv`
- `results/tables/final_delong_roc_auc.csv`

## 6.4 Model Disagreement and Error Overlap

Uses:

- `results/report_tables/report_pairwise_error_overlap.csv`
- `results/report_tables/report_prediction_disagreement_matrix.csv`
- `results/report_tables/report_disagreement_examples.csv`
- `results/model_disagreement.csv`

Inserted plot:

- `plots/report/05_model_disagreement_overlap.png`

Generated from:

- `results/report_tables/report_pairwise_error_overlap.csv`
- `results/report_tables/report_prediction_disagreement_matrix.csv`

Optional supporting table:

- `results/report_tables/report_disagreement_examples.csv`

## 7. Subgroup and Behavioral Error Analysis

### Subgroup Analysis

Uses:

- `results/report_tables/report_subgroup_metrics.csv`
- `results/test_predictions.parquet`
- `src/subgroups.py`

Inserted plot:

- `plots/report/06_subgroup_f1.png`

Generated from:

- `results/report_tables/report_subgroup_metrics.csv`

### Behavioral Error Tags

Uses:

- `results/report_tables/report_behavioral_error_tags.csv`
- `results/behavioral_error_tags.csv`
- `src/generate_phase5_artifacts.py`

Inserted plot:

- `plots/report/07_behavioral_error_tags.png`

Generated from:

- `results/report_tables/report_behavioral_error_tags.csv`

Behavioral categories used:

- FP stockpile error
- FP broken streak error
- FP recipe/novelty error
- FN sleeper staple error
- FN new habit error
- Basket size inflation
- None-collapse cases

## 8. Feature Importance

### Figure: Top Built-In Feature Importances

Inserted plot:

- `plots/report/08_feature_importance_top15.png`

Generated from:

- `results/report_tables/report_feature_top15_long.csv`

Original model-specific feature importance files:

- `results/logreg/feature_importance_logreg.csv`
- `results/rf/feature_importance_rf.csv`
- `results/adaboost/feature_importance_adaboost.csv`
- `results/xgboost/feature_importance_xgboost.csv`

Optional model-specific plots:

- `plots/logreg/final_feature_importance.png`
- `plots/rf/final_feature_importance.png`
- `plots/adaboost/final_feature_importance.png`
- `plots/xgboost/final_feature_importance.png`

### Table 4: Top-10 Feature Overlap

Inserted in report as:

- Table: `Table 4`

Uses:

- `results/report_tables/report_feature_top10_overlap_jaccard.csv`

Supporting table:

- `results/report_tables/report_feature_top10_shared_counts.csv`

Supporting plot:

- `plots/report/09_feature_top10_overlap.png`

## 9. Discussion

Uses synthesized findings from:

- `results/report_tables/report_model_metrics.csv`
- `results/report_tables/report_confusion_decomposition.csv`
- `results/report_tables/report_basket_summary.csv`
- `results/report_tables/report_pairwise_error_overlap.csv`
- `results/report_tables/report_effect_sizes_vs_rf.csv`
- `results/report_tables/report_feature_top10_overlap_jaccard.csv`

No new plot/table is directly inserted in this section.

## 10. Limitations

Uses project-plan comparison against available artifacts:

- `project_plan.md`
- `left.md`
- `results/report_tables/report_feature_top15_long.csv`

Important limitation noted:

- The project plan mentions permutation importance and retraining ablations.
- Current artifacts only include built-in feature importance.
- Therefore the report does not claim permutation-importance or ablation results.

No plot/table is directly inserted in this section.

## 11. Conclusion

Uses synthesized final claims from:

- `results/report_tables/report_model_metrics.csv`
- `results/report_tables/report_confusion_decomposition.csv`
- `results/report_tables/report_basket_summary.csv`
- `results/report_tables/report_pairwise_error_overlap.csv`

No new plot/table is directly inserted in this section.

## Full Plot Inventory

Report plots available in `plots/report/`:

- `01_model_metrics.png`
- `02_error_rates.png`
- `03_effect_sizes_vs_rf.png`
- `04_basket_metrics.png`
- `05_model_disagreement_overlap.png`
- `06_subgroup_f1.png`
- `07_behavioral_error_tags.png`
- `08_feature_importance_top15.png`
- `09_feature_top10_overlap.png`

## Full Report Table Inventory

Report tables available in `results/report_tables/`:

- `report_model_metrics.csv`
- `report_confusion_decomposition.csv`
- `report_effect_sizes_vs_rf.csv`
- `report_mcnemar_bonferroni.csv`
- `report_bootstrap_f1_diff.csv`
- `report_delong_roc_auc.csv`
- `report_basket_summary.csv`
- `report_pairwise_error_overlap.csv`
- `report_prediction_disagreement_matrix.csv`
- `report_disagreement_examples.csv`
- `report_subgroup_metrics.csv`
- `report_behavioral_error_tags.csv`
- `report_feature_top15_long.csv`
- `report_feature_top10_overlap_jaccard.csv`
- `report_feature_top10_shared_counts.csv`
- `report_figure_inventory.csv`

