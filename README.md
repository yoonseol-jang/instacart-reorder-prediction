# Next-Basket Reorder Prediction & Controlled Error Analysis

Controlled benchmark of five next-basket reorder models on the Instacart Market Basket dataset, with a focus on how evaluation metric choice affects model selection decisions.

## Summary

- **Dataset:** 3.4M orders from 206K users (Instacart Market Basket Analysis)
- **Models:** Logistic Regression, Random Forest, AdaBoost, XGBoost, Recency Baseline
- **Feature set:** 37 shared features across all models (user history, product frequency, recency, department/aisle signals)
- **Validation:** User-grouped 5-fold cross-validation (prevents user leakage)
- **Best model:** Random Forest — item-level F1 0.57, ROC-AUC 0.91
- **Key finding:** The top-performing model changed between item-level and basket-level evaluation — model rankings reverse depending on evaluation granularity

## Project Structure

```
src/              Training scripts and feature engineering
  features.py     Feature construction pipeline
  train_*.py      Per-model training scripts
  eda.ipynb       Exploratory data analysis
  results.ipynb   Results aggregation and analysis
slurm/            SLURM job scripts (NYU HPC cluster)
results/          Metrics, thresholds, model outputs
plots/            EDA and evaluation visualizations
report/           Final report (PDF + LaTeX source)
```

## Setup

Data is not included. Download from [Kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis) and place CSVs in `data/`.

```bash
pip install -r requirements.txt  # numpy, pandas, scikit-learn, xgboost, matplotlib
python src/features.py           # build feature matrix
python src/train_rf.py           # train random forest (repeat for other models)
```

## Results

| Model | Item F1 | ROC-AUC | Basket F1 |
|---|---|---|---|
| Random Forest | **0.57** | **0.91** | — |
| Logistic Regression | 0.54 | 0.88 | — |
| XGBoost | 0.56 | 0.90 | — |
| AdaBoost | 0.53 | 0.87 | — |
| Recency Baseline | 0.49 | 0.82 | — |

Tuning via grid search, randomized search, and Optuna under user-grouped CV. Behavioral error taxonomy used to classify disagreements across models.

## Tech Stack

`Python` `scikit-learn` `XGBoost` `pandas` `NumPy` `Optuna` `Matplotlib`
