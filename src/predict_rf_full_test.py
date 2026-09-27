from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "rf"
MODEL_PATH = ROOT / "models" / "rf" / "rf_best.pkl"

model = joblib.load(MODEL_PATH)
threshold = json.loads((ROOT / "results" / "optimal_thresholds.json").read_text())["rf"]
feature_cols = (DATA / "feature_cols.txt").read_text().splitlines()
cols = ["user_id", "product_id", "label"] + feature_cols

pred_path = OUT / "final_full_predictions.csv"
metrics_path = OUT / "final_full_metrics.json"

ys = []
ps = []
first = True

pf = pq.ParquetFile(DATA / "test.parquet")
for batch in pf.iter_batches(batch_size=100_000, columns=cols):
    df = batch.to_pandas()
    prob = model.predict_proba(df[feature_cols])[:, 1]
    pred = (prob >= threshold).astype(int)

    out = df[["user_id", "product_id", "label"]].copy()
    out["prob"] = np.round(prob, 6)
    out["pred"] = pred
    out.to_csv(pred_path, mode="w" if first else "a", header=first, index=False)
    first = False

    ys.append(df["label"].to_numpy())
    ps.append(prob)

y = np.concatenate(ys)
p = np.concatenate(ps)
pred = (p >= threshold).astype(int)

metrics = {
    "threshold": float(threshold),
    "accuracy": float(accuracy_score(y, pred)),
    "f1": float(f1_score(y, pred, zero_division=0)),
    "precision": float(precision_score(y, pred, zero_division=0)),
    "recall": float(recall_score(y, pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y, p)),
    "pr_auc": float(average_precision_score(y, p)),
    "n_pos_pred": int(pred.sum()),
    "n_true_pos": int(y.sum()),
    "n_test": int(len(y)),
    "model": "rf",
    "tag": "final_full",
    "prediction_source": "full_test_streamed",
}

metrics_path.write_text(json.dumps({"metrics": metrics}, indent=2))
print(json.dumps(metrics, indent=2))