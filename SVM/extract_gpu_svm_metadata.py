"""
Extract performance metadata from svm_gpu_wetland_model_20260124_185210.pkl
by reloading the model and re-evaluating it on the same random-split test set
used during training (same random_state=42, test_size=0.2).

Outputs: svm_gpu_wetland_model_20260124_185210_metadata.json
"""

import os
import json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "svm_gpu_wetland_model_20260124_185210.pkl")
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "wetland_dataset_1.5M_4Training.npz")
OUT_PATH   = os.path.join(SCRIPT_DIR, "svm_gpu_wetland_model_20260124_185210_metadata.json")

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading dataset …")
data = np.load(DATA_PATH)
X = data["X"]
y = data["y"]
class_weights_arr = data["class_weights"]
data.close()
print(f"  X shape: {X.shape},  y shape: {y.shape}")

# ── Recreate the exact same train/test split used during training ─────────────
print("Splitting data (test_size=0.2, random_state=42) …")
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Test set size: {len(y_test):,}")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model …")
model = joblib.load(MODEL_PATH)
print("  Model loaded:", type(model))

# ── Predict ───────────────────────────────────────────────────────────────────
print("Running predictions on test set … (this may take a minute)")
y_pred = model.predict(X_test)

# ── Metrics ───────────────────────────────────────────────────────────────────
print("Computing metrics …")
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
prec_w, rec_w, f1_w, _         = precision_recall_fscore_support(y_test, y_pred, average="weighted")
cm = confusion_matrix(y_test, y_pred)

n_classes = len(np.unique(y))
class_weight_dict = {i: float(class_weights_arr[i]) for i in range(n_classes)}

print(f"\nOverall Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Weighted F1      : {f1_w:.4f}")

# ── Build metadata dict ───────────────────────────────────────────────────────
metadata = {
    "timestamp": "20260124_185210",
    "model_file": "svm_gpu_wetland_model_20260124_185210.pkl",
    "model_type": type(model).__name__,
    "note": "Metrics re-extracted via extract_gpu_svm_metadata.py using same 80/20 random split (seed=42).",
    "overall_metrics": {
        "accuracy":           float(accuracy),
        "precision_weighted": float(prec_w),
        "recall_weighted":    float(rec_w),
        "f1_weighted":        float(f1_w),
    },
    "per_class_metrics": {
        str(i): {
            "precision": float(precision[i]),
            "recall":    float(recall[i]),
            "f1_score":  float(f1[i]),
            "support":   int(support[i]),
        }
        for i in range(len(support))
    },
    "confusion_matrix": cm.tolist(),
    "dataset": {
        "source":     "../wetland_dataset_1.5M_4Training.npz",
        "n_test":     int(len(y_test)),
        "n_features": int(X_test.shape[1]),
    },
    "class_weights": {str(k): v for k, v in class_weight_dict.items()},
}

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUT_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅  Metadata saved to:\n    {OUT_PATH}")
