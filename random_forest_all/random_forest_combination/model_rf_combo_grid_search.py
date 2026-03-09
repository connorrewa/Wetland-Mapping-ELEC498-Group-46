from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import numpy as np
import joblib
import os
from datetime import datetime
import json

# ======================================
# GRID SEARCH — RF Combination (Stage 1 + 2)
# Tunes: Stage 2 (n_estimators x max_depth)
# Stage 1 Fixed: n_estimators=300, max_depth=35
# ======================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'random_forest_spatial_middle', 'wetland_dataset_middle_split.npz')

# ---- Stage 1 (Background) Fixed Parameters ----
S1_N_ESTIMATORS = 300
S1_MAX_DEPTH    = 35
S1_MIN_SAMPLES_LEAF = 20
BACKGROUND_BOOST = 2

# ---- Stage 2 (Wetland) Hyperparameter grid ----
S2_N_ESTIMATORS_OPTIONS = [100, 200, 300]
S2_MAX_DEPTH_OPTIONS    = [25, 35, None]
S2_MIN_SAMPLES_LEAF     = 20
CLASS1_DAMPEN           = 0.4

# ======================================
# LOAD DATA (done once)
# ======================================
data = np.load(DATA_PATH)
X_train = data['X_train']
y_train_raw = data['y_train']
X_test  = data['X_test']
y_test_raw  = data['y_test']
test_row_min = int(data['test_row_min'])
test_row_max = int(data['test_row_max'])
data.close()

print(f"Loaded: {DATA_PATH}")
print(f"Total samples — Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}\n")

# ======================================
# PREPARE STAGE 1 (Background) DATA & WEIGHTS
# ======================================
y_train_s1 = (y_train_raw != 0).astype(int)

s1_unique_classes, s1_counts = np.unique(y_train_s1, return_counts=True)
s1_weight_dict = {
    int(cls): float(len(y_train_s1) / (len(s1_unique_classes) * count))
    for cls, count in zip(s1_unique_classes, s1_counts)
}
s1_weight_dict[0] = s1_weight_dict[0] * BACKGROUND_BOOST

print(f"--- Stage 1 Preparation ---")
print(f"  Classes present: {s1_unique_classes.tolist()}")
print(f"  Class weights (with x{BACKGROUND_BOOST} Background Boost):")
for cls, w in s1_weight_dict.items():
    print(f"    Class {cls}: weight={w:.4f}")
print()

# ======================================
# PREPARE STAGE 2 (Wetland) DATA & WEIGHTS
# ======================================
s2_train_mask = y_train_raw != 0
X_train_s2 = X_train[s2_train_mask]
y_train_s2 = y_train_raw[s2_train_mask]

s2_unique_classes, s2_counts = np.unique(y_train_s2, return_counts=True)
s2_weight_dict = {
    int(cls): float(len(y_train_s2) / (len(s2_unique_classes) * count))
    for cls, count in zip(s2_unique_classes, s2_counts)
}
s2_weight_dict[1] = s2_weight_dict[1] * CLASS1_DAMPEN

print(f"--- Stage 2 Preparation ---")
print(f"  Data shape after filtering Background: {X_train_s2.shape[0]:,}")
print(f"  Classes present: {s2_unique_classes.tolist()}")
print(f"  Class weights (with x{CLASS1_DAMPEN} Class 1 Dampen):")
for cls, w in s2_weight_dict.items():
    print(f"    Class {cls}: weight={w:.4f}")
print()

# ======================================
# TRAIN FIXED STAGE 1 MODEL
# ======================================
print(f"{'='*60}")
print("TRAINING STAGE 1 MODEL (Fixed Parameters)")
print(f"n_estimators={S1_N_ESTIMATORS}, max_depth={S1_MAX_DEPTH}")
print(f"{'='*60}")

t_start_s1 = datetime.now()
rf_stage1 = RandomForestClassifier(
    n_estimators=S1_N_ESTIMATORS,
    max_depth=S1_MAX_DEPTH,
    min_samples_leaf=S1_MIN_SAMPLES_LEAF,
    random_state=42,
    class_weight=s1_weight_dict,
    verbose=0,
    n_jobs=-1,
)
rf_stage1.fit(X_train, y_train_s1)
t_end_s1 = datetime.now()
train_secs_s1 = (t_end_s1 - t_start_s1).total_seconds()
print(f"Stage 1 Training Time: {train_secs_s1:.1f}s\n")

# Run Stage 1 inference to cache results for the combination loop
# This saves time by not running Stage 1 inference repeatedly
print("Running Stage 1 Inference on full test set...")
preds_stage1_full = rf_stage1.predict(X_test)
wetland_mask_full = (preds_stage1_full == 1)
print(f"  -> Identified {np.sum(wetland_mask_full):,} valid wetland pixels out of {len(X_test):,}")

# Filter Stage 2 test data based on Stage 1 predictions
X_test_s2_masked = X_test[wetland_mask_full]


# ======================================
# GRID SEARCH LOOP (STAGE 2)
# ======================================
results = []
total_runs = len(S2_N_ESTIMATORS_OPTIONS) * len(S2_MAX_DEPTH_OPTIONS)
run_num = 0

labels_full = [0, 1, 2, 3, 4, 5]

for n_est in S2_N_ESTIMATORS_OPTIONS:
    for max_d in S2_MAX_DEPTH_OPTIONS:
        run_num += 1
        depth_label = str(max_d) if max_d is not None else 'None'
        print(f"\n{'='*60}")
        print(f"RUN {run_num}/{total_runs} — Stage 2: n_estimators={n_est}, max_depth={depth_label}")
        print(f"{'='*60}")

        t_start_s2 = datetime.now()

        # Train Stage 2
        rf_stage2 = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_leaf=S2_MIN_SAMPLES_LEAF,
            random_state=42,
            class_weight=s2_weight_dict,
            verbose=0,
            n_jobs=-1,
        )
        rf_stage2.fit(X_train_s2, y_train_s2)

        t_end_s2 = datetime.now()
        train_secs_s2 = (t_end_s2 - t_start_s2).total_seconds()

        # Run Stage 2 inference on the masked pixels
        final_predictions = np.zeros(X_test.shape[0], dtype=np.int32)
        if np.sum(wetland_mask_full) > 0:
            preds_stage2_masked = rf_stage2.predict(X_test_s2_masked)
            final_predictions[wetland_mask_full] = preds_stage2_masked

        # Evaluate against FULL MULTI-CLASS y_test
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_raw, final_predictions, labels=labels_full, average=None, zero_division=0
        )
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
            y_test_raw, final_predictions, labels=labels_full, average='weighted', zero_division=0
        )
        conf_matrix = confusion_matrix(y_test_raw, final_predictions, labels=labels_full)
        accuracy = accuracy_score(y_test_raw, final_predictions)
        
        # Calculate mean wetland F1 (Classes 1-5 only)
        f1_wetlands_only = f1[1:]
        mean_wetland_f1 = float(np.mean(f1_wetlands_only))

        print(f"  Accuracy:         {accuracy:.4f}")
        print(f"  Weighted F1:      {f1_avg:.4f}")
        print(f"  Mean Wetland F1:  {mean_wetland_f1:.4f}")
        print(f"  Per-class F1:     { {labels_full[i]: round(f1[i], 3) for i in range(len(labels_full))} }")
        print(f"  S2 Train time:    {train_secs_s2:.1f}s")

        # Save metadata JSON (no model saving by default to save storage on combo grids)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata_filename = f'rf_combo_S1_fixed_S2_est{n_est}_depth{depth_label}_{timestamp}_metadata.json'

        metadata = {
            'timestamp': timestamp,
            'evaluation_datetime': t_end_s2.strftime('%Y-%m-%d %H:%M:%S'),
            'pipeline_stage': 'Two-Stage Combination Pipeline (grid search)',
            'split_method': 'middle_row_band',
            'test_row_min': test_row_min,
            'test_row_max': test_row_max,
            'classes': labels_full,
            'overall_metrics': {
                'accuracy':           float(accuracy),
                'precision_weighted': float(precision_avg),
                'recall_weighted':    float(recall_avg),
                'f1_weighted':        float(f1_avg),
                'mean_wetland_f1':    mean_wetland_f1,
            },
            'per_class_metrics': {
                str(labels_full[i]): {
                    'precision': float(precision[i]),
                    'recall':    float(recall[i]),
                    'f1_score':  float(f1[i]),
                    'support':   int(support[i]),
                }
                for i in range(len(labels_full))
            },
            'confusion_matrix': conf_matrix.tolist(),
            'confusion_matrix_labels': labels_full,
            'stage1_hyperparameters': {
                'n_estimators':            S1_N_ESTIMATORS,
                'max_depth':               S1_MAX_DEPTH,
                'min_samples_leaf':        S1_MIN_SAMPLES_LEAF,
                'background_boost_factor': BACKGROUND_BOOST,
            },
            'stage2_hyperparameters': {
                'n_estimators':         n_est,
                'max_depth':            max_d,
                'min_samples_leaf':     S2_MIN_SAMPLES_LEAF,
                'class1_dampen_factor': CLASS1_DAMPEN,
            },
            'dataset': {
                'source':           '../random_forest_spatial_middle/wetland_dataset_middle_split.npz',
                'n_train':          int(X_train.shape[0]),
                'n_test':           int(X_test.shape[0]),
                'n_features':       int(X_train.shape[1]),
            },
            'stage1_train_seconds':  train_secs_s1,
            'stage2_train_seconds':  train_secs_s2,
        }

        with open(os.path.join(SCRIPT_DIR, metadata_filename), 'w') as f:
            json.dump(metadata, f, indent=2)

        results.append({
            's2_n_estimators': n_est,
            's2_max_depth':    depth_label,
            'accuracy':        round(accuracy, 4),
            'f1_weighted':     round(f1_avg, 4),
            'mean_wetland_f1': round(mean_wetland_f1, 4),
            's2_train_secs':   round(train_secs_s2, 1),
        })

# ======================================
# SUMMARY TABLE
# ======================================
print(f"\n{'='*75}")
print("GRID SEARCH SUMMARY (Combo)  (sorted by f1_weighted desc)")
print(f"{'='*75}")
results.sort(key=lambda x: x['f1_weighted'], reverse=True)
print(f"{'S2 n_est':>8}  {'S2 depth':>8}  {'accuracy':>9}  {'wt_f1':>7}  {'wet_mean_f1':>11}  {'S2 secs':>8}")
print(f"{'-'*8}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*11}  {'-'*8}")
for r in results:
    print(f"{r['s2_n_estimators']:>8}  {r['s2_max_depth']:>8}  {r['accuracy']:>9.4f}  "
          f"{r['f1_weighted']:>7.4f}  {r['mean_wetland_f1']:>11.4f}  {r['s2_train_secs']:>8.1f}s")

# Save summary JSON
summary_path = os.path.join(SCRIPT_DIR, f'grid_search_summary_combo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSummary saved to: {summary_path}")
