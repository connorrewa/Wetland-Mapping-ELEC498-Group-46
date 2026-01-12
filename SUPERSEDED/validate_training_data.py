"""
Validate training_data_bow_river_FINAL.npz
Check that embeddings and labels are properly aligned and ready for model training
"""
import numpy as np

print("="*70)
print("VALIDATING TRAINING DATA")
print("="*70)

# Load the NPZ file
data = np.load('training_data_bow_river_FINAL.npz')

print("\n1. FILE CONTENTS:")
print(f"   Available arrays: {list(data.keys())}")

# Check each array
for key in data.keys():
    arr = data[key]
    print(f"\n2. ARRAY '{key}':")
    print(f"   Shape: {arr.shape}")
    print(f"   Data type: {arr.dtype}")
    print(f"   Memory size: {arr.nbytes / (1024**2):.2f} MB")
    
    if key == 'X':  # Embeddings
        print(f"\n   EMBEDDINGS VALIDATION:")
        print(f"   - Number of samples: {arr.shape[0]:,}")
        print(f"   - Number of features (bands): {arr.shape[1]}")
        print(f"   - Min value: {arr.min():.4f}")
        print(f"   - Max value: {arr.max():.4f}")
        print(f"   - Mean value: {arr.mean():.4f}")
        print(f"   - Contains NaN: {np.isnan(arr).any()}")
        print(f"   - Contains Inf: {np.isinf(arr).any()}")
        
        # Check for all-zero samples
        zero_samples = (arr == 0).all(axis=1).sum()
        print(f"   - All-zero samples: {zero_samples:,} ({100*zero_samples/arr.shape[0]:.2f}%)")
        
        # Sample some values
        print(f"\n   Sample embedding (first 10 features):")
        print(f"   {arr[0, :10]}")
        
    elif key == 'y':  # Labels
        print(f"\n   LABELS VALIDATION:")
        print(f"   - Number of labels: {arr.shape[0]:,}")
        print(f"   - Unique classes: {np.unique(arr)}")
        print(f"   - Min class: {arr.min()}")
        print(f"   - Max class: {arr.max()}")
        
        # Class distribution
        unique, counts = np.unique(arr, return_counts=True)
        print(f"\n   CLASS DISTRIBUTION:")
        for cls, count in zip(unique, counts):
            percentage = 100 * count / len(arr)
            print(f"   - Class {cls}: {count:,} samples ({percentage:.2f}%)")
            
    elif key == 'class_weights':
        print(f"\n   CLASS WEIGHTS:")
        for i, weight in enumerate(arr):
            print(f"   - Class {i}: {weight:.4f}")

# Verify alignment
X = data['X']
y = data['y']

print(f"\n3. ALIGNMENT CHECK:")
print(f"   - X has {X.shape[0]:,} samples")
print(f"   - y has {y.shape[0]:,} labels")
print(f"   - Aligned: {'✓ YES' if X.shape[0] == y.shape[0] else '✗ NO - MISMATCH!'}")

# Data quality checks
print(f"\n4. DATA QUALITY:")
all_checks_pass = True

if np.isnan(X).any():
    print(f"   ✗ FAIL: Embeddings contain NaN values")
    all_checks_pass = False
else:
    print(f"   ✓ PASS: No NaN values in embeddings")

if np.isinf(X).any():
    print(f"   ✗ FAIL: Embeddings contain Inf values")
    all_checks_pass = False
else:
    print(f"   ✓ PASS: No Inf values in embeddings")

if X.shape[0] == y.shape[0]:
    print(f"   ✓ PASS: Embeddings and labels are aligned")
else:
    print(f"   ✗ FAIL: Shape mismatch between X and y")
    all_checks_pass = False

if X.shape[1] == 64:
    print(f"   ✓ PASS: Correct number of embedding features (64)")
else:
    print(f"   ⚠ WARNING: Expected 64 features, got {X.shape[1]}")

if set(np.unique(y)) == set(range(6)):
    print(f"   ✓ PASS: All 6 classes present (0-5)")
elif len(np.unique(y)) < 6:
    print(f"   ⚠ WARNING: Only {len(np.unique(y))} classes present, expected 6")
else:
    print(f"   ✗ FAIL: Invalid class labels detected")
    all_checks_pass = False

print(f"\n{'='*70}")
if all_checks_pass:
    print("✓ DATA VALIDATION PASSED - READY FOR MODEL TRAINING!")
else:
    print("✗ DATA VALIDATION FAILED - ISSUES NEED TO BE FIXED")
print(f"{'='*70}")

data.close()
