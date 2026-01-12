import numpy as np
import os

output_lines = []

def log(msg):
    print(msg)
    output_lines.append(msg)

log("="*70)
log("COMPARING DATASETS")
log("="*70)

# Check both files
files = ['wetland_dataset_1.5M.npz', 'training_data_bow_river_FINAL.npz']

for filename in files:
    if not os.path.exists(filename):
        log(f"\n⚠ File not found: {filename}")
        continue
    
    log(f"\n{'='*70}")
    log(f"FILE: {filename}")
    log(f"{'='*70}")
    
    data = np.load(filename)
    log(f"Keys: {list(data.keys())}")
    
    for key in data.keys():
        arr = data[key]
        log(f"\n{key}:")
        log(f"  Shape: {arr.shape}")
        log(f"  Type: {arr.dtype}")
        log(f"  Size: {arr.nbytes / (1024**2):.2f} MB")
        
        if key == 'X':
            log(f"  Has NaN: {np.isnan(arr).any()}")
            log(f"  Has Inf: {np.isinf(arr).any()}")
            log(f"  Min: {arr.min():.4f}, Max: {arr.max():.4f}, Mean: {arr.mean():.4f}")
            zero_samples = (arr == 0).all(axis=1).sum()
            log(f"  All-zero samples: {zero_samples:,} ({100*zero_samples/arr.shape[0]:.2f}%)")
            
        elif key == 'y':
            unique, counts = np.unique(arr, return_counts=True)
            log(f"  Unique classes: {unique}")
            log(f"\n  Class Distribution:")
            for c, n in zip(unique, counts):
                pct = 100 * n / arr.shape[0]
                log(f"    Class {c}: {n:,} samples ({pct:.2f}%)")
                
        elif key == 'class_weights':
            log(f"  Class Weights:")
            for i, w in enumerate(arr):
                log(f"    Class {i}: {w:.4f}")
    
    data.close()

# Write to file
with open('dataset_comparison.txt', 'w') as f:
    f.write('\n'.join(output_lines))

log(f"\n✓ Full report saved to dataset_comparison.txt")
