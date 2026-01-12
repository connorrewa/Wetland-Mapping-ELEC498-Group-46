import numpy as np

data = np.load('training_data_bow_river_FINAL.npz')
print("Keys:", list(data.keys()))

for key in data.keys():
    arr = data[key]
    print(f"\n{key}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Type: {arr.dtype}")
    
    if key == 'X':
        print(f"  Has NaN: {np.isnan(arr).any()}")
        print(f"  Min: {arr.min():.4f}, Max: {arr.max():.4f}")
        print(f"  Sample: {arr[0, :5]}")
    elif key == 'y':
        unique, counts = np.unique(arr, return_counts=True)
        print(f"  Classes: {unique}")
        for c, n in zip(unique, counts):
            print(f"    Class {c}: {n:,}")

data.close()
