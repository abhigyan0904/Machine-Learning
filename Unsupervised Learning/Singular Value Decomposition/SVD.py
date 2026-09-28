# pca using single value decomposition
import numpy as np
#Step 1: Center the data ---
np.random.seed(0)
X = np.random.randn(100, 3)
X_centered = X - np.mean(X, axis=0)

# --- Step 2: SVD decomposition ---
# X_centered = U Σ V^T
U, S, Vt = np.linalg.svd(X_centered)

# --- Step 3: Choose top k principal components ---
k = 2
W = Vt.T[:, :k]  # Each column is a principal component

# --- Step 4: Project the data onto top k components ---
X_reduced = X_centered @ W
print(f"data shape:{X_centered.shape} Reduced Shape: {X_reduced.shape}")

# --- Step 5: Reconstruct Data ---
X_approx = X_reduced @ W.T + np.mean(X, axis=0)
print("Reconstructed shape:", X_approx.shape)