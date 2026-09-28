# pca using eigen values and vector
import numpy as np
# --- Step 0: Create Dummy Data ---
np.random.seed(0)
X = np.random.randn(100, 3)  # 100 samples, 3 features

# --- Step 1: Standardize ---
X_centered = X - np.mean(X, axis=0)

# --- Step 2: Covariance Matrix ---
cov = np.cov(X_centered.T)

# --- Step 3: Eigen Decomposition ---
eig_vals, eig_vecs = np.linalg.eig(cov)

# --- Step 4: Sort Eigenvectors by Eigenvalues ---
sorted_idx = np.argsort(eig_vals)[::-1]
eig_vecs = eig_vecs[:, sorted_idx]

# --- Step 5: Project to Lower Dimensions (e.g., 2D) ---
k = 2
W = eig_vecs[:, :k]
X_reduced = X_centered @ W
print(f"data shape:{X_centered.shape} Reduced Shape: {X_reduced.shape}")

# --- Step 6: Reconstruct Original Data ---
X_approx = X_reduced @ W.T + np.mean(X, axis=0)
print("Reconstructed shape:", X_approx.shape)