# Principal Component Analysis (PCA)

> Compress your data while keeping the most information.

[⬅ Back to main README](../../README.md)

---

## Definition

PCA finds new axes, called **principal components**, along which the data varies the most. Keeping only the top few reduces the number of features while preserving most of the structure.

## Intuition

Imagine a cloud of points shaped like a long cigar. Its main direction is the length of the cigar; the thickness matters much less. PCA rotates your view to line up with that main direction.

## Formulas

**Steps**

1. Center the data by subtracting each feature's mean.
2. Compute the covariance matrix:

$$\Sigma = \frac{1}{m-1}X^\top X$$

3. Solve the eigenvalue problem $\Sigma v = \lambda v$.
4. Sort by eigenvalue and keep the top $k$ eigenvectors $W_k$.
5. Project the data:

$$Z = XW_k$$

**Explained variance ratio**

$$\text{EVR}_j = \frac{\lambda_j}{\sum_{i}\lambda_i}$$

## Files in this folder

- `PCA.py`: implements PCA and plots the projected data and explained variance.

## How to run

```bash
python PCA.py
```

## Expected outcome

- A lower-dimensional version of the dataset.
- A plot of explained variance to choose how many components to keep.
- A 2D visualization of high-dimensional data.

## Tips and hyperparameters

- **Standardize features** first, or large-scale features dominate.
- Pick `k` so that cumulative explained variance reaches about 90 to 95 percent.

## Strengths and limitations

**Strengths:** reduces noise, speeds up other models, enables visualization.

**Limitations:** only finds linear structure, and components are harder to interpret than original features.

---

[⬅ Back to main README](../../README.md)
