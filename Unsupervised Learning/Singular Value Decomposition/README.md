# Singular Value Decomposition (SVD)

> Break any matrix into three simpler pieces.

[⬅ Back to main README](../../README.md)

---

## Definition

SVD factorizes any matrix into rotations and a scaling: $A = U\Sigma V^\top$. The singular values in `Σ` tell you how important each direction is, so you can keep only the strongest ones.

## Intuition

Any matrix transformation can be seen as: rotate, stretch along the axes, rotate again. SVD hands you exactly those three steps, and dropping the smallest stretches gives a compressed approximation.

## Formulas

**Decomposition**

$$A = U\Sigma V^\top$$

- $U$: left singular vectors (orthogonal)
- $\Sigma$: diagonal matrix of singular values, sorted from largest to smallest
- $V^\top$: right singular vectors (orthogonal)

**Low-rank approximation** (best rank-$k$ approximation)

$$A_k = U_k\Sigma_kV_k^\top$$

**Link to PCA** (for a centered data matrix with $m$ rows)

$$\lambda_j = \frac{\sigma_j^2}{m-1}$$

## Files in this folder

- `SVD.py`: computes the decomposition and reconstructs the matrix using the top `k` singular values.

## How to run

```bash
python SVD.py
```

## Expected outcome

- The matrices `U`, `Σ`, and `Vᵀ`.
- A reconstruction that gets closer to the original as `k` grows.
- Storage savings from keeping only a few components.

## Tips and hyperparameters

- Choose `k` from the singular value spectrum (look for a sharp drop).
- Used for image compression, noise reduction, and recommender systems.

## Strengths and limitations

**Strengths:** works on any matrix and is numerically stable.

**Limitations:** can be costly for very large matrices, and it only captures linear structure.

---

[⬅ Back to main README](../../README.md)
