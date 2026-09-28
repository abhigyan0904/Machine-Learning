# K-Means Clustering

> Group unlabeled data into K clusters.

[⬅ Back to main README](../../README.md)

---

## Definition

K-Means partitions data into `K` clusters, where every point belongs to the cluster with the nearest center (centroid).

## Intuition

Drop `K` pins on a map, assign each point to its closest pin, then move each pin to the middle of its group. Repeat until the pins stop moving.

## Formulas

**Objective (within-cluster sum of squares)**

$$J = \sum_{k=1}^{K}\sum_{x_i \in C_k}\|x_i - \mu_k\|^2$$

**Centroid update**

$$\mu_k = \frac{1}{|C_k|}\sum_{x_i \in C_k}x_i$$

**Algorithm**

1. Choose `K` initial centroids.
2. Assign each point to its nearest centroid.
3. Recompute each centroid as the mean of its points.
4. Repeat 2 and 3 until assignments stop changing.

## Files in this folder

- `K-Means.py`: implements K-Means and plots clusters with centroids.

## How to run

```bash
python K-Means.py
```

## Expected outcome

- A cluster label for every point.
- Final centroids marked on the plot.
- An elbow plot of inertia vs. `K` (if included).

## Tips and hyperparameters

- **Choosing `K`:** use the elbow method or silhouette score.
- **Initialization:** run multiple times with different seeds (or use K-Means++).
- **Scale features** first.

## Strengths and limitations

**Strengths:** simple, fast, scales well.

**Limitations:** needs `K` in advance, assumes round, similar-sized clusters, sensitive to outliers and initialization.

---

[⬅ Back to main README](../../README.md)
