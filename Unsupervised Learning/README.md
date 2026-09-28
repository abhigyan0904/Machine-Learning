# Unsupervised Learning

[⬅ Back to main README](../README.md)

In unsupervised learning, the data has **no labels**. The algorithm looks for structure on its own: groups, unusual points, or simpler representations.

> **Note on organization.** Q-Learning and DQN are **reinforcement learning** methods, and K-Nearest Neighbors is an **instance-based supervised** method. They live in this folder to keep the repository layout simple, and each module's README explains its true category.

## Modules in this folder

| Module | Type | One-line summary |
|---|---|---|
| [K_Means](./K_Means) | Clustering | Group points around `K` centroids. |
| [KNN](./KNN) | Instance-based | Predict from the `k` closest training points. |
| [Principal Component Analysis](./Principal%20Component%20Analysis) | Dimensionality reduction | Keep the directions of highest variance. |
| [Singular Value Decomposition](./Singular%20Value%20Decomposition) | Matrix factorization | Split a matrix into `U`, `Σ`, and `Vᵀ`. |
| [Anomaly Detection](./Anomaly%20Detection) | Outlier detection | Flag points with very low probability. |
| [Q_Learning](./Q_Learning) | Reinforcement learning | Learn action values in a table. |
| [DQN](./DQN) | Reinforcement learning | Learn action values with a neural network. |

## Common workflow

1. Scale or normalize the features.
2. Pick the algorithm that matches the question (groups, compression, outliers, or decisions).
3. Evaluate with internal metrics (inertia, silhouette, explained variance) or a small labeled validation set.
