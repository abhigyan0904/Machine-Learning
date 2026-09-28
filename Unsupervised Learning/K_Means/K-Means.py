import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Set seed
np.random.seed(42)

# Generate a classification dataset
x, y = make_classification(
    n_samples=200,
    n_features=2,          # Only 2 useful features for visualization
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,         # Low separation makes it harder
    flip_y=0.1,            # Add noise (10% label flipping)
    random_state=42
)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.title("Dummy Binary Classification Data (for Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape:", x.shape)
print("Target shape:", y.shape)

np.random.seed(None)

# Initialize centroids
K = 4
centroids_indices = np.random.choice(np.arange(0,x.shape[0]), K, replace = True)
centroids = x[centroids_indices]

#Assign each point a centroid
def distance(x, centroids):
    distance = np.sum(np.sqrt((x - centroids) ** 2), axis = 1)
    return distance

for _ in range(100):
    data_belongs_to = []
    cost = 0
    for i_data in x:
        i_data = i_data.reshape(1,-1)
        distances = distance(i_data, centroids)
        min_d_index = np.argmin(distances)
        data_belongs_to.append(min_d_index)
        cost = cost + distances[min_d_index]
    cost = cost / x.shape[0]

    for i in range(K):
        indices = np.where(np.array(data_belongs_to) == i)[0]
        centroids[i] = np.mean(x[indices], axis=0)


plt.figure(figsize=(8, 6))
for k in range(K):
    cluster_points = x[np.array(data_belongs_to) == k]
    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        label=f"Cluster {k}"
    )

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.title("K-Means Clustering")
plt.show()