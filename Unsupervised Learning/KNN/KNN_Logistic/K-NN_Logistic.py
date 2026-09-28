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

x = (x - (np.mean(x))) / np.std(x)
y_reshaped = np.reshape(y, (-1, 1))


test_point = 5
random_points_index = np.random.choice(np.arange(0,x.shape[0]), test_point, replace = True)
random_points = x[random_points_index]
random_points_prediction = []

K = 3
for i in range(test_point):
    distance = np.sum(np.sqrt((x - random_points[[i]]) ** 2), axis = 1)
    indexes = np.argsort(distance)[:K]
    values = y[indexes]
    print(values.shape)
    unique_values = np.unique(values)
    votes = 0
    for unique_value in unique_values:
        temp_vote = np.sum(values == unique_value)
        if temp_vote > votes:
            votes = temp_vote
            class_ = unique_value
    random_points_prediction.append(class_)

print(random_points, random_points_prediction)

# Plot training data
plt.figure(figsize=(8, 6))
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1],
            color="orange", label="Class 0 (train)", alpha=0.5)
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1],
            color="lightblue", label="Class 1 (train)", alpha=0.5)

# Convert predictions to numpy array
random_points_prediction = np.array(random_points_prediction)

# Plot predicted test points
plt.scatter(random_points[random_points_prediction == 0][:, 0],
            random_points[random_points_prediction == 0][:, 1],
            color="darkred", edgecolors="black",
            marker="X", s=120, label="Predicted Class 0")

plt.scatter(random_points[random_points_prediction == 1][:, 0],
            random_points[random_points_prediction == 1][:, 1],
            color="darkblue", edgecolors="black",
            marker="X", s=120, label="Predicted Class 1")

plt.title("KNN Classification (Train Data + Test Predictions)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
