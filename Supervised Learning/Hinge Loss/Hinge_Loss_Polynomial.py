import numpy as np
import matplotlib.pyplot as plt

# Load data
def load_coffee_data():
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5
    X[:, 0] = X[:, 0] * (285 - 150) + 150
    Y = np.zeros(len(X))
    for i, (t, d) in enumerate(X):
        y = -3/(260-175)*t + 21
        Y[i] = 1 if (175 < t < 260 and 12 < d < 15 and d <= y) else 0
    return X, Y.reshape(-1, 1)

# Polynomial feature expansion for degree 2
def polynomial_features(X):
    x1 = X[:, 0].reshape(-1, 1)
    x2 = X[:, 1].reshape(-1, 1)
    x1_sq = x1 ** 2
    x2_sq = x2 ** 2
    x1_x2 = x1 * x2
    return np.hstack([x1, x2, x1_sq, x2_sq, x1_x2])

# Load and normalize
X, y = load_coffee_data()
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std
y = y * 2 - 1  # Convert labels to {-1, 1}

# Apply polynomial feature expansion
X_poly = polynomial_features(X_norm)

# Initialize
W = np.random.randn(X_poly.shape[1], 1)
b = 0.0
C = 1.0
lr = 0.01
epochs = 1000
costs = []

# Training loop
for epoch in range(epochs):
    margin = y * (X_poly @ W + b)
    loss = 0.5 * np.sum(W**2) + C * np.sum(np.maximum(0, 1 - margin))
    indicator = (margin < 1).astype(float)
    dW = W - C * (X_poly.T @ (indicator * y))
    db = -C * np.sum(indicator * y)
    W -= lr * dW
    b -= lr * db
    costs.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Prediction
y_pred = np.sign(X_poly @ W + b)
acc = np.mean(y_pred.flatten() == y.flatten())
print("Final Accuracy:", acc * 100, "%")

# Decision boundary plotting
xx, yy = np.meshgrid(
    np.linspace(X_norm[:, 0].min() - 1, X_norm[:, 0].max() + 1, 500),
    np.linspace(X_norm[:, 1].min() - 1, X_norm[:, 1].max() + 1, 500)
)
grid = np.c_[xx.ravel(), yy.ravel()]
grid_poly = polynomial_features(grid)
Z = np.sign(grid_poly @ W + b).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
plt.scatter(X_norm[y[:, 0] == -1][:, 0], X_norm[y[:, 0] == -1][:, 1], color='blue', label="Class 0")
plt.scatter(X_norm[y[:, 0] == 1][:, 0], X_norm[y[:, 0] == 1][:, 1], color='red', label="Class 1")
plt.title("SVM Decision Boundary with Polynomial Kernel (Degree=2)")
plt.xlabel("Normalized Temperature")
plt.ylabel("Normalized Duration")
plt.legend()
plt.grid(True)
plt.show()

# Cost plot
plt.plot(costs)
plt.title("Cost Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.grid(True)
plt.show()