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

x = (x - (np.mean(x))) / np.std(x)
y_reshaped = np.reshape(y, (-1, 1))


# Cost function
def cost_function(w, x, b, y):
    m = x.shape[0]
    f_wb = 1 / ((1 + np.exp(-(np.dot(x, w) + b)))  + 1e-6)
    cost = -y * np.log(f_wb + 1e-6) - (1 - y) * np.log(1 - f_wb + 1e-6)
    return np.sum(cost) / m

#Gradient Descent:
def gradient_w(w, x, y, b):
    m = x.shape[0]
    f_wb = 1 / ((1 + np.exp(-(np.dot(x, w) + b))) + 1e-6)
    dw = np.dot(x.T, (f_wb - y))
    return dw / m

def gradient_b(w, x, y, b):
    m = x.shape[0]
    f_wb = 1 / ((1 + np.exp(-(np.dot(x, w) + b))) + 1e-6)
    db = np.sum(f_wb - y)
    return db / m

w = np.array([[0], [1]])
b = 1
alpha = 0.001
iteration = 1000
costs = []

for i in range(iteration):
    dw = gradient_w(w, x, y_reshaped, b)
    db = gradient_b(w, x, y_reshaped, b)

    w = w - alpha * dw
    b = b - alpha * db.item()

    f_wb = 1 / ((1 + np.exp(-(np.dot(x, w) + b))) + 1e-6)
    cost = -y * np.log(f_wb + 1e-6) - (1 - y) * np.log(1 - f_wb + 1e-6)
    costs.append(np.mean(cost))

#Plotting:
plt.plot(costs, color = 'red', label = 'costs')
plt.xlabel("iteration")
plt.ylabel("cost")
plt.legend()
plt.grid(True)
plt.show()

x1 = (-b - (x[:, 1] * w[1])) / w[0]

plt.plot(x1, x[:, 1] , color = 'blue', label = 'fit line')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(-2,4)
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.legend()
plt.grid(True)
plt.show()



# --- Decision Boundary (only new part) ---
x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 200),
    np.linspace(x2_min, x2_max, 200)
)

grid = np.c_[xx1.ravel(), xx2.ravel()]

# Predict probability for each point
z = 1 / (1 + np.exp(-(np.dot(grid, w) + b)))
z = z.reshape(xx1.shape)

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx1, xx2, z, levels=[0, 0.5, 1], alpha=0.3, cmap='coolwarm')
plt.contour(xx1, xx2, z, levels=[0.5], colors='green', linewidths=2)

# Original data
plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], color='red', label='Class 0', alpha=0.6)
plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], color='blue', label='Class 1', alpha=0.6)

plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()