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
    lam = 5
    f_wb = 1 / ((1 + np.exp(-(np.dot(x, w) + b)))  + 1e-6)
    cost = -y * np.log(f_wb + 1e-6) - (1 - y) * np.log(1 - f_wb + 1e-6)
    reg =  (lam / (2 * m)) * np.sum(w ** 2)
    return np.sum(cost) / m + reg


def gradient_w(w, x, y, b):
    m = x.shape[0]
    lam = 5
    f_wb = 1 / ((1 + np.exp(-(np.dot(x, w) + b))) + 1e-6)
    dw = np.dot(x.T, (f_wb - y))
    reg_grad =  (lam / m) * w
    dw = dw + reg_grad
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