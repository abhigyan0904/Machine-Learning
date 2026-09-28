import numpy as np
from matplotlib import pyplot as plt

def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5  # 12-15 min is best
    X[:, 0] = X[:, 0] * (285 - 150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        y = -3 / (260 - 175) * t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y)


X, y = load_coffee_data()
print(X.shape, y.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X[y.flatten() == 0][:, 0], X[y.flatten() == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(X[y.flatten() == 1][:, 0], X[y.flatten() == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.title("Dummy Binary Classification Data (for Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()


# Gaussian Parameters
X_normal = X[y.flatten() == 1]
Mu = np.mean(X_normal, axis = 0)
sigma = np.std(X_normal, axis = 0)

p_x = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((X_normal - Mu) ** 2) / (2 * sigma ** 2))
Gaussian = np.prod(p_x, axis = 1)

p_x = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((X - Mu) ** 2) / (2 * sigma ** 2))
Gaussian_all = np.prod(p_x, axis = 1)

epsilon = np.percentile(Gaussian,1)
anomalies = Gaussian_all < epsilon

print("Epsilon:", epsilon)

plt.figure(figsize=(6,6))

# normal points
plt.scatter(X[~anomalies, 0], X[~anomalies, 1],
            label="Normal (Blue)", alpha=0.6)

# anomaly points
plt.scatter(X[anomalies, 0], X[anomalies, 1], label="Anomaly (Red)", marker="x")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Anomaly Detection using Gaussian Model")
plt.legend()
plt.grid(True)
plt.show()