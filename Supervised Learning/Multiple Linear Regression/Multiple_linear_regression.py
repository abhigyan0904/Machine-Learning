import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)

# Number of samples and features
num_samples = 100
num_features = 3

# Generate random x values (features)
x_train = 2 * np.random.rand(num_samples, num_features)

# Define true weights (slopes) and intercept
true_weights = np.array([[3], [2], [1]])  # shape: (3, 1)
true_intercept = 4

# Generate noise
noise = np.random.randn(num_samples, 1)

# Compute y values
y_train = true_intercept + x_train @ true_weights + noise  # @ is matrix multiplication

# Plotting using only first feature for visualization
plt.figure(figsize = (8, 5))
plt.scatter(x_train[:, 0], y_train, color = "green", label = "Training data (vs 1st feature)", alpha = 0.7)
plt.title("Dummy Linear Regression Data (3 features)")
plt.xlabel("x_train[:, 0] (1st feature)")
plt.ylabel("y_train")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape is:", x_train.shape)
print("Target shape is:", y_train.shape)


# Vectorization:

w = np.array([[1], [2], [3]])
b = 0
vect = np.dot(x_train, w) + b


#Cost Function:
m = x_train.shape[0]
cost = ((vect - y_train) ** 2) / (2 * m)
print(w.shape)


# Gradient of w:
def gradient_w(w, x_train, y_train, b):
    m = x_train.shape[0]
    dw = np.dot(x_train.T, ((np.dot(x_train, w) + b)- y_train))
    return dw / m


#Gradient of b:
def gradient_b(w, x_train, y_train, b):
    m = x_train.shape[0]
    db = np.sum((np.dot(x_train, w) + b) - y_train)
    return db / m


alpha = 0.1
iteration = 100
costs = []
for i in range(iteration):
     print(w,b)
     dw = gradient_w(w, x_train, y_train, b)
     db = gradient_b(w, x_train, y_train, b)
     w = w - alpha * dw
     b = b - alpha * db

     cost = np.sum(((np.dot(x_train, w) + b - y_train) ** 2)) / (2 * m)
     costs.append(cost)

plt.plot( costs, color = "m", label = "costs")
plt.xlabel("iteration")
plt.ylabel("cost")
plt.legend()
plt.grid(True)
plt.show()