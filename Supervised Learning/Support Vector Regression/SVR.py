import numpy as np
import matplotlib.pyplot as plt


# Set the seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 100

# Generate random x values (features)
x_train = 2 * np.random.rand(num_samples, 1)

# Generate corresponding y values with a linear relationship (y = 4 + 3x + noise)
true_slope = 3
true_intercept = 4
noise = np.random.randn(num_samples, 1)

y_train = true_intercept + true_slope * x_train + noise

# Plotting the data
plt.figure(figsize = (8, 5))
plt.scatter(x_train, y_train, color = "blue", label = "Training data", alpha = 0.7)
plt.title("Dummy Linear Regression Data")
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape is:", x_train.shape)
print("Target shape is:", y_train.shape)

# SVR hyperparameters
epsilon = 0.5     # ε-tube width
C = 1.0           # penalty
learning_rate = 0.01
epochs = 1000

# Initialize model parameters
w = 0.0
b = 0.0

for epoch in range(epochs):

    dw = 0.0
    db = 0.0

    for i in range(num_samples):
        x_i = x_train[i][0]
        y_i = y_train[i][0]

        y_pred = w * x_i + b
        error = y_pred - y_i

        # Case 1: inside ε-tube → no loss
        if abs(error) <= epsilon:
            continue

        # Case 2: above ε-tube
        elif error > epsilon:
            dw += C * x_i
            db += C

        # Case 3: below ε-tube
        elif error < -epsilon:
            dw -= C * x_i
            db -= C

    # Add regularization gradient
    dw += w

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

def predict(x):
    return w * x + b

x_test = np.linspace(0, 2, 100).reshape(-1, 1)
y_pred = predict(x_test)

plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, color="blue", alpha=0.6, label="Training data")
plt.plot(x_test, y_pred, color="red", linewidth=2, label="SVR fit")

# ε-tube
plt.plot(x_test, y_pred + epsilon, "r--", linewidth=1)
plt.plot(x_test, y_pred - epsilon, "r--", linewidth=1)

plt.xlabel("x")
plt.ylabel("y")
plt.title("SVR from Scratch (ε-insensitive hinge loss)")
plt.legend()
plt.grid(True)
plt.show()

print("Learned weight (w):", w)
print("Learned bias (b):", b)