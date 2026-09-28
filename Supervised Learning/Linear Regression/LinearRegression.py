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

# Linear Regression:
def linear_regression(w, x, b):
    m = x.shape[0]
    y_hat = np.zeros(m)
    for i in range(m):
        y_hat[i] = w * x[i][0] + b
    return y_hat

# Applying Cost Function
def cost_function(w, x, b, y):
    m = x.shape[0]
    total_cost_sum = 0
    for i in range(m):
        f_wb = w * x[i][0] + b
        cost = (f_wb - y[i][0]) ** 2
        total_cost_sum += cost
    return total_cost_sum / (2 * m)

#Gradient of w and b:
def gradient_w(w, x, y, b):
    m = x.shape[0]
    dw = 0
    for i in range(m):
        dw += ((((w * x[i][0]) + b) - y[i][0]) * x[i][0])
    return dw / m


def gradient_b(w, x, y, b):
    m = x.shape[0]
    db = 0
    for i in range(m):
        db += (((w * x[i][0]) + b) - y[i][0])
    return db / m

w = 0
b = 0
alpha = 0.8
iteration = 100
costs = []
w_list = []
b_list = []
for i in range(iteration):
     dw = gradient_w(w, x_train, y_train, b)
     db = gradient_b(w, x_train, y_train, b)
     w = w - alpha * dw
     b = b - alpha * db
     costs.append(cost_function(w, x_train, b, y_train))
     w_list.append(w)
     b_list.append(b)

# Plotting Cost Function
plt.plot(costs, color = "y", label = "Cost")
plt.xlabel("iterations")
plt.ylabel("cost")
plt.legend()
plt.grid(True)
plt.show()

# Plotting Linear Regression:
plt.scatter(x_train, y_train, color = "blue", label = "Training data", alpha = 0.7)
plt.plot(x_train, linear_regression(w, x_train, b), color = "m", label = "My Prediction")
plt.title("Dummy Linear Regression Data")
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.legend()
plt.grid(True)
plt.show()

# w vs Costs
plt.plot(w_list, costs, color = "m", label = "w vs costs")
plt.xlabel("w")
plt.ylabel("cost")
plt.legend()
plt.grid(True)
plt.show()

#3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(w_list, b_list, costs, color='purple', marker='o', label='3D Line')
ax.set_xlabel('W_List')
ax.set_ylabel('B_ List')
ax.set_zlabel('Costs')
ax.set_title('3D Line Plot Example')
ax.legend()
plt.show()