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
np.random.seed(None)

test_point = 5
random_points_index = np.random.choice(np.arange(0,x_train.shape[0]), test_point, replace = True)
random_points = x_train[random_points_index]
random_points_prediction = []

K = 3
for i in range(test_point):
    distance = np.sum(np.sqrt((x_train - random_points[[i]]) ** 2), axis = 1)
    indexes = np.argsort(distance)[:K]
    avg = np.mean(y_train[indexes])
    random_points_prediction.append(avg)
print(random_points, random_points_prediction)

plt.figure(figsize = (8, 5))
plt.scatter(x_train, y_train, color = "blue", label = "Training data", alpha = 0.7)
plt.title("Dummy Linear Regression Data")
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.legend()
plt.grid(True)
plt.scatter(random_points,random_points_prediction, color = "red")
plt.show()