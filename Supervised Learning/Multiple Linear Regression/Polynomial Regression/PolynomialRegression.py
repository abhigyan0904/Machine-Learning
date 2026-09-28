import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv("polynomial_data.csv")
x_train = file["x_train"]
y_train = file["y_train"]
y_train = np.reshape(y_train, (-1, 1))

plt.scatter(x_train, y_train, label = "Polynomial Scatter", color = "red")
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.legend()
plt.show()


# polynomial feature creation
x = x_train.values.reshape(-1, 1)
x_mean = np.mean(x)
x_std = np.std(x)
x_train = (x - x_mean) / x_std

temp_poly_x = [x_train] + [(x_train ** 2)] + [(x_train ** 3)] + [(x_train ** 4)] + [(x_train ** 5)] + [(x_train ** 6)]

poly_x_train = np.hstack((temp_poly_x))

# Applying polynomial regression:
w = np.array([[0], [0], [0], [0], [0], [0]])
b = 0

#Vectorization:
f_wb = np.dot(poly_x_train, w) + b

#Cost Function:
m = poly_x_train.shape[0]
cost = ((f_wb - y_train) ** 2) / (2 * m)

 #Gradient of w:
def gradient_w(w, poly_x_train, y_train, b):
    m = poly_x_train.shape[0]
    dw = np.dot(poly_x_train.T, ((np.dot(poly_x_train, w) + b) - y_train))
    return dw / m


# Gradient of b:
def gradient_b(w, poly_x_train, y_train, b):
    m = poly_x_train.shape[0]
    db = np.sum((np.dot(poly_x_train, w) + b) - y_train)
    return db / m

alpha = 0.02
iteration = 50000
costs = []
for i in range(iteration):
     print(w,b)
     dw = gradient_w(w, poly_x_train, y_train, b)
     db = gradient_b(w, poly_x_train, y_train, b)
     w = w - alpha * dw
     b = b - alpha * db

     cost = np.sum(((np.dot(poly_x_train, w) + b - y_train) ** 2)) / (2 * m)
     costs.append(cost)

y_pred = np.dot(poly_x_train, w) + b

# Plotting:
plt.scatter(x_train, y_train, label = "Polynomial Scatter", color = "red")
plt.plot(x_train, y_pred, label = "Polynomial Scatter", color = "blue")
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.legend()
plt.show()

plt.plot( costs, color = "m", label = "costs")
plt.xlabel("iteration")
plt.ylabel("cost")
plt.legend()
#plt.grid(True)
plt.show()
