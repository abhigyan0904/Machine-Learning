import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Set seed
np.random.seed(42)

# Generate a classification dataset
X, y = make_classification(
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
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="red", label="Class 0", alpha=0.6)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue", label="Class 1", alpha=0.6)
plt.title("Dummy Binary Classification Data (for Logistic Regression)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

print("Features shape:", X.shape)
X = (X - (np.mean(X))) / np.std(X)
y = np.reshape(y, (-1, 1))
print("Target shape:", y.shape)

#Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#ReLU Function
def reLU(z):
    return np.maximum(0, z)

#Softmax
def softmax(z):
    z_j = np.exp(z - np.max(z))
    a = z_j / np.sum(z_j)
    return a

#Diffrentiation
def diff_sigmoid(z):
    return z * (1 - z)

def diff_reLU(z):
    return (z > 0).astype(float)

architecture = [512,128, 1]
activation = [reLU, reLU, sigmoid]

input_features = X.shape[1]
W = []
B = []
m = X.shape[0]
for i in architecture:
    layer_w = np.random.randn(input_features, i)/np.sqrt(i)
    layer_b = np.zeros(i)
    input_features = i
    W.append(layer_w)
    B.append(layer_b)

def forward_prop(W, B, x, activation):
    inputs = x
    activation_outputs = []
    activation_outputs.append(x)
    z_s = []

    for i in range(len(W)):
        z = np.dot(inputs, W[i]) + B[i]
        a = activation[i](z)
        inputs = a

        activation_outputs.append(a)
        z_s.append(z)


    return inputs, z_s, activation_outputs

def back_prop(activation_outputs, z_s, y, W, activation):
    dw = []
    db = []

    W_rev = W[::-1]
    A_rev = activation_outputs[::-1]
    act_rev = activation[::-1]

    # Output layer derivative
    dZ = (A_rev[0] - y ) / m                    # shape (m, 1)
    dW = A_rev[1].T @ dZ                  # (n_prev, n_curr)
    db.append(np.sum(dZ, axis=0))         # (n_curr,)
    dw.append(dW)

    for i in range(1, len(A_rev) - 1):
        dZ = (dZ @ W_rev[i-1].T)

        if act_rev[i] == reLU:
            dZ *= diff_reLU(A_rev[i])
        else:
            dZ *= diff_sigmoid(A_rev[i])

        dW = A_rev[i+1].T @ dZ
        dw.append(dW)
        db.append(np.sum(dZ, axis=0))

    dw = dw[::-1]
    db = db[::-1]
    return dw, db

loss = []

alpha = 0.001

for epoch in range(2500):
    a,z_s,activation_output = forward_prop(W, B, X, activation)
    # print(activation_output)
    dw,db = back_prop(activation_output,z_s,y,W,activation)

    cost = np.sum(-y * np.log(a) - (1 - y) * np.log(1 - a)) / m
    loss.append(cost)
    for i in range(len(W)):
        W[i] = W[i]- (alpha * dw[i])
        B[i] = B[i] -  (alpha * db[i])
    print(f" cost : {cost} , epoch : {epoch+1}")

plt.plot(loss)
plt.show()

print("\nShapes of trained W matrices:")
for i, w_matrix in enumerate(W):
    print(f"W[{i}].shape: {w_matrix.shape}")



def plot_decision_boundary(X, y, W, B, activation):
    # Create a mesh grid
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # Flatten grid → forward pass → predictions
    grid = np.c_[xx.ravel(), yy.ravel()]
    a, _, _ = forward_prop(W, B, grid, activation)
    preds = (a > 0.5).astype(int).reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, cmap="coolwarm", alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y.reshape(-1), cmap="coolwarm", edgecolor="k")
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_decision_boundary(X, y, W, B, activation)