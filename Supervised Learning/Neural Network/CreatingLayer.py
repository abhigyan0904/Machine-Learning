import numpy as np
import matplotlib.pyplot as plt

# 1. DATA GENERATION
def load_coffee_data():
    """ Creates a coffee roasting data set.
            roasting duration: 12-15 minutes is best
            temperature range: 175-260C is best
        """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5  # 12-15 min is best
    X[:, 0] = X[:, 0] * (285 - 150) + 150  # 175-260 C is best
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        y = -3 / (260 - 175) * t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1
    return (X, Y.reshape(-1, 1))


# Load Data
X, y = load_coffee_data()

print(f"Original X shape: {X.shape}, y shape: {y.shape}")

# 2. NORMALIZATION (The Fix)
# We must normalize features so they are on the same scale (approx -1 to 1)
print("Normalizing data...")
norm_l = np.mean(X, axis=0)
norm_s = np.std(X, axis=0)

# Xt is the Normalized Training Data
Xt = (X - norm_l) / norm_s

# 3. HELPER FUNCTIONS
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def reLU(z):
    return np.maximum(0, z)


def diff_sigmoid(z):
    return z * (1 - z)


def diff_reLU(z):
    return (z > 0).astype(float)

# 4. INITIALIZATION
architecture = [3, 1]  # 3 neurons in hidden layer, 1 in output
activation = [reLU, reLU, sigmoid]  # Activation for input (identity), hidden (ReLU), output (Sigmoid)

# Note: The activation list length usually matches layers.
# Here we treat index 1 as hidden activation, index 2 as output.

input_features = Xt.shape[1]
W = []
B = []
m = Xt.shape[0]

# Initialize weights
# We loop through architecture.
# 1st iter: Input(2) -> Hidden(3)
# 2nd iter: Hidden(3) -> Output(1)
curr_input = input_features
for i in architecture:
    # He/Xavier Initialization scale
    scale = np.sqrt(2.0 / curr_input)
    layer_w = np.random.randn(curr_input, i) * scale
    layer_b = np.zeros(i)

    W.append(layer_w)
    B.append(layer_b)
    curr_input = i

# 5. PROPAGATION
def forward_prop(W, B, x, activation_funcs):
    inputs = x
    activation_outputs = [x]  # Store A0 (Input)
    z_s = []

    # Loop through layers
    # Layer 0: Weights W[0], Bias B[0], Activation activation_funcs[1] (ReLU)
    # Layer 1: Weights W[1], Bias B[1], Activation activation_funcs[2] (Sigmoid)

    for i in range(len(W)):
        z = np.dot(inputs, W[i]) + B[i]

        # Use appropriate activation function
        # Note: Your activation list structure was [unused, relu, sigmoid]
        # So we use i+1
        act_func = activation_funcs[i + 1]
        a = act_func(z)

        inputs = a
        activation_outputs.append(a)
        z_s.append(z)

    return inputs, z_s, activation_outputs


def back_prop(activation_outputs, z_s, y, W, activation_funcs):
    dw = []
    db = []

    # We prefer iterating backwards explicitly
    # Final Layer (Sigmoid)
    A_final = activation_outputs[-1]
    A_prev = activation_outputs[-2]

    # Derivative of Cost with respect to Z (for Sigmoid + CrossEntropy)
    dZ = A_final - y

    dW_last = np.dot(A_prev.T, dZ) / m
    db_last = np.sum(dZ, axis=0) / m

    dw.append(dW_last)
    db.append(db_last)

    # Hidden Layer (ReLU)
    # We move backwards. Current dZ is passed back.
    # W[1] connects hidden to output.

    dA_hidden = np.dot(dZ, W[1].T)
    dZ_hidden = dA_hidden * diff_reLU(z_s[0])  # Derivative of ReLU

    dW_hidden = np.dot(activation_outputs[0].T, dZ_hidden) / m
    db_hidden = np.sum(dZ_hidden, axis=0) / m

    dw.append(dW_hidden)
    db.append(db_hidden)

    # Reverse lists to match W order [Hidden, Output]
    return dw[::-1], db[::-1]

# 6. TRAINING LOOP
loss = []
alpha = 0.5  # Increased learning rate because we normalized data
epochs = 30000

print(f"Starting training for {epochs} epochs...")

for epoch in range(epochs):
    # 1. Forward (Use Xt - Normalized Data)
    a, z_s, activation_output = forward_prop(W, B, Xt, activation)

    # 2. Backward
    dw, db = back_prop(activation_output, z_s, y, W, activation)

    # 3. Cost Calculation (Binary Cross Entropy)
    # Add small epsilon to avoid log(0) errors
    epsilon = 1e-15
    a_clipped = np.clip(a, epsilon, 1 - epsilon)
    cost = np.sum(-y * np.log(a_clipped) - (1 - y) * np.log(1 - a_clipped)) / m
    loss.append(cost)

    # 4. Update
    for i in range(len(W)):
        W[i] = W[i] - (alpha * dw[i])
        B[i] = B[i] - (alpha * db[i])

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Cost {cost:.4f}")

# Plot Loss
plt.figure(figsize=(6, 4))
plt.plot(loss)
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.show()

# 7. VISUALIZATION
def plot_decision_boundary(X_orig, y, W, B, activation, norm_l, norm_s):
    # Create meshgrid based on ORIGINAL scale
    x_min, x_max = X_orig[:, 0].min() - 1, X_orig[:, 0].max() + 1
    y_min, y_max = X_orig[:, 1].min() - 1, X_orig[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Flatten
    grid = np.c_[xx.ravel(), yy.ravel()]

    # NORMALIZE the grid using the stats we saved earlier
    grid_norm = (grid - norm_l) / norm_s

    # Predict
    a, _, _ = forward_prop(W, B, grid_norm, activation)
    preds = (a > 0.5).astype(int).reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, cmap="coolwarm", alpha=0.3)
    plt.scatter(X_orig[:, 0], X_orig[:, 1], c=y.flatten(), cmap="coolwarm", edgecolor="k")
    plt.title("Decision Boundary (Normalized Training)")
    plt.xlabel("Temperature")
    plt.ylabel("Duration")
    plt.show()


def plot_neuron_lines(X_orig, y, W, B, norm_l, norm_s):
    plt.figure(figsize=(8, 6))

    # 1. Plot the Data (Original Scale)
    plt.scatter(X_orig[:, 0], X_orig[:, 1], c=y.flatten(), cmap="coolwarm", edgecolor="k", alpha=0.6)

    # 2. Calculate and Plot Lines for each Hidden Neuron
    # The weights for the first layer are in W[0] and biases in B[0]
    # Shape of W[0] is (2, 3) -> 2 inputs, 3 neurons
    W1 = W[0]
    b1 = B[0]

    # Generate x-values in the NORMALIZED space (roughly -2 to 2)
    x_norm_range = np.linspace(-3, 3, 100)

    colors = ['purple', 'orange', 'green']

    for i in range(3):  # Loop through the 3 neurons
        # Get weights/bias for neuron i
        w_x = W1[0, i]  # Weight for Feature 1
        w_y = W1[1, i]  # Weight for Feature 2
        b = b1[i]  # Bias

        # Equation of the line: w_x * x + w_y * y + b = 0
        # Solve for y: y = -(w_x * x + b) / w_y

        # Calculate y in NORMALIZED space
        y_norm_line = - (w_x * x_norm_range + b) / w_y

        # 3. De-normalize points to plot them on the original graph
        # X_orig = X_norm * sigma + mu
        x_plot = x_norm_range * norm_s[0] + norm_l[0]
        y_plot = y_norm_line * norm_s[1] + norm_l[1]

        # Plot the line
        plt.plot(x_plot, y_plot, label=f"Neuron {i + 1} Boundary",
                 color=colors[i], linewidth=2, linestyle='--')

    plt.title("The 3 Linear Boundaries Learned by Hidden Layer")
    plt.xlabel("Temperature")
    plt.ylabel("Duration")

    # Set limits to match data so lines don't zoom out the graph
    plt.xlim(X_orig[:, 0].min() - 5, X_orig[:, 0].max() + 5)
    plt.ylim(X_orig[:, 1].min() - 1, X_orig[:, 1].max() + 1)

    plt.legend()
    plt.show()


# Call the function
plot_neuron_lines(X, y, W, B, norm_l, norm_s)