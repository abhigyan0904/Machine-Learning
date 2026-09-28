import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv("animal_features.csv")

# Encode categorical data
for col in file.columns:
    file[col] = file[col].astype("category").cat.codes

x_train = file[["EarShape", "FaceShape", "Whiskers"]].values
y_train = file["Animal"].values

# Root Node Entropy
def entropy(p1):
    if p1 == 0 or p1 == 1:
        return 0
    return -p1*np.log2(p1) - (1-p1)*np.log2(1-p1)

# Splitting tree
def split_tree(X, feature_index):
    left = []
    right = []
    for i in range(len(X)):
        if X[i][feature_index] == 1:
            left.append(i)
        else:
            right.append(i)
    return left, right

# Show probabilities
def show_probabilities(y, left, right):
    p_root = sum(y) / len(y)
    p_left = sum(y[left]) / len(left)
    p_right = sum(y[right]) / len(right)

    print(f"P(Cat at Root)  = {p_root:.3f}")
    print(f"P(Cat in Left)  = {p_left:.3f}")
    print(f"P(Cat in Right) = {p_right:.3f}")

# Weighted Entropy
def weighted_entropy(y, left, right):
    w_left = len(left) / len(y)
    w_right = len(right) / len(y)

    p_left = sum(y[left]) / len(left)
    p_right = sum(y[right]) / len(right)

    return w_left * entropy(p_left) + w_right * entropy(p_right)

# Information Gain
def information_gain(y, left, right):
    p_root = sum(y) / len(y)
    H_root = entropy(p_root)
    H_split = weighted_entropy(y, left, right)
    return H_root - H_split

# Run for each feature
features = ["EarShape", "FaceShape", "Whiskers"]

for i, name in enumerate(features):
    print("\nFeature:", name)
    left, right = split_tree(x_train, i)

    show_probabilities(y_train, left, right)

    ig = information_gain(y_train, left, right)
    print("Information Gain =", round(ig,4))

# Feature indices
x_feature = 0  # EarShape
y_feature = 1  # FaceShape

plt.scatter(
    x_train[:, x_feature],
    x_train[:, y_feature],
    c = y_train,       # color by label
    cmap = 'bwr',      # blue/red for 0/1
    edgecolor = 'k',
    s = 100
)

plt.xlabel("EarShape")
plt.ylabel("FaceShape")
plt.title("Scatter Plot of Animals by Features")
plt.show()

# --- Determine best root split for decision boundary ---
best_feature = None
best_ig = -1

for i in range(3):  # check all features: EarShape, FaceShape, Whiskers
    left, right = split_tree(x_train, i)
    ig = information_gain(y_train, left, right)
    if ig > best_ig:
        best_ig = ig
        best_feature = i
        best_left, best_right = left, right


# --- Decision Boundary (only new part) ---
x1_min, x1_max = x_train[:, 0].min() - 0.5, x_train[:, 0].max() + 0.5
x2_min, x2_max = x_train[:, 1].min() - 0.5, x_train[:, 1].max() + 0.5

xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 200),
    np.linspace(x2_min, x2_max, 200)
)

# Flatten grid to pass to prediction
grid = np.c_[xx1.ravel(), xx2.ravel()]

# Simple prediction for root node based on best feature
def predict_point(point):
    # Use best feature split (0=EarShape, 1=FaceShape)
    if best_feature is None:
        return 0
    if point[best_feature] == 1:
        # Predict majority class in left
        return int(round(np.mean(y_train[best_left])))
    else:
        # Predict majority class in right
        return int(round(np.mean(y_train[best_right])))

# Predict on the grid
z = np.array([predict_point(p) for p in grid])
z = z.reshape(xx1.shape)

# Plot decision boundary
plt.figure(figsize = (8,6))
plt.contourf(xx1, xx2, z, alpha = 0.3, cmap='coolwarm', levels=[-0.1,0.5,1.1])
plt.contour(xx1, xx2, z, levels = [0.5], colors = 'green', linewidths = 2)

# Plot original data
plt.scatter(x_train[y_train==0][:,0], x_train[y_train==0][:,1], color='red', label='Class 0', s=100, edgecolor='k')
plt.scatter(x_train[y_train==1][:,0], x_train[y_train==1][:,1], color='blue', label='Class 1', s=100, edgecolor='k')

plt.xlabel("EarShape")
plt.ylabel("FaceShape")
plt.title("Decision Tree Root Node Boundary")
plt.legend()
plt.show()