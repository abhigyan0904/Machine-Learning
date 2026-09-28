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

    return (X, Y.reshape(-1, 1))


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

#Calculate Entropy:
def entropy(p1):
    if p1 == 0 or p1 == 1:
        return 0
    return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

#Split tree:
def split_tree(X, feature_index, threshold):
    left = []
    right = []
    for i in range(len(X)):
        if X[i][feature_index] <= threshold:
            left.append(i)
        else:
            right.append(i)
    return left, right

#Calculate Information gain:
def information_gain(X, y, feature_index, threshold):
    left, right = split_tree(X, feature_index, threshold)
    if len(left) == 0 or len(right) == 0:
        return 0
    y_left = y[left]
    y_right = y[right]

    p_parent = np.mean(y)
    parent_entropy = entropy(p_parent)

    left_entropy = entropy(np.mean(y_left))
    right_entropy = entropy(np.mean(y_right))


    w_left = len(left) / len(y)
    w_right = len(right) / len(y)

    child_entropy = (w_left * left_entropy ) + (w_right * right_entropy)
    return parent_entropy - child_entropy

#Best Split:
def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        for i in range(len(X)):
            threshold = X[i, feature]
            gain = information_gain(X, y, feature, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

feature, threshold, gain = best_split(X, y)
print("best_feature", feature)
print("best_threshold", threshold)
print("best_gain", gain)

#Leaf utilities
def is_pure(y):
    return len(np.unique(y)) == 1


def leaf_prediction(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


#Build Tree with Depth
def build_tree(X, y, depth = 0, max_depth = 3):
    if is_pure(y) or depth == max_depth or len(X) == 0:
        return {
            "type": "leaf",
            "prediction": leaf_prediction(y),
            "depth": depth
        }

    feature, threshold, gain = best_split(X, y)

    if feature is None or gain == 0:
        return {
            "type": "leaf",
            "prediction": leaf_prediction(y),
            "depth": depth
        }

    left_idx, right_idx = split_tree(X, feature, threshold)

    return {
        "type": "node",
        "feature": feature,
        "threshold": threshold,
        "depth": depth,
        "left": build_tree(X[left_idx], y[left_idx], depth + 1, max_depth),
        "right": build_tree(X[right_idx], y[right_idx], depth + 1, max_depth)
    }


#Prediction
def predict_one(x, tree):
    if tree["type"] == "leaf":
        return tree["prediction"]

    if x[tree["feature"]] <= tree["threshold"]:
        return predict_one(x, tree["left"])
    else:
        return predict_one(x, tree["right"])


def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])


#Train & Evaluate
tree = build_tree(X, y, max_depth=3)

y_pred = predict(X, tree)
accuracy = np.mean(y_pred.reshape(-1, 1) == y)
print("Accuracy:", accuracy)

plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap="bwr", alpha=0.6)

#Plotting threshold:
if feature == 0:
    plt.axvline(threshold, color="black", linestyle="--")
else:
    plt.axhline(threshold, color="black", linestyle="--")

plt.title("Best Threshold")
plt.show()


# ----- DECISION BOUNDARY PLOT -----
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]

Z = predict(grid, tree)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")

plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap="bwr", edgecolor="k")

plt.title("Decision Boundary of Decision Tree")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()