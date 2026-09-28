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

#Variance
def variance(y):
    y_mean = np.mean(y)
    var = np.mean((y - y_mean)**2)
    return var

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

#Variance Reduction
def variance_reduction(X, y, feature_index, threshold):
    left, right = split_tree(X, feature_index, threshold)

    if len(left) == 0 or len(right) == 0:
        return 0

    y_left = y[left]
    y_right = y[right]

    parent_var = variance(y)

    w_left = len(left) / len(y)
    w_right = len(right) / len(y)

    child_var = w_left * variance(y_left) + w_right * variance(y_right)

    return parent_var - child_var

#Best Split:
def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature in range(X.shape[1]):
        for i in range(len(X)):
            threshold = X[i, feature]

            gain = variance_reduction(X, y, feature, threshold)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


feature, threshold, gain = best_split(x_train, y_train)

print("best_feature:", feature)
print("best_threshold:", threshold)
print("best_gain:", gain)

#Leaf utilities
def is_pure(y):
    return len(np.unique(y)) == 1


def leaf_prediction(y):
    return np.mean(y)

#Build Tree with Depth
def build_tree(X, y, depth=0, max_depth=3):
    if depth == max_depth or len(X) <= 2:
        return {
            "type": "leaf",
            "prediction": leaf_prediction(y)
        }

    feature, threshold, gain = best_split(X, y)

    if feature is None or gain == 0:
        return {
            "type": "leaf",
            "prediction": leaf_prediction(y)
        }

    left_idx, right_idx = split_tree(X, feature, threshold)

    return {
        "type": "node",
        "feature": feature,
        "threshold": threshold,
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
tree = build_tree(x_train, y_train, max_depth=3)

y_pred = predict(x_train, tree)


# accuracy = np.mean(y_pred.reshape(-1, 1) == y_pred)
# print("Accuracy:", accuracy)

# ----- 1D DECISION TREE REGRESSION PLOT -----
x_test = np.linspace(x_train.min() - 1, x_train.max() + 1, 500).reshape(-1, 1)
y_pred = predict(x_test, tree)
plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, color='blue', label='Original Data')
plt.plot(x_test, y_pred, color='red', label='Decision Tree Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Tree Regression - 1D')
plt.legend()
plt.grid(True)
plt.show()