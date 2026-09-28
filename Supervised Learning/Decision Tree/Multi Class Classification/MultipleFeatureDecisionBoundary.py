import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv("animal_features_extended.csv")

# -------- Save category mappings BEFORE encoding --------
category_maps = {}

for col in file.columns:
    if col == "Animal":
        category_maps[col] = file[col].astype("category").cat.categories
    else:
        category_maps[col] = file[col].astype("category").cat.categories


# Encode categorical data
for col in file.columns:
    file[col] = file[col].astype("category").cat.codes


x_train = file.drop("Animal", axis = 1).values
y_train = file["Animal"].values

num_classes = len(np.unique(y_train))
num_features = x_train.shape[1]

# Entropy (multiclass)
def entropy(y):
    classes, counts = np.unique(y, return_counts = True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

# Split function
def split_tree(X, feature_index, value):
    left = []
    right = []
    for i in range(len(X)):
        if X[i][feature_index] == value:
            left.append(i)
        else:
            right.append(i)
    return left, right

#Show probabilities
def show_probabilities(y, left, right):
    print("Root:", np.bincount(y, minlength = num_classes))
    print("Left:", np.bincount(y[left], minlength = num_classes))
    print("Right:", np.bincount(y[right], minlength = num_classes))

#Weighted Entropy
def weighted_entropy(y, left, right):
    w_left = len(left) / len(y)
    w_right = len(right) / len(y)
    return w_left * entropy(y[left]) + w_right * entropy(y[right])

#Information Gain
def information_gain(y, left, right):
    return entropy(y) - weighted_entropy(y, left, right)

#Find Best Root Split
best_feature = None
best_value = None
best_ig = -1

for i in range(num_features):
    print("\nFeature index:", i)
    for value in np.unique(x_train[:, i]):
        left, right = split_tree(x_train, i, value)
        if len(left) == 0 or len(right) == 0:
            continue

        show_probabilities(y_train, left, right)
        ig = information_gain(y_train, left, right)
        print(f"Split value {value} → IG = {ig:.4f}")

        if ig > best_ig:
            best_ig = ig
            best_feature = i
            best_value = value
            best_left, best_right = left, right

print("\nBest Feature Index:", best_feature)
print("Best Feature Value:", best_value)
print("Best Information Gain:", round(best_ig, 4))

# Root Node Prediction
def predict_point(point):
    if point[best_feature] == best_value:
        return np.bincount(y_train[best_left], minlength = num_classes).argmax()
    else:
        return np.bincount(y_train[best_right], minlength = num_classes).argmax()
#Predict New Sample
def predict_new_sample(sample_dict):
    new_sample = []

    feature_columns = file.drop("Animal", axis=1).columns

    for col in feature_columns:
        # Use SAVED category map
        categories = category_maps[col]
        value = categories.get_loc(sample_dict[col])
        new_sample.append(value)

    new_sample = np.array(new_sample)

    if new_sample[best_feature] == best_value:
        return np.bincount(y_train[best_left], minlength=num_classes).argmax()
    else:
        return np.bincount(y_train[best_right], minlength=num_classes).argmax()



#Predict Training Data
y_pred = np.array([predict_point(x) for x in x_train])
print("Predictions:", y_pred)

# -------- USER INPUT --------
sample = {}

print("\nEnter test sample values:")

for col in file.drop("Animal", axis=1).columns:
    sample[col] = input(f"{col}: ")

prediction = predict_new_sample(sample)

print("\nPredicted class index:", prediction)
print("Predicted Animal:",
      file["Animal"].astype("category").cat.categories[prediction])