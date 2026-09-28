import pandas as pd
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the fixed CSV
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

# One example per class
examples = train_df.groupby("label").first().reset_index()

# Plot
plt.figure(figsize=(10, 4))
for i in range(10):
    ax = plt.subplot(2, 5, i + 1)
    img = examples.loc[i].drop("label").values.astype(np.uint8).reshape(28, 28)
    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {examples.loc[i, 'label']}")
    plt.axis("off")

plt.tight_layout()
plt.show()



x_train = train_df.iloc[:, 1:].values
x_train = x_train[:]
x_train = x_train / 255 #(x_train - np.mean(x_train, axis = 0)) / (np.std(x_train, axis = 0) + 1e-8)
y_train = train_df.iloc[:, 0].values
y_train = y_train[:]
y_train = np.eye(10)[y_train]

x_test = test_df.iloc[:, 1:].values
x_test = x_test / 255 #(x_test - np.mean(x_train, axis = 0)) / (np.std(x_train, axis = 0) + 1e-8)
y_test = test_df.iloc[:, 0].values
y_test = np.eye(10)[y_test]



print(x_train.shape)        # (m, features)
print(y_train.shape) # (m, 10)


#Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#ReLU Function
def reLU(z):
    return np.maximum(0, z)

#Softmax
def softmax(z):
    exp_z = np.exp(z)
    return exp_z / (np.sum(exp_z, axis=1, keepdims = True) + 1e-8)

#Diffrentiation
def diff_sigmoid(z):
    return z * (1 - z)

def diff_reLU(z):
    return (z > 0).astype(float)

architecture = [512, 128, 10]
activation = [reLU, reLU, softmax]

input_features = x_train.shape[1]
W = []
B = []
m = x_train.shape[0]
for i in architecture:
    layer_w = np.random.randn(input_features, i) * 0.05
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

a_train_loss = []
a_test_loss = []
a_train_accuracy = []
a_test_accuracy = []
alpha = 0.3

for epoch in range(50):
    a,z_s,activation_output = forward_prop(W, B, x_train, activation)
    dw,db = back_prop(activation_output,z_s,y_train,W,activation)
    for i in range(len(W)):
        W[i] = W[i]- (alpha * dw[i])
        B[i] = B[i] -  (alpha * db[i])

    train_cost = np.sum(-y_train * np.log(a + 1e-8)) / m
    train_labels = y_train.argmax(axis=1)
    train_pred_labels = a.argmax(axis=1)
    train_accuracy = ((train_labels == train_pred_labels).sum()) / len(train_labels)
    print("train_accuracy", train_accuracy * 100)
    print(f" train_cost : {train_cost} , epoch : {epoch + 1}")
    a_train_loss.append(train_cost)
    a_train_accuracy.append(train_accuracy)


    a, z_s, activation_output = forward_prop(W, B, x_test, activation)
    test_cost = np.sum(-y_test * np.log(a + 1e-8)) / m
    test_labels = y_test.argmax(axis=1)
    test_pred_labels = a.argmax(axis=1)
    test_accuracy = ((test_labels == test_pred_labels).sum()) / len(test_labels)
    a_test_loss.append(test_cost)
    a_test_accuracy.append(test_accuracy)


    print("test accuracy", test_accuracy * 100)
    print(f" test_cost : {test_cost} , epoch : {epoch+1}")


plt.plot(a_train_loss)
plt.plot(a_test_loss)
plt.show()

plt.plot(a_train_accuracy)
plt.plot(a_test_accuracy)
plt.show()

