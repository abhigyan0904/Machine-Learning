# Supervised Learning

[⬅ Back to main README](../README.md)

In supervised learning, the model learns from **labeled data**: every input `x` comes with a known answer `y`. The model learns the mapping from `x` to `y` so it can predict answers for new, unseen inputs.

- **Regression** predicts a continuous number (a price, a temperature).
- **Classification** predicts a category (spam or not spam, which digit).

## Modules in this folder

| Module | Task | One-line summary |
|---|---|---|
| [Linear Regression](./Linear%20Regression) | Regression | Fit a straight line to one feature. |
| [Multiple Linear Regression](./Multiple%20Linear%20Regression) | Regression | Fit a plane or hyperplane to many features. |
| [Logistic Regrssion](./Logistic%20Regrssion) | Classification | Predict class probabilities with a sigmoid. |
| [Decision Tree](./Decision%20Tree) | Both | Split data with a chain of yes/no questions. |
| [Hinge Loss](./Hinge%20Loss) | Classification | The margin-based loss behind SVMs. |
| [Support Vector Machine](./Support%20Vector%20Machine) | Classification | Find the widest-margin boundary. |
| [Support Vector Regression](./Support%20Vector%20Regression) | Regression | Fit a function inside an ε-tube. |
| [Neural Network](./Neural%20Network) | Both | Stack layers of neurons to learn complex patterns. |

## Common workflow

1. Split data into train, validation, and test sets.
2. Scale features where needed.
3. Choose a model and a loss function.
4. Train, tune hyperparameters on the validation set.
5. Report final performance once on the test set.
