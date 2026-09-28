# Multiple Linear Regression

> Linear regression with many features at once.

[⬅ Back to main README](../../README.md)

---

## Definition

Multiple linear regression predicts a continuous target from several input features. Each feature gets its own weight, showing how strongly it pushes the prediction up or down.

## Intuition

Predicting a house price from size, number of rooms, and age is no longer a line but a flat surface in higher dimensions. The model finds the weights that make this surface fit the data best.

## Formulas

**Model**

$$\hat{y} = w_1x_1 + w_2x_2 + \dots + w_nx_n + b = \mathbf{w}^\top\mathbf{x} + b$$

**Cost function**

$$J(\mathbf{w},b) = \frac{1}{2m}\sum_{i=1}^{m}\left(\mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)}\right)^2$$

**Closed-form solution (Normal Equation)**

$$\mathbf{w} = (X^\top X)^{-1}X^\top\mathbf{y}$$

**Z-score feature scaling**

$$x_j' = \frac{x_j - \mu_j}{\sigma_j}$$

## Polynomial Regression (sub-folder)

Polynomial regression captures curves by adding powers of the input as new features while the model stays linear in its weights.

$$\hat{y} = w_1x + w_2x^2 + \dots + w_dx^d + b$$

A degree that is too low underfits, and one that is too high overfits. Pick the degree using a validation set.

## Files in this folder

- `Multiple_linear_regression.py`: multi-feature regression with gradient descent.
- `Polynomial Regression/`: extends the model to curved relationships.

## How to run

```bash
python Multiple_linear_regression.py
```

## Expected outcome

- One learned weight per feature.
- Predictions for new samples.
- A decreasing cost curve, which converges faster after feature scaling.

## Tips and hyperparameters

- **Always scale features** before gradient descent, otherwise large-valued features dominate.
- Watch for **multicollinearity** (highly correlated features), which makes weights unstable.
- Use the Normal Equation only when the number of features is small.

## Strengths and limitations

**Strengths:** interpretable weights, quick to train.

**Limitations:** assumes a linear relationship and is sensitive to outliers and correlated features.

---

[⬅ Back to main README](../../README.md)
