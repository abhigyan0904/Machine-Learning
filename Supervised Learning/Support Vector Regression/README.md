# Support Vector Regression (SVR)

> Regression with a tolerance tube around the prediction.

[⬅ Back to main README](../../README.md)

---

## Definition

SVR applies the SVM idea to regression. It fits a function that stays inside a tube of width `ε` around the data and **ignores errors smaller than `ε`**. Only points outside the tube shape the model.

## Intuition

Instead of chasing every tiny deviation, SVR says: "anything within this band is good enough." That makes the fit smoother and more robust to noise.

## Formulas

**ε-insensitive loss**

$$L_\epsilon(y, \hat{y}) = \max\left(0,\ |y - \hat{y}| - \epsilon\right)$$

**Objective**

$$\min_{\mathbf{w},b,\xi,\xi^*}\ \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i}\left(\xi_i + \xi_i^*\right)$$

**Constraints**

$$y^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)} - b \leq \epsilon + \xi_i, \qquad \mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)} \leq \epsilon + \xi_i^*$$

## Files in this folder

- `SVR.py`: fits an SVR model and plots the prediction with its ε-tube.

## How to run

```bash
python SVR.py
```

## Expected outcome

- A smooth regression curve with a shaded tube.
- Support vectors highlighted outside or on the tube edge.

## Tips and hyperparameters

- **`ε`:** wider tube means a simpler model with fewer support vectors.
- **`C`:** penalty for points outside the tube.
- **Kernel and `γ`:** control how flexible the curve is.
- Scale both features and the target.

## Strengths and limitations

**Strengths:** robust to noise and outliers, flexible with kernels.

**Limitations:** slow on big datasets and sensitive to hyperparameters.

---

[⬅ Back to main README](../../README.md)
