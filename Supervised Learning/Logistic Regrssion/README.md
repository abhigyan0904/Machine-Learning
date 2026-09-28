# Logistic Regression

> A classifier that turns a linear score into a probability.

[⬅ Back to main README](../../README.md)

---

## Definition

Despite its name, logistic regression is a **classification** algorithm. It estimates the probability that an input belongs to the positive class using the sigmoid function, then applies a threshold to make a decision.

## Intuition

Linear regression can output any number, but a probability must live between 0 and 1. The sigmoid squashes the linear score into that range: large positive scores become close to 1, large negative scores close to 0.

## Formulas

**Sigmoid**

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = \mathbf{w}^\top\mathbf{x} + b$$

**Prediction**

$$\hat{y} = \begin{cases} 1 & \text{if } \sigma(z) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

**Loss (Binary Cross-Entropy)**

$$J(\mathbf{w},b) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log\hat{p}^{(i)} + \left(1-y^{(i)}\right)\log\left(1-\hat{p}^{(i)}\right)\right]$$

**Gradient**

$$\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m}\left(\hat{p}^{(i)} - y^{(i)}\right)x_j^{(i)}$$

## Regularization in detail

**L2 (Ridge)** shrinks weights smoothly:

$$J_{\text{ridge}} = J(\mathbf{w},b) + \frac{\lambda}{2m}\sum_{j=1}^{n}w_j^2$$

**L1 (Lasso)** can push weights to exactly zero, acting as feature selection:

$$J_{\text{lasso}} = J(\mathbf{w},b) + \frac{\lambda}{m}\sum_{j=1}^{n}|w_j|$$

## Files in this folder

- `LogisticRegression.py`: baseline binary classifier with a linear boundary.
- `PolynomialLogisticRegression.py`: adds polynomial features for a curved boundary.
- `Regularization.py`: shows L1 and L2 penalties reducing overfitting.

## How to run

```bash
python LogisticRegression.py
python PolynomialLogisticRegression.py
python Regularization.py
```

## Expected outcome

- Class probabilities for every sample.
- A decision boundary plot (straight for the baseline, curved for the polynomial version).
- Regularization shrinks weights and produces a simpler, more general boundary.

## Tips and hyperparameters

- **Threshold:** 0.5 by default. Lower it to catch more positives (higher recall).
- **`λ` (regularization):** larger values mean a simpler model.
- **Scale features** for stable gradient descent.

## Strengths and limitations

**Strengths:** gives probabilities, fast, interpretable.

**Limitations:** linear boundary by default, so it needs feature engineering for complex data.

---

[⬅ Back to main README](../../README.md)
