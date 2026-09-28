# Hinge Loss

> The loss function that gives SVMs their wide margins.

[⬅ Back to main README](../../README.md)

---

## Definition

Hinge loss penalizes points that are misclassified **or correctly classified but too close to the boundary**. That extra pressure pushes the model to leave a healthy margin around the decision line.

## Intuition

With log loss, a correct prediction is always slightly rewarded for more confidence. With hinge loss, once a point is safely beyond the margin, the loss is exactly zero, so the model stops worrying about it and focuses on the difficult points.

## Formulas

**Hinge loss** (labels must be $y \in \{-1, +1\}$)

$$L(y, f(x)) = \max\left(0,\; 1 - y\cdot f(x)\right)$$

**Regularized SVM objective**

$$J(\mathbf{w},b) = \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{m}\max\left(0,\; 1 - y^{(i)}(\mathbf{w}^\top\mathbf{x}^{(i)} + b)\right)$$

**Subgradient** for a sample with $y f(x) < 1$:

$$\frac{\partial L}{\partial \mathbf{w}} = -y\,\mathbf{x}, \qquad \frac{\partial L}{\partial b} = -y$$

## Files in this folder

- `Hinge_Loss_SVM.py`: a linear classifier trained with hinge loss.
- `Hinge_Loss_Polynomial.py`: the same idea with polynomial features for a curved boundary.

## How to run

```bash
python Hinge_Loss_SVM.py
python Hinge_Loss_Polynomial.py
```

## Expected outcome

- A maximum-margin decision boundary.
- A loss curve that decreases and levels off.
- A visible difference between the linear and polynomial boundaries.

## Tips and hyperparameters

- **`C`:** large values punish violations heavily (narrow margin), small values allow a wider, softer margin.
- Remember to convert labels from `{0, 1}` to `{-1, +1}`.

## Strengths and limitations

**Strengths:** focuses on hard examples, robust margin-based training.

**Limitations:** not probabilistic and not differentiable at the hinge point (subgradients are used).

---

[⬅ Back to main README](../../README.md)
