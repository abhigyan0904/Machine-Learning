# KNN for Regression

> Predict a number by averaging the nearest neighbors.

[⬅ Back to main README](../../../README.md)

---

## Definition

This module uses KNN to predict a **continuous value**. The prediction for a new point is the average target of its `k` nearest training points.

## Intuition

Instead of fitting a global line, the model builds its prediction locally. Each point is estimated by asking, "what did similar points look like?"

## Formulas

**Prediction**

$$\hat{y} = \frac{1}{k}\sum_{i \in N_k(x)}y_i$$

**Distance-weighted variant**

$$\hat{y} = \frac{\sum_{i \in N_k(x)}w_iy_i}{\sum_{i \in N_k(x)}w_i}, \qquad w_i = \frac{1}{d(x, x_i)}$$

## Files in this folder

- `K-NN_Linear.py`: KNN regression with a plot of the fitted curve.

## How to run

```bash
python K-NN_Linear.py
```

## Expected outcome

- A flexible, step-like curve that follows local trends.
- Smoother output as `k` increases.

## Tips and hyperparameters

- Compare `k = 1` (overfit) with larger `k` values.
- Evaluate with RMSE or R².

## Strengths and limitations

**Strengths:** follows non-linear trends without assumptions.

**Limitations:** cannot extrapolate beyond the training range.

---

[⬅ Back to main README](../../../README.md)
