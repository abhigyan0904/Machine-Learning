# Decision Tree for Regression

> Use a tree to predict a continuous value.

[⬅ Back to main README](../../../README.md)

---

## Definition

A regression tree splits the input space into regions and predicts the **average target value** inside each region. This module compares that step-shaped prediction against a straight-line fit.

## Intuition

Instead of one smooth line, the tree builds a staircase. Each step is a region where the tree predicts a constant value. More splits mean more, smaller steps.

## Formulas

**Split criterion (minimize weighted variance)**

$$\text{MSE}_{\text{split}} = \frac{N_L}{N}\text{MSE}_L + \frac{N_R}{N}\text{MSE}_R$$

**Leaf prediction**

$$\hat{y}_{\text{leaf}} = \frac{1}{N_{\text{leaf}}}\sum_{i \in \text{leaf}}y_i$$

## Files in this folder

- `DecisionTreeOnLinearRegression.py`: fits a regression tree and plots the result.

## How to run

```bash
python DecisionTreeOnLinearRegression.py
```

## Expected outcome

- A staircase-shaped prediction curve.
- A clear view of how depth controls smoothness.

## Tips and hyperparameters

- **`max_depth`:** shallow trees underfit, deep trees memorize noise.
- Regression trees **cannot extrapolate** beyond the training range.

## Strengths and limitations

**Strengths:** captures non-linear patterns without feature engineering.

**Limitations:** predictions are piecewise constant and never go beyond seen values.

---

[⬅ Back to main README](../../../README.md)
