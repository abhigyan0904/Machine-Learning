# KNN for Classification

> Classify by letting the nearest neighbors vote.

[⬅ Back to main README](../../../README.md)

---

## Definition

This module uses KNN to predict a **class label**. A new point is assigned the most common class among its `k` nearest training points.

## Intuition

You are known by the company you keep: if most of a point's closest neighbors are of one class, the point probably belongs there too.

## Formulas

**Majority vote**

$$\hat{y} = \arg\max_{c}\sum_{i \in N_k(x)}\mathbb{1}\left(y_i = c\right)$$

**Class probability estimate**

$$P(c \mid x) = \frac{1}{k}\sum_{i \in N_k(x)}\mathbb{1}\left(y_i = c\right)$$

## Files in this folder

- `K-NN_Logistic.py`: KNN classifier with a decision boundary plot.

## How to run

```bash
python K-NN_Logistic.py
```

## Expected outcome

- A decision boundary that can take complex shapes.
- Predicted classes and probability-like scores.

## Tips and hyperparameters

- Use an odd `k` for binary problems to avoid ties.
- Evaluate with accuracy, precision, and recall.
- Compare against the logistic regression module on the same data.

## Strengths and limitations

**Strengths:** handles irregular boundaries with no training.

**Limitations:** sensitive to noisy points and to the choice of `k`.

---

[⬅ Back to main README](../../../README.md)
