# Decision Tree

> Predict by asking a series of yes/no questions.

[⬅ Back to main README](../../README.md)

---

## Definition

A decision tree splits the data again and again using simple rules on feature values until each group is mostly one class (classification) or has similar values (regression). The path from the root to a leaf is a human-readable rule.

## Intuition

Think of a game of twenty questions. Each question (a split) is chosen to narrow down the answer as much as possible. The tree learns which questions to ask and in what order.

## Formulas

**Gini impurity**

$$G = 1 - \sum_{k=1}^{K}p_k^2$$

**Entropy**

$$H = -\sum_{k=1}^{K}p_k\log_2 p_k$$

**Information gain** (choose the split that maximizes it)

$$IG = H(\text{parent}) - \sum_{c \in \text{children}}\frac{N_c}{N}H(c)$$

**Regression trees** minimize the variance (MSE) within each region; each leaf predicts the mean of its samples.

## Files in this folder

- `Binary Classification/`: two-class problem with `animal_features.csv` and a decision boundary plot.
- `Multi Class Classification/`: more than two classes with `animal_features_extended.csv`.
- `DecisionTreeOnLinearRegression/`: a regression tree on a continuous target.

## How to run

```bash
cd "Binary Classification"
python DecisionTree.py
```

## Expected outcome

- A set of readable if/else rules.
- Decision boundaries that appear as axis-aligned rectangles.
- Training accuracy that can reach 100% if the tree grows deep, a sign of overfitting.

## Tips and hyperparameters

- **`max_depth`:** the easiest way to control overfitting.
- **`min_samples_split` / `min_samples_leaf`:** stop tiny, noisy splits.
- **Pruning:** remove branches that do not improve validation performance.

## Strengths and limitations

**Strengths:** highly interpretable, needs no scaling, handles mixed data.

**Limitations:** overfits easily, unstable to small data changes, boundaries are axis-aligned only.

---

[⬅ Back to main README](../../README.md)
