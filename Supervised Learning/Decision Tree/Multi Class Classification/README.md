# Decision Tree: Multi-Class Classification

> Classify samples into three or more categories.

[⬅ Back to main README](../../../README.md)

---

## Definition

This module extends the decision tree to **more than two classes** using `animal_features_extended.csv`, and plots decision boundaries across multiple features.

## Intuition

The tree works exactly as before, but each leaf now predicts whichever class is most common among its samples, and impurity is measured across all classes at once.

## Formulas

**Gini impurity for K classes**

$$G = 1 - \sum_{k=1}^{K}p_k^2$$

**Leaf prediction**

$$\hat{y} = \arg\max_k\ p_k$$

## Files in this folder

- `animal_features_extended.csv`: the multi-class dataset.
- `MultipleFeatureDecisionBoundary.py`: trains the tree and plots the boundaries.

## How to run

```bash
python MultipleFeatureDecisionBoundary.py
```

## Expected outcome

- Several colored regions, one per class.
- Predictions for each sample.
- Optionally a confusion matrix showing which classes get mixed up.

## Tips and hyperparameters

- Check for **class imbalance**; rare classes may be ignored by a shallow tree.
- Evaluate with per-class precision and recall, not just accuracy.

## Strengths and limitations

**Strengths:** handles many classes naturally.

**Limitations:** with many classes and few samples, leaves become tiny and unreliable.

---

[⬅ Back to main README](../../../README.md)
