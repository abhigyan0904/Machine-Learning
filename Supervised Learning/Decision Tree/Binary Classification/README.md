# Decision Tree: Binary Classification

> Separate two classes using a tree of simple rules.

[⬅ Back to main README](../../../README.md)

---

## Definition

This module trains a decision tree to distinguish between **two classes** using the animal features in `animal_features.csv`. It also visualizes how the tree carves up the feature space.

## Intuition

Each split asks a question like "is this feature above a certain value?" The two answers send a sample left or right. After a few questions, every sample lands in a leaf that predicts one of the two classes.

## Formulas

**Gini impurity for two classes**

$$G = 1 - p^2 - (1-p)^2 = 2p(1-p)$$

**Best split** minimizes the weighted impurity of the children:

$$G_{\text{split}} = \frac{N_L}{N}G_L + \frac{N_R}{N}G_R$$

## Files in this folder

- `animal_features.csv`: the dataset.
- `DecisionTree.py`: trains the tree and prints or plots the result.
- `ContinuousFeatureDecisionBoundary.py`: draws the decision boundary for continuous features.

## How to run

```bash
python DecisionTree.py
python ContinuousFeatureDecisionBoundary.py
```

## Expected outcome

- The trained tree with its split rules.
- Predicted classes for each animal.
- A boundary plot where each region is colored by predicted class.

## Tips and hyperparameters

- Try different `max_depth` values and watch the boundary become more or less detailed.
- Compare Gini and entropy criteria; results are usually similar.

## Strengths and limitations

**Strengths:** the boundary is easy to visualize and explain.

**Limitations:** small datasets can produce very jagged, overfit boundaries.

---

[⬅ Back to main README](../../../README.md)
