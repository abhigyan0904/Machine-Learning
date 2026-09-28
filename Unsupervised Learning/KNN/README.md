# K-Nearest Neighbors (KNN)

> Predict from the k closest examples.

[⬅ Back to main README](../../README.md)

---

## Definition

KNN has no training phase. It stores the data and, for a new point, looks at the `k` closest stored points to decide the output. It is an **instance-based supervised** method, grouped here for layout convenience.

## Intuition

To guess a new house's price, look at the few most similar houses nearby and average their prices. To classify a new animal, let its nearest neighbors vote.

## Formulas

**Euclidean distance**

$$d(x, x') = \sqrt{\sum_{j=1}^{n}\left(x_j - x_j'\right)^2}$$

**Regression**

$$\hat{y} = \frac{1}{k}\sum_{i \in N_k(x)}y_i$$

**Classification**

$$\hat{y} = \arg\max_{c}\sum_{i \in N_k(x)}\mathbb{1}\left(y_i = c\right)$$

## Files in this folder

- `KNN_Linear/`: KNN used for regression.
- `KNN_Logistic/`: KNN used for classification.

## How to run

```bash
cd KNN_Linear
python K-NN_Linear.py
```

## Expected outcome

- Predictions based on nearby points.
- A boundary (or curve) that becomes smoother as `k` grows.

## Tips and hyperparameters

- **`k`:** small values overfit, large values oversmooth. Choose it by validation, usually an odd number for classification.
- **Scale features:** distances are meaningless otherwise.
- Consider distance-weighted voting.

## Strengths and limitations

**Strengths:** very simple, no training, naturally non-linear.

**Limitations:** slow at prediction time on big data and suffers in high dimensions.

---

[⬅ Back to main README](../../README.md)
