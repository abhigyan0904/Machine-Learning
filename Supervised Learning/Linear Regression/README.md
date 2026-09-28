# Linear Regression

> Fit the best straight line through your data.

[⬅ Back to main README](../../README.md)

---

## Definition

Linear regression predicts a continuous value from a single input feature by learning a slope `w` and an intercept `b`. It is the simplest regression model and the foundation for most of what follows in this repository.

## Intuition

Imagine plotting house size against price. Linear regression draws the one line that sits as close as possible to all the points. Training means nudging the line up, down, and tilting it until the total error is as small as it can get.

## Formulas

**Model**

$$\hat{y} = wx + b$$

**Cost function (Mean Squared Error)**

$$J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$$

**Gradients**

$$\frac{\partial J}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)x^{(i)}, \qquad \frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)$$

**Gradient descent update**

$$w := w - \alpha\frac{\partial J}{\partial w}, \qquad b := b - \alpha\frac{\partial J}{\partial b}$$

Here `α` is the learning rate and `m` is the number of training examples.

## Files in this folder

- `LinearRegression.py`: implements the model, cost function, and gradient descent.

## How to run

```bash
python LinearRegression.py
```

## Expected outcome

- A fitted line drawn over the data points.
- The learned values of `w` and `b`.
- A cost curve that steadily decreases as training progresses.

## Tips and hyperparameters

- **Learning rate (`α`):** if the cost goes up or oscillates, lower it.
- **Iterations:** train until the cost curve flattens.
- **Scaling:** scale the feature if values are very large.

## Strengths and limitations

**Strengths:** fast, easy to interpret, a great baseline.

**Limitations:** only captures straight-line trends and is sensitive to outliers.

---

[⬅ Back to main README](../../README.md)
