# Polynomial Regression

> Fit curves while keeping the model linear in its weights.

[⬅ Back to main README](../../../README.md)

---

## Definition

Polynomial regression creates new features from powers of the original input (`x`, `x²`, `x³`, ...) and then applies ordinary linear regression to them. This lets a linear model bend to follow curved data.

## Intuition

A straight line cannot follow a U-shaped trend, but a parabola can. By feeding `x²` to the model as if it were a separate feature, the same learning algorithm suddenly fits the curve.

## Formulas

**Model**

$$\hat{y} = w_1x + w_2x^2 + \dots + w_dx^d + b$$

**Feature mapping**

$$x \;\longrightarrow\; [x,\; x^2,\; x^3,\; \dots,\; x^d]$$

**Cost function**

$$J(\mathbf{w},b) = \frac{1}{2m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$$

## Files in this folder

- The Python script(s) in this folder build polynomial features and train the model.

## How to run

```bash
python <script_name>.py
```

## Expected outcome

- A smooth curve that follows the non-linear trend.
- A visible comparison between low-degree (underfit) and high-degree (overfit) fits.

## Tips and hyperparameters

- **Degree `d`:** the main knob. Start small (2 or 3).
- **Scaling is critical:** `x⁵` can be enormous, so standardize features.
- Combine with regularization (Ridge or Lasso) to tame high degrees.

## Strengths and limitations

**Strengths:** simple way to model curves.

**Limitations:** high degrees overfit quickly and extrapolate badly outside the training range.

---

[⬅ Back to main README](../../../README.md)
