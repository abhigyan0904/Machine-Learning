# Anomaly Detection

> Spot the rare points that do not fit the pattern.

[⬅ Back to main README](../../README.md)

---

## Definition

Anomaly detection flags data points that behave very differently from the majority, such as fraud, machine faults, or unusual sensor readings. This module models normal behavior with a Gaussian distribution and marks low-probability points as anomalies.

## Intuition

Learn what "normal" looks like from lots of ordinary examples. Anything that looks very unlikely under that picture is suspicious.

## Formulas

**Estimate parameters from normal data**

$$\mu_j = \frac{1}{m}\sum_{i=1}^{m}x_j^{(i)}, \qquad \sigma_j^2 = \frac{1}{m}\sum_{i=1}^{m}\left(x_j^{(i)} - \mu_j\right)^2$$

**Probability of a new point**

$$p(x) = \prod_{j=1}^{n}\frac{1}{\sqrt{2\pi}\,\sigma_j}\exp\left(-\frac{\left(x_j - \mu_j\right)^2}{2\sigma_j^2}\right)$$

**Decision rule**

$$\text{anomaly if } p(x) < \varepsilon$$

## Files in this folder

- `Anomaly_Detection.py`: fits the Gaussian model, selects a threshold, and plots anomalies.

## How to run

```bash
python Anomaly_Detection.py
```

## Expected outcome

- Normal points and flagged anomalies shown in different colors.
- A chosen threshold `ε` and its F1 score on validation data.

## Tips and hyperparameters

- Anomalies are rare, so **use precision, recall, and F1, not accuracy**.
- Choose `ε` on a small labeled validation set by maximizing F1.
- Transform skewed features (for example, with a log) so they look more Gaussian.

## Strengths and limitations

**Strengths:** simple and works with few or no anomaly examples.

**Limitations:** assumes roughly Gaussian features and treats features as independent.

---

[⬅ Back to main README](../../README.md)
