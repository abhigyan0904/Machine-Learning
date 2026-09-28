# SVM Without a Kernel (Linear SVM)

> A straight-line boundary with the widest margin.

[⬅ Back to main README](../../../README.md)

---

## Definition

The linear SVM separates classes with a flat hyperplane. It works best when the classes are linearly separable or nearly so.

## Intuition

Think of sliding a straight ruler between two clusters and adjusting it until the gap on both sides is as big as possible.

## Formulas

**Decision function**

$$f(x) = \mathbf{w}^\top\mathbf{x} + b, \qquad \hat{y} = \text{sign}\left(f(x)\right)$$

**Objective**

$$\min_{\mathbf{w},b}\ \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{m}\max\left(0,\ 1 - y^{(i)}f(x^{(i)})\right)$$

## Files in this folder

- `SVM_without_kernal.py`: trains and plots the linear SVM.

## How to run

```bash
python SVM_without_kernal.py
```

## Expected outcome

- A straight boundary with two parallel margin lines.
- Good accuracy on linearly separable data, and visible errors on curved data, which motivates the kernel version.

## Tips and hyperparameters

- Tune `C` on a validation set.
- Run the kernel version on the same data and compare.

## Strengths and limitations

**Strengths:** fast, few parameters.

**Limitations:** cannot separate data that needs a curved boundary.

---

[⬅ Back to main README](../../../README.md)
