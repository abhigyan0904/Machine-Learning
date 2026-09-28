# SVM With a Kernel

> Separate non-linear data using the kernel trick.

[⬅ Back to main README](../../../README.md)

---

## Definition

A kernel SVM behaves as if the data were mapped into a much higher-dimensional space where a straight boundary works, **without ever computing that mapping**. In the original space the boundary appears curved.

## Intuition

Points arranged in a ring cannot be split by a line. Lift the inner points up and the outer points stay low, and a flat sheet now separates them. The kernel does this lifting implicitly.

## Formulas

**Kernel form of the decision function**

$$f(x) = \sum_{i}\alpha_i\,y^{(i)}K\left(x^{(i)}, x\right) + b$$

**Common kernels**

$$K_{\text{linear}}(x, z) = x^\top z$$

$$K_{\text{poly}}(x, z) = \left(\gamma\,x^\top z + r\right)^d$$

$$K_{\text{RBF}}(x, z) = \exp\left(-\gamma\|x - z\|^2\right)$$

## Files in this folder

- `SVM_with_kernal.py`: trains an SVM with a chosen kernel and plots the boundary.

## How to run

```bash
python SVM_with_kernal.py
```

## Expected outcome

- A smooth, curved boundary that wraps around the classes.
- Higher accuracy than the linear SVM on non-linear data.

## Tips and hyperparameters

- **`γ` (RBF):** large values make a tight, wiggly boundary (overfit), small values a smooth one (underfit).
- **`C`:** as in the linear SVM.
- Start with RBF, then try polynomial.

## Strengths and limitations

**Strengths:** handles complex boundaries with a simple change of kernel.

**Limitations:** training cost grows quickly with the number of samples, and results depend heavily on `γ` and `C`.

---

[⬅ Back to main README](../../../README.md)
