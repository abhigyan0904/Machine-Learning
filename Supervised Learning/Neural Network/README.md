# Neural Network

> Stack layers of simple neurons to learn complex patterns.

[⬅ Back to main README](../../README.md)

---

## Definition

A neural network is a chain of layers. Each neuron computes a weighted sum of its inputs and passes it through a non-linear **activation function**. Stacking layers lets the network learn patterns far beyond what a linear model can.

## Intuition

Early layers learn simple features (edges in an image), and later layers combine them into more abstract ones (shapes, then digits). Training adjusts millions of small knobs so the final output matches the correct answer.

## Formulas

**Forward propagation for layer $l$**

$$\mathbf{z}^{[l]} = W^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \qquad \mathbf{a}^{[l]} = g\left(\mathbf{z}^{[l]}\right)$$

**Activations**

$$\text{ReLU}(z) = \max(0, z), \qquad \sigma(z) = \frac{1}{1+e^{-z}}, \qquad \text{softmax}(z_k) = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

**Loss (multi-class cross-entropy)**

$$J = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}y_k^{(i)}\log\hat{y}_k^{(i)}$$

**Backpropagation update**

$$W^{[l]} := W^{[l]} - \alpha\frac{\partial J}{\partial W^{[l]}}, \qquad \mathbf{b}^{[l]} := \mathbf{b}^{[l]} - \alpha\frac{\partial J}{\partial \mathbf{b}^{[l]}}$$

## Files in this folder

- `CreatingLayer.py`: builds a single dense layer, the basic building block.
- `NeuralNetwork.py`: combines layers into a full network with forward and backward passes.
- `Hand Written Digit Classification/`: recognizing handwritten digits (0 to 9).

## How to run

```bash
python CreatingLayer.py
python NeuralNetwork.py
```

## Expected outcome

- A trained network with decreasing loss and increasing accuracy over epochs.
- Correct predictions on handwritten digits.

## Tips and hyperparameters

- **Learning rate:** the most important setting. Try `0.001` to `0.1`.
- **Layers and units:** more capacity fits more, but overfits more.
- **Regularization:** dropout, L2, or early stopping.
- **Normalize inputs** (for example, pixel values to `0..1`).

## Strengths and limitations

**Strengths:** learns complex, non-linear relationships and scales to images, text, and audio.

**Limitations:** needs more data and compute, and is harder to interpret.

---

[⬅ Back to main README](../../README.md)
