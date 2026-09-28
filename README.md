# Machine Learning From Scratch

A hands-on collection of classic machine learning algorithms, each one implemented in Python with its own folder, runnable script, and README. The goal is simple: **understand how the algorithms actually work, not just call `.fit()` on them.**

Every module explains the idea in plain language, shows the math behind it, and describes what you should expect to see when you run the code.

---

## Table of Contents

1. [Why this repository exists](#why-this-repository-exists)
2. [Repository structure](#repository-structure)
3. [Getting started](#getting-started)
4. [Modules](#modules)
5. [Evaluation metrics cheat sheet](#evaluation-metrics-cheat-sheet)
6. [Choosing an algorithm](#choosing-an-algorithm)
7. [Key concepts used across modules](#key-concepts-used-across-modules)
8. [Contributing](#contributing)
9. [Author](#author)
10. [License](#license)

---

## Why this repository exists

Libraries like scikit-learn and TensorFlow are fantastic, but they can hide the interesting parts. When a model misbehaves in production, it helps a lot to know what is going on under the hood: how a loss function shapes learning, why regularization stops overfitting, or what a kernel really does to your data.

This repository is my way of working through those ideas step by step. You can:

- **Learn** a concept from a short definition and its formula.
- **Run** a small, self-contained script that demonstrates it.
- **Compare** related techniques side by side (linear vs. polynomial models, SVM with and without a kernel).

---

## Repository structure

```
Machine-Learning/
├── Supervised Learning/
│   ├── Decision Tree/
│   │   ├── Binary Classification/
│   │   ├── DecisionTreeOnLinearRegression/
│   │   └── Multi Class Classification/
│   ├── Hinge Loss/
│   ├── Linear Regression/
│   ├── Logistic Regrssion/
│   ├── Multiple Linear Regression/
│   │   └── Polynomial Regression/
│   ├── Neural Network/
│   │   └── Hand Written Digit Classification/
│   ├── Support Vector Machine/
│   │   ├── Svm With KernelSVM_with_kernal/
│   │   └── Svm Without Kernal/
│   └── Support Vector Regression/
├── Unsupervised Learning/
│   ├── Anomaly Detection/
│   ├── DQN/
│   ├── K_Means/
│   ├── KNN/
│   │   ├── KNN_Linear/
│   │   └── KNN_Logistic/
│   ├── Principal Component Analysis/
│   ├── Q_Learning/
│   └── Singular Value Decomposition/
└── README.md
```

Each algorithm folder contains a Python script and a README describing that specific module.

---

## Getting started

### Prerequisites

- Python 3.9 or newer
- `pip` (or `conda`)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Machine-Learning.git
cd Machine-Learning

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install the dependencies
pip install numpy pandas matplotlib scikit-learn tensorflow
```

### Running a module

Each script runs on its own. For example:

```bash
python "Supervised Learning/Linear Regression/LinearRegression.py"
python "Unsupervised Learning/K_Means/K-Means.py"
```

Datasets used by the decision tree modules (`animal_features.csv` and `animal_features_extended.csv`) live next to their scripts, so no extra downloads are needed.

---

## Modules

| Category | What's inside |
|---|---|
| [Supervised Learning](./Supervised%20Learning) | Regression, classification, SVM, decision trees, neural networks |
| [Unsupervised Learning](./Unsupervised%20Learning) | Clustering, dimensionality reduction, anomaly detection, reinforcement learning |

---

## Evaluation metrics cheat sheet

### Regression

| Metric | Formula | Meaning |
|---|---|---|
| **MAE** | $\frac{1}{m}\sum\|y - \hat{y}\|$ | Average absolute error. Easy to interpret. |
| **MSE** | $\frac{1}{m}\sum(y - \hat{y})^2$ | Penalizes large errors heavily. |
| **RMSE** | $\sqrt{\text{MSE}}$ | Error in the same units as the target. |
| **R² score** | $1 - \frac{\sum(y-\hat{y})^2}{\sum(y-\bar{y})^2}$ | Share of variance explained. 1 is perfect. |

### Classification

| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall share of correct predictions. |
| **Precision** | $\frac{TP}{TP + FP}$ | Of predicted positives, how many were right. |
| **Recall** | $\frac{TP}{TP + FN}$ | Of actual positives, how many were found. |
| **F1 score** | $2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$ | Balance of precision and recall. |

### Clustering

| Metric | Meaning |
|---|---|
| **Inertia (WCSS)** | Total within-cluster squared distance. Lower is tighter. |
| **Silhouette score** | Ranges from -1 to 1. Higher means better-separated clusters. |

---

## Choosing an algorithm

| If you need to... | Consider |
|---|---|
| Predict a number from a simple trend | Linear Regression |
| Predict a number from many features | Multiple Linear Regression |
| Model a curved relationship | Polynomial Regression, SVR |
| Classify into two or more categories | Logistic Regression, SVM, Decision Tree, Neural Network |
| Need easily explainable rules | Decision Tree |
| Separate classes with a clear margin | SVM |
| Recognize images or complex patterns | Neural Network |
| Group unlabeled data | K-Means |
| Reduce the number of features | PCA, SVD |
| Spot rare or unusual events | Anomaly Detection |
| Learn by interacting with an environment | Q-Learning, DQN |

---

## Key concepts used across modules

- **Overfitting:** the model memorizes training data and performs poorly on new data.
- **Underfitting:** the model is too simple to capture the pattern.
- **Bias-variance trade-off:** simple models have high bias, complex models have high variance. Aim for the middle.
- **Learning rate (α):** step size in gradient descent. Too large diverges, too small crawls.
- **Gradient descent:** iteratively moves parameters in the direction that reduces the loss.
- **Feature scaling:** puts features on a similar range. Essential for gradient descent, KNN, SVM, and PCA.
- **Train / validation / test split:** train to learn, validate to tune, test once for honest performance.
- **Hyperparameters:** settings chosen before training (`α`, `λ`, `C`, `k`, tree depth).

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m "Add: short description"`
4. Push the branch: `git push origin feature/your-idea`
5. Open a Pull Request describing what you changed and why.

Please keep new modules consistent with the existing layout: one folder, one script, one README with definition, formulas, and expected outcome.

---

## Author

**[Your Name]**
GitHub: [@your-username](https://github.com/your-username)
LinkedIn: [your-profile](https://www.linkedin.com/in/your-profile)

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

If you found this repository useful, consider giving it a star.
