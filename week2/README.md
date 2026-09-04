# Week 2: Supervised Learning — Regression & Classification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Pytest](https://img.shields.io/badge/Tested%20With-Pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Week 2 Syllabus & Objectives

This repository folder contains the complete, production-grade implementation of **Week 2: Supervised Learning (Regression & Classification)**.

```
WEEK 2 CURRICULUM
├── 🔍 Focus: Model building, training, evaluation, and accuracy testing
├── 🧠 Topics Covered:
│   ├── Linear Regression (Ordinary Least Squares, Normal Equation, Gradient Descent)
│   ├── Logistic Regression (Sigmoid, Odds-Ratio, Cross-Entropy Loss)
│   ├── Decision Trees & Random Forests (Gini Impurity, Entropy, Ensemble Bagging)
│   └── Accuracy & Evaluation Metrics (MSE, RMSE, MAE, R², Confusion Matrix, ROC-AUC)
└── 🚀 Hands-On Projects & Assignments:
    ├── Mini Project 2 & Assignment 1: House Price Prediction (Kaggle Housing Dataset)
    └── Assignment 2: Titanic Survival Classification (Logistic Regression & Ensembles)
```

---

## 🎓 Mathematical & Theoretical Foundations

### 1. Supervised Learning Overview
In **Supervised Learning**, algorithms learn a mapping function $f: \mathcal{X} \to \mathcal{Y}$ from labeled training pairs $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^m$:
- **Regression:** $y \in \mathbb{R}$ (Continuous numerical target, e.g., House Price).
- **Classification:** $y \in \{0, 1\}$ or $\{1, \dots, K\}$ (Discrete categorical class label, e.g., Titanic Survival).

---

### 2. Linear Regression (Continuous Target Estimation)
Linear regression models the relationship between target scalar $y$ and predictor vector $\mathbf{x} = [x_1, x_2, \dots, x_p]^T$:

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p = \mathbf{w}^T \mathbf{x} + b$$

#### A. Ordinary Least Squares (OLS) Cost Function
The optimal weights minimize the Mean Squared Error over $m$ training samples:

$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right)^2 = \frac{1}{2m} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$$

#### B. Closed-Form Normal Equation
Setting the gradient $\nabla_{\mathbf{w}} J(\mathbf{w}) = 0$ yields:

$$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

#### C. Gradient Descent Optimization
When matrix inversion is computationally expensive ($\mathcal{O}(p^3)$), weights update iteratively via learning rate $\alpha$:

$$w_j := w_j - \alpha \frac{\partial J(\mathbf{w})}{\partial w_j} = w_j - \alpha \frac{1}{m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}$$

---

### 3. Logistic Regression (Binary Classification)
Logistic Regression models the posterior probability $P(y=1|\mathbf{x})$ by passing the linear combination through the standard logistic (sigmoid) function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = \mathbf{w}^T \mathbf{x} + b$$

$$\hat{p} = P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

#### A. Odds and Log-Odds (Logit)
$$\text{Odds} = \frac{P(y=1|\mathbf{x})}{1 - P(y=1|\mathbf{x})} \implies \ln(\text{Odds}) = \mathbf{w}^T \mathbf{x} + b$$

#### B. Cost Function: Binary Cross-Entropy (Log-Loss)
Derived via Maximum Likelihood Estimation (MLE), the convex loss function is:

$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(\hat{p}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{p}^{(i)}) \right]$$

---

### 4. Non-Linear & Ensemble Methods

#### A. Decision Trees
A non-parametric supervised model that partitions feature space into hyper-rectangles using recursive binary splitting.
- **Classification Splitting Criteria**:
  - **Gini Impurity**: $I_G(t) = 1 - \sum_{k=1}^K p_k^2$
  - **Information Entropy**: $H(t) = -\sum_{k=1}^K p_k \log_2(p_k)$
- **Regression Splitting Criteria**:
  - Minimizes sample variance (Mean Squared Error): $\text{MSE}(t) = \frac{1}{|D_t|} \sum_{i \in D_t} (y_i - \bar{y}_t)^2$

#### B. Random Forests
An ensemble of $B$ decorrelated decision trees using **Bootstrap Aggregation (Bagging)** and random feature subspace sampling:
1. Sample $m$ observations with replacement from the training dataset.
2. At each node split, randomly choose a subset of $k = \sqrt{p}$ features.
3. Aggregate predictions:
   - **Regression:** $\hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$
   - **Classification:** $\hat{y} = \text{mode}\{T_1(\mathbf{x}), T_2(\mathbf{x}), \dots, T_B(\mathbf{x})\}$

---

### 5. Evaluation Metrics Glossary

| Metric | Formula / Definition | Use Case |
| :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | $\frac{1}{m} \sum \|y^{(i)} - \hat{y}^{(i)}\|$ | Robust to outliers |
| **MSE** (Mean Squared Error) | $\frac{1}{m} \sum (y^{(i)} - \hat{y}^{(i)})^2$ | Heavily penalizes large errors |
| **RMSE** (Root Mean Squared Error) | $\sqrt{\text{MSE}}$ | In original target units ($) |
| **$R^2$** (Coefficient of Determination) | $1 - \frac{\sum (y^{(i)} - \hat{y}^{(i)})^2}{\sum (y^{(i)} - \bar{y})^2}$ | Percentage of target variance explained |
| **Adjusted $R^2$** | $1 - \left[ \frac{(1 - R^2)(m - 1)}{m - p - 1} \right]$ | Penalizes unnecessary predictor variables |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correctness (balanced data) |
| **Precision** | $\frac{TP}{TP + FP}$ | Cost of False Positives is high |
| **Recall / Sensitivity** | $\frac{TP}{TP + FN}$ | Cost of False Negatives is high |
| **Specificity** | $\frac{TN}{TN + FP}$ | True Negative Rate |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of precision & recall |
| **ROC-AUC** | Area under True Positive Rate vs False Positive Rate curve | Threshold-independent discrimination power |

---

## 🏗️ Architecture & Module Organization

```text
week2/
├── data/
│   ├── Housing.csv                  # Kaggle Housing dataset (545 rows, 13 features)
│   └── titanic_cleaned.csv          # Preprocessed Titanic dataset (891 rows, 26 features)
│
├── src/
│   ├── __init__.py                  # Package exports
│   ├── housing_regression.py        # Linear Regression & Regressors pipeline
│   ├── titanic_classification.py    # Logistic Regression & Classifiers pipeline
│   └── model_utils.py               # Metrics, visualization, and model serialization
│
├── notebooks/
│   └── week2_supervised_learning.ipynb # Comprehensive Jupyter portfolio walkthrough
│
├── outputs/
│   ├── figures/                     # High-DPI (300 DPI) visualization charts
│   │   ├── housing_linear_regression_predicted_vs_actual.png
│   │   ├── housing_linear_regression_residuals.png
│   │   ├── housing_linear_regression_coefficients.png
│   │   ├── housing_random_forest_feature_importance.png
│   │   ├── titanic_logistic_regression_confusion_matrix.png
│   │   ├── titanic_random_forest_classifier_confusion_matrix.png
│   │   ├── titanic_roc_curves_comparison.png
│   │   ├── titanic_logistic_regression_coefficients.png
│   │   └── titanic_random_forest_feature_importance.png
│   ├── models/                      # Serialized trained models & scalers (.joblib)
│   │   ├── linear_regression.joblib
│   │   ├── ridge_regression.joblib
│   │   ├── decision_tree_regressor.joblib
│   │   ├── random_forest_regressor.joblib
│   │   ├── housing_scaler.joblib
│   │   ├── logistic_regression.joblib
│   │   ├── decision_tree_classifier.joblib
│   │   ├── random_forest_classifier.joblib
│   │   └── titanic_scaler.joblib
│   └── reports/                     # CSV tabular evaluation summaries
│       ├── housing_regression_benchmark.csv
│       └── titanic_classification_benchmark.csv
│
├── tests/
│   ├── __init__.py
│   └── test_week2_models.py         # Automated Pytest suite (8 unit tests)
│
├── main.py                          # CLI runner for the end-to-end pipeline
└── README.md                        # Documentation
```

---

## 📊 Experimental Results & Benchmark Analysis

### 1. Mini Project 2 & Assignment 1: House Price Prediction (Regression)

| Model | Train $R^2$ | Test $R^2$ | Test Adj $R^2$ | Test MAE ($) | Test RMSE ($) | Test MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **0.6859** | **0.6529** | **0.6054** | **$970,043** | **$1,324,507** | **21.04%** |
| **Ridge Regression** | 0.6859 | 0.6528 | 0.6053 | $969,858 | $1,324,703 | 21.03% |
| **Random Forest Regressor** | 0.8974 | 0.6019 | 0.5474 | $1,036,740 | $1,418,576 | 22.16% |
| **Decision Tree Regressor** | 0.7504 | 0.4654 | 0.3922 | $1,222,655 | $1,643,884 | 26.39% |

#### 🔑 Regression Insights:
- **Linear Regression achieved an $R^2$ of 0.6529**, explaining **65.29% of test set variance**.
- The most influential positive factors driving house prices are **lot area**, **number of bathrooms**, **stories**, **air conditioning**, and **preferred location area (`prefarea`)**.
- Unfurnished status is the strongest negative price driver compared to furnished homes.

---

### 2. Assignment 2: Titanic Survival Classification

| Model | Train Acc (%) | Test Acc (%) | Precision | Recall | F1-Score | Specificity | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **84.13%** | **84.92%** | **0.8281** | **0.7681** | **0.7970** | **0.9000** | **0.8734** |
| **Random Forest Classifier** | 87.64% | 82.12% | 0.7846 | 0.7391 | 0.7612 | 0.8727 | 0.8482 |
| **Decision Tree Classifier** | 86.38% | 80.45% | 0.8036 | 0.6522 | 0.7200 | 0.9000 | 0.8333 |

#### 🔑 Classification Insights:
- **Logistic Regression achieved 84.92% accuracy and 0.8734 ROC-AUC** on unseen test data.
- **Title (`Title_Mr` vs `Title_Mrs`/`Title_Miss`)**, **Passenger Class (`Pclass`)**, and **Gender (`Sex`)** remain the most statistically dominant predictive signals.
- **Specificity of 90.0%** confirms the model excels at correctly identifying passengers who did not survive.

---

## 🖼️ Visual Gallery

All generated high-resolution charts (300 DPI) are saved in [`week2/outputs/figures/`](file:///c:/Users/GOKULNATH/Desktop/Online%20ML%20INTERN/week2/outputs/figures):

1. **`housing_linear_regression_predicted_vs_actual.png`**: Predicted vs Actual scatter plot showcasing model fit alignment along the identity line ($y = x$).
2. **`housing_linear_regression_residuals.png`**: Residuals vs Fitted values & normal error distribution diagnostics.
3. **`housing_linear_regression_coefficients.png`**: Standardized feature coefficients showing feature impacts.
4. **`titanic_logistic_regression_confusion_matrix.png`**: Breakdown of True Positives, True Negatives, False Positives, and False Negatives.
5. **`titanic_roc_curves_comparison.png`**: Multi-model ROC comparison with area-under-the-curve metrics.
6. **`titanic_random_forest_feature_importance.png`**: Non-linear feature importances calculated via Mean Decrease in Impurity (MDI).

---

## 🚀 Quickstart & Usage

### 1. Run the Complete Week 2 Pipeline
To train all models, generate diagnostic plots, save joblib models, and produce benchmark tables:

```bash
python week2/main.py
```

### 2. Run Automated Pytest Suite
Run the 8 automated unit tests validating data integrity, regression $R^2$, classification accuracy, and model persistence:

```bash
pytest week2/tests/test_week2_models.py -v
```

### 3. Launch Interactive Jupyter Notebook
```bash
jupyter notebook week2/notebooks/week2_supervised_learning.ipynb
```
