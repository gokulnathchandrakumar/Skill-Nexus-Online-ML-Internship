# Skill Nexus — Online Machine Learning Internship

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Pytest](https://img.shields.io/badge/Tested%20With-Pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Internship Overview

Welcome to the **Skill Nexus Online Machine Learning Internship** portfolio repository. This repository houses all weekly assignments, mini projects, production-grade source code, diagnostic figures, serialized machine learning models, and automated test suites developed throughout the program.

The repository is modularly organized into independent, self-contained weekly modules:

```
Skill Nexus Online ML Internship
├── 📂 week1/  ->  Machine Learning Fundamentals & Data Preprocessing
└── 📂 week2/  ->  Supervised Learning (Regression & Classification)
```

---

## 🗺️ Internship Curriculum & Roadmap

```
                          Skill Nexus ML Internship
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│            WEEK 1             │               │            WEEK 2             │
│   ML Fundamentals & EDA &     │               │      Supervised Learning      │
│      Data Preprocessing       │               │  (Regression & Classification)│
├───────────────────────────────┤               ├───────────────────────────────┤
│ • Types of Machine Learning   │               │ • Linear Regression (OLS)     │
│ • Data Imputation & Cleaning  │               │ • Logistic Regression         │
│ • Domain Feature Engineering  │               │ • Decision Trees & RF         │
│ • One-Hot & Label Encoding    │               │ • Mini Project 2: House Price │
│ • Non-Leaking Standardization │               │ • Assignment 2: Titanic Class │
└──────────────┬────────────────┘               └──────────────┬────────────────┘
               ▼                                               ▼
     [ Explore Week 1 ]                              [ Explore Week 2 ]
     (week1/README.md)                               (week2/README.md)
```

---

## 📂 Repository Architecture

```text
.
├── week1/                           # Week 1: ML Fundamentals & Data Preprocessing
│   ├── data/
│   │   ├── raw/train.csv            # Raw Kaggle Titanic Dataset (891 rows)
│   │   └── processed/               # Cleaned & scaled dataset (titanic_cleaned.csv)
│   ├── src/                         # Modular Python data preprocessing engine
│   │   ├── data_loader.py           # Inspection & data quality reports
│   │   ├── data_cleaning.py         # Median/Mode/Deck imputation strategies
│   │   ├── feature_engineering.py   # FamilySize, IsAlone, and Title extraction
│   │   ├── preprocessing.py         # Encoding, stratified split, and scaling
│   │   └── visualization.py         # High-DPI EDA figure generator
│   ├── notebooks/
│   │   └── week1_titanic_analysis.ipynb # Jupyter Portfolio walkthrough
│   ├── outputs/                     # Generated charts, quality reports & CSVs
│   ├── tests/
│   │   └── test_preprocessing.py    # Automated Pytest suite (7 test cases)
│   ├── main.py                      # Week 1 CLI pipeline entrypoint
│   └── README.md                    # Dedicated Week 1 documentation
│
├── week2/                           # Week 2: Supervised Learning (Regression & Classification)
│   ├── data/
│   │   ├── Housing.csv              # Kaggle Housing Dataset (545 rows, 13 features)
│   │   └── titanic_cleaned.csv      # Cleaned Titanic Classification dataset
│   ├── src/                         # Modular model training & evaluation pipelines
│   │   ├── housing_regression.py    # Linear Regression, Decision Tree, Random Forest
│   │   ├── titanic_classification.py# Logistic Regression, Decision Tree, Random Forest
│   │   └── model_utils.py           # Metrics calculation, plotting & model serialization
│   ├── notebooks/
│   │   └── week2_supervised_learning.ipynb # Interactive portfolio notebook with math derivations
│   ├── outputs/
│   │   ├── figures/                 # Predicted vs Actual, Residuals, ROC curves, CM
│   │   ├── models/                  # Serialized .joblib model artifacts & scalers
│   │   └── reports/                 # Tabular CSV benchmark performance metrics
│   ├── tests/
│   │   └── test_week2_models.py     # Automated Pytest suite (9 test cases)
│   ├── main.py                      # Week 2 CLI pipeline entrypoint
│   └── README.md                    # Dedicated Week 2 documentation
│
├── requirements.txt                 # Project dependencies
├── LICENSE                          # MIT License
└── README.md                        # Master repository documentation
```

---

## 📊 Summary of Projects & Benchmarks

### Week 1: Data Preprocessing & Feature Engineering (Titanic)
- **Dataset:** 891 raw passenger records across 12 features.
- **Engineered Features:** Extracted titles (`Mr`, `Mrs`, `Miss`, `Master`), family dynamics (`FamilySize`, `IsAlone`), and reconstructed cabin deck categories.
- **Data Quality:** 0 remaining missing values, 0 data leakage across train/test splits.

### Week 2: Supervised Regression & Classification

#### 1. Mini Project 2 & Assignment 1: House Price Prediction (Regression)

| Model | Test $R^2$ | Test Adj $R^2$ | Test MAE ($) | Test RMSE ($) | Test MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | **0.6529** | **0.6054** | **$970,043** | **$1,324,507** | **21.04%** |
| **Ridge Regression** | 0.6528 | 0.6053 | $969,858 | $1,324,703 | 21.03% |
| **Random Forest Regressor** | 0.6019 | 0.5474 | $1,036,740 | $1,418,576 | 22.16% |
| **Decision Tree Regressor** | 0.4654 | 0.3922 | $1,222,655 | $1,643,884 | 26.39% |

#### 2. Assignment 2: Titanic Survival Classification

| Model | Test Accuracy (%) | Precision | Recall | F1-Score | Specificity | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **84.92%** | **0.8281** | **0.7681** | **0.7970** | **0.9000** | **0.8734** |
| **Random Forest Classifier** | 82.12% | 0.7846 | 0.7391 | 0.7612 | 0.8727 | 0.8482 |
| **Decision Tree Classifier** | 80.45% | 0.8036 | 0.6522 | 0.7200 | 0.9000 | 0.8333 |

---

## ⚡ Quick Start & Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/gokulnathchandrakumar/skill-nexus-online-ml-intern.git
cd skill-nexus-online-ml-intern

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Run Weekly Pipelines
```bash
# Execute Week 1 pipeline (Data Cleaning, EDA, Feature Engineering)
python week1/main.py

# Execute Week 2 pipeline (Model Training, Evaluation, Metric Reports)
python week2/main.py
```

### 3. Run Automated Tests
```bash
# Run all 16 unit tests across week1 and week2
pytest -v
```

---

## 👤 Author
**Gokulnath Chandrakumar**  
*Skill Nexus Online Machine Learning Intern*  
GitHub: [@gokulnathchandrakumar](https://github.com/gokulnathchandrakumar)

---

## 📄 License
This repository is licensed under the [MIT License](LICENSE).
