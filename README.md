# Titanic Survival Prediction — Data Cleaning & Preprocessing (Week 1 ML Project)

[![Python 3.10](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Pytest](https://img.shields.io/badge/Tested%20With-Pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview

This repository contains a professional, portfolio-grade implementation of **Week 1: Machine Learning Fundamentals & Data Preprocessing**. 

The goal of this project is to take the raw **Kaggle Titanic Survival Dataset** (`train.csv`), perform thorough exploratory data analysis (EDA), handle missing values intelligently, engineer high-signal domain features, encode categorical variables, scale numerical features without data leakage, and produce a fully cleaned dataset ready for predictive ML modeling.

---

## 🎓 Machine Learning Fundamentals

### What is Machine Learning?
**Machine Learning (ML)** is a subfield of Artificial Intelligence (AI) focused on building algorithms that learn patterns from data and improve their predictive accuracy over time without being explicitly programmed with hardcoded rules.

---

### Types of Machine Learning

```
                       ┌─────────────────────────┐
                       │    Machine Learning     │
                       └────────────┬────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│   Supervised    │        │  Unsupervised   │        │  Reinforcement  │
│    Learning     │        │    Learning     │        │    Learning     │
└────────┬────────┘        └────────┬────────┘        └────────┬────────┘
         │                          │                          │
  ┌──────┴──────┐            ┌──────┴──────┐            ┌──────┴──────┐
  ▼             ▼            ▼             ▼            ▼             ▼
Classification Regression Clustering Dimensionality  Agent       Environment
(Titanic)      (Housing)  (Customers)  Reduction   (Self-Driving)  (State/Reward)
```

#### 1. Supervised Learning
In Supervised Learning, the algorithm learns from a labeled dataset containing both input features ($X$) and ground-truth targets ($y$).
- **Classification:** Predicts discrete categories or labels.
  - *Example:* **Titanic Survival Prediction** — Predicting whether a passenger survived ($y=1$) or did not survive ($y=0$).
- **Regression:** Predicts continuous numerical values.
  - *Example:* Predicting house prices based on square footage and location.

#### 2. Unsupervised Learning
In Unsupervised Learning, the model analyzes unlabeled data ($X$) to discover hidden patterns, groupings, or structural representations without human annotations.
- **Clustering:** Grouping similar data points together.
  - *Example:* Customer market segmentation based on purchasing behavior.
- **Dimensionality Reduction:** Compressing high-dimensional feature spaces while preserving essential variance.
  - *Example:* Principal Component Analysis (PCA) or t-SNE for dataset visualization.

#### 3. Reinforcement Learning (RL)
Reinforcement Learning involves an **Agent** interacting with a dynamic **Environment**. The agent takes an **Action**, observes the updated state, and receives a **Reward** or penalty. Through trial and error, the agent learns an optimal policy to maximize cumulative rewards.
- *Real-World Example:* Automated trading bots or self-driving cars adjusting steering angle and acceleration based on environmental sensors and collision rewards/penalties.

---

## 🏗️ Project Architecture & Directory Structure

```text
c:\Users\GOKULNATH\Desktop\Online ML INTERN\
├── data/
│   ├── raw/
│   │   └── train.csv                # Raw Titanic training dataset (891 rows, 12 columns)
│   └── processed/
│       └── titanic_cleaned.csv      # Cleaned, engineered & ML-ready dataset
│
├── notebooks/
│   └── week1_titanic_analysis.ipynb # End-to-end Jupyter Notebook portfolio walkthrough
│
├── src/
│   ├── __init__.py                  # Package initializer
│   ├── data_loader.py               # Dataset loading & automated quality reporting
│   ├── data_cleaning.py             # Imputation strategies (Age median, Embarked mode, Cabin Deck)
│   ├── feature_engineering.py       # Domain features (FamilySize, IsAlone, Title extraction)
│   ├── preprocessing.py             # Scikit-Learn encoding, scaling & stratified train/test split
│   └── visualization.py             # Plot generation & figure export module
│
├── outputs/
│   ├── figures/                     # Generated high-DPI exploratory charts
│   │   ├── age_distribution.png
│   │   ├── survival_distribution.png
│   │   ├── survival_by_gender.png
│   │   ├── survival_by_class.png
│   │   └── correlation_heatmap.png
│   ├── reports/
│   │   └── data_quality_report.txt  # Automated text inspection summary
│   └── titanic_cleaned.csv          # Cleaned CSV output copy
│
├── tests/
│   ├── __init__.py                  # Test package initializer
│   └── test_preprocessing.py        # Automated Pytest suite (7 test cases)
│
├── main.py                          # Main Python execution pipeline script
├── requirements.txt                 # Project dependencies
├── README.md                        # Documentation & Learning Guide
├── .gitignore                       # Version control exclusion rules
└── LICENSE                          # Open-source MIT License
```

---

## 🛠️ Data Preprocessing & Engineering Workflow

```text
Raw Dataset (train.csv)
          │
          ▼
   Load & Inspect Dataset (.info, .describe)
          │
          ▼
   Generate Data Quality Report
          │
          ▼
   Missing Value Handling (Age Median by Title/Pclass, Embarked Mode, Cabin Deck)
          │
          ▼
   Feature Engineering (Title Extraction, FamilySize, IsAlone, Cabin_Known)
          │
          ▼
   Categorical Encoding (Sex → LabelEncoder, Embarked/Title/Deck → OneHotEncoder)
          │
          ▼
   Stratified Train/Test Split (80% Train / 20% Test, stratify=Survived)
          │
          ▼
   Feature Scaling (StandardScaler fit on Train ONLY → zero leakage)
          │
          ▼
   Visual Analytics & Chart Export
          │
          ▼
   Export Cleaned Dataset (titanic_cleaned.csv)
```

### 1. Data Quality Analysis & Missing Data Strategy
- **Row Count:** 891 passengers
- **Column Count:** 12 attributes
- **Missing Value Handling Strategy:**
  - **`Age` (177 missing / 19.8%):** Instead of global mean filling, missing ages are imputed using the **median age grouped by passenger `Title` and `Pclass`**, preserving demographic nuance.
  - **`Embarked` (2 missing / 0.22%):** Imputed with mode (`'S'`).
  - **`Cabin` (687 missing / 77.1%):** High missingness is transformed into a high-signal binary feature `Cabin_Known` ($1$ if cabin recorded, $0$ otherwise) and `Deck` feature extracted from the cabin prefix letter (`'Unknown'` for missing). Rows are **never** blindly deleted!

### 2. Feature Engineering
- **`FamilySize`:** `SibSp` + `Parch` + 1
- **`IsAlone`:** Binary indicator ($1$ if `FamilySize` == 1 else $0$).
- **`Title`:** Extracted from passenger names (`Mr`, `Mrs`, `Miss`, `Master`, `Rare`).

### 3. Categorical Encoding
- **Binary Features (`Sex`):** Transformed via `LabelEncoder` (`male`: 1, `female`: 0).
- **Nominal Multi-Category Features (`Embarked`, `Title`, `Deck`):** Encoded using Scikit-Learn `OneHotEncoder` with `handle_unknown='ignore'`.

### 4. Feature Scaling & Zero Data Leakage
- `StandardScaler` is applied to numerical features (`Age`, `Fare`, `FamilySize`).
- **Data Leakage Prevention:** The scaler is fitted **STRICTLY** on training features (`X_train`) and only transforms test features (`X_test`).

---

## 📊 Visualizations Generated

All plots are automatically generated and saved to `outputs/figures/` in high resolution (300 DPI):

| Chart File | Description | Key Insight |
| :--- | :--- | :--- |
| [`age_distribution.png`](file:///c:/Users/GOKULNATH/Desktop/Online%20ML%20INTERN/outputs/figures/age_distribution.png) | Histogram + KDE of passenger age | Bimodal distribution peaking around young adults (20–30 years) with an infant spike. |
| [`survival_distribution.png`](file:///c:/Users/GOKULNATH/Desktop/Online%20ML%20INTERN/outputs/figures/survival_distribution.png) | Overall survival counts | 549 passengers did not survive (61.6%), while 342 survived (38.4%). |
| [`survival_by_gender.png`](file:///c:/Users/GOKULNATH/Desktop/Online%20ML%20INTERN/outputs/figures/survival_by_gender.png) | Survival breakdown by gender | Female survival rate (~74%) significantly surpassed male survival rate (~19%). |
| [`survival_by_class.png`](file:///c:/Users/GOKULNATH/Desktop/Online%20ML%20INTERN/outputs/figures/survival_by_class.png) | Survival breakdown by class | 1st Class passengers had highest survival rate (>62%), while 3rd Class suffered highest mortality (>75%). |
| [`correlation_heatmap.png`](file:///c:/Users/GOKULNATH/Desktop/Online%20ML%20INTERN/outputs/figures/correlation_heatmap.png) | Heatmap of numerical feature correlations | Strong correlation between `Pclass` & `Fare`, as well as `Fare` & `Survived`. |

---

## 🚀 How to Run the Project

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

### 2. Set Up Virtual Environment

#### Windows (PowerShell / Command Prompt)
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Execute Main Pipeline
Run the main modular script to process data, generate figures, and export cleaned CSVs:
```bash
python main.py
```

### 5. Run Automated Tests
Execute Pytest suite to verify dataset loading, cleaning, feature creation, and scaling:
```bash
pytest -v
```

### 6. Run Jupyter Notebook
Launch interactive walkthrough:
```bash
jupyter notebook notebooks/week1_titanic_analysis.ipynb
```

---

## 🧪 Automated Testing Verification

The project includes 7 unit tests in `tests/test_preprocessing.py`:

- [x] **Test 1 (`test_data_loading`):** Validates raw dataset loads 891 rows successfully.
- [x] **Test 2 (`test_required_columns`):** Ensures all original columns are present.
- [x] **Test 3 (`test_target_column`):** Verifies target binary format (`Survived` $\in \{0, 1\}$).
- [x] **Test 4 (`test_missing_values_handled`):** Confirms zero NaNs remain after imputation.
- [x] **Test 5 (`test_feature_engineering`):** Tests calculation logic for `FamilySize`, `IsAlone`, and `Title`.
- [x] **Test 6 (`test_processed_data_dimensions`):** Verifies 80/20 train/test split shape (712 train, 179 test).
- [x] **Test 7 (`test_feature_scaling_no_nans`):** Confirms zero data leakage and scaled bounds.

---

## 📈 Key Learnings & Future Improvements

### Key Learnings
1. **Domain Imputation over Row Dropping:** Imputing missing values using grouped statistics (such as median age per title/class) preserves statistical power.
2. **Preventing Data Leakage:** Preprocessing parameters (scaler mean/variance, encoder maps) must be fit exclusively on training data.
3. **High-Signal Features:** Abstracting `Name` into `Title` and `Cabin` into `Cabin_Known`/`Deck` creates actionable features for downstream estimators.

### Future Improvements
- Implement automated outlier detection for `Fare`.
- Experiment with hyperparameter-tuned classifiers (Logistic Regression, Random Forest, XGBoost) in Week 2.
- Build an automated MLflow pipeline for model artifact tracking.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
