"""Automated Pytest Suite for Week 2 Supervised Learning (Regression and Classification).

Validates:
- Housing data loading and categorical encoding
- Linear Regression, Decision Tree, and Random Forest regressor performance and shapes
- Titanic classification feature preparation and stratified splitting
- Logistic Regression, Decision Tree, and Random Forest classification metrics
- Model serialization (save/load) fidelity
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from week2.src.housing_regression import HousingRegressionPipeline
from week2.src.model_utils import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    load_model,
    save_model,
)
from week2.src.titanic_classification import TitanicClassificationPipeline


@pytest.fixture
def housing_pipeline():
    data_path = PROJECT_ROOT / "week2" / "data" / "Housing.csv"
    pipeline = HousingRegressionPipeline(data_path=data_path, random_state=42)
    return pipeline


@pytest.fixture
def titanic_pipeline():
    data_path = PROJECT_ROOT / "week2" / "data" / "titanic_cleaned.csv"
    pipeline = TitanicClassificationPipeline(data_path=data_path, random_state=42)
    return pipeline


# ----------------------------------------------------------------------
# 1. Housing Regression Tests (Mini Project 2 & Assignment 1)
# ----------------------------------------------------------------------
def test_housing_data_loading(housing_pipeline):
    df = housing_pipeline.load_data()
    assert df is not None
    assert len(df) == 545
    assert "price" in df.columns
    assert "area" in df.columns
    assert "furnishingstatus" in df.columns


def test_housing_preprocessing(housing_pipeline):
    X, y = housing_pipeline.preprocess_data()
    assert X.shape[0] == 545
    assert y.shape[0] == 545
    assert "price" not in X.columns
    assert not X.isnull().any().any(), "Preprocessed features should contain no nulls"
    # Ensure binary columns are encoded as 0 or 1
    for col in housing_pipeline.BINARY_COLS:
        unique_vals = set(X[col].unique())
        assert unique_vals.issubset({0, 1})


def test_housing_linear_regression_metrics(housing_pipeline):
    housing_pipeline.train_models()
    lr_metrics = housing_pipeline.metrics_test.get("Linear Regression")

    assert lr_metrics is not None
    assert lr_metrics["R2"] > 0.60, f"Expected R² > 0.60, got {lr_metrics['R2']}"
    assert lr_metrics["RMSE"] > 0
    assert lr_metrics["MAE"] > 0
    assert lr_metrics["MAPE(%)"] < 30.0


def test_housing_tree_regressors(housing_pipeline):
    housing_pipeline.train_models()
    assert "Decision Tree Regressor" in housing_pipeline.models
    assert "Random Forest Regressor" in housing_pipeline.models

    rf_metrics = housing_pipeline.metrics_test.get("Random Forest Regressor")
    assert rf_metrics is not None
    assert rf_metrics["R2"] > 0.50
    assert len(housing_pipeline.predictions["Random Forest Regressor"]) == len(housing_pipeline.y_test)


# ----------------------------------------------------------------------
# 2. Titanic Classification Tests (Assignment 2)
# ----------------------------------------------------------------------
def test_titanic_data_loading_and_prep(titanic_pipeline):
    X, y = titanic_pipeline.prepare_features()
    assert X.shape[0] == 891
    assert y.shape[0] == 891
    assert "Survived" not in X.columns
    assert not X.isnull().any().any(), "Features should have no missing values"


def test_titanic_logistic_regression_metrics(titanic_pipeline):
    titanic_pipeline.train_models()
    lr_metrics = titanic_pipeline.metrics_test.get("Logistic Regression")

    assert lr_metrics is not None
    assert lr_metrics["Accuracy"] >= 0.75, f"Expected Accuracy >= 0.75, got {lr_metrics['Accuracy']}"
    assert lr_metrics["ROC_AUC"] >= 0.80, f"Expected ROC-AUC >= 0.80, got {lr_metrics['ROC_AUC']}"
    assert 0.0 <= lr_metrics["Precision"] <= 1.0
    assert 0.0 <= lr_metrics["Recall"] <= 1.0
    assert 0.0 <= lr_metrics["F1_Score"] <= 1.0


def test_titanic_classification_models_predict(titanic_pipeline):
    titanic_pipeline.train_models()
    for name in ["Decision Tree Classifier", "Random Forest Classifier"]:
        assert name in titanic_pipeline.models
        m = titanic_pipeline.metrics_test[name]
        assert m["Accuracy"] >= 0.75
        assert len(titanic_pipeline.predictions[name]) == len(titanic_pipeline.y_test)
        assert len(titanic_pipeline.probabilities[name]) == len(titanic_pipeline.y_test)


# ----------------------------------------------------------------------
# 3. Model Serialization & Utility Function Tests
# ----------------------------------------------------------------------
def test_model_serialization(tmp_path, housing_pipeline):
    housing_pipeline.train_models()
    lr = housing_pipeline.models["Linear Regression"]

    save_file = tmp_path / "test_lr_model.joblib"
    save_model(lr, save_file)
    assert save_file.exists()

    loaded_lr = load_model(save_file)
    preds_orig = lr.predict(housing_pipeline.X_test_scaled)
    preds_loaded = loaded_lr.predict(housing_pipeline.X_test_scaled)

    np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)


def test_metric_helpers():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 31.0, 39.0])
    reg_metrics = calculate_regression_metrics(y_true, y_pred, n_features=2)
    assert reg_metrics["MAE"] == 1.0
    assert reg_metrics["MSE"] == 1.0
    assert reg_metrics["RMSE"] == 1.0
    assert reg_metrics["R2"] > 0.95

    y_cls_true = np.array([0, 0, 1, 1])
    y_cls_pred = np.array([0, 1, 1, 1])
    cls_metrics = calculate_classification_metrics(y_cls_true, y_cls_pred)
    assert cls_metrics["Accuracy"] == 0.75
    assert cls_metrics["True_Positives"] == 2
    assert cls_metrics["False_Positives"] == 1
