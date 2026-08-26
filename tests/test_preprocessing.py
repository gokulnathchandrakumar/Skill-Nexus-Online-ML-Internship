"""
Pytest Unit Tests for Week 1 Titanic Data Preprocessing Pipeline.
Validates loading, cleaning, feature engineering, encoding, and scaling.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_titanic_data, inspect_data
from src.feature_engineering import engineer_features
from src.data_cleaning import clean_missing_values
from src.preprocessing import (
    encode_categorical_features,
    prepare_train_test_split,
    scale_features,
)


@pytest.fixture
def raw_data_path() -> Path:
    """Fixture returning path to raw CSV."""
    return PROJECT_ROOT / "data" / "raw" / "train.csv"


@pytest.fixture
def loaded_df(raw_data_path: Path) -> pd.DataFrame:
    """Fixture returning loaded raw DataFrame."""
    return load_titanic_data(raw_data_path)


# Test 1: Dataset loads successfully
def test_data_loading(raw_data_path: Path):
    df = load_titanic_data(raw_data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 891


# Test 2: Required original columns exist
def test_required_columns(loaded_df: pd.DataFrame):
    required_cols = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    for col in required_cols:
        assert col in loaded_df.columns, f"Required column {col} missing from dataset."


# Test 3: Target column exists and is binary
def test_target_column(loaded_df: pd.DataFrame):
    assert "Survived" in loaded_df.columns
    unique_vals = set(loaded_df["Survived"].unique())
    assert unique_vals.issubset({0, 1})


# Test 4: Missing-value processing handles NaNs
def test_missing_values_handled(loaded_df: pd.DataFrame):
    df_engineered = engineer_features(loaded_df)
    df_cleaned = clean_missing_values(df_engineered)

    assert df_cleaned["Age"].isnull().sum() == 0, "Age column still contains missing values."
    assert df_cleaned["Embarked"].isnull().sum() == 0, "Embarked column still contains missing values."
    assert df_cleaned.isnull().sum().sum() == 0, "Cleaned dataset contains unexpected NaN values."


# Test 5: Feature engineering creates expected columns
def test_feature_engineering(loaded_df: pd.DataFrame):
    df_engineered = engineer_features(loaded_df)

    assert "FamilySize" in df_engineered.columns
    assert "IsAlone" in df_engineered.columns
    assert "Title" in df_engineered.columns

    # Verify logic
    assert (df_engineered["FamilySize"] == df_engineered["SibSp"] + df_engineered["Parch"] + 1).all()
    assert set(df_engineered["IsAlone"].unique()).issubset({0, 1})
    assert set(df_engineered["Title"].unique()).issubset({"Mr", "Mrs", "Miss", "Master", "Rare"})


# Test 6: Preprocessing dimensions and split proportion
def test_processed_data_dimensions(loaded_df: pd.DataFrame):
    df_engineered = engineer_features(loaded_df)
    df_cleaned = clean_missing_values(df_engineered)
    df_encoded, _ = encode_categorical_features(df_cleaned)

    X_train, X_test, y_train, y_test = prepare_train_test_split(df_encoded, target_col="Survived", test_size=0.2)

    # 80% of 891 is 712, 20% of 891 is 179
    assert len(X_train) + len(X_test) == len(loaded_df)
    assert len(X_train) == 712
    assert len(X_test) == 179
    assert len(y_train) == 712
    assert len(y_test) == 179


# Test 7: Feature scaling and zero NaN verification
def test_feature_scaling_no_nans(loaded_df: pd.DataFrame):
    df_engineered = engineer_features(loaded_df)
    df_cleaned = clean_missing_values(df_engineered)
    df_encoded, _ = encode_categorical_features(df_cleaned)

    X_train, X_test, y_train, y_test = prepare_train_test_split(df_encoded, target_col="Survived", test_size=0.2)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, num_cols=["Age", "Fare", "FamilySize"])

    assert not X_train_scaled.isnull().values.any(), "Scaled X_train contains NaNs."
    assert not X_test_scaled.isnull().values.any(), "Scaled X_test contains NaNs."

    # Check scaled mean near 0 and std near 1 for training set
    assert abs(X_train_scaled["Age"].mean()) < 1e-5
    assert abs(X_train_scaled["Age"].std() - 1.0) < 1e-2
