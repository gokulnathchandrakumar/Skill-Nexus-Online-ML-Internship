"""
Preprocessing Module for Categorical Encoding, Feature Scaling, and Train/Test Splitting.
Ensures zero data leakage and professional Scikit-Learn integration.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encodes categorical features using LabelEncoder for binary features
    and OneHotEncoder for multi-class nominal features.

    Args:
        df (pd.DataFrame): Preprocessed dataset.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Encoded DataFrame and encoder metadata.
    """
    encoded_df = df.copy()
    encoders = {}

    # 1. Label Encoding for Binary Feature: Sex
    if "Sex" in encoded_df.columns:
        label_encoder = LabelEncoder()
        encoded_df["Sex"] = label_encoder.fit_transform(encoded_df["Sex"])
        encoders["Sex_LabelEncoder"] = label_encoder

    # 2. One-Hot Encoding for Nominal Features: Embarked, Title, Deck (if present)
    ohe_cols = [col for col in ["Embarked", "Title", "Deck"] if col in encoded_df.columns]

    if ohe_cols:
        ohe = OneHotEncoder(sparse_output=False, drop=None, handle_unknown="ignore")
        ohe_features = ohe.fit_transform(encoded_df[ohe_cols])
        feature_names = ohe.get_feature_names_out(ohe_cols)

        ohe_df = pd.DataFrame(ohe_features, columns=feature_names, index=encoded_df.index)

        # Drop original nominal columns and concat OHE features
        encoded_df = encoded_df.drop(columns=ohe_cols)
        encoded_df = pd.concat([encoded_df, ohe_df], axis=1)
        encoders["OneHotEncoder"] = ohe

    return encoded_df, encoders


def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str = "Survived",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features (X) and target (y) into stratified train and test sets.

    Args:
        df (pd.DataFrame): Preprocessed dataset containing target column.
        target_col (str): Target column name. Default 'Survived'.
        test_size (float): Proportion of dataset for test set. Default 0.2.
        random_state (int): Random seed for reproducibility. Default 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # Drop non-predictive identifiers or raw text columns if still present
    cols_to_drop = [c for c in ["PassengerId", "Name", "Ticket"] if c in df.columns]
    df_model = df.drop(columns=cols_to_drop)

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, num_cols: list = None
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scales numerical features using StandardScaler.
    Prevents data leakage by fitting ONLY on X_train.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        num_cols (list, optional): List of numerical column names to scale.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]: Scaled X_train, Scaled X_test, Fitted Scaler.
    """
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if num_cols is None:
        num_cols = ["Age", "Fare", "FamilySize"]
        num_cols = [c for c in num_cols if c in X_train.columns]

    scaler = StandardScaler()

    # FIT ONLY ON TRAINING DATA
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])

    # TRANSFORM TEST DATA USING TRAINED SCALER
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, scaler


def export_cleaned_data(df: pd.DataFrame, output_path: Path) -> Path:
    """
    Exports the cleaned and processed dataset to CSV format.

    Args:
        df (pd.DataFrame): Processed dataset.
        output_path (Path): Path to output CSV file.

    Returns:
        Path: Path to saved file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
