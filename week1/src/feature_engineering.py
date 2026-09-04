"""
Feature Engineering Module for Titanic Dataset.
Creates domain-specific features to improve predictive capability.
"""

import pandas as pd
import numpy as np


def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts passenger titles from the 'Name' column and categorizes them.

    Args:
        df (pd.DataFrame): Dataset containing 'Name' column.

    Returns:
        pd.DataFrame: Dataset with new 'Title' column.
    """
    processed_df = df.copy()
    if "Name" in processed_df.columns:
        # Extract title using regular expression
        extracted_titles = processed_df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Standardize titles
        title_mapping = {
            "Mr": "Mr",
            "Miss": "Miss",
            "Mrs": "Mrs",
            "Master": "Master",
            "Dr": "Rare",
            "Rev": "Rare",
            "Col": "Rare",
            "Major": "Rare",
            "Mlle": "Miss",
            "Countess": "Rare",
            "Ms": "Miss",
            "Lady": "Rare",
            "Jonkheer": "Rare",
            "Don": "Rare",
            "Dona": "Rare",
            "Mme": "Mrs",
            "Capt": "Rare",
            "Sir": "Rare",
        }

        processed_df["Title"] = extracted_titles.map(title_mapping).fillna("Rare")

    return processed_df


def create_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates family size and isolation indicator features.

    Features engineered:
    - FamilySize = SibSp + Parch + 1
    - IsAlone = 1 if FamilySize == 1 else 0

    Args:
        df (pd.DataFrame): Dataset containing 'SibSp' and 'Parch' columns.

    Returns:
        pd.DataFrame: Dataset with 'FamilySize' and 'IsAlone' columns.
    """
    processed_df = df.copy()

    if "SibSp" in processed_df.columns and "Parch" in processed_df.columns:
        processed_df["FamilySize"] = processed_df["SibSp"] + processed_df["Parch"] + 1
        processed_df["IsAlone"] = (processed_df["FamilySize"] == 1).astype(int)

    return processed_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline function to apply all feature engineering steps.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Feature-engineered dataset.
    """
    engineered_df = df.copy()
    engineered_df = extract_title(engineered_df)
    engineered_df = create_family_features(engineered_df)
    return engineered_df
