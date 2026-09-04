"""
Data Cleaning Module for Handling Missing Values and Data Imputation.
"""

from typing import Optional
import pandas as pd
import numpy as np


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans missing values in the dataset using statistically sound imputation strategies.

    Strategies applied:
    - Age: Imputed using median age grouped by passenger Title and Pclass.
    - Embarked: Imputed using mode ('S').
    - Cabin: High missingness (~77%) converted into Deck feature ('Unknown' for missing)
      and binary feature 'Cabin_Known' (1 if cabin recorded, 0 otherwise).

    Args:
        df (pd.DataFrame): Raw or partially preprocessed dataset.

    Returns:
        pd.DataFrame: Cleaned dataset with zero missing values.
    """
    cleaned_df = df.copy()

    # 1. Handle Embarked: Mode Imputation
    if "Embarked" in cleaned_df.columns:
        mode_embarked = cleaned_df["Embarked"].mode()[0] if not cleaned_df["Embarked"].mode().empty else "S"
        cleaned_df["Embarked"] = cleaned_df["Embarked"].fillna(mode_embarked)

    # 2. Handle Cabin: Intelligent Feature Extraction
    if "Cabin" in cleaned_df.columns:
        cleaned_df["Cabin_Known"] = cleaned_df["Cabin"].notnull().astype(int)
        cleaned_df["Deck"] = cleaned_df["Cabin"].apply(
            lambda x: str(x)[0].upper() if pd.notnull(x) and len(str(x)) > 0 else "Unknown"
        )
        # Drop original raw Cabin column to avoid sparse high-cardinality issues
        cleaned_df = cleaned_df.drop(columns=["Cabin"])

    # 3. Handle Age: Median Imputation by Title/Pclass or Overall Median
    if "Age" in cleaned_df.columns:
        # Extract title temporarily if available for granular median calculation
        if "Title" in cleaned_df.columns:
            cleaned_df["Age"] = cleaned_df.groupby(["Title", "Pclass"])["Age"].transform(
                lambda x: x.fillna(x.median())
            )
        elif "Name" in cleaned_df.columns:
            temp_titles = cleaned_df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
            cleaned_df["Age"] = cleaned_df.groupby([temp_titles, cleaned_df["Pclass"]])["Age"].transform(
                lambda x: x.fillna(x.median())
            )

        # Fallback median fill if any NaN remains
        overall_median_age = cleaned_df["Age"].median()
        cleaned_df["Age"] = cleaned_df["Age"].fillna(overall_median_age)

    # 4. Handle Fare if any missing values exist
    if "Fare" in cleaned_df.columns and cleaned_df["Fare"].isnull().sum() > 0:
        median_fare = cleaned_df.groupby("Pclass")["Fare"].transform("median")
        cleaned_df["Fare"] = cleaned_df["Fare"].fillna(median_fare)

    return cleaned_df
