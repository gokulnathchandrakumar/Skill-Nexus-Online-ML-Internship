"""
Data Loader Module for Titanic Dataset Exploration and Quality Reporting.
"""

from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np


def load_titanic_data(filepath: Path) -> pd.DataFrame:
    """
    Loads the Titanic dataset from the specified CSV file path.

    Args:
        filepath (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the dataset is empty.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    return df


def inspect_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs initial exploration and statistical inspection on the dataset.

    Args:
        df (pd.DataFrame): Input Titanic dataset.

    Returns:
        Dict[str, Any]: Dictionary containing exploratory metrics.
    """
    num_rows, num_cols = df.shape
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0].to_dict()

    summary_stats = df.describe(include="all").to_dict()

    inspection_results = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "target_col": "Survived" if "Survived" in df.columns else None,
        "missing_values": missing_cols,
        "total_missing_cells": int(missing_values.sum()),
        "summary_stats": summary_stats,
    }

    return inspection_results


def generate_quality_report(df: pd.DataFrame, output_path: Path) -> str:
    """
    Generates a comprehensive text quality report and writes it to a file.

    Args:
        df (pd.DataFrame): Dataset to analyze.
        output_path (Path): Path where the report will be saved.

    Returns:
        str: Generated report content.
    """
    inspection = inspect_data(df)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "==================================================",
        "          TITANIC DATA QUALITY REPORT            ",
        "==================================================",
        f"Total Passengers (Rows)   : {inspection['num_rows']}",
        f"Total Attributes (Columns): {inspection['num_cols']}",
        f"Target Feature            : {inspection['target_col']}",
        "",
        "--- Column Data Types ---",
    ]
    for col, dtype in inspection["dtypes"].items():
        lines.append(f"  - {col:<15}: {dtype}")

    lines.extend([
        "",
        "--- Feature Classification ---",
        f"Numerical Features ({len(inspection['numerical_cols'])}): {', '.join(inspection['numerical_cols'])}",
        f"Categorical Features ({len(inspection['categorical_cols'])}): {', '.join(inspection['categorical_cols'])}",
        "",
        "--- Missing Values Analysis ---",
    ])

    if inspection["missing_values"]:
        for col, count in inspection["missing_values"].items():
            pct = (count / inspection["num_rows"]) * 100
            lines.append(f"  - {col:<12}: {count:>3} missing ({pct:.2f}%)")
    else:
        lines.append("  No missing values detected.")

    lines.extend([
        "",
        "--- Target Distribution (Survival Rate) ---",
    ])
    if "Survived" in df.columns:
        survived_counts = df["Survived"].value_counts()
        survived_pct = df["Survived"].value_counts(normalize=True) * 100
        lines.append(f"  - Did Not Survive (0): {survived_counts.get(0, 0)} ({survived_pct.get(0, 0):.2f}%)")
        lines.append(f"  - Survived (1)        : {survived_counts.get(1, 0)} ({survived_pct.get(1, 0):.2f}%)")

    lines.extend([
        "",
        "==================================================",
        "               END OF REPORT                      ",
        "==================================================",
    ])

    report_content = "\n".join(lines)
    out_file.write_text(report_content, encoding="utf-8")
    return report_content
