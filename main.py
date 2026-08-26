"""
Main Execution Script for Week 1 ML Fundamentals + Data Preprocessing Project.
Orchestrates end-to-end Titanic data pipeline.
"""

import sys
import logging
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_titanic_data, inspect_data, generate_quality_report
from src.feature_engineering import engineer_features
from src.data_cleaning import clean_missing_values
from src.preprocessing import (
    encode_categorical_features,
    prepare_train_test_split,
    scale_features,
    export_cleaned_data,
)
from src.visualization import generate_all_visualizations

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Week1_ML_Pipeline")


def main() -> None:
    """Executes complete Week 1 Machine Learning Preprocessing Pipeline."""
    logger.info("==================================================")
    logger.info("  STARTING WEEK 1 ML PREPROCESSING PIPELINE       ")
    logger.info("==================================================")

    # File Paths
    raw_csv_path = PROJECT_ROOT / "data" / "raw" / "train.csv"
    processed_csv_data_path = PROJECT_ROOT / "data" / "processed" / "titanic_cleaned.csv"
    processed_csv_output_path = PROJECT_ROOT / "outputs" / "titanic_cleaned.csv"
    report_output_path = PROJECT_ROOT / "outputs" / "reports" / "data_quality_report.txt"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"

    # Step 1: Load Dataset
    logger.info("Step 1: Loading raw Titanic dataset from %s...", raw_csv_path)
    df_raw = load_titanic_data(raw_csv_path)
    logger.info("Loaded dataset with shape: %s", df_raw.shape)

    # Step 2: Explore Data & Generate Quality Report
    logger.info("Step 2: Performing data quality inspection...")
    inspection = inspect_data(df_raw)
    report_text = generate_quality_report(df_raw, report_output_path)
    logger.info("Data Quality Report generated and saved to: %s", report_output_path)

    # Step 3: Feature Engineering
    logger.info("Step 3: Engineering domain features (FamilySize, IsAlone, Title)...")
    df_engineered = engineer_features(df_raw)
    logger.info("New columns created: Title, FamilySize, IsAlone")

    # Step 4: Missing Value Handling
    logger.info("Step 4: Handling missing values (Age median, Embarked mode, Cabin deck)...")
    df_cleaned = clean_missing_values(df_engineered)
    logger.info("Remaining missing values count: %d", df_cleaned.isnull().sum().sum())

    # Step 5: Data Visualization
    logger.info("Step 5: Generating and saving 5 core EDA plots to %s...", figures_dir)
    fig_paths = generate_all_visualizations(df_cleaned, figures_dir)
    for p in fig_paths:
        logger.info("  - Saved plot: %s", p.name)

    # Step 6: Categorical Encoding
    logger.info("Step 6: Performing Categorical Encoding (LabelEncoder for Sex, OHE for Embarked/Title/Deck)...")
    df_encoded, encoders = encode_categorical_features(df_cleaned)
    logger.info("Dataset shape post-encoding: %s", df_encoded.shape)

    # Step 7: Train/Test Split
    logger.info("Step 7: Performing Stratified Train/Test Split (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(
        df_encoded, target_col="Survived", test_size=0.2, random_state=42
    )
    logger.info("Train shape: X=%s, y=%s | Test shape: X=%s, y=%s", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Step 8: Feature Scaling (Zero Data Leakage)
    logger.info("Step 8: Scaling numerical features with StandardScaler (Fit on Train ONLY)...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, num_cols=["Age", "Fare", "FamilySize"])
    logger.info("Feature scaling completed successfully without data leakage.")

    # Step 9: Export Cleaned Dataset
    logger.info("Step 9: Exporting final cleaned dataset to CSV...")
    export_cleaned_data(df_encoded, processed_csv_data_path)
    export_cleaned_data(df_encoded, processed_csv_output_path)
    logger.info("Saved cleaned CSV to: %s and %s", processed_csv_data_path, processed_csv_output_path)

    # Step 10: Execution Summary
    logger.info("==================================================")
    logger.info("  WEEK 1 ML PREPROCESSING PIPELINE COMPLETED!     ")
    logger.info("==================================================")
    logger.info("Summary Statistics:")
    logger.info("  - Initial Dataset Rows/Cols : %d / %d", inspection["num_rows"], inspection["num_cols"])
    logger.info("  - Final Encoded Features   : %d", df_encoded.shape[1])
    logger.info("  - Train Set Size           : %d samples", len(X_train))
    logger.info("  - Test Set Size            : %d samples", len(X_test))
    logger.info("  - Total Missing Values Left: %d", df_encoded.isnull().sum().sum())
    logger.info("==================================================")


if __name__ == "__main__":
    main()
