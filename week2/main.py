"""Week 2: Supervised Learning — Main Execution Script.

Executes both core tasks:
1. Mini Project 2 / Assignment 1: House Price Prediction using Linear Regression, Decision Tree, & Random Forest.
2. Assignment 2: Titanic Survival Classification using Logistic Regression, Decision Tree, & Random Forest.
Generates all metrics tables, high-resolution plots, and saves serialized model artifacts.
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from week2.src.housing_regression import run_housing_regression
from week2.src.titanic_classification import run_titanic_classification


def print_banner(title: str):
    width = 75
    print("\n" + "=" * width)
    print(f" {title.center(width - 2)} ")
    print("=" * width + "\n")


def main():
    start_time = time.time()
    print_banner("WEEK 2: SUPERVISED LEARNING (REGRESSION & CLASSIFICATION)")

    output_dir = PROJECT_ROOT / "week2" / "outputs"
    housing_data = PROJECT_ROOT / "week2" / "data" / "Housing.csv"
    titanic_data = PROJECT_ROOT / "week2" / "data" / "titanic_cleaned.csv"

    # -------------------------------------------------------------
    # TASK 1: Mini Project 2 & Assignment 1 — Housing Price Prediction
    # -------------------------------------------------------------
    print_banner("1. Mini Project 2 & Assignment 1: House Price Prediction (Regression)")
    print(f"[*] Dataset: {housing_data}")
    housing_pipeline, housing_metrics = run_housing_regression(
        data_path=housing_data, output_dir=output_dir
    )

    print("\n[+] Regression Models Trained Successfully:")
    print(housing_metrics.to_string(index=False))

    lr_r2 = housing_pipeline.metrics_test["Linear Regression"]["R2"]
    print(f"\n[OK] Linear Regression Test R2 Score: {lr_r2:.4f} ({lr_r2 * 100:.2f}% variance explained)")

    # -------------------------------------------------------------
    # TASK 2: Assignment 2 — Titanic Survival Classification
    # -------------------------------------------------------------
    print_banner("2. Assignment 2: Titanic Survival Prediction (Classification)")
    print(f"[*] Dataset: {titanic_data}")
    titanic_pipeline, titanic_metrics = run_titanic_classification(
        data_path=titanic_data, output_dir=output_dir
    )

    print("\n[+] Classification Models Trained Successfully:")
    print(titanic_metrics.to_string(index=False))

    lr_acc = titanic_pipeline.metrics_test["Logistic Regression"]["Accuracy"]
    lr_auc = titanic_pipeline.metrics_test["Logistic Regression"]["ROC_AUC"]
    print(f"\n[OK] Logistic Regression Test Accuracy: {lr_acc * 100:.2f}% | Test ROC-AUC: {lr_auc:.4f}")

    # -------------------------------------------------------------
    # Summary of Generated Artifacts
    # -------------------------------------------------------------
    print_banner("Summary of Generated Artifacts")
    fig_dir = output_dir / "figures"
    model_dir = output_dir / "models"
    report_dir = output_dir / "reports"

    print(f"[+] Figures saved to: {fig_dir}")
    for fig_file in sorted(fig_dir.glob("*.png")):
        print(f"    - {fig_file.name}")

    print(f"\n[+] Trained Model Artifacts saved to: {model_dir}")
    for model_file in sorted(model_dir.glob("*.joblib")):
        print(f"    - {model_file.name}")

    print(f"\n[+] Tabular Benchmark Reports saved to: {report_dir}")
    for report_file in sorted(report_dir.glob("*.csv")):
        print(f"    - {report_file.name}")

    elapsed = time.time() - start_time
    print(f"\n[DONE] Week 2 Pipeline execution completed successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
