"""Titanic Survival Classification Module — Logistic Regression, Decision Tree, & Random Forest.

Implements Assignment 2 and Supervised Classification:
- Loads Cleaned Titanic Dataset
- Selects engineered features and performs stratified train/test split
- Scales features and trains Logistic Regression, Decision Tree, and Random Forest Classifiers
- Evaluates Accuracy, Precision, Recall, F1-Score, Specificity, and ROC-AUC
- Generates Confusion Matrix heatmaps, ROC curve comparisons, and Feature Importance charts
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from .model_utils import (
        calculate_classification_metrics,
        plot_confusion_matrix_heatmap,
        plot_feature_importance_bar,
        plot_roc_curves_comparison,
        save_model,
    )
except (ImportError, ValueError):
    # Support direct execution (python week2/src/titanic_classification.py)
    root_dir = Path(__file__).resolve().parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from week2.src.model_utils import (
        calculate_classification_metrics,
        plot_confusion_matrix_heatmap,
        plot_feature_importance_bar,
        plot_roc_curves_comparison,
        save_model,
    )

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "titanic_cleaned.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


class TitanicClassificationPipeline:
    """End-to-end classification modeling pipeline for Titanic Survival Prediction."""

    DROP_COLS = ["PassengerId", "Name", "Ticket", "Survived"]

    def __init__(
        self,
        data_path: Union[str, Path] = DEFAULT_DATA_PATH,
        random_state: int = 42,
    ):
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.df_raw: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        self.scaler = StandardScaler()
        self.models: Dict[str, object] = {}
        self.metrics_train: Dict[str, Dict[str, float]] = {}
        self.metrics_test: Dict[str, Dict[str, float]] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.probabilities: Dict[str, np.ndarray] = {}

    def load_data(self) -> pd.DataFrame:
        """Load the preprocessed Titanic dataset."""
        if not self.data_path.exists():
            # Fallback to outputs/titanic_cleaned.csv if needed
            fallback = Path("outputs/titanic_cleaned.csv")
            if fallback.exists():
                self.data_path = fallback
            else:
                raise FileNotFoundError(f"Titanic dataset not found at {self.data_path}")
        self.df_raw = pd.read_csv(self.data_path)
        return self.df_raw

    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Filter out identifier columns and separate target y."""
        if self.df_raw is None:
            self.load_data()

        df = self.df_raw.copy()
        cols_to_drop = [c for c in self.DROP_COLS if c in df.columns]
        X = df.drop(columns=cols_to_drop)
        y = df["Survived"].astype(int)

        self.feature_names = list(X.columns)
        return X, y

    def split_and_scale(
        self, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified train-test split and standard feature scaling."""
        X, y = self.prepare_features()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train.values
        self.y_test = y_test.values

        return X_train_scaled, X_test_scaled, self.y_train, self.y_test

    def train_models(self) -> Dict[str, object]:
        """Train Logistic Regression, Decision Tree, and Random Forest models."""
        if not hasattr(self, "X_train_scaled"):
            self.split_and_scale()

        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=self.random_state, C=1.0
            ),
            "Decision Tree Classifier": DecisionTreeClassifier(
                max_depth=5, min_samples_split=6, min_samples_leaf=3, random_state=self.random_state
            ),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=150, max_depth=6, min_samples_split=4, random_state=self.random_state
            ),
        }

        for name, model in self.models.items():
            model.fit(self.X_train_scaled, self.y_train)

            # Predictions & probabilities
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            self.predictions[name] = y_pred_test

            y_prob_train = model.predict_proba(self.X_train_scaled)[:, 1]
            y_prob_test = model.predict_proba(self.X_test_scaled)[:, 1]
            self.probabilities[name] = y_prob_test

            # Metrics
            self.metrics_train[name] = calculate_classification_metrics(
                self.y_train, y_pred_train, y_prob_train
            )
            self.metrics_test[name] = calculate_classification_metrics(
                self.y_test, y_pred_test, y_prob_test
            )

        return self.models

    def get_metrics_table(self) -> pd.DataFrame:
        """Return a combined DataFrame comparing classification metrics for all models."""
        records = []
        for name in self.models.keys():
            m_train = self.metrics_train[name]
            m_test = self.metrics_test[name]
            records.append(
                {
                    "Model": name,
                    "Train Acc (%)": m_train["Accuracy"] * 100,
                    "Test Acc (%)": m_test["Accuracy"] * 100,
                    "Precision": m_test["Precision"],
                    "Recall": m_test["Recall"],
                    "F1-Score": m_test["F1_Score"],
                    "Specificity": m_test["Specificity"],
                    "ROC-AUC": m_test.get("ROC_AUC", np.nan),
                }
            )
        df_metrics = pd.DataFrame(records)
        return df_metrics

    def save_artifacts(
        self,
        output_dir: Union[str, Path] = "week2/outputs",
    ) -> Dict[str, Path]:
        """Save trained models, high-DPI figures, and evaluation reports."""
        out_dir = Path(output_dir)
        fig_dir = out_dir / "figures"
        model_dir = out_dir / "models"
        report_dir = out_dir / "reports"

        for d in [fig_dir, model_dir, report_dir]:
            d.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # 1. Save Models
        for name, model in self.models.items():
            safe_name = name.lower().replace(" ", "_")
            m_path = save_model(model, model_dir / f"{safe_name}.joblib")
            saved_paths[f"model_{safe_name}"] = m_path

        # Save Scaler
        saved_paths["scaler"] = save_model(self.scaler, model_dir / "titanic_scaler.joblib")

        # 2. Confusion Matrices (Logistic Regression & Random Forest)
        for name in ["Logistic Regression", "Random Forest Classifier"]:
            safe_name = name.lower().replace(" ", "_")
            p_cm = fig_dir / f"titanic_{safe_name}_confusion_matrix.png"
            plot_confusion_matrix_heatmap(
                self.y_test,
                self.predictions[name],
                classes=["Did Not Survive (0)", "Survived (1)"],
                model_name=name,
                save_path=p_cm,
            )
            saved_paths[f"cm_{safe_name}"] = p_cm

        # 3. ROC Curves Comparison
        roc_dict = {}
        for name, model in self.models.items():
            prob = self.probabilities[name]
            fpr, tpr, _ = roc_curve(self.y_test, prob)
            auc_val = self.metrics_test[name].get("ROC_AUC", 0.0)
            roc_dict[name] = (fpr, tpr, auc_val)

        p_roc = fig_dir / "titanic_roc_curves_comparison.png"
        plot_roc_curves_comparison(roc_dict, save_path=p_roc)
        saved_paths["plot_roc_curves"] = p_roc

        # 4. Feature Coefficients / Importances
        log_reg = self.models["Logistic Regression"]
        p_log_coef = fig_dir / "titanic_logistic_regression_coefficients.png"
        plot_feature_importance_bar(
            self.feature_names,
            log_reg.coef_[0],
            title="Logistic Regression Feature Coefficients (Log-Odds)",
            save_path=p_log_coef,
        )
        saved_paths["plot_logistic_coef"] = p_log_coef

        rf_clf = self.models["Random Forest Classifier"]
        p_rf_imp = fig_dir / "titanic_random_forest_feature_importance.png"
        plot_feature_importance_bar(
            self.feature_names,
            rf_clf.feature_importances_,
            title="Random Forest Classifier Feature Importances",
            save_path=p_rf_imp,
        )
        saved_paths["plot_rf_importance"] = p_rf_imp

        # 5. Save Metrics Summary Report
        metrics_df = self.get_metrics_table()
        report_csv = report_dir / "titanic_classification_benchmark.csv"
        metrics_df.to_csv(report_csv, index=False)
        saved_paths["report_csv"] = report_csv

        return saved_paths


def run_titanic_classification(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
) -> Tuple[TitanicClassificationPipeline, pd.DataFrame]:
    """Convenience helper to run the entire titanic classification pipeline."""
    pipeline = TitanicClassificationPipeline(data_path=data_path)
    pipeline.train_models()
    pipeline.save_artifacts(output_dir=output_dir)
    return pipeline, pipeline.get_metrics_table()


if __name__ == "__main__":
    pipeline, df_res = run_titanic_classification()
    print("=== Titanic Survival Classification Benchmark Results ===")
    print(df_res.to_string(index=False))
