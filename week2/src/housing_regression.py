"""Housing Price Prediction Module — Linear Regression, Decision Tree, & Random Forest.

Implements Mini Project 2 and Assignment 1:
- Loads Kaggle Housing Dataset
- Preprocesses features (categorical encoding, feature scaling)
- Trains Linear Regression, Ridge, Decision Tree, and Random Forest Regressors
- Evaluates R^2, MSE, RMSE, MAE
- Generates Predicted vs Actual plots and Residual diagnostics
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from .model_utils import (
        calculate_regression_metrics,
        plot_feature_importance_bar,
        plot_predicted_vs_actual,
        plot_residuals,
        save_model,
    )
except (ImportError, ValueError):
    # Support direct execution (python week2/src/housing_regression.py)
    root_dir = Path(__file__).resolve().parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    from week2.src.model_utils import (
        calculate_regression_metrics,
        plot_feature_importance_bar,
        plot_predicted_vs_actual,
        plot_residuals,
        save_model,
    )

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Housing.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


class HousingRegressionPipeline:
    """End-to-end regression modeling pipeline for House Price Prediction."""

    BINARY_COLS = [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
    ]
    CATEGORICAL_COLS = ["furnishingstatus"]
    NUMERICAL_COLS = ["area", "bedrooms", "bathrooms", "stories", "parking"]

    def __init__(
        self,
        data_path: Union[str, Path] = DEFAULT_DATA_PATH,
        random_state: int = 42,
    ):
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        self.scaler = StandardScaler()
        self.models: Dict[str, object] = {}
        self.metrics_train: Dict[str, Dict[str, float]] = {}
        self.metrics_test: Dict[str, Dict[str, float]] = {}
        self.predictions: Dict[str, np.ndarray] = {}

    def load_data(self) -> pd.DataFrame:
        """Load the raw Housing dataset from CSV."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Housing dataset not found at {self.data_path}")
        self.df_raw = pd.read_csv(self.data_path)
        return self.df_raw

    def preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Encode categorical features and prepare feature matrix X and target y."""
        if self.df_raw is None:
            self.load_data()

        df = self.df_raw.copy()

        # 1. Map binary categorical columns (yes -> 1, no -> 0)
        for col in self.BINARY_COLS:
            if col in df.columns:
                df[col] = df[col].map({"yes": 1, "no": 0}).astype(int)

        # 2. One-hot encode furnishingstatus with drop_first=True to prevent dummy variable trap
        if "furnishingstatus" in df.columns:
            df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True, dtype=int)

        self.df_processed = df
        X = df.drop(columns=["price"])
        y = df["price"]
        self.feature_names = list(X.columns)
        return X, y

    def split_and_scale(
        self, test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/test sets and scale features with StandardScaler."""
        X, y = self.preprocess_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Scale features based only on training distribution to prevent data leakage
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.X_train_raw = X_train
        self.X_test_raw = X_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train.values
        self.y_test = y_test.values

        return X_train_scaled, X_test_scaled, self.y_train, self.y_test

    def train_models(self) -> Dict[str, object]:
        """Train Linear Regression, Ridge, Decision Tree, and Random Forest models."""
        if not hasattr(self, "X_train_scaled"):
            self.split_and_scale()

        self.models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0, random_state=self.random_state),
            "Decision Tree Regressor": DecisionTreeRegressor(
                max_depth=5, min_samples_split=5, random_state=self.random_state
            ),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_split=4, random_state=self.random_state
            ),
        }

        n_features = len(self.feature_names)

        for name, model in self.models.items():
            model.fit(self.X_train_scaled, self.y_train)

            # Predict on train & test
            y_pred_train = model.predict(self.X_train_scaled)
            y_pred_test = model.predict(self.X_test_scaled)
            self.predictions[name] = y_pred_test

            # Metrics
            self.metrics_train[name] = calculate_regression_metrics(
                self.y_train, y_pred_train, n_features=n_features
            )
            self.metrics_test[name] = calculate_regression_metrics(
                self.y_test, y_pred_test, n_features=n_features
            )

        return self.models

    def get_metrics_table(self) -> pd.DataFrame:
        """Return a combined DataFrame comparing train and test metrics for all models."""
        records = []
        for name in self.models.keys():
            m_train = self.metrics_train[name]
            m_test = self.metrics_test[name]
            records.append(
                {
                    "Model": name,
                    "Train R2": m_train["R2"],
                    "Test R2": m_test["R2"],
                    "Test Adj R2": m_test.get("Adjusted_R2", np.nan),
                    "Test MAE": m_test["MAE"],
                    "Test RMSE": m_test["RMSE"],
                    "Test MAPE (%)": m_test["MAPE(%)"],
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
        saved_paths["scaler"] = save_model(self.scaler, model_dir / "housing_scaler.joblib")

        # 2. Save Plots for Linear Regression (Mini Project 2 Primary Task)
        lr_pred = self.predictions["Linear Regression"]
        lr_r2 = self.metrics_test["Linear Regression"]["R2"]

        p_scatter = fig_dir / "housing_linear_regression_predicted_vs_actual.png"
        plot_predicted_vs_actual(
            self.y_test, lr_pred, model_name="Linear Regression", r2_val=lr_r2, save_path=p_scatter
        )
        saved_paths["plot_pred_vs_actual"] = p_scatter

        p_residuals = fig_dir / "housing_linear_regression_residuals.png"
        plot_residuals(self.y_test, lr_pred, model_name="Linear Regression", save_path=p_residuals)
        saved_paths["plot_residuals"] = p_residuals

        # 3. Coefficients / Feature Importances
        lr_model = self.models["Linear Regression"]
        p_coef = fig_dir / "housing_linear_regression_coefficients.png"
        plot_feature_importance_bar(
            self.feature_names,
            lr_model.coef_,
            title="Linear Regression Feature Coefficients (Standardized)",
            save_path=p_coef,
        )
        saved_paths["plot_coefficients"] = p_coef

        rf_model = self.models["Random Forest Regressor"]
        p_rf_imp = fig_dir / "housing_random_forest_feature_importance.png"
        plot_feature_importance_bar(
            self.feature_names,
            rf_model.feature_importances_,
            title="Random Forest Regressor Feature Importances",
            save_path=p_rf_imp,
        )
        saved_paths["plot_rf_importance"] = p_rf_imp

        # 4. Save Metrics Summary Report
        metrics_df = self.get_metrics_table()
        report_csv = report_dir / "housing_regression_benchmark.csv"
        metrics_df.to_csv(report_csv, index=False)
        saved_paths["report_csv"] = report_csv

        return saved_paths


def run_housing_regression(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
) -> Tuple[HousingRegressionPipeline, pd.DataFrame]:
    """Convenience helper to run the entire housing regression pipeline."""
    pipeline = HousingRegressionPipeline(data_path=data_path)
    pipeline.train_models()
    pipeline.save_artifacts(output_dir=output_dir)
    return pipeline, pipeline.get_metrics_table()


if __name__ == "__main__":
    pipeline, df_res = run_housing_regression()
    print("=== Housing Price Prediction Benchmark Results ===")
    print(df_res.to_string(index=False))
