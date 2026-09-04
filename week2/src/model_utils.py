"""Utility functions for model evaluation, metrics calculation, artifact persistence, and visualization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_features: Optional[int] = None
) -> Dict[str, float]:
    """Calculate comprehensive regression evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values.
    y_pred : np.ndarray
        Model predicted target values.
    n_features : Optional[int]
        Number of predictor features (for Adjusted R^2 calculation).

    Returns
    -------
    Dict[str, float]
        Dictionary with MAE, MSE, RMSE, R2, MAPE, and optionally Adjusted R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE(%)": float(mape),
    }

    if n_features is not None and len(y_true) > n_features + 1:
        n = len(y_true)
        p = n_features
        adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
        metrics["Adjusted_R2"] = float(adj_r2)

    return metrics


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calculate comprehensive binary classification evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    y_prob : Optional[np.ndarray]
        Predicted probability of positive class (for ROC-AUC).

    Returns
    -------
    Dict[str, float]
        Dictionary containing Accuracy, Precision, Recall, F1-Score, Specificity, and ROC-AUC.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1_Score": float(f1),
        "Specificity": float(specificity),
        "True_Positives": int(tp),
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
    }

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            metrics["ROC_AUC"] = float(auc)
        except Exception:
            metrics["ROC_AUC"] = 0.0

    return metrics


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Linear Regression",
    r2_val: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot Predicted vs Actual scatter plot with identity reference line.

    Parameters
    ----------
    y_true : np.ndarray
        Actual target prices.
    y_pred : np.ndarray
        Predicted target prices.
    model_name : str
        Name of model for chart title.
    r2_val : Optional[float]
        R2 score to annotate on plot.
    save_path : Optional[Union[str, Path]]
        Destination path to save the generated figure.

    Returns
    -------
    plt.Figure
    """
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.scatter(y_true, y_pred, alpha=0.65, color="#1f77b4", edgecolors="w", s=65, label="Data Points")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="#d62728", linestyle="--", linewidth=2, label="Perfect Fit (y = x)")

    title = f"Predicted vs Actual House Prices ({model_name})"
    if r2_val is not None:
        title += f"\n$R^2 = {r2_val:.4f}$"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Actual Price ($ / ₹)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Predicted Price ($ / ₹)", fontsize=11, fontweight="semibold")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Linear Regression",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot Residual distribution and Residuals vs Fitted.

    Parameters
    ----------
    y_true : np.ndarray
        Actual targets.
    y_pred : np.ndarray
        Predicted targets.
    model_name : str
        Model name.
    save_path : Optional[Union[str, Path]]
        Destination path.

    Returns
    -------
    plt.Figure
    """
    residuals = y_true - y_pred
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, color="#2ca02c", edgecolors="w", s=55)
    axes[0].axhline(0, color="#d62728", linestyle="--", linewidth=1.8)
    axes[0].set_title(f"Residuals vs Fitted ({model_name})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Fitted (Predicted) Values", fontsize=10)
    axes[0].set_ylabel("Residuals ($y - \\hat{y}$)", fontsize=10)

    # 2. Residual Distribution (KDE)
    sns.histplot(residuals, kde=True, ax=axes[1], color="#9467bd", bins=20)
    axes[1].axvline(0, color="#d62728", linestyle="--", linewidth=1.8)
    axes[1].set_title(f"Residual Error Distribution ({model_name})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Residual Error", fontsize=10)
    axes[1].set_ylabel("Density / Count", fontsize=10)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str] = ["Did Not Survive", "Survived"],
    model_name: str = "Logistic Regression",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot styled Confusion Matrix heatmap.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    classes : List[str]
        Class label names.
    model_name : str
        Model title.
    save_path : Optional[Union[str, Path]]
        Destination path.

    Returns
    -------
    plt.Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        linewidths=1.5,
        linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=10, fontweight="semibold")
    ax.set_ylabel("True Class", fontsize=10, fontweight="semibold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_roc_curves_comparison(
    models_dict: Dict[str, Tuple[np.ndarray, float]],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot overlaid ROC curves for multiple models.

    Parameters
    ----------
    models_dict : Dict[str, Tuple[np.ndarray, float]]
        Mapping of model_name -> (fpr, tpr, roc_auc)
    save_path : Optional[Union[str, Path]]
        Save destination.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, (name, (fpr, tpr, auc_val)) in enumerate(models_dict.items()):
        c = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="semibold")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_feature_importance_bar(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    top_n: int = 15,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot sorted horizontal bar chart of feature importances or model coefficients.

    Parameters
    ----------
    feature_names : List[str]
        List of predictor feature names.
    importances : np.ndarray
        Importance values or regression coefficients.
    title : str
        Chart title.
    top_n : int
        Number of features to display.
    save_path : Optional[Union[str, Path]]
        Save file destination.

    Returns
    -------
    plt.Figure
    """
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp["AbsImportance"] = df_imp["Importance"].abs()
    df_imp = df_imp.sort_values(by="AbsImportance", ascending=False).head(top_n)
    df_imp = df_imp.sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(5, int(top_n * 0.35))), dpi=300)
    colors = ["#d62728" if x < 0 else "#1f77b4" for x in df_imp["Importance"]]
    ax.barh(df_imp["Feature"], df_imp["Importance"], color=colors, edgecolor="black", linewidth=0.6, alpha=0.85)

    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Value / Weight", fontsize=10, fontweight="semibold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def save_model(model: object, filepath: Union[str, Path]) -> Path:
    """Serialize model artifact to disk using joblib.

    Parameters
    ----------
    model : object
        Trained model instance or Pipeline.
    filepath : Union[str, Path]
        Path to save joblib file.

    Returns
    -------
    Path
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(filepath: Union[str, Path]) -> object:
    """Deserialize model artifact from disk.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to saved joblib file.

    Returns
    -------
    object
    """
    return joblib.load(filepath)
