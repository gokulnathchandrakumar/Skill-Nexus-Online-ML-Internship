"""
Visualization Module for Exploratory Data Analysis (EDA).
Generates and saves publication-quality charts using Matplotlib and Seaborn.
"""

from pathlib import Path
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_style() -> None:
    """Sets global visual style parameters for professional charts."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 18,
    })


def plot_age_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generates and saves Age distribution histogram with KDE overlay."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(df["Age"].dropna(), kde=True, color="#2b5c8f", bins=30, ax=ax)
    ax.set_title("Age Distribution of Passengers", fontweight="bold", pad=15)
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Passenger Count")

    output_path = output_dir / "age_distribution.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_survival_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generates and saves Overall Survival count distribution."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    palette = {0: "#d9534f", 1: "#5cb85c"}
    sns.countplot(x="Survived", hue="Survived", data=df, palette=palette, ax=ax, legend=False)

    ax.set_title("Overall Survival Count", fontweight="bold", pad=15)
    ax.set_xlabel("Survival Status")
    ax.set_ylabel("Passenger Count")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Did Not Survive (0)", "Survived (1)"])

    # Add count labels on bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="center",
                xytext=(0, 8),
                textcoords="offset points",
                fontweight="bold",
            )

    output_path = output_dir / "survival_distribution.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_survival_by_gender(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generates and saves Survival comparison grouped by Gender."""
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    palette = {0: "#d9534f", 1: "#5cb85c"}
    sns.countplot(x="Sex", hue="Survived", data=df, palette=palette, ax=ax)

    ax.set_title("Survival Breakdown by Gender", fontweight="bold", pad=15)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Passenger Count")
    ax.legend(["Did Not Survive", "Survived"], title="Status")

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="center",
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=11,
            )

    output_path = output_dir / "survival_by_gender.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_survival_by_class(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generates and saves Survival comparison grouped by Passenger Class."""
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    palette = {0: "#d9534f", 1: "#5cb85c"}
    sns.countplot(x="Pclass", hue="Survived", data=df, palette=palette, ax=ax)

    ax.set_title("Survival Breakdown by Passenger Class", fontweight="bold", pad=15)
    ax.set_xlabel("Passenger Class (Pclass)")
    ax.set_ylabel("Passenger Count")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["1st Class", "2nd Class", "3rd Class"])
    ax.legend(["Did Not Survive", "Survived"], title="Status")

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="center",
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=11,
            )

    output_path = output_dir / "survival_by_class.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generates and saves Correlation Heatmap for numerical features."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    num_df = df.select_dtypes(include=["float64", "int64", "int32"])
    # Exclude raw PassengerId if present
    if "PassengerId" in num_df.columns:
        num_df = num_df.drop(columns=["PassengerId"])

    corr = num_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, linewidths=0.5)

    ax.set_title("Numerical Features Correlation Heatmap", fontweight="bold", pad=15)

    output_path = output_dir / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def generate_all_visualizations(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """
    Executes and saves all 5 required visualizations.

    Args:
        df (pd.DataFrame): Dataset with raw or cleaned columns.
        output_dir (Path): Output directory for figures.

    Returns:
        List[Path]: List of saved file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_age_distribution(df, out_dir),
        plot_survival_distribution(df, out_dir),
        plot_survival_by_gender(df, out_dir),
        plot_survival_by_class(df, out_dir),
        plot_correlation_heatmap(df, out_dir),
    ]

    return paths
