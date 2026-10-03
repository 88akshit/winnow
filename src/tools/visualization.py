from langchain_core.tools import tool
import matplotlib.pyplot as plt
import seaborn as sns
import os
import uuid
from .data_loader import DATASETS

OUTPUT_DIR = os.path.join(os.getcwd(), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_plot(name_prefix: str) -> str:
    filename = f"{name_prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    return filepath

@tool
def plot_histogram(dataset_name: str, column: str, bins: int = 30) -> str:
    """Create histogram."""
    if dataset_name not in DATASETS: return "Dataset not found."
    sns.histplot(DATASETS[dataset_name][column], bins=bins)
    plt.title(f"Histogram of {column}")
    return f"Plot saved to: {save_plot('hist')}"

@tool
def plot_scatter(dataset_name: str, x: str, y: str, hue: str = None) -> str:
    """Scatter plot."""
    if dataset_name not in DATASETS: return "Dataset not found."
    sns.scatterplot(data=DATASETS[dataset_name], x=x, y=y, hue=hue)
    plt.title(f"Scatter plot of {x} vs {y}")
    return f"Plot saved to: {save_plot('scatter')}"

@tool
def plot_correlation_heatmap(dataset_name: str) -> str:
    """Correlation heatmap."""
    if dataset_name not in DATASETS: return "Dataset not found."
    df = DATASETS[dataset_name].select_dtypes(include='number')
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Correlation Heatmap ({dataset_name})")
    return f"Plot saved to: {save_plot('heatmap')}"

@tool
def plot_boxplot(dataset_name: str, column: str, by: str = None) -> str:
    """Box plot."""
    if dataset_name not in DATASETS: return "Dataset not found."
    sns.boxplot(data=DATASETS[dataset_name], x=by, y=column)
    plt.title(f"Boxplot of {column}" + (f" by {by}" if by else ""))
    return f"Plot saved to: {save_plot('boxplot')}"

@tool
def plot_pairplot(dataset_name: str, columns: list[str] = None) -> str:
    """Pair plot."""
    if dataset_name not in DATASETS: return "Dataset not found."
    df = DATASETS[dataset_name]
    if columns: df = df[columns]
    sns.pairplot(df.select_dtypes(include='number'))
    return f"Plot saved to: {save_plot('pairplot')}"

@tool
def plot_distribution(dataset_name: str, column: str) -> str:
    """KDE + histogram."""
    if dataset_name not in DATASETS: return "Dataset not found."
    sns.histplot(DATASETS[dataset_name][column], kde=True)
    plt.title(f"Distribution of {column}")
    return f"Plot saved to: {save_plot('dist')}"
