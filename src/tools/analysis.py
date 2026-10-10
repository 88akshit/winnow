from langchain_core.tools import tool
import pandas as pd
from tabulate import tabulate
from .data_loader import DATASETS

@tool
def describe_data(dataset_name: str) -> str:
    """Full descriptive statistics."""
    if dataset_name not in DATASETS: return f"Dataset {dataset_name} not found."
    desc = DATASETS[dataset_name].describe(include='all')
    return tabulate(desc, headers='keys', tablefmt='psql')

@tool
def correlation_matrix(dataset_name: str, method: str = 'pearson') -> str:
    """Compute and display correlation matrix for numeric columns."""
    if dataset_name not in DATASETS: return f"Dataset {dataset_name} not found."
    df = DATASETS[dataset_name].select_dtypes(include='number')
    if df.empty: return "No numeric columns to correlate."
    corr = df.corr(method=method)
    return tabulate(corr, headers='keys', tablefmt='psql')

@tool
def value_counts(dataset_name: str, column: str) -> str:
    """Value counts for a column."""
    if dataset_name not in DATASETS: return "Dataset not found."
    df = DATASETS[dataset_name]
    if column not in df.columns: return "Column not found."
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    return tabulate(counts, headers='keys', tablefmt='psql')

@tool
def group_statistics(dataset_name: str, group_by: str, agg_column: str, agg_func: str) -> str:
    """GroupBy analysis. agg_func can be mean, sum, count, max, min."""
    if dataset_name not in DATASETS: return "Dataset not found."
    df = DATASETS[dataset_name]
    if group_by not in df.columns or agg_column not in df.columns: return "Column not found."
    res = df.groupby(group_by)[agg_column].agg(agg_func).reset_index()
    return tabulate(res, headers='keys', tablefmt='psql')

@tool
def statistical_test(dataset_name: str, column1: str, column2: str, test_type: str) -> str:
    """Run a t-test or Pearson correlation on two numeric dataset columns."""
    import scipy.stats as stats
    if dataset_name not in DATASETS: return "Dataset not found."
    df = DATASETS[dataset_name]
    missing = [column for column in (column1, column2) if column not in df.columns]
    if missing: return f"Column not found: {', '.join(missing)}."

    # CSV/Excel inference can leave numeric-looking or mixed columns as object
    # dtype. scipy.stats requires actual numeric arrays, so coerce values here.
    import numpy as np
    first = pd.to_numeric(df[column1], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    second = pd.to_numeric(df[column2], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if test_type.lower() in ['t-test', 'ttest']:
        first = first[np.isfinite(first)]
        second = second[np.isfinite(second)]
        if len(first) < 2 or len(second) < 2:
            return "A t-test needs at least two valid numeric values in each column."
        stat, p = stats.ttest_ind(first, second)
        return f"T-test results: statistic={stat}, p-value={p}"
    elif test_type.lower() == 'pearson':
        valid = np.isfinite(first) & np.isfinite(second)
        first, second = first[valid], second[valid]
        if len(first) < 2:
            return "Pearson correlation needs at least two rows with valid numeric values in both columns."
        if np.ptp(first) == 0 or np.ptp(second) == 0:
            return "Pearson correlation is undefined because one of the columns has no variation."
        stat, p = stats.pearsonr(first, second)
        return f"Pearson correlation: statistic={stat}, p-value={p}"
    return "Unsupported test_type."

@tool
def detect_outliers(dataset_name: str, column: str, method: str = 'iqr') -> str:
    """Detect outliers using IQR."""
    if dataset_name not in DATASETS: return "Dataset not found."
    df = DATASETS[dataset_name]
    if column not in df.columns: return "Column not found."
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))]
    return f"Detected {len(outliers)} outliers in {column} using IQR method."
