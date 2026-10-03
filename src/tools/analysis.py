from langchain_core.tools import tool
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
    """Run t-test, chi-square, etc."""
    import scipy.stats as stats
    if dataset_name not in DATASETS: return "Dataset not found."
    df = DATASETS[dataset_name].dropna(subset=[column1, column2])
    if test_type.lower() in ['t-test', 'ttest']:
        stat, p = stats.ttest_ind(df[column1], df[column2])
        return f"T-test results: statistic={stat}, p-value={p}"
    elif test_type.lower() == 'pearson':
        stat, p = stats.pearsonr(df[column1], df[column2])
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
