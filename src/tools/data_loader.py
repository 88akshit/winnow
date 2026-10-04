from langchain_core.tools import tool
import pandas as pd
import json
from src.utils.security import validate_safe_read_path

DATASETS = {}

@tool
def load_csv(file_path: str, name: str) -> str:
    """Load a CSV file from an allowed data directory into memory."""
    try:
        safe_path = validate_safe_read_path(file_path)
        df = pd.read_csv(safe_path)
        DATASETS[name] = df
        return f"Successfully loaded {name} from {safe_path.name} with shape {df.shape}."
    except Exception as e:
        return f"Error loading CSV: {str(e)}"

@tool
def load_excel(file_path: str, sheet_name: str, name: str) -> str:
    """Load an Excel file from an allowed data directory into memory."""
    try:
        safe_path = validate_safe_read_path(file_path)
        df = pd.read_excel(safe_path, sheet_name=sheet_name)
        DATASETS[name] = df
        return f"Successfully loaded {name} from {safe_path.name} (sheet {sheet_name}) with shape {df.shape}."
    except Exception as e:
        return f"Error loading Excel: {str(e)}"

@tool
def list_datasets() -> str:
    """List all loaded datasets."""
    if not DATASETS: return "No datasets loaded."
    return json.dumps({name: list(df.columns) for name, df in DATASETS.items()})

@tool
def preview_data(name: str, rows: int = 5) -> str:
    """Show first N rows of a dataset."""
    if name not in DATASETS: return f"Dataset {name} not found."
    from tabulate import tabulate
    return tabulate(DATASETS[name].head(rows), headers='keys', tablefmt='psql')

@tool
def get_column_info(name: str) -> str:
    """Show dtypes, nulls, unique counts for a dataset."""
    if name not in DATASETS: return f"Dataset {name} not found."
    df = DATASETS[name]
    info = pd.DataFrame({
        'dtype': df.dtypes,
        'null_count': df.isnull().sum(),
        'unique_count': df.nunique()
    })
    from tabulate import tabulate
    return tabulate(info, headers='keys', tablefmt='psql')
