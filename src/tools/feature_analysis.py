from langchain_core.tools import tool
from .data_loader import DATASETS
import pandas as pd
from tabulate import tabulate

@tool
def feature_importance(dataset_name: str, target: str, method: str = 'random_forest') -> str:
    """Compute feature importance."""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    if dataset_name not in DATASETS: return "Dataset not found."
    df = DATASETS[dataset_name].dropna()
    if target not in df.columns: return "Target not found."
    
    X = df.drop(columns=[target]).select_dtypes(include='number')
    y = df[target]
    
    if y.dtype == 'object' or len(y.unique()) < 20:
        model = RandomForestClassifier(random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)
        
    model.fit(X, y)
    imp = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    imp = imp.sort_values(by='Importance', ascending=False)
    return tabulate(imp, headers='keys', tablefmt='psql')

@tool
def feature_correlation_with_target(dataset_name: str, target: str) -> str:
    """Show correlations with target."""
    if dataset_name not in DATASETS: return "Dataset not found."
    df = DATASETS[dataset_name].select_dtypes(include='number')
    if target not in df.columns: return "Target not found or not numeric."
    corr = df.corr()[[target]].sort_values(by=target, ascending=False)
    return tabulate(corr, headers='keys', tablefmt='psql')

@tool
def suggest_features(dataset_name: str, target: str) -> str:
    """Recommend best features for modeling."""
    return feature_importance.invoke({"dataset_name": dataset_name, "target": target, "method": "random_forest"})
