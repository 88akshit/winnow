from langchain_core.tools import tool
from src.utils.sandbox import run_in_sandbox
from .data_loader import DATASETS

@tool
def execute_python(code: str) -> str:
    """Execute arbitrary Python code in a sandboxed environment."""
    return run_in_sandbox(code, DATASETS)
