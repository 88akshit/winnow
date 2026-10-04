import io
import sys
import ast
import signal
import threading
from contextlib import redirect_stdout, redirect_stderr

# Modules strictly prohibited from being imported or accessed
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "pty", "commands",
    "posix", "nt", "ctypes", "multiprocessing", "threading", "signal",
    "importlib", "pickle", "shelve", "webbrowser", "http", "urllib", "requests"
}

# Functions/attributes strictly prohibited from being called or accessed
BLOCKED_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "getattr", "setattr", "delattr"
}

class SecurityASTValidator(ast.NodeVisitor):
    """Inspects the Python AST to block dangerous imports, attribute lookups, and builtins."""

    def visit_Import(self, node):
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module in BLOCKED_MODULES:
                raise PermissionError(f"Security Violation: Import of module '{alias.name}' is strictly blocked.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module in BLOCKED_MODULES:
                raise PermissionError(f"Security Violation: Importing from module '{node.module}' is strictly blocked.")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in BLOCKED_CALLS and isinstance(node.ctx, ast.Load):
            raise PermissionError(f"Security Violation: Direct access to '{node.id}' is strictly blocked.")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr in {"__subclasses__", "__bases__", "__class__", "__globals__", "__code__"}:
            raise PermissionError(f"Security Violation: Access to introspective attribute '{node.attr}' is blocked.")
        self.generic_visit(node)


def timeout_handler(signum, frame):
    raise TimeoutError("Execution exceeded the 30-second time limit.")


def run_in_sandbox(code: str, datasets: dict) -> str:
    """Run validated Python code with a 30-second timeout on the main thread.

    Execution uses a hardened environment with:
    1. Pre-execution static AST security check (blocks malicious imports/introspection).
    2. Restricted globals with safe builtins only.
    3. A 30-second SIGALRM timeout when called from the main thread.
    """
    # 1. Parse and validate syntax and security via AST
    try:
        parsed_ast = ast.parse(code)
    except SyntaxError as se:
        return f"Syntax Error: {str(se)}"

    validator = SecurityASTValidator()
    try:
        validator.visit(parsed_ast)
    except PermissionError as pe:
        return str(pe)

    ALLOWED_IMPORTS = {"pandas", "numpy", "seaborn", "matplotlib", "scipy", "sklearn", "math", "random", "json", "tabulate"}

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        base_name = name.split(".")[0]
        if base_name in BLOCKED_MODULES or base_name not in ALLOWED_IMPORTS:
            raise PermissionError(f"Security Violation: Import of '{name}' is not allowed in this sandbox.")
        return __import__(name, globals, locals, fromlist, level)

    # 2. Configure safe builtins (strip dangerous functions like open, eval, exec)
    safe_builtins = {
        "__import__": safe_import,
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float, "format": format,
        "int": int, "isinstance": isinstance, "issubclass": issubclass, "iter": iter,
        "len": len, "list": list, "map": map, "max": max, "min": min, "next": next,
        "pow": pow, "print": print, "range": range, "reversed": reversed,
        "round": round, "set": set, "slice": slice, "sorted": sorted,
        "str": str, "sum": sum, "tuple": tuple, "type": type, "zip": zip,
        "True": True, "False": False, "None": None,
    }

    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy
    import sklearn


    sandbox_globals = {
        "pd": pd,
        "np": np,
        "sns": sns,
        "plt": plt,
        "scipy": scipy,
        "sklearn": sklearn,
        "datasets": datasets,
        "__builtins__": safe_builtins,
    }

    # Python only allows signal handlers to be installed from the main thread.
    # LangChain may run synchronous tools in a worker thread, where SIGALRM
    # would raise ValueError before the sandbox can execute.
    has_signal_timeout = threading.current_thread() is threading.main_thread()
    if has_signal_timeout:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)

    output = io.StringIO()
    err = io.StringIO()

    try:
        with redirect_stdout(output), redirect_stderr(err):
            compiled_code = compile(parsed_ast, filename="<sandboxed_exec>", mode="exec")
            exec(compiled_code, sandbox_globals)

        res = output.getvalue()
        error_output = err.getvalue()
        if error_output:
            res += f"\nErrors:\n{error_output}"
        return res if res.strip() else "Code executed successfully with no output."
    except Exception as e:
        return f"Execution Error: {type(e).__name__}: {str(e)}"
    finally:
        if has_signal_timeout:
            signal.alarm(0)
