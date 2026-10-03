import io
import sys
import signal
from contextlib import redirect_stdout, redirect_stderr

def timeout_handler(signum, frame):
    raise TimeoutError("Execution exceeded the 30-second time limit.")

def run_in_sandbox(code: str, datasets: dict) -> str:
    """Executes python code in a sandboxed environment with timeout."""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy
    import sklearn
    
    sandbox_globals = {
        "pd": pd, "np": np, "sns": sns, "plt": plt,
        "scipy": scipy, "sklearn": sklearn,
        "datasets": datasets,
        "__builtins__": __builtins__
    }
    
    output = io.StringIO()
    err = io.StringIO()
    
    try:
        with redirect_stdout(output), redirect_stderr(err):
            exec(code, sandbox_globals)
        res = output.getvalue()
        error_output = err.getvalue()
        if error_output:
            res += f"\nErrors:\n{error_output}"
        return res if res.strip() else "Code executed successfully with no output."
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        signal.alarm(0)
