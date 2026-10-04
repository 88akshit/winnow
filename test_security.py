import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.sandbox import run_in_sandbox
from src.tools.data_loader import load_csv

print("=" * 60)
print("🛡️ Running Winnow Security Tests")
print("=" * 60)

# Test 1: Malicious import os
print("\nTest 1: Attempting to import 'os' inside sandbox...")
res1 = run_in_sandbox("import os\nos.system('echo hacked')", {})
print(f"Result: {res1}")
assert "Security Violation" in res1, "Failed: Malicious import was not blocked!"
print("✅ PASSED: Blocked unauthorized module import.")

# Test 2: Attempting to use open() to read files
print("\nTest 2: Attempting to call 'open()' inside sandbox...")
res2 = run_in_sandbox("f = open('.env', 'r')\nprint(f.read())", {})
print(f"Result: {res2}")
assert "Security Violation" in res2 or "Execution Error" in res2, "Failed: open() was not blocked!"
print("✅ PASSED: Blocked file open access.")

# Test 3: Path traversal on load_csv
print("\nTest 3: Attempting path traversal on load_csv('/etc/passwd')...")
res3 = load_csv.invoke({"file_path": "/etc/passwd", "name": "passwd"})
print(f"Result: {res3}")
assert "Security Error: Access denied" in res3, "Failed: Path traversal was not blocked!"
print("✅ PASSED: Path traversal rejected.")

# Test 4: Legitimate math & pandas code
print("\nTest 4: Running legitimate data code in sandbox...")
res4 = run_in_sandbox("""
import pandas as pd
import numpy as np
df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
print('Mean of b:', df['b'].mean())
""", {})
print(f"Result: {res4.strip()}")
assert "Mean of b: 20.0" in res4, "Failed: Legitimate code did not execute!"
print("✅ PASSED: Legitimate data analysis code runs perfectly.")

print("\n" + "=" * 60)
print("🎉 ALL SECURITY CHECKS PASSED SUCCESSFULLY!")
print("=" * 60)
