import os
from pathlib import Path

# Project root directory (anchor for safe operations)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Allowed directory paths that the agent is permitted to read data from
ALLOWED_DATA_DIRS = [
    PROJECT_ROOT / "sample_data",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "outputs",
]

# Create standard data folder if not present
(PROJECT_ROOT / "data").mkdir(exist_ok=True)
(PROJECT_ROOT / "outputs").mkdir(exist_ok=True)

def validate_safe_read_path(file_path: str) -> Path:
    """
    Validates that a requested file path resides strictly inside allowed directories.
    Prevents path traversal attacks (e.g., ../../.ssh/id_rsa, /etc/passwd).
    """
    candidate = Path(file_path).expanduser()

    # If a relative path was given, resolve it relative to PROJECT_ROOT
    if not candidate.is_absolute():
        resolved = (PROJECT_ROOT / candidate).resolve()
    else:
        resolved = candidate.resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check if resolved path is contained within any allowed directory
    allowed = False
    for allowed_dir in ALLOWED_DATA_DIRS:
        resolved_allowed = allowed_dir.resolve()
        try:
            resolved.relative_to(resolved_allowed)
            allowed = True
            break
        except ValueError:
            continue

    if not allowed:
        allowed_names = [str(d.relative_to(PROJECT_ROOT)) for d in ALLOWED_DATA_DIRS]
        raise PermissionError(
            f"Security Error: Access denied to '{file_path}'. "
            f"Files can only be read from safe directories: {allowed_names}"
        )

    return resolved
