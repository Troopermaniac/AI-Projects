"""
Filesystem sandbox   resolves symlinks and checks real paths.
Only allows file operations inside the project directory (and its subdirectories).
Prevents access to files outside the project, even via relative or absolute paths.
"""

import os


def get_project_root():
    """Return the resolved (real) path of the project root."""
    return os.path.realpath(os.path.dirname(__file__))


# Resolve once at import time so it can't be tampered with later.
PROJECT_ROOT = get_project_root()


class FilesystemViolationError(Exception):
    """Raised when a file operation tries to escape the project directory."""
    pass


def enforce_path(path: str, mode: str = "rw") -> None:
    """
    Resolve the real path of *path* and verify it stays inside PROJECT_ROOT.

    Parameters
    
    path : str
        Absolute or relative file / directory path to validate.
    mode : str
        One of 'r', 'w', 'a', 'x', 'd' (read, write, append, create, delete).
        Used only for logging / future policy expansion.

    Raises
    
    FilesystemViolationError
        If the resolved path is outside PROJECT_ROOT.
    """
    # Resolve symlinks and normalize
    real_path = os.path.realpath(path)

    # Ensure it starts with project root (with trailing separator to avoid
    # prefix collisions like /AI Project vs /AI Project2)
    if not (real_path == PROJECT_ROOT or real_path.startswith(PROJECT_ROOT + os.sep)):
        raise FilesystemViolationError(
            f"Filesystem violation: '{path}' resolves to '{real_path}', "
            f"which is outside the project directory ({PROJECT_ROOT})."
        )


def enforce_read(path: str) -> None:
    """Validate that a read path stays inside the project."""
    enforce_path(path, mode="r")


def enforce_write(path: str) -> None:
    """Validate that a write/create path stays inside the project."""
    enforce_path(path, mode="w")


def enforce_delete(path: str) -> None:
    """Validate that a delete/remove path stays inside the project."""
    enforce_path(path, mode="d")
