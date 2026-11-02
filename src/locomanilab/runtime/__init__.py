"""Locate packaged child-process programs without loading model dependencies.

The selected model/data/simulator interpreter runs these files by absolute path.
It need not install locomanilab, and never inherits the CLI's site-packages directory.
"""
from pathlib import Path


def script_path(relative: str) -> Path:
    """A file or directory below this package, suitable for an external Python.

    Both wheel and editable installs provide these ordinary on-disk resources.
    Refuse unknown resources early, before starting an expensive model process.
    """
    root = Path(__file__).resolve().parent
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"runtime resource must be inside locomanilab.runtime: {relative!r}")
    if not path.exists():
        raise FileNotFoundError(f"packaged runtime resource missing: {relative}")
    return path
