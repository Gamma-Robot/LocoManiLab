"""Locate source checkouts without confusing installed code with user storage."""
from __future__ import annotations

import os
import re
from pathlib import Path


def source_root(start: Path | None = None) -> Path | None:
    """Find this project's source tree, including a source snapshot without Git."""
    start = (start or Path(__file__)).resolve()
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file() and (
            (parent / "src/locomanilab/config.py").is_file() or (parent / "locomanilab/config.py").is_file()
        ):
            return parent
    return None


def absolute_path(value: str | Path, base: Path | None = None) -> Path:
    """Anchor a path while retaining a symlink's user-facing dataset name."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("filesystem paths must be non-empty strings")
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if re.search(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)", expanded):
        raise ValueError(f"unset environment variable in path {value!r}")
    path = Path(expanded)
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    # Keep '..' for the filesystem to interpret after following symlinks.
    # Lexical normalization can silently select a different dataset.
    return path.absolute()
