"""Dataset contracts and tooling; importing this package needs only the stdlib."""
from pathlib import Path


def prepare_script_path() -> Path:
    """Locate the data-environment entrypoint without importing pandas or numpy."""
    return Path(__file__).resolve().parent / "prepare.py"
