"""Public CLI entry point."""
from .app import app, main
from . import benchmark as benchmark, data as data, maintenance as maintenance, models as models
__all__ = ["app", "main"]
