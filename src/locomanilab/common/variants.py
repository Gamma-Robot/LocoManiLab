"""Compatibility alias; implementation now lives in locomanilab.data.variants."""
if __name__ == "__main__":
    from pathlib import Path
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'data/variants.py'), run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module('locomanilab.data.variants')
