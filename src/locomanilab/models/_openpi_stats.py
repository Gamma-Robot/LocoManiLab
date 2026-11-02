"""Run native openpi normalization in its own environment with local assets.

This file is executed by path in the model environment. Keep its top-level
imports standard-library-only and do not require locomanilab to be installed there.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-home", type=Path, required=True)
    parser.add_argument("--assets-base-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_LEROBOT_HOME"] = str(args.dataset_home)
    sys.path.insert(0, str(args.repo / "src"))
    sys.path.insert(0, str(args.repo))
    config_module = importlib.import_module("openpi.training.config")
    config = config_module.get_config(args.config_name)
    if config.data.repo_id != args.repo_id:
        raise ValueError("openpi recipe changed after planning; recompute the command")
    assets_dir = args.assets_base_dir / args.config_name
    assets = dataclasses.replace(config.data.assets, assets_dir=str(assets_dir), asset_id=args.repo_id)
    bound = dataclasses.replace(
        config, data=dataclasses.replace(config.data, assets=assets),
        assets_base_dir=str(args.assets_base_dir),
    )
    # Update only this process's recipe registry. Backend source files stay untouched.
    config_module._CONFIGS_DICT[args.config_name] = bound
    native = runpy.run_path(str(args.repo / "scripts/compute_norm_stats.py"))
    native["main"](args.config_name)


if __name__ == "__main__":
    main()
