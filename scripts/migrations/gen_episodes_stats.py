#!/usr/bin/env python3
"""Generate numeric per-episode statistics for a LeRobot v2.1 dataset.

Computes statistics from episode parquet files, skipping non-numeric/video
columns. This utility is for a loader that requires episodes_stats.jsonl; check
that model's normalization contract before changing existing dataset metadata.
An existing output is preserved unless --force is provided.

Usage:
    python scripts/migrations/gen_episodes_stats.py /path/to/dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Quantiles lerobot/PRTS expect alongside mean/std/min/max.
QUANTILES = {"q01": 0.01, "q99": 0.99}


def episode_files(root: Path) -> dict[int, list[Path]]:
    """Map episode_index -> parquet shard paths (data/chunk-*/episode_*.parquet)."""
    out: dict[int, list[Path]] = {}
    for p in sorted((root / "data").rglob("episode_*.parquet")):
        try:
            idx = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        out.setdefault(idx, []).append(p)
    return out


def column_stats(col: pd.Series) -> dict | None:
    """mean/std/min/max/q01/q99 for one column, shaped like meta/stats.json."""
    first = col.iloc[0]
    if isinstance(first, (list, np.ndarray)):
        arr = np.stack([np.asarray(v, dtype=np.float64) for v in col.to_numpy()])
    else:
        arr = np.asarray(col.to_numpy(), dtype=np.float64).reshape(-1, 1)
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return None

    stats = {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        # lerobot's aggregate_stats() weights each episode's mean/variance by "count"
        # and asserts its shape is exactly (1,) -- so it's a 1-element list, not a scalar.
        "count": [int(arr.shape[0])],
    }
    for name, q in QUANTILES.items():
        stats[name] = np.quantile(arr, q, axis=0).tolist()
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args()

    out_path = args.root / "meta" / "episodes_stats.jsonl"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} already exists (use --force to regenerate)")
        return 0

    files = episode_files(args.root)
    if not files:
        print(f"[err ] no episode parquet files under {args.root/'data'}", file=sys.stderr)
        return 1

    lines: list[str] = []
    for ep_idx in sorted(files):
        df = pd.concat([pd.read_parquet(p) for p in files[ep_idx]], ignore_index=True)
        ep_stats: dict[str, dict] = {}
        for col in df.columns:
            try:
                s = column_stats(df[col])
            except (TypeError, ValueError):
                continue          # non-numeric (e.g. string task) -> not needed
            if s is not None:
                ep_stats[col] = s
        lines.append(json.dumps({"episode_index": ep_idx, "stats": ep_stats}))
        print(f"  ep {ep_idx:>4}: {len(df):>5} frames, {len(ep_stats)} cols", flush=True)

    out_path.write_text("\n".join(lines) + "\n")
    print(f"[ok  ] wrote {out_path} ({len(lines)} episodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
