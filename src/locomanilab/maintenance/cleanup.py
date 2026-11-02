"""Find (and let the CLI remove) empty checkpoint run-dirs and empty dataset dirs.

A crashed / aborted training still creates its run directory (and launch scaffolding
like experiment_cfg/, an empty checkpoints/ tree, …) but never writes a real checkpoint.
`lab clean-ckpts` uses this to sweep each model's default ckpt root. The same happens
one stage earlier: a collection you started to test the setup and stopped before
recording anything still leaves `outputs_sim/<backend>/<env>_<ts>/` behind with meta
(and, for --fixed-task, its sidecars) but no episode — `lab clean-datasets` sweeps those.

Each model declares in its manifest:

    ckpt:
      root:      outputs/gr00t_log          # repo-relative base holding the run dirs
      run_glob:  "*"                         # each run dir under <root>
      ckpt_glob: "checkpoint-*"              # a real checkpoint inside a run dir (str or list)

A run dir is "empty" only if it matches NO `ckpt_glob` AND holds no large file — the
large-file guard (`BIG_FILE_BYTES`) means even a wrong/incomplete glob can never flag a dir
that clearly contains weights. We also never flag a run a live process references, or one
touched within `min_age_min` minutes (a live run that just hasn't saved yet).
"""
from __future__ import annotations

import time
from pathlib import Path

from locomanilab.common import lerobot_meta

# A real checkpoint is always large; a pre-first-save crash leaves only small scaffolding
# (config, stats, logs). If a "no-ckpt-match" dir still holds a file this big, treat it as
# NOT empty — guards against a wrong ckpt_glob deleting a dir that actually has weights.
BIG_FILE_BYTES = 50 * 1024 * 1024


def _as_list(v) -> list[str]:
    if v is None:
        return []
    return [v] if isinstance(v, str) else list(v)


def _scan(root: Path) -> tuple[int, int, float]:
    """Single walk: (total_size, largest_file_size, newest_mtime) over root's subtree."""
    total = 0
    largest = 0
    newest = root.stat().st_mtime
    for p in root.rglob("*"):
        if p.is_symlink():
            # A linked file or directory may contain weights; never classify it as empty.
            return total, BIG_FILE_BYTES, newest
        try:
            st = p.stat()
        except OSError:
            continue
        newest = max(newest, st.st_mtime)
        if p.is_file() and not p.is_symlink():
            total += st.st_size
            largest = max(largest, st.st_size)
    return total, largest, newest


def _process_cmdlines() -> list[str]:
    """Best-effort list of running process command lines (Linux /proc; empty elsewhere),
    so we never delete a run dir a live training still references."""
    out: list[str] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return out
    for pd in proc.iterdir():
        if not pd.name.isdigit():
            continue
        try:
            raw = (pd / "cmdline").read_bytes()
        except OSError:
            continue
        if raw:
            out.append(raw.replace(b"\x00", b" ").decode("utf-8", "replace"))
    return out


def find_empty_ckpt_dirs(repo: Path, ckpt_cfg: dict, *, min_age_min: float = 0.0) -> list[dict]:
    """Run dirs under repo/<root> matching <run_glob> that hold no checkpoint.

    Returns [{path, mtime, size, active}]. Dirs that match a `ckpt_glob` or contain a
    >=BIG_FILE_BYTES file are skipped entirely (not empty). `active` marks the rest that a
    live process references or that were touched within `min_age_min` minutes — the caller
    must not delete those.
    """
    root = (repo / ckpt_cfg.get("root", ".")).resolve()
    if not root.exists():
        return []
    run_glob = ckpt_cfg.get("run_glob", "*")
    globs = _as_list(ckpt_cfg.get("ckpt_glob", "*"))
    now = time.time()
    cmdlines = _process_cmdlines()

    out: list[dict] = []
    for run in sorted(root.glob(run_glob)):
        if not run.is_dir() or run.is_symlink():
            continue
        if any(any(run.glob(g)) for g in globs):
            continue                                   # has a real checkpoint -> keep
        total, largest, mtime = _scan(run)
        if largest >= BIG_FILE_BYTES:
            continue                                   # holds weights in an unexpected layout -> keep
        recent = bool(min_age_min) and (now - mtime) < min_age_min * 60
        # Match the run as a path component; `/name` avoids bare-substring false positives.
        referenced = any(str(run) in c or f"/{run.name}" in c for c in cmdlines)
        out.append({"path": run, "mtime": mtime, "size": total, "active": recent or referenced})
    return out


# --------------------------------------------------------------------------- #
# Dataset dirs with no episode in them
# --------------------------------------------------------------------------- #

# What a recorded episode leaves on disk. A dataset dir holding one of these is NOT
# empty however its meta reads — the verdict must never rest on meta alone, since a
# collector killed mid-episode can leave the parquet/video written and episodes.jsonl not.
DATA_SUFFIXES = {".parquet", ".mp4", ".avi", ".mkv", ".hdf5", ".h5", ".npy", ".npz", ".bin"}
# ...and any sizeable file at all, whatever it is called: meta + the fixed-task sidecars
# are a few KB, so anything this big is data we do not recognise, not scaffolding.
DATA_FILE_BYTES = 2 * 1024 * 1024


def _episode_count(dataset: Path) -> int:
    """How many episodes this dataset's meta claims (meta/episodes.jsonl lines)."""
    try:
        text = (dataset / "meta" / "episodes.jsonl").read_text()
    except OSError:
        return 0                                        # absent = the collector never got there
    return sum(1 for line in text.splitlines() if line.strip())


def _dataset_scan(dataset: Path) -> tuple[bool, int, float]:
    """(holds_data, total_size, newest_mtime), stopping at the first data file found."""
    total = 0
    newest = dataset.stat().st_mtime
    for p in dataset.rglob("*"):
        if p.is_symlink():
            # Even a dangling link carries user intent and may be temporarily offline.
            return True, total, newest
        try:
            st = p.stat()
        except OSError:
            continue
        newest = max(newest, st.st_mtime)
        if not p.is_file() or p.is_symlink():
            continue
        total += st.st_size
        if p.suffix.lower() in DATA_SUFFIXES or st.st_size >= DATA_FILE_BYTES:
            return True, total, newest                  # sizes past here are never reported
    return False, total, newest


def _lerobot_dirs(root: Path, depth: int) -> list[Path]:
    """LeRobot dataset dirs at or under root, descending at most `depth` levels.

    Bounded on purpose: it must find `outputs_sim/<backend>/<env>_<ts>` given either the
    tree root or one dataset, without walking through the videos of every real dataset
    on the way. A dataset dir is never descended into.
    """
    if root.is_symlink():
        return []
    if lerobot_meta.is_dataset_dir(root):
        return [root]
    if depth <= 0:
        return []
    out: list[Path] = []
    try:
        children = sorted(root.iterdir())
    except OSError:
        return out
    for child in children:
        if child.is_dir() and not child.is_symlink():
            out += _lerobot_dirs(child, depth - 1)
    return out


def find_empty_datasets(roots, *, min_age_min: float = 0.0, max_depth: int = 3) -> list[dict]:
    """Dataset dirs under `roots` that recorded no episode at all.

    Returns [{path, episodes, mtime, size, active}] sorted by path. A dir counts as empty
    only if its meta lists zero episodes AND its subtree holds no data file (see
    `_dataset_scan`) — a dataset whose meta is broken but whose parquet/videos are there
    is kept. `active` marks the ones a live process references or that were touched within
    `min_age_min` minutes (a collection running right now looks exactly like an aborted
    one until its first episode lands); the caller must not delete those.
    """
    now = time.time()
    cmdlines = _process_cmdlines()
    out: list[dict] = []
    seen: set[Path] = set()
    for raw in roots:
        root = Path(raw).expanduser().absolute()
        if not root.is_dir():
            continue
        for ds in _lerobot_dirs(root, max_depth):
            if ds in seen:
                continue
            seen.add(ds)
            episodes = _episode_count(ds)
            holds_data, size, mtime = _dataset_scan(ds)
            if episodes or holds_data:
                continue
            recent = bool(min_age_min) and (now - mtime) < min_age_min * 60
            referenced = any(str(ds) in c or f"/{ds.name}" in c for c in cmdlines)
            out.append({"path": ds, "episodes": episodes, "mtime": mtime, "size": size,
                        "active": recent or referenced})
    return sorted(out, key=lambda e: str(e["path"]))
