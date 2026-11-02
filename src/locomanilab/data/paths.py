"""Stdlib-only path guards shared by standalone data conversion programs."""
from __future__ import annotations

import os
import hashlib
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from string import Formatter
from uuid import uuid4


_FIELDS = {"episode_index", "episode_chunk", "video_key"}


def dataset_identity(dataset: Path) -> dict:
    """Bind an artifact to the caller's source path and schema/episode metadata.

    This hashes metadata, not all video/parquet contents. It catches another
    source with the same basename and ordinary schema/episode changes without
    scanning an entire training dataset.
    """
    digest = hashlib.sha256()
    for name in ("info.json", "modality.json", "episodes.jsonl", "episodes_stats.jsonl",
                 "tasks.jsonl", "stats.json"):
        path = dataset / "meta" / name
        if path.is_file():
            digest.update(name.encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return {"dataset": str(dataset), "source": str(dataset.resolve()), "metadata_sha256": digest.hexdigest()}


def dataset_file(root: Path, pattern: str, *, episode_index: int,
                 episode_chunk: int = 0, video_key: str = "", output: bool = False) -> Path:
    """Expand a dataset-local template; writing may never traverse a symlink.

    Source files may deliberately be symlinks (conversion staging commonly shares
    videos). Templates themselves must always be relative, for reads and writes.
    """
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("dataset path template must be a nonempty string")
    for _, name, _, conversion in Formatter().parse(pattern):
        if name is not None and (name not in _FIELDS or conversion is not None):
            raise ValueError(f"unsupported field in dataset path template: {name!r}")
    try:
        rendered = pattern.format(episode_index=episode_index, episode_chunk=episode_chunk, video_key=video_key)
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        raise ValueError(f"invalid dataset path template {pattern!r}: {exc}") from exc
    relative = Path(rendered)
    if relative.is_absolute() or PureWindowsPath(rendered).is_absolute() or ".." in relative.parts:
        raise ValueError(f"dataset path template must stay inside the dataset: {pattern!r}")
    if not relative.parts or relative == Path("."):
        raise ValueError("dataset path template must name a file")
    root = Path(root).absolute()
    path = root / relative
    if output:
        if root.is_symlink():
            raise ValueError(f"output directory must not be a symlink: {root}")
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"output path must not traverse a symlink: {current}")
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"output path escapes the dataset: {path}")
    return path


def output_directory(sources, destination: Path, *, force: bool = False) -> Path:
    """Check physical source/output separation before creating or replacing anything."""
    raw = Path(destination).expanduser().absolute()
    if raw.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {raw}")
    target = raw.resolve()
    for source in sources:
        source = Path(source).expanduser().resolve()
        if target == source or target.is_relative_to(source) or source.is_relative_to(target):
            raise ValueError(f"output directory must not overlap source {source}: {target}")
    if target.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {target}; choose a new destination")
        if not target.is_dir():
            raise ValueError(f"output destination is not a directory: {target}")
    return target


@contextmanager
def staged_output(sources, destination: Path, *, force: bool = False):
    """Publish a complete new directory only after conversion succeeds.

    A failed --force conversion keeps the previous output. The source is never a
    valid replacement target, including through symlinked parent directories.
    """
    sources = tuple(Path(source) for source in sources)
    target = output_directory(sources, destination, force=force)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    backup = None
    try:
        yield staging
        # Recheck after child workers have finished, immediately before publication.
        checked = output_directory(sources, destination, force=force)
        if checked != target:
            raise ValueError("output directory changed while conversion was running")
        if target.exists():
            backup = target.with_name(f".{target.name}.backup-{uuid4().hex}")
            target.rename(backup)
        try:
            os.rename(staging, target)
        except BaseException:
            if backup is not None:
                backup.rename(target)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
