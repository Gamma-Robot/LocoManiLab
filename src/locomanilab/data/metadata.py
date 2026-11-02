"""Read a LeRobot dataset's `meta/` files -- the one place that knows the layout.

Pure stdlib on purpose: metadata and controller selection are also used by
the standalone prepare entrypoint in a separate data environment.

`is_dataset_dir` is the single test for "this directory is a LeRobot dataset":
meta/info.json exists. cli's range selection, clean-datasets and prepare all use it;
three slightly different judgements (meta/ present, meta+data present, info.json
present) used to make the same tree look different to each command.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def meta_path(dataset: Path | str, name: str) -> Path:
    return Path(dataset) / "meta" / name


def is_dataset_dir(path: Path | str) -> bool:
    """A LeRobot dataset root: has meta/info.json."""
    return meta_path(path, "info.json").is_file()


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text())


def read_jsonl(path: Path | str, limit: int | None = None) -> list[dict]:
    out: list[dict] = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
        if limit and len(out) >= limit:
            break
    return out


def load_meta(dataset: Path | str, name: str) -> Any | None:
    """Parsed meta/<name> (json or jsonl by suffix), or None when it is absent.

    A present-but-unparsable file is a real error and is raised as such -- the
    callers that used to swallow it reported "not a LeRobot dataset" for what was
    a half-written file.
    """
    p = meta_path(dataset, name)
    if not p.is_file():
        return None
    return read_jsonl(p) if p.suffix == ".jsonl" else read_json(p)


def load_info(dataset: Path | str) -> dict | None:
    return load_meta(dataset, "info.json")


def load_modality(dataset: Path | str) -> dict | None:
    return load_meta(dataset, "modality.json")


def load_tasks(dataset: Path | str) -> list[dict] | None:
    return load_meta(dataset, "tasks.jsonl")


def modality_keys(modality: dict | None, group: str) -> set[str]:
    """Key names under one modality group ("state" / "action" / "video" / ...)."""
    return set(((modality or {}).get(group) or {}).keys())
