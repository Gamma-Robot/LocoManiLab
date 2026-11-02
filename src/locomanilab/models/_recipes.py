"""Dataset-bound recipe helpers that never import a model framework."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

from locomanilab.data.paths import dataset_identity


def openpi_repo_id(repo: Path, config_name: str, *, expected_factory: str | None = None) -> str:
    """Inspect a literal upstream TrainConfig without importing JAX or torch.

    Dynamically generated recipes are deliberately refused: claiming to know their
    dataset without running their configuration code would hide the original bug.
    """
    source = repo / "src/openpi/training/config.py"
    try:
        tree = ast.parse(source.read_text())
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"cannot inspect openpi recipes in {source}: {exc}") from exc
    matches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        if target != "TrainConfig":
            continue
        values = {kw.arg: kw.value for kw in node.keywords}
        name = values.get("name")
        if not isinstance(name, ast.Constant) or name.value != config_name:
            continue
        data = values.get("data")
        if isinstance(data, ast.Call):
            factory = data.func.id if isinstance(data.func, ast.Name) else getattr(data.func, "attr", "")
            if expected_factory and factory != expected_factory:
                raise ValueError(
                    f"openpi recipe {config_name!r} uses {factory}, but this controller needs {expected_factory}"
                )
            repo_id = next((kw.value for kw in data.keywords if kw.arg == "repo_id"), None)
            if isinstance(repo_id, ast.Constant) and isinstance(repo_id.value, str):
                matches.append(repo_id.value)
    if len(matches) != 1:
        raise ValueError(
            f"cannot establish the dataset of openpi recipe {config_name!r}; "
            "select a literal TrainConfig with an explicit data.repo_id using --config-name"
        )
    return matches[0]


def dataset_home(dataset: Path, repo_id: str) -> Path:
    """The LeRobot cache root that makes this repo_id read this exact directory."""
    parts = PurePosixPath(repo_id).parts
    if not parts or PurePosixPath(repo_id).is_absolute() or any(p in (".", "..") for p in parts):
        raise ValueError(f"unsafe dataset repo_id in recipe: {repo_id!r}")
    if tuple(dataset.parts[-len(parts):]) != parts:
        raise ValueError(
            f"--dataset {dataset} does not match recipe data.repo_id={repo_id!r}. "
            "Choose a matching --config-name or register a matching recipe in your openpi checkout. "
            "No training was started."
        )
    return dataset.parents[len(parts) - 1]


def recipe_key(dataset: Path, name: str) -> str:
    payload = json.dumps({"name": name, **dataset_identity(dataset)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
