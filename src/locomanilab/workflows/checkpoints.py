"""Locate checkpoint run roots used by local training and cleanup."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from locomanilab.config import Context, ModelManifest


def checkpoint_scans(ctx: Context, manifest: ModelManifest) -> list[tuple[Path, dict]]:
    """Return ``(root, cleanup_config)`` pairs without requiring model backends.

    Workspace outputs use the adapter's native run nesting. The manifest's
    repository-relative legacy root keeps its original patterns, so existing
    local runs remain visible after changing the default output location.
    """
    legacy = manifest.raw.get("ckpt")
    if not legacy:
        return []  # Cleanup needs a declared checkpoint pattern to be safe.
    name = manifest.name
    run_glob = {"fastwam": "*", "pi05": "*/*", "psi0": "*/*"}.get(name, "*")
    native = {**legacy, "root": ".", "run_glob": run_glob}
    scans: list[tuple[Path, dict]] = []

    def add(root: Path, config: dict) -> None:
        if not any(old_root == root and old_config == config for old_root, old_config in scans):
            scans.append((root, config))

    add(ctx.workspace_root / "checkpoints" / name, native)
    configured = ctx.path(f"{name}_outputs")
    if configured is not None:
        add(configured, native)
    if name in ("act", "dp"):
        shared = ctx.path("lerobot_outputs")
        if shared is not None:
            add(shared / name, native)
    legacy_root = ctx.anchor(ctx.repo_path(manifest.repo_key) / legacy.get("root", "."))
    add(legacy_root, {**legacy, "root": "."})
    return scans
