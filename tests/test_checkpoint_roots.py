"""Checkpoint cleanup follows local output roots without importing model backends."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from locomanilab.cli import app
from locomanilab.config import load_manifests
from locomanilab.workflows.checkpoints import checkpoint_scans


@pytest.mark.parametrize("model,run,checkpoint", [
    ("gr00t", "run", "checkpoint-100"),
    ("fastwam", "20261021_120000", "checkpoints/weights/step_000100.pt"),
    ("pi05", "recipe/experiment", "100/params"),
    ("psi0", "sonic/experiment.20261021", "checkpoints/ckpt_100"),
    ("act", "task", "checkpoints/000100"),
])
def test_workspace_native_nesting_and_checkpoint_patterns(ctx, model, run, checkpoint, monkeypatch):
    from locomanilab.maintenance import cleanup
    monkeypatch.setattr(cleanup, "_process_cmdlines", lambda: [])
    manifest = load_manifests()[model]
    root = ctx.workspace_root / "checkpoints" / model
    complete = root / run
    (complete / checkpoint).mkdir(parents=True)
    empty = complete.with_name(complete.name + "_empty")
    empty.mkdir()
    scans = checkpoint_scans(ctx, manifest)
    native = next(config for path, config in scans if path == root)
    rows = cleanup.find_empty_ckpt_dirs(root, native)
    assert [row["path"] for row in rows] == [empty]


def test_roots_include_configured_outputs_and_unchanged_legacy_patterns(ctx):
    ctx.paths["fastwam_outputs"] = "/local/fastwam-runs"
    manifest = load_manifests()["fastwam"]
    scans = checkpoint_scans(ctx, manifest)
    assert (Path("/local/fastwam-runs"), {**manifest.raw["ckpt"], "root": ".", "run_glob": "*"}) in scans
    legacy = ctx.repo_path("fastwam") / "runs"
    assert (legacy, {**manifest.raw["ckpt"], "root": "."}) in scans


@pytest.mark.parametrize("model", ["act", "dp"])
def test_lerobot_shared_output_root_includes_model_subdirectory(ctx, model):
    ctx.paths["lerobot_outputs"] = "/local/lerobot-runs"
    roots = [root for root, _ in checkpoint_scans(ctx, load_manifests()[model])]
    assert Path("/local/lerobot-runs") / model in roots
    assert Path("/local/lerobot-runs") not in roots


def test_explicit_absolute_legacy_root_is_preserved(ctx):
    manifest = load_manifests()["gr00t"]
    manifest.raw["ckpt"]["root"] = "/archived/gr00t"
    roots = [root for root, _ in checkpoint_scans(ctx, manifest)]
    assert Path("/archived/gr00t") in roots


def test_clean_ckpts_lists_and_rechecks_workspace_configured_and_legacy_roots(ctx, monkeypatch):
    import locomanilab.cli.maintenance as commands
    from locomanilab.maintenance import cleanup
    monkeypatch.setattr(cleanup, "_process_cmdlines", lambda: [])
    ctx.paths["gr00t_outputs"] = str(ctx.workspace_root / "custom-gr00t")
    manifest = load_manifests()["gr00t"]
    monkeypatch.setattr(commands, "_ctx", lambda: ctx)
    monkeypatch.setattr(commands, "available_models", lambda: {"gr00t": manifest})
    empty_runs = []
    complete_runs = []
    for root, _ in checkpoint_scans(ctx, manifest):
        empty = root / "aborted"
        empty.mkdir(parents=True)
        (empty / "config.json").write_text("{}")
        empty_runs.append(empty)
        complete = root / "complete"
        (complete / "checkpoint-100").mkdir(parents=True)
        complete_runs.append(complete)
    listed = CliRunner().invoke(app, ["clean-ckpts", "-M", "gr00t", "--min-age", "0"])
    assert listed.exit_code == 0, listed.output
    assert all(str(path) in listed.output and path.exists() for path in empty_runs)
    deleted = CliRunner().invoke(app, ["clean-ckpts", "-M", "gr00t", "--min-age", "0", "--yes"])
    assert deleted.exit_code == 0, deleted.output
    assert not any(path.exists() for path in empty_runs)
    assert all(path.exists() for path in complete_runs)
