"""Real CPU conversion provenance; policy backends are never executed."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from locomanilab.cli import app
from locomanilab.config import Context
from locomanilab.plugin import StageResult
from locomanilab.registry import build_plugin
from locomanilab.runtime import script_path

pytest.importorskip("pyarrow")
pytest.importorskip("av")


def sample(root: Path, destination: Path, frames: int) -> Path:
    result = CliRunner().invoke(app, ["--workspace", str(root), "sample-data", "--out", str(destination),
                                     "--episodes", "1", "--frames", str(frames)])
    assert result.exit_code == 0, result.output
    assert next((destination / "data").rglob("*.parquet")).is_file()
    return destination


def convert(source: Path, destination: Path, cwd: Path) -> None:
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    environment.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(script_path("lerobot/convert.py")),
                             "--src", str(source), "--out", str(destination), "--action-parts", "action"],
                            cwd=cwd, env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (destination / "meta/episodes_stats.jsonl").is_file()


def snapshot(root: Path) -> dict:
    return {str(path.relative_to(root)): ("symlink:" + os.readlink(path) if path.is_symlink()
            else hashlib.sha256(path.read_bytes()).hexdigest())
            for path in root.rglob("*") if path.is_symlink() or path.is_file()}


@pytest.fixture
def converted(tmp_path):
    first = sample(tmp_path, tmp_path / "source-a/shared", frames=3)
    second = sample(tmp_path, tmp_path / "source-b/shared", frames=4)
    output = tmp_path / "converted/shared"
    convert(first, output, cwd=tmp_path)
    context = Context(workspace=tmp_path, repos={"lerobot": str(tmp_path / "backend-not-installed")},
                      paths={"lerobot_data_root": "converted", "lerobot_outputs": "checkpoints"})
    return context, first, second, output


def backend_forbidden(context: Context, model: str, monkeypatch):
    adapter = build_plugin(model, context)
    monkeypatch.setattr(adapter.runner, "run", lambda *a, **kw: pytest.fail("policy backend must not start"))
    return adapter


@pytest.mark.parametrize("model", ["act", "dp"])
def test_convert_rejects_same_basename_from_a_different_source(converted, model, monkeypatch):
    context, _, second, output = converted
    before = snapshot(output)
    result = backend_forbidden(context, model, monkeypatch).convert(dataset=second)
    assert not result.ok and "source provenance" in result.message
    assert json.loads((second / "meta/info.json").read_text())["total_frames"] == 4
    assert json.loads((output / "meta/info.json").read_text())["total_frames"] == 3
    assert snapshot(output) == before


@pytest.mark.parametrize("model", ["act", "dp"])
def test_convert_reuses_the_unchanged_source_without_starting_a_backend(converted, model, monkeypatch):
    context, first, _, output = converted
    before = snapshot(output)
    result = backend_forbidden(context, model, monkeypatch).convert(dataset=first)
    assert result.ok, result.message
    assert "already converted" in result.message
    assert snapshot(output) == before


def change_metadata(source: Path) -> None:
    path = source / "meta/info.json"
    info = json.loads(path.read_text())
    info["robot_type"] = "changed-source-metadata"
    path.write_text(json.dumps(info))


def change_tasks(source: Path) -> None:
    path = source / "meta/tasks.jsonl"
    task = json.loads(path.read_text())
    task["task"] = "changed task instruction"
    path.write_text(json.dumps(task) + "\n")


@pytest.mark.parametrize("reason", ["metadata_changed", "tasks_changed", "legacy_without_record"])
def test_convert_refuses_unproven_existing_data(converted, reason, monkeypatch):
    context, first, _, output = converted
    if reason == "metadata_changed":
        change_metadata(first)
    elif reason == "tasks_changed":
        change_tasks(first)
    else:
        (output / "meta/locomanilab_source.json").unlink()
    before = snapshot(output)
    result = backend_forbidden(context, "act", monkeypatch).convert(dataset=first)
    assert not result.ok and "source provenance" in result.message
    assert snapshot(output) == before


@pytest.mark.parametrize("model", ["act", "dp"])
@pytest.mark.parametrize("reason", ["other_source", "metadata_changed", "tasks_changed", "legacy_without_record"])
def test_train_checks_provenance_before_starting_the_backend(converted, model, reason, monkeypatch):
    context, first, second, output = converted
    selected = second if reason == "other_source" else first
    if reason == "metadata_changed":
        change_metadata(first)
    elif reason == "tasks_changed":
        change_tasks(first)
    elif reason == "legacy_without_record":
        (output / "meta/locomanilab_source.json").unlink()
    before = snapshot(output)
    result = backend_forbidden(context, model, monkeypatch).train(dataset=selected, gpus=1)
    assert not result.ok and "source provenance" in result.message
    assert snapshot(output) == before
    assert not (context.workspace_root / "checkpoints").exists()


@pytest.mark.parametrize("model", ["act", "dp"])
def test_train_accepts_matching_provenance_at_the_backend_boundary(converted, model, monkeypatch):
    context, first, _, _ = converted
    adapter = build_plugin(model, context)
    calls = []

    def intercepted_backend(*args, **kwargs):
        calls.append(kwargs["stage"])
        return StageResult(ok=True, message="backend intercepted; no model executed")

    monkeypatch.setattr(adapter.runner, "run", intercepted_backend)
    result = adapter.train(dataset=first, gpus=1)
    assert result.ok, result.message
    assert calls == ["train"]


@pytest.mark.parametrize("model", ["act", "dp"])
@pytest.mark.parametrize("override", ["dataset.root=/different-unverified-dataset",
                                      "--dataset.repo_id=another/source", "dataset={root: /another-source}"])
def test_train_overrides_cannot_bypass_the_verified_source(converted, model, override, monkeypatch):
    context, first, _, output = converted
    before = snapshot(output)
    result = backend_forbidden(context, model, monkeypatch).train(dataset=first, gpus=1, overrides=[override])
    assert not result.ok and "--dataset" in result.message and "dataset-bound" in result.message
    assert snapshot(output) == before


def test_converter_preserves_the_callers_alias_and_rejects_retargeting(tmp_path, monkeypatch):
    original = sample(tmp_path, tmp_path / "physical-a", frames=3)
    replacement = sample(tmp_path, tmp_path / "physical-b", frames=3)
    alias = tmp_path / "shared"
    alias.symlink_to(original, target_is_directory=True)
    output = tmp_path / "converted/shared"
    convert(alias, output, cwd=tmp_path)
    provenance = json.loads((output / "meta/locomanilab_source.json").read_text())
    assert provenance["dataset"] == str(alias)
    assert provenance["source"] == str(original.resolve())
    assert len(provenance["metadata_sha256"]) == 64
    context = Context(workspace=tmp_path, paths={"lerobot_data_root": "converted"})
    adapter = backend_forbidden(context, "act", monkeypatch)
    assert adapter.convert(dataset=alias).ok
    alias.unlink()
    alias.symlink_to(replacement, target_is_directory=True)
    result = adapter.convert(dataset=alias)
    assert not result.ok and "source provenance" in result.message
