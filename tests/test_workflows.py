"""Workflow services have no model-plugin dependency."""
from pathlib import Path
import sys

from locomanilab.runner import Runner
from locomanilab.workflows import Workflows


def test_collection_service_works_without_constructing_a_model(ctx, tmp_path, monkeypatch):
    import locomanilab.registry
    monkeypatch.setattr(locomanilab.registry, "build_plugin", lambda *a, **kw: (_ for _ in ()).throw(AssertionError()))
    workflow = Workflows(ctx, Runner(dry_run=True, log_root=tmp_path / "logs"))
    result = workflow.collect(task_prompt="test", out_dir=tmp_path / "demo", sim=True)
    assert result.ok and "--env EmptyEnv" in result.command


def test_every_model_retains_compatibility_access_to_prepare_and_replay(plugin, tmp_path):
    # These methods used to exist only on two plugins despite being model-independent.
    adapter = plugin("pi05")
    missing = adapter.prepare(datasets=[])
    assert not missing.ok and "no datasets" in missing.message
    replay = adapter.replay(source_runs=[])
    assert not replay.ok and "no source runs" in replay.message


def test_data_workflow_uses_packaged_script_and_current_interpreter_in_plan(ctx, tmp_path):
    source = tmp_path / "dataset"
    (source / "meta").mkdir(parents=True)
    workflow = Workflows(ctx, Runner(dry_run=True, log_root=tmp_path / "logs"))
    result = workflow.prepare(datasets=[source], out=Path("prepared"))
    assert result.ok, result.message
    assert sys.executable in result.command
    assert "locomanilab/data/prepare.py" in result.command
    assert str(ctx.workspace_root / "prepared") in result.command


def test_evaluation_dry_run_never_sources_a_user_setup_script(plugin, ctx, tmp_path):
    marker = tmp_path / "setup_was_executed"
    setup = tmp_path / "setup.sh"
    setup.write_text(f"touch '{marker}'\n")
    ctx.paths["conda_setup"] = str(setup)
    result = plugin("gr00t").eval(model_path="/tmp/planned-checkpoint")
    assert result.ok, result.message
    assert "readiness was not checked" in result.message
    assert not marker.exists()
