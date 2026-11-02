"""The public package name changes without breaking existing local configuration."""
from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys

import pytest

from locomanilab import __version__
from locomanilab.config import Context, load_context


@pytest.fixture
def isolated_config_environment(monkeypatch):
    for key in ("LOCOMANILAB_CONFIG", "LOCOMANILAB_WORKSPACE", "SOMA_CONFIG", "SOMA_WORKSPACE"):
        monkeypatch.delenv(key, raising=False)


def test_new_distribution_and_compatibility_command_share_one_entrypoint():
    distribution = metadata.distribution("locomanilab")
    assert distribution.metadata["Name"] == "locomanilab"
    commands = {entry.name: entry.value for entry in distribution.entry_points if entry.group == "console_scripts"}
    assert commands["lab"] == commands["locomanilab"] == commands["soma"] == "locomanilab.cli:main"


def test_package_module_command_runs_outside_checkout(tmp_path):
    environment = {key: value for key, value in os.environ.items()
                   if key != "PYTHONPATH" and not key.startswith(("LOCOMANILAB_", "SOMA_"))}
    result = subprocess.run([sys.executable, "-m", "locomanilab", "--version"], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == __version__


@pytest.mark.parametrize("prefix", ["LOCOMANILAB", "SOMA"])
def test_new_and_legacy_configuration_environment_variables(tmp_path, monkeypatch, isolated_config_environment, prefix):
    configuration = tmp_path / "settings.yaml"
    configuration.write_text("schema_version: 2\n")
    workspace = tmp_path / "workspace"
    monkeypatch.setenv(f"{prefix}_CONFIG", str(configuration))
    monkeypatch.setenv(f"{prefix}_WORKSPACE", str(workspace))
    context = load_context()
    assert context.source == configuration
    assert context.workspace_root == workspace


def test_new_environment_names_take_precedence_over_legacy(tmp_path, monkeypatch, isolated_config_environment):
    current = tmp_path / "new.yaml"
    current.write_text("schema_version: 2\n")
    monkeypatch.setenv("LOCOMANILAB_CONFIG", str(current))
    monkeypatch.setenv("LOCOMANILAB_WORKSPACE", str(tmp_path / "current"))
    monkeypatch.setenv("SOMA_CONFIG", str(tmp_path / "obsolete-missing.yaml"))
    monkeypatch.setenv("SOMA_WORKSPACE", str(tmp_path / "previous"))
    context = load_context()
    assert context.source == current
    assert context.workspace_root == tmp_path / "current"


def test_benchmark_default_uses_new_repository_name_and_explicit_legacy_path_still_works(tmp_path):
    context = Context(workspace=tmp_path)
    assert context.sim_repo() == tmp_path / "external/humanoid_benchmark"
    context.repos["sim_repo"] = "external/tele_humanoid_benchmark"
    assert context.sim_repo() == tmp_path / "external/tele_humanoid_benchmark"


def test_new_dataset_registry_names_use_the_project_name(plugin, dataset):
    from tests.conftest import SONIC_MODALITY
    source = dataset(SONIC_MODALITY, name="task")
    assert plugin("dm05")._dataset_name(source) == "locomanilab_task"
    assert plugin("xr1")._config_name(source) == "locomanilab_task"
    assert Path(plugin("dm05")._runtime_file("scripts/dm05/soma_dm05_exp.py", "dm05/experiment.py")).is_file()
