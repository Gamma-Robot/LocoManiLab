"""Guide checks use real configurations/pinned source plans, without model setup."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location(
        "policy_guide_checker", ROOT / "scripts/diagnostics/check_policy_guides.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def staged_source(checker, tmp_path_factory):
    destination = tmp_path_factory.mktemp("guide-source") / "source"
    checker.stage_source(ROOT, destination)
    return destination


@pytest.fixture
def example(staged_source, tmp_path):
    checkout = tmp_path / "source"
    shutil.copytree(staged_source, checkout)
    workspace = checkout / "workspaces/act"
    workspace.mkdir(parents=True)
    shutil.copy2(checkout / "policies/act/workspace.example.yaml", workspace / "env.local.yaml")
    return checkout


@pytest.fixture
def environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("LOCOMANILAB_", "SOMA_", "GIT_"))}


def test_real_example_resolves_without_backend_checkouts(checker, example, environment, monkeypatch):
    monkeypatch.setenv("SOMA_WORKSPACE", str(example / "wrong-workspace"))
    checker.check_workspace(example, "act", environment)
    assert os.environ["SOMA_WORKSPACE"] == str(example / "wrong-workspace")
    assert not list((example / "external/lerobot").iterdir())


@pytest.mark.parametrize("failure,match", [
    ("legacy_schema", "schema_version"),
    ("different_workspace", "different workspace"),
    ("missing_model_repo", "explicitly configure repos.lerobot"),
    ("wrong_repo_anchor", "declared gitlink"),
    ("unset_asset_path", "unset environment variable"),
    ("wrong_source_profile", "source profile does not cover"),
    ("stale_readme_pin", "complete fixed SHA"),
])
def test_invalid_guide_contracts_are_rejected(checker, example, environment, failure, match):
    path = example / "policies/act/workspace.example.yaml"
    config = yaml.safe_load(path.read_text())
    if failure == "legacy_schema":
        config["schema_version"] = 1
    elif failure == "different_workspace":
        config["workspace_root"] = "../another-workspace"
    elif failure == "missing_model_repo":
        del config["repos"]["lerobot"]
    elif failure == "wrong_repo_anchor":
        config["repos"]["lerobot"] = "../../../external/lerobot"
    elif failure == "unset_asset_path":
        config["paths"]["assets"] = "${LOCOMANILAB_UNSET_GUIDE_ASSETS}/weights"
    elif failure == "wrong_source_profile":
        profiles = example / "dependencies/profiles.json"
        value = json.loads(profiles.read_text())
        value["profiles"]["act"] = ["external/openpi"]
        profiles.write_text(json.dumps(value))
    else:
        readme = example / "policies/act/README.md"
        pin = checker.gitlinks(example)["external/lerobot"]
        readme.write_text(readme.read_text().replace(pin, "0" * 40))
    path.write_text(yaml.safe_dump(config))
    shutil.copy2(path, example / "workspaces/act/env.local.yaml")
    with pytest.raises((ValueError, RuntimeError), match=match):
        checker.check_workspace(example, "act", environment)


def test_real_readme_refuses_repeated_workspace_without_changing_data(checker, example, environment):
    workspace = example / "workspaces/act"
    for name in ("datasets/source/episode.parquet", "datasets/prepared/episode.parquet"):
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserve existing demonstration bytes")
    readme = (example / "policies/act/README.md").read_text()
    checker.check_repeat(checker.CPU_BLOCK.findall(readme)[0], example, "act", environment)


@pytest.mark.parametrize("block,match", [
    ("true", "must refuse"),
    ("printf changed > workspaces/act/env.local.yaml; false", "changed the existing workspace"),
])
def test_repeat_check_detects_success_or_mutation(checker, example, environment, block, match):
    with pytest.raises(ValueError, match=match):
        checker.check_repeat(block, example, "act", environment)


@pytest.mark.parametrize("leader_waits", [False, True])
def test_cpu_timeout_stops_real_descendants_even_after_leader_exit(checker, tmp_path, environment, leader_waits):
    marker = tmp_path / "worker.pid"
    worker = ("import os,signal,time; from pathlib import Path; "
              "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
              f"Path({str(marker)!r}).write_text(str(os.getpid())); "
              "print('worker ready',flush=True); time.sleep(30)")
    block = f"{shlex.quote(sys.executable)} -B -c {shlex.quote(worker)} & " + ("wait" if leader_waits else "true")
    try:
        with pytest.raises(subprocess.TimeoutExpired) as error:
            checker.run_cpu_block(block, tmp_path, environment, timeout=0.5)
        assert "worker ready" in error.value.output
        status = Path(f"/proc/{int(marker.read_text())}/stat")
        try:
            # Linux can briefly expose the dead state before removing /proc.
            assert status.read_text().split()[2] in {"Z", "X"}
        except (FileNotFoundError, ProcessLookupError):
            pass
    finally:
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
