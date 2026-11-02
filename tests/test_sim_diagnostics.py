"""Selected-controller diagnostics; no simulator, model, or downloads execute."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner
import yaml

from locomanilab.cli import app
from locomanilab.common import sim
from locomanilab.config import Context
from locomanilab.maintenance.doctor import diagnose


@pytest.fixture
def local_sources(tmp_path, monkeypatch):
    for name in ("LOCOMANILAB_CONFIG", "LOCOMANILAB_WORKSPACE", "SOMA_CONFIG", "SOMA_WORKSPACE"):
        monkeypatch.delenv(name, raising=False)
    runtime, simulator = tmp_path / "controller", tmp_path / "simulator"
    for package in ("g1_bridge", "sim2real"):
        (runtime / package).mkdir(parents=True)
        (runtime / package / "__init__.py").touch()
    (simulator / "robocasa").mkdir(parents=True)
    source = tmp_path / "env.local.yaml"
    values = {"schema_version": 2,
              "repos": {"g1_backend": str(runtime), "sim_repo": str(simulator)},
              "paths": {"conda_setup": "setup.sh"}}
    source.write_text(yaml.safe_dump(values))
    context = Context(source=source, repos=values["repos"], paths=values["paths"])
    monkeypatch.setattr(sim, "_conda_run", lambda *args, **kwargs: (True, "CPU import probe stub"))

    def forbidden(*args, **kwargs):
        pytest.fail("diagnostics must not launch model processes or downloads")

    monkeypatch.setattr(subprocess, "run", forbidden)
    return context


def selected_controller(context, selection):
    if selection == "sonic_v1_1":
        manifest = context.repo_path("g1_backend") / "checkpoints/sonic/v1_1/g1/policy.yaml"
        sizes = sim._SONIC_EXPORT_SIZES["v1_1"]
        content = "model_path: model_decoder.onnx\nencoder_model_path: model_encoder.onnx\n"
        options = ["--sonic-ckpt", "v1_1"]
    else:
        manifest = context.repo_path("g1_backend") / "checkpoints/heft/pmg/policy.yaml"
        sizes, content, options = {"policy.onnx": 0}, "model_path: policy.onnx\n", ["--backend", "heft"]
    manifest.parent.mkdir(parents=True)
    manifest.write_text(content)
    for name, size in sizes.items():
        # Sparse files satisfy local size checks without storing or reading weights.
        with (manifest.parent / name).open("wb") as stream:
            stream.truncate(size)
    return manifest, options


@pytest.mark.parametrize("selection", ["heft", "sonic_v1_1"])
def test_doctor_checks_only_the_selected_controller(local_sources, selection):
    context = local_sources
    manifest, options = selected_controller(context, selection)
    result = CliRunner().invoke(app, ["--config", str(context.source), "doctor", "--sim",
                                      "--stage", "collect", "--json", *options])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["ok"] and any(check["detail"] == str(manifest) for check in report["checks"])
    assert "low_latency" not in result.output
    assert not (context.repo_path("g1_backend") / "checkpoints/sonic/low_latency").exists()


def test_doctor_resolves_custom_policy_from_the_callers_directory(local_sources, tmp_path, monkeypatch):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    manifest = caller / "custom.yaml"
    manifest.write_text("model_path: custom.onnx\n")
    (caller / "custom.onnx").touch()
    result = CliRunner().invoke(app, ["--config", str(local_sources.source), "doctor", "--sim",
                                      "--backend", "amo", "--policy-config", "custom.yaml", "--json"])
    assert result.exit_code == 0, result.output
    assert str(manifest) in result.output and "low_latency" not in result.output


@pytest.mark.parametrize("missing", ["manifest", "weights"])
def test_doctor_reports_missing_selected_resources_instead_of_requiring_downloads(local_sources, missing):
    context = local_sources
    manifest = context.repo_path("g1_backend") / "checkpoints/sonic/v1_1/g1/policy.yaml"
    if missing == "weights":
        manifest.parent.mkdir(parents=True)
        manifest.write_text("model_path: model_decoder.onnx\nencoder_model_path: model_encoder.onnx\n")
    result = CliRunner().invoke(app, ["--config", str(context.source), "doctor", "--sim",
                                      "--sonic-ckpt", "v1_1", "--json"])
    assert result.exit_code == 1, result.output
    report = json.loads(result.output)
    assert not report["ok"] and "checks" in report and "error" not in report
    assert any(not check["ok"] and str(manifest.parent) in check["detail"] for check in report["checks"])
    assert "low_latency" not in result.output


@pytest.mark.parametrize("flags,match", [
    (["--sim", "--sonic-ckpt", "v1_1", "--policy-config", "custom.yaml"], "not both"),
    (["--sim", "--backend", "heft", "--sonic-ckpt", "v1_1"], "requires --backend sonic"),
    (["--sim", "--backend", "unknown"], "unknown backend"),
    (["--backend", "heft"], "require --sim"),
])
def test_doctor_selection_errors_are_clear_json(local_sources, flags, match):
    result = CliRunner().invoke(app, ["--config", str(local_sources.source), "doctor", "--json", *flags])
    assert result.exit_code == 1, result.output
    assert match in json.loads(result.output)["error"]


def test_unselected_doctor_preserves_the_existing_detect_defaults(local_sources, monkeypatch):
    calls = []

    def detect(context, **options):
        calls.append(options)
        return sim.SimSupport(ok=True)

    monkeypatch.setattr(sim, "detect", detect)
    assert diagnose(local_sources, stage="collect", sim=True)["ok"]
    assert calls == [{}]


def test_doctor_probes_the_selected_sim_env(local_sources, monkeypatch):
    calls = []

    def detect(context, **options):
        calls.append(options)
        return sim.SimSupport(ok=True)

    monkeypatch.setattr(sim, "detect", detect)
    assert diagnose(local_sources, stage="collect", sim=True, sim_env="my_sim")["ok"]
    assert calls == [{"conda_env": "my_sim"}]


@pytest.fixture
def preflight(local_sources, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/diagnostics/preflight_sim.py"
    spec = importlib.util.spec_from_file_location("preflight_selected_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.cfg, "load_context", lambda: local_sources)
    return module


@pytest.mark.parametrize("selection", ["heft", "sonic_v1_1", "custom"])
def test_preflight_runs_the_selected_local_paths_without_unrelated_weights(
    local_sources, preflight, selection, monkeypatch, tmp_path
):
    context = local_sources
    if selection == "custom":
        caller = tmp_path / "caller"
        caller.mkdir()
        monkeypatch.chdir(caller)
        manifest = caller / "custom.yaml"
        manifest.write_text("model_path: custom.onnx\n")
        (caller / "custom.onnx").touch()
        options = ["--backend", "amo", "--policy-config", "custom.yaml"]
    else:
        manifest, options = selected_controller(context, selection)
    calls = []

    def wrap(self, command, **kwargs):
        assert kwargs["setup"] == str(context.workspace_root / "setup.sh")
        return command

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(preflight.Runner, "wrap", wrap)
    monkeypatch.setattr(preflight.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["preflight_sim.py", "--env", "EmptyEnv", *options])
    assert preflight.main() == 1  # Stubbed execution deliberately produces no video.
    assert len(calls) == 1
    command, process_options = calls[0]
    if selection == "heft":
        assert "--policy-config" not in command
    else:
        assert command[command.index("--policy-config") + 1] == str(manifest)
    expected_backend = {"heft": "heft", "sonic_v1_1": "sonic", "custom": "amo"}[selection]
    assert command[command.index("--backend") + 1] == expected_backend
    assert process_options["env"]["PYTHONPATH"].startswith(str(context.repo_path("g1_backend")))
    assert process_options["cwd"] == str(context.sim_repo())


@pytest.mark.parametrize("flags,match", [
    (["--sonic-ckpt", "v1_1", "--policy-config", "custom.yaml"], "not both"),
    (["--backend", "heft", "--sonic-ckpt", "v1_1"], "requires --backend sonic"),
])
def test_preflight_refuses_conflicting_selection_before_any_probe(preflight, monkeypatch, capsys, flags, match):
    monkeypatch.setattr(preflight.simmod, "detect", lambda *a, **kw: pytest.fail("must reject before probing"))
    monkeypatch.setattr(sys, "argv", ["preflight_sim.py", "--env", "EmptyEnv", *flags])
    with pytest.raises(SystemExit) as error:
        preflight.main()
    assert error.value.code == 2
    assert match in capsys.readouterr().err
