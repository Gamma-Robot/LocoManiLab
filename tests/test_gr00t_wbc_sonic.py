"""SONIC data provenance selects matching GR00T-WBC hardware deployment assets.

Collection uses g1_backend; GR00T-WBC is retained only for real-robot deployment.
Deployment command checks use a fake asset tree and never launch a controller.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from locomanilab import cli
from locomanilab.runner import Runner
from locomanilab.workflows import Workflows
from locomanilab.workflows.collection import CollectionWorkflow


def _wbc_assets(root: Path, variant: str) -> None:
    d = root / "gear_sonic_deploy" / "policy" / variant
    d.mkdir(parents=True)
    for f in ("model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"):
        (d / f).write_text("")


def _v1_1_manifest(tree) -> str:
    return str(Path(tree["repos"]["g1_backend"]) / "checkpoints/sonic/v1_1/g1/policy.yaml")


def test_deploy_default_has_no_controller_flag(plugin):
    r = plugin("gr00t").deploy(model_path="/tmp/ckpt", prompt="test")
    assert r.ok and "--deploy-checkpoint" not in r.command and "--low-latency" not in r.command


def test_deploy_v1_1_forwards_wbc_deploy_assets(plugin, tree):
    _wbc_assets(Path(tree["repos"]["gr00t_wbc"]), "sonic_v1_1")
    r = plugin("gr00t").deploy(model_path="/tmp/ckpt", prompt="test",
                               policy_config=_v1_1_manifest(tree))
    assert r.ok
    assert ("bash start_inference.sh --model-path /tmp/ckpt --prompt test "
            "--deploy-checkpoint policy/sonic_v1_1/model "
            "--deploy-obs-config policy/sonic_v1_1/observation_config.yaml") in r.command


def test_deploy_low_latency_uses_launcher_preset(plugin, tree):
    pc = _v1_1_manifest(tree).replace("v1_1", "low_latency")
    r = plugin("gr00t").deploy(model_path="/tmp/ckpt", prompt="test", policy_config=pc)
    assert r.ok and "--prompt test --low-latency" in r.command and "--deploy-checkpoint" not in r.command


def test_deploy_v1_1_without_assets_is_refused_with_fetch_hint(plugin, tree):
    r = plugin("gr00t").deploy(model_path="/tmp/ckpt", prompt="test",
                               policy_config=_v1_1_manifest(tree))
    assert not r.ok and "download_from_hf.py --sonic-v1-1" in r.message
    assert "policy/sonic_v1_1/model_encoder.onnx" in r.message


def test_deploy_unknown_variant_is_refused(plugin, tree):
    pc = _v1_1_manifest(tree).replace("v1_1", "custom")
    r = plugin("gr00t").deploy(model_path="/tmp/ckpt", prompt="test", policy_config=pc)
    assert not r.ok and "custom" in r.message and "--sonic-ckpt" in r.message


def test_server_only_refuses_controller_switch(plugin, tree):
    r = plugin("gr00t").deploy(model_path="/tmp/ckpt", server_only=True,
                               policy_config=_v1_1_manifest(tree))
    assert not r.ok and "--server-only" in r.message


def test_g1_collection_and_wbc_deployment_select_the_same_sonic_variant(plugin, tree):
    _wbc_assets(Path(tree["repos"]["gr00t_wbc"]), "sonic_v1_1")
    p = plugin("gr00t")
    c = p.collect(task_prompt="test", out_dir=Path("/tmp/out"), backend="sonic",
                  pico_source="mock", camera_runtime="none", real_runtime="g1_backend",
                  policy_config=_v1_1_manifest(tree))
    d = p.deploy(model_path="/tmp/ckpt", prompt="test", policy_config=_v1_1_manifest(tree))
    assert c.ok and d.ok
    assert "g1_bridge.collect_real" in c.command
    assert f"--policy-config {_v1_1_manifest(tree)}" in c.command
    assert ("--deploy-checkpoint policy/sonic_v1_1/model "
            "--deploy-obs-config policy/sonic_v1_1/observation_config.yaml") in d.command


@pytest.mark.parametrize("sim", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_wbc_collection_is_refused_before_runtime_or_device_start(ctx, tmp_path, monkeypatch, sim, dry_run):
    from locomanilab.common import xrt_service

    def forbidden(*args, **kwargs):
        pytest.fail("removed WBC collection must not start a runtime or device")

    workflow = Workflows(ctx, Runner(dry_run=dry_run, log_root=tmp_path / "logs"))
    monkeypatch.setattr(workflow.runner, "run", forbidden)
    monkeypatch.setattr(CollectionWorkflow, "_collect_sim", forbidden)
    monkeypatch.setattr(CollectionWorkflow, "_collect_real_g1_backend", forbidden)
    monkeypatch.setattr(xrt_service, "ensure_service", forbidden)
    monkeypatch.setattr(workflow, "_start_headset_video", forbidden)
    monkeypatch.setattr(workflow, "_bring_up_robot_camera", forbidden)
    result = workflow.collect(task_prompt="test", out_dir=tmp_path / "output", sim=sim,
                              real_runtime="gr00t_wbc", pico_source="real", headset_video=True,
                              policy_config="${LOCOMANILAB_UNSET_REMOVED_RUNTIME}/policy.yaml")
    assert not result.ok
    assert "collection uses g1_backend" in result.message
    assert "only for real-robot deployment" in result.message
    assert not (tmp_path / "output").exists() and not (tmp_path / "logs").exists()


@pytest.mark.parametrize("sim", [False, True])
def test_collect_cli_rejects_wbc_before_configuration_or_sonic_asset_checks(monkeypatch, tmp_path, sim):
    def forbidden(*args, **kwargs):
        pytest.fail("removed runtime must be rejected before configuration or asset checks")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.data, "_workflow", forbidden)
    monkeypatch.setattr(cli.data, "_resolve_sonic_ckpt", forbidden)
    argv = ["collect", "--real-runtime", "gr00t_wbc", "--pico-source", "real"]
    if sim:
        argv += ["--sim", "--headset-video"]
    result = CliRunner().invoke(cli.app, argv)
    assert result.exit_code == 1, result.output
    assert "collection uses g1_backend" in result.output
    assert "only for real-robot deployment" in result.output
    assert not list(tmp_path.iterdir())


def test_collect_help_advertises_only_the_shared_g1_runtime():
    result = CliRunner().invoke(cli.app, ["collect", "--help"])
    assert result.exit_code == 0, result.output
    assert "g1_backend" in result.output and "heft" in result.output
    assert "gr00t_wbc" not in result.output and "GR00T-WBC" not in result.output


def test_fastwam_deploy_v1_1_replaces_low_latency_preset(plugin, tree):
    _wbc_assets(Path(tree["repos"]["gr00t_wbc"]), "sonic_v1_1")
    r = plugin("fastwam").deploy(run_dir="/tmp/run", prompt="test", base_model="/tmp/base",
                                 policy_config=_v1_1_manifest(tree))
    assert r.ok
    assert ("--deploy-checkpoint policy/sonic_v1_1/model "
            "--deploy-obs-config policy/sonic_v1_1/observation_config.yaml") in r.command


def test_fastwam_deploy_low_latency_adds_nothing(plugin, tree):
    """The launcher's own preset IS low_latency -- no duplicate --low-latency."""
    pc = _v1_1_manifest(tree).replace("v1_1", "low_latency")
    a = plugin("fastwam").deploy(run_dir="/tmp/run", prompt="test", base_model="/tmp/base")
    b = plugin("fastwam").deploy(run_dir="/tmp/run", prompt="test", base_model="/tmp/base",
                                 policy_config=pc)
    assert a.ok and b.ok and a.command == b.command
