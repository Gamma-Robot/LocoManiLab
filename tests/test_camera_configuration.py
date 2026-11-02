"""Remote camera startup must use the robot chosen in workspace settings."""
from types import SimpleNamespace

import pytest

from locomanilab.workflows.devices import DeviceWorkflow


@pytest.mark.parametrize("deploy, runtime, hint", [
    ({}, "orin_sender", "deploy.orin_ssh_host"),
    ({"orin_ssh_host": "user@chosen-robot"}, "orin_sender", "deploy.orin_listen"),
])
def test_camera_without_configuration_does_not_contact_a_robot(
    tmp_path, monkeypatch, capsys, deploy, runtime, hint,
):
    script = tmp_path / "scripts/g1/start_orin_camera.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\n")
    workflow = DeviceWorkflow(SimpleNamespace(deploy=deploy), SimpleNamespace(dry_run=False))

    def forbidden(*args, **kwargs):
        raise AssertionError("camera startup contacted a robot without configuration")

    monkeypatch.setattr("locomanilab.workflows.devices.subprocess.run", forbidden)
    workflow._bring_up_robot_camera(tmp_path, "tcp://robot-camera:5555", runtime)
    assert hint in capsys.readouterr().out


def test_camera_uses_configured_robot_and_headset_endpoint(tmp_path, monkeypatch):
    script = tmp_path / "scripts/g1/start_orin_camera.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\n")
    workflow = DeviceWorkflow(SimpleNamespace(deploy={
        "orin_ssh_host": "user@chosen-robot", "orin_listen": "headset-link:13579",
    }), SimpleNamespace(dry_run=False))
    calls = []
    monkeypatch.setattr("locomanilab.workflows.devices.subprocess.run",
                        lambda argv: calls.append(argv) or SimpleNamespace(returncode=0))
    workflow._bring_up_robot_camera(tmp_path, "tcp://robot-camera:5594", "orin_sender")
    assert calls == [[str(script), "--mode", "listen", "--host", "user@chosen-robot",
                      "--zmq", "tcp://*:5594", "--listen", "headset-link:13579"]]


def test_headless_camera_needs_only_the_configured_host(tmp_path, monkeypatch):
    script = tmp_path / "scripts/g1/start_orin_camera.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/bash\n")
    workflow = DeviceWorkflow(SimpleNamespace(deploy={"orin_ssh_host": "user@chosen-robot"}),
                              SimpleNamespace(dry_run=False))
    calls = []
    monkeypatch.setattr("locomanilab.workflows.devices.subprocess.run",
                        lambda argv: calls.append(argv) or SimpleNamespace(returncode=0))
    workflow._bring_up_robot_camera(tmp_path, "tcp://robot-camera:5555", "orin_headless")
    assert calls == [[str(script), "--mode", "headless", "--host", "user@chosen-robot",
                      "--zmq", "tcp://*:5555"]]
