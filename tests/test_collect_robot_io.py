"""Real-robot collection chooses the latency-appropriate RobotIO path."""

import pytest


def _collect(plugin, tmp_path, *, backend: str, robot_io=None, real_runtime="auto"):
    return plugin("gr00t").collect(
        task_prompt="test",
        out_dir=tmp_path / backend,
        backend=backend,
        pico_source="mock",
        max_episodes=None,
        camera_runtime="none",
        camera_connect="tcp://127.0.0.1:5594",
        tmux=True,
        robot_io=robot_io,
        real_runtime=real_runtime,
    )


@pytest.mark.parametrize("backend", ["almi", "heft", "sonic", "scalebfm", "amo"])
@pytest.mark.parametrize("real_runtime", ["auto", "g1_backend"])
def test_every_backend_defaults_to_inline_and_skips_the_bridge(plugin, tmp_path, backend, real_runtime):
    result = _collect(plugin, tmp_path, backend=backend, real_runtime=real_runtime)
    assert result.ok
    assert "g1_bridge.launch_collect_real" in result.command
    assert "--robot-io inline" in result.command
    assert "--no-bridge" in result.command


@pytest.mark.parametrize("backend", ["almi", "heft"])
def test_explicit_zmq_remains_available_for_ab_testing(plugin, tmp_path, backend):
    result = _collect(plugin, tmp_path, backend=backend, robot_io="zmq")
    assert result.ok
    assert "--robot-io zmq" in result.command
    assert "--no-bridge" not in result.command
