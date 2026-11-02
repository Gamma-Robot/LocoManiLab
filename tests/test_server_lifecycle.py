"""Real local sockets and subprocesses; no model, GPU, or external service."""
from __future__ import annotations

import socket
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from locomanilab.base import StageResult
from locomanilab.common import infer_check, sim
from locomanilab.config import Context


class Client:
    def __init__(self):
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        return StageResult(ok=True, message="client ran")


def free_port():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return server.getsockname()[1]


def invoke(which, tmp_path, argv, port, runner, *, dry_run=False):
    if which == "eval":
        return sim.orchestrate(support=sim.SimSupport(sim_repo=tmp_path), server_argv=argv,
            server_cwd=tmp_path, runner=runner, port=port, client_env_name="unused",
            env_name="EmptyEnv", prompt="test", episodes=1, max_steps=1, action_horizon=1,
            render="none", video_dir=None, conda_setup=None, dry_run=dry_run)
    return infer_check.orchestrate(server_argv=argv, server_cwd=tmp_path, runner=runner,
        port=port, client_env_name="unused", conda_setup=None, dataset=tmp_path, episode=0,
        frames=1, out=str(tmp_path / "plots/result.png"), prompt="test", dry_run=dry_run)


@pytest.mark.parametrize("which", ["eval", "infer"])
def test_occupied_port_never_starts_or_reuses_a_server(which, tmp_path):
    marker = tmp_path / "server-started"
    client = Client()
    with socket.socket() as existing:
        existing.bind(("127.0.0.1", 0))
        existing.listen(5)
        port = existing.getsockname()[1]
        argv = [sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"]
        result = invoke(which, tmp_path, argv, port, client)
        assert not result.ok and "already in use" in result.message
        assert client.calls == 0 and not marker.exists()
        assert existing.fileno() >= 0
        assert sim._port_in_use(port)


@pytest.mark.parametrize("which", ["eval", "infer"])
def test_server_early_exit_fails_without_waiting_for_timeout(which, tmp_path, monkeypatch):
    monkeypatch.setenv("SOMA_SERVER_WAIT_SECS", "1")
    client = Client()
    start = time.monotonic()
    result = invoke(which, tmp_path, [sys.executable, "-c", "raise SystemExit(7)"], free_port(), client)
    assert not result.ok and "exit 7" in result.message and client.calls == 0
    assert time.monotonic() - start < 2


def test_eval_runs_client_only_after_its_own_local_server_is_ready(tmp_path):
    port = free_port()
    code = ("import socket; s=socket.socket(); s.bind(('127.0.0.1'," + str(port) + ")); s.listen(5); "
            "exec(\"while True:\\n c,a=s.accept(); data=c.recv(4); c.sendall(b'pong') if data else None; c.close()\")")
    class ProtocolClient(Client):
        def run(self, *args, **kwargs):
            with socket.create_connection(("127.0.0.1", port), timeout=2) as connection:
                connection.sendall(b"ping")
                assert connection.recv(4) == b"pong"
            return super().run(*args, **kwargs)
    client = ProtocolClient()
    result = invoke("eval", tmp_path, [sys.executable, "-c", code], port, client)
    assert result.ok and client.calls == 1
    assert not sim._port_in_use(port)


@pytest.mark.parametrize("leader_exits", [False, True])
def test_shutdown_stops_workers_even_when_leader_exits_first(tmp_path, leader_exits):
    marker = tmp_path / "worker.pid"
    worker = ("import os,signal,time; from pathlib import Path; "
              "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
              f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)")
    parent = ("import subprocess,sys,time; from pathlib import Path; "
              f"p=subprocess.Popen([sys.executable,'-c',{worker!r}],stdout=subprocess.DEVNULL); "
              f"exec(\"while not Path({str(marker)!r}).exists(): time.sleep(.01)\"); "
              "print(p.pid,flush=True); " + ("" if leader_exits else "time.sleep(30)"))
    process = subprocess.Popen([sys.executable, "-c", parent], start_new_session=True,
                               stdout=subprocess.PIPE, text=True)
    try:
        worker_pid = int(process.stdout.readline().strip())
        if leader_exits:
            process.wait(timeout=2)
        sim._shutdown_server(process)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status = Path(f"/proc/{worker_pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(.02)
        else:
            pytest.fail("owned worker survived shutdown")
        assert process.poll() is not None
    finally:
        sim._shutdown_server(process)
        process.stdout.close()


@pytest.mark.parametrize("backend", ["amo", "heft", "scalebfm", "almi"])
def test_diagnosis_checks_selected_controller_without_requiring_sonic(tmp_path, monkeypatch, backend):
    runtime = tmp_path / "runtime"
    simulator = tmp_path / "sim"
    (simulator / "robocasa").mkdir(parents=True)
    folders = {"amo": "amo", "heft": "heft/pmg", "scalebfm": "scalebfm/humanoid_transformer_m",
               "almi": "almi/flat_dual"}
    folder = runtime / "checkpoints" / folders[backend]
    folder.mkdir(parents=True)
    (folder / "policy.yaml").write_text("model_path: policy.onnx\n")
    (folder / "policy.onnx").write_bytes(b"local test resource")
    for package in ("g1_bridge", "sim2real"):
        (runtime / package).mkdir()
        (runtime / package / "__init__.py").touch()
    monkeypatch.setattr(sim, "_conda_run", lambda *args, **kwargs: (True, "local test environment"))
    context = Context(repos={"g1_backend": str(runtime), "sim_repo": str(simulator)})
    report = sim.detect(context, backend=backend)
    assert report.ok, report.summary()
    assert not (runtime / "checkpoints/sonic").exists()


def test_selected_sonic_v11_does_not_require_low_latency(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    simulator = tmp_path / "sim"
    (simulator / "robocasa").mkdir(parents=True)
    folder = runtime / "checkpoints/sonic/v1_1/g1"
    folder.mkdir(parents=True)
    (folder / "policy.yaml").write_text("model_path: model_decoder.onnx\nencoder_model_path: model_encoder.onnx\n")
    for package in ("g1_bridge", "sim2real"):
        (runtime / package).mkdir()
        (runtime / package / "__init__.py").touch()
    for name, size in sim._SONIC_EXPORT_SIZES["v1_1"].items():
        with (folder / name).open("wb") as stream:
            stream.truncate(size)  # sparse placeholders; no weights are downloaded or read
    monkeypatch.setattr(sim, "_conda_run", lambda *args, **kwargs: (True, "local test environment"))
    context = Context(repos={"g1_backend": str(runtime), "sim_repo": str(simulator)})
    assert sim.detect(context, policy_config=folder / "policy.yaml").ok
    assert not (runtime / "checkpoints/sonic/low_latency").exists()


def test_term_to_runner_parent_does_not_leave_its_child_running(tmp_path):
    import signal
    marker = tmp_path / "owned-child.pid"
    worker = f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)"
    parent = ("from locomanilab.runner import Runner; from pathlib import Path; import sys; "
              f"Runner(log_root=Path({str(tmp_path / 'logs')!r})).run([sys.executable,'-c',{worker!r}],cwd={str(tmp_path)!r})")
    process = subprocess.Popen([sys.executable, "-c", parent], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    child_pid = None
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert marker.exists(), "owned child never started"
        child_pid = int(marker.read_text())
        process.terminate()
        assert process.wait(timeout=5) == 128 + signal.SIGTERM
        status = Path(f"/proc/{child_pid}/stat")
        assert not status.exists() or status.read_text().split()[2] == "Z"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if child_pid is not None:
            try:
                os.killpg(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.fixture
def configured_sim(tmp_path, monkeypatch):
    """Tiny importable checkouts and a local activation shim; no conda/GPU required."""
    workspace = tmp_path / "workspace"
    runtime = workspace / "configured-runtime"
    simulator = workspace / "configured-simulator"
    other = tmp_path / "other-editable-install"
    files = {
        runtime / "g1_bridge/__init__.py": "ORIGIN = 'configured'\n",
        runtime / "g1_bridge/_cuda_order.py": "",
        runtime / "sim2real/__init__.py": "",
        simulator / "robocasa/__init__.py": "",
        simulator / "robocasa/g1_host/__init__.py": "",
        simulator / "robocasa/g1_host/wbt.py": "",
        simulator / "robocasa/scripts/__init__.py": "",
        simulator / "robocasa/scripts/collect_demos_g1.py": "import g1_bridge; print(g1_bridge.__file__)\n",
        simulator / "robocasa/scripts/replay_augment.py": "import g1_bridge; print(g1_bridge.__file__)\n",
        simulator / "zmq.py": "",
        simulator / "msgpack_numpy.py": "",
        other / "g1_bridge/__init__.py": "ORIGIN = 'other'\n",
        runtime / "checkpoints/amo/policy.yaml": "model_path: policy.onnx\n",
        runtime / "checkpoints/amo/policy.onnx": "local test resource",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    # Runner's activation wrapper selects this existing CPU interpreter.
    (workspace / "setup.sh").write_text(f"conda() {{ return 0; }}\nexport CONDA_PREFIX={sys.prefix!r}\n")
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setenv("PYTHONPATH", str(other))
    ctx = Context(workspace=workspace, repos={"g1_backend": str(runtime), "sim_repo": str(simulator)},
                  paths={"conda_setup": "setup.sh"})
    return ctx, runtime, simulator, other


def test_diagnosis_resolves_relative_setup_and_does_not_write_bytecode(configured_sim):
    ctx, runtime, simulator, _ = configured_sim
    report = sim.detect(ctx, backend="amo")
    assert report.ok, report.summary()
    assert report.runtime_repo == runtime
    assert not list(runtime.rglob("__pycache__"))
    assert not list(simulator.rglob("__pycache__"))


def test_eval_client_imports_the_same_configured_runtime_as_diagnosis(configured_sim):
    ctx, runtime, simulator, _ = configured_sim
    report = sim.detect(ctx, backend="amo")
    assert report.ok, report.summary()
    _, extra_env = sim.build_client_cmd(sim_repo=report.sim_repo, runtime_repo=report.runtime_repo,
        env_name="unused", host="localhost", port=5550, model="EmptyEnv", prompt="test", episodes=1,
        max_steps=1, action_horizon=1, render="video", video_dir=None)
    process = subprocess.run([sys.executable, "-B", "-c",
        "import g1_bridge,robocasa; print(g1_bridge.__file__); print(robocasa.__file__)"],
        env={**os.environ, **extra_env}, text=True, capture_output=True, check=True)
    assert process.stdout.splitlines() == [str(runtime / "g1_bridge/__init__.py"),
                                           str(simulator / "robocasa/__init__.py")]


@pytest.mark.parametrize("workflow", ["collect", "replay"])
def test_simulation_workflows_import_configured_controller(configured_sim, workflow):
    from locomanilab.runner import Runner
    from locomanilab.workflows import Workflows
    ctx, runtime, _, _ = configured_sim
    workflows = Workflows(ctx, Runner(log_root=ctx.workspace_root / "logs"))
    if workflow == "collect":
        result = workflows.collect(task_prompt="test", out_dir=Path("output"), sim=True)
    else:
        result = workflows.replay(source_runs=[ctx.workspace_root / "synthetic-source"])
    assert result.ok, result.message
    assert str(runtime / "g1_bridge/__init__.py") in result.output.splitlines()


def test_manifest_with_non_string_keys_reports_invalid_instead_of_crashing(configured_sim):
    ctx, runtime, _, _ = configured_sim
    (runtime / "checkpoints/amo/policy.yaml").write_text("1: unsupported\n")
    report = sim.detect(ctx, backend="amo")
    assert not report.ok and "keys must be strings" in report.summary()


@pytest.mark.parametrize("missing_package", ["g1_bridge", "sim2real"])
@pytest.mark.parametrize("workflow", ["detect", "collect", "replay"])
def test_incomplete_configured_runtime_never_falls_back_to_old_environment(configured_sim, missing_package, workflow):
    from locomanilab.runner import Runner
    from locomanilab.workflows import Workflows
    ctx, runtime, _, other = configured_sim
    (runtime / missing_package / "__init__.py").unlink()
    (other / "sim2real").mkdir(exist_ok=True)
    (other / "sim2real/__init__.py").touch()
    if workflow == "detect":
        result = sim.detect(ctx, backend="amo")
        message = result.summary()
    else:
        workflows = Workflows(ctx, Runner(log_root=ctx.workspace_root / "logs"))
        if workflow == "collect":
            result = workflows.collect(task_prompt="test", out_dir=Path("output"), sim=True)
        else:
            result = workflows.replay(source_runs=[ctx.workspace_root / "synthetic-source"])
        message = result.message
    assert not result.ok and "checkout is incomplete" in message
    assert not (ctx.workspace_root / "logs").exists()
