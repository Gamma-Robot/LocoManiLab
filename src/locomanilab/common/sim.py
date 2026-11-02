"""Simulation eval support: detect whether this machine can run the humanoid_benchmark
sim, and launch the closed-loop eval client that connects to a gr00t-compat policy server.

The sim repo (LocoManiLab/humanoid_benchmark) is the
`external/humanoid_benchmark` submodule (env.local `paths.sim_repo` still overrides
for machines that keep a separate clone); it runs in its own conda env (default
`tele_benchmark`). See docs/sim_validation.md.
"""
from __future__ import annotations

import os

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SimSupport:
    ok: bool = True
    sim_repo: Path | None = None
    conda_env: str = "tele_benchmark"
    checks: list[tuple[str, bool, str]] = field(default_factory=list)  # (name, ok, detail)
    runtime_repo: Path | None = None

    def summary(self) -> str:
        lines = ["sim eval support:"]
        lines.append(f"  sim_repo: {self.sim_repo or '(missing submodule external/humanoid_benchmark)'}")
        lines.append(f"  conda env: {self.conda_env}")
        for name, ok, detail in self.checks:
            lines.append(f"  [{'ok ' if ok else 'MISSING'}] {name}: {detail}")
        lines.append(f"  => {'READY' if self.ok else 'NOT READY'}")
        return "\n".join(lines)


def _conda_run(conda_root: str | None, env: str, code: str,
               setup: str | None = None) -> tuple[bool, str]:
    """Run a python snippet inside a conda env; return (ok, stdout/err tail).

    Source the same optional environment setup used by actual runs before
    activating conda. Suppress bytecode writes while importing local checkouts.
    """
    from locomanilab.runner import Runner
    # Share Runner's activation and PATH handling so the probe uses the same
    # interpreter as the client, even when locomanilab itself runs inside a virtualenv.
    argv = Runner(conda_root=conda_root).wrap(["python", "-B", "-c", code], env_kind="conda",
                                              env_name=env, setup=setup)
    try:
        p = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    except subprocess.TimeoutExpired:
        return False, "simulation environment import check timed out after 30 seconds"
    except OSError as exc:
        return False, f"could not start simulation environment check: {exc}"
    lines = (p.stdout or "").strip().splitlines()
    return p.returncode == 0, lines[-1] if lines else ""


# Known SONIC export sizes distinguish variants whose filenames are identical.
# A motion token must be decoded by the variant used during collection. These
# lightweight size checks detect common export mixups; they are not checksums.
_SONIC_EXPORT_SIZES = {
    "low_latency": {
        "model_encoder.onnx": 45_933_505,
        "model_decoder.onnx": 149_763_762,
    },
    "v1_1": {
        "model_encoder.onnx": 50_117_796,
        "model_decoder.onnx": 149_778_638,
    },
}


def _wrong_sonic_export(ckpt_dir: Path, variant: str = "low_latency") -> str | None:
    """Return a complaint if this dir holds an export other than ``variant``."""
    bad = []
    for name, want in _SONIC_EXPORT_SIZES[variant].items():
        got = (ckpt_dir / name).stat().st_size          # follows symlinks
        if got != want:
            bad.append(f"{name} is {got:,} B, expected {want:,} B")
    if not bad:
        return None
    return (f"wrong SONIC export for {variant} (a policy's motion_token will "
            "decode as noise, silently): " + "; ".join(bad))


def detect(ctx, conda_env: str = "tele_benchmark", *, backend: str = "sonic",
           policy_config: str | Path | None = None) -> SimSupport:
    """Inspect only the selected controller and local simulation environment."""
    import yaml

    support = SimSupport(conda_env=conda_env, sim_repo=ctx.sim_repo())
    sim_repo = support.sim_repo
    present = sim_repo.is_dir() and (sim_repo / "robocasa").is_dir()
    support.checks.append(("sim_repo", present, str(sim_repo)))
    client = client_path()
    support.checks.append(("eval client", client.is_file(), str(client)))
    defaults = {
        "sonic": "sonic/low_latency/g1/policy.yaml", "amo": "amo/policy.yaml",
        "heft": "heft/pmg/policy.yaml", "scalebfm": "scalebfm/humanoid_transformer_m/policy.yaml",
        "almi": "almi/flat_dual/policy.yaml",
    }
    if backend not in defaults:
        support.checks.append(("controller", False, f"unknown backend {backend!r}"))
        support.ok = False
        return support
    runtime = ctx.repo_path("g1_backend")
    support.runtime_repo = runtime
    source_error = runtime_checkout_error(runtime)
    support.checks.append(("controller source", source_error is None, source_error or str(runtime)))
    selected = (ctx.input_path(policy_config) if policy_config else runtime / "checkpoints" / defaults[backend])
    support.checks.append((f"{backend} manifest", selected.is_file(), str(selected)))
    if selected.is_file():
        try:
            config = yaml.safe_load(selected.read_text())
            if not isinstance(config, dict):
                raise ValueError("policy manifest must be a mapping")
            if not all(isinstance(key, str) for key in config):
                raise ValueError("policy manifest keys must be strings")
            weights = [value for key, value in config.items()
                       if (key == "model_path" or key.endswith("_model_path") or key in
                           {"amo_adapter_path", "amo_norm_stats_path"}) and value]
            if not all(isinstance(value, str) for value in weights):
                raise ValueError("model paths must be strings")
            required = [Path(value) if Path(value).is_absolute() else selected.parent / value for value in weights]
            if not required:
                required = list(selected.parent.glob("*.onnx")) + list(selected.parent.glob("*.pt"))
                if not required:
                    support.checks.append((f"{backend} weights", False, f"no local weights beside {selected}"))
            for path in required:
                support.checks.append(("controller resource", path.is_file(), str(path)))
            if backend == "sonic":
                variant = next((name for name in _SONIC_EXPORT_SIZES if name in selected.parts), None)
                if variant and all((selected.parent / name).is_file() for name in _SONIC_EXPORT_SIZES[variant]):
                    wrong = _wrong_sonic_export(selected.parent, variant)
                    support.checks.append((f"SONIC {variant} export", not wrong, wrong or str(selected.parent)))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            support.checks.append(("controller manifest", False, str(exc)))
    if not all(ok for _, ok, _ in support.checks):
        support.ok = False
        return support
    # Probe the configured checkouts, not a different editable install on sys.path.
    setup = ctx.path("conda_setup")
    ok_env, detail = _conda_run(ctx.conda_root, conda_env,
        "import os, sys; os.environ['PYTHONDONTWRITEBYTECODE']='1'; "
        f"sys.path[:0] = {[str(runtime), str(sim_repo)]!r}; "
        "gl=os.environ.setdefault('MUJOCO_GL', 'egl'); "
        "os.environ.setdefault('PYOPENGL_PLATFORM', gl); "
        "import g1_bridge._cuda_order; from robocasa.g1_host import wbt; "
        "import zmq, msgpack_numpy; print('robocasa+g1_bridge+zmq ok')",
        setup=str(setup) if setup else None)
    support.checks.append((f"conda env '{conda_env}'", ok_env, detail or "import failed"))
    support.ok = all(ok for _, ok, _ in support.checks)
    return support


def _port_in_use(port: int, host: str = "localhost") -> bool:
    import socket
    with socket.socket() as connection:
        connection.settimeout(0.25)
        return connection.connect_ex((host, port)) == 0


def _wait_port(port: int, host: str = "localhost", timeout: float = 900, *, process=None) -> bool:
    """Wait for a new server, stopping immediately if its process exits."""
    import time
    timeout = float(os.environ.get("LOCOMANILAB_SERVER_WAIT_SECS") or os.environ.get("SOMA_SERVER_WAIT_SECS") or timeout)
    if timeout <= 0:
        raise ValueError("LOCOMANILAB_SERVER_WAIT_SECS (legacy SOMA_SERVER_WAIT_SECS) must be positive")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        if _port_in_use(port, host):
            # A process that failed to bind can exit just after the connect attempt.
            if process is not None:
                time.sleep(0.05)
            return process is None or process.poll() is None
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    return False


def orchestrate(*, support: "SimSupport", server_argv: list, server_cwd, runner, port: int,
                client_env_name: str, env_name: str, prompt: str, episodes: int, max_steps: int,
                action_horizon: int, render: str, video_dir,
                backend: str = "sonic",
                conda_setup: str | None, dry_run: bool,
                extra_client_args: list[str] | None = None):
    """Start a policy server (bg) -> wait for port -> run the sim eval client (blocking) -> teardown.
    server_argv is the fully env-wrapped command to Popen; the client runs in the sim conda env.
    extra_client_args are appended verbatim to the client command (e.g. --inference-rate, --scene-id)."""
    import subprocess
    from locomanilab.base import StageResult

    extra_client_args = extra_client_args or []
    client, extra_env = build_client_cmd(
        sim_repo=support.sim_repo, env_name=client_env_name, host="localhost", port=port,
        model=env_name, prompt=prompt, episodes=episodes, max_steps=max_steps,
        action_horizon=action_horizon, render=render, video_dir=video_dir,
        backend=backend, extra=extra_client_args, runtime_repo=support.runtime_repo)

    if dry_run:
        print(f"[locomanilab:eval] would start server:\n  cd {server_cwd}\n  {' '.join(map(str, server_argv))}")
        print(f"[locomanilab:eval] then sim client (conda {client_env_name}):\n  cd {support.sim_repo}\n  {' '.join(client)}")
        return StageResult(ok=True, message="dry-run")

    # start_new_session puts the server in its own process group so teardown can signal the
    # whole tree. The server is launched via `bash -lc '... exec python ...'`, so without
    # this a stray descendant can survive terminate() and linger as a zombie the parent
    # never reaps -- which hangs whatever is waiting on this eval even though every episode
    # already finished.
    if _port_in_use(port):
        return StageResult(ok=False, message=f"port :{port} is already in use; choose --port or stop your own service")
    try:
        proc = subprocess.Popen([str(a) for a in server_argv], cwd=str(server_cwd),
                                start_new_session=True)
    except OSError as exc:
        return StageResult(ok=False, message=f"could not start policy server: {exc}")
    from locomanilab.processes import managed_process
    with managed_process(proc):
        if not _wait_port(port, process=proc):
            detail = f" (exit {proc.returncode})" if proc.poll() is not None else " (startup timeout)"
            return StageResult(ok=False, message=f"policy server did not open :{port}{detail}")
        return runner.run(client, cwd=support.sim_repo, env_kind="conda", env_name=client_env_name,
                          setup=conda_setup, extra_env=extra_env or None, stage="eval")



def _shutdown_server(proc) -> None:
    """Stop the entire process group created for this server, including workers."""
    from locomanilab.processes import stop_process_group
    stop_process_group(proc)


def client_extra_args(inference_rate: float | None, scene_id: int | None, *,
                      robot: str | None = None, results_json: str | None = None,
                      genuine_min_steps: int | None = None,
                      token_ema_alpha: float | None = None,
                      inference_backend: str | None = None,
                      fixed_scene: bool = False,
                      scene_seed: int | None = None,
                      policy_config: str | None = None,
                      reset_jitter: float | None = None,
                      reset_object_jitter: float | None = None,
                      jitter_seed: int | None = None,
                      randomize: str | None = None,
                      randomize_seed: int | None = None,
                      render_scale: int | None = None) -> list[str]:
    """Build the optional sim-eval client args shared by every model's eval().

    inference_rate: None keeps the client default (2.5Hz PyTorch); pass a value to override.
    scene_id: pin a specific iTHOR FloorPlan number -- **the scene the data was collected in**.
        The dataset-name suffix IS the FloorPlan (IThorBathroomApproachToilet_401 -> 401).
        Leaving this unset randomises the scene, i.e. tests the policy in rooms it never saw:
        success rates come out far lower and are NOT comparable to same-scene numbers.
    robot: robosuite robot class; None keeps the client default (G1FloatingBody).
    results_json: write the per-episode table here, for batch sweeps.
    genuine_min_steps: report successes shorter than this separately -- they usually mean
        the episode spawned already in the goal state rather than doing the task.
    token_ema_alpha: EMA on the first action of each new chunk (1.0 = snap).
    policy_config: an explicit sim2real policy.yaml for the low-level controller,
        overriding --backend's default checkpoint. This is how you reach a
        non-default SONIC variant, e.g.
        ``external/g1_backend/checkpoints/sonic/v1_1/g1/policy.yaml``. **A policy's
        motion_token is only meaningful to the decoder it was trained against**,
        so pass the same variant the dataset was collected through, or the run
        scores zero with nothing in the log to say why.

    Two flags that used to be forwarded from here are not:

    * ``--sonic-ckpt``: the CLI's `lab eval --sonic-ckpt <name>` still exists, but
      it is resolved in cli.py (`_resolve_sonic_ckpt`) into the absolute
      ``policy_config`` above -- after checking that manifest and its ONNX weights
      are really in the g1_backend tree -- so this function only ever sees the
      path. The earlier incarnation was a free-floating old/new alias that could
      name an export the tree did not hold, and defaulted to the wrong one; do not
      bring that back as a client flag.
    * ``--warmup-secs``: a fixed standing token depends on the controller export.
      The current client starts the policy directly and has no warmup stage.
    """
    extra: list[str] = []
    if inference_rate is not None:
        extra += ["--inference-rate", str(inference_rate)]
    if scene_id is not None:
        extra += ["--scene-id", str(scene_id)]
    for flag, value in (("--robot", robot), ("--results-json", results_json),
                        ("--genuine-min-steps", genuine_min_steps),
                        ("--ema-alpha", token_ema_alpha),
                        ("--inference-backend", inference_backend),
                        ("--scene-seed", scene_seed),
                        ("--policy-config", policy_config),
                        ("--reset-jitter", reset_jitter),
                        ("--jitter-seed", jitter_seed),
                        ("--randomize", randomize),
                        ("--randomize-seed",
                         randomize_seed if randomize is not None else None),
                        ("--reset-object-jitter", reset_object_jitter),
                        ("--render-scale", render_scale)):
        if value is not None:
            extra += [flag, str(value)]
    if fixed_scene:
        extra.append("--fixed-scene")
    return extra


def client_path() -> Path:
    """locomanilab's own eval client (locomanilab/runtime/sim_client.py).

    It lives here rather than in the benchmark submodule because what it
    controls -- which scene, how many episodes, what counts as a success -- is
    the experiment, not the simulator. The benchmark's own scripts additionally
    cannot pin an iTHOR FloorPlan, which silently randomises the scene and
    makes success rates incomparable between runs.
    """
    from locomanilab.runtime import script_path
    return script_path("sim_client.py")


def runtime_checkout_error(runtime_repo: Path) -> str | None:
    """Reject incomplete checkouts that would fall back to an installed copy."""
    missing = [runtime_repo / package / "__init__.py" for package in ("g1_bridge", "sim2real")
               if not (runtime_repo / package / "__init__.py").is_file()]
    if missing:
        return (f"configured g1_backend checkout is incomplete: missing {', '.join(map(str, missing))}; "
                "initialize the checkout or correct repos.g1_backend in env.local.yaml")
    return None


def runtime_pythonpath(*, sim_repo: Path, runtime_repo: Path | None = None) -> str:
    """Use the configured simulator/controller checkouts before installed copies."""
    roots = [str(path) for path in (runtime_repo, sim_repo) if path is not None]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    return os.pathsep.join(roots)


def build_client_cmd(*, sim_repo: Path, env_name: str, host: str, port: int,
                     model: str, prompt: str, episodes: int, max_steps: int,
                     action_horizon: int, render: str, video_dir: str | None,
                     backend: str = "sonic",
                     extra: list[str] | None = None,
                     policy: str = "server",
                     runtime_repo: Path | None = None,
                     ) -> tuple[list[str], dict]:
    """Build the sim eval client command (runs in the sim conda env).

    `backend` picks the whole-body controller (sonic/amo/heft/scalebfm); the
    action space follows from it. (The old `pose` flag -- sonic's reference-motion
    alias -- was accepted and never used; it is gone.)"""
    cmd = ["python", "-u", str(client_path()),
           "--policy", policy, "--host", host, "--port", str(port),
           "--backend", backend,
           "--env", model, "--prompt", prompt,
           "--episodes", str(episodes), "--max-steps", str(max_steps),
           "--action-horizon", str(action_horizon)]
    if render == "video":
        cmd += ["--render", "video"]
        if video_dir:
            cmd += ["--video-dir", video_dir]
    else:
        cmd += ["--render", "viewer"]
    cmd += list(extra or [])
    extra_env = {"MUJOCO_GL": os.environ.get("LOCOMANILAB_MUJOCO_GL") or os.environ.get("SOMA_MUJOCO_GL", "egl")} if render == "video" else {}
    if "MUJOCO_GL" in extra_env:
        extra_env["PYOPENGL_PLATFORM"] = extra_env["MUJOCO_GL"]
    # An absolute client script does not put its working directory on sys.path.
    # Select configured checkouts before any editable installs in the environment.
    extra_env["PYTHONPATH"] = runtime_pythonpath(sim_repo=sim_repo, runtime_repo=runtime_repo)
    return cmd, extra_env
