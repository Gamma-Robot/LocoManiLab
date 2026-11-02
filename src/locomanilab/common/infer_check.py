"""Offline inference-check orchestration.

Start a model's gr00t-compat policy server, run the model-agnostic offline client
(locomanilab/runtime/infer_client.py) against a training dataset, tear the server down.
The client plots predicted vs ground-truth actions. Mirrors sim.orchestrate but the
"client" is the dataset-driven infer_client instead of the sim rollout.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from locomanilab.base import StageResult
from locomanilab.common.sim import _port_in_use, _wait_port

from locomanilab.runtime import script_path

CLIENT = script_path("infer_client.py")


def orchestrate(*, server_argv, server_cwd, runner, port: int, client_env_name: str,
                conda_setup, dataset, episode: int, frames, out, prompt: str, dry_run: bool,
                out_root: Path | None = None):
    # The client runs with cwd=<dataset>, so a relative --out (the default is
    # `runs/infer_check/<model>_ep<N>.png`) used to be written INSIDE the dataset dir --
    # and, since that subdirectory does not exist there, the run died at the very last
    # line with `FileNotFoundError: runs/infer_check/..._pred.npy`, after every inference
    # had already been paid for. Anchor it at the caller's root and create it up front.
    out = Path(out).expanduser()
    if not out.is_absolute():
        out = (out_root or Path.cwd()) / out
    client = ["python", str(CLIENT), "--dataset", str(dataset), "--host", "localhost",
              "--port", str(port), "--episode", str(episode), "--out", str(out), "--prompt", prompt]
    if frames is not None:
        client += ["--frames", str(frames)]

    if dry_run:
        print(f"[locomanilab:infer-check] would start server:\n  cd {server_cwd}\n  {' '.join(map(str, server_argv))}")
        print(f"[locomanilab:infer-check] then offline client (conda {client_env_name}):\n  {' '.join(client)}")
        return StageResult(ok=True, message="dry-run")

    # Own process group, same as the sim harness: the server is `bash -lc '... exec
    # python ...'`, and without this a descendant could outlive terminate() as a
    # zombie the parent never reaps.
    if _port_in_use(port):
        return StageResult(ok=False, message=f"port :{port} is already in use; choose --port or stop your own service")
    out.parent.mkdir(parents=True, exist_ok=True)
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
        res = runner.run(client, cwd=Path(dataset), env_kind="conda", env_name=client_env_name,
                         setup=conda_setup, stage="infer-check")
        if res.ok:
            res.artifacts["plot"] = Path(out)
        return res

