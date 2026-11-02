"""Runner: execute a command *locally* in the correct environment.

LocoManiLab never imports model code (envs conflict). Instead it shells out. The runner
wraps a command in the right activation:
  - conda: source <conda_root>/etc/profile.d/conda.sh && conda activate <name>
  - venv:  source <venv_path>/bin/activate
  - uv:    uv run <cmd>            (cwd must be the uv project)
  - none:  run as-is

Every run prints the exact command first (transparency); with dry_run it only
prints. Output is streamed live to the workspace's runs/<model>/<stage>/ directory
and echoed to the console.
Model-independent stages pass `label=` to file themselves under something truthful
instead of the model whose plugin they borrowed (see `run`).

`wrap()` is public on purpose: the sim-eval / infer-check paths Popen a policy
server themselves instead of going through `run()`, and every plugin used to
hand-roll its own `bash -lc "source … && conda activate … && exec …"` for that.
Ten copies in four spellings, and none of them carried the PATH fix below —
the one thing that makes `conda activate` inside a wrapper actually select the
env's python when the caller's shell already has some other venv first on PATH.
"""
from __future__ import annotations

import os
import json
import shlex
import subprocess
import time
from collections import deque
from pathlib import Path

from locomanilab.plan import ExecutionPlan
from locomanilab.processes import managed_process

from locomanilab.base import StageResult


def export_line(key: str, value: str) -> str:
    """`export KEY=<quoted value>` for `wrap(pre=[...])`.

    For values that must stay shell expressions (``${HF_HUB_CACHE:-…}``,
    ``${PYTHONPATH:+:…}``), write the fragment by hand instead.
    """
    return f"export {key}={shlex.quote(str(value))}"


def default_conda_roots() -> list[Path]:
    """Conda installations to try when `conda_root` is not configured.

    The active installation (``$CONDA_EXE`` is ``<root>/bin/conda`` once conda is
    initialized) comes first, then the usual per-user install locations.
    """
    roots = []
    if os.environ.get("CONDA_EXE"):
        roots.append(Path(os.environ["CONDA_EXE"]).parent.parent)
    roots += [Path.home() / name for name in ("anaconda3", "miniconda3", "miniforge3", "mambaforge")]
    return roots


class Runner:
    def __init__(self, *, conda_root: str | None = None, log_root: Path | None = None,
                 dry_run: bool = False, model: str = "locomanilab"):
        self.conda_root = conda_root
        self.log_root = Path(log_root) if log_root else Path.cwd() / "runs"
        self.dry_run = dry_run
        self.model = model

    def wrap(self, cmd: list[str], *, env_kind: str = "none", env_name: str | None = None,
             venv_path: str | None = None, setup: str | None = None,
             cwd: Path | str | None = None, pre: list[str] | None = None) -> list[str]:
        """The argv that runs `cmd` inside the requested environment.

        Order inside the `bash -lc` script, when one is needed::

            source <setup>      # optional local initialization (paths.*_setup)
            cd <cwd>            # only when `cwd` is given -- uv needs to be in its project
            <pre...>            # caller's raw fragments: exports, PYTHONPATH tweaks
            <activation>        # conda.sh + conda activate + PATH fix | venv activate | (uv: none)
            exec <cmd>          # `exec uv run --no-sync <cmd>` for uv

        `pre` fragments go BEFORE activation so a setup script cannot undo them and
        so `export PYTHONPATH=<repo>${PYTHONPATH:+:${PYTHONPATH}}` prepends onto
        whatever the setup script left. Returns a plain argv (no bash) when nothing
        has to be wrapped.
        """
        inner = shlex.join([str(c) for c in cmd])
        prefixes: list[str] = []
        if setup:
            prefixes.append(f"source {shlex.quote(setup)}")
        if cwd is not None:
            prefixes.append(f"cd {shlex.quote(str(cwd))}")
        prefixes += list(pre or [])
        if env_kind == "conda":
            if not env_name:
                raise ValueError("env_kind=conda requires env_name")
            # A configured setup script loads conda; otherwise use conda_root.
            if not setup:
                prefixes.append(f"source {shlex.quote(self._find_conda_sh())}")
            prefixes.append(f"conda activate {shlex.quote(env_name)}")
            # a virtualenv inherited from the caller's shell (e.g. the locomanilab CLI .venv)
            # can sit ahead of the conda env on PATH — `conda activate` swaps the
            # previously-active env's segment in place instead of prepending, so
            # `python` would still resolve to the venv. Force the env bin to the front.
            prefixes.append('export PATH="$CONDA_PREFIX/bin:$PATH"')
            exec_cmd = f"exec {inner}"
        elif env_kind == "venv":
            if not venv_path:
                raise ValueError("env_kind=venv requires venv_path")
            prefixes.append(f"source {shlex.quote(str(Path(venv_path) / 'bin' / 'activate'))}")
            exec_cmd = f"exec {inner}"
        elif env_kind == "uv":
            # The backend environment is provisioned separately. Running a command
            # must not unexpectedly sync dependencies or rebuild native extensions.
            exec_cmd = f"exec uv run --no-sync {inner}"
        elif env_kind in ("none", None):
            exec_cmd = f"exec {inner}"
        else:
            raise ValueError(f"unknown env_kind: {env_kind}")

        if not prefixes:                            # no wrapping needed
            if env_kind == "uv":
                return ["uv", "run", "--no-sync", *cmd]
            if env_kind in ("none", None):
                return list(cmd)
        return ["bash", "-lc", " && ".join(prefixes + [exec_cmd])]

    # kept for callers that still spell it the old way
    _wrap = wrap

    def _find_conda_sh(self) -> str:
        candidates = []
        if self.conda_root:
            candidates.append(Path(self.conda_root) / "etc/profile.d/conda.sh")
        candidates += [root / "etc/profile.d/conda.sh" for root in default_conda_roots()]
        for c in candidates:
            if c.exists():
                return str(c)
        # fall back to first candidate; error surfaces at run time
        return str(candidates[0]) if candidates else "conda.sh"

    def run(self, cmd: list[str], *, cwd: Path | str, env_kind: str = "none",
            env_name: str | None = None, venv_path: str | None = None,
            setup: str | None = None,
            extra_env: dict[str, str] | None = None, stage: str = "run",
            label: str | None = None) -> StageResult:
        """Shell out, printing the command and streaming output to one log file.

        ``label`` overrides the model name in the prefix and the log path. It
        exists for stages whose work is model-independent: collection is driven
        entirely by g1_bridge / humanoid_benchmark and only borrows a plugin
        to reach this runner, so filing it under whichever model happened to be
        the CLI default said something untrue about the run.
        """
        cwd = Path(cwd)
        wrapped = self.wrap(cmd, env_kind=env_kind, env_name=env_name,
                            venv_path=venv_path, setup=setup)
        who = label or self.model
        plan = ExecutionPlan(argv=wrapped, cwd=cwd, stage=stage, model=who,
                             environment=extra_env or {})
        pretty = plan.display()
        print(f"\n[locomanilab:{who}:{stage}] would run:" if self.dry_run
              else f"\n[locomanilab:{who}:{stage}] running:")
        print(pretty)

        if self.dry_run:
            return StageResult(ok=True, command=pretty, message="dry-run")

        if not cwd.exists():
            return StageResult(ok=False, command=pretty,
                               message=f"cwd does not exist: {cwd}")

        # one canonical log per run, under runs/<who>/<stage>/, with a self-identifying
        # filename <who>_<stage>_<ts>.log so it's distinguishable from other runs'
        # logs even when copied out of its directory. <who> is the model, or the
        # `label` override for model-independent stages. (No external tee — this is
        # the single copy.)
        log_dir = self.log_root / who / stage
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{who}_{stage}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{time.time_ns()}.log"

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")   # child python flushes stdout live
        if extra_env:
            env.update({k: str(v) for k, v in extra_env.items()})

        # stream output live: write each line to the log file (flushed) and echo to console,
        # so long-running training logs land in locomanilab/runs/... in real time (tail -f works).
        captured: deque[str] = deque(maxlen=2000)
        log_path.with_suffix(".json").write_text(json.dumps(plan.as_dict(), indent=2) + "\n")
        with open(log_path, "w", buffering=1) as lf:
            lf.write(pretty + "\n\n")
            lf.flush()
            try:
                proc = subprocess.Popen(wrapped, cwd=str(cwd), env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, bufsize=1, start_new_session=True)
            except OSError as exc:
                lf.write(str(exc) + "\n")
                return StageResult(ok=False, command=pretty, log_path=log_path,
                                   message=f"could not start command: {exc}")
            with managed_process(proc):
                try:
                    for line in proc.stdout:
                        lf.write(line)
                        print(line, end="", flush=True)
                        captured.append(line)
                    proc.wait()
                except KeyboardInterrupt:
                    return StageResult(ok=False, command=pretty, log_path=log_path,
                                       message="interrupted; the child process was stopped", output="".join(captured))
                finally:
                    if proc.stdout:
                        proc.stdout.close()
        ok = proc.returncode == 0
        return StageResult(ok=ok, command=pretty, log_path=log_path,
                           message="ok" if ok else f"exit code {proc.returncode}",
                           output="".join(captured))

    @staticmethod
    def _pretty(wrapped: list[str], cwd: Path, extra_env: dict[str, str] | None) -> str:
        return ExecutionPlan(argv=wrapped, cwd=cwd, environment=extra_env or {}).display()

    @staticmethod
    def _stop(proc: subprocess.Popen) -> None:
        from locomanilab.processes import stop_process_group
        stop_process_group(proc)
