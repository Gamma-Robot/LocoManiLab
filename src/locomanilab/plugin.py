"""LocoManiLab plugin interface: the abstract contract every model adapter implements.

A plugin is a *thin* wrapper: its job is to translate locomanilab's uniform stage calls
(convert / train / deploy) into the model repo's own native scripts, then run them
locally in the right environment via the injected Runner. No model business logic
lives here.
"""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from locomanilab.config import Context, ModelManifest
    from locomanilab.runner import Runner


@dataclass
class StageResult:
    """Outcome of one pipeline stage."""

    ok: bool
    command: str | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)
    log_path: Path | None = None
    message: str = ""
    output: str = ""          # captured stdout (for stages that need to parse it)


# name -> plugin class, populated by the @register decorator.
REGISTRY: dict[str, type["ModelPlugin"]] = {}


def register(name: str):
    """Class decorator registering a plugin under a model name."""

    def _wrap(cls: type["ModelPlugin"]):
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return _wrap


@dataclass
class SimServer:
    """What a plugin's `_sim_server()` hook hands the shared eval / infer-check harness.

    argv            fully env-wrapped command (see `ModelPlugin.server_argv`), Popen'd as-is
    cwd             directory to Popen it from
    prompt          what the CLIENT should send as the task prompt (None -> "test")
    inference_rate  client-rate override the server implies (gr00t --trt -> 10 Hz);
                    an explicit --inference-rate from the user still wins
    """

    argv: list[str]
    cwd: Path
    prompt: str | None = None
    inference_rate: float | None = None


class ModelPlugin:
    """Base class for a model adapter.

    Subclasses implement convert/train/deploy and, for the sim harness, the single
    `_sim_server()` hook that both `eval` and `infer_check` below are built on.
    Collection, preparation and replay are independent workflow services.
    Compatibility methods below delegate to those services.
    """

    name: str = "<unnamed>"

    #: CLI options that only some models implement. The CLI exposes all of them to
    #: every model; a plugin that does not name one in the signature of the method
    #: handling the stage (or of `_sim_server` for eval / infer-check) refuses it
    #: instead of letting `**opts` swallow it -- `lab eval -M psi0 --ckpt-step 15000`
    #: used to evaluate the newest checkpoint and say nothing.
    MODEL_OPTS = ("model_path", "run_dir", "ckpt_step", "base_model",
                  "output_dir", "trt", "trt_engine", "embodiment_tag", "t5_mode", "config_name",
                  "action_space", "per_gpu_batch", "server_only",
                  "master_port")

    def __init__(self, manifest: "ModelManifest", ctx: "Context", runner: "Runner"):
        self.manifest = manifest
        self.ctx = ctx
        self.runner = runner

    # ---- helpers -------------------------------------------------------
    def repo(self, key: str) -> Path:
        """Resolve a repo path for this machine from env.local.yaml."""
        return self.ctx.repo_path(key)

    def env_setup(self, kind: str) -> str | None:
        """Optional machine-local shell initialization for an environment manager."""
        key = {"uv": "uv_setup", "conda": "conda_setup"}.get(kind)
        path = self.ctx.path(key) if key else None
        return str(path) if path else None

    def server_argv(self, cmd: list[str], *, env_kind: str | None = None,
                    env_name: str | None = None, venv_path: str | None = None,
                    cwd: Path | str | None = None, pre: list[str] | None = None) -> list[str]:
        """Env-wrapped argv for a policy server the sim harness Popen's directly.

        Same wrapper `runner.run` uses (so the conda PATH fix and the uv
        `--no-sync` apply here too), with the isolation script for `env_kind`
        sourced first. `env_kind=None` means the manifest's `deploy` env.
        `pre` are raw shell fragments run before activation (exports); `cwd`
        adds a `cd` inside the script for servers that must start in their repo.
        """
        if env_kind is None:
            spec = self.manifest.env("deploy")
            env_kind, env_name, venv_path = spec.kind, env_name or spec.name, venv_path or spec.venv_path
        return self.runner.wrap(cmd, env_kind=env_kind, env_name=env_name, venv_path=venv_path,
                                setup=self.env_setup(env_kind), cwd=cwd, pre=pre)

    def resolve_gpus(self, gpus: int, gpu_ids: str | None,
                     extra_env: dict[str, str]) -> int:
        """Honour --gpu-ids: pin CUDA_VISIBLE_DEVICES and return the process count.

        `--gpus` left at its default of 1 with several ids given means "one per id";
        an explicit --gpus wins. Every launcher (torchrun / accelerate / NGPU=) reads
        the count from what this returns, so the two flags cannot disagree.
        """
        if gpu_ids:
            extra_env["CUDA_VISIBLE_DEVICES"] = gpu_ids
            n = len([x for x in gpu_ids.split(",") if x != ""])
            if gpus <= 1 and n > 1:
                gpus = n
        return gpus

    def resolve_ckpt(self, *candidates: str | Path | None, env_key: str | None = None,
                     stage: str, hint: str) -> Path | StageResult:
        """The checkpoint a stage should use: first explicit candidate, else env.local
        `paths.<env_key>`; absolute, symlinks kept (see psi0's `_dataset_path` for why).

        Returns a StageResult (ok=False) when nothing names one, or -- outside
        dry-run -- when the path is not on disk. The second check is what turns
        "policy server did not open :5550" after a 900 s wait into the one-line
        truth: the server was handed a path that does not exist. `hint` is the
        human spelling of where the checkpoint may come from.
        """
        chosen = next((c for c in candidates if c), None)
        if chosen is None and env_key:
            p = self.ctx.path(env_key)
            chosen = str(p) if p is not None else None
        if chosen is None:
            return StageResult(ok=False, message=f"{self.name} {stage} needs {hint}")
        path = self.ctx.input_path(chosen)
        if not self.runner.dry_run and not path.exists():
            return StageResult(ok=False, message=(
                f"{self.name} {stage}: checkpoint path does not exist: {path} "
                f"(given via {hint})"))
        return path

    def raise_nofile(self, want: int = 65536, model: str = "locomanilab") -> None:
        """Raise the inherited file-descriptor limit for worker-based dataloaders.

        Tensor sharing between workers can exhaust a low descriptor limit. The
        adjustment is best effort and is skipped while building a dry-run plan.
        """
        if self.runner.dry_run:
            return
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft < want:
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(want, hard), hard))
        except Exception as e:  # noqa: BLE001  — best effort, never fatal
            print(f"[locomanilab:{model}] could not raise RLIMIT_NOFILE: {e}")

    @staticmethod
    def drop_expandable_segments(extra_env: dict[str, str], model: str) -> None:
        """Exclude inherited expandable_segments for backend allocator compatibility.

        Adapters opt into this existing compatibility setting. Other allocator
        options are preserved in the child environment.
        """
        alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in alloc:
            return
        extra_env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(
            p for p in alloc.split(",") if "expandable_segments" not in p)
        print(f"[locomanilab:{model}] excluding inherited expandable_segments "
              "from PYTORCH_CUDA_ALLOC_CONF for backend compatibility")

    def unsupported_opts(self, method, opts: dict[str, Any]) -> list[str]:
        """Which MODEL_OPTS were given a value but are not parameters of `method`."""
        params = inspect.signature(method).parameters
        return [k for k in self.MODEL_OPTS
                if k in opts and opts[k] not in (None, False) and k not in params]

    def refuse_unsupported(self, stage: str, method, opts: dict[str, Any]) -> StageResult | None:
        """StageResult(ok=False) naming the flags `method` would have ignored, or None."""
        bad = self.unsupported_opts(method, opts)
        if not bad:
            return None
        flags = ", ".join("--" + k.replace("_", "-") for k in bad)
        return StageResult(ok=False, message=(
            f"{self.name} {stage} does not use {flags} -- refused rather than ignored, so a "
            f"flag meant for another model cannot silently run the wrong thing here. "
            f"Drop it, or check `locomanilab {stage} --help` for which model each option is for."))

    def _not_implemented(self, stage: str) -> StageResult:
        return StageResult(
            ok=False,
            message=f"{self.name}: stage '{stage}' not implemented for this plugin",
        )

    # ---- stages (override in subclasses) -------------------------------
    def convert(self, *, dataset: Path, **opts: Any) -> StageResult:
        return self._not_implemented("convert")

    def train(self, *, dataset: Path, gpus: int, **opts: Any) -> StageResult:
        return self._not_implemented("train")

    def deploy(self, *, prompt: str | None = None, **opts: Any) -> StageResult:
        return self._not_implemented("deploy")

    # ---- sim harness: one hook, two stages -----------------------------
    def _sim_server(self, *, stage: str, prompt: str | None, port: int,
                    **opts: Any) -> "SimServer | StageResult":
        """Resolve the checkpoint and build this model's gr00t-compat policy server.

        The ONE thing a plugin has to provide for `lab eval --sim` and
        `lab infer-check`: both stages start the same server and differ only in
        the client they run against it. `stage` is "eval" / "infer-check", for
        error messages. Return a StageResult(ok=False) to refuse.
        """
        return self._not_implemented(stage)

    def eval(self, *, prompt: str | None = None, port: int = 5550,
             env_name: str = "TabletopPickPlace", episodes: int = 1, max_steps: int = 60,
             action_horizon: int = 64, render: str = "video", video_dir: str | None = None,
             sim_env: str = "tele_benchmark", inference_rate: float | None = None,
             scene_id: int | None = None, backend: str = "sonic",
             **opts: Any) -> StageResult:
        """Run the shared closed-loop evaluation workflow."""
        from locomanilab.workflows.evaluation import evaluate
        return evaluate(self, prompt=prompt, port=port, env_name=env_name, episodes=episodes,
                        max_steps=max_steps, action_horizon=action_horizon, render=render,
                        video_dir=video_dir, sim_env=sim_env, inference_rate=inference_rate,
                        scene_id=scene_id, backend=backend, **opts)

    def infer_check(self, *, dataset: Path, episode: int = 0, frames: int | None = None,
                    out: str | None = None, prompt: str | None = None, port: int = 5550,
                    sim_env: str = "tele_benchmark", **opts: Any) -> StageResult:
        """Run the shared offline policy check."""
        from locomanilab.workflows.evaluation import infer_check
        return infer_check(self, dataset=dataset, episode=episode, frames=frames, out=out,
                           prompt=prompt, port=port, sim_env=sim_env, **opts)

    # Backwards-compatible access to stages that do not depend on a model.
    def collect(self, **opts: Any) -> StageResult:
        from locomanilab.workflows import Workflows
        return Workflows(self.ctx, self.runner).collect(**opts)

    def prepare(self, **opts: Any) -> StageResult:
        from locomanilab.workflows import Workflows
        return Workflows(self.ctx, self.runner).prepare(**opts)

    def replay(self, **opts: Any) -> StageResult:
        from locomanilab.workflows import Workflows
        return Workflows(self.ctx, self.runner).replay(**opts)

    def gr00t_wbc_sonic_args(self, *args, **opts):
        from locomanilab.workflows.devices import DeviceWorkflow
        return DeviceWorkflow(self.ctx, self.runner).gr00t_wbc_sonic_args(*args, **opts)
