"""ACT / Diffusion Policy baseline plugins (lerobot v0.3.3 in external/lerobot).

One shared plugin class serves both models — they differ only in manifest values
(policy_type, batch size, policy_args). Stages map onto lerobot's own scripts,
all run via `uv run --no-sync` inside external/lerobot:

  convert -> locomanilab/runtime/lerobot/convert.py      (packaged script, run in the
             lerobot env; merges the 78-D action = motion_token(64)+hands(14)
             into one `action` column, symlinks videos, rebuilds episode stats)
  train   -> src/lerobot/scripts/train.py --policy.type=act|diffusion (draccus CLI)
  deploy  -> locomanilab/runtime/lerobot/server.py        (gr00t :5550 / get_action)
  eval / infer-check -> the same server + the shared sim / offline client

Both scripts live in locomanilab rather than in the model repo — the exception to the
rule the other six plugins follow — because external/lerobot is unforked
upstream and its working tree stays pristine. They still run in its uv env.

Neither baseline is language-conditioned: prompts are accepted for protocol
parity and ignored (the server warns once). The converted dataset is SHARED by
act and dp — convert with either model name once.

Machine specifics from env.local.yaml: repos.lerobot (default external/lerobot),
paths.lerobot_data_root, paths.lerobot_outputs. `--set k=v` appends/overrides
draccus args by dotted name (e.g. --set steps=500 --set policy.chunk_size=64).
"""
from __future__ import annotations

import os
import json

from pathlib import Path
from typing import Any

from locomanilab.plugin import ModelPlugin, SimServer, StageResult, register
from locomanilab.data.paths import dataset_identity


class LerobotBaselinePlugin(ModelPlugin):

    # ---- helpers -------------------------------------------------------
    def _data_root(self) -> Path | None:
        p = self.ctx.path("lerobot_data_root")
        return Path(p) if p is not None else None

    def _out_root(self) -> Path | None:
        p = self.ctx.path("lerobot_outputs")
        return Path(p) if p is not None else None

    @staticmethod
    def _dataset_name(dataset: Path) -> str:
        """Name the converted copy / run after the path the CALLER gave us.

        Datasets are often reached through a symlink whose name carries the variant
        (bench_sonic_X, bench_heft_X -> .../{sonic,heft}/X). Resolving first would
        collapse both to "X" and let two different action layouts (78-D vs 50-D)
        overwrite each other's conversion. Keep the caller's name, resolve only the
        path that gets passed to the tools.
        """
        return Path(dataset).expanduser().absolute().name

    def _converted_dir(self, dataset: Path) -> Path | None:
        root = self._data_root()
        return (root / self._dataset_name(dataset)) if root is not None else None

    @staticmethod
    def _source_matches(converted: Path, dataset: Path) -> bool:
        try:
            recorded = json.loads((converted / "meta/locomanilab_source.json").read_text())
            return recorded == dataset_identity(dataset)
        except (OSError, ValueError):
            return False

    def _runtime_script(self, rel: str) -> Path:
        from locomanilab.runtime import script_path
        packaged = {
            "scripts/convert_lerobot_baseline.py": "lerobot/convert.py",
            "scripts/run_lerobot_server.py": "lerobot/server.py",
        }
        if rel in packaged:
            return script_path(packaged[rel])
        if rel in ("lerobot/convert.py", "lerobot/server.py"):
            return script_path(rel)
        return self.ctx.anchor(rel)

    # ---- convert -------------------------------------------------------
    def convert(self, *, dataset: Path, **opts: Any) -> StageResult:
        dataset = self.ctx.input_path(dataset)
        name = self._dataset_name(dataset)
        root = self._data_root()
        if root is None:
            return StageResult(ok=False,
                               message="set paths.lerobot_data_root in env.local.yaml")
        out = root / name
        if not self.runner.dry_run and out.exists():
            if (out / "meta/episodes_stats.jsonl").is_file() and self._source_matches(out, dataset):
                return StageResult(ok=True, message=f"already converted: {out} (source verified; shared by act+dp)")
            return StageResult(ok=False, message=(
                f"{out} has no matching source provenance for {dataset}. Existing data was left unchanged. "
                "Choose a fresh paths.lerobot_data_root or a unique source directory name and run lab convert again."))
        script = self._runtime_script(self.manifest.convert.get(
            "script", "lerobot/convert.py"))
        env = self.manifest.env("convert")
        # Which source features get concatenated into the single `action` column that
        # ACT/DP consume. Default (manifest-less) = the 78-D SONIC layout; heft-collected
        # datasets carry ref_qpos(36)+hands(14) instead, declared per manifest.
        cmd = ["python", str(script), "--src", str(dataset), "--out", str(out)]
        if (parts := self.manifest.convert.get("action_parts")):
            cmd += ["--action-parts", str(parts)]
        r = self.runner.run(
            cmd,
            cwd=self.repo(self.manifest.repo_key), env_kind=env.kind, env_name=env.name,
            setup=self.env_setup(env.kind), stage="convert")
        if not r.ok:
            return r
        return StageResult(ok=True, message=f"OK — converted to {out} (action[78], videos symlinked)")

    # ---- train ---------------------------------------------------------
    def train(self, *, dataset: Path, gpus: int, gpu_ids: str | None = None,
              master_port: int | None = None, task: str | None = None,
              output_dir: str | None = None, overrides: list[str] | None = None,
              **opts: Any) -> StageResult:
        for override in overrides or []:
            key = override.lstrip("-").split("=", 1)[0]
            if key in {"dataset", "dataset.root", "dataset.repo_id"}:
                return StageResult(ok=False, message=(
                    f"--set {key} would bypass the dataset-bound conversion. "
                    "Select the source with --dataset and configure paths.lerobot_data_root instead."))
        dataset = self.ctx.input_path(dataset)
        name = self._dataset_name(dataset)
        conv = self._converted_dir(dataset)
        tr = self.manifest.train
        if conv is None:
            return StageResult(ok=False, message="set paths.lerobot_data_root in env.local.yaml")
        if not self.runner.dry_run and (not (conv / "meta/episodes_stats.jsonl").is_file()
                                        or not self._source_matches(conv, dataset)):
            return StageResult(
                ok=False,
                message=f"{conv} is missing or its source provenance does not match. "
                        f"Run `lab convert --model {self.name} --dataset {dataset}` first; "
                        "if an older output exists, choose a fresh paths.lerobot_data_root.")
        out_root = self._out_root() or self.ctx.workspace_root / "checkpoints"
        if out_root is None:
            return StageResult(ok=False,
                               message="set paths.lerobot_outputs in env.local.yaml or pass --output-dir")
        run_dir = (out_root / self.name / (task or name)) if not output_dir \
            else self.ctx.anchor(output_dir)
        # `--set resume=true` has to get PAST this guard, or the advice in its own
        # error message is a dead end: a run that died mid-training leaves the dir
        # behind, every retry is rejected here, and the cell can never finish
        # (measured 2026-08-08: dp/KitchenCloseDrawer stuck at 17600/20000 this way).
        resuming = str(dict(
            s.split("=", 1) for s in (overrides or []) if "=" in s
        ).get("resume", "")).lower() in ("true", "1", "yes")
        if not self.runner.dry_run and run_dir.exists() and not resuming:
            return StageResult(
                ok=False,
                message=f"{run_dir} already exists — lerobot refuses to overwrite; "
                        f"delete it, pass --output-dir, or resume with --set resume=true")
        if resuming and not (run_dir / "checkpoints" / "last").exists():
            return StageResult(
                ok=False,
                message=f"--set resume=true but {run_dir}/checkpoints/last does not exist; "
                        f"there is nothing to resume from")

        # draccus args as an ordered mapping so --set can REPLACE by dotted key.
        args: dict[str, Any] = {
            "dataset.repo_id": f"locomanilab/{name}",
            "dataset.root": str(conv),
            "dataset.video_backend": "pyav",
            "policy.type": tr.get("policy_type", "act"),
            "policy.device": "cuda",
            "policy.push_to_hub": "false",
            "output_dir": str(run_dir),
            "job_name": f"{self.name}_{name}",
            "steps": str(tr.get("steps", 50000)),
            "batch_size": str(tr.get("batch_size", 8)),
            "save_freq": str(tr.get("save_freq", 10000)),
            "num_workers": str(tr.get("num_workers", 4)),
            "wandb.enable": "false",
        }
        for k, v in (tr.get("policy_args") or {}).items():
            args[f"policy.{k}"] = str(v)
        for kv in (overrides or []):
            kv = kv.lstrip("-")
            if "=" in kv:
                k, v = kv.split("=", 1)
                args[k] = v
            else:
                args[kv] = "true"

        cmd = ["python", tr.get("entrypoint", "src/lerobot/scripts/train.py")]
        cmd += [f"--{k}={v}" for k, v in args.items()]

        extra_env: dict[str, str] = {"TOKENIZERS_PARALLELISM": "false",
                                     "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
                                     "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1")}
        # lerobot's train.py is a single process on one device: `--gpus 2` / two ids used
        # to be accepted and quietly run on the first visible card. Say so instead.
        if self.resolve_gpus(gpus, gpu_ids, extra_env) > 1:
            return StageResult(ok=False, message=(
                f"{self.name} trains on ONE GPU (lerobot train.py is single-process); "
                f"pass a single id in --gpu-ids and leave --gpus at 1"))
        env = self.manifest.env("train")
        return self.runner.run(cmd, cwd=self.repo(self.manifest.repo_key),
                               env_kind=env.kind, env_name=env.name,
                               setup=self.env_setup(env.kind),
                               extra_env=extra_env, stage="train")

    # ---- deploy --------------------------------------------------------
    def deploy(self, *, model_path: str | None = None, run_dir: str | None = None,
               prompt: str | None = None, port: int = 5550, **opts: Any) -> StageResult:
        ckpt = self.resolve_ckpt(model_path, run_dir, stage="deploy", hint=self.CKPT_HINT)
        if isinstance(ckpt, StageResult):
            return ckpt
        env = self.manifest.env("deploy")
        return self.runner.run(self._server_cmd(str(ckpt), port),
                               cwd=self.repo(self.manifest.repo_key),
                               env_kind=env.kind, env_name=env.name,
                               setup=self.env_setup(env.kind), stage="deploy")

    # ---- sim harness (eval + infer-check share this) --------------------
    CKPT_HINT = ("--model-path/--run-dir (lerobot train output dir, checkpoints/<step>, "
                 "or pretrained_model)")

    def _sim_server(self, *, stage: str, prompt: str | None, port: int,
                    model_path: str | None = None, run_dir: str | None = None,
                    **opts: Any) -> SimServer | StageResult:
        """gr00t-compat server for the 78-D act/dp checkpoints: they emit motion_token
        like gr00t/fastwam/prts, so the sonic backend drives them unchanged. The prompt
        is passed for protocol parity only (neither baseline is language-conditioned)."""
        ckpt = self.resolve_ckpt(model_path, run_dir, stage=stage, hint=self.CKPT_HINT)
        if isinstance(ckpt, StageResult):
            return ckpt
        return SimServer(argv=self._server_argv(str(ckpt), port),
                         cwd=self.repo(self.manifest.repo_key), prompt=prompt)

    # ---- server helpers ------------------------------------------------
    def _server_cmd(self, ckpt: str, port: int) -> list[str]:
        script = self._runtime_script(self.manifest.deploy.get(
            "module", "lerobot/server.py"))
        # Absolute: the server runs with cwd=<lerobot repo>, so a relative
        # checkpoint path typed at the locomanilab prompt would resolve against the
        # wrong directory -- silently, since it just looks like a missing ckpt.
        return ["python", str(script),
                "--ckpt", str(self.ctx.input_path(ckpt)),
                "--host", "0.0.0.0", "--port", str(port)]

    def _server_argv(self, ckpt: str, port: int) -> list[str]:
        """Launch the policy server, preferring the venv's python over `uv run`.

        `uv run` takes a GLOBAL lock on ~/.cache/uv/.lock, so parallel evals do not
        start in parallel: measured 2026-08-08 with 8 lanes, every server after the
        first sat in futex_wait_queue_me on that lock and came up ~12 minutes apart,
        which caps a sweep's throughput no matter how many GPUs are free. The venv
        interpreter needs no lock and no resolution step -- `--no-sync` already said
        we do not want either -- so use it directly when it exists.
        """
        repo = self.repo(self.manifest.repo_key)
        cmd = self._server_cmd(ckpt, port)
        venv_py = Path(repo) / ".venv" / "bin" / "python3"
        if venv_py.exists():
            return self.server_argv([str(venv_py)] + cmd[1:], env_kind="none", cwd=repo)
        return self.server_argv(cmd, env_kind="uv", cwd=repo)


@register("act")
class ACTPlugin(LerobotBaselinePlugin):
    pass


@register("dp")
class DPPlugin(LerobotBaselinePlugin):
    pass
