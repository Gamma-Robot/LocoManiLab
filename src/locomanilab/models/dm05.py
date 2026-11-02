"""OpenDM model adapter using locomanilab's packaged converter, experiment and server.

The upstream OpenDM checkout remains unmodified; only its native launcher and
installed environment are used for training and serving.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from locomanilab.plugin import ModelPlugin, SimServer, StageResult, register

from locomanilab.runtime import script_path

DM05_SCRIPTS = script_path("dm05")


@register("dm05")
class DM05Plugin(ModelPlugin):

    # ---- helpers -------------------------------------------------------
    def _runtime_file(self, configured: str | None, packaged: str) -> Path:
        legacy = {"scripts/dm05/soma_dm05_exp.py", "scripts/dm05/run_dm05_server.py"}
        if not configured or configured in legacy or configured == packaged:
            return script_path(packaged)
        return self.ctx.anchor(configured)

    def _dataset_name(self, dataset: Path) -> str:
        stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(dataset).name).strip("_")
        return f"{self.manifest.convert.get('dataset_prefix', 'locomanilab_')}{stem}"

    def _converted_dir(self, dataset: Path) -> Path | None:
        root = self.ctx.path("dm05_data_root")
        return (Path(root) / Path(dataset).name) if root is not None else None

    def _stage_env(self, stage: str) -> dict[str, str]:
        block = getattr(self.manifest, stage, None) or {}
        env = {k: str(v) for k, v in (block.get("env") or {}).items()}
        root = self.ctx.path("dm05_data_root")
        if root is not None:
            env["LOCOMANILAB_DM05_DATA_ROOT"] = str(root)
        env["OPENDM_DATA_PATH"] = str(DM05_SCRIPTS / "registry")
        return env

    # ---- convert -------------------------------------------------------
    def convert(self, *, dataset: Path, **opts: Any) -> StageResult:
        dataset = self.ctx.input_path(dataset)
        out = self._converted_dir(dataset)
        if out is None:
            return StageResult(ok=False, message="set paths.dm05_data_root in env.local.yaml")
        env = self.manifest.env("convert")
        r = self.runner.run(
            ["python", str(script_path("dm05/convert.py")), "--data-root", str(dataset),
             "--out-dir", str(out), "--name", self._dataset_name(dataset)],
            cwd=self.ctx.config_root, env_kind=env.kind, env_name=env.name,
            setup=self.env_setup(env.kind), stage="convert")
        if r.ok:
            r.artifacts["converted"] = out
            r.message = (r.message or "") + (
                f"\nregistered as {self._dataset_name(dataset)} ({out}/opendm_dataset.json)\n"
                f"Train:\n  lab train --model dm05 --dataset {dataset} --gpu-ids 0,1,2,3")
        return r

    # ---- train ---------------------------------------------------------
    def train(self, *, dataset: Path, gpus: int, gpu_ids: str | None = None,
              master_port: int | None = None, task: str | None = None,
              output_dir: str | None = None, overrides: list[str] | None = None,
              base_model: str | None = None, **opts: Any) -> StageResult:
        dataset = self.ctx.input_path(dataset)
        converted = self._converted_dir(dataset)
        record = converted / "opendm_dataset.json" if converted is not None else None
        if not self.runner.dry_run and (record is None or not record.exists()):
            return StageResult(ok=False, message=(
                f"{record} missing -- run `lab convert --model dm05 --dataset {dataset}` first"))
        name = self._dataset_name(dataset)
        if record is not None and record.exists():
            name = json.loads(record.read_text())["name"]

        pretrained = self.ctx.input_path(base_model) if base_model else self.ctx.path("dm05_pretrained")
        if pretrained is None:
            return StageResult(ok=False, message=(
                "set paths.dm05_pretrained in env.local.yaml (or --base-model) to the widened base:\n"
                f"  python {script_path('dm05/widen.py')} --src <DM05> --dst <DM05-sonic78> "
                "--action-dim 78"))
        if not self.runner.dry_run:
            cfg = Path(pretrained) / "config.json"
            width = json.loads(cfg.read_text()).get("action_dim") if cfg.exists() else None
            want = int(self.manifest.action.get("model_dim", 78))
            if width != want:
                return StageResult(ok=False, message=(
                    f"{pretrained} has action_dim {width}, need {want}: widen it first "
                    f"({script_path('dm05/widen.py')})"))

        out = self.ctx.anchor(output_dir) if output_dir else (
            self.ctx.path("dm05_outputs") or self.ctx.workspace_root / "checkpoints" / "dm05")
        if out is None:
            return StageResult(ok=False, message="set paths.dm05_outputs in env.local.yaml or pass --output-dir")
        out = Path(out)
        if output_dir is None or task:
            out = out / (task or name)

        if gpu_ids:
            n = len([x for x in gpu_ids.split(",") if x != ""])
            if gpus <= 1 and n > 1:
                gpus = n
        tr = self.manifest.train
        flags: dict[str, str] = {
            "--task": "train",
            "--data-config.dataset-name": name,
            "--model-config.model-name-or-path": str(pretrained),
            "--model-config.chunk-size": str(tr.get("chunk_size", 50)),
            "--model-config.vision-attn-implementation": str(tr.get("vision_attn_implementation", "sdpa")),
            "--trainer-config.output-dir": str(out),
            "--trainer-config.per-device-train-batch-size": str(tr.get("per_device_batch_size", 8)),
            "--trainer-config.num-train-steps": str(tr.get("max_steps", 50000)),
            "--trainer-config.save-steps": str(tr.get("save_steps", 10000)),
            "--trainer-config.save-total-limit": str(tr.get("save_total_limit", 3)),
        }
        for kv in (overrides or []):
            kv = kv.lstrip("-")
            if "=" not in kv:
                return StageResult(ok=False, message=f"--set expects k=v, got {kv!r}")
            key, value = kv.split("=", 1)
            flags[f"--{key}"] = value

        cmd = ["bash", tr.get("launcher", "script/dm05_launcher.sh"),
               "--exp", str(self._runtime_file(tr.get("exp"), "dm05/experiment.py")),
               "--nproc_per_node", str(gpus)]
        if master_port is not None:
            cmd += ["--master_port", str(master_port)]
        for k, v in flags.items():
            cmd += [k, v]

        extra_env = self._stage_env("train")
        if gpu_ids:
            extra_env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        env = self.manifest.env("train")
        r = self.runner.run(cmd, cwd=self.repo(self.manifest.repo_key), env_kind=env.kind,
                            env_name=env.name, setup=self.env_setup(env.kind),
                            extra_env=extra_env, stage="train")
        if r.ok:
            r.artifacts["run_dir"] = out
            r.message = (r.message or "") + f"\nrun dir: {out}"
        return r

    # ---- deploy / sim eval ---------------------------------------------
    def _resolve_ckpt(self, path: str | None, ckpt_step: int | None = None) -> Path | None:
        """A servable dir holds config.json + model weights: a checkpoint-N, or a run's final
        top-level save. A run dir plus --ckpt-step N picks <run>/checkpoint-N."""
        if path is None:
            p = self.ctx.path("dm05_run")
            path = str(p) if p is not None else None
        if path is None:
            return None
        d = self.ctx.input_path(path)
        if ckpt_step not in (None, "", "latest"):
            d = d / f"checkpoint-{int(ckpt_step)}"
        if (d / "config.json").exists() or self.runner.dry_run:
            return d
        return None

    def _server_cmd(self, ckpt: Path, port: int, prompt: str | None) -> list[str]:
        dep = self.manifest.deploy
        dataset_name = str(dep.get("dataset_name", "locomanilab_train50_merged"))
        if dataset_name == "locomanilab_train50_merged":
            root = self.ctx.path("dm05_data_root")
            record = root / "train50_merged" / "opendm_dataset.json" if root is not None else None
            if record is not None and record.is_file():
                dataset_name = str(json.loads(record.read_text())["name"])
        cmd = ["python", str(self._runtime_file(dep.get("module"), "dm05/server.py")),
               "--model-path", str(ckpt), "--host", "0.0.0.0", "--port", str(port),
               "--dataset-name", dataset_name,
               "--diffusion-steps", str(int(dep.get("diffusion_steps", 10)))]
        prompt = prompt or dep.get("prompt_fixed")
        if prompt:
            cmd += ["--initial-prompt", prompt]
        return cmd

    def deploy(self, *, model_path: str | None = None, run_dir: str | None = None,
               prompt: str | None = None, port: int = 5550,
               ckpt_step: int | None = None, **opts: Any) -> StageResult:
        ckpt = self._resolve_ckpt(model_path or run_dir, ckpt_step)
        if ckpt is None:
            return StageResult(ok=False, message="deploy needs --model-path (checkpoint-N dir or run dir "
                                                 "+ --ckpt-step)")
        env = self.manifest.env("deploy")
        extra_env = self._stage_env("deploy")
        return self.runner.run(self._server_cmd(ckpt, port, prompt), cwd=self.ctx.config_root,
                               env_kind=env.kind, env_name=env.name, setup=self.env_setup(env.kind),
                               extra_env=extra_env, stage="deploy")

    def _sim_server(self, *, stage: str, prompt: str | None, port: int,
                    model_path: str | None = None, run_dir: str | None = None,
                    ckpt_step: int | None = None, **opts: Any) -> SimServer | StageResult:
        from locomanilab.runner import export_line
        ckpt = self._resolve_ckpt(model_path or run_dir, ckpt_step)
        if ckpt is None:
            return StageResult(ok=False, message=f"{stage} needs --model-path (DM05 checkpoint dir)")
        env = self.manifest.env("deploy")
        pre = [export_line(key, value) for key, value in self._stage_env("deploy").items()]
        return SimServer(
            argv=self.server_argv(self._server_cmd(ckpt, port, prompt), env_kind=env.kind,
                                  env_name=env.name, venv_path=env.venv_path,
                                  cwd=self.ctx.config_root, pre=pre),
            cwd=self.ctx.config_root, prompt=prompt,
            inference_rate=self.manifest.deploy.get("inference_rate"))
