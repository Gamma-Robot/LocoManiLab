"""Xiaomi-Robotics-1 (XR-1) plugin.

Maps locomanilab's uniform stages onto XR-1's native scripts (all run in the `xr1` conda
env inside the fork at env.local `repos.xr1`):
  convert -> xr1/scripts/data/locomanilab_sonic_to_xr1_json.py  (LeRobot -> JSON episodes)
           + xr1/scripts/data/compute_xr1_stats.py       (-> configs/data/locomanilab_<ds>.yaml)
  train   -> RESOURCE_GPU=N bash xr1/scripts/train.sh <hydra dotlist>
  deploy  -> python deploy/run_xr1_server.py             (gr00t :5550 / get_action)

XR-1 is 60-D bimanual-EE-delta native; G1 SONIC is 78-D motion_token. The fork
carries the SONIC action space (`mibot/utils/sonic.py`), a SonicJsonDataset /
SonicDataModule, the widened `posttrain_sonic` model config and the checkpoint
surgery that makes the released 5B weights loadable at action_dim 80. See
locomanilab/configs/xr1.yaml for the full mismatch table and the hazard list.

Machine specifics come from env.local.yaml: repos.xr1 and
paths.xr1_data_root / xr1_outputs / xr1_pretrained.
`--set k=v` appends a raw hydra dotlist override (e.g. `--set trainer.max_steps=200`);
hydra takes the LAST occurrence of a key, so overrides always win.
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from locomanilab.plugin import ModelPlugin, SimServer, StageResult, register
from locomanilab.models._legacy import converter_path


@register("xr1")
class XR1Plugin(ModelPlugin):

    # ---- helpers -------------------------------------------------------
    def _xr1_dir(self) -> Path:
        """The `xr1/` package dir: training and the data scripts run from there."""
        return self.repo(self.manifest.repo_key) / self.manifest.train.get("subdir", "xr1")

    def _json_dir(self, dataset: Path) -> Path | None:
        root = self.ctx.path("xr1_data_root")
        return (Path(root) / Path(dataset).name) if root is not None else None

    def _config_name(self, dataset: Path) -> str:
        stem = re.sub(r"[^A-Za-z0-9_]+", "_", Path(dataset).name).strip("_")
        return f"{self.manifest.convert.get('config_prefix', 'locomanilab_')}{stem}"

    _prefix_len: int | None = None

    def _stage_env(self, stage: str) -> dict[str, str]:
        """Manifest env for a stage, with env.local paths.xr1_hf_home overriding HF_HOME.

        Both training and serving use the same user-provided Qwen3-VL cache.
        """
        block = getattr(self.manifest, stage, None) or {}
        env = {k: str(v) for k, v in (block.get("env") or {}).items()}
        hf_home = self.ctx.path("xr1_hf_home")
        if hf_home is not None:
            env["HF_HOME"] = str(hf_home)
        return env

    def _exp_name(self, task: str | None, dataset: Path) -> str:
        # hydra puts this straight into a directory name; keep it filesystem-safe.
        return re.sub(r"[^A-Za-z0-9_-]+", "-", task or Path(dataset).name).strip("-").lower()

    # ---- convert -------------------------------------------------------
    def convert(self, *, dataset: Path, **opts: Any) -> StageResult:
        """locomanilab-prepared GR00T LeRobot -> XR-1 JSON episodes + a hydra data config
        carrying the normalization stats (XR-1 keeps them in the config, not a file)."""
        dataset = self.ctx.input_path(dataset)
        cv = self.manifest.convert

        modality = dataset / "meta" / "modality.json"
        if not modality.exists():
            return StageResult(ok=False, message=f"{modality} not found — not a locomanilab-prepared dataset?")
        try:
            mod = json.loads(modality.read_text())
        except Exception as e:  # noqa: BLE001
            return StageResult(ok=False, message=f"cannot parse {modality}: {e}")
        missing = [f"{group}.{key}"
                   for group, keys in (cv.get("require_modality") or {}).items()
                   for key in keys
                   if key not in set((mod.get(group) or {}).keys())]
        if missing:
            return StageResult(
                ok=False,
                message=f"dataset modality.json is missing keys xr1 needs: {', '.join(missing)}")

        json_dir = self._json_dir(dataset)
        if json_dir is None:
            return StageResult(ok=False,
                               message="set paths.xr1_data_root in env.local.yaml "
                                       "(where the converted JSON episode dirs go)")
        cwd = self._xr1_dir()
        env = self.manifest.env("convert")
        name = self._config_name(dataset)

        r = self.runner.run(
            ["python", converter_path(cwd, cv.get("converter", "scripts/data/locomanilab_sonic_to_xr1_json.py")),
             "--data-root", str(dataset), "--out-dir", str(json_dir), "--verify-frames"],
            cwd=cwd, env_kind=env.kind, env_name=env.name,
            setup=self.env_setup(env.kind), stage="convert")
        if not r.ok:
            return r

        r2 = self.runner.run(
            ["python", cv.get("stats_script", "scripts/data/compute_xr1_stats.py"),
             "--json-dir", str(json_dir), "--name", name,
             "--action-length", str(cv.get("action_length", 30)),
             "--batch-size", str(cv.get("batch_size", 4))],
            cwd=cwd, env_kind=env.kind, env_name=env.name,
            setup=self.env_setup(env.kind), stage="convert")
        if not r2.ok:
            return r2

        generated = cwd / "configs" / "data" / f"{name}.yaml"
        if not self.runner.dry_run and not generated.exists():
            return StageResult(ok=False, message=f"{generated} missing after compute_xr1_stats")
        return StageResult(
            ok=True,
            artifacts={"json_dir": json_dir, "data_config": generated},
            message=(f"OK — {json_dir} + {generated} (action 80 / state 60 / chunk "
                     f"{cv.get('action_length', 30)}).\n"
                     f"Train:\n  lab train --model xr1 --dataset {dataset} --gpu-ids 0,1"))

    # ---- train ---------------------------------------------------------
    def train(self, *, dataset: Path, gpus: int, gpu_ids: str | None = None,
              master_port: int | None = None, task: str | None = None,
              output_dir: str | None = None, overrides: list[str] | None = None,
              base_model: str | None = None, **opts: Any) -> StageResult:
        dataset = self.ctx.input_path(dataset)
        cwd = self._xr1_dir()
        env = self.manifest.env("train")
        tr = self.manifest.train

        name = self._config_name(dataset)
        data_config = cwd / "configs" / "data" / f"{name}.yaml"
        if not data_config.exists() and self.manifest.convert.get("config_prefix", "locomanilab_") == "locomanilab_":
            legacy_name = "soma_" + name.removeprefix("locomanilab_")
            legacy_config = data_config.with_name(f"{legacy_name}.yaml")
            if legacy_config.is_file():
                name, data_config = legacy_name, legacy_config
        if not self.runner.dry_run and not data_config.exists():
            return StageResult(
                ok=False,
                message=f"{data_config} missing — run "
                        f"`lab convert --model xr1 --dataset {dataset}` first")

        pretrained = self.ctx.input_path(base_model) if base_model else self.ctx.path("xr1_pretrained")
        if pretrained is None:
            return StageResult(
                ok=False,
                message="set paths.xr1_pretrained in env.local.yaml (or pass --base-model) to the "
                        "widened base checkpoint:\n"
                        "  python tools/widen_action_dim.py --input pretrained_ckpt/model_states.pt "
                        "--output pretrained_ckpt/model_states_sonic80.pt --action-dim 80")
        out_root = self.ctx.anchor(output_dir) if output_dir else (
            self.ctx.path("xr1_outputs") or self.ctx.workspace_root / "checkpoints" / "xr1")
        if out_root is None:
            return StageResult(ok=False,
                               message="set paths.xr1_outputs in env.local.yaml or pass --output-dir")

        if gpu_ids:
            n = len([x for x in gpu_ids.split(",") if x != ""])
            if gpus <= 1 and n > 1:
                gpus = n
        exp = self._exp_name(task, dataset)

        # hydra dotlist: an ordered mapping so `--set k=v` replaces a known key instead
        # of appending a second one (hydra would take the last, but a duplicated key in
        # the printed command is exactly the kind of thing --dry-run review misses).
        args: dict[str, str] = {
            "trainer.project": str(tr.get("project", "locomanilab")),
            "trainer.exp_name": exp,
            "trainer.default_root_dir": str(out_root),
            "trainer.max_steps": str(tr.get("max_steps", 20000)),
            "trainer.save_interval": str(tr.get("save_interval", 5000)),
            "data": name,
            "model": str(tr.get("model_config", "posttrain_sonic")),
            "model.params.pretrained": str(pretrained),
        }
        for kv in (overrides or []):
            kv = kv.lstrip("-")
            if "=" not in kv:
                return StageResult(ok=False, message=f"--set expects k=v for hydra, got {kv!r}")
            key, value = kv.split("=", 1)
            args[key] = value

        cmd = ["bash", tr.get("launcher", "scripts/train.sh")]
        cmd += [f"{k}={v}" for k, v in args.items()]

        extra_env: dict[str, str] = {
            "RESOURCE_GPU": str(gpus),
            "MAX_LENGTH": str(tr.get("max_length", 20000)),
        }
        extra_env.update(self._stage_env("train"))
        if master_port is not None:
            extra_env["MASTER_PORT"] = str(master_port)
        if gpu_ids:
            extra_env["CUDA_VISIBLE_DEVICES"] = gpu_ids

        r = self.runner.run(cmd, cwd=cwd, env_kind=env.kind, env_name=env.name,
                            setup=self.env_setup(env.kind), extra_env=extra_env, stage="train")
        run_dir = Path(out_root) / f"project_{tr.get('project', 'locomanilab')}" / exp
        if r.ok:
            r.artifacts["run_dir"] = run_dir
            r.message = (r.message or "") + f"\nrun dir: {run_dir}"
        return r

    # ---- deploy --------------------------------------------------------
    def deploy(self, *, model_path: str | None = None, run_dir: str | None = None,
               prompt: str | None = None, port: int = 5550,
               ckpt_step: int | None = None, prefix_len: int | None = None,
               **opts: Any) -> StageResult:
        """Start the xr1 gr00t-compat policy server (tcp://0.0.0.0:<port>/get_action).
        --model-path/--run-dir must point at a training run dir (config.py + last.ckpt/)."""
        self._prefix_len = prefix_len
        run = self._resolve_run(model_path or run_dir)
        if run is None:
            return StageResult(ok=False,
                               message="deploy needs --model-path/--run-dir pointing at an XR-1 "
                                       "run dir (<xr1_outputs>/project_<project>/<exp>)")
        env = self.manifest.env("deploy")
        return self.runner.run(self._server_cmd(run, port, prompt, self._ckpt_name(run, ckpt_step)),
                               cwd=self.repo(self.manifest.repo_key),
                               env_kind=env.kind, env_name=env.name,
                               setup=self.env_setup(env.kind),
                               extra_env=self._stage_env("deploy"), stage="deploy")

    # ---- eval / infer-check (shared harness) ---------------------------
    def _sim_server(self, *, stage: str, prompt: str | None, port: int,
                    model_path: str | None = None, run_dir: str | None = None,
                    ckpt_step: int | None = None, prefix_len: int | None = None,
                    **opts: Any) -> "SimServer | StageResult":
        """XR-1 gr00t-compat server (motion_token space), same harness as
        gr00t / fastwam / starvla / psi0 / internvla_a15.

        """
        self._prefix_len = prefix_len
        run = self._resolve_run(model_path or run_dir)
        if run is None:
            return StageResult(ok=False, message=(
                f"{stage} needs --model-path/--run-dir (XR-1 run dir), or "
                "env.local paths.xr1_run"))
        return SimServer(
            argv=self._server_argv(run, port, prompt, self._ckpt_name(run, ckpt_step)),
            cwd=self.repo(self.manifest.repo_key), prompt=prompt,
            inference_rate=self.manifest.deploy.get("inference_rate"))

    # ---- deploy helpers ------------------------------------------------
    def _resolve_run(self, path: str | None) -> Path | None:
        """A run dir holds config.py (dumped by cfg_utils.process_save_cfg) next to
        the Lightning checkpoints."""
        if path is None:
            p = self.ctx.path("xr1_run")
            path = str(p) if p is not None else None
        if path is None:
            return None
        run = self.ctx.input_path(path)
        if run.name.endswith(".ckpt"):        # someone passed .../last.ckpt
            run = run.parent
        if (run / "config.py").exists() or self.runner.dry_run:
            return run
        return None

    def _ckpt_name(self, run: Path, ckpt_step: int | None = None,
                   ckpt_name: str | None = None) -> str:
        """Which checkpoint dir inside the run to serve.

        Lightning writes `epoch=<E>-step=<N>.ckpt` per save plus a `last.ckpt`
        alias, so `--ckpt-step N` selects an intermediate save without anyone
        having to know the epoch number.
        """
        default = self.manifest.deploy.get("ckpt_name", "last.ckpt")
        if ckpt_name:
            return str(ckpt_name)
        step = ckpt_step
        if step in (None, "", "latest"):
            return default
        matches = sorted(run.glob(f"epoch=*-step={int(step)}.ckpt"))
        if matches:
            return matches[-1].name
        if self.runner.dry_run:
            return f"epoch=0-step={int(step)}.ckpt"
        raise FileNotFoundError(
            f"no checkpoint for step {step} in {run} "
            f"(have: {sorted(p.name for p in run.glob('*.ckpt')) or 'none'})")

    def _server_cmd(self, run: Path, port: int, prompt: str | None,
                    ckpt_name: str | None = None) -> list[str]:
        dep = self.manifest.deploy
        cmd = ["python", dep.get("module", "deploy/run_xr1_server.py"),
               "--run-dir", str(run),
               "--ckpt-name", str(ckpt_name or dep.get("ckpt_name", "last.ckpt")),
               "--host", "0.0.0.0", "--port", str(port)]
        prefix_len = self._prefix_len if self._prefix_len is not None else dep.get("prefix_len", 0)
        if int(prefix_len) > 0:
            cmd += ["--prefix-len", str(int(prefix_len)),
                    "--replan-steps", str(dep.get("replan_steps", 20))]
        if dep.get("num_steps"):
            cmd += ["--num-steps", str(int(dep["num_steps"]))]
        if float(dep.get("chunk_gain", 1.0)) != 1.0:
            cmd += ["--chunk-gain", str(float(dep["chunk_gain"]))]
        prompt = prompt or dep.get("prompt_fixed")
        if prompt:
            cmd += ["--initial-prompt", prompt]
        return cmd

    def _server_argv(self, run: Path, port: int, prompt: str | None,
                     ckpt_name: str | None = None) -> list[str]:
        """Fully env-wrapped server command for sim/infer-check (Popen'd directly, not
        through runner.run): source the conda isolation script, activate, cd, exec.

        `export PATH="$CONDA_PREFIX/bin:$PATH"` is not optional on shared machines — a bare
        `python` gets stolen by whatever venv is ahead on PATH and the imports fail.
        """
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("deploy")
        parts = []
        setup = self.env_setup(env.kind)
        if setup:
            parts.append(f"source {shlex.quote(setup)}")
        if env.kind == "conda":
            parts.append(f"conda activate {shlex.quote(env.name)}")
            parts.append('export PATH="$CONDA_PREFIX/bin:$PATH"')
        for key, value in self._stage_env("deploy").items():
            parts.append(f"export {key}={shlex.quote(value)}")
        parts.append(f"cd {shlex.quote(str(repo))}")
        parts.append(f"exec {shlex.join(self._server_cmd(run, port, prompt, ckpt_name))}")
        return ["bash", "-lc", " && ".join(parts)]
