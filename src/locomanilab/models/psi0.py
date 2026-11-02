"""Psi0 plugin.

Maps locomanilab's uniform stages onto Psi0's native scripts (all run via `uv run --no-sync`
inside external/Psi0, mirroring scripts/train/psi0/finetune-real-sonic-psi0.sh):
  convert -> scripts/data/locomanilab_sonic_to_psi_lerobot.py + calc_modality_stats.py
             + cp stats.json stats_psi0.json          (Psi0 LeRobot layout rewrite)
  train   -> torchrun scripts/train.py finetune_real_psi0_config --train.name=sonic ...
  deploy  -> python src/psi/deploy/run_psi0_server.py   (gr00t :5550 / get_action)
  eval / infer-check -> the same server + the shared sim / offline client

Psi0 supports G1 SONIC natively (action 78 = motion_token 64 + hands 14, odim 43,
chunk 30) — no embodiment registration was needed; only the converters and the
gr00t-compat server were added (all new files, no upstream file modified).

locomanilab collects with three different whole-body controllers, and each records a
different action space, so convert picks the converter by looking at the source
dataset's meta/modality.json (configs/psi0.yaml `variants:`):
    sonic  action 78 / states 43   heft  action 50 / states 29
    amo    action 36 / states 32
train then reads the dims back out of the converted data (meta/stats_psi0.json)
instead of assuming — the pretrained action expert tolerates the mismatch by
keeping only its transformer blocks.

Machine specifics come from env.local.yaml: repos.psi0 (default external/Psi0) and
paths.psi0_data_root / psi0_vlm_base / psi0_action_expert / psi0_outputs.
`--set k=v` overrides any tyro training arg by its dotted name (e.g.
--set train.max_training_steps=200 --set train.train_batch_size=8).
"""
from __future__ import annotations

import os

import json
import re
import shutil
from pathlib import Path
from typing import Any

from locomanilab.plugin import ModelPlugin, SimServer, StageResult, register
from locomanilab.models._legacy import converter_path


def _dataset_path(dataset: Path | str) -> Path:
    """Absolute, but WITHOUT resolving symlinks.

    The converted dir and the run dir are named after the dataset's basename, and the
    bench datasets are reached through `datasets/bench_<backend>_<task>` symlinks that
    all point at `bench_dataset/<backend>/<task>`. `.resolve()` collapses those to the
    same basename, so heft's KitchenOpenFridge silently overwrote sonic's — and a psi0
    heft run then trained on 43/78-D sonic data with 29/50-D dims and died at step 0
    (`shape '[16, -1, 50]' is invalid for input of size 37440`, 2026-08-05).
    """
    return Path(dataset).expanduser().absolute()


@register("psi0")
class Psi0Plugin(ModelPlugin):

    # ---- helpers -------------------------------------------------------
    def _data_root(self) -> Path | None:
        p = self.ctx.path("psi0_data_root")
        return Path(p) if p is not None else None

    def _converted_dir(self, dataset: Path) -> Path | None:
        root = self._data_root()
        return (root / Path(dataset).name) if root is not None else None

    def _variant(self, dataset: Path) -> tuple[str, dict[str, Any]] | str:
        """Which controller produced this dataset (sonic | amo | heft) — shared helper."""
        from locomanilab.common import variants as variantmod
        return variantmod.detect(dataset, self.manifest.raw.get("variants") or {})

    def _dims(self, conv: Path) -> tuple[int, int] | None:
        """(action_dim, state_dim) as actually written into the converted dataset.
        Authoritative over the manifest — the converter decides the layout."""
        stats = conv / "meta" / "stats_psi0.json"
        if not stats.exists():
            return None
        try:
            s = json.loads(stats.read_text())
            return len(s["action"]["min"]), len(s["states"]["min"])
        except Exception:  # noqa: BLE001
            return None

    # ---- convert -------------------------------------------------------
    def convert(self, *, dataset: Path, **opts: Any) -> StageResult:
        """locomanilab-prepared GR00T LeRobot -> Psi0 LeRobot (states 43 reordered, action 78,
        egocentric video key), then dataset statistics -> meta/stats_psi0.json."""
        dataset = self.ctx.input_path(dataset)
        cv = self.manifest.convert

        found = self._variant(dataset)
        if isinstance(found, str):
            return StageResult(ok=False, message=found)
        variant, spec = found
        print(f"[locomanilab:psi0:convert] source schema: {variant} "
              f"(action {spec.get('action_dim')} / states {spec.get('state_dim')})")

        root = self._data_root()
        if root is None:
            return StageResult(ok=False,
                               message="set paths.psi0_data_root in env.local.yaml "
                                       "(e.g. <PSI_HOME>/data/sonic/lerobot)")
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("convert")
        repo_id = dataset.name

        r = self.runner.run(
            ["python", converter_path(repo, spec["script"]),
             "--data-root", str(dataset), "--work-dir", str(root), "--repo-id", repo_id],
            cwd=repo, env_kind=env.kind, env_name=env.name,
            setup=self.env_setup(env.kind), stage="convert")
        if not r.ok:
            return r

        r2 = self.runner.run(
            ["python", cv.get("stats_script", "scripts/data/calc_modality_stats.py"),
             "--work-dir", str(root), "--task", repo_id],
            cwd=repo, env_kind=env.kind, env_name=env.name,
            setup=self.env_setup(env.kind), stage="convert")
        if not r2.ok:
            return r2

        out = root / repo_id
        if not self.runner.dry_run:
            stats = out / "meta" / "stats.json"
            if not stats.exists():
                return StageResult(ok=False, message=f"{stats} missing after calc_modality_stats")
            # Psi0-format stats: upstream README says "simply a copy for now"
            shutil.copyfile(stats, out / "meta" / "stats_psi0.json")
        dims = self._dims(out) or (spec.get("action_dim"), spec.get("state_dim"))
        return StageResult(
            ok=True,
            message=(f"OK — converted to {out} ({variant}: states {dims[1]} / action {dims[0]} "
                     f"/ stats_psi0.json).\n"
                     f"Train:\n  lab train --model psi0 --dataset {dataset} --gpu-ids N"))

    # ---- train ---------------------------------------------------------
    def train(self, *, dataset: Path, gpus: int, gpu_ids: str | None = None,
              master_port: int | None = None, task: str | None = None,
              output_dir: str | None = None, overrides: list[str] | None = None,
              **opts: Any) -> StageResult:
        dataset = self.ctx.input_path(dataset)
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("train")
        tr = self.manifest.train

        root = self._data_root()
        conv = self._converted_dir(dataset)
        if conv is None:
            return StageResult(ok=False, message="set paths.psi0_data_root in env.local.yaml")
        if not self.runner.dry_run and not (conv / "meta" / "stats_psi0.json").exists():
            return StageResult(
                ok=False,
                message=f"{conv}/meta/stats_psi0.json missing — run "
                        f"`lab convert --model psi0 --dataset {dataset}` first")

        # Which trainer: psi.trainers.<name>.<Name>Trainer, and also the run-dir segment
        # (<psi0_outputs>/<name>/<exp>.<ts>) and wandb group — so an amo run says amo.
        found = self._variant(dataset)
        trainer_name = (found[1].get("trainer") if not isinstance(found, str) else None) \
            or tr.get("trainer_name", "sonic")

        # The action space is whatever the converter wrote (sonic 78/43, amo 36/32,
        # heft 50/29) — read it back off the converted data rather than assuming.
        dims = self._dims(conv)
        if dims is not None:
            action_dim, state_dim = dims
            print(f"[locomanilab:psi0:train] trainer={trainer_name} action_dim={action_dim} "
                  f"odim={state_dim} (from {conv.name})")
        else:
            action_dim = int(tr.get("action_dim", 78))
            state_dim = int(tr.get("state_dim", 43))

        vlm = self.ctx.path("psi0_vlm_base")
        header = self.ctx.path("psi0_action_expert")
        if vlm is None or header is None:
            return StageResult(ok=False,
                               message="set paths.psi0_vlm_base and paths.psi0_action_expert in "
                                       "env.local.yaml (download from HF USC-PSI-Lab/psi-model)")
        out_root = self.ctx.anchor(output_dir) if output_dir else (
            self.ctx.path("psi0_outputs") or self.ctx.workspace_root / "checkpoints" / "psi0")
        if out_root is None:
            return StageResult(ok=False,
                               message="set paths.psi0_outputs in env.local.yaml or pass --output-dir")
        # exp names the run dir (<out_root>/sonic/<exp>.<timestamp>); dots would confuse
        # the trainer's timestamp parsing (it splits on '.'), so sanitize.
        exp = re.sub(r"[^A-Za-z0-9_-]+", "-", task or dataset.name).strip("-").lower()

        extra_env: dict[str, str] = {"OMP_NUM_THREADS": "16", "TOKENIZERS_PARALLELISM": "false",
                                     "PSI_NUM_WORKERS": str(tr.get("num_workers", 4))}
        gpus = self.resolve_gpus(gpus, gpu_ids, extra_env)

        # tyro args as an ordered mapping so --set can REPLACE (tyro rejects duplicate
        # flags, unlike OmegaConf dotlists). Value None = bare boolean flag.
        args: dict[str, Any] = {
            "seed": "292285",
            "exp": exp,
            "train.name": trainer_name,
            "train.data_parallel": "ddp",
            "train.mixed_precision": "bf16",
            "train.train_batch_size": str(tr.get("train_batch_size", 16)),
            "train.max_checkpoints_to_keep": "5",
            "train.gradient_accumulation_steps": "1",
            "train.learning_rate": str(tr.get("learning_rate", "1e-4")),
            "train.max_training_steps": str(tr.get("max_training_steps", 20000)),
            "train.warmup_ratio": "None",
            "train.warmup_steps": str(tr.get("warmup_steps", 1000)),
            "train.checkpointing_steps": str(tr.get("checkpointing_steps", 5000)),
            "train.validation_steps": "1000",
            "train.val_num_batches": "20",
            "train.max_grad_norm": "1.0",
            "train.lr_scheduler_type": "cosine",
            "train.lr_scheduler_kwargs.weight_decay": "1e-6",
            "train.lr_scheduler_kwargs.betas": ("0.95", "0.999"),
            "train.output_dir": str(out_root),
            "log.report_to": "wandb",
            "data.root_dir": str(root),
            "data.train_repo_ids": dataset.name,
            "data.transform.field.stat-path": "meta/stats_psi0.json",
            "data.transform.field.stat-action-key": "action",
            "data.transform.field.stat-state-key": "states",
            "data.transform.field.action_norm_type": "bounds",
            "data.transform.field.no-use-norm-mask": None,
            "data.transform.field.normalize-state": None,
            "data.transform.model.img-aug": None,
            "data.transform.model.resize.size": ("240", "320"),
            "data.transform.model.center_crop.size": ("240", "320"),
            "model.model_name_or_path": str(vlm),
            "model.pretrained-action-header-path": str(header),
            "model.noise-scheduler": "flow",
            "model.train-diffusion-steps": "1000",
            "model.n_conditions": "0",
            "model.action-chunk-size": str(tr.get("action_chunk_size", 30)),
            "model.action-dim": str(action_dim),
            "model.action-exec-horizon": str(tr.get("action_chunk_size", 30)),
            "model.observation-horizon": "1",
            "model.odim": str(state_dim),
            "model.view_feature_dim": "2048",
            "model.no-tune-vlm": None,
            "model.no-use_film": None,
            "model.no-combined_temb": None,
            "model.rtc": None,
            "model.max-delay": "8",
        }
        for kv in (overrides or []):
            kv = kv.lstrip("-")
            if "=" in kv:
                k, v = kv.split("=", 1)
                args[k] = v          # replace known key or append a new one
            else:
                args[kv] = None      # bare flag, e.g. --set model.no-rtc

        cmd = ["torchrun", f"--nproc_per_node={gpus}"]
        if master_port is not None:
            cmd.append(f"--master_port={master_port}")
        cmd += [tr.get("entrypoint", "scripts/train.py"), tr.get("config_name", "finetune_real_psi0_config")]
        for k, v in args.items():
            if v is None:
                cmd.append(f"--{k}")
            elif isinstance(v, tuple):
                cmd.append(f"--{k}")
                cmd.extend(v)
            else:
                cmd.append(f"--{k}={v}")

        if tr.get("wandb_offline", True):
            extra_env.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))
        # base ckpts are local paths — keep transformers from probing HF through the
        # unavailable network proxy (same failure mode as starVLA's Warming Up hang).
        extra_env.setdefault("HF_HUB_OFFLINE", os.environ.get("HF_HUB_OFFLINE", "1"))
        extra_env.setdefault("TRANSFORMERS_OFFLINE", os.environ.get("TRANSFORMERS_OFFLINE", "1"))

        # Psi0's sonic trainer hardcodes num_workers=12 for train AND val and validates at
        # step 0 -> 24 workers; under a 1024 soft limit that hangs (see raise_nofile).
        self.raise_nofile(model="psi0")
        return self.runner.run(cmd, cwd=repo, env_kind=env.kind, env_name=env.name,
                               setup=self.env_setup(env.kind),
                               extra_env=extra_env, stage="train")

    # ---- deploy --------------------------------------------------------
    def deploy(self, *, model_path: str | None = None, run_dir: str | None = None,
               prompt: str | None = None, port: int = 5550,
               **opts: Any) -> StageResult:
        """Start the psi0 gr00t-compat policy server (tcp://0.0.0.0:<port>/get_action).
        --model-path/--run-dir accept either the run dir (has run_config.json) or a
        checkpoints/ckpt_<N> dir inside it (pins that step)."""
        resolved = self._resolve_run(model_path or run_dir)
        if resolved is None:
            return StageResult(ok=False,
                               message="deploy needs --model-path/--run-dir pointing at a Psi0 "
                                       "run dir (or its checkpoints/ckpt_<N>)")
        run, step = resolved
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("deploy")
        cmd = self._server_cmd(run, step, port, prompt)
        return self.runner.run(cmd, cwd=repo, env_kind=env.kind, env_name=env.name,
                               setup=self.env_setup(env.kind),
                               extra_env=self._hf_env(), stage="deploy")

    # ---- sim harness (eval + infer-check share this) --------------------
    def _sim_server(self, *, stage: str, prompt: str | None, port: int,
                    model_path: str | None = None, run_dir: str | None = None,
                    **opts: Any) -> SimServer | StageResult:
        """psi0 gr00t-compat server (uv, run dir or checkpoints/ckpt_<N>); motion_token space."""
        resolved = self._resolve_run(model_path or run_dir)
        if resolved is None:
            return StageResult(ok=False, message=f"{stage} needs --model-path/--run-dir (Psi0 run dir)")
        run, step = resolved
        return SimServer(argv=self._server_argv(run, step, port, prompt),
                         cwd=self.repo(self.manifest.repo_key), prompt=prompt)

    # ---- deploy helpers ------------------------------------------------
    def _resolve_run(self, path: str | None) -> tuple[Path, str] | None:
        """Accept a run dir or a checkpoints/ckpt_<N> dir; return (run_dir, step)."""
        if path is None:
            p = self.ctx.path("psi0_run")
            path = str(p) if p is not None else None
        if path is None:
            return None
        p = self.ctx.input_path(path)
        m = re.match(r"ckpt_(\d+)$", p.name)
        if m and p.parent.name == "checkpoints":
            return p.parent.parent, m.group(1)
        if (p / "run_config.json").exists() or self.runner.dry_run:
            return p, "latest"
        return None

    def _server_cmd(self, run: Path, step: str, port: int, prompt: str | None) -> list[str]:
        dep = self.manifest.deploy
        cmd = ["python", dep.get("module", "src/psi/deploy/run_psi0_server.py"),
               "--run-dir", str(run), "--ckpt-step", str(step),
               "--host", "0.0.0.0", "--port", str(port),
               "--num-inference-steps", str(dep.get("num_inference_steps", 8))]
        if prompt:
            cmd += ["--initial-prompt", prompt]
        return cmd

    def _hf_env(self) -> dict[str, str]:
        """Point the HF cache at psi0_home and forbid network lookups.

        The server loads a Qwen tokenizer/config through transformers, which by
        default resolves against the Hub. Training already pins this; deploy did
        not, so on a machine without egress the policy server died at startup
        with a DNS failure for hf-mirror.com -- while the very files it wanted
        sat in <psi0_home>/hf_hub. Offline is the right default either way: a
        deploy that silently re-downloads a backbone is a deploy that can change
        under you.
        """
        env = {"HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
                                     "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1")}
        home = self.ctx.path("psi0_home")
        if home is not None:
            env["HF_HOME"] = str(home / "hf_hub")
            env["HF_HUB_CACHE"] = str(home / "hf_hub")
        return env

    def _server_argv(self, run: Path, step: str, port: int, prompt: str | None) -> list[str]:
        """Env-wrapped server for the sim harness (Popen'd directly): source the uv
        isolation script, cd into the repo, export the HF offline env (nothing else
        applies extra_env for us here), exec via `uv run --no-sync`."""
        from locomanilab.runner import export_line
        repo = self.repo(self.manifest.repo_key)
        pre = [export_line(k, v) for k, v in self._hf_env().items()]
        return self.server_argv(self._server_cmd(run, step, port, prompt),
                                env_kind="uv", cwd=repo, pre=pre)
