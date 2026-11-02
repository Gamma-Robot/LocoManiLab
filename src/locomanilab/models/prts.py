"""Native PRTS adapter with explicit local dataset YAML recipes.

PRTS training and serving dependencies remain in the backend subprocess.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from locomanilab.plugin import ModelPlugin, SimServer, StageResult, register
from locomanilab.models._recipes import dataset_identity
from locomanilab.models._prts_recipe import validate_recipe


@register("prts")
class PRTSPlugin(ModelPlugin):

    def _recipe(self, dataset: Path, config_name: str | None):
        """Generate a PRTS YAML whose root is exactly the requested dataset."""
        from locomanilab.common import lerobot_meta, variants as variantmod
        found = variantmod.detect(dataset, self.manifest.raw.get("variants") or {})
        if isinstance(found, str):
            return StageResult(ok=False, message=f"PRTS dataset cannot be selected: {found}")
        name, variant = found
        if name != "sonic":
            return StageResult(ok=False, message=(
                f"PRTS {name} training is unavailable: the pinned PRTS policy server supports only "
                "the SONIC 43-D state / 78-D motion-token-and-hands contract. "
                "This dataset is a different controller contract; use an adapter that supports it."))
        problem = variantmod.check_modality(
            dataset, variant.get("require_modality") or {}, what="PRTS training recipe")
        if problem:
            return StageResult(ok=False, message=problem)
        row = {"repo_id": f"locomanilab/{dataset.name}", "root": str(dataset),
               "load_quantile_stats": True, "state_relative_action": False,
               "select_video_keys": ["observation.images.ego_view"],
               "select_state_keys": ["observation.state"],
               "select_action_keys": ["action.motion_token", "teleop.left_hand_joints", "teleop.right_hand_joints"],
               "embodiment_tag": variant.get("embodiment_tag", "unitree_g1_sonic")}
        if config_name:
            template = self.ctx.input_path(config_name)
            try:
                payload = yaml.safe_load(template.read_text())
                rows = payload.get("lerobot_datasets") or []
                if payload.get("mm_datasets") or len(rows) != 1:
                    raise ValueError("the template must contain exactly one LeRobot dataset and no multimodal datasets")
                row.update(rows[0])
            except (OSError, ValueError, AttributeError, TypeError, yaml.YAMLError) as exc:
                return StageResult(ok=False, message=f"cannot read PRTS recipe template {template}: {exc}")
            row.update(root=str(dataset), repo_id=f"locomanilab/{dataset.name}",
                       embodiment_tag=variant.get("embodiment_tag", "unitree_g1_sonic"))
        for field in ("select_video_keys", "select_state_keys", "select_action_keys"):
            keys = row.get(field)
            if not isinstance(keys, list) or not keys or any(not isinstance(key, str) for key in keys):
                return StageResult(ok=False, message=f"PRTS recipe {field} must be an explicit nonempty list of columns")
        info = lerobot_meta.load_info(dataset) or {}
        features = info.get("features") or {}
        if features:
            needed = [key for field in ("select_video_keys", "select_state_keys", "select_action_keys")
                      for key in row.get(field) or []]
            missing = [key for key in needed if key not in features]
            if missing:
                return StageResult(ok=False, message=(
                    f"PRTS recipe requires missing dataset columns: {', '.join(missing)}. "
                    "Prepare the dataset or provide a matching --config-name YAML template."))
        problem = validate_recipe(row, info, lerobot_meta.load_modality(dataset) or {}, name)
        if problem:
            return StageResult(ok=False, message=problem)
        payload = {"lerobot_datasets": [row]}
        identity = json.dumps({"recipe": payload, **dataset_identity(dataset)}, sort_keys=True)
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        path = self.ctx.config_root / "artifacts" / "prts" / "recipes" / f"{digest}.yaml"
        return path, payload, variant

    def _write_recipe(self, path: Path, payload: dict) -> None:
        if not self.runner.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(payload, sort_keys=False))

    # ---- convert -------------------------------------------------------
    def convert(self, *, dataset: Path, config_name: str | None = None, **opts: Any) -> StageResult:
        """Validate and compute native PRTS statistics using a local dataset recipe."""
        dataset = self.ctx.input_path(dataset)
        recipe = self._recipe(dataset, config_name)
        if isinstance(recipe, StageResult):
            return recipe
        path, payload, _ = recipe
        self._write_recipe(path, payload)
        env = self.manifest.env("convert")
        extra_env = {"HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE") or str(Path.home() / ".cache/huggingface/hub")}
        command = ["python", self.manifest.convert.get("compute_stats", "scripts/compute_stats.py"),
                   "--data_path", str(path)]
        result = self.runner.run(command, cwd=self.repo(self.manifest.repo_key), env_kind=env.kind,
                                 env_name=env.name, setup=self.env_setup(env.kind),
                                 extra_env=extra_env, stage="convert")
        result.artifacts["recipe"] = path
        result.message += f"; dataset={dataset}; recipe={path}"
        return result

    def _base_model(self, override: str | None, tr: dict[str, Any]) -> str:
        """Prefer the explicit local base checkpoint, then machine configuration."""
        if override:
            return str(self.ctx.input_path(override))
        p = self.ctx.path("prts_base")
        if p is not None:
            return str(p)
        return tr.get("base_model", "TeleEmbodied/PRTS-4B")


    # ---- train ---------------------------------------------------------
    def train(self, *, dataset: Path, gpus: int, gpu_ids: str | None = None,
              master_port: int | None = None, task: str | None = None,
              base_model: str | None = None, output_dir: str | None = None,
              config_name: str | None = None, overrides: list[str] | None = None, **opts: Any) -> StageResult:
        dataset = self.ctx.input_path(dataset)
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("train")
        tr = self.manifest.train
        d = tr.get("defaults", {})
        recipe = self._recipe(dataset, config_name)
        if isinstance(recipe, StageResult):
            return recipe
        recipe_path, recipe_payload, variant = recipe
        for override in overrides or []:
            key = override.lstrip("-").split("=", 1)[0].replace("_", "-")
            if key in ("data-path", "max-action-dim", "embodiment-tag"):
                return StageResult(ok=False, message=(
                    f"--set {key} conflicts with the dataset/controller contract; "
                    "use --dataset and a verified --config-name recipe template"))
        self._write_recipe(recipe_path, recipe_payload)
        run_name = task or ("locomanilab_prts_" + time.strftime("%Y%m%d_%H%M%S"))

        extra_env: dict[str, str] = dict(tr.get("env", {}))
        self.drop_expandable_segments(extra_env, "prts")
        gpus = self.resolve_gpus(gpus, gpu_ids, extra_env)
        extra_env.setdefault("HF_HUB_OFFLINE", os.environ.get("HF_HUB_OFFLINE", "1"))   # Explicit online/offline settings are inherited.
        # PRTS's training/config.py reads os.environ["HF_HUB_CACHE"] at import time (for the
        # default vlm_name_or_path), so an unset var is a KeyError before training starts —
        # the deploy path already guards this; train needs the same.
        extra_env.setdefault(
            "HF_HUB_CACHE",
            os.environ.get("HF_HUB_CACHE") or str(Path.home() / ".cache/huggingface/hub"))

        # env_kind="conda" -> runner wraps `conda activate <env>`; cmd is the torchrun line.
        cmd = ["torchrun", f"--nproc_per_node={gpus}"]
        if master_port is not None:
            cmd += [f"--master_port={master_port}"]
        cmd += [tr.get("entrypoint", "scripts/train.py"),
                "--deepspeed", tr.get("deepspeed", "configs/zero2.json"),
                "--data-path", str(recipe_path),
                "--embodiment-tag", variant.get("embodiment_tag")
                or tr.get("embodiment_tag", "unitree_g1_sonic"),
                "--model-name-or-path", self._base_model(base_model, tr),
                "--state-mode", str(d.get("state_mode", "QUANTILE")),
                "--chunk-size", str(d.get("chunk_size", 20)),
                "--max-action-dim", str(variant.get("action_dim", d.get("action_dim", 78))),
                "--per-device-train-batch-size", str(d.get("per_device_batch_size", 4)),
                "--max-steps", str(d.get("max_train_steps", 30000)),
                "--save-steps", str(d.get("save_steps", 10000)),
                "--warmup-steps", str(d.get("warmup_steps", 5000)),
                "--attn-implementation", tr.get("attn", "sdpa"),
                "--rope-type", tr.get("rope_type", "qwen3_vl"),
                "--bf16", "True", "--run-name", run_name, "--report-to", "wandb"]
        output = self.ctx.anchor(output_dir) if output_dir else (
            self.ctx.path("prts_outputs") or self.ctx.workspace_root / "checkpoints" / "prts") / run_name
        cmd += ["--output-dir", str(output)]
        for kv in (overrides or []):
            cmd.append(f"--{kv}" if not kv.startswith("--") else kv)

        result = self.runner.run(cmd, cwd=repo, env_kind=env.kind, env_name=env.name,
                                 setup=self.env_setup(env.kind),
                                 extra_env=extra_env or None, stage="train")
        result.artifacts.update(recipe=recipe_path, run_dir=output)
        result.message += f"; dataset={dataset}; recipe={recipe_path}"
        return result

    # ---- deploy --------------------------------------------------------
    def deploy(self, *, model_path: str | None = None, run_dir: str | None = None,
               prompt: str | None = None, port: int = 5550, native: bool = False,
               **opts: Any) -> StageResult:
        """Start a gr00t-compatible policy server (:5550/get_action) for a PRTS checkpoint,
        so the real G1 loop and the tele sim client can drive it like any other model.
        Pass native=True for PRTS's own openpi websocket server (:10000) instead."""
        dep = self.manifest.deploy
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("deploy")

        if native:
            # The native server can start without a checkpoint (its own default).
            given = model_path or run_dir or self.ctx.path("prts_ckpt")
            cmd = ["python", dep.get("websocket_server", "scripts/serve_policy.py"),
                   "--port", str(dep.get("websocket_port", 10000))]
            if given:
                cmd += ["--ckpt", str(self.ctx.input_path(given))]
        else:
            ckpt = self.resolve_ckpt(model_path, run_dir, env_key="prts_ckpt", stage="deploy",
                                     hint=self.CKPT_HINT)
            if isinstance(ckpt, StageResult):
                return ckpt
            cmd = self._gr00t_server_cmd(str(ckpt), port, prompt)
        return self.runner.run(cmd, cwd=repo, env_kind=env.kind, env_name=env.name,
                               setup=self.env_setup(env.kind), stage="deploy")

    # ---- sim harness (eval + infer-check share this) --------------------
    CKPT_HINT = ("--model-path (a PRTS checkpoint dir, e.g. outputs/prts/<run>/checkpoint-20000), "
                 "--run-dir, or paths.prts_ckpt in env.local.yaml")

    def _sim_server(self, *, stage: str, prompt: str | None, port: int,
                    model_path: str | None = None, run_dir: str | None = None,
                    **opts: Any) -> SimServer | StageResult:
        """PRTS gr00t-compat server (conda prts). PRTS emits latent motion_token (like
        fastwam/starVLA)."""
        ckpt = self.resolve_ckpt(model_path, run_dir, env_key="prts_ckpt", stage=stage,
                                 hint=self.CKPT_HINT)
        if isinstance(ckpt, StageResult):
            return ckpt
        prompt = prompt or self.manifest.deploy.get("prompt_fixed")
        return SimServer(argv=self._server_argv(str(ckpt), port, prompt),
                         cwd=self.repo(self.manifest.repo_key), prompt=prompt)

    # ---- deploy helpers ------------------------------------------------
    def _gr00t_server_cmd(self, ckpt: str, port: int, prompt: str | None) -> list[str]:
        dep = self.manifest.deploy
        server = dep.get("gr00t_server", "deployment/run_prts_server.py")
        repo = self.repo(self.manifest.repo_key)
        cmd = ["python", server, "--ckpt-path", str(ckpt), "--port", str(port),
               "--prts-repo", str(repo)]
        prompt = prompt or dep.get("prompt_fixed")
        if prompt:
            cmd += ["--initial-prompt", str(prompt)]
        if dep.get("state_mode"):
            cmd += ["--state-mode", str(dep["state_mode"])]
        return cmd

    def _server_argv(self, ckpt: str, port: int, prompt: str | None) -> list[str]:
        """Env-wrapped gr00t-compat server for the sim harness (Popen'd directly, not via
        runner.run). HF_HUB_CACHE is read with os.environ[...] in PRTS's config.py --
        unset means KeyError, so it is exported (keeping any value already set)."""
        import shlex
        repo = self.repo(self.manifest.repo_key)
        pre = [f"export PYTHONPATH={shlex.quote(str(repo))}${{PYTHONPATH:+:${{PYTHONPATH}}}}",
               'export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"']
        return self.server_argv(self._gr00t_server_cmd(ckpt, port, prompt), pre=pre)
