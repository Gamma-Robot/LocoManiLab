"""Native openpi adapters with verified local dataset and normalization bindings.

Model dependencies are loaded only in the backend subprocess.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from locomanilab.plugin import ModelPlugin, SimServer, StageResult, register
from locomanilab.models._recipes import dataset_home, dataset_identity, openpi_repo_id, recipe_key


@register("pi05")
class Pi05Plugin(ModelPlugin):

    # ---- convert -------------------------------------------------------
    def _openpi_env(self) -> dict[str, str]:
        """Optional local tokenizer/assets cache and LeRobot data root."""
        env = {}
        for path_key, env_key in (("openpi_data_home", "OPENPI_DATA_HOME"),
                                  ("datasets_root", "HF_LEROBOT_HOME")):
            path = self.ctx.path(path_key)
            if path is not None:
                env[env_key] = str(path)
        if "HF_LEROBOT_HOME" not in env and self.ctx.path("datasets_dir"):
            env["HF_LEROBOT_HOME"] = str(self.ctx.path("datasets_dir"))
        return env

    def _recipe(self, dataset: Path, config_name: str | None):
        """Resolve a recipe and prove which on-disk dataset it will read."""
        from locomanilab.common import variants as variantmod
        found = variantmod.detect(dataset, self.manifest.raw.get("variants") or {})
        if isinstance(found, str):
            return StageResult(ok=False, message=f"pi05 dataset cannot be selected: {found}")
        controller, variant = found
        if reason := variant.get("unavailable_reason"):
            return StageResult(ok=False, message=f"pi05 {controller} conversion/training unavailable: {reason}")
        problem = variantmod.check_modality(
            dataset, variant.get("require_modality") or {}, what="pi05 training recipe")
        if problem:
            return StageResult(ok=False, message=problem)
        name = config_name or variant.get("config_name") or self.manifest.train.get("config_name")
        try:
            factories = {"sonic": "LeRobotG1SonicDataConfig", "heft": "LeRobotG1HeftDataConfig",
                         "amo": "LeRobotG1AmoDataConfig"}
            if controller not in factories:
                raise ValueError(f"pi05 has no data recipe for controller {controller!r}")
            repo_id = openpi_repo_id(self.repo(self.manifest.repo_key), name,
                                    expected_factory=factories[controller])
            home = dataset_home(dataset, repo_id)
        except ValueError as exc:
            return StageResult(ok=False, message=str(exc))
        root = self.ctx.path("pi05_assets") or self.ctx.config_root / "artifacts" / "pi05"
        return name, repo_id, home, root / recipe_key(dataset, name)


    def convert(self, *, dataset: Path, config_name: str | None = None, **opts: Any) -> StageResult:
        """Compute native openpi normalization for this dataset and matching recipe."""
        dataset = self.ctx.input_path(dataset)
        recipe = self._recipe(dataset, config_name)
        if isinstance(recipe, StageResult):
            return recipe
        name, repo_id, home, assets_base = recipe
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("convert")
        command = ["python", str(Path(__file__).with_name("_openpi_stats.py")),
                   "--repo", str(repo), "--config-name", name, "--repo-id", repo_id,
                   "--dataset-home", str(home), "--assets-base-dir", str(assets_base)]
        extra_env = self._openpi_env()
        extra_env["HF_LEROBOT_HOME"] = str(home)
        result = self.runner.run(command, cwd=repo, env_kind=env.kind, env_name=env.name,
                                 setup=self.env_setup(env.kind), extra_env=extra_env, stage="convert")
        stats_dir = assets_base / name / repo_id
        if result.ok:
            if not self.runner.dry_run:
                if not (stats_dir / "norm_stats.json").is_file():
                    return StageResult(ok=False, message=f"openpi did not write {stats_dir / 'norm_stats.json'}")
                (stats_dir / "locomanilab_dataset.json").write_text(json.dumps(dataset_identity(dataset), indent=2) + "\n")
            result.artifacts["norm_stats"] = stats_dir / "norm_stats.json"
            result.message += f"; dataset={dataset}; recipe={name}; statistics={stats_dir}"
        return result

    # ---- train ---------------------------------------------------------
    def train(self, *, dataset: Path, gpus: int, gpu_ids: str | None = None,
              task: str | None = None, config_name: str | None = None,
              base_model: str | None = None, output_dir: str | None = None,
              overrides: list[str] | None = None, **opts: Any) -> StageResult:
        """Train a matching native JAX recipe on the explicitly selected local dataset."""
        dataset = self.ctx.input_path(dataset)
        tr = self.manifest.train
        if tr.get("entry", "jax") != "jax":
            return StageResult(ok=False, message=(
                "pi05 supports the native JAX training entry. The PyTorch base-weight "
                "conversion is not configured; refusing to start from scratch."))
        recipe = self._recipe(dataset, config_name)
        if isinstance(recipe, StageResult):
            return recipe
        config_name, repo_id, home, assets_base = recipe
        stats_dir = assets_base / config_name / repo_id
        if not self.runner.dry_run:
            identity_file = stats_dir / "locomanilab_dataset.json"
            if not identity_file.exists():
                identity_file = stats_dir / "soma_dataset.json"  # Existing normalization outputs.
            try:
                matches = json.loads(identity_file.read_text()) == dataset_identity(dataset)
            except (OSError, ValueError):
                matches = False
            if not matches or not (stats_dir / "norm_stats.json").is_file():
                return StageResult(ok=False, message=(
                    f"pi05 needs dataset-bound normalization. Run lab convert -M pi05 "
                    f"-d {dataset} --config-name {config_name} first."))
        base_model = base_model or self.ctx.path("pi05_base") or tr.get("base_model")
        if not base_model:
            return StageResult(ok=False, message=(
                "pi05 train needs --base-model <local params directory> or paths.pi05_base"))
        reserved = ("data.repo-id", "data.assets.", "assets-base-dir",
                    "weight-loader.params-path", "checkpoint-base-dir")
        for override in overrides or []:
            key = override.lstrip("-").split("=", 1)[0].replace("_", "-")
            if any(key == field or key.startswith(field) for field in reserved):
                return StageResult(ok=False, message=(
                    f"--set {key} conflicts with the dataset-bound recipe; use the named CLI option"))
        exp_name = task or ("locomanilab_pi05_" + time.strftime("%Y%m%d_%H%M%S"))
        output_root = self.ctx.anchor(output_dir) if output_dir else (
            self.ctx.path("pi05_outputs") or self.ctx.workspace_root / "checkpoints" / "pi05")
        extra_env = self._openpi_env()
        extra_env["HF_LEROBOT_HOME"] = str(home)
        for key, value in (tr.get("fsdp_env") or {}).items():
            extra_env.setdefault(str(key), str(value))
        gpus = self.resolve_gpus(gpus, gpu_ids, extra_env)
        self.raise_nofile(model="pi05")
        command = [tr.get("jax_entrypoint", "scripts/train.py"), config_name,
                   "--exp-name", exp_name, "--data.repo-id", repo_id,
                   "--data.assets.assets-dir", str(assets_base / config_name),
                   "--data.assets.asset-id", repo_id, "--assets-base-dir", str(assets_base),
                   "--weight-loader.params-path", str(self.ctx.input_path(base_model)),
                   "--checkpoint-base-dir", str(output_root)]
        if gpus > 1:
            command += ["--fsdp-devices", str(gpus)]
        for value in overrides or []:
            command.append(value if value.startswith("--") else "--" + value)
        env = self.manifest.env("train")
        result = self.runner.run(command, cwd=self.repo(self.manifest.repo_key), env_kind=env.kind,
                                 env_name=env.name, setup=self.env_setup(env.kind),
                                 extra_env=extra_env, stage="train")
        result.artifacts["run_dir"] = output_root / config_name / exp_name
        result.message += f"; dataset={dataset}; recipe={config_name}"
        return result

    # ---- deploy --------------------------------------------------------
    def deploy(self, *, model_path: str | None = None, run_dir: str | None = None,
               prompt: str | None = None, port: int = 5550, config_name: str | None = None,
               native: bool = False, **opts: Any) -> StageResult:
        """Start a gr00t-compatible policy server (:5550/get_action) for a pi05 checkpoint,
        so the real G1 loop and the tele sim client drive it like any other model.
        Pass native=True for openpi's own websocket server (:8000) instead."""
        dep = self.manifest.deploy
        ckpt = self.resolve_ckpt(model_path, run_dir, stage="deploy", hint=self.CKPT_HINT)
        if isinstance(ckpt, StageResult):
            return ckpt
        ckpt = str(ckpt)
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("deploy")
        cfg = config_name or self.manifest.train.get("config_name")

        # env_kind="uv" -> runner adds `uv run --no-sync`; don't repeat it in cmd.
        if native:
            cmd = [dep.get("websocket_server", "scripts/serve_policy.py"),
                   "policy:checkpoint", f"--policy.config={cfg}", f"--policy.dir={ckpt}"]
        else:
            cmd = self._gr00t_server_cmd(ckpt, port, prompt, cfg)
        return self.runner.run(cmd, cwd=repo, env_kind=env.kind, env_name=env.name,
                               setup=self.env_setup(env.kind), extra_env=self._openpi_env(), stage="deploy")

    # ---- sim harness (eval + infer-check share this) --------------------
    CKPT_HINT = "--model-path (checkpoint dir, e.g. checkpoints/<config>/<exp>/20000)"

    def _sim_server(self, *, stage: str, prompt: str | None, port: int,
                    model_path: str | None = None, run_dir: str | None = None,
                    config_name: str | None = None, **opts: Any) -> SimServer | StageResult:
        """openpi gr00t-compat server (uv). pi0.5 emits latent motion_token (like
        fastwam/starVLA/PRTS)."""
        ckpt = self.resolve_ckpt(model_path, run_dir, stage=stage, hint=self.CKPT_HINT)
        if isinstance(ckpt, StageResult):
            return ckpt
        prompt = prompt or self.manifest.deploy.get("prompt_fixed")
        cfg = config_name or self.manifest.train.get("config_name")
        return SimServer(argv=self._server_argv(str(ckpt), port, prompt, cfg),
                         cwd=self.repo(self.manifest.repo_key), prompt=prompt)

    # ---- deploy helpers ------------------------------------------------
    def _gr00t_server_cmd(self, ckpt: str, port: int, prompt: str | None,
                          config_name: str | None) -> list[str]:
        dep = self.manifest.deploy
        server = dep.get("gr00t_server", "deployment/run_openpi_server.py")
        cmd = ["python", server, "--config-name", str(config_name),
               "--ckpt-path", str(ckpt), "--port", str(port)]
        prompt = prompt or dep.get("prompt_fixed")
        if prompt:
            cmd += ["--initial-prompt", str(prompt)]
        return cmd

    def _server_argv(self, ckpt: str, port: int, prompt: str | None,
                     config_name: str | None) -> list[str]:
        """Env-wrapped server for the sim harness (Popen'd directly, not via runner.run).
        openpi runs under uv (`cd` into its project first), and HF_LEROBOT_HOME must point
        at the dataset root so the config's repo_id resolves (openpi builds LeRobotDataset
        without root=). Same env the other stages get (asset cache + dataset root): this
        is Popen'd directly, so nothing else applies extra_env for us -- and without
        OPENPI_DATA_HOME the server dies fetching the tokenizer from googleapis, which
        aiohttp tries without the proxy."""
        from locomanilab.runner import export_line
        repo = self.repo(self.manifest.repo_key)
        pre = [export_line(k, v) for k, v in sorted(self._openpi_env().items())]
        return self.server_argv(self._gr00t_server_cmd(ckpt, port, prompt, config_name),
                                env_kind="uv", cwd=repo, pre=pre)
