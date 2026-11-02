"""Targeted, read-only readiness checks. No downloads or model imports."""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from locomanilab.config import ConfigError, Context, EnvSpec, load_manifests
from locomanilab.runner import default_conda_roots


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def controller_manifest(ctx: Context, *, backend: str | None = None,
                        sonic_ckpt: str | None = None,
                        policy_config: str | Path | None = None) -> Path | None:
    """Resolve a selected manifest without requiring its local resources to exist."""
    selected_backend = backend or "sonic"
    if selected_backend not in {"sonic", "amo", "heft", "scalebfm", "almi"}:
        raise ConfigError(f"unknown backend {selected_backend!r}; choose sonic, amo, heft, scalebfm, or almi")
    if sonic_ckpt is not None:
        if policy_config is not None:
            raise ConfigError("pass either --sonic-ckpt or --policy-config, not both")
        if selected_backend != "sonic":
            raise ConfigError(f"--sonic-ckpt requires --backend sonic, not {selected_backend!r}")
        from locomanilab.data.variants import canonical_sonic_ckpt
        try:
            variant = canonical_sonic_ckpt(sonic_ckpt)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        return ctx.repo_path("g1_backend") / "checkpoints" / "sonic" / variant / "g1/policy.yaml"
    return ctx.input_path(policy_config) if policy_config is not None else None


def diagnose(ctx: Context, *, model: str | None = None, stage: str = "core",
             sim: bool = False, backend: str | None = None, sonic_ckpt: str | None = None,
             policy_config: str | Path | None = None, sim_env: str | None = None) -> dict:
    checks = [Check("python", sys.version_info >= (3, 10), sys.version.split()[0]),
              Check("workspace", ctx.workspace_root.is_dir(), str(ctx.workspace_root),
                    "Create the directory with `lab init <directory>`."),
              Check("configuration", True, str(ctx.source) if ctx.source else "automatic local defaults")]
    manifests = load_manifests()
    if stage not in {"core", "data", "collect", "train", "deploy", "eval"}:
        raise ConfigError("stage must be core, data, collect, train, deploy, or eval")
    if model and model not in manifests:
        raise ConfigError(f"unknown model {model!r}; run `lab list-models`")
    if stage in {"train", "deploy", "eval"} and not model:
        raise ConfigError(f"--stage {stage} requires --model; use --stage core or data for standalone tools")
    if not sim and any(value is not None for value in (backend, sonic_ckpt, policy_config)):
        raise ConfigError("--backend, --sonic-ckpt, and --policy-config require --sim")
    if stage == "data":
        for module in ("pandas", "pyarrow", "av"):
            checks.append(Check(module, importlib.util.find_spec(module) is not None,
                                "installed" if importlib.util.find_spec(module) else "not installed",
                                'Install data tools with `python -m pip install -e ".[data]"`.'))
    keys = []
    if model and stage != "collect":
        keys.append(manifests[model].repo_key)
        if stage == "deploy" and model in {"gr00t", "fastwam"}:
            keys.append("gr00t_wbc")
    if stage == "collect":
        keys.append("g1_backend")
    if sim:
        keys.extend(("sim_repo", "g1_backend"))
    for key in dict.fromkeys(keys):
        try:
            path = ctx.sim_repo() if key == "sim_repo" else ctx.repo_path(key)
            present = path.is_dir() and any(path.iterdir())
            checks.append(Check(f"repository:{key}", present, str(path),
                                f"Initialize the selected backend, or set repos.{key} to an existing checkout."))
        except (ConfigError, OSError) as exc:
            checks.append(Check(f"repository:{key}", False, str(exc)))
    if model and stage in {"train", "deploy", "eval"}:
        manifest = manifests[model]
        spec = manifest.env("train" if stage == "train" else "deploy")
        if model == "gr00t" and stage == "eval":
            spec = EnvSpec(kind="uv")
        repo = ctx.repo_path(manifest.repo_key)
        if spec.kind == "uv":
            checks.append(Check("uv", shutil.which("uv") is not None, "uv executable", "Install uv."))
            interpreter = repo / ".venv/bin/python"
            checks.append(Check("backend environment", interpreter.is_file(), str(interpreter),
                                "Follow the selected model's environment setup instructions."))
        elif spec.kind == "conda":
            name = ctx.conda_envs.get(model) or spec.name
            roots = [Path(ctx.conda_root)] if ctx.conda_root else default_conda_roots()
            candidates = [root / "envs" / str(name) / "bin/python" for root in roots]
            candidates += [Path(root) / str(name) / "bin/python"
                           for root in os.environ.get("CONDA_ENVS_PATH", "").split(os.pathsep) if root]
            if os.environ.get("CONDA_DEFAULT_ENV") == name and os.environ.get("CONDA_PREFIX"):
                candidates.append(Path(os.environ["CONDA_PREFIX"]) / "bin/python")
            checks.append(Check("backend environment", any(item.is_file() for item in candidates),
                                f"conda environment {name!r}", "Configure conda_root/conda_envs or install this environment."))
        elif spec.kind == "venv":
            interpreter = (ctx.repo_path("gr00t_wbc") / ".venv_inference/bin/python"
                           if model == "gr00t" else repo / ".venv/bin/python")
            checks.append(Check("backend environment", interpreter.is_file(), str(interpreter),
                                "Follow the selected model's environment setup instructions."))
    if sim:
        from locomanilab.common.sim import detect
        selected = controller_manifest(ctx, backend=backend, sonic_ckpt=sonic_ckpt, policy_config=policy_config)
        options = {}
        if backend is not None:
            options["backend"] = backend
        if selected is not None:
            options["policy_config"] = selected
        if sim_env is not None:
            options["conda_env"] = sim_env
        support = detect(ctx, **options)
        checks.extend(Check(name, ok, detail) for name, ok, detail in support.checks)
    return {"ok": all(check.ok for check in checks), "model": model, "stage": stage,
            "workspace": str(ctx.workspace_root), "checks": [asdict(check) for check in checks],
            "scope": "Paths and selected local prerequisites; no training or inference is performed."}
