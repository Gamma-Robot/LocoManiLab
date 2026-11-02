"""Plugin registry: pair each configs/<model>.yaml manifest with its models/<model>.py
plugin class (registered via @register)."""
from __future__ import annotations

import importlib
import pkgutil

from locomanilab import models as models_pkg
from locomanilab.base import REGISTRY, ModelPlugin
from locomanilab.config import Context, ModelManifest, load_manifests
from locomanilab.runner import Runner


def _import_all_plugins() -> None:
    for mod in pkgutil.iter_modules(models_pkg.__path__):
        importlib.import_module(f"{models_pkg.__name__}.{mod.name}")


def available_models() -> dict[str, ModelManifest]:
    return load_manifests()


def build_plugin(name: str, ctx: Context, *, dry_run: bool = False) -> ModelPlugin:
    _import_all_plugins()
    manifests = load_manifests()
    if name not in manifests:
        raise KeyError(f"no manifest configs/{name}.yaml (known: {list(manifests)})")
    if name not in REGISTRY:
        raise KeyError(f"no plugin registered for '{name}' (models/{name}.py must @register)")
    manifest = manifests[name]
    # env.local can override the conda env name for this model (machine-specific).
    override = ctx.conda_envs.get(name)
    if override:
        for spec in manifest.envs.values():
            if spec.kind == "conda":
                spec.name = override
    runner = Runner(conda_root=ctx.conda_root, log_root=ctx.log_root, dry_run=dry_run, model=name)
    return REGISTRY[name](manifest, ctx, runner)
