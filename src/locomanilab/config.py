"""Model manifests and local workspace configuration; never import model code."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from locomanilab.workspace import absolute_path, source_root

PKG_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = PKG_DIR / "configs"
SOURCE_ROOT = source_root(PKG_DIR)


class ConfigError(RuntimeError):
    """An actionable configuration error suitable for display by the CLI."""


def _checked_path(value, base: Path | None = None) -> Path:
    try:
        return absolute_path(value, base)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _mapping(value: Any, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a YAML mapping")
    return value


@dataclass
class EnvSpec:
    kind: str = "none"
    name: str | None = None
    venv_path: str | None = None

    @classmethod
    def parse(cls, value: dict | None) -> "EnvSpec":
        value = _mapping(value, "environment")
        kind = value.get("kind", "none")
        if kind not in {"none", "conda", "uv", "venv"}:
            raise ConfigError(f"unknown environment kind {kind!r}; use none, conda, uv, or venv")
        return cls(kind=kind, name=value.get("name"), venv_path=value.get("venv_path"))


@dataclass
class ModelManifest:
    name: str
    embodiment: str = ""
    repo_key: str = ""
    action: dict[str, Any] = field(default_factory=dict)
    envs: dict[str, EnvSpec] = field(default_factory=dict)
    convert: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    deploy: dict[str, Any] = field(default_factory=dict)
    action_spaces: dict[str, Any] = field(default_factory=dict)
    default_action_space: str = "motion_token"
    raw: dict[str, Any] = field(default_factory=dict)

    def env(self, stage: str) -> EnvSpec:
        return self.envs.get(stage, EnvSpec())

    def action_space(self, name: str | None) -> dict[str, Any]:
        name = name or self.default_action_space
        if name not in self.action_spaces:
            raise ConfigError(f"{self.name}: action space {name!r} is not declared; "
                              f"choose from {list(self.action_spaces) or '[none]'}")
        return self.action_spaces[name] or {}

    @classmethod
    def from_file(cls, path: Path) -> "ModelManifest":
        try:
            value = _mapping(yaml.safe_load(path.read_text()), str(path))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read model manifest {path}: {exc}") from exc
        if not isinstance(value.get("name"), str) or not value["name"]:
            raise ConfigError(f"{path}: missing model name")
        envs = {key: EnvSpec.parse(spec) for key, spec in _mapping(value.get("envs"), "envs").items()}
        return cls(name=value["name"], embodiment=value.get("embodiment", ""),
                   repo_key=value.get("repo_key", value["name"]),
                   action=_mapping(value.get("action"), "action"), envs=envs,
                   convert=_mapping(value.get("convert"), "convert"),
                   train=_mapping(value.get("train"), "train"),
                   deploy=_mapping(value.get("deploy"), "deploy"),
                   action_spaces=_mapping(value.get("action_spaces"), "action_spaces"),
                   default_action_space=value.get("default_action_space", "motion_token"), raw=value)


def load_manifests() -> dict[str, ModelManifest]:
    result: dict[str, ModelManifest] = {}
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        manifest = ModelManifest.from_file(path)
        if manifest.name in result:
            raise ConfigError(f"duplicate model name {manifest.name!r} in {path}")
        result[manifest.name] = manifest
    return result


@dataclass
class Context:
    role: list[str] = field(default_factory=list)
    conda_root: str | None = None
    repos: dict[str, str] = field(default_factory=dict)
    paths: dict[str, str] = field(default_factory=dict)
    deploy: dict[str, Any] = field(default_factory=dict)
    conda_envs: dict[str, str] = field(default_factory=dict)
    source: Path | None = None
    schema_version: int = 2
    workspace: Path | None = None

    _REPO_DEFAULTS = {
        "fastwam": "external/FastWAM", "isaac_gr00t": "external/Isaac-GR00T",
        "gr00t_wbc": "external/GR00T-WholeBodyControl", "starvla": "external/starVLA",
        "sim_repo": "external/humanoid_benchmark", "g1_backend": "external/g1_backend",
        "xr1": "external/Xiaomi-Robotics-1", "opendm": "external/opendm",
        "openpi": "external/openpi", "prts": "external/PRTS", "psi0": "external/Psi0",
        "dreamzero": "external/dreamzero", "internvla_a15": "external/InternVLA-A1.5",
        "lerobot": "external/lerobot",
    }

    @property
    def workspace_root(self) -> Path:
        if self.workspace is not None:
            return _checked_path(self.workspace)
        return _checked_path(self.source.parent if self.source else Path.cwd())

    @property
    def config_root(self) -> Path:
        """Compatibility name for the workspace where outputs are written."""
        return self.workspace_root

    @property
    def backend_root(self) -> Path:
        return self._path_base / "external"

    @property
    def _path_base(self) -> Path:
        # Unversioned configurations historically anchored paths at the source tree.
        # Retain that convention where the source tree is actually available.
        return SOURCE_ROOT if self.schema_version == 1 and SOURCE_ROOT else self.workspace_root

    def _resolve(self, value: str | Path) -> Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"expected a filesystem path, got {type(value).__name__}")
        expanded = os.path.expandvars(os.path.expanduser(str(value)))
        if re.search(r"\$(?:\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)", expanded):
            raise ConfigError(f"unset environment variable in path {value!r}; set it or use a local path")
        if re.search(r"<(?:name|run|path|your[^>]*)>", expanded, re.IGNORECASE):
            raise ConfigError(f"replace the placeholder in path {value!r}")
        return _checked_path(expanded, self._path_base)

    def anchor(self, path: Path | str) -> Path:
        return _checked_path(path, self.workspace_root)

    def input_path(self, path: Path | str) -> Path:
        """CLI inputs are relative to the caller; symlink names are retained."""
        return _checked_path(path)

    @property
    def log_root(self) -> Path:
        return self.path("logs") or self.workspace_root / "runs"

    def repo_path(self, key: str) -> Path:
        value = self.repos.get(key) or self._REPO_DEFAULTS.get(key)
        if value is None:
            raise ConfigError(f"repository {key!r} is not configured in repos")
        return self._resolve(value)

    def path(self, key: str, default: str | None = None) -> Path | None:
        value = self.paths.get(key, default)
        return self._resolve(value) if value else None

    def data_env(self) -> tuple[str, str | None, str | None]:
        name = self.conda_envs.get("data")
        if name:
            return "conda", name, None
        venv = self.paths.get("data_venv")
        if venv:
            return "venv", None, str(self._resolve(venv))
        return "none", None, None

    def sim_repo(self) -> Path:
        legacy = self.paths.get("sim_repo")
        return self._resolve(legacy) if legacy else self.repo_path("sim_repo")


def find_env_local(start: Path | None = None) -> Path | None:
    current = _checked_path(start or Path.cwd())
    for parent in (current, *current.parents):
        candidate = parent / "env.local.yaml"
        if candidate.is_file():
            return candidate
    if SOURCE_ROOT and (SOURCE_ROOT / "env.local.yaml").is_file():
        return SOURCE_ROOT / "env.local.yaml"
    return None


def load_context(path: Path | None = None, *, workspace: Path | None = None,
                 required: bool = True) -> Context:
    # The former project name remains accepted for existing local configurations.
    explicit = path or os.environ.get("LOCOMANILAB_CONFIG") or os.environ.get("SOMA_CONFIG")
    workspace_value = workspace or os.environ.get("LOCOMANILAB_WORKSPACE") or os.environ.get("SOMA_WORKSPACE")
    workspace_path = _checked_path(workspace_value) if workspace_value else None
    if explicit:
        path = _checked_path(explicit)
    elif workspace_path is not None:
        candidate = workspace_path / "env.local.yaml"
        path = candidate if candidate.is_file() else None
    else:
        path = find_env_local()
    if path is None:
        if required:
            raise ConfigError("No env.local.yaml found. Run `lab init <directory>` or use --config PATH.")
        return Context(workspace=workspace_path)
    try:
        value = _mapping(yaml.safe_load(path.read_text()), str(path))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read configuration {path}: {exc}") from exc
    version = value.get("schema_version", 1)
    if type(version) is not int or version not in (1, 2):
        raise ConfigError(f"{path}: unsupported schema_version {version!r}; supported versions are 1 and 2")
    role = value.get("role", [])
    if isinstance(role, str):
        role = [role]
    if not isinstance(role, list) or not all(isinstance(item, str) for item in role):
        raise ConfigError("role must be a string or a list of strings")
    mappings = {name: _mapping(value.get(name), name) for name in ("repos", "paths", "deploy", "conda_envs")}
    for name in ("repos", "paths", "conda_envs"):
        if not all(isinstance(key, str) and isinstance(item, str) for key, item in mappings[name].items()):
            raise ConfigError(f"{name} entries must map names to strings")
    if workspace_path is None and value.get("workspace_root") is not None:
        workspace_path = _checked_path(value["workspace_root"], path.parent)
    conda_root = value.get("conda_root")
    if conda_root is not None:
        if not isinstance(conda_root, str):
            raise ConfigError("conda_root must be a filesystem path string")
        conda_root = str(_checked_path(conda_root, path.parent))
    return Context(role=role, conda_root=conda_root, source=path, schema_version=version,
                   workspace=workspace_path, **mappings)
