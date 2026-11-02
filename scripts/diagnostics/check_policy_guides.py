#!/usr/bin/env python3
"""Replay policy README CPU examples in a temporary copy of the tracked source.

Install the main package with its data extra before running this check. Only the
explicit cpu-smoke block is executed; model installation/training examples are
outside that block. No backend checkout or existing user workspace is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CPU_BLOCK = re.compile(
    r"<!-- cpu-smoke:start -->\s*```bash\n(.*?)\n```\s*<!-- cpu-smoke:end -->", re.DOTALL
)


def _git(root: Path, *args: str) -> bytes:
    # Ambient Git settings must not redirect the copy into another repository.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0")
    return subprocess.check_output(["git", "-C", str(root), *args], env=env, stderr=subprocess.PIPE)


def gitlinks(root: Path) -> dict[str, str]:
    """Map each submodule path in the index to its pinned commit."""
    links = {}
    for entry in _git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if entry.startswith(b"160000 "):
            metadata, path = entry.split(b"\t", 1)
            links[os.fsdecode(path)] = metadata.split()[1].decode("ascii")
    return links


def stage_source(root: Path, destination: Path) -> None:
    """Copy the tracked working-tree files into a new repository with the same gitlinks."""
    links = gitlinks(root)
    for name in _git(root, "ls-files", "-z").decode().split("\0"):
        if name and name not in links:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target, follow_symlinks=False)
    _git(destination, "init", "--quiet", "--template=", "-b", "main")
    _git(destination, "add", "--all", "--force")
    for path, commit in links.items():
        # Git represents an uninitialized submodule as an empty directory.
        (destination / path).mkdir(parents=True, exist_ok=True)
        _git(destination, "update-index", "--add", "--cacheinfo", f"160000,{commit},{path}")


@contextmanager
def _configuration_environment(environment: dict[str, str]):
    previous = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(environment)
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def check_workspace(checkout: Path, model: str, environment: dict[str, str]) -> None:
    """Validate only the guide's copied example, never a user's configuration."""
    from locomanilab.config import Context, ModelManifest, load_context

    workspace = checkout / "workspaces" / model
    configuration = workspace / "env.local.yaml"
    example = checkout / "policies" / model / "workspace.example.yaml"
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError(f"expected a new directory at workspaces/{model}")
    if configuration.read_bytes() != example.read_bytes():
        raise ValueError(f"workspaces/{model}/env.local.yaml differs from its guide example")
    manifest = ModelManifest.from_file(checkout / "src/locomanilab/configs" / f"{model}.yaml")
    if manifest.name != model:
        raise ValueError(f"{model}: model manifest name differs from the guide ID")
    links = gitlinks(checkout)
    with _configuration_environment(environment):
        context = load_context(configuration)
        if context.schema_version != 2:
            raise ValueError(f"{model}: workspace example must use schema_version: 2")
        if context.workspace_root.resolve() != workspace.resolve():
            raise ValueError(f"{model}: configuration resolves to a different workspace")
        if not context.repos.get(manifest.repo_key):
            raise ValueError(f"{model}: example must explicitly configure repos.{manifest.repo_key}")
        defaults = Context(workspace=checkout)
        for key, value in context.repos.items():
            resolved = context.repo_path(key).resolve()
            expected = defaults.repo_path(key).resolve()
            relative = expected.relative_to(checkout.resolve()).as_posix()
            if not value.strip() or resolved != expected or relative not in links:
                raise ValueError(f"{model}: repos.{key} must select this checkout's declared gitlink")
        for key, value in context.paths.items():
            if value:
                context.path(key).resolve()
        model_repository = context.repo_path(manifest.repo_key).resolve()
    # This is the real setup helper's read-only plan, using the copied index.
    result = subprocess.run([sys.executable, str(checkout / "scripts/setup/profiles.py"), model, "--json"],
                            cwd=checkout, env=environment, text=True, capture_output=True,
                            check=True, timeout=30)
    profile = json.loads(result.stdout)
    if profile["profile"] != model or model_repository not in {
        (checkout / row["path"]).resolve() for row in profile["dependencies"]
    }:
        raise ValueError(f"{model}: source profile does not cover the model repository")
    readme = (checkout / "policies" / model / "README.md").read_text()
    if not re.search(r"scripts/setup/profiles\.py\s+" + re.escape(model) + r"(?=[\s`])", readme):
        raise ValueError(f"{model}: README must name its source profile command")
    for row in profile["dependencies"]:
        revision = row["revision"]
        if links.get(row["path"]) != revision:
            raise ValueError(f"{model}: profile revision differs from the checkout's gitlink")
        if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.search(
            r"(?<![0-9a-f])" + re.escape(revision) + r"(?![0-9a-f])", readme
        ):
            raise ValueError(f"{model}: README must contain the complete fixed SHA for {row['path']}")


def run_cpu_block(block: str, checkout: Path, environment: dict[str, str], *,
                  timeout: float = 120) -> subprocess.CompletedProcess:
    from locomanilab.processes import managed_process, stop_process_group

    argv = ["bash", "-euo", "pipefail", "-c", block]
    with subprocess.Popen(argv, cwd=checkout, env=environment, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          start_new_session=True) as process, managed_process(process):
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_process_group(process)
            stdout, stderr = process.communicate(timeout=5)
            raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr) from None
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def workspace_snapshot(workspace: Path) -> dict:
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("existing workspace was removed or replaced")
    result = {}
    for path in workspace.rglob("*"):
        if path.is_symlink():
            value = ("symlink", os.readlink(path))
        elif path.is_dir():
            value = ("directory",)
        elif path.is_file():
            value = ("file", hashlib.sha256(path.read_bytes()).hexdigest())
        else:
            raise ValueError("workspace contains an unsupported file type")
        result[path.relative_to(workspace).as_posix()] = value
    return result


def check_repeat(block: str, checkout: Path, model: str, environment: dict[str, str]) -> None:
    workspace = checkout / "workspaces" / model
    before = workspace_snapshot(workspace)
    result = run_cpu_block(block, checkout, environment)
    if result.returncode == 0:
        raise ValueError(f"{model}: CPU example must refuse an existing workspace")
    if workspace_snapshot(workspace) != before:
        raise ValueError(f"{model}: repeated CPU example changed the existing workspace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="Policy IDs; defaults to all policy guides")
    args = parser.parse_args()
    models = args.models or sorted(path.parent.name for path in (ROOT / "policies").glob("*/README.md"))
    if not models:
        parser.error("no policy guides found")
    blocks = {}
    for model in models:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", model):
            parser.error(f"invalid policy ID: {model!r}")
        path = ROOT / "policies" / model / "README.md"
        if not path.is_file():
            parser.error(f"missing policies/{model}/README.md")
        matches = CPU_BLOCK.findall(path.read_text())
        if len(matches) != 1:
            parser.error(f"policies/{model}/README.md must contain one marked bash CPU example")
        blocks[model] = matches[0]
    executable_directory = Path(sys.executable).parent
    if not (executable_directory / "lab").is_file():
        parser.error('install this project with its data extra in the current Python environment first')

    environment = {
        key: value for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"}
        and not key.startswith(("LOCOMANILAB_", "SOMA_", "GIT_"))
    }
    environment.update(
        PATH=str(executable_directory) + os.pathsep + os.environ.get("PATH", ""),
        CUDA_VISIBLE_DEVICES="", UV_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1", GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0",
    )
    with tempfile.TemporaryDirectory(prefix="locomanilab-policy-guides-") as directory:
        checkout = Path(directory) / "source"
        stage_source(ROOT, checkout)
        for model, block in blocks.items():
            syntax = subprocess.run(["bash", "-n"], input=block, text=True, capture_output=True)
            if syntax.returncode:
                print(f"FAIL {model}: {syntax.stderr}", file=sys.stderr)
                return 1
            try:
                result = run_cpu_block(block, checkout, environment)
            except subprocess.TimeoutExpired:
                print(f"FAIL {model}: CPU example exceeded 120 seconds", file=sys.stderr)
                return 1
            if result.returncode:
                print(f"FAIL {model}\n{result.stdout}\n{result.stderr}", file=sys.stderr)
                return 1
            try:
                check_workspace(checkout, model, environment)
                check_repeat(block, checkout, model, environment)
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                print(f"FAIL {model}: {exc}", file=sys.stderr)
                return 1
            print(f"PASS {model}: README CPU example, configuration/source paths, and repeat protection")
    print(f"{len(blocks)} policy CPU examples passed; no model training or inference was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
