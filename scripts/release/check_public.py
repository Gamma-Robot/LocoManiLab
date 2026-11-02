#!/usr/bin/env python3
"""Check the public source tree for private infrastructure markers and credentials."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PUBLIC_DIRECTORIES = ("src", "tests", "docs", "policies", "scripts", "dependencies", "examples", ".github")
PUBLIC_FILES = ("README.md", "CONTRIBUTING.md", "DESIGN.md", "CHANGELOG.md", "AGENTS.md", "THIRD_PARTY.md",
                "RELEASING.md", "LICENSE", "NOTICE", "pyproject.toml", "uv.lock", ".gitignore", ".gitmodules",
                "env.local.yaml.example", "MANIFEST.in")
PATTERNS = {
    "private infrastructure path": re.compile(r"/data_[A-Za-z0-9_.-]+/|/(?:home|Users)/[A-Za-z0-9_.-]+/"),
    "private key": re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    "GitHub access token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{50,}\b"),
    "OpenAI access token": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b"),
}
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", "node_modules", ".venv"}
ASSET_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine", ".parquet", ".mp4", ".npz"}
LOCAL_FILES = {".env", "env.local.yaml", "env.local.yml", ".netrc", ".pypirc"}
MAX_FILE_BYTES = 20 * 1024 * 1024


def public_paths(root: Path):
    for name in PUBLIC_FILES:
        path = root / name
        if path.exists() or path.is_symlink():
            yield path
    for name in PUBLIC_DIRECTORIES:
        directory = root / name
        # rglob on a symlinked root can enumerate files outside this repository.
        if directory.is_symlink():
            yield directory
            continue
        if directory.is_dir():
            for path in sorted(directory.rglob("*")):
                if not any(part in SKIP_PARTS or part.startswith(".venv") or
                           part.endswith((".egg-info", ".dist-info")) for part in path.relative_to(root).parts):
                    if path.is_file() or path.is_symlink():
                        yield path


def check(root: Path) -> list[str]:
    if not root.is_dir():
        return ["source root is not a directory"]
    findings = []
    for path in public_paths(root):
        relative = path.relative_to(root)
        if path.is_symlink():
            findings.append(f"{relative}: symlinks must be resolved explicitly before export")
            continue
        if not path.is_file():
            findings.append(f"{relative}: expected a regular source file")
            continue
        if path.name in LOCAL_FILES or path.suffix.lower() in ASSET_SUFFIXES:
            findings.append(f"{relative}: local configuration or runtime asset is outside the public source scope")
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            findings.append(f"{relative}: unexpected file over 20 MiB")
            continue
        content = path.read_bytes()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            findings.append(f"{relative}: binary content requires explicit public-source review")
            continue
        if "\0" in text:
            findings.append(f"{relative}: binary content requires explicit public-source review")
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for description, pattern in PATTERNS.items():
                if pattern.search(line):
                    # Report locations, never print a potentially sensitive value.
                    findings.append(f"{relative}:{number}: {description}")
    for name in ("CLAUDE.md", ".claude"):
        if (root / name).exists():
            findings.append(f"{name}: internal assistant material is outside the public source scope")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        findings = check(args.root.resolve())
    except OSError:
        # Avoid echoing arbitrary source contents through exception messages.
        print("Could not read the public source tree; no export was created.")
        return 1
    if findings:
        for finding in findings:
            print(finding)
        return 1
    print("Public source checks passed (working tree only; Git history is not checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
