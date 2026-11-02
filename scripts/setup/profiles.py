#!/usr/bin/env python3
"""Plan optional code checkouts using committed submodule gitlinks as the lock.

Only --apply can initialize missing checkouts. Existing checkouts are never reset,
updated, cleaned, or removed. Model environments and binary assets are out of scope.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def profiles(root: Path = ROOT) -> dict[str, list[str]]:
    data = json.loads((root / "dependencies/profiles.json").read_text())
    if data.get("schema_version") != 1 or data.get("revision_source") != "gitlink":
        raise ValueError("unsupported dependency profile schema")
    return data["profiles"]


def declarations(root: Path = ROOT) -> dict[str, str]:
    config = configparser.ConfigParser(interpolation=None)
    config.read(root / ".gitmodules")
    return {config[s]["path"]: config[s]["url"] for s in config.sections()
            if s.startswith("submodule ")}


def git(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False)


def locked_revision(path: str, root: Path = ROOT) -> str:
    result = git("ls-files", "--stage", "--", path, cwd=root)
    fields = result.stdout.strip().split()
    if result.returncode or len(fields) < 4 or fields[0] != "160000":
        raise ValueError(f"{path}: missing gitlink; use a Git source checkout")
    return fields[1]


def plan(name: str, root: Path = ROOT) -> list[dict]:
    selected = profiles(root)
    if name not in selected:
        raise ValueError(f"unknown profile {name!r}; use --list")
    urls = declarations(root)
    rows = []
    for path in selected[name]:
        if path not in urls or not path.startswith("external/") or ".." in Path(path).parts:
            raise ValueError(f"{path}: not a declared external submodule")
        revision = locked_revision(path, root)
        initialized = (root / path / ".git").exists()
        actual = git("rev-parse", "HEAD", cwd=root / path).stdout.strip() if initialized else None
        rows.append({"path": path, "url": urls[path], "revision": revision,
                     "initialized": initialized, "actual_revision": actual})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", default="core")
    parser.add_argument("--list", action="store_true", help="list code profiles")
    parser.add_argument("--apply", action="store_true", help="initialize missing selected submodules")
    parser.add_argument("--json", action="store_true", help="print the plan as JSON")
    args = parser.parse_args()
    try:
        available = profiles()
        if args.list:
            for name, paths in available.items():
                print(f"{name:18s} {', '.join(paths) or '(no submodules)'}")
            return 0
        rows = plan(args.profile)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps({"profile": args.profile, "dependencies": rows}, indent=2))
    else:
        print(f"Code profile: {args.profile}")
        if not rows:
            print("No submodules required.")
        for row in rows:
            status = "already initialized" if row["initialized"] else "not initialized"
            print(f"  {row['path']} @ {row['revision']} ({status})")
            if row["initialized"] and row["actual_revision"] != row["revision"]:
                print("    Checkout differs from the gitlink. Review it with standard Git; it will not be changed.")
        if rows:
            paths = [row["path"] for row in rows if not row["initialized"]]
            if paths:
                print("Plan: GIT_LFS_SKIP_SMUDGE=1 " + shlex.join(["git", "submodule", "update", "--init", "--", *paths]))
            print("Each repository is fetched from its .gitmodules URL at the recorded commit.")
        print("No weights, SDKs, scene assets, or model environments are installed.")
    if not args.apply:
        return 0
    missing = [row["path"] for row in rows if not row["initialized"]]
    if not missing:
        return 0
    # LFS smudging can download binary model/scene assets while cloning source code.
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    result = subprocess.run(["git", "submodule", "update", "--init", "--", *missing], cwd=ROOT, env=env)
    if result.returncode:
        print("Initialization failed. Check repository access and locked commits; no fallback revision was selected.",
              file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
