#!/usr/bin/env python3
"""Download LocoManiLab release resources from the Hugging Face Hub.

The release lives in one dataset repository (default ``GammaRobotics/LocoManiLab``,
override with ``--repo`` or ``LOCOMANILAB_HF_REPO``):

    asset/        manifest.tsv.gz (every asset file: sha256 + source), tasks.json.gz
                  (task -> files), local.tar.gz (the few files hosted here; see NOTICE.md)
    dataset/      LeRobot v2.1 benchmark data (main/, data_generation/, ...)
    checkpoint/   gr00t/, dm05/, xr1/ inference checkpoints; controller/ G1 whole-body controller
                  weights (fetched by humanoid_benchmark's download_g1_checkpoints.py)

Usage (requires ``huggingface_hub``; install with ``pip install -e '.[hf]'``):

    python scripts/hf/download.py list
    python scripts/hf/download.py dataset --task IThorKitchenOpenFridge_scene13 --out datasets/hf
    python scripts/hf/download.py dataset --group main/atomic/seen --out datasets/hf
    python scripts/hf/download.py assets --task IThorKitchenOpenFridge_scene13 \\
        --assets-dir external/humanoid_benchmark/robocasa/models/assets
    python scripts/hf/download.py checkpoint gr00t --out checkpoints/hf

``assets`` fetches only the files the selected tasks load, each from its original host
(see upstream.py), and verifies every file's sha256. Paths are relative to the simulator's
``robocasa/models/assets`` directory.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
from pathlib import Path

DEFAULT_REPO = "GammaRobotics/LocoManiLab"
REPO_TYPE = "dataset"
CHECKPOINTS = ("gr00t", "dm05", "xr1")
# Asset presets select tasks by name prefix; "all" selects every task in asset/tasks.json.gz.
PRESET_PREFIXES = {
    "ithor": ("IThor",),
    "holodeck": ("Holodeck",),
    "telelab": ("TeleLab",),
}
WEIGHT_PATTERNS = ["*.safetensors", "*.pt", "*.bin"]


def _hub():
    try:
        import huggingface_hub
    except ImportError:
        sys.exit("huggingface_hub is required: pip install -e '.[hf]' (or pip install huggingface_hub)")
    return huggingface_hub


def _repo(args) -> str:
    return args.repo or os.environ.get("LOCOMANILAB_HF_REPO") or DEFAULT_REPO


def _snapshot(args, allow, ignore=None) -> Path:
    hub = _hub()
    out = Path(args.out).resolve()
    hub.snapshot_download(repo_id=_repo(args), repo_type=REPO_TYPE, revision=args.revision,
                          allow_patterns=allow, ignore_patterns=ignore, local_dir=out)
    return out


def cmd_list(args) -> int:
    api = _hub().HfApi()
    repo = _repo(args)

    def tree(path, recursive=False):
        return list(api.list_repo_tree(repo, repo_type=REPO_TYPE, revision=args.revision,
                                       path_in_repo=path, recursive=recursive))

    print(f"repository: {repo} ({REPO_TYPE})")
    print("asset files:")
    for item in tree("asset"):
        size = getattr(item, "size", None)
        if size is not None:
            print(f"  {item.path.split('/', 1)[1]:42s} {size / 1e9:6.2f} GB")
    print("checkpoints:")
    for name in CHECKPOINTS:
        files = [f for f in tree(f"checkpoint/{name}", recursive=True) if getattr(f, "size", None) is not None]
        print(f"  {name:8s} {len(files):3d} files {sum(f.size for f in files) / 1e9:6.2f} GB")
    files = [f for f in tree("checkpoint/controller", recursive=True) if getattr(f, "size", None) is not None]
    print(f"  {'controller':8s} {len(files):3d} files {sum(f.size for f in files) / 1e9:6.2f} GB"
          "  (G1 controller weights; fetch with humanoid_benchmark's download_g1_checkpoints.py)")
    print("dataset groups (use with `dataset --group`):")
    for top in tree("dataset"):
        for sub in tree(top.path):
            if getattr(sub, "size", None) is None:
                print(f"  {sub.path.split('/', 1)[1]}")
    print(f"asset presets: {', '.join(sorted(PRESET_PREFIXES))}, all (or --task / --task-set)")
    return 0


def cmd_dataset(args) -> int:
    group = args.group.strip("/")
    if args.task:
        allow = []
        for task in args.task:
            allow += [f"dataset/{group}/{task}/*", f"dataset/{group}/*/{task}/*"]
    else:
        allow = [f"dataset/{group}/*"]
    out = _snapshot(args, allow)
    found = sorted({p.parent.parent for p in (out / "dataset" / group).rglob("meta/info.json")})
    if not found:
        print(f"no dataset matched group={group!r} task={args.task}; run `list` to see groups", file=sys.stderr)
        return 1
    if args.task:
        missing = [t for t in args.task if not any(p.name == t for p in found)]
        if missing:
            print(f"not found in dataset/{group}: {missing}", file=sys.stderr)
            return 1
    for path in found:
        print(path)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(hub, repo: str, revision):
    """Rows of asset/manifest.tsv.gz and the task -> files map."""
    import gzip
    import json

    manifest = hub.hf_hub_download(repo, "asset/manifest.tsv.gz", repo_type=REPO_TYPE, revision=revision)
    rows = {}
    with gzip.open(manifest, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            path, sha, size, source = line.rstrip("\n").split("\t")
            rows[path] = (sha, int(size), source)
    tasks_file = hub.hf_hub_download(repo, "asset/tasks.json.gz", repo_type=REPO_TYPE, revision=revision)
    with gzip.open(tasks_file, "rt", encoding="utf-8") as fh:
        tasks = json.load(fh)
    return rows, tasks


def _select_files(args, rows, tasks) -> list[str] | None:
    names: list[str] = list(args.task or [])
    if args.task_set:
        from locomanilab.benchmark import load_tasks, select_tasks

        names += [t.name for t in select_tasks(args.task_set, load_tasks())]
    if not names:
        if args.preset == "all":
            names = sorted(tasks)
        else:
            names = sorted(t for t in tasks if t.startswith(PRESET_PREFIXES[args.preset]))
    unknown = [n for n in names if n not in tasks]
    if unknown:
        print(f"no asset list for {unknown}; tasks with assets: `lab tasks` (plus the dataset-only "
              f"tasks {sorted(t for t in tasks if t.startswith('TeleLab'))})", file=sys.stderr)
        return None
    return sorted({path for name in names for path in tasks[name]})


def _present(dest: Path, path: str, sha: str, size: int) -> bool:
    target = dest / path
    return target.is_file() and target.stat().st_size == size and _sha256(target) == sha


def cmd_assets(args) -> int:
    hub = _hub()
    dest = Path(args.assets_dir).resolve()
    if not dest.is_dir():
        print(f"assets dir does not exist: {dest}\n"
              "Point --assets-dir at <humanoid_benchmark>/robocasa/models/assets of an initialized "
              "simulator checkout (python3 scripts/setup/profiles.py sim --apply).", file=sys.stderr)
        return 1
    repo = _repo(args)
    rows, tasks = _load_manifest(hub, repo, args.revision)
    wanted = _select_files(args, rows, tasks)
    if wanted is None:
        return 1

    import upstream

    # --force refetches downloaded files; files of the simulator checkout are never fetched.
    todo = [p for p in wanted
            if (args.force and rows[p][2] != "git") or not _present(dest, p, rows[p][0], rows[p][1])]
    missing_git = [p for p in todo if rows[p][2] == "git"]
    if missing_git:
        print(f"{len(missing_git)} files belong to the simulator checkout but are missing or changed "
              f"(e.g. {missing_git[0]}); check out the pinned humanoid_benchmark commit", file=sys.stderr)
        return 1
    print(f"{len(wanted)} files selected, {len(todo)} to fetch", flush=True)

    groups: dict[tuple, list[str]] = {}
    for path in todo:
        source = rows[path][2]
        kind, _, rest = source.partition(":")
        if kind == "hf":
            key = ("hf", rest)
        elif kind == "zip":
            key = ("zip", rest.split(".zip:", 1)[0] + ".zip")
        elif kind == "molmospaces":
            src, package, _member = rest.split(":", 2)
            key = ("molmospaces", src, package)
        elif kind == "file":
            key = ("file", rest)
        else:
            print(f"unknown source {source!r} for {path}", file=sys.stderr)
            return 1
        groups.setdefault(key, []).append(path)

    sources = {}
    molmospaces = sorted({key[1] for key in groups if key[0] == "molmospaces"})
    if molmospaces:
        try:
            import pyarrow  # noqa: F401
            import zstandard  # noqa: F401
        except ImportError as exc:
            print(f"{exc.name} is required for MolmoSpaces assets: pip install -e '.[hf]'", file=sys.stderr)
            return 1
    for src in molmospaces:
        sources[src] = upstream.MolmoSpacesSource(src, upstream.molmospaces_index(src, hub))
    zips: dict[str, object] = {}

    def member_of(path: str) -> str:
        # Member names may contain ":"; split on the fixed fields before them.
        source = rows[path][2]
        if source.startswith("molmospaces:"):
            return source.split(":", 3)[3]
        return source.split(".zip:", 1)[1]

    def fetch(key, paths):
        kind = key[0]
        if kind == "hf":
            archive = Path(hub.hf_hub_download(repo, f"asset/{key[1]}.tar.gz", repo_type=REPO_TYPE,
                                               revision=args.revision))
            want = set(paths)
            with tarfile.open(archive, "r:gz") as tar:
                for member in tar:
                    if member.isfile() and member.name in want:
                        upstream.write_verified(dest, member.name, tar.extractfile(member).read(),
                                                rows[member.name][0])
                        want.discard(member.name)
            if want:
                raise OSError(f"{len(want)} files missing from shard {key[1]}, e.g. {sorted(want)[0]}")
        elif kind == "zip":
            for path in paths:
                upstream.write_verified(dest, path, zips[key[1]].read(member_of(path)), rows[path][0])
        elif kind == "molmospaces":
            members = sources[key[1]].members(key[2])
            for path in paths:
                upstream.write_verified(dest, path, members[member_of(path)], rows[path][0])
        else:
            upstream.write_verified(dest, paths[0], upstream.fetch_url(key[1]), rows[paths[0]][0])
        return len(paths)

    # Zip members are fetched one file per task so that large archives download in parallel.
    units = []
    for key, paths in groups.items():
        if key[0] == "zip":
            if key[1] not in zips:
                zips[key[1]] = upstream.RemoteZip(key[1])
            units += [(key, [p]) for p in paths]
        else:
            units.append((key, paths))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    done = failed = 0
    errors: list[str] = []
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(fetch, key, paths): (key, paths) for key, paths in units}
        for i, future in enumerate(as_completed(futures), 1):
            key, paths = futures[future]
            try:
                done += future.result()
            except Exception as exc:
                failed += len(paths)
                errors.append(f"{paths[0]}: {exc}")
            if i % 500 == 0:
                print(f"[progress] {done}/{len(todo)} files", flush=True)
    for line in errors[:10]:
        print(f"[fail] {line}", file=sys.stderr)
    if failed:
        print(f"{failed} files failed; rerun the same command to retry them", file=sys.stderr)
        return 1
    print(f"assets ready in {dest}: {len(wanted)} files ({done} fetched, all sha256-verified)")
    return 0


def cmd_checkpoint(args) -> int:
    ignore = WEIGHT_PATTERNS if args.no_weights else None
    out = _snapshot(args, [f"checkpoint/{args.name}/*"], ignore)
    path = out / "checkpoint" / args.name
    if not path.is_dir():
        print(f"checkpoint/{args.name} not found in {_repo(args)}", file=sys.stderr)
        return 1
    print(path)
    if args.name == "gr00t":
        print("note: GR00T loads the gated backbone nvidia/Cosmos-Reason2-2B at startup; accept its "
              "terms on huggingface.co and log in (`hf auth login`) on the serving machine.")
    if args.name == "xr1":
        print("note: serve with `--ckpt-step 50000` (the run directory has no last.ckpt).")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None, help=f"Hub dataset repo (default {DEFAULT_REPO} or $LOCOMANILAB_HF_REPO)")
    ap.add_argument("--revision", default=None, help="branch, tag, or commit of the Hub repo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show shards, checkpoints, and dataset groups").set_defaults(func=cmd_list)

    ds = sub.add_parser("dataset", help="download benchmark data (LeRobot v2.1)")
    ds.add_argument("--group", default="main", help="e.g. main, main/atomic/seen, data_generation/kimodo, real")
    ds.add_argument("--task", action="append", help="task directory name (repeatable); default: whole group")
    ds.add_argument("--out", default="datasets/hf", help="local directory (mirrors the repo layout)")
    ds.set_defaults(func=cmd_dataset)

    at = sub.add_parser("assets", help="fetch the simulator assets a set of tasks needs (sha256-verified)")
    at.add_argument("--assets-dir", required=True, help="<humanoid_benchmark>/robocasa/models/assets")
    group = at.add_mutually_exclusive_group()
    group.add_argument("--preset", default="ithor", choices=[*PRESET_PREFIXES, "all"],
                       help="task family (default ithor); all = every task")
    group.add_argument("--task", action="append", help="benchmark task name (repeatable), e.g. IThorKitchenOpenFridge_scene13")
    group.add_argument("--task-set", help="selection as in `lab tasks`, e.g. atomic/seen")
    at.add_argument("--jobs", type=int, default=8, help="parallel downloads (default 8)")
    at.add_argument("--force", action="store_true", help="fetch files even if they are already present")
    at.set_defaults(func=cmd_assets)

    ck = sub.add_parser("checkpoint", help="download one released checkpoint")
    ck.add_argument("name", choices=CHECKPOINTS)
    ck.add_argument("--out", default="checkpoints/hf", help="local directory (mirrors the repo layout)")
    ck.add_argument("--no-weights", action="store_true", help="configs only (skip *.safetensors/*.pt/*.bin)")
    ck.set_defaults(func=cmd_checkpoint)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
