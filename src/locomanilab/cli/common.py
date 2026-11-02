"""Shared CLI parsing and path helpers."""
from __future__ import annotations
import re
import shlex
from datetime import datetime
from pathlib import Path
from typing import Optional
import typer
from locomanilab import config as cfg
from locomanilab.registry import build_plugin


def _ctx() -> cfg.Context:
    try:
        from .app import CLI_OPTIONS
        options = CLI_OPTIONS.get()
        return cfg.load_context(options.get("config"), workspace=options.get("workspace"), required=False)
    except cfg.ConfigError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)


def _plugin(model: str, dry_run: bool):
    ctx = _ctx()
    try:
        return build_plugin(model, ctx, dry_run=dry_run), ctx
    except (KeyError, cfg.ConfigError) as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)


def _stage(plugin, stage: str, **kw):
    """Run one plugin stage, refusing model-specific flags this plugin does not implement
    (see ModelPlugin.MODEL_OPTS). eval / infer-check do the same check inside the base
    class against the plugin's `_sim_server` hook."""
    try:
        method = getattr(plugin, stage)
        refused = plugin.refuse_unsupported(stage, method, kw)
        return refused if refused else method(**kw)
    except (cfg.ConfigError, ValueError, FileNotFoundError) as exc:
        from locomanilab.base import StageResult
        return StageResult(ok=False, message=str(exc))


def _workflow(dry_run: bool):
    from locomanilab.runner import Runner
    from locomanilab.workflows import Workflows

    ctx = _ctx()
    runner = Runner(conda_root=ctx.conda_root, log_root=ctx.log_root, dry_run=dry_run)
    return Workflows(ctx, runner), ctx


def _finish(res) -> None:
    color = typer.colors.GREEN if res.ok else typer.colors.RED
    typer.secho(f"\n[{'OK' if res.ok else 'FAIL'}] {res.message}", fg=color)
    if res.log_path:
        typer.echo(f"log: {res.log_path}")
    if not res.ok:
        raise typer.Exit(1)


_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[_-]\d{2}-\d{2}(?:-\d{2})?")


def _ts_key(name: str) -> Optional[str]:
    """The dataset's timestamp, normalised so both spellings sort together."""
    m = _TS_RE.search(name)
    return m.group(0).replace("_", "-") if m else None


_COLLECT_ROOTS = ("outputs_sim", "outputs_real", "outputs")


def _collect_roots(config_root: Path) -> list[Path]:
    """The collect roots that exist under *config_root*, in _COLLECT_ROOTS order."""
    return [config_root / name for name in _COLLECT_ROOTS if (config_root / name).is_dir()]


def _dataset_dirs(root: Path):
    """Dataset dirs under *root*: the ones named with a timestamp, one level deep too.

    Collected datasets live at `outputs_{sim,real}/<backend>/<env>_<ts>`, so a scan of
    a root has to descend through the per-backend dirs. Anything that is itself
    timestamped, or is already a LeRobot root (meta/info.json), is taken as-is and
    not descended into.
    """
    from locomanilab.common.lerobot_meta import is_dataset_dir

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if _ts_key(d.name) or is_dataset_dir(d):
            yield d
            continue
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and _ts_key(sub.name):
                yield sub


def _range_datasets(ctx: cfg.Context, start: Optional[str], end: Optional[str]) -> list[Path]:
    """Select dataset dirs whose timestamp falls in [start, end] (inclusive).

    Bounds may be bare dir names (looked for under every <repo>/outputs_* collect root
    and its per-backend subdirs) or paths, in which case only that parent dir is
    scanned. A bound may also be just a timestamp or its prefix (e.g. 2026-08-11-11
    covers that hour), and the two spellings are interchangeable — comparison is on the
    timestamp, not the name, so `<env>_<ts>` dirs order by time rather than by env.

    A bare range therefore spans sim and real together, which is the honest reading of
    "everything I collected in this window" and stays the caller's to narrow: point
    --start/--end at one root's path to scan only that one.
    """
    parents: list[Path] = []
    bounds: dict[str, str] = {}
    for label, bound in (("start", start), ("end", end)):
        if bound is None:
            continue
        p = Path(bound.rstrip("/"))
        if p.parent != Path("."):
            parents.append(p.parent)
        bounds[label] = _ts_key(p.name) or p.name.replace("_", "-")
    if len(parents) == 2 and parents[0] != parents[1]:
        raise typer.BadParameter("--start and --end point to different parent directories")
    # resolve now: the runner executes from another repo, so relative paths must
    # be anchored to the invoking cwd before they reach the underlying script
    if parents:
        roots = [parents[0].expanduser().resolve()]
        if not roots[0].is_dir():
            raise typer.BadParameter(f"dataset root not found: {roots[0]}")
    else:
        config_root = ctx.config_root.expanduser().resolve()
        roots = _collect_roots(config_root)
        if not roots:
            raise typer.BadParameter(
                f"no collect roots under {config_root} (looked for "
                f"{', '.join(_COLLECT_ROOTS)})")
    lo, hi = bounds.get("start"), bounds.get("end")
    dated = sorted(((_ts_key(d.name), d) for root in roots for d in _dataset_dirs(root)),
                   key=lambda kd: (kd[0] or "", kd[1].name))
    picked = [d for ts, d in dated
              if ts is not None
              and (lo is None or ts >= lo)
              and (hi is None or ts <= hi or ts.startswith(hi))]
    if not picked:
        where = ", ".join(str(r) for r in roots)
        raise typer.BadParameter(f"no dataset dirs in {where} within [{lo or '…'}, {hi or '…'}]")
    return picked


def _default_collect_out(ctx: cfg.Context, *, backend: str, name: str, sim: bool,
                         fixed_task: bool = False, scene_id: Optional[int] = None,
                         policy_config: Optional[str] = None,
                         now: Optional[datetime] = None) -> Path:
    """Return the next unused timestamped collection directory.

    This uses the same root/lane/timestamp layout as the collectors' own defaults.
    `name` is the sim env id, or "real" on hardware where there is no env. LocoManiLab also
    makes a fixed-task dataset's selected scene visible in the directory name:
    ``_fixed_scene<scene-id>`` when the scene is explicit, and the legacy ``_fixed``
    suffix otherwise. Rooted at env.local.yaml's directory, so it does not depend on
    where the command was typed.

    Sim and real split at the top rather than deeper because the tree is what people
    rsync, glob and hand to `lab prepare`: `--backend` is recorded inside every
    dataset and re-checked at merge time, but nothing downstream re-checks where the
    frames came from, so that distinction has to be visible in the path.

    The per-controller directory is a *lane*, not the bare backend (see
    `locomanilab.common.variants.collection_lane` for why): v1_1 files itself under
    `sonic_v1_1/` next to the historical low_latency `sonic/` lane, so old data
    keeps its existing path even though new collections default to v1_1.
    """
    from locomanilab.common.variants import collection_lane

    lane = collection_lane(backend, policy_config)
    root = ctx.config_root / ("outputs_sim" if sim else "outputs_real") / lane
    stamp = (now or datetime.now()).strftime("%Y-%m-%d-%H-%M-%S")
    suffix = ""
    if fixed_task:
        suffix = f"_fixed_scene{scene_id}" if scene_id is not None else "_fixed"
    candidate = root / f"{name}{suffix}_{stamp}"
    n = 2
    while candidate.exists():
        candidate = root / f"{name}{suffix}_{stamp}_{n}"
        n += 1
    return candidate


def _hsize(n: float) -> str:
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024 or u == "T":
            return f"{n:.0f}{u}"
        n /= 1024


def _hage(sec: float) -> str:
    if sec < 3600:
        return f"{sec / 60:.0f}m"
    if sec < 86400:
        return f"{sec / 3600:.1f}h"
    return f"{sec / 86400:.1f}d"


def _resolve_sonic_ckpt(ctx: cfg.Context, name: str, *, backend: str,
                        policy_config: Optional[str]) -> str:
    """`--sonic-ckpt <name>` -> the absolute `--policy-config` it stands for.

    The shortcut exists so the variant does not have to be spelled as a
    machine-specific absolute path; it is resolved against this machine's
    g1_backend tree and must exist there, then travels as a plain
    `--policy-config` so the collector, the lane (`outputs_*/sonic_v1_1/`) and
    `collection_info.policy_config` see exactly what they would have seen
    typed out. Exits with the reason on anything that would otherwise fail
    later or, worse, collect under the wrong variant.
    """
    from locomanilab.common.variants import (SONIC_DEFAULT_VARIANT, canonical_sonic_ckpt,
                                      missing_sonic_weights, sonic_policy_config)
    if backend != "sonic":
        typer.secho(f"--sonic-ckpt is a --backend sonic option, not {backend!r}",
                    fg=typer.colors.RED)
        raise typer.Exit(1)
    if policy_config:
        typer.secho("pass either --sonic-ckpt or --policy-config, not both (the "
                    "first is shorthand for the second)", fg=typer.colors.RED)
        raise typer.Exit(1)
    try:
        variant = canonical_sonic_ckpt(name)
        path = sonic_policy_config(ctx.repo_path("g1_backend"), variant)
    except (ValueError, FileNotFoundError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    missing = missing_sonic_weights(path)
    if missing:
        # A manifest without its weights is the one state that starts cleanly
        # and dies at load -- say so now, with the command that fixes it.
        typer.secho(f"sonic {variant}: {path.parent} is missing {', '.join(missing)} -- "
                    "download the selected weights first: `python robocasa/scripts/"
                    "download_g1_checkpoints.py --family "
                    f"{'sonic' if variant == SONIC_DEFAULT_VARIANT else 'sonic_' + variant}"
                    f" --dest {shlex.quote(str(ctx.repo_path('g1_backend') / 'checkpoints'))}"
                    "` (from the configured simulator checkout)", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo(f"sonic ckpt: {variant} -> {path}")
    return str(path)


def _replay_sources(ctx, source_runs, start, end) -> list[Path]:
    """Run dirs for replay/augment, from explicit paths or a --start/--end range.

    Both commands need the fixed-task sidecars (the per-reset jitter is what makes
    a replay land on the recorded start pose), so a dir without them is refused
    here rather than after a sim env has been stood up.
    """
    if (start or end) and source_runs:
        typer.secho("pass either explicit run dirs or --start/--end, not both",
                    fg=typer.colors.RED)
        raise typer.Exit(1)
    if start or end:
        source_runs = _range_datasets(ctx, start, end)
        typer.echo(f"selected {len(source_runs)} run(s):")
        for d in source_runs:
            typer.echo(f"  {d}")
    if not source_runs:
        typer.secho("no source runs given: pass dirs or --start/--end", fg=typer.colors.RED)
        raise typer.Exit(1)
    runs = [Path(d).expanduser().resolve() for d in source_runs]
    missing = [d for d in runs if not (d / "fixed_task_manifest.json").is_file()]
    if missing:
        typer.secho("not fixed-task runs (no fixed_task_manifest.json):", fg=typer.colors.RED)
        for d in missing:
            typer.echo(f"  {d}")
        typer.secho("replay re-applies the recorded per-reset jitter, so it needs the "
                    "sidecars `lab collect --sim --fixed-task` writes (runs from "
                    "`lab augment` carry them too).", fg=typer.colors.RED)
        raise typer.Exit(1)
    return runs

