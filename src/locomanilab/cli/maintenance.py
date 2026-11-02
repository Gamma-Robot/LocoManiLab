"""Maintenance commands."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import typer
from locomanilab import config as cfg
from locomanilab.registry import available_models
from .app import app
from .common import _COLLECT_ROOTS, _collect_roots, _ctx, _hage, _hsize

@app.command("clean-ckpts")
def clean_ckpts(
    model: Optional[str] = typer.Option(None, "--model", "-M", help="only this model (default: all)"),
    min_age: float = typer.Option(30.0, "--min-age", min=0,
                                  help="treat run dirs modified within N minutes as active and skip"),
    yes: bool = typer.Option(False, "--yes", help="actually delete (default: list only / dry-run)"),
):
    """Sweep empty checkpoint run-dirs under each model's default ckpt root.

    A crashed/aborted run still creates its run dir (scaffolding, empty checkpoints/ tree) but
    never saves a checkpoint. This lists such dirs; pass --yes to delete them. Runs a live
    process still references, or that were touched within --min-age minutes, are always skipped.
    """
    import shutil
    import time
    from locomanilab.common.cleanup import find_empty_ckpt_dirs
    from locomanilab.workflows.checkpoints import checkpoint_scans

    ctx = _ctx()
    models = available_models()
    names = [model] if model else list(models)
    rows: list[tuple[str, dict]] = []
    seen_paths: set[Path] = set()
    for name in names:
        m = models.get(name)
        if m is None:
            raise typer.BadParameter(f"unknown model {name!r}; run lab list-models", param_hint="--model")
        if not m.raw.get("ckpt"):
            continue
        for root, scan in checkpoint_scans(ctx, m):
            for e in find_empty_ckpt_dirs(root, scan, min_age_min=min_age):
                identity = e["path"].resolve()
                if identity not in seen_paths:
                    seen_paths.add(identity)
                    e["scan_root"], e["scan_config"] = root, scan
                    rows.append((name, e))

    if not rows:
        typer.secho("no empty ckpt run-dirs found.", fg=typer.colors.GREEN)
        return

    now = time.time()
    deletable = [(n, e) for n, e in rows if not e["active"]]
    for name, e in rows:
        tag = (typer.style("active→skip", fg=typer.colors.YELLOW) if e["active"]
               else typer.style("empty", fg=typer.colors.RED))
        typer.echo(f"  [{name:10s}] {tag}  {e['path']}  ({_hsize(e['size'])}, {_hage(now - e['mtime'])} old)")

    if not deletable:
        typer.secho("\nall empties are active (in-use / recent) — nothing to delete.", fg=typer.colors.YELLOW)
        return
    if not yes:
        typer.secho(f"\n{len(deletable)} deletable empty run-dir(s). Re-run with --yes to delete.",
                    fg=typer.colors.YELLOW)
        return
    freed = 0.0
    deleted = 0
    for name, e in deletable:
        # Recheck before removal: a run may have produced a checkpoint since the scan.
        current = find_empty_ckpt_dirs(e["scan_root"], e["scan_config"], min_age_min=min_age)
        if not any(row["path"] == e["path"] and not row["active"] for row in current):
            typer.echo(f"  skip changed or active directory: {e['path']}")
            continue
        freed += e["size"]
        shutil.rmtree(e["path"])
        deleted += 1
    typer.secho(f"\ndeleted {deleted} empty run-dir(s), freed ~{_hsize(freed)}.", fg=typer.colors.GREEN)


@app.command("clean-datasets")
def clean_datasets(
    dirs: Optional[list[Path]] = typer.Argument(None,
                                                help="dirs to sweep — a collect root, a "
                                                     "per-backend dir, or single dataset dirs "
                                                     "(default: every outputs_* root next to "
                                                     "env.local.yaml)"),
    min_age: float = typer.Option(10.0, "--min-age", min=0,
                                  help="treat dataset dirs touched within N minutes as an "
                                       "in-flight collection and skip them"),
    yes: bool = typer.Option(False, "--yes", help="actually delete (default: list only / dry-run)"),
):
    """Sweep dataset dirs that recorded no episode at all.

    A collection you start to check the setup and stop before the first episode still
    leaves `outputs_sim/<backend>/<env>_<ts>/` behind — meta/ written, plus the
    fixed-task sidecars, and nothing else. Those pile up and are what `lab prepare`
    skips as "0 episodes recorded"; this lists them, and --yes deletes them.

    A dir is only ever flagged when its meta lists zero episodes AND its subtree holds
    no parquet/video/large file, so a dataset with broken meta but real frames is kept.
    Dirs a live process references, or that were touched within --min-age minutes, are
    skipped — a collection running right now looks exactly like an aborted one until
    its first episode lands.
    """
    import shutil
    import time
    from locomanilab.common.cleanup import find_empty_datasets

    if dirs:
        roots = [Path(d).expanduser().absolute() for d in dirs]
        missing = [r for r in roots if not r.is_dir()]
        if missing:
            typer.secho("dir(s) not found: " + " ".join(map(str, missing)), fg=typer.colors.RED)
            raise typer.Exit(1)
    else:
        ctx = _ctx()
        config_root = ctx.config_root.expanduser().resolve()
        roots = _collect_roots(config_root)
        if not roots:
            typer.secho(f"no collect roots under {config_root} (looked for "
                        f"{', '.join(_COLLECT_ROOTS)}) — pass the dirs to sweep",
                        fg=typer.colors.RED)
            raise typer.Exit(1)
        typer.echo("scanning: " + ", ".join(str(r) for r in roots))

    rows = find_empty_datasets(roots, min_age_min=min_age)
    if not rows:
        typer.secho("no empty dataset dirs found.", fg=typer.colors.GREEN)
        return

    now = time.time()
    deletable = [e for e in rows if not e["active"]]
    for e in rows:
        tag = (typer.style("active→skip", fg=typer.colors.YELLOW) if e["active"]
               else typer.style("no data", fg=typer.colors.RED))
        typer.echo(f"  {tag}  {e['path']}  ({_hsize(e['size'])}, {_hage(now - e['mtime'])} old)")

    if not deletable:
        typer.secho("\nall empties are active (in-use / recent) — nothing to delete.",
                    fg=typer.colors.YELLOW)
        return
    if not yes:
        typer.secho(f"\n{len(deletable)} deletable empty dataset dir(s). Re-run with --yes to delete.",
                    fg=typer.colors.YELLOW)
        return
    deleted = 0
    for e in deletable:
        current = find_empty_datasets([e["path"]], min_age_min=min_age)
        if not any(row["path"] == e["path"] and not row["active"] for row in current):
            typer.echo(f"  skip changed or active directory: {e['path']}")
            continue
        shutil.rmtree(e["path"])
        deleted += 1
    typer.secho(f"\ndeleted {deleted} empty dataset dir(s).", fg=typer.colors.GREEN)


@app.command()
def doctor(
    model: Optional[str] = typer.Option(None, "--model", "-M", help="Check one selected model."),
    stage: str = typer.Option("core", "--stage", help="core, data, collect, train, deploy, or eval."),
    sim: bool = typer.Option(False, "--sim", help="Also inspect the optional simulator prerequisites."),
    backend: Optional[str] = typer.Option(None, "--backend",
                                          help="--sim: selected controller (sonic, amo, heft, scalebfm, almi)."),
    sonic_ckpt: Optional[str] = typer.Option(None, "--sonic-ckpt",
                                             help="--sim --backend sonic: selected checkpoint variant."),
    policy_config: Optional[Path] = typer.Option(None, "--policy-config",
                                                  help="--sim: explicit local controller policy.yaml."),
    sim_env: Optional[str] = typer.Option(None, "--sim-env",
                                          help="--sim: conda env of the simulator (default: tele_benchmark)."),
    json_: bool = typer.Option(False, "--json", help="Emit a machine-readable report."),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Return failure when a selected check fails."),
):
    """Check selected local prerequisites without downloading assets or loading a model."""
    import json
    from locomanilab.maintenance.doctor import diagnose

    try:
        if json_:
            from .app import CLI_OPTIONS
            options = CLI_OPTIONS.get()
            ctx = cfg.load_context(options.get("config"), workspace=options.get("workspace"), required=False)
        else:
            ctx = _ctx()
        report = diagnose(ctx, model=model, stage=stage, sim=sim, backend=backend,
                          sonic_ckpt=sonic_ckpt, policy_config=policy_config, sim_env=sim_env)
    except cfg.ConfigError as exc:
        if json_:
            typer.echo(json.dumps({"ok": False, "error": str(exc)}))
        else:
            typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    if json_:
        typer.echo(json.dumps(report, indent=2))
    else:
        typer.echo(f"Workspace: {report['workspace']}")
        for check in report["checks"]:
            mark = "OK" if check["ok"] else "MISSING"
            typer.echo(f"[{mark}] {check['name']}: {check['detail']}")
            if not check["ok"] and check["hint"]:
                typer.echo(f"  {check['hint']}")
        typer.echo(report["scope"])
    if strict and not report["ok"]:
        raise typer.Exit(1)
