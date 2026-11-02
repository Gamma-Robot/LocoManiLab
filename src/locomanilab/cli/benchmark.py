"""Benchmark task table commands and the `lab eval --task/--task-set` helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .app import app


@app.command("tasks")
def tasks_(
    selection: str = typer.Argument("all", help="all | atomic | composite | seen | unseen_task | "
                                                "unseen_scene | <type>/<split> | task name; "
                                                "a comma separates alternatives"),
    names_only: bool = typer.Option(False, "--names", help="print task names only"),
):
    """List benchmark tasks and their evaluation settings (used by `lab eval --task`)."""
    from locomanilab.benchmark import select_tasks

    tasks = select_tasks(selection)
    for t in tasks:
        if names_only:
            typer.echo(t.name)
        else:
            typer.echo(f"{t.name:80s} {t.type:9s} {t.split:12s} episodes={t.episodes:<3d} "
                       f"max_steps={t.max_steps:<5d} genuine={t.genuine_min_steps:<4d} {t.prompt!r}")
    if not names_only:
        typer.echo(f"{len(tasks)} tasks")


def _explicit(typer_ctx: typer.Context, name: str) -> bool:
    # Compared by name: the enum lives in click or in typer's vendored copy, by typer version.
    source = typer_ctx.get_parameter_source(name)
    return source is not None and source.name not in ("DEFAULT", "DEFAULT_MAP")


def plan_task_runs(typer_ctx: typer.Context, model: str, task: Optional[str],
                   task_set: Optional[str], opts: dict) -> list[tuple]:
    """[(BenchmarkTask | None, eval options)] for this `lab eval` call.

    Without --task/--task-set this is the single run the options describe. With them,
    each task's table settings replace the options left at their defaults, and every
    task writes its videos and results.json to its own directory: --video-dir (default
    runs/eval/<model>) is the root for --task-set, and the directory itself for --task.
    """
    if task and task_set:
        raise ValueError("pass either --task or --task-set, not both")
    if not (task or task_set):
        return [(None, dict(opts))]
    from locomanilab.benchmark import load_tasks, select_tasks

    if task:
        table = load_tasks()
        if task not in table:
            raise ValueError(f"unknown benchmark task {task!r}; `lab tasks` lists them "
                             "(use --task-set for a selection)")
        selected = [table[task]]
    else:
        selected = select_tasks(task_set)
        if _explicit(typer_ctx, "results_json"):
            raise ValueError("--task-set writes one results.json per task; use --video-dir "
                             "to choose the output root instead of --results-json")
    root = opts.get("video_dir") or f"runs/eval/{model}"
    runs = []
    for t in selected:
        kw = dict(opts)
        for name, value in t.eval_options().items():
            if name == "sonic_ckpt" and _explicit(typer_ctx, "policy_config"):
                continue
            if _explicit(typer_ctx, name):
                if kw.get(name) != value:
                    typer.echo(f"[task] {t.name}: keeping {name}={kw.get(name)!r} from the "
                               f"command line (table: {value!r})")
                continue
            kw[name] = value
        out = root if (task and _explicit(typer_ctx, "video_dir")) else f"{root}/{t.name}"
        kw["video_dir"] = out
        if not _explicit(typer_ctx, "results_json"):
            kw["results_json"] = f"{out}/results.json"
        runs.append((t, kw))
    return runs


def finished_episodes(results_json: Path) -> int:
    """Episodes already recorded in a results.json (0 when absent or unreadable)."""
    try:
        return len(json.loads(Path(results_json).read_text()).get("results") or [])
    except (OSError, ValueError, AttributeError):
        return 0


def summarize(rows: list[dict], out: Path) -> dict:
    """Per-task, per-group (type/split) and overall success rates; written to ``out``."""
    groups: dict[str, list[float]] = {}
    for r in rows:
        if r.get("success_rate") is not None:
            for key in (f"{r['type']}/{r['split']}", r["type"], "all"):
                groups.setdefault(key, []).append(r["success_rate"])
    summary = {"tasks": {r["task"]: r for r in rows},
               "groups": {k: {"tasks": len(v), "mean_success_rate": sum(v) / len(v)}
                          for k, v in sorted(groups.items())}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    return summary


def task_row(task, results_json: Path, ok: bool, message: str = "") -> dict:
    row = {"task": task.name, "type": task.type, "split": task.split, "episodes": task.episodes,
           "ok": ok, "success": None, "genuine": None, "success_rate": None}
    try:
        data = json.loads(Path(results_json).read_text())
        n = len(data.get("results") or [])
        row.update(success=int(data.get("success", 0)), genuine=int(data.get("genuine", 0)),
                   evaluated=n, success_rate=(int(data.get("success", 0)) / n) if n else None)
    except (OSError, ValueError, TypeError):
        row["error"] = message or "no results.json"
    return row
