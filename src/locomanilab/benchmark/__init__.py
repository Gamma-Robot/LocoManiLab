"""The benchmark task table: per-task closed-loop evaluation settings (tasks.yaml).

`lab eval --task <name>` and `lab eval --task-set <selection>` fill the eval options
from it; `lab tasks` lists it. Selections combine a type (atomic | composite) and a
split (seen | unseen_task | unseen_scene), e.g. ``atomic/seen``; ``all`` selects
every task and a comma separates alternatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TASKS_FILE = Path(__file__).with_name("tasks.yaml")
TYPES = ("atomic", "composite")
SPLITS = ("seen", "unseen_task", "unseen_scene")


@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    env: str
    scene_id: int
    type: str
    split: str
    prompt: str
    max_steps: int
    genuine_min_steps: int
    episodes: int
    robot: str
    backend: str
    sonic_ckpt: str
    scene_seed: int
    reset_jitter: float
    reset_object_jitter: float
    jitter_seed: int
    action_horizon: int

    def eval_options(self) -> dict:
        """The `lab eval` options this task fixes, keyed by their eval() parameter names."""
        return dict(env_name=self.env, scene_id=self.scene_id, prompt=self.prompt,
                    max_steps=self.max_steps, genuine_min_steps=self.genuine_min_steps,
                    episodes=self.episodes, robot=self.robot, backend=self.backend,
                    sonic_ckpt=self.sonic_ckpt, fixed_scene=True, scene_seed=self.scene_seed,
                    reset_jitter=self.reset_jitter, reset_object_jitter=self.reset_object_jitter,
                    jitter_seed=self.jitter_seed, action_horizon=self.action_horizon)


def load_tasks(path: Path = TASKS_FILE) -> dict[str, BenchmarkTask]:
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    defaults = data.get("defaults") or {}
    tasks = {}
    for name, entry in (data.get("tasks") or {}).items():
        task = BenchmarkTask(name=name, **{**defaults, **entry})
        if task.type not in TYPES or task.split not in SPLITS:
            raise ValueError(f"{name}: unknown type/split {task.type}/{task.split}")
        tasks[name] = task
    return tasks


def select_tasks(selection: str, tasks: dict[str, BenchmarkTask] | None = None) -> list[BenchmarkTask]:
    """Tasks matching ``selection`` (see the module docstring), in table order."""
    tasks = load_tasks() if tasks is None else tasks
    chosen: dict[str, BenchmarkTask] = {}
    for part in (p.strip() for p in selection.split(",")):
        if not part:
            continue
        if part in tasks:
            chosen[part] = tasks[part]
            continue
        words = part.split("/") if part != "all" else []
        unknown = [w for w in words if w not in TYPES + SPLITS]
        if unknown or len(words) > 2:
            raise ValueError(f"unknown task or selection {part!r}: use a task name, all, "
                             f"{'|'.join(TYPES)}, {'|'.join(SPLITS)}, or <type>/<split>")
        for task in tasks.values():
            if all(w in (task.type, task.split) for w in words):
                chosen[task.name] = task
    if not chosen:
        raise ValueError(f"no benchmark task matches {selection!r}")
    return [t for t in tasks.values() if t.name in chosen]
