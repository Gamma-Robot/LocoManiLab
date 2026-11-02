"""The benchmark task table and the `lab eval --task/--task-set` option planning."""
import json
import re
from collections import Counter

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from locomanilab.benchmark import SPLITS, TASKS_FILE, TYPES, load_tasks, select_tasks
from locomanilab.cli import app
from locomanilab.cli.benchmark import finished_episodes, plan_task_runs, summarize, task_row


def test_table_covers_every_task_once_with_consistent_settings():
    tasks = load_tasks()
    assert TASKS_FILE.is_file() and len(tasks) == 234
    counts = Counter((t.type, t.split) for t in tasks.values())
    assert counts == {("atomic", "seen"): 50, ("atomic", "unseen_task"): 46, ("atomic", "unseen_scene"): 6,
                      ("composite", "seen"): 50, ("composite", "unseen_task"): 77,
                      ("composite", "unseen_scene"): 5}
    for t in tasks.values():
        assert t.name == f"{t.env}_scene{t.scene_id}"
        assert t.episodes == (20 if t.split == "seen" else 10)
        assert t.prompt.strip() and 0 < t.genuine_min_steps < t.max_steps
        assert (t.robot, t.backend, t.sonic_ckpt) == ("G1FloatingBody", "sonic", "v1_1")


def test_selection_by_type_split_name_and_union():
    tasks = load_tasks()
    assert len(select_tasks("all", tasks)) == 234
    assert len(select_tasks("atomic", tasks)) == 102 and len(select_tasks("composite", tasks)) == 132
    assert len(select_tasks("unseen_scene", tasks)) == 11
    assert len(select_tasks("composite/unseen_task", tasks)) == 77
    assert len(select_tasks("seen/atomic", tasks)) == 50
    name = "IThorKitchenOpenFridge_scene13"
    assert [t.name for t in select_tasks(f"{name},{name}", tasks)] == [name]
    assert len(select_tasks("atomic/seen,composite/seen", tasks)) == 100
    for bad in ("atomic/kitchen", "atomic/seen/x", "NoSuchTask_scene1", ","):
        with pytest.raises(ValueError):
            select_tasks(bad, tasks)
    assert set(TYPES) == {"atomic", "composite"} and set(SPLITS) == {"seen", "unseen_task", "unseen_scene"}


class _Ctx:
    def __init__(self, *explicit):
        self.explicit = set(explicit)

    def get_parameter_source(self, name):
        return SimpleNamespace(name="COMMANDLINE" if name in self.explicit else "DEFAULT")


OPTS = dict(env_name="TabletopPickPlace", scene_id=None, prompt=None, max_steps=60, genuine_min_steps=None,
            episodes=1, robot=None, backend="sonic", sonic_ckpt=None, fixed_scene=False, scene_seed=None,
            reset_jitter=None, reset_object_jitter=None, jitter_seed=None, action_horizon=64,
            policy_config=None, video_dir=None, results_json=None, port=5550)


def test_task_fills_defaults_and_keeps_command_line_values():
    ((task, kw),) = plan_task_runs(_Ctx("max_steps"), "gr00t", "IThorKitchenOpenFridge_scene13", None,
                                   dict(OPTS, max_steps=900))
    assert task.name == "IThorKitchenOpenFridge_scene13"
    assert (kw["env_name"], kw["scene_id"], kw["episodes"], kw["genuine_min_steps"]) == (
        "IThorKitchenOpenFridge", 13, 20, 66)
    assert kw["max_steps"] == 900  # given on the command line
    assert kw["fixed_scene"] and (kw["reset_jitter"], kw["reset_object_jitter"], kw["jitter_seed"]) == (.05, .02, 0)
    assert kw["sonic_ckpt"] == "v1_1" and kw["prompt"] == "open the fridge door in front of you"
    assert kw["video_dir"] == "runs/eval/gr00t/IThorKitchenOpenFridge_scene13"
    assert kw["results_json"] == "runs/eval/gr00t/IThorKitchenOpenFridge_scene13/results.json"


def test_explicit_policy_config_suppresses_the_sonic_shortcut():
    ((_, kw),) = plan_task_runs(_Ctx("policy_config"), "gr00t", "IThorKitchenOpenFridge_scene13", None,
                                dict(OPTS, policy_config="/x/policy.yaml"))
    assert kw["sonic_ckpt"] is None and kw["policy_config"] == "/x/policy.yaml"


def test_task_set_uses_one_directory_per_task_under_the_root():
    runs = plan_task_runs(_Ctx("video_dir"), "act", None, "atomic/unseen_scene", dict(OPTS, video_dir="out"))
    assert len(runs) == 6
    assert {kw["results_json"] for _, kw in runs} == {f"out/{t.name}/results.json" for t, _ in runs}
    with pytest.raises(ValueError, match="results.json per task"):
        plan_task_runs(_Ctx("results_json"), "act", None, "atomic", dict(OPTS, results_json="r.json"))
    with pytest.raises(ValueError, match="either"):
        plan_task_runs(_Ctx(), "act", "IThorKitchenOpenFridge_scene13", "all", OPTS)
    ((task, kw),) = plan_task_runs(_Ctx(), "act", None, None, OPTS)
    assert task is None and kw == OPTS


def test_summary_and_resume_bookkeeping(tmp_path):
    tasks = load_tasks()
    a, b = tasks["IThorKitchenOpenFridge_scene13"], tasks["HolodeckApproachDesk_scene117"]
    ra = tmp_path / "a.json"
    ra.write_text(json.dumps({"success": 3, "genuine": 2, "results": [{}] * 4}))
    assert finished_episodes(ra) == 4 and finished_episodes(tmp_path / "missing.json") == 0
    rows = [task_row(a, ra, ok=True), task_row(b, tmp_path / "missing.json", ok=False, message="boom")]
    assert rows[0]["success_rate"] == .75 and rows[1]["error"] == "boom"
    summary = summarize(rows, tmp_path / "out" / "summary.json")
    assert summary["groups"]["atomic/seen"] == {"tasks": 1, "mean_success_rate": .75}
    assert json.loads((tmp_path / "out" / "summary.json").read_text())["tasks"].keys() == {a.name, b.name}


def test_cli_lists_tasks_and_documents_the_eval_options():
    runner = CliRunner()
    res = runner.invoke(app, ["tasks", "composite/unseen_scene", "--names"])
    assert res.exit_code == 0, res.output
    assert len(res.output.split()) == 5
    res = runner.invoke(app, ["eval", "--help"])
    # CI forces colored help; option names are split by ANSI style codes.
    text = re.sub(r"\x1b\[[0-9;]*m", "", res.output)
    assert "--task-set" in text and "--task" in text
