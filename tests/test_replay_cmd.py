"""`lab replay`: the argv handed to the sim repo's replay_augment.

Golden, like tests/test_sim_servers.py -- this is the exact command a machine
runs, and a real invocation costs a full replay of every episode in a run.
"""
from __future__ import annotations

from pathlib import Path


def _run(plugin, tree, **kw):
    p = plugin("fastwam")
    src = Path(tree["repos"]["sim_repo"]).parent.parent / "outputs_sim" / "sonic" / "run_a"
    src.mkdir(parents=True, exist_ok=True)
    kw.setdefault("source_runs", [src])
    res = p.replay(**kw)
    assert res.ok, res.message
    return res.command, src


def test_default_command_is_the_standard_preset(plugin, tree):
    cmd, src = _run(plugin, tree, randomize="standard")
    assert "-m robocasa.scripts.replay_augment" in cmd
    assert f"--source-run {src}" in cmd
    # the spec, not a pile of appearance flags
    assert "--randomize standard" in cmd
    for gone in ("--texture-frac", "--light-jitter", "--material-pbr-jitter",
                 "--cam-pos", "--robot-tint", "--distractors "):
        assert gone not in cmd, f"{gone} came back"
    assert "--variants 1" in cmd and "--seed 0" in cmd
    assert "--episodes all" in cmd and "--reroll 0" in cmd
    # no --out-base: the script's own default puts it in the sibling _replay lane
    assert "--out-base" not in cmd


def test_spec_and_knobs_are_forwarded_verbatim(plugin, tree):
    cmd, _ = _run(plugin, tree, randomize="standard,textures=0.4,distractors=2",
                  variants=4, seed=7, episodes="0,1,2",
                  reroll="1", object_textures=True,
                  distractor_groups="cup,mug", preroll_ticks=150, keep_failed=True)
    assert "--randomize standard,textures=0.4,distractors=2" in cmd
    assert "--variants 4" in cmd and "--seed 7" in cmd
    assert "--episodes 0,1,2" in cmd and "--reroll 1" in cmd
    assert "--randomize-object-textures" in cmd
    assert "--distractor-groups cup,mug" in cmd
    assert "--preroll-ticks 150" in cmd and "--keep-failed" in cmd


def test_runs_in_the_sim_repo_under_its_conda_env(plugin, tree):
    cmd, _ = _run(plugin, tree)
    assert "conda activate tele_benchmark" in cmd
    assert "MUJOCO_GL=egl" in cmd
    assert tree["repos"]["sim_repo"] in cmd   # cwd + PYTHONPATH


def test_missing_sim_submodule_refuses_politely(plugin, ctx, tmp_path):
    ctx.repos["sim_repo"] = str(tmp_path / "gone")
    res = plugin("fastwam").replay(source_runs=[tmp_path])
    assert not res.ok and "humanoid_benchmark" in res.message


def test_no_source_runs_refuses(plugin, tree):
    res = plugin("fastwam").replay(source_runs=[])
    assert not res.ok and "no source runs" in res.message


# --------------------------------------------------------------------------- #
def test_replay_defaults_to_no_randomization_and_no_output(plugin, tree):
    # `lab replay` answers a question; producing data is `lab augment`
    cmd, _ = _run(plugin, tree, randomize="off", write=False)
    assert "--randomize off" in cmd and "--no-write" in cmd
    assert "--viewer" not in cmd


def test_view_opens_a_window_and_drops_the_headless_gl(plugin, tree):
    cmd, _ = _run(plugin, tree, randomize="off", viewer=True, write=False)
    assert "--viewer" in cmd and "--no-write" in cmd
    # egl is headless-only; the interactive window comes up black under it
    assert "MUJOCO_GL" not in cmd


def test_replay_writes_only_when_asked(plugin, tree):
    cmd, _ = _run(plugin, tree, randomize="standard", write=True)
    assert "--write" in cmd and "--no-write" not in cmd


def test_headless_runs_still_set_egl(plugin, tree):
    cmd, _ = _run(plugin, tree, randomize="off", write=False)
    assert "MUJOCO_GL=egl" in cmd


def test_reroll_period_is_forwarded(plugin, tree):
    for n in ("0", "1", "4"):
        cmd, _ = _run(plugin, tree, randomize="standard", reroll=n, write=False)
        assert f"--reroll {n}" in cmd
