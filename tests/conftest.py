"""Shared fixtures: a machine-independent Context + dry-run plugins.

Nothing here touches env.local.yaml, external/ submodules or any model env. Every
plugin is built against a throw-away tree under tmp_path, with a fake conda.sh so
`Runner._find_conda_sh` is deterministic on any machine, and with `dry_run=True`
so the only observable product of a stage is the command it would have run
(`StageResult.command`) or the server argv it built (`SimServer.argv`).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from locomanilab.config import Context
from locomanilab.registry import build_plugin

MODELS = ("gr00t", "fastwam", "starvla", "prts", "pi05", "psi0", "act", "dp",
          "dreamzero", "internvla_a15", "xr1", "dm05")


@pytest.fixture
def tree(tmp_path: Path) -> dict:
    """A fake machine: conda root (with conda.sh), one dir per repo, a few asset paths."""
    conda_root = tmp_path / "conda"
    (conda_root / "etc" / "profile.d").mkdir(parents=True)
    (conda_root / "etc" / "profile.d" / "conda.sh").write_text("# fake conda.sh\n")
    repos = {}
    for key, name in (("fastwam", "FastWAM"),
                      ("isaac_gr00t", "Isaac-GR00T"), ("gr00t_wbc", "GR00T-WholeBodyControl"),
                      ("starvla", "starVLA"), ("sim_repo", "humanoid_benchmark"),
                      ("g1_backend", "g1_backend"), ("openpi", "openpi"), ("prts", "PRTS"),
                      ("psi0", "Psi0"), ("lerobot", "lerobot"), ("dreamzero", "dreamzero"),
                      ("internvla_a15", "InternVLA-A1.5"), ("xr1", "Xiaomi-Robotics-1"),
                      ("opendm", "opendm")):
        d = tmp_path / "external" / name
        d.mkdir(parents=True)
        repos[key] = str(d)
    (tmp_path / "env.local.yaml").write_text("role: [test]\n")
    return {"root": tmp_path, "conda_root": conda_root, "repos": repos,
            "conda_sh": str(conda_root / "etc" / "profile.d" / "conda.sh")}


@pytest.fixture
def ctx(tree) -> Context:
    return Context(role=["test"], conda_root=str(tree["conda_root"]), repos=tree["repos"],
                   paths={}, deploy={}, conda_envs={},
                   source=tree["root"] / "env.local.yaml")


@pytest.fixture
def plugin(ctx):
    """Factory: plugin(name, **ctx_overrides) -> dry-run plugin."""
    def make(name: str, *, paths: dict | None = None, conda_envs: dict | None = None):
        if paths:
            ctx.paths.update(paths)
        if conda_envs:
            ctx.conda_envs.update(conda_envs)
        return build_plugin(name, ctx, dry_run=True)
    return make


@pytest.fixture
def dataset(tmp_path: Path):
    """Factory: dataset(modality: dict, name=...) -> LeRobot-looking dir with meta/."""
    import json

    def make(modality: dict, name: str = "ds", **info):
        d = tmp_path / "datasets" / name
        (d / "meta").mkdir(parents=True)
        (d / "meta" / "modality.json").write_text(json.dumps(modality))
        (d / "meta" / "info.json").write_text(json.dumps({"fps": 50, **info}))
        return d
    return make


# modality.json groups as the collectors write them (keys only matter here)
SONIC_MODALITY = {
    "state": {k: {} for k in ("left_leg", "right_leg", "waist", "left_arm", "left_hand",
                              "right_arm", "right_hand")},
    "action": {k: {} for k in ("motion_token", "left_hand", "right_hand")},
    "video": {"ego_view": {}},
}
HEFT_MODALITY = {
    "state": {k: {} for k in ("left_leg", "right_leg", "waist", "left_arm", "right_arm",
                              "projected_gravity", "root_orientation")},
    "action": {k: {} for k in ("ref_joint_pos", "ref_root_pos", "ref_root_quat",
                               "left_hand", "right_hand")},
    "video": {"ego_view": {}},
}
AMO_MODALITY = {
    "state": {k: {} for k in ("left_hand", "right_hand", "left_arm", "right_arm", "rpy",
                              "height")},
    "action": {k: {} for k in ("left_hand", "right_hand", "left_arm", "right_arm", "rpy",
                               "height", "torso_vx", "torso_vy", "torso_vyaw",
                               "target_yaw")},
    "video": {"ego_view": {}},
}
