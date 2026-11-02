"""Closed-loop sim evaluation of a VLA, on any G1_BRIDGE whole-body controller.

This is the inverse of ``robocasa.scripts.collect_demos_g1``: instead of an
operator teleoperating through a backend and recording what the backend
consumed, a policy predicts it and the same backend consumes it. Both sides go
through ``G1_BRIDGE``, so eval and collection cannot drift apart in the way a
separately-written eval path silently can.

Two axes, and they are orthogonal:

* ``--backend``  which whole-body controller runs the robot (sonic, amo, heft,
  scalebfm, almi). It decides the *command space*.
* the policy behind ``--host:--port`` -- any gr00t-compatible server (gr00t,
  fastwam, starVLA, pi0.5, PRTS). It must emit that command space,
  which in practice means it was trained on data collected through that same
  backend. The mismatch is caught on the first response, not discovered as a
  robot that stands still.

Everything about talking to the policy and pacing a 2.5 Hz model against a
50 Hz controller lives in ``sim2real.vla``; this file is the experiment: which
scene, how many episodes, what counts as a success.

Server-free smoke tests, in increasing order of what they prove:

    --policy zero     controller stack alone: does the robot stay up?
    --policy replay   + adapter/command space: replay a collected episode's
                      recorded actions; the robot should redo the demo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="sonic",
                   help="whole-body controller: sonic | amo | heft | scalebfm | almi. "
                        "Must match the backend the training data was collected "
                        "through -- it fixes the action space the policy has to emit.")
    p.add_argument("--policy-config", default=None,
                   help="override the backend's default checkpoint policy.yaml")
    p.add_argument("--inference-backend", default=None,
                   help="onnx-cpu (default) | onnx-gpu | tensorrt | torch-jit, "
                        "for the whole-body controller (not the VLA).")
    p.add_argument("--env", default="IThorKitchenCloseOven", help="sim task env")
    p.add_argument("--robot", default="G1FloatingBody")
    p.add_argument("--scene-id", type=int, default=None,
                   help="pin an iTHOR FloorPlan -- THE SCENE THE DATA WAS COLLECTED "
                        "IN. Unset randomises it over every downloaded FloorPlan of "
                        "that room type, so the policy is tested in rooms it never "
                        "saw and the number is not comparable to a same-scene run.")
    p.add_argument("--layout", type=int, default=None)
    p.add_argument("--style", type=int, default=None)
    p.add_argument("--fixed-scene", action="store_true")
    p.add_argument("--scene-seed", type=int, default=0)
    p.add_argument("--no-task-overrides", action="store_true",
                   help="build the task exactly as the simulator revision defines it, "
                        "without sim_task_overrides.yaml (the per-task settings that keep "
                        "a few tasks at the configuration they were evaluated with).")

    p.add_argument("--policy", default="server", choices=["server", "replay", "zero"])
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5550)
    p.add_argument("--policy-timeout-ms", type=int, default=120000,
                   help="how long one get_action may take before the episode is "
                        "abandoned as an error (default 120 s, like infer-check). The "
                        "sim2real default of 20 s is too short for a 4B model's first "
                        "request (CUDA warmup), and a timeout used to kill the whole "
                        "batch instead of one episode.")
    p.add_argument("--replay-parquet", default=None,
                   help="--policy replay: a collected episode's parquet")
    p.add_argument("--prompt", default=None,
                   help="language instruction. MUST match what the checkpoint was "
                        "trained with (the dataset's meta/tasks.jsonl) -- an "
                        "out-of-distribution prompt fails silently.")

    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=1200,
                   help="per-episode step budget. G1_BRIDGE spends the start "
                        "of every episode on a settling window, so size this "
                        "generously -- >=2x a typical episode. Measured on "
                        "prts-closeoven11/scene 11, 20 eps per arm: at 2400 "
                        "steps 14/20 succeed, at 1200 a minority of episodes "
                        "finish too late to count. See docs/sim_validation.md.")
    p.add_argument("--control-freq", type=int, default=50)
    p.add_argument("--inference-rate", type=float, default=2.5,
                   help="policy re-query rate (Hz). 2.5 is what a PyTorch policy "
                        "server sustains; 10 is for TRT. Too high over-replans -- "
                        "the loop throws away most of each chunk.")
    p.add_argument("--action-horizon", type=int, default=64)
    p.add_argument("--ema-alpha", type=float, default=1.0,
                   help="EMA on the first action of each new chunk (1.0 = snap)")
    p.add_argument("--max-latency-skip", type=int, default=3)
    p.add_argument("--genuine-min-steps", type=int, default=0,
                   help="successes shorter than this are reported separately: they "
                        "usually mean the episode spawned in the goal state.")
    p.add_argument("--reset-jitter", type=float, default=0.0, metavar="R",
                   help="XY jitter radius (m) applied to the ROBOT after every reset, "
                        "the way collection did. --fixed-scene alone re-seeds the env "
                        "RNG on every scene build, so the robot spawns at the identical "
                        "nominal pose and all N episodes become ONE rollout repeated N "
                        "times (measured: 20 byte-identical videos, success can only be "
                        "0%% or 100%%). Collection instead froze the scene and then "
                        "jittered per reset from an unseeded RNG -- see the per-episode "
                        "g1_xy_jitter in fixed_task_resets.jsonl. Pass the manifest's "
                        "jitter.g1_radius (0.05) to reproduce the training distribution.")
    p.add_argument("--reset-object-jitter", type=float, default=0.0, metavar="R",
                   help="same, for task objects (manifest jitter.object_radius, 0.02)")
    p.add_argument("--jitter-seed", type=int, default=None,
                   help="nonnegative reset-jitter base seed; episode index starts at 0 "
                        "and uses base + index. Unset keeps unseeded jitter.")
    p.add_argument("--snap-hands", type=float, default=None, metavar="T",
                   help="quantise the predicted grasp: any hand whose closure "
                        "exceeds T (of the full close pose) is sent fully shut, "
                        "the rest fully open. Collection only ever recorded those "
                        "two, so a partly-closed prediction is out of distribution "
                        "and grips nothing. 0.5 is the natural starting point. "
                        "Leave unset to send the raw prediction.")

    p.add_argument("--randomize", default=None, metavar="SPEC",
                   help="eval-time scene randomization for robustness tests. "
                        "Comma-separated tokens, each with an optional =value, "
                        "plus the presets off/mild/standard/stress which expand "
                        "in place and can be overridden by later tokens "
                        "('standard,textures=0.4'). Pass --randomize help for "
                        "the full reference. The same syntax replay_augment "
                        "and preview_randomization take, so a spec copies "
                        "between them; eval draws the HELD-OUT 'test' half of "
                        "the texture pool, augmentation draws 'train'. Some "
                        "defaults resolve per scene family (a bare 'lights' is "
                        "0.15 in an iTHOR room, 0.08 in a kitchen) -- the "
                        "resolved values are echoed at startup and written to "
                        "--results-json. Fully reproducible from "
                        "--randomize-seed; numbers only compare across runs "
                        "with the SAME spec and seed."),
    p.add_argument("--randomize-seed", type=int, default=0,
                   help="seed for --randomize. Per-episode variation is "
                        "derived from it deterministically, and it is echoed "
                        "into --results-json so a run can be reproduced.")

    p.add_argument("--render", default="video", choices=["video", "viewer"])
    p.add_argument("--render-scale", type=int, default=1, metavar="N",
                   help="render camera at N times the policy resolution, then area-average "
                        "down to 320x240 (N=1 disables supersampling; try N=2 for "
                        "OSMesa texture aliasing)")
    p.add_argument("--fix-z-fighting", action="store_true",
                   default=os.environ.get("LOCOMANILAB_FIX_Z_FIGHTING", os.environ.get("SOMA_FIX_Z_FIGHTING")) == "1",
                   help="hide duplicate Holodeck wall visual geoms that overlap exactly")
    p.add_argument("--video-dir", default=None)
    p.add_argument("--results-json", default=None, help="write per-episode results here")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.jitter_seed is not None and args.jitter_seed < 0:
        p.error("--jitter-seed must be nonnegative")
    if args.render_scale < 1:
        p.error("--render-scale must be >= 1")
    return args


# --------------------------------------------------------------------------- #
def build_controller_config(args):
    """A G1_BRIDGE controller config -- the same one collection builds.

    One runtime for every backend. sonic used to have a second one (the
    vendored WHOLE_BODY_G1 SONIC controller, reachable as
    ``--controller-runtime wbt``); it was retired 2026-08-25 after the two were
    measured equivalent -- offline A/B bit-identical decoder inputs/outputs, and
    13 vs 14 of 20 in closed loop at a 2400-step budget (docs/sim_validation.md).
    """
    # The robosuite host adapter -- lives in the sim repo, while the bridge
    # itself (g1_bridge) now ships from g1_backend. Importing it also registers
    # the G1_BRIDGE composite controller.
    from robocasa.g1_host.bridge import load_sim2real_controller_config

    return load_sim2real_controller_config(
        backend=args.backend,
        # No operator: a mock source keeps the ModeManager and the backend's
        # own sample plumbing alive without opening a headset connection.
        # Nothing reads it on the VLA path.
        pico_source="mock",
        policy_config=args.policy_config,
        inference_backend=args.inference_backend or _default_inference_backend(),
    )


def _default_inference_backend() -> str:
    """Where the whole-body controller runs when --inference-backend is unset.

    The controller JSON says onnx-cpu, which is the right default for a robot's
    onboard computer and the wrong one here: this controller steps at 50 Hz
    inside an eval that is already sharing the box with a policy server, and on
    CPU its decoder alone takes ~19 ms of every 20 ms tick. Measured over 100
    episodes of IThorKitchenOpenFridge: 29 steps/s on CPU against 52 on GPU, for
    identical trajectories -- sim time drives the rollout, so this buys wall
    clock and changes nothing about the result.

    Falls back to onnx-cpu when the CUDA provider is not compiled in (the
    CPU-only onnxruntime wheel), which is exactly when requesting GPU would
    raise instead.
    """
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "onnx-gpu"
    except Exception:
        pass
    return "onnx-cpu"


#: Per-(env, scene) settings applied to the task class before the env is built;
#: see the file's header for what each key does.
TASK_OVERRIDES = pathlib.Path(__file__).with_name("sim_task_overrides.yaml")
_OVERRIDE_KEYS = {"inherit_methods", "fixed_attrs", "scene_spec", "scene_fixtures"}


def load_task_overrides(env_name, scene_id, path=TASK_OVERRIDES):
    """The override entry for ``env_name`` pinned to ``scene_id``, or None."""
    if scene_id is None:
        return None
    import yaml

    table = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    return (table.get(env_name) or {}).get(int(scene_id))


def apply_task_overrides(env_name, scene_id, path=TASK_OVERRIDES) -> list[str]:
    entry = load_task_overrides(env_name, scene_id, path)
    if not entry:
        return []
    from robosuite.environments.base import REGISTERED_ENVS

    return override_task_class(REGISTERED_ENVS[env_name], scene_id, entry)


def override_task_class(cls, scene_id, entry) -> list[str]:
    """Patch ``cls`` (in this process only) with one sim_task_overrides.yaml entry.

    Returns one line per applied setting. Applying the same entry twice is a no-op,
    so a second env built in the same process is not patched again.
    """
    unknown = set(entry) - _OVERRIDE_KEYS
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown task override keys {sorted(unknown)}")
    if vars(cls).get("_task_overrides_scene") == scene_id:
        return []
    applied = []
    for name in entry.get("inherit_methods") or ():
        if name not in vars(cls):
            raise ValueError(f"{cls.__name__} does not define {name}: nothing to bypass")
        setattr(cls, name, _inherited_method(cls, name))
        applied.append(f"{name}: inherited implementation")
    for name, value in (entry.get("fixed_attrs") or {}).items():
        setattr(cls, name, property(lambda self, value=value: value, lambda self, _value: None))
        applied.append(f"{name} = {value!r}")
    if entry.get("scene_spec"):
        _merge_scene_spec(cls, scene_id, entry["scene_spec"])
        applied.append(f"SCENE_SPECS[{scene_id}] <- {sorted(entry['scene_spec'])}")
    fixtures = entry.get("scene_fixtures") or ()
    if fixtures:
        _add_scene_fixtures(cls, fixtures)
        applied.append("scene fixtures: " + ", ".join(f["file"] for f in fixtures))
    cls._task_overrides_scene = scene_id
    return applied


def _inherited_method(cls, name):
    def method(self, *args, **kwargs):
        return getattr(super(cls, self), name)(*args, **kwargs)

    method.__name__ = name
    return method


def _merge_scene_spec(cls, scene_id, values):
    owner = next((c for c in cls.__mro__ if "SCENE_SPECS" in vars(c)), None)
    if owner is None:
        raise ValueError(f"{cls.__name__} has no SCENE_SPECS")
    specs = dict(owner.SCENE_SPECS)
    key = next((k for k in (f"FloorPlan{scene_id}", f"train_{scene_id}", scene_id, str(scene_id))
                if k in specs), None)
    if key is None:
        raise ValueError(f"{cls.__name__}.SCENE_SPECS has no entry for scene {scene_id}")
    specs[key] = {**specs[key], **values}
    cls.SCENE_SPECS = specs


def _add_scene_fixtures(cls, fixtures):
    import xml.etree.ElementTree as ET

    original = cls._setup_baked_scene_fixtures

    def setup(self):
        original(self)
        from robocasa.models.arenas.molmospaces_arena import inline_defaults, remap_geom_groups

        scene_xml = pathlib.Path(self.mujoco_arena.scene_xml)
        for spec in fixtures:
            fixture = ET.parse(scene_xml.parent / spec["file"]).getroot()
            # The extracted bodies use the source scene's named defaults.
            fixture.insert(0, ET.parse(scene_xml).getroot().find("default"))
            inline_defaults(fixture)
            remap_geom_groups(fixture)
            insert_children(self.mujoco_arena.worldbody, fixture.find("worldbody"), spec.get("before_body"))
            contact = fixture.find("contact")
            if contact is not None:
                insert_children(self.mujoco_arena.contact, contact, None, at_start=True)

    cls._setup_baked_scene_fixtures = setup


def insert_children(parent, source, before_name, at_start=False):
    """Insert ``source``'s children into ``parent`` ahead of the child named ``before_name``
    (or at the start / end), keeping their order -- body order fixes the qpos layout."""
    children = list(parent)
    if before_name is not None:
        index = next((i for i, el in enumerate(children) if el.get("name") == before_name), None)
        if index is None:
            raise ValueError(f"no element named {before_name!r} to insert before")
    else:
        index = 0 if at_start else len(children)
    for offset, element in enumerate(list(source)):
        parent.insert(index + offset, element)


def make_env(args, controller_config):
    import robosuite

    import robocasa  # noqa: F401  -- registers the task envs

    if not getattr(args, "no_task_overrides", False):
        for line in apply_task_overrides(args.env, args.scene_id):
            print(f"[eval] task override ({args.env}, scene {args.scene_id}): {line}", flush=True)

    scene_kwargs = {}
    if args.layout is not None:
        scene_kwargs["layout_ids"] = args.layout
    if args.style is not None:
        scene_kwargs["style_ids"] = args.style
    if args.scene_id is not None:
        scene_kwargs["scene_ids"] = args.scene_id
    if args.fixed_scene:
        scene_kwargs["fixed_scene_seed"] = args.scene_seed

    viewer = args.render == "viewer"
    return robosuite.make(
        env_name=args.env,
        robots=args.robot,
        controller_configs=controller_config,
        # Both modes render the ego camera offscreen -- it is the policy's
        # input, not a convenience.
        has_renderer=viewer,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[EGO_CAMERA],
        camera_heights=EGO_H * args.render_scale,
        camera_widths=EGO_W * args.render_scale,
        render_camera=None,
        control_freq=args.control_freq,
        ignore_done=True,
        renderer="mjviewer" if viewer else "mujoco",
        seed=args.seed,
        **scene_kwargs,
    )


EGO_CAMERA = "robot0_head_camera"
EGO_H, EGO_W = 240, 320


def _downsample_image(img: np.ndarray, scale: int) -> np.ndarray:
    """Area-average an integer-factor supersampled RGB camera image.

    OSMesa/llvmpipe can alias high-frequency wall textures at the policy's
    320x240 resolution. Rendering at ``scale`` times that resolution and
    averaging each scale-by-scale pixel block applies a deterministic box
    filter before decimation, without adding a PIL/OpenCV dependency or
    changing the image dtype handed to the policy.
    """
    if scale == 1:
        return img
    if img.ndim != 3 or img.shape[0] % scale or img.shape[1] % scale:
        raise ValueError(
            f"cannot area-downsample image shape {img.shape} by integer scale {scale}")
    h, w, channels = img.shape
    # Keep the accumulation in float32: uint8 overflow would otherwise turn
    # bright texels dark before the average. Round back to the camera dtype so
    # callers (video writer and policy encoder) see the same type as scale=1.
    out = img.reshape(h // scale, scale, w // scale, scale, channels).astype(
        np.float32, copy=False).mean(axis=(1, 3))
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(np.rint(out), info.min, info.max).astype(img.dtype)
    else:
        out = out.astype(img.dtype, copy=False)
    return np.ascontiguousarray(out)


def make_policy(args, command_space):
    """The thing behind get_action: a real server, or a stand-in."""
    from sim2real.vla import PolicyClient, ReplayPolicy, ZeroPolicy

    if args.policy == "server":
        client = PolicyClient(args.host, args.port, timeout_ms=args.policy_timeout_ms)
        print(f"[eval] policy server tcp://{args.host}:{args.port}: "
              f"{'reachable' if client.ping() else 'NOT REACHABLE'}", flush=True)
        return client
    if args.policy == "replay":
        if not args.replay_parquet:
            raise SystemExit("--policy replay needs --replay-parquet")
        from g1_bridge.modality import layout_for

        return ReplayPolicy(
            load_replay_frames(args.replay_parquet, command_space,
                               layout=layout_for(args.backend))
        )
    return ZeroPolicy(zero_shapes(command_space))


#: What a zero action looks like per command space -- key -> width.
_ZERO_SHAPES = {
    "motion_token": {"motion_token": 64, "left_hand_joints": 7, "right_hand_joints": 7},
    "ref_qpos": {"ref_root_pos": 3, "ref_root_quat": 4, "ref_joint_pos": 29,
                 "left_hand_joints": 7, "right_hand_joints": 7},
    "smpl": {"smpl_joints": 72, "left_hand_joints": 7, "right_hand_joints": 7},
    "velocity_cmd": {"torso_vx": 1, "torso_vy": 1, "torso_vyaw": 1, "target_yaw": 1,
                     "rpy": 3, "height": 1, "left_arm": 7, "right_arm": 7,
                     "left_hand_joints": 7, "right_hand_joints": 7},
    # twist_ee (almi) is deliberately absent -- see zero_shapes().
}

#: Command spaces whose null action is not a vector of zeros, with the reason.
#: Better a refusal that says why than a table of plausible zeros: this smoke
#: exists to isolate "the controller stack is broken", and it cannot do that if
#: the action it sends is itself out of distribution.
_NO_ZERO_ACTION = {
    "twist_ee": (
        "a zero action is not a null command for almi: ee_quat would be a zero "
        "quaternion (AlmiCommands.validated() rejects it), a zero base_height "
        "clips to the bottom of the trained band (0.35 m -- a crouch, not a "
        "stand), and a zero ref_joint_pos disagrees with any EE target, which "
        "is the exact pairing that saturates the upper branch. Its real null "
        "pose is the checkpoint's own default_joint_pos, which only the loaded "
        "runtime resolves.\n"
        "  Use --policy replay --replay-parquet <collected almi episode> "
        "instead: it exercises the same adapter with in-distribution actions "
        "and proves more.\n"
        "  For the 'does the stack hold itself up' question on its own, "
        "`lab collect --sim --backend almi --pico-source mock` runs the "
        "identical controller on an idle command."
    ),
}


def zero_shapes(command_space):
    if command_space in _NO_ZERO_ACTION:
        raise SystemExit(
            f"--policy zero is not available for {command_space!r}: "
            f"{_NO_ZERO_ACTION[command_space]}"
        )
    try:
        return _ZERO_SHAPES[command_space]
    except KeyError:
        raise SystemExit(f"--policy zero has no shape table for {command_space!r}") from None


def _action_groups(layout):
    """``[(group name, blob column, start, end)]`` for a blob-schema layout.

    Every backend except sonic packs its action into a couple of concatenated
    LeRobot columns (``action.twist``, ``action.upper_ref``, ``action.hands``,
    ...); the per-group names a policy server returns exist only as byte ranges
    inside them, declared in meta/modality.json. So a replay of a blob-schema
    episode has to slice, not look up a column -- without this, replay found
    none of its keys on heft/amo/almi and said the parquet was wrong.
    """
    blobs = getattr(layout, "blobs", None)
    if not blobs:
        return []
    out = []
    for blob in blobs:
        if blob.section != "action":
            continue
        offset = 0
        for g in blob.groups:
            out.append((g.name, blob.feature_name, offset, offset + g.dim))
            offset += g.dim
    return out


def load_replay_frames(parquet_path, command_space, layout=None):
    """Recorded actions from a collected episode, as the policy would emit them.

    Collection writes ``teleop.*``/``action.*`` columns (see g1_bridge.modality);
    both spellings are accepted since preprocessing renames some of them, and a
    blob-schema backend's groups are sliced out of their packed column.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    groups = _action_groups(layout)

    def col(name):
        for c in (f"action.{name}", f"teleop.{name}", name):
            if c in df.columns:
                return np.stack(df[c].values).astype(np.float32)
        for gname, feature, lo, hi in groups:
            if gname == name and feature in df.columns:
                return np.stack(df[feature].values).astype(np.float32)[:, lo:hi]
        return None

    # The recorded action groups ARE what the policy emits, so they are the
    # better key list whenever the layout gives them. _ZERO_SHAPES is the
    # fallback for sonic, whose legacy schema is one column per signal.
    keys = [name for name, *_ in groups] or list(_ZERO_SHAPES[command_space])
    cols = {k: col(k) for k in keys}
    missing = [k for k, v in cols.items() if v is None]
    have = {k: v for k, v in cols.items() if v is not None}
    if not have:
        raise SystemExit(
            f"{parquet_path} has none of {keys} (columns: {sorted(df.columns)[:20]}...)"
        )
    if missing:
        print(f"[eval] replay: columns not in this parquet, left at zero: {missing}")
    n = min(len(v) for v in have.values())
    return [{k: v[i] for k, v in have.items()} for i in range(n)]


#: The fully-closed Dex3 pose per side, in actuator order (thumb_0,1,2,
#: index_0,1, middle_0,1) -- sim2real.utils.dex3.DEX3_CLOSE_POSE. Used only by
#: --snap-hands; kept here so this module stays importable without sim2real.
_DEX3_CLOSE_POSE = {
    "left": (0.0, 0.724, 1.05, -1.5, -1.5, -1.5, -1.5),
    "right": (0.0, -0.724, -1.05, 1.5, 1.5, 1.5, 1.5),
}


def hand_closure(action: dict) -> tuple:
    """``(left, right)`` closure fractions of the RAW prediction, or None per
    side when the action carries no hand for it.

    1.0 is the operator's fully-closed pose and 0.0 is fully open, so a policy
    that learned the grasp should spend an episode near one or the other. A
    distribution that instead piles up in between is the signature the snapping
    experiment is looking for.
    """
    import numpy as np

    from sim2real.vla.client import action_field

    out = []
    for side in ("left", "right"):
        close = np.asarray(_DEX3_CLOSE_POSE[side], dtype=np.float32)
        frac = None
        for key in (f"{side}_hand_joints", f"{side}_hand"):
            try:
                pred = np.asarray(action_field(action, key), np.float32).reshape(-1)[:7]
            except KeyError:
                continue
            if pred.size == 7:
                frac = float(pred @ close / (close @ close))
            break
        out.append(frac)
    return tuple(out)


def snap_hand_action(action: dict, threshold: float) -> dict:
    """Quantise the predicted grasp to fully-open or fully-closed.

    A diffusion policy regresses a continuous pose, and a hand that is only
    PARTLY closed neither grips nor releases: the Dex3 PD loop drives to
    whatever it is handed, so a prediction that sits at 60% of the close pose
    holds the object with 60% of the finger travel and it slips. Collection
    never produced such a pose -- the trigger is effectively binary -- so this
    snaps each side back onto the two values the operator could actually
    command.

    Returns a NEW dict; the caller's action is not mutated, so the raw
    prediction stays available for logging and comparison.

    This is a deploy-time hypothesis test, not a fix. If it helps, the model's
    grasp signal is directionally right but too weak to close the fingers --
    which points at the training target. If it does not, the failure is
    upstream of the hand entirely (reach, timing, or the object never being in
    the gripper), and no post-processing of the hand will recover it.
    """
    import numpy as np

    from sim2real.vla.client import action_field

    out = dict(action)
    for side in ("left", "right"):
        close = np.asarray(_DEX3_CLOSE_POSE[side], dtype=np.float32)
        for key in (f"{side}_hand_joints", f"{side}_hand"):
            try:
                pred = np.asarray(action_field(action, key), np.float32)
            except KeyError:
                continue
            flat = pred.reshape(-1, 7) if pred.size % 7 == 0 else None
            if flat is None:
                continue
            # Closure as a single scalar: how far along the close-pose
            # direction this prediction sits. Per-joint thresholding would let
            # one finger close while another stays open, which is not a pose
            # the hand can be commanded into.
            frac = (flat @ close) / float(close @ close)
            snapped = (frac > threshold)[:, None] * close[None, :]
            out[key if key in action else f"action.{key}"] = snapped.reshape(pred.shape)
            break
    return out


def torso_state(obs, composite=None) -> dict:
    """`state.*` quantities that are NOT part of the 43-DoF joint vector.

    rpy / height (the torso pose the operator commands) and the raw root quaternion.
    `state_observation` folds whichever of these the backend's own layout packs into
    `observation.state` (amo: rpy + height) into the state vector and hands the rest
    over as extra state; a policy server keeps the keys its embodiment declares and
    ignores the rest.

    These must be the SAME quantities the collector records into `state.rpy` /
    `state.height` (g1_bridge/backends/amo.py::_amo_teleop_data_sample). Since
    schema v7 that is **the torso command in force during the previous tick**
    (Psi0's definition for its G1 wholebody datasets), NOT a measurement -- read
    from the amo backend (``AmoBackend.torso_command_state``) BEFORE this tick's
    action is applied. The previous definition here (waist joints + robosuite's
    `robot0_base_pos` z) was both a different physical quantity from what a v7
    dataset carries and, for the height, robosuite's root frame, which sits
    0.793 m below the pelvis. It survives as the fallback for any backend that
    exposes no ``torso_command_state`` -- today every one except amo, and those
    are exactly the backends whose layouts do not pack rpy/height into
    `observation.state`, so what it produces only ever rides along as extra
    state a policy server is free to ignore.
    """
    import numpy as np

    out: dict = {}
    backend = getattr(getattr(composite, "session", None), "backend", None)
    torso = getattr(backend, "torso_command_state", None)
    if torso is not None:
        rpy, height = torso()
        out["rpy"] = np.asarray(rpy, dtype=np.float32).reshape(3)[None, None]
        out["height"] = np.asarray([height], dtype=np.float32)[None, None]
    else:
        # Legacy fallback (pre-v7 semantics): waist joints + base z.
        q = canonical_joint_pos(composite, obs) if composite is not None else \
            np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
        if q.shape[0] >= 15:
            out["rpy"] = np.array([q[13], q[14], q[12]], dtype=np.float32)[None, None]
        base_pos = obs.get("robot0_base_pos")
        if base_pos is not None:
            out["height"] = np.asarray([np.asarray(base_pos).reshape(-1)[2]],
                                       dtype=np.float32)[None, None]
    quat = base_quat_wxyz(obs)
    if quat is not None:  # heft's state.root_orientation is the raw wxyz quaternion
        out["root_orientation"] = quat[None, None]
    return out


def canonical_joint_pos(composite, obs):
    """The 43-entry state vector, exactly as collection records it.

    29 body joints in the canonical (``G1_JOINT_NAMES``) order with the 7+7
    measured Dex3 fingers interleaved between the arms. Built by the RECORDER's
    own ``sonic_state_43`` rather than assembled here, because the two things a
    re-implementation gets wrong -- where the fingers sit, and which finger
    order they are in -- are both invisible afterwards: the result is 43 floats
    either way, and no consumer downstream can tell a correct one from a
    permuted one.

    Two orderings have to be right, and neither raises when it is not:

    * ``obs["robot0_joint_pos"]`` is MJCF body-tree order -- legs interleaved,
      fingers between the arms. Datasets are in the mapping's order instead,
      because the recording layer reads through the host. Feeding tree order to
      a policy trained on mapping order gives it a permuted state where every
      limb group mixes the wrong joints; it simply sees a pose that never
      occurs and does nothing.
    * the fingers come from ``read_hand_state()`` in Dex3 actuator order
      (thumb_0,1,2, index_0,1, middle_0,1), NOT the MJCF tree's
      thumb/middle/index -- the same mismatch that made sonic's own hand
      columns disagree with its state vector before 2026-08-11.

    Falls back to the observable only when the composite cannot be read at all
    (no session, or a host with no hand reader), which yields the 29-entry
    vector -- build_observation's width check is what says so.
    """
    import numpy as np

    try:
        from g1_bridge.recording.frames import sonic_state_43

        return np.asarray(sonic_state_43(composite), dtype=np.float32).reshape(-1)
    except Exception:
        pass
    reader = getattr(composite, "read_joint_state", None)
    if reader is not None:
        try:
            qpos, _qvel = reader()
            return np.asarray(qpos, dtype=np.float32).reshape(-1)
        except Exception:
            pass
    return np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)


def state_layout_for(backend: str):
    """``(slices, width)`` of this backend's ``observation.state``, from the SAME layout
    table the recorder writes with (``g1_bridge.modality.layout_for``), or None for
    sonic's plain 43-DoF schema (``G1_43DOF_STATE_SLICES``, build_observation's default).

    heft / scalebfm / almi declare the 43-DoF order sonic uses; amo declares 32 entries
    (left_hand, right_hand, left_arm, right_arm, rpy, height). Before this the client
    sent every backend the 43 vector under the 43 slices and relied on the policy
    server picking its groups by NAME -- which happened to be right for amo because its
    hand/arm groups are the same quantities under the same names, and wrong in principle
    (a layout with a different group order or a group the 43 vector does not carry would
    have been fed silently). Reading the layout here makes the two sides agree by
    construction rather than by coincidence.
    """
    from g1_bridge.modality import layout_for

    blobs = getattr(layout_for(backend), "blobs", None)
    if not blobs:
        return None
    for blob in blobs:
        if blob.section == "state" and blob.feature_name == "observation.state":
            slices, offset = {}, 0
            for g in blob.groups:
                slices[g.name] = (offset, offset + g.dim)
                offset += g.dim
            return slices, offset
    return None


def state_observation(composite, obs, backend: str):
    """``(joint_pos, state_slices, extra_state)`` for build_observation, per backend.

    The 43-DoF host read (``sonic_state_43``: 29 joints + measured fingers) and the
    torso / attitude quantities from ``torso_state`` are the pool every layout is
    assembled from; which groups go into ``observation.state``, in what order, is the
    backend's layout. Groups the layout declares that the pool cannot supply are an
    error here, not a zero-filled column the policy reads as a pose that never occurs.
    """
    from sim2real.vla.observation import G1_43DOF_STATE_SLICES

    q43 = canonical_joint_pos(composite, obs)
    extra = torso_state(obs, composite)            # rpy / height / root_orientation, batched
    layout = state_layout_for(backend)
    if layout is None:
        return q43, None, extra
    slices, width = layout
    pool = {name: q43[a:b] for name, (a, b) in G1_43DOF_STATE_SLICES.items()}         if q43.shape[0] == 43 else {}
    pool.update({k: np.asarray(v, dtype=np.float32).reshape(-1) for k, v in extra.items()})
    parts = []
    for name, (a, b) in slices.items():
        v = pool.get(name)
        if v is None or v.shape[0] != b - a:
            raise SystemExit(
                f"--backend {backend}: its observation.state layout needs group "
                f"{name!r} ({b - a} wide), which this host read cannot supply "
                f"(have {sorted(pool)}; {'' if v is None else f'{v.shape[0]} wide'})")
        parts.append(v)
    vec = np.concatenate(parts).astype(np.float32)
    assert vec.shape[0] == width, (vec.shape, width)
    # the groups inside the vector are split out by build_observation; only the
    # remaining measured quantities ride along as extra_state
    extra = {k: v for k, v in extra.items() if k not in slices}
    return vec, slices, extra


def base_quat_wxyz(obs):
    """``robot0_base_quat`` as wxyz -- the convention every consumer assumes.

    robosuite's sensor is ``T.mat2quat(...)``, which returns **xyzw**, while the
    real host's ``read_pelvis_state()[0]`` is **wxyz**; the recording layer
    converts (``robocasa.utils.g1_data_collection._base_quat_wxyz``) so datasets
    carry wxyz. Eval did not, so it fed the model a reordered quaternion: for a
    G1 standing at yaw -0.043 the observation said ``[0, 0, -0.021, 1.0]``,
    which reads back as yaw +pi -- the robot appearing to face exactly backwards
    in every frame the policy saw, in a scene where the task is in front of it.
    A checkpoint that tracks the recording to 1.5 cm open-loop then does nothing
    at all in closed loop, with no error anywhere to say why.

    Returns None when the observation carries no base quaternion.
    """
    import numpy as np

    quat = obs.get("robot0_base_quat")
    if quat is None:
        return None
    q = np.asarray(quat, dtype=np.float32).reshape(-1)
    # xyzw -> wxyz, spelled out rather than via robosuite's convert_quat so this
    # client keeps importing nothing from the simulator package.
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


# Preserve the legacy unseeded stream when --jitter-seed is omitted.
_JITTER_RNG = None


def _apply_reset_jitter(env, base, args, obs, episode=0):
    """Use collection's jitter utility with an independent stream per seeded reset.

    Episode indices start at zero. A fresh generator makes a given episode's
    perturbations independent of RNG consumption in earlier episodes.
    """
    if args.reset_jitter <= 0 and args.reset_object_jitter <= 0:
        return obs
    try:
        from robocasa.utils.scene_initial_jitter import apply_initial_pose_jitter
    except Exception as exc:
        if getattr(args, "jitter_seed", None) is not None:
            raise RuntimeError("seeded evaluation requires the benchmark jitter utility") from exc
        print(f"[locomanilab-eval] --reset-jitter unavailable ({exc}); start pose NOT jittered",
              flush=True)
        return obs

    global _JITTER_RNG
    seed = getattr(args, "jitter_seed", None)
    if seed is None:
        if _JITTER_RNG is None:
            _JITTER_RNG = np.random.default_rng()
        rng = _JITTER_RNG
    else:
        rng = np.random.default_rng(seed + episode)

    if args.reset_jitter > 0:
        # A falsy return here is a real failure: no collision-free pose in 32 tries,
        # so the robot stayed at the nominal placement. Objects are different -- see below.
        if not apply_initial_pose_jitter(
                base, robot_radius=float(args.reset_jitter), object_radius=0.0,
                rng=rng, include_robot=True, include_objects=False):
            print("[locomanilab-eval] robot jitter found no free pose; kept the nominal one",
                  flush=True)
    if args.reset_object_jitter > 0:
        # Falsy is EXPECTED on tasks with no movable objects (articulation tasks own
        # nothing to move), where the helper returns False simply because the joint
        # list came out empty. Only say something when there was something to move.
        moved = apply_initial_pose_jitter(
            base, robot_radius=0.0, object_radius=float(args.reset_object_jitter),
            rng=rng, include_robot=False, include_objects=True)
        if not moved and getattr(base, "objects", None):
            print("[locomanilab-eval] object jitter found no free placement; kept the nominal one",
                  flush=True)
    # The observation was built before the perturbation, so it now describes the old
    # pose. Re-read it, or the policy's first chunk is conditioned on a stale state.
    return base._get_observations() if hasattr(base, "_get_observations") else obs

def _initial_state_digest(base):
    """Fingerprint pre-rollout physics for cross-run seed audits (not a snapshot)."""
    fields = ("qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat")
    digest = hashlib.sha256()
    sizes = {}
    for name in fields:
        values = np.asarray(getattr(base.sim.data, name), dtype="<f8")
        sizes[name] = list(values.shape)
        digest.update(name.encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return {"sha256": digest.hexdigest(), "shapes": sizes}


# --------------------------------------------------------------------------- #
class SceneRandomizer:
    """--randomize: eval-time scene randomization, for robustness sweeps.

    Reuses the benchmark's own ``robocasa.utils.scene_randomization`` /
    ``robocasa.utils.appearance`` utilities -- the SAME code replay
    augmentation runs, driven by the SAME spec parser -- so what a policy was
    trained against under augmentation is what eval perturbs. Everything
    derives deterministically from --randomize-seed (per-episode seeds are
    seed*10000+ep for appearance, seed*1000+ep for the camera, (seed, ep, 555)
    for distractor placement), and the resolved spec is echoed into the results
    json: same spec + seed = same scenes.
    """

    DISTRACTOR_GROUPS = ["cup", "mug", "banana", "can", "bowl", "sponge",
                         "orange", "lemon"]

    def __init__(self, spec: str, seed: int, args) -> None:
        from robocasa.utils.scene_randomization import (
            parse_randomize_spec, randomize_help)

        self.seed = int(seed)
        if str(spec).strip() in ("help", "?", "list"):
            raise SystemExit(randomize_help())
        self.spec_str = str(spec)
        try:
            #: eval draws HELD-OUT textures by default: replay augmentation
            #: writes training data from the 'train' half of the pool, so
            #: 'test' here measures generalization to textures never seen.
            self.spec = parse_randomize_spec(spec, texture_split="test")
        except ValueError as exc:
            raise SystemExit(f"--randomize: {exc}") from None
        if self.spec.edits_appearance and args.scene_id is None \
                and getattr(args, "layout", None) is None:
            # No longer an error: the appearance is regenerated per episode
            # from the seed and applies to whatever room got loaded. But an
            # unpinned eval already varies the room, so the two sources of
            # variation are confounded in the success rate.
            print("[eval] randomize: no --scene-id/--layout -- the room itself "
                  "varies per episode too, so appearance is not the only thing "
                  "changing between episodes", flush=True)
        if self.spec.distractors and not args.fixed_scene:
            print("[eval] randomize: distractors without --fixed-scene -- the "
                  "task placement is freshly sampled every episode anyway, so "
                  "the baseline-restore guard is off", flush=True)
        self._env_name = args.env
        self._fixed_scene = bool(args.fixed_scene)
        self._restores: list = []
        self._appearance = None
        self._lens_warp = None
        self._ref: dict = {}
        self._task_names: list[str] = []
        self._distractor_names: list[str] = []
        self.last_episode_info: dict = {}
        import atexit

        atexit.register(self.cleanup)

    # Spec fields, read straight through so the rest of the class (and the
    # results json) keeps its old shape.
    @property
    def textures(self):
        return self.spec.textures

    @property
    def lights(self):
        return self.spec.lights

    @property
    def camera(self):
        return self.spec.camera

    @property
    def lens(self):
        return self.spec.lens

    @property
    def robot(self):
        return self.spec.robot

    @property
    def distractors(self):
        return self.spec.distractors

    @distractors.setter
    def distractors(self, value):
        self.spec.distractors = int(value)

    def describe(self) -> dict:
        """The spec as the results json should record it.

        Two tokens resolve against the env's scene family (``exclude`` and a
        bare ``lights``), so before the profile is known this is what the user
        typed, and afterwards it is what actually ran.
        """
        out = dict(seed=self.seed, **self.spec.describe())
        if self._appearance is not None:
            cfg = self._appearance.cfg
            out["profile"] = self._appearance.profile.name
            out["exclude"] = list(cfg.texture_exclude)
            if self.spec.lights is not None:
                out["lights"] = cfg.light_jitter
        return out

    def install_appearance(self, base) -> None:
        """After make_env: register the appearance randomizer on the env.

        One seam for every scene family (robosuite's own xml processor, run on
        the merged model xml at every hard reset), so kitchens randomize the
        same way MolmoSpaces rooms do. Raises SystemExit with the profile's own
        message when the env's scene family has no profile.
        """
        if not self.spec.edits_appearance:
            return
        from robocasa.utils.appearance import (
            UnsupportedScene, install_appearance, profile_for)

        try:
            profile = profile_for(base)
        except UnsupportedScene as exc:
            raise SystemExit(f"--randomize: {exc}") from None
        self._appearance = install_appearance(
            base, self.spec.appearance_config(profile), profile)
        self._restores.append(self._appearance.restore)
        print(self.spec.format_resolved(self.spec_str, profile, seed=self.seed),
              flush=True)

    def setup_distractors(self, base) -> None:
        """After make_env: reference reset (distractor-free baseline), then
        wrap the env class's _get_obj_cfgs to append the distractor cfgs."""
        if not self.distractors:
            return
        from robocasa.utils.scene_randomization import (
            capture_baseline, install_distractors)

        base.reset()
        self._task_names = list(getattr(base, "objects", {}) or {})
        if not self._task_names:
            print("[eval] randomize: this task has no movable task objects to "
                  "anchor distractors on; distractors skipped", flush=True)
            self.distractors = 0
            return
        if self._fixed_scene:
            self._ref = capture_baseline(base, self._task_names)
        names, restore, placed = install_distractors(
            base, self._env_name, self.distractors, self.DISTRACTOR_GROUPS,
            np.random.default_rng(self.seed + 777),
            registries=tuple(getattr(base, "obj_registries",
                                     ("objaverse", "lightwheel"))))
        self.distractors = placed
        self._distractor_names = names
        if restore is not None:
            self._restores.append(restore)
        print(f"[eval] randomize: distractors {names}", flush=True)

    def before_reset(self, ep: int) -> None:
        """Point this episode's compile at a fresh appearance."""
        if self._appearance is not None:
            self._appearance.new_episode(self.seed * 10000 + ep)

    def after_reset(self, base) -> None:
        """Pre-jitter: put the task objects back on the distractor-free
        baseline (adding cfgs shifts the env's own rng consumption)."""
        if self._ref:
            from robocasa.utils.scene_randomization import restore_baseline

            if not restore_baseline(base, self._ref):
                print("[eval] randomize: baseline restore failed this episode",
                      flush=True)

    def after_jitter(self, base, ep: int, obs):
        """Post-jitter: place distractors, perturb the camera, refresh obs."""
        info: dict = {}
        if self._distractor_names:
            from robocasa.utils.scene_randomization import teleport_distractors

            info["distractors"] = teleport_distractors(
                base, self._distractor_names, self._task_names,
                np.random.default_rng((self.seed, ep, 555)))
        if self.camera:
            from robocasa.utils.scene_randomization import randomize_camera

            randomize_camera(base, [EGO_CAMERA], self.seed * 1000 + ep,
                             pos=0.01 * self.camera, rot=0.05 * self.camera,
                             fovy=2.0 * self.camera)
            info["camera_seed"] = self.seed * 1000 + ep
        if self.robot:
            from robocasa.utils.scene_randomization import tint_robot_geoms

            info["robot_geoms_tinted"] = tint_robot_geoms(
                base, np.random.default_rng((self.seed, ep, 99)), self.robot)
        if self.lens:
            from robocasa.utils.scene_randomization import LensWarp

            self._lens_warp = LensWarp(EGO_H, EGO_W,
                                       np.random.default_rng((self.seed, ep, 77)),
                                       scale=self.lens)
            info["lens"] = self._lens_warp.describe()
        if self._appearance is not None:
            info["appearance_seed"] = self.seed * 10000 + ep
            # the RESOLVED appearance: which textures, final light params --
            # auditable / reproducible independently of the seed
            info["appearance"] = dict(self._appearance.report)
        self.last_episode_info = info
        if info and hasattr(base, "_get_observations"):
            try:
                obs = base._get_observations(force_update=True)
            except TypeError:
                obs = base._get_observations()
        return obs

    def image(self, img: np.ndarray) -> np.ndarray:
        """The frame as the policy (and the video) should see it: through
        this episode's randomized lens, if any."""
        return self._lens_warp(img) if self._lens_warp is not None else img

    def cleanup(self) -> None:
        # Nothing on disk to remove any more: the appearance randomizer
        # rewrites the model xml in memory on its way into MjSim.
        for restore in self._restores:
            try:
                restore()
            except Exception:
                pass
        self._restores = []
        self._appearance = None


# --------------------------------------------------------------------------- #
def _fix_scene_z_fighting(base) -> int:
    """Hide duplicate Holodeck brick visual layers while preserving physics.

    Some train_117 wall mesh pairs are geometrically identical but carry
    different visual materials. Hiding the brick visual geom avoids depth
    fighting; all affected geoms are non-colliding visual-only geoms.
    """
    import mujoco

    model = base.sim.model
    native = getattr(model, "_model", model)
    hidden = 0
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        matid = int(model.geom_matid[gid])
        matname = (mujoco.mj_id2name(native, mujoco.mjtObj.mjOBJ_MATERIAL, matid)
                   if matid >= 0 else "") or ""
        if (name.startswith("wall_0_") and
                "LivingRoomFireplacebrickTexWHITE" in matname and
                int(model.geom_contype[gid]) == 0 and
                int(model.geom_conaffinity[gid]) == 0):
            model.geom_matid[gid] = -1
            model.geom_rgba[gid, 3] = 0.0
            hidden += 1
    return hidden


def run(args) -> dict:
    from sim2real.vla import VlaDriver, VlaDriverConfig, build_observation
    from g1_bridge.backends import get_backend_class

    command_space = get_backend_class(args.backend).vla_action_space
    if not command_space:
        raise SystemExit(
            f"backend {args.backend!r} has no VLA command space yet -- it can be "
            "teleoperated but not driven by a policy."
        )
    # One line the log auditors can key on (scripts/audit_successes.py): which scene
    # was pinned (or RANDOM), which controller checkpoint, and the fixed-scene seed.
    print(f"[eval] backend={args.backend}  "
          f"command space={command_space}  "
          f"scene={'RANDOM' if args.scene_id is None else args.scene_id}  "
          f"policy_config={args.policy_config or 'default'}  "
          f"fixed_scene={args.fixed_scene}", flush=True)

    rnd = (SceneRandomizer(args.randomize, args.randomize_seed, args)
           if args.randomize else None)
    controller_config = build_controller_config(args)
    env = make_env(args, controller_config)
    base = getattr(env, "unwrapped", env)
    if rnd is not None:
        rnd.install_appearance(base)
        rnd.setup_distractors(base)

    policy = make_policy(args, command_space)
    # No adapter: actions go straight to composite.set_vla_action() below, so
    # the backend builds its own adapter with host-side constructor args --
    # e.g. ref_qpos's frame_builder -- that this client cannot supply. The
    # driver still validates the policy's action keys against the command space.
    driver = VlaDriver(
        policy,
        command_space=command_space,
        config=VlaDriverConfig(
            control_freq=args.control_freq,
            inference_rate=args.inference_rate,
            action_horizon=args.action_horizon,
            ema_alpha=args.ema_alpha,
            max_latency_skip=args.max_latency_skip,
            # A simulator is not realtime, so re-planning synchronously costs
            # nothing and keeps the rollout reproducible. Real deployment is
            # the case that needs the async worker.
            async_inference=False,
        ),
    )

    save_video = args.render == "video" and bool(args.video_dir)
    if save_video:
        pathlib.Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    print(f"[eval] control {args.control_freq}Hz, inference {args.inference_rate}Hz "
          f"-> re-plan every {driver.replan_steps} steps", flush=True)
    print(f"[eval] camera render {EGO_W * args.render_scale}x{EGO_H * args.render_scale} "
          f"-> policy {EGO_W}x{EGO_H} (area scale {args.render_scale}x)", flush=True)

    zero_ac = np.zeros(env.action_dim)
    results = []

    def ego_image(obs) -> np.ndarray:
        img = np.ascontiguousarray(obs[f"{EGO_CAMERA}_image"][::-1])
        img = _downsample_image(img, args.render_scale)
        return rnd.image(img) if rnd is not None else img

    for ep in range(args.episodes):
        if rnd is not None:
            rnd.before_reset(ep)
        obs = env.reset()
        if rnd is not None:
            rnd.after_reset(base)
        if args.fix_z_fighting and args.env.startswith("Holodeck") and args.scene_id == 117:
            hidden = _fix_scene_z_fighting(base)
            print(f"[eval] z-fighting visual fix: hidden {hidden} duplicate geoms", flush=True)
            obs = base._get_observations(force_update=True)
        obs = _apply_reset_jitter(env, base, args, obs, episode=ep)
        if rnd is not None:
            obs = rnd.after_jitter(base, ep, obs)
        initial_state = _initial_state_digest(base)
        composite = env.robots[0].composite_controller  # rebuilt by every reset
        prompt = args.prompt or base.get_ep_meta().get("lang", args.env)
        driver.reset()
        if hasattr(policy, "i"):
            policy.i = 0
        frames, success, t0 = [], False, time.time()
        grip_trace: list = []
        error: str | None = None

        for step in range(args.max_steps):
            # Some controllers open an episode with a settling window during
            # which they discard handed-over actions (robocasa spawns the robot
            # airborne). Querying the policy through it would spend the head of
            # the first action chunk on frames the robot was still falling
            # through, and hand over mid-chunk. Let the controller land first.
            if not getattr(composite, "accepts_vla_action", True):
                obs = env.step(zero_ac)[0]
                continue
            try:
                joint_pos, state_slices, extra_state = state_observation(composite, obs, args.backend)
                action = driver.step(
                    build_observation(
                        joint_pos=joint_pos,
                        state_slices=state_slices,
                        images={"ego_view": ego_image(obs)},
                        prompt=prompt,
                        base_quat_wxyz=base_quat_wxyz(obs),
                        extra_state=extra_state,
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- a timeout / ERROR reply / bad payload
                # One bad request ends THIS episode as an error, not the batch: a
                # 20-episode sweep used to die at the first ZMQ timeout with every
                # finished episode unreported. The REQ socket is stuck mid-handshake
                # after a timeout, so ping() -- which rebuilds it on failure -- is
                # called to leave the client usable for the next episode.
                error = f"{type(exc).__name__}: {exc}"
                print(f"[eval] ep {ep+1}: policy request failed at step {step}: {error}",
                      flush=True)
                if hasattr(policy, "ping"):
                    policy.ping()
                break
            # Record the RAW closure before any snapping: whether the model's
            # grasp signal is merely weak or absent entirely is the question
            # this experiment exists to answer, and snapping destroys the
            # evidence.
            if action is not None:
                grip_trace.append(hand_closure(action))
            if args.snap_hands is not None and action is not None:
                action = snap_hand_action(action, args.snap_hands)
            # G1_BRIDGE routes the dict through the backend's own command space.
            composite.set_vla_action(action)
            obs = env.step(zero_ac)[0]
            if args.render == "viewer":
                env.render()
            elif save_video:
                frames.append(ego_image(obs))
            if bool(base._check_success()):
                success = True
                break

        steps, dt = step + 1, time.time() - t0
        genuine = success and steps >= args.genuine_min_steps
        # What the model ASKED the hand to do this episode, before any snapping.
        # Reported per episode rather than only in aggregate because "never
        # tried to close" and "tried and was too weak" are different failures
        # and the success count alone cannot separate them.
        grip = {}
        for i, side in enumerate(("left", "right")):
            vals = [g[i] for g in grip_trace if g[i] is not None]
            if vals:
                arr = np.asarray(vals, dtype=np.float32)
                grip[side] = dict(
                    peak=round(float(arr.max()), 3),
                    frac_over_half=round(float((arr > 0.5).mean()), 3),
                    frac_partial=round(float(((arr > 0.15) & (arr < 0.85)).mean()), 3),
                )
        results.append(dict(episode=ep + 1, success=success, genuine=genuine,
                            steps=steps, seconds=round(dt, 1), grip=grip,
                            jitter_seed=(args.jitter_seed + ep
                                         if args.jitter_seed is not None else None),
                            initial_state=initial_state,
                            **({"error": error} if error else {}),
                            **({"randomize": rnd.last_episode_info}
                               if rnd is not None and rnd.last_episode_info
                               else {})))
        flag = "" if genuine or not success else f"  (under --genuine-min-steps {args.genuine_min_steps})"
        gtxt = "  ".join(
            f"{s}: peak={g['peak']:.2f} closed={g['frac_over_half']:.0%} "
            f"partial={g['frac_partial']:.0%}" for s, g in grip.items())
        outcome = "SUCCESS" if success else ("ERROR" if error else "timeout")
        print(f"[eval] ep {ep+1}/{args.episodes}: {outcome} "
              f"in {steps} steps ({dt:.1f}s, {steps/max(dt,1e-6):.0f} steps/s){flag}"
              + (f"\n         grip  {gtxt}" if gtxt else ""), flush=True)

        if save_video and frames:
            try:
                import imageio

                tag = "success" if success else ("error" if error else "timeout")
                vp = pathlib.Path(args.video_dir) / f"{args.env}_{args.backend}_ep{ep+1:03d}_{tag}.mp4"
                imageio.mimwrite(vp, frames, fps=args.control_freq)
            except Exception as exc:
                print(f"[eval]   video save failed: {exc}")

    driver.close()
    if rnd is not None:
        rnd.cleanup()
    n = len(results)
    n_succ = sum(r["success"] for r in results)
    n_genuine = sum(r["genuine"] for r in results)
    n_error = sum(1 for r in results if r.get("error"))
    summary = dict(env=args.env, backend=args.backend, command_space=command_space,
                   scene_id=args.scene_id, episodes=n, success=n_succ,
                   genuine=n_genuine, errors=n_error,
                   reset_config=dict(fixed_scene=args.fixed_scene,
                                     scene_seed=args.scene_seed, env_seed=args.seed,
                                     robot_radius=args.reset_jitter,
                                     object_radius=args.reset_object_jitter,
                                     jitter_seed=args.jitter_seed),
                   **({"randomize": rnd.describe()} if rnd is not None else {}),
                   results=results)
    print(f"\n[eval] === {args.env} / {args.backend}: success {n_succ}/{n} "
          f"({100.0*n_succ/max(n,1):.0f}%)"
          + (f", genuine {n_genuine}/{n}" if args.genuine_min_steps else "")
          + (f", {n_error} episode(s) ended in a policy ERROR (counted as failures)" if n_error else "")
          + " ===", flush=True)
    if args.results_json:
        pathlib.Path(args.results_json).write_text(json.dumps(summary, indent=2))
        print(f"[eval] results -> {args.results_json}")
    return summary


def main(argv=None):
    args = parse_args(argv)
    if args.render == "video" and args.video_dir is None:
        args.video_dir = "outputs/vla_eval"
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
