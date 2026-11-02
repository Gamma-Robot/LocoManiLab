#!/usr/bin/env python3
"""Visual sanity check for an already provisioned simulator/controller environment.

Runs the sim client with a zero policy and samples video brightness. A dark image
can indicate occlusion, lighting, capture failure, or a fallen robot. This heuristic
neither measures physical stability nor certifies a model evaluation environment.

Usage:
    python scripts/diagnostics/preflight_sim.py --env EmptyEnv --backend sonic
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from locomanilab import config as cfg  # noqa: E402
from locomanilab.common import sim as simmod  # noqa: E402
from locomanilab.maintenance.doctor import controller_manifest  # noqa: E402
from locomanilab.runner import Runner  # noqa: E402

# Darkness thresholds are a visual heuristic; interpretation depends on the scene.
NEAR_BLACK_LEVEL = 20      # pixel mean below this counts as "black"
MAX_NEAR_BLACK = 0.05


def frame_stats(mp4: Path, n: int):
    """(mean brightness, near-black fraction) of frame n, or None if it doesn't exist."""
    import numpy as np
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "f.png")
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4),
                        "-vf", rf"select=eq(n\,{n})", "-vsync", "0", out, "-y"],
                       check=False)
        if not os.path.exists(out):
            return None
        a = np.asarray(Image.open(out).convert("RGB"), dtype="float32")
    return float(a.mean()), float((a.mean(axis=2) < NEAR_BLACK_LEVEL).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--scene-id", type=int, default=None)
    ap.add_argument("--backend", default="sonic", help="sonic | amo | heft | scalebfm (almi has no zero action)")
    ap.add_argument("--sonic-ckpt", default=None, help="low_latency | v1_1 (sonic only)")
    ap.add_argument("--policy-config", default=None, help="explicit sim2real policy.yaml")
    ap.add_argument("--sim-env", default="tele_benchmark", help="conda env the sim client runs in")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=200)
    a = ap.parse_args()

    try:
        ctx = cfg.load_context()
        policy_config = controller_manifest(ctx, backend=a.backend, sonic_ckpt=a.sonic_ckpt,
                                            policy_config=a.policy_config)
    except cfg.ConfigError as exc:
        ap.error(str(exc))
    support = simmod.detect(ctx, conda_env=a.sim_env, backend=a.backend, policy_config=policy_config)
    if not support.ok:
        print(support.summary())
        return 2
    with tempfile.TemporaryDirectory() as vd:
        extra = []
        if policy_config:
            extra += ["--policy-config", str(policy_config)]
        cmd, extra_env = simmod.build_client_cmd(
            sim_repo=support.sim_repo, env_name=a.sim_env, host="localhost", port=5550,
            model=a.env, prompt="preflight", episodes=a.episodes, max_steps=a.max_steps,
            action_horizon=1, render="video", video_dir=vd, backend=a.backend, policy="zero",
            runtime_repo=support.runtime_repo,
            extra=extra + (["--scene-id", str(a.scene_id)] if a.scene_id is not None else []))
        setup = ctx.path("conda_setup")
        argv = Runner(conda_root=ctx.conda_root).wrap(
            cmd, env_kind="conda", env_name=a.sim_env, setup=str(setup) if setup else None)
        print("preflight (no model, --policy zero):")
        print("  cd " + str(support.sim_repo))
        print("  " + " ".join(argv) + "\n")
        subprocess.run(argv, cwd=str(support.sim_repo), env={**os.environ, **extra_env},
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        vids = sorted(Path(vd).glob("*.mp4"))
        if not vids:
            print("FAIL no videos produced -- the sim client did not run (re-run without "
                  "stdout suppressed: the command is printed above)")
            return 1

        bad = []
        sampled = 0
        for v in vids:
            for n in (0, 20, a.max_steps - 1):
                s = frame_stats(v, n)
                if s is None:
                    continue
                sampled += 1
                mean, black = s
                flag = "DARK" if black > MAX_NEAR_BLACK else "ok"
                print(f"  {v.name[:44]:44s} f{n:<4d} brightness={mean:6.1f} "
                      f"near-black={black*100:5.1f}%  {flag}")
                if black > MAX_NEAR_BLACK:
                    bad.append((v.name, n, black))

    if not sampled:
        print("FAIL no video frames could be decoded; check ffmpeg and the capture output.")
        return 1
    if bad:
        print(f"\nCHECK {len(bad)} frame(s) exceeded {MAX_NEAR_BLACK*100:.0f}% near-black.")
        print("Inspect capture, lighting, occlusion, and robot state before interpreting the run.")
        return 1

    print("\nVisual brightness check passed. This does not establish physical stability or task success.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
