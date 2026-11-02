#!/usr/bin/env python3
"""Open-loop check of a running GR00T policy server on recorded benchmark frames.

Sends real frames (ego video + 43-D joint state + projected gravity + task prompt) from a
LeRobot v2.1 SONIC dataset to the server and compares the first --horizon predicted steps
with the recorded actions, next to two trivial baselines (all zeros, hold previous action).
PASS when every prediction is finite and the motion-token error beats the zero baseline.

Run it in an environment with zmq, msgpack-numpy, opencv, and pandas (the Isaac-GR00T
environment has all of them):

    external/Isaac-GR00T/.venv/bin/python scripts/hf/gr00t_openloop.py \\
        --dataset datasets/hf/dataset/main/atomic/seen/IThorKitchenOpenFridge_scene13 --port 5550

This checks the served model against its training data; it is not a closed-loop success rate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

STATE_GROUPS = ["left_leg", "right_leg", "waist", "left_arm", "left_hand", "right_arm", "right_hand"]
ACTION_COLUMNS = {"motion_token": "action.motion_token", "left_hand_joints": "teleop.left_hand_joints",
                  "right_hand_joints": "teleop.right_hand_joints"}


def chunk(action, key):
    # Same normalisation as the sim client's `as_chunk`: (1, T, D) -> (T, D).
    arr = np.asarray(action.get(key, action.get(f"action.{key}")), np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def main() -> int:
    import cv2
    import msgpack_numpy as mnp
    import pandas as pd
    import zmq

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="one task directory (meta/, data/, videos/)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5550)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 10, 25])
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=180_000, help="per request, milliseconds")
    a = ap.parse_args()

    root = Path(a.dataset)
    modality = json.loads((root / "meta/modality.json").read_text())
    prompt = json.loads((root / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
    total = json.loads((root / "meta/info.json").read_text())["total_episodes"]

    sock = zmq.Context().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, a.timeout)
    sock.connect(f"tcp://{a.host}:{a.port}")

    def call(endpoint, data=None):
        sock.send(mnp.packb({"endpoint": endpoint, "data": data or {}}))
        return mnp.unpackb(sock.recv(), raw=False)

    def frame(episode, t):
        video = root / f"videos/chunk-000/observation.images.ego_view/episode_{episode:06d}.mp4"
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"cannot read frame {t} of {video}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    print(f"task: {prompt!r}; ping: {call('ping')}")
    rows = []
    for episode in [e for e in a.episodes if e < total]:
        df = pd.read_parquet(root / f"data/chunk-000/episode_{episode:06d}.parquet")
        n = len(df)
        joints = np.stack(df["observation.state"].to_numpy())
        gravity = np.stack(df["observation.projected_gravity"].to_numpy())
        for t in (5, n // 4, n // 2, 3 * n // 4):
            if t < 1 or t + a.horizon >= n:
                continue
            state = {g: joints[t, modality["state"][g]["start"]:modality["state"][g]["end"]][None, None]
                     .astype(np.float32) for g in STATE_GROUPS}
            state["projected_gravity"] = gravity[t][None, None].astype(np.float32)
            obs = {"video": {"ego_view": frame(episode, t)[None, None]}, "state": state,
                   "language": {"annotation.human.task_description": [[prompt]]}}
            resp = call("get_action", {"observation": obs, "options": None})
            if isinstance(resp, dict) and "error" in resp:
                print(f"FAIL server error: {str(resp['error'])[:300]}")
                return 1
            for key, column in ACTION_COLUMNS.items():
                truth = np.stack(df[column].to_numpy()).astype(np.float32)
                pred = chunk(resp[0], key)[:a.horizon]
                gt = truth[t:t + a.horizon]
                rows.append({"key": key, "finite": bool(np.isfinite(pred).all()),
                             "model": float(((pred - gt) ** 2).mean()),
                             "zero": float((gt ** 2).mean()),
                             "hold": float(((truth[t - 1][None] - gt) ** 2).mean())})
    if not rows:
        print("FAIL no usable samples (check --episodes)")
        return 1
    table = pd.DataFrame(rows)
    print(table.groupby("key")[["model", "zero", "hold"]].mean().round(4).to_string())
    motion = table[table.key == "motion_token"]
    ok = bool(table.finite.all()) and motion.model.mean() < motion.zero.mean()
    print(f"{len(rows) // len(ACTION_COLUMNS)} samples; all finite: {bool(table.finite.all())}")
    print("PASS" if ok else "FAIL", "— motion_token MSE vs zero baseline:",
          f"{motion.model.mean():.4f} vs {motion.zero.mean():.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
