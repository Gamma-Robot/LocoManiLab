#!/usr/bin/env python3
"""Check a gr00t-compatible policy server against the shared wire contract.

Verifies that get_action returns the expected response envelope, action keys,
chunk dimensions, and finite values. Optionally derives the observation/action
layout from a LeRobot dataset instead of the default SONIC layout.

Usage:
    python scripts/diagnostics/check_gr00t_server.py --port 5550 --dataset /path/to/dataset
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# Default contract = the sonic embodiment. Pass --dataset to take BOTH tables from that
# dataset's meta/modality.json instead (any backend: heft/amo/almi state and action
# groups), which is what keeps this script from carrying its own copy of a layout --
# the earlier hardcoded 43-DoF table reported FAIL against every non-sonic server.
SONIC_ACTION_DIMS = {"motion_token": 64, "left_hand_joints": 7, "right_hand_joints": 7}


def sonic_state_dims() -> dict[str, int]:
    try:
        from sim2real.vla.observation import G1_43DOF_STATE_SLICES
        return {k: hi - lo for k, (lo, hi) in G1_43DOF_STATE_SLICES.items()}
    except ImportError:   # not in the sim env: the one place the layout is spelled out here
        return {"left_leg": 6, "right_leg": 6, "waist": 3, "left_arm": 7,
                "left_hand": 7, "right_arm": 7, "right_hand": 7}


def dims_from_dataset(dataset: str) -> tuple[dict[str, int], dict[str, int]]:
    """(state dims, action dims) declared by <dataset>/meta/modality.json."""
    import json
    from pathlib import Path
    mod = json.loads((Path(dataset) / "meta" / "modality.json").read_text())
    state = {k: int(v.get("end", 0)) - int(v.get("start", 0)) for k, v in (mod.get("state") or {}).items()}
    action = {k: int(v.get("end", 0)) - int(v.get("start", 0)) for k, v in (mod.get("action") or {}).items()}
    return state, action


def main() -> int:
    import msgpack_numpy as mnp
    import zmq

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5550)
    ap.add_argument("--prompt", default="walk over to the toilet and stand next to it")
    ap.add_argument("--timeout", type=int, default=180_000)
    ap.add_argument("--dataset", default=None,
                    help="take the state/action contract from this dataset's meta/modality.json "
                         "(default: the sonic embodiment)")
    a = ap.parse_args()
    if a.dataset:
        state_dims, action_dims = dims_from_dataset(a.dataset)
    else:
        state_dims, action_dims = sonic_state_dims(), dict(SONIC_ACTION_DIMS)

    sock = zmq.Context().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, a.timeout)
    sock.connect(f"tcp://{a.host}:{a.port}")

    def call(endpoint, data=None):
        sock.send(mnp.packb({"endpoint": endpoint, "data": data or {}}))
        return mnp.unpackb(sock.recv(), raw=False)

    fail = []
    print("ping ->", call("ping"))

    # EXACTLY what deploy_living_room_vla.py builds -- copying its shapes is the whole point.
    # It sends img[np.newaxis, np.newaxis] => (1, 1, H, W, 3) under key "ego_view", and each
    # state slice as (1, 1, D), plus an EXTRA "projected_gravity" that is not part of the 43.
    # An earlier version of this script used (1, H, W, 3)/(1, 43) and therefore PASSED against
    # a server that then crashed on the real client with a 5-dim tensor.
    img = np.random.randint(0, 255, (240, 320, 3), np.uint8)
    state = {k: np.zeros((1, 1, d), np.float32) for k, d in state_dims.items()}
    state.setdefault("projected_gravity", np.array([[[0.0, 0.0, -1.0]]], np.float32))
    obs = {
        "video": {"ego_view": img[np.newaxis, np.newaxis]},
        "state": state,
        "language": {"annotation.human.task_description": [[a.prompt]]},
    }
    resp = call("get_action", {"observation": obs, "options": None})

    # THE check the old test skipped: the client indexes resp[0]/resp[1].
    if isinstance(resp, dict) and "error" in resp:
        print(f"FAIL server returned an error: {str(resp['error'])[:300]}")
        return 1
    if not isinstance(resp, (list, tuple)):
        print(f"FAIL get_action returned {type(resp).__name__}, but the sim client does "
              f"`resp[0]` and needs a list/tuple [action, info]. A dict here makes the "
              f"client die with a misleading `KeyError: 0`.")
        return 1
    action = resp[0]
    print(f"ok   wire shape: list(len={len(resp)}) -> [action, info]")

    def field(name):               # servers answer either "motion_token" or "action.motion_token"
        for k in (name, f"action.{name}"):
            if k in action:
                return k
        return None

    def chunk(k):
        # Same normalisation as the sim client's `as_chunk`: (1, T, D) -> (T, D).
        arr = np.asarray(action[k])
        while arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    found = {}
    for name, dim in action_dims.items():
        k = field(name)
        if k is None:
            fail.append(f"missing key {name} (as '{name}' or 'action.{name}')"); continue
        found[name] = k
        arr = chunk(k)
        if arr.ndim != 2 or arr.shape[1] != dim:
            fail.append(f"{k}: shape {arr.shape}, expected (T, {dim})")
        elif arr.dtype != np.float32:
            fail.append(f"{k}: dtype {arr.dtype}, expected float32")
        else:
            print(f"ok   {k}: {arr.shape} {arr.dtype}")

    horizons = {chunk(k).shape[0] for k in found.values()}
    if len(horizons) > 1:
        fail.append(f"action streams disagree on horizon: {horizons}")

    if fail:
        print("\nFAIL:"); [print("  -", f) for f in fail]
        return 1
    print("\nPASS — server speaks the sim client's contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
