#!/usr/bin/env python3
"""Round-trip an XR-1 compatible policy server with synthetic observations.

Checks ping and get_action response keys, numeric values, and chunk shapes. This
verifies protocol structure only; it does not establish task performance or robot
readiness. Run in an environment with the server client's numerical dependencies.

Usage:
    python scripts/diagnostics/xr1_roundtrip.py --port 5550 --prompt 'example instruction'
"""
from __future__ import annotations

import argparse
import sys
import time

import msgpack_numpy as mnp
import numpy as np
import zmq

EGO_H, EGO_W = 240, 320
STATE_GROUPS = {
    "left_leg": 6, "right_leg": 6, "waist": 3,
    "left_arm": 7, "left_hand": 7, "right_arm": 7, "right_hand": 7,
}
EXPECTED = {
    "action.motion_token": 64,
    "action.left_hand_joints": 7,
    "action.right_hand_joints": 7,
}


def call(sock, endpoint, data=None, timeout_ms=600_000):
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.send(mnp.packb({"endpoint": endpoint, "data": data or {}}, use_bin_type=True))
    return mnp.unpackb(sock.recv(), raw=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--prompt", default="walk over to the toilet and stand next to it")
    parser.add_argument("--calls", type=int, default=2, help="get_action calls (2nd shows warm latency)")
    args = parser.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{args.host}:{args.port}")

    print(f"ping -> {call(sock, 'ping')}")

    rng = np.random.default_rng(0)
    observation = {
        "video": {"ego_view": rng.integers(0, 255, (1, 1, EGO_H, EGO_W, 3), dtype=np.uint8)},
        "state": {
            name: rng.normal(0, 0.3, (1, 1, width)).astype(np.float32)
            for name, width in STATE_GROUPS.items()
        },
        "language": {"annotation.human.task_description": [[args.prompt]]},
    }
    observation["state"]["projected_gravity"] = np.array([[[0.0, 0.0, -1.0]]], dtype=np.float32)

    failures = []
    for index in range(args.calls):
        t0 = time.monotonic()
        reply = call(sock, "get_action", {"observation": observation})
        elapsed = time.monotonic() - t0
        if isinstance(reply, dict) and "error" in reply:
            print(f"call {index}: SERVER ERROR {reply['error']}")
            return 1
        action, info = reply
        print(f"call {index}: {elapsed * 1000:.0f} ms wall, info={info}")
        for key, width in EXPECTED.items():
            if key not in action:
                failures.append(f"missing key {key} (got {sorted(action)})")
                continue
            arr = np.asarray(action[key])
            ok_shape = arr.ndim == 2 and arr.shape[1] == width
            ok_dtype = arr.dtype == np.float32
            ok_finite = np.isfinite(arr).all()
            print(f"    {key:26s} shape={arr.shape} dtype={arr.dtype} "
                  f"std={arr.std():.4f} range=[{arr.min():.3f}, {arr.max():.3f}]")
            if not ok_shape:
                failures.append(f"{key}: shape {arr.shape}, expected (H, {width})")
            if not ok_dtype:
                failures.append(f"{key}: dtype {arr.dtype}, expected float32")
            if not ok_finite:
                failures.append(f"{key}: contains non-finite values")

    if failures:
        print("\nFAIL:")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("\nOK: ping + get_action return motion_token(H,64) / left_hand(H,7) / right_hand(H,7) float32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
