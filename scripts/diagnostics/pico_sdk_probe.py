"""Inspect controller input from an already provisioned xrobotoolkit_sdk service.

Stop competing collector clients first; the service may support one client only.
Wear the headset, keep its compatible application in the foreground, and exercise
controller buttons and sticks. This diagnostic does not install an SDK or pair a
headset.

Usage:
    python scripts/diagnostics/pico_sdk_probe.py --seconds 30
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _headset_connected() -> str:
    try:
        out = subprocess.run(
            ["ss", "-tnH", "state", "established", "( sport = :63901 )"],
            capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.split()[-1] if out else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    peer = _headset_connected()
    if not peer:
        print("no headset TCP connection on :63901 -- connect the PC in the headset app first "
              "(see docs/pico_teleop_troubleshooting.md §3)")
        return 1
    print(f"headset connected from {peer}")
    if subprocess.run(["pgrep", "-f", "g1_bridge.collect_real"], capture_output=True).stdout.strip():
        print("WARNING: a collector is running; this second SDK connection will read zeros. Stop it first.")

    import xrobotoolkit_sdk as xrt  # noqa: WPS433
    xrt.init()
    time.sleep(1.0)
    t0 = time.monotonic()
    pressed: set[str] = set()
    max_trig = max_axis = 0.0
    ctrl_seen = body_seen = False
    n = 0
    while time.monotonic() - t0 < args.seconds:
        n += 1
        btn = {k: bool(getattr(xrt, f"get_{k}_button")()) for k in "ABXY"}
        pressed |= {k for k, v in btn.items() if v}
        lt, rt = float(xrt.get_left_trigger()), float(xrt.get_right_trigger())
        la, ra = list(xrt.get_left_axis()), list(xrt.get_right_axis())
        max_trig = max(max_trig, lt, rt)
        max_axis = max(max_axis, *(abs(v) for v in la + ra))
        lp = [round(v, 2) for v in list(xrt.get_left_controller_pose())[:3]]
        rp = [round(v, 2) for v in list(xrt.get_right_controller_pose())[:3]]
        ctrl_seen |= any(lp) or any(rp)
        body = bool(xrt.is_body_data_available())
        body_seen |= body
        if n % 10 == 0:
            print(f"{time.monotonic() - t0:5.1f}s btn={''.join(k for k, v in btn.items() if v) or '-':4} "
                  f"trig=({lt:.2f},{rt:.2f}) axis={la}/{ra} Lpos={lp} Rpos={rp} "
                  f"handL={xrt.get_left_hand_is_active()} handR={xrt.get_right_hand_is_active()} body={body}",
                  flush=True)
        time.sleep(0.1)
    xrt.close()
    print("\nSUMMARY")
    print(f"  buttons pressed : {sorted(pressed) or 'NONE'}")
    print(f"  max trigger     : {max_trig:.2f}   max stick: {max_axis:.2f}")
    print(f"  controller pose : {'seen' if ctrl_seen else 'NEVER (controllers not streaming)'}")
    print(f"  body tracking   : {'seen' if body_seen else 'NEVER'}")
    if not pressed and not ctrl_seen:
        print("  -> headset is not sending controller data: wake/pair the controllers, keep the "
              "XRoboToolkit app in the foreground, wear the headset (proximity sensor) and retry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
