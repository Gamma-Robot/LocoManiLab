"""Offline open-loop inference check (client).

Feed frames from a *training* LeRobot dataset to a running gr00t-compat policy
server (tcp://<host>:<port>, `get_action`) and plot the predicted actions against
the dataset's ground-truth actions, per dimension.

This is model-agnostic: every locomanilab model exposes the same `get_action` server, so
one client validates all of them. It runs in the sim client conda env (needs zmq +
msgpack_numpy + opencv + matplotlib, and sim2real -- g1_backend -- for the wire
client and the state layout); locomanilab starts the right model server and invokes this
via `lab infer-check`.

Observation / action wire format mirrors the sim client (locomanilab/runtime/sim_client.py)
and the dataset's meta/modality.json (unitree_g1_sonic embodiment). The client and
the 43-DoF state table come from `sim2real.vla` rather than being copied here: a
copy that drifts produces a check that passes while the deployed policy reads the
wrong slice (observation.py's own header says as much).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from sim2real.vla.client import PolicyClient as ServerPolicy  # zmq REQ + msgpack_numpy
from sim2real.vla.observation import G1_43DOF_STATE_SLICES as STATE_SLICES

# policy action key (as returned by get_action) -> ground-truth column in the dataset parquet.
# Different checkpoints predict different heads (motion_token latent, SMPL joints, hand joints);
# we compare whatever the model returns against the matching GT column.
ACTION_MAP = {
    "motion_token": "action.motion_token",
    "smpl_joints": "teleop.smpl_joints",
    "smpl_pose": "teleop.smpl_pose",
    "left_hand_joints": "teleop.left_hand_joints",
    "right_hand_joints": "teleop.right_hand_joints",
}


def _as_chunk(a) -> np.ndarray:
    """Coerce a policy field to (T, D)."""
    a = np.asarray(a, dtype=np.float32)
    a = np.squeeze(a)
    if a.ndim == 1:
        a = a[None]
    return a.reshape(a.shape[0], -1)


def _field(action: dict, key: str):
    for k in (key, f"action.{key}"):
        if k in action and action[k] is not None:
            return _as_chunk(action[k])
    return None


def build_obs(state_vec, proj_grav, img, prompt) -> dict:
    """GR00T observation dict with leading [batch=1, time=1] dims (mirrors the sim client)."""
    state = {n: np.asarray(state_vec[a:b], np.float32)[None, None] for n, (a, b) in STATE_SLICES.items()}
    if proj_grav is not None:
        state["projected_gravity"] = np.asarray(proj_grav, np.float32)[None, None]
    return {
        "video": {"ego_view": np.ascontiguousarray(img).astype(np.uint8)[None, None]},
        "state": state,
        "language": {"annotation.human.task_description": [[prompt]]},
    }


def _episode_files(dataset: str, episode: int):
    info = json.load(open(os.path.join(dataset, "meta", "info.json")))
    pqs = sorted(glob.glob(os.path.join(dataset, "data", "chunk-*", "episode_*.parquet")))
    if not pqs:
        raise SystemExit(f"no episode parquet under {dataset}/data/chunk-*/")
    if episode >= len(pqs):
        raise SystemExit(f"--episode {episode} out of range (dataset has {len(pqs)} episodes)")
    pq = pqs[episode]
    idx = int(os.path.basename(pq).split("_")[1].split(".")[0])
    chunk = idx // int(info.get("chunks_size", 1000))
    vid = os.path.join(dataset, "videos", f"chunk-{chunk:03d}", "observation.images.ego_view",
                       f"episode_{idx:06d}.mp4")
    return pq, vid, idx, int(info.get("fps", 50))


def _read_video(path: str, n: int) -> np.ndarray:
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit(f"could not read frames from {path}")
    while len(frames) < n:              # pad with last frame if video is short
        frames.append(frames[-1])
    return np.stack(frames[:n])


def run(dataset: str, *, host: str, port: int, episode: int, frames: int | None,
        out: str, prompt: str, timeout_ms: int = 120000) -> None:
    import pandas as pd
    pq, vid, idx, fps = _episode_files(dataset, episode)
    df = pd.read_parquet(pq).sort_values("frame_index").reset_index(drop=True)
    n = len(df) if frames is None else min(frames, len(df))
    df = df.iloc[:n]
    state = np.stack(df["observation.state"].values).astype(np.float32)
    pg = (np.stack(df["observation.projected_gravity"].values).astype(np.float32)
          if "observation.projected_gravity" in df.columns else None)
    imgs = _read_video(vid, n)
    print(f"[infer-check] episode {idx}: {n} frames @ {fps}fps, state={state.shape}")

    policy = ServerPolicy(host, port, timeout_ms=timeout_ms)

    def obs_at(t):
        return build_obs(state[t], pg[t] if pg is not None else None, imgs[t], prompt)

    # discover which action heads this checkpoint returns; keep those with a matching GT column
    first = policy.get_action(obs_at(0))
    keys = []
    for k in list(first):
        base = k[len("action."):] if k.startswith("action.") else k
        if base in ACTION_MAP and ACTION_MAP[base] in df.columns and first[k] is not None:
            keys.append(base)
    if not keys:
        raise SystemExit(f"no comparable action heads: server returned {list(first)}; "
                         f"known heads {list(ACTION_MAP)}")
    dims = {k: int(_field(first, k).shape[-1]) for k in keys}
    print("[infer-check] comparing heads: " + ", ".join(f"{k}({dims[k]})->{ACTION_MAP[k]}" for k in keys))

    gt = {k: np.stack(df[ACTION_MAP[k]].values).astype(np.float32).reshape(n, -1)[:, :dims[k]] for k in keys}
    pred = {k: np.full((n, dims[k]), np.nan, np.float32) for k in keys}

    def place(action, t):
        H = 1
        for k in keys:
            f = _field(action, k)
            if f is None:
                continue
            f = f[:, :dims[k]]
            H = max(H, f.shape[0])
            m = min(f.shape[0], n - t)
            pred[k][t:t + m] = f[:m]
        return H

    H = place(first, 0)
    t = H
    while t < n:
        H = place(policy.get_action(obs_at(t)), t)
        t += max(1, min(H, n - t))
    print(f"[infer-check] action horizon per get_action = {H}")

    mses = {k: float(np.nanmean((pred[k] - gt[k]) ** 2)) for k in keys}
    print("[infer-check] per-head MSE: " + ", ".join(f"{k}={v:.5f}" for k, v in mses.items()))
    # temporal std per head (pred vs GT) — a flat pred (std<<GT) means the model
    # barely moves in open loop; compare against the closed-loop applied-token std.
    for k in keys:
        ps = float(np.nanmean(np.nanstd(pred[k], axis=0)))
        gs = float(np.nanmean(np.nanstd(gt[k], axis=0)))
        print(f"[infer-check] temporal std {k}: pred={ps:.4f}  GT={gs:.4f}")
    # Per-dimension TEMPORAL CORRELATION between pred and GT. This is the metric that
    # actually says whether the policy tracks the demonstration, and MSE alone can hide a
    # total failure: a latent action like SONIC's motion_token has a small absolute range,
    # so a model that emits a roughly-right-magnitude constant scores a respectable MSE
    # while being uncorrelated with the observation. Measured on a G1 SONIC checkpoint
    # whose MSE (0.044) and temporal-std ratio (1.38x) both looked reasonable: median
    # correlation was +0.04 with 0/64 dims above 0.5 -- i.e. the predictions did not follow
    # the demo at all, which in closed loop is a robot drifting one way until it falls.
    # Read this BEFORE the MSE.
    for k in keys:
        p_, g_ = pred[k], gt[k]
        cs = []
        for d in range(p_.shape[1]):
            a, b = p_[:, d], g_[:, d]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 2 and np.std(a[m]) > 1e-9 and np.std(b[m]) > 1e-9:
                cs.append(float(np.corrcoef(a[m], b[m])[0, 1]))
        if cs:
            cs = np.asarray(cs)
            print(f"[infer-check] temporal corr {k}: median={np.median(cs):+.3f}  "
                  f">0.5: {int((cs > 0.5).sum())}/{len(cs)} dims  "
                  f"<0: {int((cs < 0).sum())}/{len(cs)}")
    if out:
        np.save(str(out).rsplit(".", 1)[0] + "_pred.npy", pred[keys[0]])
    ds_name = os.path.basename(dataset.rstrip("/"))
    _plot(gt, pred, keys, dims, out, fps=fps,
          title=f"{ds_name} ep{idx}  " + "  ".join(f"{k} MSE={mses[k]:.4f}" for k in keys))
    print(f"[infer-check] saved plot -> {out}")


def _plot(gt, pred, keys, dims, out, *, fps, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{k}[{i}]" for k in keys for i in range(dims[k])]
    G = np.concatenate([gt[k] for k in keys], axis=1)
    P = np.concatenate([pred[k] for k in keys], axis=1)
    dim = G.shape[1]
    cols = 6
    rows = (dim + cols - 1) // cols
    t = np.arange(G.shape[0]) / fps
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 1.8 * rows), squeeze=False)
    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= dim:
            ax.axis("off")
            continue
        ax.plot(t, G[:, i], color="#1f77b4", lw=1.2, label="GT")
        ax.plot(t, P[:, i], color="#ff7f0e", lw=1.0, ls="--", label="pred")
        ax.set_title(labels[i], fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="offline open-loop pred-vs-GT action check")
    p.add_argument("--dataset", required=True)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5550)
    p.add_argument("--episode", type=int, default=0, help="ordinal index into the dataset's episodes")
    p.add_argument("--frames", type=int, default=None, help="cap frames (default: whole episode)")
    p.add_argument("--out", default="runs/infer_check/pred_vs_gt.png")
    p.add_argument("--prompt", default="test")
    p.add_argument("--timeout-ms", type=int, default=120000)
    a = p.parse_args()
    run(a.dataset, host=a.host, port=a.port, episode=a.episode, frames=a.frames,
        out=a.out, prompt=a.prompt, timeout_ms=a.timeout_ms)


if __name__ == "__main__":
    main()
