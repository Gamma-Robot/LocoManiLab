#!/usr/bin/env python3
"""gr00t-compat policy server for lerobot baseline policies (ACT / Diffusion Policy).

locomanilab drives every model — real G1 and tele sim alike — through one contract: a
gr00t-compatible ZMQ REQ/REP server on tcp://0.0.0.0:5550 answering ``get_action``
(msgpack-numpy wire, endpoints ping/kill/reset/get_action/get_modality_config —
mirrors external/Psi0/src/psi/deploy/psi0_gr00t_server.py, the newest precedent).
This file wraps a lerobot ``ACTPolicy`` / ``DiffusionPolicy`` checkpoint trained on
the flat-action datasets produced by locomanilab.runtime.lerobot.convert
(observation.state[43] + action[78] = motion_token 64 ⊕ left_hand 7 ⊕ right_hand 7).

Every other model's server lives in that model's own repo (FastWAM's
deployment/run_fastwam_server.py, PRTS's, starVLA's, Psi0's). This one is the
exception: external/lerobot is UNFORKED upstream, so its working tree stays
pristine and the server lives in locomanilab instead. It still RUNS inside the lerobot env:
    cd external/lerobot && uv run --no-sync python /path/to/locomanilab/runtime/lerobot/server.py \
        --ckpt <run_output_dir | .../pretrained_model> --port 5550

Notes
-----
* ACT.predict_action_chunk is self-contained (normalizes inputs itself).
* DiffusionPolicy.predict_action_chunk expects its obs queues pre-populated with
  NORMALIZED inputs (select_action does that) — we replicate that dance and call
  policy.reset() on the reset endpoint so episode boundaries clear the history.
* Neither policy is language-conditioned: the prompt is accepted and ignored
  (warned once) so the shared sim client/protocol needs no special-casing.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

MOTION_TOKEN_DIM, HAND_DIM, ACTION_DIM = 64, 7, 78
PROPRIO_DIM = 43
STATE_GROUPS = ["left_leg", "right_leg", "waist", "left_arm", "left_hand",
                "right_arm", "right_hand"]


def resolve_pretrained(path: Path) -> Path:
    """Accept a pretrained_model dir, a checkpoints/<step> dir, or a train output
    dir (with checkpoints/) — return the pretrained_model dir (newest step)."""
    path = path.expanduser().resolve()
    if (path / "config.json").exists():
        return path
    if (path / "pretrained_model" / "config.json").exists():
        return path / "pretrained_model"
    ckpts = sorted(
        (int(re.sub(r"\D", "", p.name) or 0), p)
        for p in (path / "checkpoints").glob("*")
        if (p / "pretrained_model" / "config.json").exists())
    if not ckpts:
        raise FileNotFoundError(f"no pretrained_model under {path}")
    return ckpts[-1][1] / "pretrained_model"


class LerobotGrootPolicy:
    def __init__(self, ckpt: Path, device: str, exec_horizon: int | None):
        pretrained = resolve_pretrained(ckpt)
        ptype = json.loads((pretrained / "config.json").read_text())["type"]
        if ptype == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy
            self.policy = ACTPolicy.from_pretrained(pretrained)
        elif ptype == "diffusion":
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
            self.policy = DiffusionPolicy.from_pretrained(pretrained)
        else:
            raise ValueError(f"unsupported lerobot policy type {ptype!r} (want act|diffusion)")
        self.ptype = ptype
        self.device = torch.device(device)
        self.policy.to(self.device).eval()
        self.policy.reset()
        self.exec_horizon = exec_horizon
        self.image_keys = list(self.policy.config.image_features)
        # action dim comes from the checkpoint: 78 = old SONIC layout (motion_token+hands,
        # split three ways on the wire), 29 = bench-phase G1FloatingBody wbc (served whole).
        self.action_dim = self.policy.config.output_features["action"].shape[0]
        self._prompt_warned = False
        print(f"[lerobot] loaded {ptype} from {pretrained} "
              f"(image_features={self.image_keys}, action_dim={self.action_dim})", flush=True)

    # ---- endpoints ----------------------------------------------------------
    def ping(self) -> dict:
        return {"status": "ok", "message": f"lerobot {self.ptype} gr00t-compat server"}

    def get_modality_config(self) -> dict:
        return {}

    def reset(self, options: dict | None = None) -> dict:
        self.policy.reset()      # clears DP obs/action queues; ACT action queue
        return {"status": "reset"}

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        t0 = time.monotonic()
        if not self._prompt_warned and (observation.get("language") or {}):
            print(f"[lerobot] NOTE: {self.ptype} is not language-conditioned; "
                  "the prompt is ignored", flush=True)
            self._prompt_warned = True

        frame = self._extract_image(observation)          # [H,W,3] uint8
        img = torch.from_numpy(frame).to(self.device).permute(2, 0, 1).float() / 255.0
        q = torch.from_numpy(self._extract_proprio(observation)).to(self.device)
        batch = {"observation.state": q.unsqueeze(0)}
        for k in self.image_keys:
            batch[k] = img.unsqueeze(0)

        with torch.no_grad():
            if self.ptype == "act":
                actions = self.policy.predict_action_chunk(batch)
            else:
                from lerobot.policies.utils import populate_queues
                from lerobot.constants import OBS_IMAGES
                nbatch = self.policy.normalize_inputs(batch)
                if self.image_keys:
                    nbatch = dict(nbatch)
                    nbatch[OBS_IMAGES] = torch.stack(
                        [nbatch[k] for k in self.image_keys], dim=-4)
                self.policy._queues = populate_queues(self.policy._queues, nbatch)
                actions = self.policy.predict_action_chunk(nbatch)

        raw = actions[0].float().cpu().numpy()            # (H, action_dim)
        if raw.shape[-1] != self.action_dim:
            raise RuntimeError(f"checkpoint action dim {raw.shape[-1]} != {self.action_dim}")
        if self.exec_horizon is not None:
            raw = raw[: self.exec_horizon]
        if self.action_dim == ACTION_DIM:      # 78-D SONIC layout
            action_dict = {
                "action.motion_token": np.ascontiguousarray(raw[:, :MOTION_TOKEN_DIM], np.float32),
                "action.left_hand_joints": np.ascontiguousarray(
                    raw[:, MOTION_TOKEN_DIM:MOTION_TOKEN_DIM + HAND_DIM], np.float32),
                "action.right_hand_joints": np.ascontiguousarray(
                    raw[:, MOTION_TOKEN_DIM + HAND_DIM:ACTION_DIM], np.float32),
            }
        else:                                   # bench wbc (29-D pre-v6 / 43-D v6) — served whole
            action_dict = {"action.wbc": np.ascontiguousarray(raw, np.float32)}
        info = {"inference_time_s": time.monotonic() - t0, "chunk_horizon": int(raw.shape[0])}
        return action_dict, info

    # ---- obs unpack (same tolerant shapes as the psi0 server) --------------
    @staticmethod
    def _extract_image(obs: dict) -> np.ndarray:
        video = obs.get("video") or {}
        img = None
        for k in ("ego_view", "observation.images.ego_view"):
            if k in video:
                img = video[k]
                break
        if img is None:
            if len(video) == 1:
                img = next(iter(video.values()))
            else:
                raise KeyError(f"No ego_view in obs['video']; keys={list(video.keys())}")
        img = np.asarray(img)
        while img.ndim > 3 and img.shape[0] >= 1:
            img = img[-1]
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"ego_view frame must be [H,W,3], got {img.shape}")
        return img.astype(np.uint8)

    @staticmethod
    def _extract_proprio(obs: dict) -> np.ndarray:
        q = obs.get("q")
        if q is not None:
            q = np.asarray(q, dtype=np.float32).reshape(-1)
        else:
            state = obs.get("state") or {}
            missing = [g for g in STATE_GROUPS if g not in state]
            if missing:
                raise KeyError(f"obs['state'] missing groups {missing}; got {list(state.keys())}")
            parts = []
            for g in STATE_GROUPS:
                v = np.asarray(state[g], dtype=np.float32)
                while v.ndim > 1:
                    v = v[-1]
                parts.append(v.reshape(-1))
            q = np.concatenate(parts)
        if q.shape[0] != PROPRIO_DIM:
            raise ValueError(f"proprio is {q.shape[0]}-dim, expected {PROPRIO_DIM}")
        return q


def serve(policy: LerobotGrootPolicy, host: str, port: int) -> None:
    import msgpack_numpy as mnp
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[lerobot] PolicyServer ready on tcp://{host}:{port}", flush=True)

    running = True
    while running:
        try:
            req = mnp.unpackb(sock.recv(), raw=False)
            endpoint = req.get("endpoint", "get_action")
            data = req.get("data") or {}
            if endpoint == "ping":
                resp = policy.ping()
            elif endpoint == "kill":
                running = False
                resp = {"status": "killing"}
            elif endpoint == "reset":
                resp = policy.reset(options=data.get("options"))
            elif endpoint == "get_modality_config":
                resp = policy.get_modality_config()
            elif endpoint == "get_action":
                action, info = policy.get_action(observation=data.get("observation", {}),
                                                 options=data.get("options"))
                resp = (action, info)
            else:
                resp = {"error": f"unknown endpoint {endpoint!r}"}
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            resp = {"error": f"{type(e).__name__}: {e}"}
        sock.send(mnp.packb(resp, use_bin_type=True))
    print("[lerobot] server stopped", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path,
                    help="train output dir, checkpoints/<step>, or pretrained_model dir")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5550)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--exec-horizon", type=int, default=None,
                    help="cap the served chunk length (default: policy's full chunk)")
    ap.add_argument("--initial-prompt", default=None,
                    help="accepted for CLI parity with other servers; ignored (no language)")
    args = ap.parse_args()
    serve(LerobotGrootPolicy(args.ckpt, args.device, args.exec_horizon), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
