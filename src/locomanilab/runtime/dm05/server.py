#!/usr/bin/env python3
"""gr00t-compatible policy server (ZMQ REQ/REP + msgpack-numpy) for a locomanilab DM0.5 checkpoint.

OpenDM serves DM05 over HTTP (Flask ``/v1/infer``) with base64/multipart images. locomanilab drives the
real G1 and the humanoid_benchmark sim through ONE contract instead -- a gr00t-compatible
server on ``tcp://<host>:<port>`` answering ``get_action`` with raw observations -- so this file
wraps OpenDM's own inference runtime (``DM05Exp._initialize_inference_runtime`` +
``DM05InferenceConfig._prepare_model_input`` / ``_predict``) without re-implementing any of it:
same chat template, state tokens, quantile (de)normalisation and flow sampling as ``/v1/infer``.

Per get_action():
  1. obs['video']['ego_view'] newest frame -> PIL (the single "Head" image)
  2. obs['state'][left_leg|right_leg|waist|left_arm|left_hand|right_arm|right_hand] -> 43-D
  3. prompt = obs['language']['annotation.human.task_description'] (else reset()/--initial-prompt)
  4. OpenDM prediction -> (chunk 50, 78) ABSOLUTE action -> split:
        action.motion_token [H, 64], action.left_hand_joints [H, 7], action.right_hand_joints [H, 7]

Endpoints: ping, kill, reset, get_action, get_modality_config (= gr00t PolicyServer).

    python src/locomanilab/runtime/dm05/server.py --model-path <run>/checkpoint-N --port 5550
"""
from __future__ import annotations

import argparse
import os
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np

if __package__:
    from . import experiment as experiment_config
else:
    import experiment as experiment_config  # noqa: E402  (sets OPENDM_DATA_PATH before opendm imports)
from PIL import Image  # noqa: E402

STATE_GROUPS = [("left_leg", 6), ("right_leg", 6), ("waist", 3), ("left_arm", 7),
                ("left_hand", 7), ("right_arm", 7), ("right_hand", 7)]   # 43-D observation.state order
ACTION_SPLIT = [("action.motion_token", 64), ("action.left_hand_joints", 7),
                ("action.right_hand_joints", 7)]


def state_from_groups(groups: dict) -> np.ndarray:
    parts = []
    for name, width in STATE_GROUPS:
        if name not in groups:
            raise KeyError(f"obs['state'] is missing group {name!r} (have {sorted(groups)})")
        v = np.asarray(groups[name], dtype=np.float32).reshape(-1)
        if v.size < width:
            raise ValueError(f"state group {name!r}: {v.size} values, expected {width}")
        parts.append(v[-width:])          # groups arrive as [B=1, T=1, width]; take the newest
    return np.concatenate(parts)


class DM05GrootPolicy:
    def __init__(self, model_path: Path, dataset_name: str, initial_prompt: Optional[str],
                 diffusion_steps: int, exec_horizon: Optional[int], suffix_cuda_graph: bool = False):
        model_path = model_path.resolve()
        # Dataclass __post_init__ reads the checkpoint immediately. Supply all
        # path-dependent configuration at construction and initialise only once.
        data = experiment_config.LocoManiLabDM05DataConfig(dataset_name=dataset_name)
        if not (model_path / "norm_stats.json").exists():
            data.norm_stats_root = str(model_path / "norm_stats")
        exp = experiment_config.LocoManiLabDM05Exp(
            task="inference",
            model_config=experiment_config.LocoManiLabDM05ModelConfig(model_name_or_path=str(model_path)),
            data_config=data,
            inference_config=experiment_config.LocoManiLabDM05InferenceConfig(diffusion_steps=diffusion_steps),
        )
        exp._initialize_inference_runtime()
        self.exp = exp
        self.icfg = exp.inference_config
        if not suffix_cuda_graph:
            # OpenDM records a CUDA Graph for the action-expert suffix decode per exact prefix length
            # (the state tokens change it from call to call), keeping an LRU of 8. After evictions,
            # replay hits "illegal memory access" (2026-10-13 sim eval: 7/8 servers within 4-36 calls,
            # also with one server alone on a GPU), and every graph keeps its own memory pool
            # (23 -> 32 GiB per server). Eager decode runs the same modules without graph replay.
            from opendm.model.dm05.dm05_lora import unwrap_dm05_model
            unwrap_dm05_model(self.icfg.model)._can_use_suffix_graph = lambda *_a, **_k: False
        self.suffix_cuda_graph = suffix_cuda_graph
        self.prompt = initial_prompt
        self.exec_horizon = exec_horizon
        self._prev_token = None
        print(f"[dm05] loaded {model_path} (dataset profile {dataset_name}, "
              f"robot {self.icfg.default_robot_type!r}, diffusion_steps {diffusion_steps}, "
              f"suffix_cuda_graph {suffix_cuda_graph})", flush=True)

    def ping(self) -> dict:
        return {"status": "ok", "message": "dm05 gr00t-compat server is running"}

    def get_modality_config(self) -> dict:
        return {}

    def reset(self, options: dict | None = None) -> dict:
        prompt = (options or {}).get("prompt")
        if prompt is not None:
            self.prompt = prompt
        self._prev_token = None
        return {"status": "reset", "prompt": self.prompt}

    def get_action(self, observation: dict[str, Any], options: dict | None = None):
        del options
        t0 = time.monotonic()
        prompt = self._extract_prompt(observation) or self.prompt
        if not prompt:
            raise ValueError("No prompt: pass --initial-prompt, reset(prompt=...), or "
                             "obs['language']['annotation.human.task_description'].")
        q = observation.get("q")
        state = (np.asarray(q, dtype=np.float32).reshape(-1) if q is not None
                 else state_from_groups(observation.get("state") or {}))
        if state.shape[0] != 43:
            raise ValueError(f"state must be 43-D, got {state.shape[0]}")
        image = Image.fromarray(self._extract_image(observation)).convert("RGB")
        data = self.icfg._prepare_model_input(text=prompt, images=[image],
                                              states=state.tolist(), robot_type=None)
        chunk = np.asarray(self.icfg._predict(data), dtype=np.float32)   # (H, 78) absolute
        if self.exec_horizon is not None:
            chunk = chunk[: self.exec_horizon]
        out, s = {}, 0
        for key, width in ACTION_SPLIT:
            out[key] = chunk[:, s:s + width]
            s += width
        tokens = out["action.motion_token"]
        tok_std = float(np.std(tokens, axis=0).mean())
        d_prev = ("n/a" if self._prev_token is None
                  else f"{float(np.linalg.norm(tokens[0] - self._prev_token)):.4f}")
        self._prev_token = tokens[-1].copy()
        lat = self.icfg.last_model_latency_sec or 0.0
        print(f"[dm05] get_action: H={chunk.shape[0]} model={lat * 1000:.0f} ms "
              f"tok_std={tok_std:.4f} d_prev={d_prev} prompt={prompt!r}", flush=True)
        return out, {"inference_time_s": time.monotonic() - t0, "chunk_horizon": int(chunk.shape[0])}

    @staticmethod
    def _extract_prompt(obs: dict) -> Optional[str]:
        raw = (obs.get("language") or {}).get("annotation.human.task_description")
        while isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return raw if isinstance(raw, str) and raw else None

    @staticmethod
    def _extract_image(obs: dict) -> np.ndarray:
        video = obs.get("video") or {}
        image = None
        for key in ("ego_view", "observation.images.ego_view"):
            if key in video:
                image = video[key]
                break
        if image is None:
            if len(video) != 1:
                raise KeyError(f"no ego_view in obs['video']; keys={list(video)}")
            image = next(iter(video.values()))
        image = np.asarray(image)
        while image.ndim > 3:              # peel [B, T, ...] to the newest frame
            image = image[-1]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"ego_view frame must be [H, W, 3], got {image.shape}")
        return image.astype(np.uint8)


def serve(policy: DM05GrootPolicy, host: str, port: int) -> None:
    import msgpack_numpy as mnp
    import zmq

    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[dm05] PolicyServer ready on tcp://{host}:{port}", flush=True)
    running = True
    while running:
        try:
            req = mnp.unpackb(sock.recv(), raw=False)
            endpoint = req.get("endpoint", "get_action")
            data = req.get("data") or {}
            if endpoint == "ping":
                resp = policy.ping()
            elif endpoint == "kill":
                running, resp = False, {"status": "killing"}
            elif endpoint == "reset":
                resp = policy.reset(options=data.get("options"))
            elif endpoint == "get_modality_config":
                resp = policy.get_modality_config()
            elif endpoint == "get_action":
                resp = policy.get_action(observation=data.get("observation", {}),
                                         options=data.get("options"))
            else:
                resp = {"error": f"unknown endpoint {endpoint!r}"}
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            resp = {"error": f"{type(error).__name__}: {error}"}
        try:
            sock.send(mnp.packb(resp, use_bin_type=True))
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            break
    print("[dm05] server shut down", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-path", required=True, help="checkpoint-N dir (or a run's final save)")
    ap.add_argument("--dataset-name", default="locomanilab_train50_merged",
                    help="registered dataset whose robot_type/state_desc the prompt uses")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5550)
    ap.add_argument("--initial-prompt", default=None)
    ap.add_argument("--diffusion-steps", type=int, default=10, help="flow steps (OpenDM default 10)")
    ap.add_argument("--action-horizon", type=int, default=None, help="truncate the served chunk")
    ap.add_argument("--suffix-cuda-graph", action="store_true",
                    help="keep OpenDM's suffix CUDA Graph (off by default: replay crashes after LRU eviction)")
    a = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    policy = DM05GrootPolicy(Path(a.model_path), a.dataset_name, a.initial_prompt,
                             a.diffusion_steps, a.action_horizon, a.suffix_cuda_graph)
    serve(policy, a.host, a.port)


if __name__ == "__main__":
    main()
