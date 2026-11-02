#!/usr/bin/env python3
"""G1 SONIC LeRobot (v2.1) dataset -> OpenDM JSONL episodes (DM0.5, github.com/dexmal/opendm).

OpenDM reads one ``.jsonl`` per episode, one JSON object per frame. Images are NOT re-encoded:
``images_1`` is ``{"type": "video", "url": <mp4 relative to the source root>, "frame_idx": i}``
and OpenDM's ``_load_video`` seeks the SOURCE mp4 (frames are 1:1 with the parquet rows), so the
registered ``image_dir`` is the LeRobot dataset root itself.

Per frame:
  state  : observation.state (43-D: legs 12, waist 3, arms 14, hands 14)
  action : action.motion_token (64) + teleop.left_hand_joints (7) + teleop.right_hand_joints (7)
           = 78-D, ABSOLUTE (the same layout gr00t/fastwam/xr1 deploy). An explicit ``action`` is
           required: OpenDM's relative mode needs action_dim == state_dim (78 != 43).
  prompt : tasks.jsonl[task_index] -- never meta/episodes.jsonl, whose "tasks" is the "test"
           placeholder in single-task SONIC dirs.

Also writes ``<out>/opendm_dataset.json``: the registration record that
``locomanilab.runtime.dm05.registry.locomanilab_dm05_datasets`` (OPENDM_DATA_PATH) turns into a
``register_dataset`` call, so nothing machine-specific is hard-coded in the registry.

    python src/locomanilab/runtime/dm05/convert.py --data-root <lerobot ds> --out-dir <root>/<name>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Load only the stdlib safety helper; the model interpreter need not install locomanilab.
import importlib.util
_path_spec = importlib.util.spec_from_file_location(
    "_locomanilab_data_paths", Path(__file__).resolve().parents[2] / "data" / "paths.py")
_paths = importlib.util.module_from_spec(_path_spec)
_path_spec.loader.exec_module(_paths)


import numpy as np
import pyarrow.parquet as pq

VIDEO_KEY = "observation.images.ego_view"
STATE_COL = "observation.state"
ACTION_COLS = [("action.motion_token", 64), ("teleop.left_hand_joints", 7),
               ("teleop.right_hand_joints", 7)]
STATE_DIM = 43
ACTION_DIM = sum(d for _, d in ACTION_COLS)  # 78
ROBOT_TYPE = "Unitree G1"                     # rendered into the prompt as "Robot: Unitree G1"
IMAGE_PROMPTS = ["Head"]                      # G1's single head-mounted ego camera


def _fmt(row: np.ndarray) -> str:
    # %.9g round-trips float32 exactly and keeps the JSONL ~2x smaller than float64 repr.
    return "[" + ",".join("%.9g" % v for v in row.tolist()) + "]"


def _nb_frames(mp4: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(mp4)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out.split(",")[0])


def convert_episode(args: tuple) -> dict:
    root, out_jsonl, info, tasks, ep, verify = args
    idx = ep["episode_index"]
    chunk = idx // info["chunks_size"]
    pq_path = _paths.dataset_file(root, info["data_path"], episode_chunk=chunk, episode_index=idx)
    video = _paths.dataset_file(root, info["video_path"], episode_chunk=chunk,
                                video_key=VIDEO_KEY, episode_index=idx)
    rel_video = video.relative_to(root).as_posix()
    t = pq.read_table(pq_path, columns=[STATE_COL, "task_index", "frame_index"]
                      + [c for c, _ in ACTION_COLS]).to_pandas().sort_values("frame_index")
    n = len(t)
    if n < 2:
        raise SystemExit(f"episode {idx}: {n} frames (OpenDM needs >= 2)")
    if not np.array_equal(t["frame_index"].to_numpy(), np.arange(n)):
        raise SystemExit(f"episode {idx}: frame_index is not 0..{n - 1}")
    state = np.stack(t[STATE_COL].to_numpy()).astype(np.float32)
    action = np.concatenate([np.stack(t[c].to_numpy()).astype(np.float32).reshape(n, d)
                             for c, d in ACTION_COLS], axis=1)
    if state.shape != (n, STATE_DIM) or action.shape != (n, ACTION_DIM):
        raise SystemExit(f"episode {idx}: state {state.shape} action {action.shape}")
    if not (np.isfinite(state).all() and np.isfinite(action).all()):
        raise SystemExit(f"episode {idx}: non-finite values")
    task_idx = t["task_index"].unique()
    if len(task_idx) != 1:
        raise SystemExit(f"episode {idx}: {len(task_idx)} task_index values")
    prompt = tasks[int(task_idx[0])]
    if prompt.strip().lower() == "test":
        raise SystemExit(f"episode {idx}: placeholder prompt 'test'")
    if verify:
        nv = _nb_frames(root / rel_video)
        if nv != n:
            raise SystemExit(f"episode {idx}: video has {nv} frames, parquet {n}")
    ptxt = json.dumps(prompt)
    video_url = json.dumps(rel_video)
    lines = []
    for i in range(n):
        lines.append('{"images_1":{"type":"video","url":%s,"frame_idx":%d},"state":%s,"action":%s,'
                     '"prompt":%s,"is_robot":true}' % (video_url, i, _fmt(state[i]), _fmt(action[i]), ptxt))
    dst = out_jsonl / f"episode_{idx:06d}.jsonl"
    tmp = dst.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, dst)
    return {"episode_index": idx, "frames": n, "prompt": prompt}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True, type=Path, help="G1 SONIC LeRobot v2.1 dataset")
    ap.add_argument("--out-dir", required=True, type=Path, help="output dir (jsonl/ + opendm_dataset.json)")
    ap.add_argument("--name", default=None, help="registered dataset name (default locomanilab_<out-dir name>)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-verify-frames", action="store_true",
                    help="skip the ffprobe check that every mp4 has exactly as many frames as its parquet")
    a = ap.parse_args()

    root = a.data_root.expanduser().resolve()
    try:
        destination = _paths.output_directory([root], a.out_dir)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    if a.workers < 1:
        ap.error("--workers must be positive")
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        print(f"warning: codebase_version {info.get('codebase_version')!r} (written for v2.1)", file=sys.stderr)
    if VIDEO_KEY not in info["features"]:
        raise SystemExit(f"{root}: no {VIDEO_KEY} feature")
    tasks = {}
    for line in (root / "meta/tasks.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            tasks[int(r["task_index"])] = r["task"]
    eps = [json.loads(l) for l in (root / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    if not eps or len(eps) != info.get("total_episodes"):
        ap.error("source must contain episodes matching info.total_episodes")
    indices = [ep["episode_index"] for ep in eps]
    if len(set(indices)) != len(indices):
        ap.error("duplicate source episode_index")
    for ep in eps:
        index = ep["episode_index"]
        chunk = index // info["chunks_size"]
        for pattern, key in ((info["data_path"], ""), (info["video_path"], VIDEO_KEY)):
            file = _paths.dataset_file(root, pattern, episode_index=index, episode_chunk=chunk, video_key=key)
            if not file.is_file():
                ap.error(f"missing source resource: {file}")
    with _paths.staged_output([root], destination) as out:
        jsonl_dir = out / "jsonl"
        jsonl_dir.mkdir()
        jobs = [(root, jsonl_dir, info, tasks, ep, not a.no_verify_frames) for ep in eps]
        with ProcessPoolExecutor(a.workers) as ex:
            done = list(ex.map(convert_episode, jobs, chunksize=8))
        frames = sum(d["frames"] for d in done)
        if frames != info["total_frames"] or len(done) != info["total_episodes"]:
            raise SystemExit(f"converted {len(done)} eps / {frames} frames, info.json says "
                             f"{info['total_episodes']} / {info['total_frames']}")
        name = a.name or f"locomanilab_{destination.name}"
        record = {
            "name": name,
            "jsonl_dir": str(destination / "jsonl"),
            "image_dir": str(root),
            "image_keys": ["images_1"],
            "image_prompts": IMAGE_PROMPTS,
            "robot_type": ROBOT_TYPE,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "action_layout": [f"{c}:{d}" for c, d in ACTION_COLS],
            "fps": info["fps"],
            "source": str(root),
            "episodes": len(done),
            "frames": frames,
            "unique_prompts": len({d["prompt"] for d in done}),
            "frames_verified_against_video": not a.no_verify_frames,
        }
        (out / "opendm_dataset.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
