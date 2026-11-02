#!/usr/bin/env python3
"""Convert a locomanilab-prepared v2.1 LeRobot dataset into the flat-action variant the
lerobot baseline policies (ACT / Diffusion Policy) consume.

Why: lerobot's ACTPolicy/DiffusionPolicy expect a single `action` output feature,
but locomanilab datasets carry the 78-D G1 SONIC action split across three features
(`action.motion_token`[64] + `teleop.left_hand_joints`[7] + `teleop.right_hand_joints`[7]
— the same order every locomanilab policy-server adapter splits on). This script:

  data/…/episode_*.parquet   rewritten with columns:
                             observation.state[43] · action[78] · bookkeeping
  meta/episodes_stats.jsonl  action stats concatenated per-dim from the three source
                             features (exact, no decoding); image stats computed by
                             sampling frames (source has none for video features)
  meta/{episodes,tasks}.jsonl copied · meta/info.json rewritten (feature subset)
  videos/                    symlinked to the source (video key unchanged -> no re-encode)

Usage (run with the lerobot env's python — needs pyarrow + av):
  external/lerobot/.venv/bin/python src/locomanilab/runtime/lerobot/convert.py \
      --src /path/to/dataset \
      --out /path/to/converted-dataset

Existing --out is refused unless --force (never converts in place).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Load only the stdlib safety helper; the model interpreter need not install locomanilab.
import importlib.util
_path_spec = importlib.util.spec_from_file_location(
    "_locomanilab_data_paths", Path(__file__).resolve().parents[2] / "data" / "paths.py")
_paths = importlib.util.module_from_spec(_path_spec)
_path_spec.loader.exec_module(_paths)


# Default = the G1 SONIC 78-D layout (old IThor validation datasets). The bench-phase
# datasets (robot G1FloatingBody) carry a single
# `action.wbc` instead (29-D before schema v6, 43-D = body + commanded fingers in
# the state's slots from v6 on) — pass --action-parts action.wbc for those.
ACTION_PARTS = ["action.motion_token", "teleop.left_hand_joints", "teleop.right_hand_joints"]
STATE = "observation.state"
VIDEO = "observation.images.ego_view"
BOOKKEEPING = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
STAT_FIELDS = ["mean", "std", "min", "max", "q01", "q99"]


def detect_action_parts(feats: dict) -> list[str]:
    """Pick the action layout from what the dataset actually carries.

    Two collection stacks are in play and they record different things:
      sonic -> action.motion_token(64) + teleop.{left,right}_hand_joints(7) = 78-D
      heft  -> action.ref_qpos(36: root_pos3+quat4+joint29) + action.hands(14) = 50-D
    Guessing wrong is silent (you just train on the wrong columns), so detect rather
    than default. --action-parts still overrides for anything unusual.
    """
    for parts in (["action.motion_token", "teleop.left_hand_joints", "teleop.right_hand_joints"],
                  ["action.ref_qpos", "action.hands"]):
        if all(p in feats for p in parts):
            return parts
    raise SystemExit(f"[convert] cannot tell the action layout from features "
                     f"{sorted(k for k in feats if k.startswith(('action', 'teleop')))} "
                     f"— pass --action-parts explicitly")


def concat_stats(src_stats: dict) -> dict:
    """Per-dim concat of the three action parts' stats — exact for independent dims."""
    out = {}
    for f in STAT_FIELDS:
        # only fields every part carries (bench datasets ship without q01/q99 —
        # lerobot's normalizers need just mean/std/min/max)
        if any(f not in src_stats[k] for k in ACTION_PARTS):
            continue
        parts = []
        for k in ACTION_PARTS:
            parts.extend(src_stats[k][f])
        out[f] = parts
    for need in ("mean", "std", "min", "max"):
        if need not in out:
            raise KeyError(f"action parts missing required stat field {need}")
    out["count"] = src_stats[ACTION_PARTS[0]]["count"]
    return out


def image_stats(video_path: Path, n_frames: int) -> dict:
    import av
    import numpy as np

    with av.open(str(video_path)) as c:
        stream = c.streams.video[0]
        total = stream.frames or 0
        want = sorted({int(i * max(total - 1, 0) / max(n_frames - 1, 1)) for i in range(n_frames)})
        frames, idx = [], 0
        for frame in c.decode(stream):
            if idx in want:
                frames.append(frame.to_ndarray(format="rgb24"))
            idx += 1
            if len(frames) >= len(want):
                break
    px = (np.stack(frames).astype("float64") / 255.0)          # [n, H, W, 3]
    def chan(fn):
        return [[[float(v)]] for v in fn(px, axis=(0, 1, 2))]   # shape (3,1,1) convention
    return {"mean": chan(np.mean), "std": chan(np.std), "min": chan(np.min),
            "max": chan(np.max), "count": [len(frames)]}


def main() -> int:
    global ACTION_PARTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--image-stats-frames", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--action-parts", default=None,
                    help="comma-separated source features to concat into the action column; "
                         "default: auto-detect (sonic 78-D vs heft 50-D) from the dataset")
    args = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_alias = args.src.expanduser().absolute()
    source_identity = _paths.dataset_identity(source_alias)
    src = source_alias.resolve()
    try:
        destination = _paths.output_directory([src], args.out, force=args.force)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    info = json.loads((src / "meta/info.json").read_text())
    feats = info["features"]
    ACTION_PARTS = ([s for s in args.action_parts.split(",") if s] if args.action_parts
                    else detect_action_parts(feats))
    dim = sum(feats[k]["shape"][0] for k in ACTION_PARTS)
    state_feat = dict(feats[STATE], dtype="float32")   # source declares float64; we rewrite
    new_feats = {VIDEO: feats[VIDEO], STATE: state_feat,
                 "action": {"dtype": "float32", "shape": [dim],
                            "names": None},
                 **{k: feats[k] for k in BOOKKEEPING if k in feats}}
    # Preflight inputs before replacing any previous converted dataset.
    parquets = sorted(src.glob("data/chunk-*/episode_*.parquet"))
    if not parquets:
        ap.error("source contains no episode parquet files")
    episodes = [json.loads(line) for line in (src / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    if len(episodes) != len(parquets) or len(episodes) != info.get("total_episodes"):
        ap.error("source episode metadata and parquet file counts disagree")
    for episode in episodes:
        index = episode["episode_index"]
        chunk = index // info.get("chunks_size", 1000)
        for pattern, key in ((info["data_path"], ""), (info["video_path"], VIDEO)):
            file = _paths.dataset_file(src, pattern, episode_index=index, episode_chunk=chunk, video_key=key)
            if not file.is_file():
                ap.error(f"missing source resource: {file}")
    if not (src / "meta/episodes_stats.jsonl").is_file():
        ap.error("source is missing meta/episodes_stats.jsonl")
    with _paths.staged_output([src], destination, force=args.force) as out:
        (out / "meta").mkdir()
        info["features"] = new_feats
        (out / "meta/info.json").write_text(json.dumps(info, indent=4))
        for f in ("episodes.jsonl", "tasks.jsonl"):
            shutil.copy(src / "meta" / f, out / "meta" / f)
    
        # ---- parquet rewrite --------------------------------------------------
        n_eps = 0
        for pq_path in parquets:
            rel = pq_path.relative_to(src)
            t = pq.read_table(pq_path)
            import numpy as np
    
            def f32_col(arr2d: "np.ndarray") -> pa.Array:
                # pyarrow infers list<double> from float rows — pin list<float32> so the
                # on-disk type matches the declared dtype (lerobot feeds it straight to torch)
                return pa.array([np.asarray(r, dtype=np.float32) for r in arr2d],
                                type=pa.list_(pa.float32()))
    
            state = np.stack(t[STATE].to_pylist()).astype("float32")
            cols, names = [f32_col(state)], [STATE]
            parts = [np.stack(t[k].to_pylist()).astype("float32") for k in ACTION_PARTS]
            action = np.concatenate(parts, axis=1)
            assert action.shape[1] == dim, (action.shape, dim)
            cols.append(f32_col(action)); names.append("action")
            for k in BOOKKEEPING:
                if k in t.column_names:
                    cols.append(t[k]); names.append(k)
            (out / rel.parent).mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(dict(zip(names, cols))), out / rel)
            n_eps += 1
    
        # ---- stats ------------------------------------------------------------
        vids = {int(p.stem.split("_")[-1]): p
                for p in (src / "videos").glob(f"chunk-*/{VIDEO}/episode_*.mp4")}
        with open(src / "meta/episodes_stats.jsonl") as fin, \
             open(out / "meta/episodes_stats.jsonl", "w") as fout:
            for line in fin:
                rec = json.loads(line)
                s = rec["stats"]
                new_s = {STATE: s[STATE], "action": concat_stats(s),
                         **{k: s[k] for k in BOOKKEEPING if k in s}}
                ep = rec["episode_index"]
                if ep in vids:
                    new_s[VIDEO] = image_stats(vids[ep], args.image_stats_frames)
                fout.write(json.dumps({"episode_index": ep, "stats": new_s}) + "\n")
    
        (out / "videos").symlink_to(src / "videos")
        if _paths.dataset_identity(source_alias) != source_identity:
            raise ValueError("source metadata changed during conversion; retry with a stable source")
        (out / "meta/locomanilab_source.json").write_text(json.dumps(source_identity, indent=2) + "\n")
        print(f"[convert] {n_eps} episodes -> {destination}  (action dim {dim}; videos symlinked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
