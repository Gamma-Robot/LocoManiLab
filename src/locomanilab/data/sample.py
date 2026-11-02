"""Create deterministic synthetic data for exercising the CPU data pipeline.

This is a format example, not a robot demonstration or a training benchmark.
No assets, model weights, simulator, network or GPU are used.
"""
from __future__ import annotations

import json
import shutil
from fractions import Fraction
from pathlib import Path


DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
VIDEO_KEY = "observation.images.ego_view"


def create_sample(out: Path, episodes: int = 2, frames: int = 20) -> Path:
    """Write a new, fully declared v2.1 dataset; refuse any existing destination."""
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in (episodes, frames)):
        raise ValueError("episodes and frames must be positive integers")
    out = Path(out).expanduser().absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"sample destination already exists: {out}")
    try:
        import av
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("sample-data needs locomanilab[data] (numpy, pyarrow and av)") from exc

    fps = 10
    features = {
        "observation.state": {"dtype": "float32", "shape": [4], "names": None},
        "action": {"dtype": "float32", "shape": [2], "names": None},
        VIDEO_KEY: {"dtype": "video", "shape": [64, 64, 3], "names": ["height", "width", "channels"],
                    "info": {"video.fps": fps, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                             "video.is_depth_map": False, "has_audio": False}},
        **{key: {"dtype": "int64", "shape": [1], "names": None}
           for key in ("episode_index", "frame_index", "index", "task_index")},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
        "next.reward": {"dtype": "float32", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1", "robot_type": "synthetic", "fps": fps,
        "total_episodes": episodes, "total_frames": episodes * frames, "total_tasks": 1,
        "total_videos": episodes, "total_chunks": (episodes + 999) // 1000, "chunks_size": 1000,
        "data_path": DATA_PATH, "video_path": VIDEO_PATH, "splits": {"train": f"0:{episodes}"},
        "features": features,
        "collection_info": {"backend": "synthetic", "schema_version": 1, "synthetic": True,
                            "purpose": "CPU data-pipeline example; not a robot demonstration"},
    }
    modality = {
        "state": {"proprio": {"start": 0, "end": 4, "original_key": "observation.state"}},
        "action": {"command": {"start": 0, "end": 2, "original_key": "action"}},
        "video": {"ego_view": {"original_key": VIDEO_KEY}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    prompt = "synthetic color-block data pipeline example"
    records, episode_stats, accumulated = [], [], {}

    def statistics(matrix):
        array = np.asarray(matrix, dtype=np.float64)
        if array.ndim == 1:
            array = array[:, None]
        return {"mean": array.mean(axis=0).tolist(), "std": array.std(axis=0).tolist(),
                "min": array.min(axis=0).tolist(), "max": array.max(axis=0).tolist(),
                "q01": np.quantile(array, 0.01, axis=0).tolist(),
                "q99": np.quantile(array, 0.99, axis=0).tolist(), "count": [len(array)]}

    out.mkdir(parents=True, exist_ok=False)
    try:
        meta = out / "meta"
        meta.mkdir()
        for ep in range(episodes):
            t = np.arange(frames, dtype=np.int64)
            phase = t.astype(np.float32) / max(1, frames - 1)
            state = np.column_stack((phase, 1 - phase, np.full(frames, ep), np.zeros(frames))).astype("float32")
            action = np.column_stack((phase * 0.1, -phase * 0.1)).astype("float32")
            values = {
                "observation.state": state, "action": action,
                "frame_index": t, "episode_index": np.full(frames, ep, dtype=np.int64),
                "index": ep * frames + t, "task_index": np.zeros(frames, dtype=np.int64),
                "timestamp": t.astype("float32") / fps,
                "next.done": t == frames - 1, "next.reward": np.zeros(frames, dtype="float32"),
            }
            arrays = {}
            for key, value in values.items():
                scalar = {"float32": pa.float32(), "int64": pa.int64(), "bool": pa.bool_()}[features[key]["dtype"]]
                arrays[key] = pa.array(value.tolist(), type=pa.list_(scalar) if value.ndim == 2 else scalar)
            parquet = out / DATA_PATH.format(episode_chunk=ep // 1000, episode_index=ep)
            parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(arrays), parquet)
            video = out / VIDEO_PATH.format(episode_chunk=ep // 1000, episode_index=ep, video_key=VIDEO_KEY)
            video.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(video), mode="w") as container:
                stream = container.add_stream("libx264", rate=fps)
                stream.width = stream.height = 64
                stream.pix_fmt = "yuv420p"
                # Encode losslessly on one thread so the decoded format example
                # preserves its generated YUV pattern exactly.
                stream.codec_context.thread_count = 1
                stream.options = {"crf": "0", "preset": "veryfast"}
                for frame in range(frames):
                    image = av.VideoFrame(64, 64, format="yuv420p")
                    # Initialize the entire planes, including row padding, and avoid
                    # an implicit RGB-to-YUV conversion of the synthetic pattern.
                    for index, plane in enumerate(image.planes):
                        level = 32 + (frame * 9 + ep * 31) % 96 if index == 0 else 128
                        pixels = np.full(plane.buffer_size, level, dtype=np.uint8)
                        rows = pixels[:plane.height * plane.line_size].reshape(plane.height, plane.line_size)
                        if index == 0:
                            rows[:, :plane.width] = level + np.arange(plane.width, dtype=np.uint8)
                        else:
                            rows[8:24, 8:24] = 160 if index == 1 else 96
                        plane.update(pixels)
                    image.pts = frame
                    image.time_base = Fraction(1, fps)
                    for packet in stream.encode(image):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            stats = {key: statistics(value) for key, value in values.items()}
            episode_stats.append({"episode_index": ep, "stats": stats})
            for key, value in values.items():
                accumulated.setdefault(key, []).append(value)
            records.append({"episode_index": ep, "tasks": [prompt], "length": frames})
        for name, value in (("info.json", info), ("modality.json", modality),
                            ("stats.json", {key: statistics(np.concatenate(parts))
                                            for key, parts in accumulated.items()})):
            (meta / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        for name, rows in (("tasks.jsonl", [{"task_index": 0, "task": prompt}]),
                           ("episodes.jsonl", records), ("episodes_stats.jsonl", episode_stats)):
            (meta / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    except BaseException:
        # Only remove this invocation's new directory; existing data is never touched.
        shutil.rmtree(out)
        raise
    return out
