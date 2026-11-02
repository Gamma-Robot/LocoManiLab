"""Strict, read-only validation of locomanilab's LeRobot v2.1 interchange contract.

Imports are intentionally light. Arrow and video decoding are loaded only when
validate() is called, so listing models and printing CLI help need no data stack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from . import metadata
from .paths import dataset_file


@dataclass
class DatasetReport:
    path: Path
    ok: bool = True
    fps: float | None = None
    total_episodes: int | None = None
    total_frames: int | None = None
    tasks: list[str] = field(default_factory=list)
    video_keys: list[str] = field(default_factory=list)
    columns: dict[str, tuple | None] = field(default_factory=dict)
    state_dim: int | None = None
    action_keys: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.ok = False
        # A broken vector column can fail on every frame. Keep the report usable.
        if len(self.errors) < 100:
            self.errors.append(message)

    def summary(self) -> str:
        lines = [f"dataset: {self.path}", f"  status: {'OK' if self.ok else 'INVALID'}"]
        if self.fps is not None:
            lines.append(f"  fps={self.fps}  episodes={self.total_episodes}  frames={self.total_frames}")
        if self.tasks:
            lines.append(f"  tasks: {self.tasks}")
        if self.video_keys:
            lines.append(f"  video: {self.video_keys}")
        if self.state_dim is not None:
            lines.append(f"  observation.state dim={self.state_dim}")
        if self.action_keys:
            lines.append("  action keys: " + ", ".join(f"{k}={v}" for k, v in self.action_keys.items()))
        lines.extend(f"  ! {w}" for w in self.warnings)
        lines.extend(f"  x {e}" for e in self.errors)
        return "\n".join(lines)


def _integer(value, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _shape(value) -> tuple[int, ...] | None:
    if not isinstance(value, list) or not value or not all(_integer(n, 1) for n in value):
        return None
    return tuple(value)


def _read(report: DatasetReport, name: str, expected_type):
    path = report.path / "meta" / name
    if not path.is_file():
        report.error(f"missing meta/{name}")
        return {} if expected_type is dict else []
    try:
        value = metadata.read_jsonl(path) if name.endswith(".jsonl") else metadata.read_json(path)
        if not isinstance(value, expected_type):
            raise ValueError(f"expected {expected_type.__name__}")
        if isinstance(value, list) and not all(isinstance(row, dict) for row in value):
            raise ValueError("each JSONL record must be an object")
        return value
    except (OSError, ValueError, TypeError) as exc:
        report.error(f"cannot read meta/{name}: {exc}")
        return {} if expected_type is dict else []


def _feature(section: str, name: str, spec: dict) -> str:
    return spec.get("original_key") or {
        "state": "observation.state", "action": "action",
        "video": f"observation.images.{name}",
    }.get(section, section)


def _dataset_file(report: DatasetReport, pattern, *, episode: int, chunk: int,
                  video_key: str = "") -> Path | None:
    try:
        return dataset_file(report.path, pattern, episode_index=episode,
                            episode_chunk=chunk, video_key=video_key)
    except (ValueError, TypeError) as exc:
        report.error(str(exc))
        return None


def _numeric_shape(value, shape: tuple[int, ...]) -> bool:
    """Scalar LeRobot bookkeeping fields may declare shape [1]."""
    if not isinstance(value, (list, tuple)):
        return shape == (1,) and isinstance(value, (int, float, bool)) and math.isfinite(value)
    if len(value) != shape[0]:
        return False
    if len(shape) == 1:
        return all(isinstance(v, (int, float, bool)) and math.isfinite(v) for v in value)
    return all(_numeric_shape(v, shape[1:]) for v in value)


def _video(report: DatasetReport, path: Path, length: int, shape, fps: float | None) -> None:
    try:
        import av
    except ImportError:
        report.error("video validation needs PyAV; install locomanilab[data] in this environment")
        return
    try:
        with av.open(str(path)) as container:
            streams = container.streams.video
            if len(streams) != 1:
                raise ValueError(f"expected one video stream, found {len(streams)}")
            stream = streams[0]
            if shape and len(shape) == 3 and shape != (stream.height, stream.width, 3):
                report.error(f"{path.relative_to(report.path)} dimensions {(stream.height, stream.width, 3)} "
                             f"do not match features {shape}")
            if fps and stream.average_rate is not None and not math.isclose(float(stream.average_rate), fps,
                                                                          rel_tol=1e-3):
                report.error(f"{path.relative_to(report.path)} fps={float(stream.average_rate)} != info fps={fps}")
            count = sum(1 for _ in container.decode(stream))
            if count != length:
                report.error(f"{path.relative_to(report.path)} video frames={count} != episode rows={length}")
    except Exception as exc:
        report.error(f"cannot decode {path.relative_to(report.path)}: {exc}")


def validate(path: Path) -> DatasetReport:
    """Inspect every episode, action/state value and declared video, without writes.

    Missing data dependencies are failures, never a partially checked green result.
    The supported layout is v2.1 (one parquet/video per episode); v3 needs conversion.
    """
    # Template expansion returns absolute paths; discovery must use the same
    # root so valid relative CLI inputs compare equal without resolving aliases.
    report = DatasetReport(path=Path(path).expanduser().absolute())
    for directory in ("meta", "data"):
        if not (report.path / directory).is_dir():
            report.error(f"missing {directory}/ directory")
    if not (report.path / "meta").is_dir():
        return report
    info = _read(report, "info.json", dict)
    modality = _read(report, "modality.json", dict)
    tasks = _read(report, "tasks.jsonl", list)
    episodes = _read(report, "episodes.jsonl", list)
    version = info.get("codebase_version")
    if version != "v2.1":
        report.error(f"unsupported codebase_version {version!r}; expected LeRobot v2.1")
    fps = info.get("fps")
    if isinstance(fps, (int, float)) and not isinstance(fps, bool) and math.isfinite(fps) and fps > 0:
        report.fps = fps
    else:
        report.error("info.fps must be positive and finite")
    for key in ("total_episodes", "total_frames", "total_tasks", "total_videos", "total_chunks"):
        value = info.get(key)
        if not _integer(value):
            report.error(f"info.{key} must be a nonnegative integer")
        elif key in ("total_episodes", "total_frames"):
            setattr(report, key, value)
    chunks_size = info.get("chunks_size", 1000)
    if not _integer(chunks_size, 1):
        report.error("info.chunks_size must be a positive integer")
        chunks_size = 1000
    features = info.get("features")
    if not isinstance(features, dict) or not features:
        report.error("info.features must be a nonempty object")
        features = {}
    shapes = {}
    for name, spec in features.items():
        if not isinstance(spec, dict) or _shape(spec.get("shape")) is None:
            report.error(f"features.{name} needs a positive integer shape")
            continue
        shapes[name] = _shape(spec["shape"])
        if spec.get("dtype") == "video":
            report.video_keys.append(name)
            if len(shapes[name]) != 3 or shapes[name][2] != 3:
                report.error(f"video feature {name} must declare shape [height, width, 3]")
    if "observation.state" not in features:
        report.error("features missing observation.state")
    elif shapes.get("observation.state"):
        report.state_dim = shapes["observation.state"][0]

    task_ids = set()
    for task in tasks:
        task_id, text = task.get("task_index"), task.get("task")
        if not _integer(task_id) or not isinstance(text, str) or not text.strip():
            report.error("tasks.jsonl requires nonnegative task_index and nonempty task text")
            continue
        if task_id in task_ids:
            report.error(f"tasks.jsonl duplicate task_index {task_id}")
        task_ids.add(task_id)
        report.tasks.append(text)
    if not task_ids:
        report.error("tasks.jsonl must contain at least one task")
    if info.get("total_tasks") != len(task_ids):
        report.error(f"info.total_tasks={info.get('total_tasks')} != tasks.jsonl count={len(task_ids)}")
    if report.tasks == ["test"]:
        report.warnings.append("tasks == ['test'] is a placeholder; use the training prompt at deployment")

    required = {"observation.state", "episode_index", "frame_index", "index", "timestamp", "task_index"}
    for section in ("state", "action", "video"):
        groups = modality.get(section, {})
        if not isinstance(groups, dict):
            report.error(f"modality.{section} must be an object")
            continue
        if section in ("state", "action") and not groups:
            report.error(f"modality.{section} must contain at least one group")
        for name, spec in groups.items():
            if not isinstance(spec, dict):
                report.error(f"modality.{section}.{name} must be an object")
                continue
            key = _feature(section, name, spec)
            if not isinstance(key, str) or key not in features:
                report.error(f"modality.{section}.{name} references undeclared feature {key!r}")
                continue
            if section == "video":
                if key not in report.video_keys:
                    report.error(f"modality.video.{name} must reference a video feature")
                continue
            if key not in shapes:
                continue
            dtype = features[key].get("dtype", "")
            if not isinstance(dtype, str) or not dtype.startswith(("float", "int", "uint")):
                report.error(f"modality.{section}.{name} must reference a numeric feature")
            required.add(key)
            start, end = spec.get("start"), spec.get("end")
            if not _integer(start) or not _integer(end, 1) or end <= start:
                report.error(f"modality.{section}.{name} needs bounds 0 <= start < end")
            elif shapes.get(key) and end > shapes[key][0]:
                report.error(f"modality.{section}.{name} end={end} exceeds {key} width={shapes[key][0]}")
            elif section == "action":
                report.action_keys[name] = end - start
    mapped_videos = set()
    video_groups = modality.get("video") or {}
    if isinstance(video_groups, dict):
        for name, spec in video_groups.items():
            if isinstance(spec, dict):
                key = _feature("video", name, spec)
                if isinstance(key, str):
                    mapped_videos.add(key)
    for key in report.video_keys:
        if key not in mapped_videos:
            report.error(f"video feature {key} has no modality.video mapping")

    parquets = sorted((report.path / "data").rglob("*.parquet"))
    if not parquets:
        report.error("no parquet files under data/chunk-*/")
    if len(parquets) != len(episodes) or info.get("total_episodes") != len(episodes):
        report.error(f"episode count mismatch: info={info.get('total_episodes')}, "
                     f"episodes.jsonl={len(episodes)}, parquet files={len(parquets)}")
    if not episodes:
        report.error("episodes.jsonl must contain at least one episode")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        report.error("parquet validation needs pyarrow; install locomanilab[data] in this environment")
        return report
    seen = set()
    expected_parquets = set()
    expected_videos = set()
    total_frames = 0
    chunks = set()
    for episode in episodes:
        number, length = episode.get("episode_index"), episode.get("length")
        if not _integer(number) or not _integer(length, 1):
            report.error("episodes.jsonl requires nonnegative episode_index and positive length")
            continue
        if number in seen:
            report.error(f"episodes.jsonl duplicate episode_index {number}")
            continue
        seen.add(number)
        chunk = number // chunks_size
        chunks.add(chunk)
        file = _dataset_file(report, info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"),
                             episode=number, chunk=chunk)
        if file is None:
            continue
        expected_parquets.add(file)
        if not file.is_file():
            report.error(f"missing episode parquet {file.relative_to(report.path)}")
            continue
        try:
            table = pq.read_table(file)
            nrows = table.num_rows
            if nrows != length:
                report.error(f"episode {number} rows={nrows} != episodes.jsonl length={length}")
            columns = set(table.column_names)
            for name in sorted(required - columns):
                report.error(f"episode {number} parquet missing column '{name}'")
            for name in sorted(set(features) - set(report.video_keys) - columns):
                if name not in required:
                    report.error(f"episode {number} missing declared feature '{name}'")
            for name in sorted(columns - set(features)):
                report.error(f"episode {number} column '{name}' is absent from info.features")
            for name in sorted(columns & set(features)):
                shape = shapes.get(name)
                if not shape:
                    continue
                report.columns[name] = shape
                values = table[name].to_pylist()
                if features[name].get("dtype", "").startswith(("float", "int", "uint", "bool")):
                    for row, value in enumerate(values):
                        if not _numeric_shape(value, shape):
                            report.error(f"episode {number} {name} row {row} must contain finite numeric values "
                                         f"of shape {shape}")
                            break
            for name, expected in (("episode_index", [number] * nrows),
                                   ("frame_index", list(range(nrows))),
                                   ("index", list(range(total_frames, total_frames + nrows)))):
                if name in columns and table[name].to_pylist() != expected:
                    report.error(f"episode {number} {name} is inconsistent with episode/frame order")
            if "task_index" in columns:
                used = table["task_index"].to_pylist()
                if any(not _integer(value) or value not in task_ids for value in used):
                    report.error(f"episode {number} has task_index values absent from tasks.jsonl")
                actual_tasks = {task["task"] for task in tasks if task.get("task_index") in used
                                and isinstance(task.get("task"), str)}
                declared_tasks = episode.get("tasks")
                if not isinstance(declared_tasks, list) or set(declared_tasks) != actual_tasks:
                    report.error(f"episode {number} tasks disagree with parquet task_index/tasks.jsonl")
            if "timestamp" in columns:
                times = table["timestamp"].to_pylist()
                if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in times):
                    report.error(f"episode {number} timestamp must contain finite nonnegative scalars")
                elif any(b <= a for a, b in zip(times, times[1:])):
                    report.error(f"episode {number} timestamp must increase with frame_index")
            total_frames += nrows
        except Exception as exc:
            report.error(f"could not read parquet episode {number}: {exc}")
        for key in report.video_keys:
            video = _dataset_file(report, info.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"),
                                  episode=number, chunk=chunk, video_key=key)
            if video is None:
                continue
            expected_videos.add(video)
            if not video.is_file():
                report.error(f"missing video {video.relative_to(report.path)}")
            else:
                _video(report, video, length, shapes.get(key), report.fps)
    if seen != set(range(len(episodes))):
        report.error("episode_index must be unique and contiguous from zero")
    for extra in set(parquets) - expected_parquets:
        report.error(f"unexpected parquet {extra.relative_to(report.path)}")
    if total_frames != info.get("total_frames"):
        report.error(f"info.total_frames={info.get('total_frames')} != parquet rows={total_frames}")
    if len(chunks) != info.get("total_chunks"):
        report.error(f"info.total_chunks={info.get('total_chunks')} != episode chunks={len(chunks)}")
    videos = set((report.path / "videos").rglob("*.mp4"))
    if len(expected_videos) != info.get("total_videos") or videos != expected_videos:
        report.error(f"video count/file mismatch: info={info.get('total_videos')}, "
                     f"expected={len(expected_videos)}, files={len(videos)}")
    return report
