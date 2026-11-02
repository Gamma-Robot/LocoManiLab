"""locomanilab/data/prepare.py on tiny synthetic LeRobot datasets (no video).

Needs pandas + pyarrow (the `data` / `dev` extras); skipped without them, since the
script itself runs in a data env and the locomanilab CLI env may not have them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from locomanilab.data import prepare as prep  # noqa: E402

STATE_W = 14            # left_hand(7) + right_hand(7) -- enough for the hand stages
HAND = 7


def make_dataset(root: Path, name: str, *, task: str, frames: int = 60,
                 stale: tuple[int, int] | None = None,
                 left_hand_fn=None, extra_features: dict | None = None,
                 extra_modality: dict | None = None) -> Path:
    """One-episode dataset. `stale=(a, b)` zeroes the SMPL column on frames [a, b)
    so prepare removes them; `left_hand_fn(t)` gives the raw left-hand joints."""
    d = root / name
    (d / "meta").mkdir(parents=True)
    (d / "data" / "chunk-000").mkdir(parents=True)
    t = np.arange(frames)
    lh = np.stack([left_hand_fn(i) if left_hand_fn else np.full(HAND, 0.1, np.float32) for i in t])
    rh = np.full((frames, HAND), 0.2, np.float32)
    smpl = np.stack([np.array([1.0 + 0.01 * i, 2.0, 3.0], np.float32) for i in t])
    if stale:
        smpl[stale[0]:stale[1]] = 0.0
    state = np.concatenate([lh, rh], axis=1).astype(np.float32)
    df = pd.DataFrame({
        "observation.state": list(state),
        "action.motion_token": list(np.random.default_rng(0).random((frames, 4), dtype=np.float32)),
        "teleop.left_hand_joints": list(lh.astype(np.float32)),
        "teleop.right_hand_joints": list(rh),
        "teleop.smpl_joints": list(smpl),
        "annotation.human.task_description": np.zeros(frames, np.int64),
        "timestamp": (t / 50).astype(np.float32),
        "frame_index": t, "episode_index": np.zeros(frames, np.int64),
        "index": t, "task_index": np.zeros(frames, np.int64),
        "next.reward": np.r_[np.zeros(max(0, frames - 1), np.float32), np.ones(1, np.float32)],
        "next.done": np.r_[np.zeros(max(0, frames - 1), bool), np.ones(1, bool)],
    })
    df.to_parquet(d / "data" / "chunk-000" / "episode_000000.parquet", index=False)
    features = {
        "observation.state": {"dtype": "float32", "shape": [STATE_W], "names": None},
        "action.motion_token": {"dtype": "float32", "shape": [4], "names": None},
        "teleop.left_hand_joints": {"dtype": "float32", "shape": [HAND], "names": None},
        "teleop.right_hand_joints": {"dtype": "float32", "shape": [HAND], "names": None},
        "teleop.smpl_joints": {"dtype": "float32", "shape": [3], "names": None},
        "annotation.human.task_description": {"dtype": "int64", "shape": [1], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.reward": {"dtype": "float32", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
        **(extra_features or {}),
    }
    info = {"codebase_version": "v2.1", "fps": 50, "total_episodes": 1, "total_frames": frames,
            "total_tasks": 1, "total_videos": 0, "total_chunks": 1, "chunks_size": 1000,
            "splits": {"train": "0:1"}, "data_path": prep.DEFAULT_DATA_PATH,
            "video_path": prep.DEFAULT_VIDEO_PATH, "features": features,
            "collection_info": {"backend": "sonic", "schema_version": 6}}
    modality = {
        "state": {"left_hand": {"start": 0, "end": HAND}, "right_hand": {"start": HAND, "end": 2 * HAND}},
        "action": {"motion_token": {"start": 0, "end": 4, "original_key": "action.motion_token"},
                   "left_hand_joints": {"start": 0, "end": HAND, "original_key": "teleop.left_hand_joints"},
                   "right_hand_joints": {"start": 0, "end": HAND, "original_key": "teleop.right_hand_joints"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
        **(extra_modality or {}),
    }
    (d / "meta" / "info.json").write_text(json.dumps(info))
    (d / "meta" / "modality.json").write_text(json.dumps(modality))
    (d / "meta" / "episodes.jsonl").write_text(json.dumps(
        {"episode_index": 0, "tasks": [task], "length": frames}) + "\n")
    (d / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    return d


def run_prepare(monkeypatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_dataset.py", *args])
    prep.main()


def read_out(out: Path):
    df = pd.read_parquet(out / "data" / "chunk-000" / "episode_000000.parquet")
    meta = {n: json.loads((out / "meta" / n).read_text()) for n in ("info.json", "modality.json")}
    stats = json.loads((out / "meta" / "stats.json").read_text()) if (out / "meta" / "stats.json").exists() else None
    ep_stats = [json.loads(l) for l in (out / "meta" / "episodes_stats.jsonl").read_text().splitlines()]
    return df, meta, stats, ep_stats


# --------------------------------------------------------------------------- #
def test_smoothing_does_not_cross_a_removed_gap(tmp_path, monkeypatch):
    # left hand: 0.0 before the dropout, 1.0 after it. The dropout (frames 20..29, SMPL
    # all-zero) is removed; a window that slid across the seam would blend 0 and 1.
    ds = make_dataset(tmp_path / "in", "gap", task="t", frames=60, stale=(20, 30),
                      left_hand_fn=lambda i: np.full(HAND, 0.0 if i < 20 else 1.0, np.float32))
    out = tmp_path / "out"
    run_prepare(monkeypatch, "--dataset-paths", str(ds), "--output-path", str(out), "--window", "11")
    df, meta, stats, _ = read_out(out)
    assert len(df) == 50                                          # 10 stale frames gone
    sm = np.stack(df["teleop.left_hand_joints_smooth"].values)
    assert np.all(sm[:20] == 0.0), sm[:20, 0]                    # segment 1 untouched
    assert np.all(sm[20:] == 1.0), sm[20:, 0]                    # segment 2 untouched
    openl = np.stack(df["teleop.left_hand_open_smooth"].values)   # the labels too
    assert np.all(openl[:20] == 0.0) and np.all(openl[20:] == 1.0)
    assert meta["info.json"]["total_frames"] == 50


def test_stats_cover_every_numeric_column_with_quantiles(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path / "in", "one", task="t")
    out = tmp_path / "out"
    run_prepare(monkeypatch, "--dataset-paths", str(ds), "--output-path", str(out))
    df, meta, stats, ep_stats = read_out(out)
    numeric = [c for c in df.columns]                              # no string columns here
    for col in numeric:
        assert col in stats, f"stats.json misses {col}"
        assert set(stats[col]) >= {"mean", "std", "min", "max", "q01", "q99"}
        assert len(stats[col]["mean"]) == len(np.asarray(df[col].iloc[0]).reshape(-1))
    # derived columns are there too, and per-episode stats carry the same fields
    assert "teleop.left_hand_open_smooth" in stats
    for col, s in ep_stats[0]["stats"].items():
        assert set(s) >= {"mean", "std", "min", "max", "q01", "q99", "count"}, col


def test_merge_only_still_writes_full_stats(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path / "in", "one", task="t")
    out = tmp_path / "out"
    run_prepare(monkeypatch, "--dataset-paths", str(ds), "--output-path", str(out),
                "--no-discard", "--no-smpl", "--no-open", "--no-smooth")
    df, meta, stats, _ = read_out(out)
    assert stats is not None and "observation.state" in stats and "action.motion_token" in stats
    assert "teleop.left_hand_open" not in df.columns                # nothing derived, as asked


def test_terminal_hold_appends_final_prepared_row_and_updates_meta(tmp_path, monkeypatch):
    ds = make_dataset(tmp_path / "in", "one", task="t")
    source = pd.read_parquet(ds / "data" / "chunk-000" / "episode_000000.parquet")
    out = tmp_path / "out"
    run_prepare(monkeypatch, "--dataset-paths", str(ds), "--output-path", str(out),
                "--no-smpl", "--no-open", "--no-smooth", "--hold-frames", "4")
    df, meta, stats, ep_stats = read_out(out)

    assert len(df) == len(source) + 4
    np.testing.assert_array_equal(
        np.stack(df["action.motion_token"].iloc[-4:]),
        np.repeat(np.asarray(source["action.motion_token"].iloc[-1])[None], 4, axis=0),
    )
    assert np.array_equal(df["frame_index"], np.arange(len(df)))
    assert np.array_equal(df["index"], np.arange(len(df)))
    assert np.allclose(df["timestamp"], np.arange(len(df)) / 50)
    assert df["next.done"].sum() == 1 and bool(df["next.done"].iloc[-1])
    assert df["next.reward"].iloc[len(source) - 1] == 1
    assert np.all(df["next.reward"].iloc[len(source):] == 0)
    assert meta["info.json"]["total_frames"] == len(df)
    assert meta["info.json"]["collection_info"]["terminal_hold"]["frames_appended"] == 4
    assert ep_stats[0]["stats"]["action.motion_token"]["count"] == [len(df)]
    assert len(stats["action.motion_token"]["mean"]) == 4


def test_terminal_hold_reencodes_video_with_matching_frame_count(tmp_path):
    av = pytest.importorskip("av")
    src, dst = tmp_path / "src.mp4", tmp_path / "dst.mp4"
    with av.open(str(src), mode="w") as container:
        stream = container.add_stream("h264", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for value in (0, 80, 160):
            image = np.full((16, 16, 3), value, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    prep.copy_or_filter_video(src, dst, np.array([0, 2]), fps=10, hold_frames=3)
    with av.open(str(dst)) as container:
        frames = list(container.decode(container.streams.video[0]))
    assert len(frames) == 5  # two kept source frames + three terminal copies


def test_merge_remaps_annotation_task_column_with_task_index(tmp_path, monkeypatch):
    a = make_dataset(tmp_path / "in", "a", task="open the door")
    b = make_dataset(tmp_path / "in", "b", task="close the door")
    out = tmp_path / "out"
    run_prepare(monkeypatch, "--dataset-paths", str(a), str(b), "--output-path", str(out))
    df_b = pd.read_parquet(out / "data" / "chunk-000" / "episode_000001.parquet")
    assert set(df_b["task_index"]) == {1}
    assert set(df_b["annotation.human.task_description"]) == {1}, "annotation column not remapped"
    tasks = [json.loads(l) for l in (out / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert [t["task"] for t in tasks] == ["open the door", "close the door"]


def test_prune_modality_covers_video_and_annotation_sections():
    modality = {
        "state": {"left_hand": {"start": 0, "end": 7}},
        "action": {"motion_token": {"start": 0, "end": 4, "original_key": "action.motion_token"}},
        "video": {"ego_view": {"original_key": "observation.images.ego_view"},
                  "wrist": {"original_key": "observation.images.wrist"}},
        "annotation": {"human.task_description": {"original_key": "task_index"},
                       "human.validity": {"original_key": "annotation.human.validity"}},
    }
    kept = {"observation.state", "action.motion_token", "task_index"}
    pruned = prep.prune_modality(modality, kept, video_keys={"observation.images.ego_view"})
    assert list(pruned["video"]) == ["ego_view"]
    assert list(pruned["annotation"]) == ["human.task_description"]
    assert pruned["state"] and pruned["action"]
    with pytest.raises(prep.PrepareError):
        prep.prune_modality(modality, {"observation.state"}, video_keys=set())


def test_segmented_moving_average_is_per_segment():
    arr = np.array([[0.0]] * 5 + [[1.0]] * 5, np.float32)
    whole = prep.moving_average(arr, 5)
    assert 0.0 < whole[4, 0] < 1.0                                 # the plain version blends
    seg = prep.moving_average(arr, 5, segments=[(0, 5), (5, 10)])
    assert np.all(seg[:5] == 0.0) and np.all(seg[5:] == 1.0)
    assert prep.contiguous_segments(None, 10) == [(0, 10)]
    assert prep.contiguous_segments(np.array([0, 1, 2, 7, 8, 20]), 6) == [(0, 3), (3, 5), (5, 6)]


def _snapshot_files(root: Path):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("field", ["data_path", "video_path"])
@pytest.mark.parametrize("escape", ["absolute", "parent"])
def test_prepare_refuses_template_escape_without_modifying_source(tmp_path, monkeypatch, field, escape):
    from locomanilab.data.sample import create_sample
    source = create_sample(tmp_path / "source", episodes=1, frames=3)
    file = source / "meta/info.json"
    info = json.loads(file.read_text())
    info[field] = str(source / info[field]) if escape == "absolute" else "../source/" + info[field]
    file.write_text(json.dumps(info))
    before = _snapshot_files(source)
    out = tmp_path / "prepared"
    with pytest.raises((SystemExit, ValueError), match="inside the dataset"):
        run_prepare(monkeypatch, "--dataset-paths", str(source), "--output-path", str(out),
                    "--hold-frames", "1", "--no-smpl", "--no-open", "--no-smooth")
    assert _snapshot_files(source) == before
    assert not out.exists()


@pytest.mark.parametrize("where", ["same", "child", "parent", "symlink_parent"])
def test_prepare_refuses_output_source_overlap(tmp_path, monkeypatch, where):
    from locomanilab.data.sample import create_sample
    source = create_sample(tmp_path / "source", episodes=1, frames=3)
    if where == "symlink_parent":
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        out = alias / "new"
    else:
        out = {"same": source, "child": source / "new", "parent": tmp_path}[where]
    before = _snapshot_files(source)
    with pytest.raises(SystemExit):
        run_prepare(monkeypatch, "--dataset-paths", str(source), "--output-path", str(out))
    assert _snapshot_files(source) == before
    assert not (source / "new").exists()


def test_prepare_refuses_missing_declared_video_before_creating_output(tmp_path, monkeypatch):
    from locomanilab.data.sample import create_sample
    source = create_sample(tmp_path / "source", episodes=1, frames=3)
    next((source / "videos").rglob("*.mp4")).unlink()
    before = _snapshot_files(source)
    out = tmp_path / "prepared"
    with pytest.raises(SystemExit, match="missing declared.*video"):
        run_prepare(monkeypatch, "--dataset-paths", str(source), "--output-path", str(out))
    assert not out.exists()
    assert _snapshot_files(source) == before


def test_prepare_refuses_fractional_fps_without_truncating_source(tmp_path, monkeypatch):
    from fractions import Fraction
    from locomanilab.data.sample import create_sample
    from locomanilab.data.validation import validate
    av = pytest.importorskip("av")
    source = create_sample(tmp_path / "source", episodes=1, frames=3)
    video = next((source / "videos").rglob("*.mp4"))
    with av.open(str(video)) as container:
        images = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    rate = Fraction(30000, 1001)
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for image in images:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    file = source / "meta/info.json"
    info = json.loads(file.read_text())
    info["fps"] = float(rate)
    file.write_text(json.dumps(info))
    assert validate(source).ok
    before = _snapshot_files(source)
    out = tmp_path / "prepared"
    with pytest.raises(SystemExit, match="positive integer fps"):
        run_prepare(monkeypatch, "--dataset-paths", str(source), "--output-path", str(out))
    assert not out.exists()
    assert _snapshot_files(source) == before
