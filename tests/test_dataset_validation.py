"""CPU data contract checks against an actual small parquet + MP4 dataset."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("av")

from locomanilab.data.sample import create_sample
from locomanilab.data.validation import validate


@pytest.fixture
def sample(tmp_path):
    return create_sample(tmp_path / "sample", episodes=2, frames=4)


def edit_meta(sample: Path, name: str, edit):
    file = sample / "meta" / name
    value = json.loads(file.read_text())
    edit(value)
    file.write_text(json.dumps(value))


def rewrite_episode(sample: Path, change, episode: int = 1):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = sample / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    rows = pq.read_table(path).to_pydict()
    change(rows)
    pq.write_table(pa.table(rows), path)


def test_synthetic_sample_is_a_real_valid_dataset(sample):
    report = validate(sample)
    assert report.ok, report.summary()
    assert report.total_episodes == 2 and report.total_frames == 8
    assert report.state_dim == 4 and report.action_keys == {"command": 2}
    assert report.video_keys == ["observation.images.ego_view"]
    info = json.loads((sample / "meta/info.json").read_text())
    assert info["collection_info"]["synthetic"] is True


def test_sample_is_deterministic_and_never_overwrites(sample, tmp_path):
    import av
    import numpy as np
    import pyarrow.parquet as pq

    def decoded_video(path):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            shape = (stream.width, stream.height)
            codec = stream.codec_context.name
            rate = stream.average_rate
            # Compare all native YUV planes without RGB converter rounding.
            frames = [(frame.pts * frame.time_base, frame.to_ndarray(format="yuv420p"))
                      for frame in container.decode(stream)]
        return shape, codec, rate, frames

    other = create_sample(tmp_path / "other", episodes=2, frames=4)
    for original in sample.rglob("*"):
        counterpart = other / original.relative_to(sample)
        if original.suffix == ".parquet":
            assert pq.read_table(original).equals(pq.read_table(counterpart))
        elif original.suffix in (".json", ".jsonl"):
            assert original.read_bytes() == counterpart.read_bytes()
        elif original.suffix == ".mp4":
            # Encoded bytes may vary; the sample's pixels and timeline must not.
            shape, codec, rate, frames = decoded_video(original)
            other_shape, other_codec, other_rate, other_frames = decoded_video(counterpart)
            assert (shape, codec, rate) == (other_shape, other_codec, other_rate)
            assert len(frames) == len(other_frames)
            for (timestamp, pixels), (other_timestamp, other_pixels) in zip(frames, other_frames):
                assert timestamp == other_timestamp
                assert np.array_equal(pixels, other_pixels)
    before = (sample / "meta/info.json").read_bytes()
    with pytest.raises(FileExistsError):
        create_sample(sample)
    assert (sample / "meta/info.json").read_bytes() == before


def test_sample_video_preserves_the_synthetic_yuv_pattern(sample):
    import av
    import numpy as np

    for episode, path in enumerate(sorted((sample / "videos").rglob("*.mp4"))):
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                pixels = frame.to_ndarray(format="yuv420p").reshape(-1)
                y = pixels[:4096].reshape(64, 64)
                u, v = pixels[4096:5120].reshape(32, 32), pixels[5120:].reshape(32, 32)
                # A lossless encode/decode must preserve the declared source pattern,
                # including every luma level and the two colored chroma blocks.
                level = 32 + (index * 9 + episode * 31) % 96
                assert np.array_equal(y, np.broadcast_to(np.arange(64) + level, (64, 64)))
                for plane, color in ((u, 160), (v, 96)):
                    expected = np.full((32, 32), 128)
                    expected[8:24, 8:24] = color
                    assert np.array_equal(plane, expected)


def test_validation_inspects_later_episodes_and_every_action_row(sample):
    rewrite_episode(sample, lambda rows: rows["action"].__setitem__(2, [1.0]))
    report = validate(sample)
    assert not report.ok and any("episode 1 action row 2" in e for e in report.errors)


def test_validation_refuses_missing_action_feature(sample):
    rewrite_episode(sample, lambda rows: rows.pop("action"))
    report = validate(sample)
    assert not report.ok and any("missing column 'action'" in e for e in report.errors)


def test_validation_refuses_nonfinite_action(sample):
    rewrite_episode(sample, lambda rows: rows["action"].__setitem__(0, [float("nan"), 0.0]))
    report = validate(sample)
    assert not report.ok and any("finite numeric" in e for e in report.errors)


def test_validation_refuses_invalid_action_slice(sample):
    edit_meta(sample, "modality.json", lambda obj: obj["action"]["command"].update(end=3))
    report = validate(sample)
    assert not report.ok and any("exceeds action width" in e for e in report.errors)


def test_validation_refuses_undeclared_task(sample):
    rewrite_episode(sample, lambda rows: rows["task_index"].__setitem__(0, 99))
    report = validate(sample)
    assert not report.ok and any("task_index values absent" in e for e in report.errors)


def test_validation_refuses_metadata_count_mismatch(sample):
    edit_meta(sample, "info.json", lambda obj: obj.update(total_frames=999, total_episodes=9))
    report = validate(sample)
    assert not report.ok
    assert any("episode count mismatch" in e for e in report.errors)
    assert any("total_frames=999" in e for e in report.errors)


def test_validation_refuses_missing_video(sample):
    next((sample / "videos").rglob("*.mp4")).unlink()
    report = validate(sample)
    assert not report.ok and any("missing video" in e for e in report.errors)


def test_validation_checks_decoded_video_frame_count(sample):
    import av
    import numpy as np
    path = next((sample / "videos").rglob("*.mp4"))
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        image = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")
        for packet in stream.encode(image):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    report = validate(sample)
    assert not report.ok and any("video frames=1 != episode rows=4" in e for e in report.errors)


def test_validation_reports_broken_json_without_traceback(sample):
    (sample / "meta/info.json").write_text("{")
    report = validate(sample)
    assert not report.ok and any("cannot read meta/info.json" in e for e in report.errors)


def test_validation_reports_malformed_modality_without_traceback(sample):
    edit_meta(sample, "modality.json", lambda obj: obj.update(action=["bad"], video=["bad"]))
    report = validate(sample)
    assert not report.ok and any("modality.action must be an object" in e for e in report.errors)


def test_validation_reports_malformed_feature_without_traceback(sample):
    edit_meta(sample, "info.json", lambda obj: obj["features"].update(action=["bad"]))
    report = validate(sample)
    assert not report.ok and any("features.action needs" in e for e in report.errors)


def test_validation_refuses_missing_video_dependencies(sample, monkeypatch):
    import builtins
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "av":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    report = validate(sample)
    assert not report.ok and any("video validation needs PyAV" in e for e in report.errors)


def test_validation_refuses_missing_data_dependencies(sample, monkeypatch):
    import builtins
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pyarrow.parquet":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    report = validate(sample)
    assert not report.ok and any("install locomanilab[data]" in e for e in report.errors)


def test_validation_does_not_read_paths_outside_the_dataset(sample):
    edit_meta(sample, "info.json", lambda obj: obj.update(data_path="../outside.parquet"))
    report = validate(sample)
    assert not report.ok and any("must stay inside" in e for e in report.errors)


def test_validation_reports_nonstring_video_original_key(sample):
    edit_meta(sample, "modality.json", lambda obj: obj["video"]["ego_view"].update(original_key=["bad"]))
    report = validate(sample)
    assert not report.ok and any("undeclared feature" in error for error in report.errors)


def test_real_cli_relative_workspace_data_roundtrip(tmp_path):
    """Exercise documented relative paths in fresh CLI processes outside the repo."""
    workspace = tmp_path / "example workspace"
    environment = {key: value for key, value in os.environ.items() if not key.startswith(("LOCOMANILAB_", "SOMA_"))}
    environment.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")

    def cli(*args, cwd):
        result = subprocess.run([sys.executable, "-m", "locomanilab.cli", *args], cwd=cwd, env=environment,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        assert result.returncode == 0, result.stdout
        return result.stdout

    cli("init", str(workspace), cwd=tmp_path)
    cli("sample-data", "--out", "datasets/example", "--episodes", "2", "--frames", "4", cwd=workspace)
    assert "status: OK" in cli("validate", "datasets/example", cwd=workspace)
    cli("prepare", "datasets/example", "--merge-only", "--out", "datasets/prepared", cwd=workspace)
    assert "status: OK" in cli("validate", "datasets/prepared", cwd=workspace)
    info = json.loads((workspace / "datasets/prepared/meta/info.json").read_text())
    assert info["total_episodes"] == 2 and info["total_frames"] == 8
