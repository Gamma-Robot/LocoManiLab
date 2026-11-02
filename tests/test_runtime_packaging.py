"""Packaged scripts stay runnable with the chosen child interpreter."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from locomanilab.data import prepare_script_path
from locomanilab.runtime import script_path


RUNTIME_FILES = (
    "sim_client.py", "infer_client.py", "headset_video.py",
    "lerobot/convert.py", "lerobot/server.py", "dm05/convert.py",
    "dm05/experiment.py", "dm05/server.py", "dm05/widen.py",
    "dm05/registry/locomanilab_dm05_datasets.py",
)


@pytest.mark.parametrize("relative", RUNTIME_FILES)
def test_runtime_resources_are_shipped_files(relative):
    path = script_path(relative)
    assert path.is_absolute() and path.is_file()
    assert path.suffix == ".py"


def test_runtime_locator_rejects_outside_or_missing_resources():
    with pytest.raises(ValueError):
        script_path("../config.py")
    with pytest.raises(ValueError):
        script_path("/etc/passwd")
    with pytest.raises(FileNotFoundError):
        script_path("not-packaged.py")


def test_data_and_runtime_locators_do_not_import_model_or_data_stacks(tmp_path):
    code = '''
import sys
import locomanilab.runtime
import locomanilab.data
import locomanilab.data.validation
import locomanilab.data.sample
from locomanilab.common import dataset, lerobot_meta, variants
for name in ('torch', 'pandas', 'numpy', 'pyarrow', 'av', 'mujoco', 'sim2real', 'opendm'):
    assert name not in sys.modules, name
'''
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True,
                   env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})


def test_prepare_executes_by_absolute_path_outside_checkout(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    result = subprocess.run([sys.executable, str(prepare_script_path()), "--help"],
                            cwd=tmp_path, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--dataset-paths" in result.stdout


def test_packaged_headset_runtime_runs_outside_checkout(tmp_path):
    result = subprocess.run([sys.executable, str(script_path("headset_video.py")),
                             "--dry-run", "--source", "zmq:tcp://127.0.0.1:5557"],
                            cwd=tmp_path, capture_output=True, text=True, stdin=subprocess.DEVNULL,
                            env={**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, "PATH": ""},
                            timeout=10)
    assert result.returncode == 0, result.stderr
    assert "would answer OPEN_CAMERA" in result.stdout


@pytest.mark.parametrize("source", ["zmq:tcp://127.0.0.1:5557", "x11:auto", "x11:name=G1 ego view"])
def test_headset_cli_dry_run_does_not_probe_encoders_windows_or_network(tmp_path, source):
    # Execute the packaged argument parser while making any attempted
    # child process or network operation fail, including optional probes.
    launcher = '''
import runpy
import sys
def refuse_external_operation(event, args):
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'socket.connect', 'socket.bind'):
        raise RuntimeError('dry-run attempted an external operation: ' + event)
sys.addaudithook(refuse_external_operation)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-c", launcher, str(script_path("headset_video.py")),
                             "--dry-run", "--source", source], cwd=tmp_path,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            env={**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, "PATH": ""},
                            timeout=10)
    assert result.returncode == 0, result.stderr
    assert "readiness were not checked" in result.stdout
    assert "ffmpeg -hide_banner" in result.stdout
    if source.startswith("x11:"):
        assert "-window_id 0x0" in result.stdout


def test_headset_real_cli_still_requires_ffmpeg_before_listening(tmp_path):
    result = subprocess.run([sys.executable, str(script_path("headset_video.py")),
                             "--source", "zmq:tcp://127.0.0.1:5557"], cwd=tmp_path,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            env={**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"}, "PATH": ""},
                            timeout=10)
    assert result.returncode != 0
    assert "ffmpeg is not on PATH" in result.stderr


def test_legacy_module_aliases_share_the_implementation():
    from locomanilab.common import dataset, lerobot_meta, variants
    from locomanilab.data import metadata, validation
    from locomanilab.data import variants as new_variants
    assert dataset is validation
    assert lerobot_meta is metadata
    assert variants is new_variants


def test_prepare_merge_only_round_trip_uses_real_video(tmp_path):
    pytest.importorskip("av")
    pytest.importorskip("pyarrow")
    pytest.importorskip("pandas")
    from locomanilab.data.sample import create_sample
    from locomanilab.data.validation import validate
    source = create_sample(tmp_path / "source", episodes=2, frames=4)
    out = tmp_path / "prepared"
    result = subprocess.run([sys.executable, str(prepare_script_path()),
                             "--dataset-paths", str(source), "--output-path", str(out),
                             "--no-discard", "--no-smpl", "--no-open", "--no-smooth"],
                            cwd=tmp_path, capture_output=True, text=True,
                            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert result.returncode == 0, result.stderr
    report = validate(out)
    assert report.ok, report.summary()
    assert report.total_episodes == 2 and report.total_frames == 8
    assert list(out.rglob("*.mp4"))


def _snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def _convert(relative, *args):
    return subprocess.run([sys.executable, str(script_path(relative)), *map(str, args)],
                          capture_output=True, text=True,
                          env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"})


@pytest.mark.parametrize("where", ["same", "child", "parent", "symlink_parent"])
def test_lerobot_force_never_removes_a_source_or_its_parent(tmp_path, where):
    from locomanilab.data.sample import create_sample
    source = create_sample(tmp_path / "source", episodes=1, frames=3)
    if where == "symlink_parent":
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        out = alias / "converted"
    else:
        out = {"same": source, "child": source / "converted", "parent": tmp_path}[where]
    before = _snapshot(source)
    result = _convert("lerobot/convert.py", "--src", source, "--out", out,
                      "--force", "--action-parts", "action")
    assert result.returncode != 0
    assert "overlap source" in result.stderr
    assert _snapshot(source) == before
    assert not (source / "converted").exists()


def test_failed_force_conversion_keeps_the_previous_output(tmp_path):
    from locomanilab.data.sample import create_sample
    source = create_sample(tmp_path / "source", episodes=1, frames=3)
    out = tmp_path / "converted"
    first = _convert("lerobot/convert.py", "--src", source, "--out", out, "--action-parts", "action")
    assert first.returncode == 0, first.stderr
    before = _snapshot(out)
    (source / "meta/episodes_stats.jsonl").write_text("broken\n")
    second = _convert("lerobot/convert.py", "--src", source, "--out", out,
                      "--action-parts", "action", "--force")
    assert second.returncode != 0
    assert _snapshot(out) == before
    assert not list(tmp_path.glob(".converted.staging-*"))


def _sonic_sample(out, episodes=2):
    import json
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from locomanilab.data.sample import create_sample
    source = create_sample(out, episodes=episodes, frames=3)
    for path in (source / "data").rglob("*.parquet"):
        rows = pq.read_table(path).to_pydict()
        rows.pop("action")
        rows["observation.state"] = np.zeros((3, 43), np.float32).tolist()
        rows["action.motion_token"] = np.zeros((3, 64), np.float32).tolist()
        rows["teleop.left_hand_joints"] = np.zeros((3, 7), np.float32).tolist()
        rows["teleop.right_hand_joints"] = np.zeros((3, 7), np.float32).tolist()
        pq.write_table(pa.table(rows), path)
    file = source / "meta/info.json"
    info = json.loads(file.read_text())
    info["features"].pop("action")
    for key, width in (("observation.state", 43), ("action.motion_token", 64),
                       ("teleop.left_hand_joints", 7), ("teleop.right_hand_joints", 7)):
        info["features"][key] = {"dtype": "float32", "shape": [width], "names": None}
    file.write_text(json.dumps(info))
    (source / "meta/modality.json").write_text(json.dumps({
        "state": {"proprio": {"start": 0, "end": 43, "original_key": "observation.state"}},
        "action": {key: {"start": 0, "end": width, "original_key": key}
                   for key, width in (("action.motion_token", 64),
                                      ("teleop.left_hand_joints", 7), ("teleop.right_hand_joints", 7))},
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
    }))
    from locomanilab.data.validation import validate
    report = validate(source)
    assert report.ok, report.summary()
    return source


def test_lerobot_sonic_conversion_preserves_action_order_statistics_and_source(tmp_path):
    import json
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = _sonic_sample(tmp_path / "source", episodes=1)
    parquet = next((source / "data").rglob("*.parquet"))
    rows = pq.read_table(parquet).to_pydict()
    stats_path = source / "meta/episodes_stats.jsonl"
    record = json.loads(stats_path.read_text())
    record["stats"].pop("action")
    for key, width, offset in (("observation.state", 43, 0), ("action.motion_token", 64, 10),
                               ("teleop.left_hand_joints", 7, 20), ("teleop.right_hand_joints", 7, 30)):
        values = np.arange(3 * width, dtype=np.float32).reshape(3, width) / 4 + offset
        rows[key] = values.tolist()
        record["stats"][key] = {field: getattr(values, field)(axis=0).tolist()
                                for field in ("mean", "std", "min", "max")}
        record["stats"][key]["count"] = [3]
    pq.write_table(pa.table(rows), parquet)
    stats_path.write_text(json.dumps(record) + "\n")
    before = _snapshot(source)
    out = tmp_path / "converted"
    result = _convert("lerobot/convert.py", "--src", source, "--out", out)
    assert result.returncode == 0, result.stderr
    assert _snapshot(source) == before
    parts = ("action.motion_token", "teleop.left_hand_joints", "teleop.right_hand_joints")
    expected = np.concatenate([rows[key] for key in parts], axis=1)
    converted = pq.read_table(out / parquet.relative_to(source))
    np.testing.assert_allclose(converted["action"].to_pylist(), expected)
    assert converted.schema.field("action").type.value_type == pa.float32()
    info = json.loads((out / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["features"]["action"]["shape"] == [78]
    statistics = json.loads((out / "meta/episodes_stats.jsonl").read_text())["stats"]
    np.testing.assert_allclose(statistics["action"]["mean"], expected.mean(axis=0))
    assert len(statistics["observation.images.ego_view"]["mean"]) == 3
    assert (out / "videos").resolve() == source / "videos"


def test_dm05_refuses_reuse_so_old_episode_files_cannot_survive(tmp_path):
    import json
    source = _sonic_sample(tmp_path / "two", episodes=2)
    out = tmp_path / "converted"
    first = _convert("dm05/convert.py", "--data-root", source, "--out-dir", out,
                     "--workers", "1", "--no-verify-frames")
    assert first.returncode == 0, first.stderr
    record = json.loads((out / "opendm_dataset.json").read_text())
    assert Path(record["jsonl_dir"]) == out / "jsonl"
    assert len(list((out / "jsonl").glob("episode_*.jsonl"))) == record["episodes"] == 2
    before = _snapshot(out)
    other = _sonic_sample(tmp_path / "one", episodes=1)
    second = _convert("dm05/convert.py", "--data-root", other, "--out-dir", out,
                      "--workers", "1", "--no-verify-frames")
    assert second.returncode != 0 and "already exists" in second.stderr
    assert _snapshot(out) == before


@pytest.mark.parametrize("where", ["same", "child", "parent"])
def test_dm05_never_converts_into_its_source_tree(tmp_path, where):
    source = _sonic_sample(tmp_path / "source", episodes=1)
    out = {"same": source, "child": source / "converted", "parent": tmp_path}[where]
    before = _snapshot(source)
    result = _convert("dm05/convert.py", "--data-root", source, "--out-dir", out, "--no-verify-frames")
    assert result.returncode != 0 and "overlap source" in result.stderr
    assert _snapshot(source) == before


def test_output_file_guard_refuses_a_symlinked_parent(tmp_path):
    from locomanilab.data.paths import dataset_file
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "data").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        dataset_file(output, "data/episode_{episode_index:06d}.parquet", episode_index=0, output=True)
    assert list(source.iterdir()) == []


def test_dm05_server_supplies_checkpoint_before_dataclass_post_init(tmp_path):
    """Run the real experiment/server with CPU-only stand-ins for upstream OpenDM."""
    import textwrap
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"action_dim": 78}')
    code = textwrap.dedent('''
        import dataclasses as dc
        import sys, types
        from pathlib import Path
        def module(name, package=False):
            result = types.ModuleType(name)
            if package:
                result.__path__ = []
            sys.modules[name] = result
            if '.' in name:
                parent, child = name.rsplit('.', 1)
                setattr(sys.modules[parent], child, result)
            return result
        module('tyro')
        module('PIL', True)
        module('PIL.Image')
        for name in ('opendm', 'opendm.exp', 'opendm.constants', 'opendm.data'):
            module(name, True)
        upstream = module('opendm.exp.dm05_exp')
        for name in ('DM05DataConfig', 'DM05ModelConfig', 'DM05OptimizerConfig', 'DM05TrainerConfig'):
            setattr(upstream, name, dc.dataclass(type(name, (), {})))
        @dc.dataclass
        class Inference:
            diffusion_steps: int = 5
            default_robot_type: str = 'Unitree G1'
        upstream.DM05InferenceConfig = Inference
        @dc.dataclass
        class Experiment:
            task: str = 'train'
            def __post_init__(self):
                assert self.model_config.model_name_or_path is not None
                assert Path(self.model_config.model_name_or_path, 'config.json').is_file()
                self.post_init_count = getattr(self, 'post_init_count', 0) + 1
            def _initialize_inference_runtime(self):
                self.initialized = True
        upstream.DM05Exp = Experiment
        module('opendm.constants.precision').FP32_MIXED_PRECISION_POLICY = 'fp32_mixed'
        module('opendm.constants.robot').ActionMode = types.SimpleNamespace(ABSOLUTE='absolute')
        module('opendm.data.transforms').PadAction = lambda width: width
        from locomanilab.runtime.dm05.server import DM05GrootPolicy
        policy = DM05GrootPolicy(Path(sys.argv[1]), 'synthetic', 'test', 7, None, suffix_cuda_graph=True)
        assert policy.exp.post_init_count == 1
        assert policy.exp.initialized
        assert policy.exp.data_config.dataset_name == 'synthetic'
        assert policy.exp.inference_config.diffusion_steps == 7
        assert policy.exp.data_config.norm_stats_root == str(Path(sys.argv[1]) / 'norm_stats')
    ''')
    result = subprocess.run([sys.executable, "-c", code, str(checkpoint)],
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
