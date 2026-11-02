"""locomanilab.common.variants (lanes, sonic ckpt names, modality checks) and lerobot_meta."""
from __future__ import annotations

import json

import pytest

from locomanilab.common import lerobot_meta, variants
from tests.conftest import AMO_MODALITY, HEFT_MODALITY, SONIC_MODALITY


def test_collection_lane_only_splits_sonic_non_default():
    assert variants.collection_lane("heft", None) == "heft"
    assert variants.collection_lane("sonic", None) == "sonic"
    assert variants.collection_lane("sonic", "/x/checkpoints/sonic/low_latency/g1/policy.yaml") == "sonic"
    assert variants.collection_lane("sonic", "/x/checkpoints/sonic/v1_1/g1/policy.yaml") == "sonic_v1_1"
    assert variants.collection_lane("sonic", "/elsewhere/policy.yaml") == "sonic"   # unknown -> default lane
    assert variants.sonic_ckpt_variant(None) is None


def test_canonical_sonic_ckpt_aliases():
    for alias, want in (("new", "low_latency"), ("low-latency", "low_latency"),
                        ("v1.1", "v1_1"), ("V11", "v1_1"), ("old", "release")):
        assert variants.canonical_sonic_ckpt(alias) == want
    with pytest.raises(ValueError):
        variants.canonical_sonic_ckpt("v9")


def test_sonic_policy_config_and_missing_weights(tmp_path):
    d = tmp_path / "checkpoints" / "sonic" / "v1_1" / "g1"
    d.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        variants.sonic_policy_config(tmp_path, "v1_1")
    (d / "policy.yaml").write_text("x")
    p = variants.sonic_policy_config(tmp_path, "v11")
    assert p == d / "policy.yaml"
    assert variants.missing_sonic_weights(p) == ["model_encoder.onnx", "model_decoder.onnx"]
    (d / "model_encoder.onnx").write_text("")
    assert variants.missing_sonic_weights(p) == ["model_decoder.onnx"]


def test_detect_and_same_controller(dataset):
    table = {"heft": {"detect": {"action": ["ref_joint_pos"]}, "tag": "H"},
             "amo": {"detect": {"action": ["torso_vx", "target_yaw"]}, "tag": "A"},
             "sonic": {"detect": {"action": ["motion_token"]},
                       "require_modality": {"state": ["left_hand", "right_hand"]}, "tag": "S"}}
    s, h, a = dataset(SONIC_MODALITY, "s"), dataset(HEFT_MODALITY, "h"), dataset(AMO_MODALITY, "a")
    assert variants.detect(s, table)[0] == "sonic"
    assert variants.detect(h, table)[0] == "heft"
    assert variants.detect(a, table)[0] == "amo"
    assert isinstance(variants.detect(s, {}), str)
    assert variants.same_controller([s, s]) is None
    msg = variants.same_controller([s, h])
    assert msg and "different controllers" in msg


def test_check_modality_messages(dataset, tmp_path):
    s = dataset(SONIC_MODALITY, "s2")
    assert variants.check_modality(s, {"action": ["motion_token"], "video": ["ego_view"]}, what="X") is None
    msg = variants.check_modality(s, {"action": ["motion_token", "ref_qpos"], "state": ["rpy"]}, what="PRTS")
    assert "required by PRTS: action.ref_qpos, state.rpy" in msg and "present: action=[" in msg
    assert "not found" in variants.check_modality(tmp_path / "nope", {}, what="X")
    (tmp_path / "bad" / "meta").mkdir(parents=True)
    (tmp_path / "bad" / "meta" / "modality.json").write_text("{not json")
    assert "cannot parse" in variants.check_modality(tmp_path / "bad", {}, what="X")


def test_lerobot_meta_readers(dataset, tmp_path):
    d = dataset(SONIC_MODALITY, "m", total_episodes=3)
    assert lerobot_meta.is_dataset_dir(d) and not lerobot_meta.is_dataset_dir(tmp_path)
    assert lerobot_meta.load_info(d)["total_episodes"] == 3
    assert lerobot_meta.modality_keys(lerobot_meta.load_modality(d), "video") == {"ego_view"}
    assert lerobot_meta.load_tasks(d) is None
    (d / "meta" / "tasks.jsonl").write_text('{"task_index": 0, "task": "a"}\n\n{"task_index": 1, "task": "b"}\n')
    assert [t["task"] for t in lerobot_meta.load_tasks(d)] == ["a", "b"]
    assert lerobot_meta.read_jsonl(d / "meta" / "tasks.jsonl", limit=1) == [{"task_index": 0, "task": "a"}]
    with pytest.raises(json.JSONDecodeError):
        (d / "meta" / "info.json").write_text("{")
        lerobot_meta.load_info(d)


def test_validate_is_a_real_gate(dataset, tmp_path):
    from locomanilab.common import dataset as dsmod
    d = dataset(SONIC_MODALITY, "v", total_episodes=1)
    (d / "data" / "chunk-000").mkdir(parents=True)
    r = dsmod.validate(d)
    assert not r.ok and any("tasks.jsonl" in e for e in r.errors)      # no tasks, no parquet
    (d / "meta" / "tasks.jsonl").write_text('{"task_index": 0, "task": "t"}\n')
    r = dsmod.validate(d)
    assert not r.ok and any("no parquet" in e for e in r.errors)
    bare = tmp_path / "bare"
    (bare / "meta").mkdir(parents=True)
    (bare / "data").mkdir()
    r = dsmod.validate(bare)
    assert not r.ok and any("info.json" in e for e in r.errors) and any("modality.json" in e for e in r.errors)
