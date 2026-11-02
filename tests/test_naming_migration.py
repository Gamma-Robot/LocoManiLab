"""New outputs use current names while existing local artifacts remain usable."""
import json
from pathlib import Path

import pytest

from locomanilab.models._legacy import converter_path
from locomanilab.models._recipes import dataset_identity
from locomanilab.plugin import StageResult
from tests.conftest import SONIC_MODALITY
from tests.test_model_datasets import _openpi_config


def test_sim_collection_stream_endpoint_reaches_current_and_existing_checkouts(ctx, tmp_path, monkeypatch):
    from locomanilab.runner import Runner
    from locomanilab.workflows import Workflows

    workflow = Workflows(ctx, Runner(dry_run=True, log_root=tmp_path / "logs"))
    launches = []
    monkeypatch.setattr(workflow, "_start_headset_video", lambda *a: None)
    monkeypatch.setattr(workflow, "_stop_headset_video", lambda *a: None)
    monkeypatch.setattr(workflow.runner, "run", lambda *a, **kw: launches.append(kw) or StageResult(ok=True))
    result = workflow.collect(task_prompt="test", out_dir=tmp_path / "data", sim=True,
                              headset_video=True, headset_source="sim")
    assert result.ok, result.message
    environment = launches[0]["extra_env"]
    assert environment["LOCOMANILAB_EGO_STREAM"] == environment["SOMA_EGO_STREAM"]


@pytest.mark.parametrize("suffix", ["sonic_to_psi_lerobot", "amo_to_psi_lerobot",
                                   "heft_to_psi_lerobot", "lerobot_v21_to_v30", "sonic_to_xr1_json"])
def test_existing_model_checkout_converter_and_current_name_precedence(tmp_path, suffix):
    canonical = f"scripts/data/locomanilab_{suffix}.py"
    legacy = f"scripts/data/soma_{suffix}.py"
    old = tmp_path / legacy
    old.parent.mkdir(parents=True)
    old.write_text("legacy converter")
    assert converter_path(tmp_path, canonical) == legacy
    (tmp_path / canonical).write_text("current converter")
    assert converter_path(tmp_path, canonical) == canonical
    assert old.read_text() == "legacy converter"
    assert converter_path(tmp_path, "custom.py") == "custom.py"


@pytest.mark.parametrize("marker", ["locomanilab_dataset.json", "soma_dataset.json"])
def test_pi05_existing_normalization_marker_is_accepted_without_rewriting(
    plugin, dataset, tree, tmp_path, monkeypatch, marker,
):
    _openpi_config(tree)
    source = dataset(SONIC_MODALITY, name="my_dataset")
    adapter = plugin("pi05")
    _, _, _, base = adapter._recipe(source, "recipe")
    stats = base / "recipe/my_dataset"
    stats.mkdir(parents=True)
    (stats / "norm_stats.json").write_text("{}")
    (stats / marker).write_text(json.dumps(dataset_identity(source)))
    before = {p.name: p.read_bytes() for p in stats.iterdir()}
    adapter.runner.dry_run = False
    monkeypatch.setattr(adapter.runner, "run", lambda *a, **kw: StageResult(ok=True))
    assert adapter.train(dataset=source, gpus=1, config_name="recipe", base_model=str(tmp_path)).ok
    assert {p.name: p.read_bytes() for p in stats.iterdir()} == before


def test_pi05_current_mismatching_marker_never_falls_back_to_an_old_matching_one(
    plugin, dataset, tree, monkeypatch,
):
    _openpi_config(tree)
    source = dataset(SONIC_MODALITY, name="my_dataset")
    adapter = plugin("pi05")
    _, _, _, base = adapter._recipe(source, "recipe")
    stats = base / "recipe/my_dataset"
    stats.mkdir(parents=True)
    (stats / "norm_stats.json").write_text("{}")
    (stats / "soma_dataset.json").write_text(json.dumps(dataset_identity(source)))
    (stats / "locomanilab_dataset.json").write_text("{}")
    adapter.runner.dry_run = False
    monkeypatch.setattr(adapter.runner, "run", lambda *a, **kw: pytest.fail("mismatched training launched"))
    assert not adapter.train(dataset=source, gpus=1, config_name="recipe", base_model="local").ok


def test_pi05_conversion_writes_only_current_identity_name(plugin, dataset, tree, monkeypatch):
    _openpi_config(tree)
    source = dataset(SONIC_MODALITY, name="my_dataset")
    adapter = plugin("pi05")
    _, _, _, base = adapter._recipe(source, "recipe")
    stats = base / "recipe/my_dataset"
    adapter.runner.dry_run = False

    def complete_native_statistics(*args, **kwargs):
        stats.mkdir(parents=True)
        (stats / "norm_stats.json").write_text("{}")
        return StageResult(ok=True)

    monkeypatch.setattr(adapter.runner, "run", complete_native_statistics)
    assert adapter.convert(dataset=source, config_name="recipe").ok
    assert json.loads((stats / "locomanilab_dataset.json").read_text()) == dataset_identity(source)
    assert not (stats / "soma_dataset.json").exists()


@pytest.mark.parametrize("record_name", ["locomanilab_train50_merged", "soma_train50_merged"])
def test_dm05_serving_uses_existing_conversion_registration(plugin, tmp_path, record_name):
    root = tmp_path / "converted"
    record = root / "train50_merged/opendm_dataset.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"name": record_name}))
    adapter = plugin("dm05", paths={"dm05_data_root": str(root)})
    command = adapter._server_cmd(tmp_path / "checkpoint", 5550, None)
    assert command[command.index("--dataset-name") + 1] == record_name
    assert "SOMA_DM05_DATA_ROOT" not in adapter._stage_env("deploy")


@pytest.mark.parametrize("current_exists", [False, True])
def test_xr1_training_reads_existing_config_and_prefers_current_name(
    plugin, dataset, tree, tmp_path, monkeypatch, current_exists,
):
    source = dataset(SONIC_MODALITY, name="task")
    adapter = plugin("xr1", paths={"xr1_pretrained": str(tmp_path / "checkpoint")})
    directory = Path(tree["repos"]["xr1"]) / "xr1/configs/data"
    directory.mkdir(parents=True)
    (directory / "soma_task.yaml").write_text("legacy config")
    if current_exists:
        (directory / "locomanilab_task.yaml").write_text("current config")
    commands = []
    adapter.runner.dry_run = False
    monkeypatch.setattr(adapter.runner, "run", lambda command, **kw: commands.append(command) or StageResult(ok=True))
    result = adapter.train(dataset=source, gpus=1)
    assert result.ok, result.message
    selected = "locomanilab_task" if current_exists else "soma_task"
    assert any(f"data={selected}" == str(arg) for arg in commands[0]), commands
    assert (directory / "soma_task.yaml").read_text() == "legacy config"
