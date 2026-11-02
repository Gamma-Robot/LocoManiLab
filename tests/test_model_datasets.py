"""CPU regressions for dataset identity across CLI and backend processes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner
import yaml

from locomanilab.cli import app
from locomanilab.plugin import StageResult
from tests.conftest import AMO_MODALITY, HEFT_MODALITY, SONIC_MODALITY


SONIC_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": [43]},
    "observation.images.ego_view": {"dtype": "video", "shape": [480, 640, 3]},
    "action.motion_token": {"dtype": "float32", "shape": [64]},
    "teleop.left_hand_joints": {"dtype": "float32", "shape": [7]},
    "teleop.right_hand_joints": {"dtype": "float32", "shape": [7]},
}


def _openpi_config(tree, name="recipe", repo_id="my_dataset", factory="LeRobotG1SonicDataConfig"):
    source = Path(tree["repos"]["openpi"]) / "src/openpi/training/config.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    # This is parsed, never imported: no model environment is required.
    source.write_text(
        f'_CONFIGS = [TrainConfig(name={name!r}, '
        f'data={factory}(repo_id={repo_id!r}))]\n'
    )
    return source


def test_cli_convert_and_train_keep_the_same_dataset_alias(plugin, dataset, tmp_path, monkeypatch):
    import locomanilab.cli.models as commands

    source = dataset(SONIC_MODALITY, name="X")
    alias = tmp_path / "bench_sonic_X"
    alias.symlink_to(source, target_is_directory=True)
    adapter = plugin("act", paths={"lerobot_data_root": str(tmp_path / "converted"),
                                   "lerobot_outputs": str(tmp_path / "output")})
    monkeypatch.setattr(commands, "_plugin", lambda *args: (adapter, adapter.ctx))
    convert = CliRunner().invoke(app, ["convert", "-M", "act", "-d", str(alias), "--dry-run"])
    train = CliRunner().invoke(app, ["train", "-M", "act", "-d", str(alias), "--dry-run"])
    expected = str(tmp_path / "converted" / "bench_sonic_X")
    assert convert.exit_code == train.exit_code == 0, convert.output + train.output
    assert f"--out {expected}" in convert.output
    assert f"--dataset.root={expected}" in train.output


@pytest.mark.parametrize("model", ["gr00t", "fastwam", "starvla"])
def test_relative_training_dataset_is_absolute_before_backend_changes_directory(plugin, dataset, ctx, model, monkeypatch):
    source = dataset(SONIC_MODALITY)
    alias = ctx.workspace_root / "relative_dataset"
    alias.symlink_to(source, target_is_directory=True)
    monkeypatch.chdir(ctx.workspace_root)
    adapter = plugin(model)
    options = {"base_model": "/local/base"} if model == "gr00t" else {}
    result = adapter.train(dataset=Path("relative_dataset"), gpus=1, **options)
    assert result.ok, result.message
    assert str(alias) in result.command
    assert str(source) not in result.command


def test_pi05_refuses_recipe_for_another_dataset(plugin, dataset, tree):
    _openpi_config(tree, repo_id="dataset_A")
    source = dataset(SONIC_MODALITY, name="dataset_B")
    result = plugin("pi05").train(dataset=source, gpus=1, config_name="recipe", base_model="/local/params")
    assert not result.ok
    assert "does not match" in result.message and "dataset_A" in result.message
    assert result.command is None


@pytest.mark.parametrize("native", [False, True])
def test_pi05_deploy_uses_the_configured_local_asset_and_dataset_caches(plugin, tmp_path, monkeypatch, native):
    adapter = plugin("pi05", paths={"openpi_data_home": str(tmp_path / "assets"),
                                   "datasets_root": str(tmp_path / "datasets")})
    launches = []

    def record_launch(command, **options):
        launches.append(options)
        return StageResult(ok=True)

    monkeypatch.setattr(adapter.runner, "run", record_launch)
    result = adapter.deploy(model_path=str(tmp_path / "checkpoint"), native=native)
    assert result.ok
    assert launches[0]["extra_env"] == {"OPENPI_DATA_HOME": str(tmp_path / "assets"),
                                       "HF_LEROBOT_HOME": str(tmp_path / "datasets")}


def test_pi05_refuses_recipe_for_another_controller(plugin, dataset, tree):
    _openpi_config(tree, factory="LeRobotG1HeftDataConfig")
    source = dataset(SONIC_MODALITY, name="my_dataset")
    result = plugin("pi05").train(dataset=source, gpus=1, config_name="recipe", base_model="/local/params")
    assert not result.ok and "this controller needs LeRobotG1SonicDataConfig" in result.message


@pytest.mark.parametrize("stage", ["convert", "train"])
def test_pi05_blocks_the_pinned_amo_data_transform_mismatch_before_launch(plugin, dataset, monkeypatch, stage):
    adapter = plugin("pi05")
    monkeypatch.setattr(adapter.runner, "run", lambda *a, **kw: pytest.fail("unsupported AMO backend launched"))
    result = getattr(adapter, stage)(dataset=dataset(AMO_MODALITY), gpus=1, base_model="local/params")
    assert not result.ok and result.command is None
    assert "AMO data factory" in result.message and "separate action and hands" in result.message


def test_pi05_binds_dataset_home_statistics_base_model_and_output(plugin, dataset, tree, tmp_path):
    _openpi_config(tree)
    source = dataset(SONIC_MODALITY, name="my_dataset")
    adapter = plugin("pi05")
    result = adapter.train(dataset=source, gpus=2, config_name="recipe", base_model="/local/params",
                           output_dir=str(tmp_path / "checkpoints"), task="run1")
    assert result.ok, result.message
    assert f"HF_LEROBOT_HOME={source.parent}" in result.command
    assert "--data.repo-id my_dataset" in result.command
    assert "--weight-loader.params-path /local/params" in result.command
    assert f"--checkpoint-base-dir {tmp_path / 'checkpoints'}" in result.command
    assert "--fsdp-devices 2" in result.command
    assert "--overwrite" not in result.command
    assert result.artifacts["run_dir"] == tmp_path / "checkpoints" / "recipe" / "run1"
    converted = adapter.convert(dataset=source, config_name="recipe")
    assert converted.ok, converted.message
    assert "--config-name recipe" in converted.command
    assert "--dataset-home" in converted.command
    assert "artifacts/pi05" in converted.command
    assert not converted.artifacts["norm_stats"].exists()  # dry-run is read-only


def test_pi05_requires_statistics_for_this_dataset_before_real_training(plugin, dataset, tree, monkeypatch):
    _openpi_config(tree)
    source = dataset(SONIC_MODALITY, name="my_dataset")
    adapter = plugin("pi05")
    adapter.runner.dry_run = False
    monkeypatch.setattr(adapter.runner, "run", lambda *a, **kw: pytest.fail("training must not start"))
    result = adapter.train(dataset=source, gpus=1, config_name="recipe", base_model="/local/params")
    assert not result.ok and "lab convert" in result.message


def test_pi05_rejects_passthrough_that_replaces_dataset_binding(plugin, dataset, tree):
    _openpi_config(tree)
    source = dataset(SONIC_MODALITY, name="my_dataset")
    result = plugin("pi05").train(dataset=source, gpus=1, config_name="recipe", base_model="/local/params",
                                  overrides=["data.repo_id=another_dataset"])
    assert not result.ok and "conflicts" in result.message


def test_prts_generated_recipe_selects_the_requested_dataset(plugin, dataset, monkeypatch):
    first = dataset(SONIC_MODALITY, name="A", features=SONIC_FEATURES)
    second = dataset(SONIC_MODALITY, name="B", features=SONIC_FEATURES)
    adapter = plugin("prts")
    a = adapter.train(dataset=first, gpus=1, task="fixed")
    b = adapter.train(dataset=second, gpus=1, task="fixed")
    assert a.ok and b.ok
    assert a.artifacts["recipe"] != b.artifacts["recipe"]
    assert not a.artifacts["recipe"].exists()  # no writes while planning
    assert str(first) in a.message and str(second) in b.message

    # Exercise the actual recipe write while replacing only the costly backend process.
    adapter.runner.dry_run = False
    commands = []
    monkeypatch.setattr(adapter.runner, "run", lambda command, **kw: (
        commands.append(command) or StageResult(ok=True, message="ok")))
    result = adapter.train(dataset=second, gpus=1, task="fixed")
    config = yaml.safe_load(result.artifacts["recipe"].read_text())
    row = config["lerobot_datasets"][0]
    assert row["root"] == str(second) and row["repo_id"] == "locomanilab/B"
    assert row["select_action_keys"] == ["action.motion_token", "teleop.left_hand_joints", "teleop.right_hand_joints"]
    assert commands[0][commands[0].index("--data-path") + 1] == str(result.artifacts["recipe"])


def test_prts_template_keeps_layout_but_replaces_stale_dataset_root(plugin, dataset, tmp_path):
    modality = json.loads(json.dumps(SONIC_MODALITY))
    for group, start, end in (("motion_token", 0, 64), ("left_hand", 64, 71), ("right_hand", 71, 78)):
        modality["action"][group] = {"original_key": "action.custom", "start": start, "end": end}
    source = dataset(modality, features={**SONIC_FEATURES, "action.custom": {"dtype": "float32", "shape": [78]}})
    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump({"lerobot_datasets": [{
        "repo_id": "someone/old", "root": "/old/dataset", "select_action_keys": ["action.custom"]}]}))
    adapter = plugin("prts")
    result = adapter._recipe(source, str(template))
    assert not isinstance(result, StageResult)
    _, config, _ = result
    row = config["lerobot_datasets"][0]
    assert row["root"] == str(source) and row["repo_id"] == "locomanilab/ds"
    assert row["select_action_keys"] == ["action.custom"]


@pytest.mark.parametrize("row,features,message", [
    ({"select_action_keys": ["action.wbc"]}, {"action.wbc": {"dtype": "float32", "shape": [43]}}, "action width is 43"),
    ({"state_relative_action": True}, {}, "state_relative_action: false"),
    ({"select_state_keys": ["observation.short"]}, {"observation.short": {"dtype": "float32", "shape": [32]}}, "state width is 32"),
    ({"select_action_keys": ["action.unknown"]}, {"action.unknown": {"dtype": "float32", "shape": [78]}}, "cannot prove renamed action"),
    ({"select_action_keys": ["teleop.left_hand_joints", "action.motion_token", "teleop.right_hand_joints"]}, {}, "cannot prove renamed action"),
])
def test_prts_refuses_template_that_breaks_server_contract(plugin, dataset, tmp_path, row, features, message):
    source = dataset(SONIC_MODALITY, features={**SONIC_FEATURES, **features})
    template = tmp_path / "invalid.yaml"
    template.write_text(yaml.safe_dump({"lerobot_datasets": [row]}))
    result = plugin("prts").train(dataset=source, gpus=1, config_name=str(template))
    assert not result.ok and message in result.message
    assert result.command is None


def test_prts_requires_width_metadata_before_launch(plugin, dataset):
    result = plugin("prts").train(dataset=dataset(SONIC_MODALITY), gpus=1)
    assert not result.ok and "features" in result.message


def test_prts_accepts_current_sonic_hand_group_names_and_explicit_export_slices(plugin, dataset):
    # The pinned g1_backend G1_modality.json spells action hands *_hand_joints,
    # and includes overlapping diagnostic groups alongside the policy inputs.
    modality = json.loads(json.dumps(SONIC_MODALITY))
    offset = 0
    for name, width in (("left_leg", 6), ("right_leg", 6), ("waist", 3), ("left_arm", 7),
                        ("left_hand", 7), ("right_arm", 7), ("right_hand", 7)):
        modality["state"][name] = {"original_key": "observation.state", "start": offset, "end": offset + width}
        offset += width
    modality["state"]["joint_pos"] = {"original_key": "observation.state", "start": 0, "end": 43}
    modality["action"] = {
        name: {"original_key": column, "start": 0, "end": width}
        for name, column, width in (("motion_token", "action.motion_token", 64),
                                    ("left_hand_joints", "teleop.left_hand_joints", 7),
                                    ("right_hand_joints", "teleop.right_hand_joints", 7),
                                    ("wbc", "action.wbc", 43))}
    source = dataset(modality, features={**SONIC_FEATURES, "action.wbc": {"dtype": "float64", "shape": [43]}})
    result = plugin("prts").train(dataset=source, gpus=1)
    assert result.ok, result.message
    assert "--max-action-dim 78" in result.command


@pytest.mark.parametrize("modality,controller", [(HEFT_MODALITY, "heft"), (AMO_MODALITY, "amo")])
def test_prts_refuses_controller_that_pinned_server_cannot_decode(plugin, dataset, modality, controller):
    result = plugin("prts").train(dataset=dataset(modality), gpus=1)
    assert not result.ok and f"PRTS {controller} training is unavailable" in result.message
    assert "pinned PRTS policy server supports only" in result.message
    assert result.command is None


@pytest.mark.parametrize("model,base_flag,out_flag", [
    ("gr00t", "--base-model-path", "--output-dir"),
    ("starvla", "--framework.qwenvl.base_vlm", "--run_root_dir"),
    ("dm05", "--model-config.model-name-or-path", "--trainer-config.output-dir"),
    ("xr1", "model.params.pretrained=", "trainer.default_root_dir="),
    ("internvla_a15", "--policy.pretrained_path=", "--output_dir="),
    ("pi05", "--weight-loader.params-path", "--checkpoint-base-dir"),
    ("prts", "--model-name-or-path", "--output-dir"),
])
def test_relative_base_is_caller_input_and_output_is_workspace(plugin, dataset, tree, tmp_path, monkeypatch, model, base_flag, out_flag):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    source = dataset(SONIC_MODALITY, name="my_dataset", features=SONIC_FEATURES)
    paths = {"dm05_data_root": str(tmp_path / "dmdata"),
             "xr1_data_root": str(tmp_path / "xrdata"),
             "internvla_a15_data_root": str(tmp_path / "ivdata")}
    adapter = plugin(model, paths=paths)
    options = {}
    if model == "pi05":
        _openpi_config(tree)
        options["config_name"] = "recipe"
    result = adapter.train(dataset=source, gpus=1, base_model="local-base", output_dir="local-output", **options)
    assert result.ok, result.message
    assert f"{base_flag}{'' if base_flag.endswith('=') else ' '}{caller / 'local-base'}" in result.command
    assert f"{out_flag}{'' if out_flag.endswith('=') else ' '}{adapter.ctx.workspace_root / 'local-output'}" in result.command


@pytest.mark.parametrize("model", ["act", "dp", "fastwam"])
def test_relative_training_output_does_not_follow_backend_cwd(plugin, dataset, tmp_path, monkeypatch, model):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    adapter = plugin(model, paths={"lerobot_data_root": str(tmp_path / "converted")})
    result = adapter.train(dataset=dataset(SONIC_MODALITY), gpus=1, output_dir="checkpoints/run")
    assert result.ok, result.message
    assert str(adapter.ctx.workspace_root / "checkpoints/run") in result.command


def test_fastwam_server_relative_base_keeps_caller_location(plugin, tmp_path, monkeypatch):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    adapter = plugin("fastwam")
    server = adapter._sim_server(stage="eval", prompt="test", port=5550, run_dir="run", base_model="local-base")
    assert not isinstance(server, StageResult)
    assert str(caller / "local-base") in " ".join(server.argv)
    assert str(caller / "run") in " ".join(server.argv)


@pytest.mark.parametrize("model", ["gr00t", "fastwam"])
def test_removed_checkpoint_transfer_is_refused_by_legacy_python_calls(plugin, monkeypatch, model):
    adapter = plugin(model)
    monkeypatch.setattr(adapter.runner, "run", lambda *a, **kw: pytest.fail("checkpoint transfer must not run"))
    result = adapter.deploy(fetch=True)
    assert not result.ok and "local checkpoint" in result.message
    assert result.command is None


def test_pi05_statistics_wrapper_keeps_native_imports_in_subprocess(tmp_path, monkeypatch):
    import dataclasses
    import sys
    import types
    from locomanilab.models import _openpi_stats

    @dataclasses.dataclass
    class Assets:
        assets_dir: str | None = None
        asset_id: str | None = None

    @dataclasses.dataclass
    class Data:
        repo_id: str
        assets: Assets = dataclasses.field(default_factory=Assets)

    @dataclasses.dataclass
    class Config:
        data: Data
        assets_base_dir: str = "./assets"

    source = tmp_path / "openpi"
    script = source / "scripts/compute_norm_stats.py"
    script.parent.mkdir(parents=True)
    output = tmp_path / "observed.json"
    script.write_text(
        "import json, os\nfrom openpi.training import config\n"
        "def main(name):\n"
        "    selected = config.get_config(name)\n"
        f"    open({str(output)!r}, 'w').write(json.dumps([os.environ['HF_LEROBOT_HOME'], "
        "selected.data.repo_id, selected.data.assets.assets_dir]))\n"
    )
    config = types.ModuleType("openpi.training.config")
    config._CONFIGS_DICT = {"recipe": Config(Data("my_dataset"))}
    config.get_config = lambda name: config._CONFIGS_DICT[name]
    package = types.ModuleType("openpi")
    training = types.ModuleType("openpi.training")
    training.config = config
    monkeypatch.setitem(sys.modules, "openpi", package)
    monkeypatch.setitem(sys.modules, "openpi.training", training)
    monkeypatch.setitem(sys.modules, "openpi.training.config", config)
    monkeypatch.setenv("HF_LEROBOT_HOME", "/old/cache")
    arguments = ["stats", "--repo", str(source), "--config-name", "recipe", "--repo-id", "my_dataset",
                 "--dataset-home", str(tmp_path / "datasets"), "--assets-base-dir", str(tmp_path / "assets")]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(sys, "path", list(sys.path))
    _openpi_stats.main()
    assert json.loads(output.read_text()) == [str(tmp_path / "datasets"), "my_dataset", str(tmp_path / "assets/recipe")]
    assert not (source / "src/openpi/training/config.py").exists()
