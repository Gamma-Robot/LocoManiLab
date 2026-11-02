"""Public CLI and filesystem boundaries, exercised without model environments."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from locomanilab import cli, config
from locomanilab.base import StageResult
from locomanilab.config import ConfigError, Context, load_context
from locomanilab.maintenance.cleanup import find_empty_ckpt_dirs, find_empty_datasets
from locomanilab.plan import ExecutionPlan
from locomanilab.runner import Runner


@pytest.fixture
def command(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOCOMANILAB_CONFIG", raising=False)
    monkeypatch.delenv("SOMA_CONFIG", raising=False)
    monkeypatch.delenv("LOCOMANILAB_WORKSPACE", raising=False)
    monkeypatch.delenv("SOMA_WORKSPACE", raising=False)
    return CliRunner()


def test_default_help_and_version_need_no_configuration(command):
    result = command.invoke(cli.app, [])
    assert result.exit_code == 0 and "sample-data" in result.output
    result = command.invoke(cli.app, ["--version"])
    assert result.exit_code == 0 and "1.0.0" in result.output


def test_init_never_overwrites_local_configuration(command, tmp_path):
    target = tmp_path / "my workspace"
    assert command.invoke(cli.app, ["init", str(target)]).exit_code == 0
    path = target / "env.local.yaml"
    before = path.read_bytes()
    result = command.invoke(cli.app, ["init", str(target)])
    assert result.exit_code != 0 and "unchanged" in result.output
    assert path.read_bytes() == before


def test_core_doctor_does_not_require_optional_backends(command):
    result = command.invoke(cli.app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["ok"] and not any("repository:" in row["name"] for row in report["checks"])


def test_doctor_selected_missing_repository_fails_as_json(command):
    result = command.invoke(cli.app, ["doctor", "--model", "act", "--stage", "train", "--json"])
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert not report["ok"]
    assert any(row["name"] == "repository:lerobot" and not row["ok"] for row in report["checks"])


@pytest.mark.parametrize("body", ["schema_version: 99\n", "repos: [broken\n", "paths: []\n"])
def test_doctor_configuration_errors_are_json(command, tmp_path, body):
    path = tmp_path / "broken.yaml"
    path.write_text(body)
    result = command.invoke(cli.app, ["--config", str(path), "doctor", "--json"])
    assert result.exit_code == 1
    report = json.loads(result.stdout)
    assert not report["ok"] and report["error"]
    assert "Traceback" not in result.output


def test_unset_output_variable_is_a_readable_error(command, tmp_path, monkeypatch):
    monkeypatch.delenv("LOCOMANILAB_TEST_MISSING", raising=False)
    cfg = tmp_path / "env.local.yaml"
    cfg.write_text("schema_version: 2\npaths:\n  logs: ${LOCOMANILAB_TEST_MISSING}/logs\n")
    result = command.invoke(cli.app, ["prepare", str(tmp_path), "--dry-run"])
    assert result.exit_code == 1 and "unset environment variable" in result.output
    assert "Traceback" not in result.output


def test_workspace_option_controls_outputs(command, tmp_path):
    work = tmp_path / "another workspace"
    work.mkdir()
    result = command.invoke(cli.app, ["--workspace", str(work), "sample-data", "--out", "sample"])
    assert result.exit_code == 0, result.output
    assert (work / "sample/meta/info.json").is_file()
    assert not (tmp_path / "sample").exists()


def test_config_relative_paths_are_versioned(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(config, "SOURCE_ROOT", checkout)
    workspace = tmp_path / "work"
    workspace.mkdir()
    path = workspace / "env.local.yaml"
    path.write_text("repos: {lerobot: backends/lerobot}\n")
    assert load_context(path).repo_path("lerobot") == checkout / "backends/lerobot"
    path.write_text("schema_version: 2\nrepos: {lerobot: backends/lerobot}\n")
    assert load_context(path).repo_path("lerobot") == workspace / "backends/lerobot"


def test_invalid_path_types_and_unset_variables_are_rejected(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("schema_version: 2\npaths: {logs: 123}\n")
    with pytest.raises(ConfigError, match="strings"):
        load_context(path)
    monkeypatch.delenv("LOCOMANILAB_TEST_MISSING", raising=False)
    with pytest.raises(ConfigError, match="unset environment"):
        Context(paths={"logs": "${LOCOMANILAB_TEST_MISSING}/logs"}).log_root


def test_cli_convert_and_train_preserve_the_same_dataset_alias(command, tmp_path, monkeypatch):
    source = tmp_path / "source" / "X"
    source.mkdir(parents=True)
    alias = tmp_path / "bench_sonic_X"
    alias.symlink_to(source, target_is_directory=True)
    captured = []
    monkeypatch.setattr(cli.models, "_plugin", lambda *_: (object(), Context()))
    def capture(_plugin, stage, **options):
        captured.append((stage, options["dataset"]))
        return StageResult(ok=True, message="captured")
    monkeypatch.setattr(cli.models, "_stage", capture)
    for stage in ("convert", "train"):
        result = command.invoke(cli.app, [stage, "--model", "act", "--dataset", alias.name, "--dry-run"])
        assert result.exit_code == 0, result.output
    assert captured == [("convert", alias), ("train", alias)]


def test_input_path_preserves_parent_traversal_after_a_symlink(tmp_path, monkeypatch):
    work, external = tmp_path / "work", tmp_path / "external"
    (work / "dataset").mkdir(parents=True)
    (external / "child").mkdir(parents=True)
    (external / "dataset").mkdir()
    (work / "dataset/marker").write_text("wrong source")
    (external / "dataset/marker").write_text("chosen source")
    (work / "link").symlink_to(external / "child", target_is_directory=True)
    monkeypatch.chdir(work)
    selected = Context().input_path("link/../dataset")
    assert selected.is_absolute() and "link" in selected.parts
    assert (selected / "marker").read_text() == "chosen source"


def test_eval_check_returns_failure_when_simulator_is_unavailable(command, monkeypatch):
    from locomanilab.common import sim
    monkeypatch.setattr(sim, "detect", lambda *args, **kw: SimpleNamespace(ok=False, summary=lambda: "NOT READY"))
    result = command.invoke(cli.app, ["eval", "--check"])
    assert result.exit_code == 1 and "NOT READY" in result.output


def test_eval_check_uses_the_selected_controller(command, monkeypatch):
    from locomanilab.common import sim
    captured = {}
    def detect(_ctx, **options):
        captured.update(options)
        return SimpleNamespace(ok=True, summary=lambda: "READY")
    monkeypatch.setattr(sim, "detect", detect)
    result = command.invoke(cli.app, ["eval", "--check", "--backend", "amo", "--policy-config", "amo.yaml"])
    assert result.exit_code == 0, result.output
    assert captured == {"conda_env": "tele_benchmark", "backend": "amo", "policy_config": "amo.yaml"}


def test_internal_operations_are_absent_from_public_cli(command):
    for name in ("push-dataset", "repo-push", "repo-setup-tunnel", "sync-upstream"):
        result = command.invoke(cli.app, [name, "--help"])
        assert result.exit_code == 2 and "No such command" in result.output
    result = command.invoke(cli.app, ["deploy", "--model", "act", "--fetch"])
    assert result.exit_code == 2 and "No such option" in result.output


def test_runner_records_local_plan_without_environment_credentials(tmp_path):
    import sys
    values = {"SSHPASS": "local-test-value", "SSH_PRIVATE_KEY": "local-key-value", "VISIBLE": "yes"}
    runner = Runner(log_root=tmp_path / "runs")
    result = runner.run([sys.executable, "-c", "import os; assert os.environ['VISIBLE'] == 'yes'; print('done')"],
                        cwd=tmp_path, extra_env=values)
    assert result.ok and result.output == "done\n"
    text = result.log_path.read_text()
    plan = json.loads(result.log_path.with_suffix(".json").read_text())
    assert "local-test-value" not in text and "local-key-value" not in text
    assert plan["environment"]["SSHPASS"] == "[redacted]"
    assert plan["cwd"] == str(tmp_path)


def test_runner_missing_executable_is_an_actionable_result(tmp_path):
    result = Runner(log_root=tmp_path / "runs").run(["locomanilab-command-that-does-not-exist"], cwd=tmp_path)
    assert not result.ok and "could not start" in result.message


def test_dry_run_does_not_create_logs(tmp_path):
    result = Runner(log_root=tmp_path / "logs", dry_run=True).run(["not-installed"], cwd=tmp_path)
    assert result.ok and not (tmp_path / "logs").exists()


def test_execution_plan_redacts_credentials_and_preserves_local_paths(tmp_path):
    plan = ExecutionPlan(["python", "a.py"], tmp_path, environment={"API_KEY": "dummy", "PATH": "/bin"})
    assert plan.as_dict()["environment"] == {"API_KEY": "[redacted]", "PATH": "/bin"}


def test_cleanup_preserves_linked_and_offline_data(tmp_path, monkeypatch):
    from locomanilab.maintenance import cleanup
    monkeypatch.setattr(cleanup, "_process_cmdlines", lambda: [])
    for kind in ("valid", "broken", "directory"):
        dataset = tmp_path / kind
        (dataset / "meta").mkdir(parents=True)
        (dataset / "meta/info.json").write_text("{}")
        (dataset / "data").mkdir()
        target = tmp_path / f"outside-{kind}"
        if kind == "valid":
            target.write_bytes(b"a parquet placeholder")
        elif kind == "directory":
            target.mkdir()
        (dataset / "data/episode.parquet").symlink_to(target, target_is_directory=kind == "directory")
        assert find_empty_datasets([dataset]) == []


def test_cleanup_preserves_linked_checkpoint_runs(tmp_path, monkeypatch):
    from locomanilab.maintenance import cleanup
    monkeypatch.setattr(cleanup, "_process_cmdlines", lambda: [])
    run = tmp_path / "outputs/run"
    run.mkdir(parents=True)
    (run / "weights").symlink_to(tmp_path / "currently-offline-weights", target_is_directory=True)
    assert find_empty_ckpt_dirs(tmp_path, {"root": "outputs", "run_glob": "*", "ckpt_glob": "*.pt"}) == []


@pytest.mark.parametrize("body", ["schema_version: true\n", "schema_version: 2\nworkspace_root: {bad: value}\n",
                                    "schema_version: 2\nworkspace_root: ${LOCOMANILAB_MISSING_ROOT}/work\n"])
def test_bad_workspace_configuration_never_creates_literal_directories(command, tmp_path, monkeypatch, body):
    monkeypatch.delenv("LOCOMANILAB_MISSING_ROOT", raising=False)
    cfg = tmp_path / "bad-root.yaml"
    cfg.write_text(body)
    result = command.invoke(cli.app, ["--config", str(cfg), "doctor", "--json"])
    assert result.exit_code == 1 and not json.loads(result.stdout)["ok"]
    assert not (tmp_path / "${LOCOMANILAB_MISSING_ROOT}").exists()


def test_prepare_relative_output_uses_selected_workspace(command, tmp_path, monkeypatch):
    work = tmp_path / "workspace"
    source = tmp_path / "source"
    work.mkdir()
    source.mkdir()
    captured = {}
    class Workflow:
        def prepare(self, **options):
            captured.update(options)
            return StageResult(ok=True, message="planned")
    monkeypatch.setattr(cli.data, "_workflow", lambda _: (Workflow(), Context(workspace=work)))
    result = command.invoke(cli.app, ["--workspace", str(work), "prepare", str(source), "--out", "prepared", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert captured["out"] == work / "prepared"


def test_recipe_reaches_deploy_and_offline_inference(command, tmp_path, monkeypatch):
    captured = []
    class Plugin:
        def infer_check(self, **options):
            captured.append(options["config_name"])
            return StageResult(ok=True, message="planned")
    monkeypatch.setattr(cli.models, "_plugin", lambda *_: (Plugin(), Context()))
    def stage(_plugin, _stage, **options):
        captured.append(options["config_name"])
        return StageResult(ok=True, message="planned")
    monkeypatch.setattr(cli.models, "_stage", stage)
    for subcommand, args in (("deploy", []), ("infer-check", ["--dataset", str(tmp_path)])):
        result = command.invoke(cli.app, [subcommand, "--model", "pi05", "--config-name", "my_recipe", "--dry-run", *args])
        assert result.exit_code == 0, result.output
    assert captured == ["my_recipe", "my_recipe"]
