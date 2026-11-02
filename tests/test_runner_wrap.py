"""Runner.wrap: the one env wrapper every run AND every Popen'd policy server goes through."""
from __future__ import annotations

import shlex

import pytest

from locomanilab.runner import Runner, export_line


@pytest.fixture
def runner(tree):
    return Runner(conda_root=str(tree["conda_root"]), dry_run=True)


def _script(argv):
    assert argv[:2] == ["bash", "-lc"], argv
    return argv[2]


def test_conda_wrap_activates_then_forces_env_bin_first(runner, tree):
    s = _script(runner.wrap(["python", "-c", "print(1)"], env_kind="conda", env_name="foo"))
    parts = s.split(" && ")
    assert parts == [
        f"source {tree['conda_sh']}",
        "conda activate foo",
        'export PATH="$CONDA_PREFIX/bin:$PATH"',     # the fix for a venv inherited on PATH
        "exec python -c 'print(1)'",
    ]


def test_conda_with_setup_script_skips_conda_sh(runner):
    s = _script(runner.wrap(["python"], env_kind="conda", env_name="foo",
                            setup="/data/conda-env.sh"))
    assert s.startswith("source /data/conda-env.sh && conda activate foo && ")
    assert "profile.d/conda.sh" not in s


def test_conda_requires_env_name(runner):
    with pytest.raises(ValueError):
        runner.wrap(["python"], env_kind="conda", env_name=None)


def test_uv_without_prefixes_is_plain_argv(runner):
    assert runner.wrap(["python", "x.py"], env_kind="uv") == \
        ["uv", "run", "--no-sync", "python", "x.py"]


def test_uv_with_setup_cwd_and_pre_keeps_order(runner):
    s = _script(runner.wrap(["python", "srv.py"], env_kind="uv", setup="/s.sh",
                            cwd="/repo dir", pre=[export_line("A", "x y"), "export B=1"]))
    assert s == ("source /s.sh && cd '/repo dir' && export A='x y' && export B=1 "
                 "&& exec uv run --no-sync python srv.py")


def test_none_kind_returns_cmd_unchanged_or_cd_wrapped(runner):
    assert runner.wrap(["ls", "-l"]) == ["ls", "-l"]
    assert _script(runner.wrap(["ls"], cwd="/r")) == "cd /r && exec ls"


def test_venv_wrap(runner):
    s = _script(runner.wrap(["python"], env_kind="venv", venv_path="/v"))
    assert s == "source /v/bin/activate && exec python"


def test_arguments_with_spaces_survive(runner):
    s = _script(runner.wrap(["python", "a.py", "--prompt", "pick up the mug"],
                            env_kind="conda", env_name="e"))
    assert s.endswith("exec python a.py --prompt 'pick up the mug'")
    # and round-trips through the shell's own splitter
    assert shlex.split(s.split("exec ", 1)[1]) == ["python", "a.py", "--prompt", "pick up the mug"]


def test_unknown_kind(runner):
    with pytest.raises(ValueError):
        runner.wrap(["x"], env_kind="docker")


def test_run_dry_run_reports_command(runner, tmp_path):
    res = runner.run(["echo", "hi"], cwd=tmp_path, env_kind="none", stage="t")
    assert res.ok and res.message == "dry-run"
    assert res.command.endswith("echo hi")


def test_unconfigured_conda_root_finds_active_and_miniforge_installs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    forge = home / "miniforge3/etc/profile.d"
    forge.mkdir(parents=True)
    (forge / "conda.sh").write_text("# fake\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CONDA_EXE", raising=False)
    assert Runner()._find_conda_sh() == str(forge / "conda.sh")

    active = tmp_path / "opt/conda"
    (active / "etc/profile.d").mkdir(parents=True)
    (active / "etc/profile.d/conda.sh").write_text("# fake\n")
    monkeypatch.setenv("CONDA_EXE", str(active / "bin/conda"))
    assert Runner()._find_conda_sh() == str(active / "etc/profile.d/conda.sh")
