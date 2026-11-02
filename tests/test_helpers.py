"""ModelPlugin helpers shared by the plugins, and Context path anchoring."""
from __future__ import annotations

from pathlib import Path

from locomanilab.base import StageResult
from locomanilab.config import Context


def test_resolve_gpus_infers_count_from_ids_unless_explicit(plugin):
    p = plugin("gr00t")
    env = {}
    assert p.resolve_gpus(1, "4,5,6", env) == 3 and env["CUDA_VISIBLE_DEVICES"] == "4,5,6"
    assert p.resolve_gpus(2, "4,5,6", {}) == 2          # explicit --gpus wins
    assert p.resolve_gpus(1, "7,", {}) == 1              # trailing comma is not a GPU
    assert p.resolve_gpus(4, None, env) == 4


def test_resolve_ckpt_prefers_explicit_then_env_local(plugin):
    p = plugin("starvla", paths={"starvla_ckpt": "/from/env.pt"})
    assert p.resolve_ckpt(None, None, env_key="starvla_ckpt", stage="eval", hint="h") == Path("/from/env.pt")
    assert p.resolve_ckpt("rel/c.pt", None, env_key="starvla_ckpt", stage="eval", hint="h").is_absolute()
    assert p.resolve_ckpt(None, "/b", env_key="starvla_ckpt", stage="eval", hint="h") == Path("/b")
    res = p.resolve_ckpt(None, None, stage="deploy", hint="--model-path <x>")
    assert isinstance(res, StageResult) and "starvla deploy needs --model-path <x>" == res.message


def test_resolve_ckpt_checks_existence_outside_dry_run(plugin, tmp_path):
    p = plugin("prts")
    p.runner.dry_run = False
    missing = p.resolve_ckpt(str(tmp_path / "nope"), stage="eval", hint="h")
    assert isinstance(missing, StageResult) and "does not exist" in missing.message
    (tmp_path / "ok").mkdir()
    assert p.resolve_ckpt(str(tmp_path / "ok"), stage="eval", hint="h") == tmp_path / "ok"


def test_resolve_ckpt_keeps_symlink_names(plugin, tmp_path):
    real = tmp_path / "bench_dataset" / "sonic" / "X"
    real.mkdir(parents=True)
    link = tmp_path / "bench_sonic_X"
    link.symlink_to(real)
    p = plugin("prts")
    assert p.resolve_ckpt(str(link), stage="eval", hint="h").name == "bench_sonic_X"


def test_context_config_root_and_anchor(tmp_path):
    c = Context(source=tmp_path / "env.local.yaml")
    assert c.config_root == tmp_path
    assert c.anchor("outputs_sim/x") == tmp_path / "outputs_sim" / "x"
    assert c.anchor("/abs/y") == Path("/abs/y")
    assert Context().config_root == Path.cwd()


def test_lerobot_train_refuses_more_than_one_gpu(plugin, tmp_path):
    p = plugin("act", paths={"lerobot_data_root": str(tmp_path / "d"),
                             "lerobot_outputs": str(tmp_path / "o")})
    res = p.train(dataset=tmp_path / "ds", gpus=1, gpu_ids="4,5")
    assert not res.ok and "ONE GPU" in res.message
    ok = p.train(dataset=tmp_path / "ds", gpus=1, gpu_ids="4")
    assert ok.ok and "CUDA_VISIBLE_DEVICES=4" in ok.command


def test_gr00t_set_bare_flag_and_torchrun(plugin, tmp_path):
    p = plugin("gr00t")
    res = p.train(dataset=tmp_path / "ds", gpus=1, gpu_ids="2,3", master_port=29777,
                  base_model="/tmp/base", output_dir="/tmp/out",
                  overrides=["max_steps=10", "no_wandb"])
    assert res.ok
    cmd = res.command.splitlines()[-1]
    assert "torchrun --nproc_per_node=2 --master-port=29777" in cmd
    assert "--max-steps 10 --no-wandb" in cmd and "--no-wandb ''" not in cmd
    assert "--num-gpus 2" in cmd and "--global-batch-size 512" in cmd
