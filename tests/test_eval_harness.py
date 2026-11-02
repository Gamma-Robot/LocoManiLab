"""The shared eval / infer-check harness in ModelPlugin.

Regression for the bug that motivated it: a client flag added to one plugin's
eval() was silently dropped by the other seven. Now there is one eval(); this
checks every plugin forwards the whole client-arg set and the prompt / rate rules.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from locomanilab.common import sim as simmod
from tests.conftest import MODELS

CKPT_KW = {"gr00t": dict(model_path="/tmp/ckpt"), "fastwam": dict(run_dir="/tmp/run"),
           "starvla": dict(model_path="/tmp/c.pt"), "prts": dict(model_path="/tmp/c"),
           "pi05": dict(model_path="/tmp/c", config_name="cfg"),
           "psi0": dict(model_path="/tmp/run"), "act": dict(model_path="/tmp/c"),
           "dp": dict(model_path="/tmp/c"), "dreamzero": dict(model_path="/tmp/c"),
           "internvla_a15": dict(model_path="/tmp/c"), "xr1": dict(model_path="/tmp/run"),
           "dm05": dict(model_path="/tmp/c")}


@pytest.fixture
def harness(monkeypatch, tree):
    """Pretend the sim is installed and capture what eval() hands to orchestrate()."""
    calls = []
    monkeypatch.setattr(simmod, "detect",
                        lambda ctx, conda_env="tele_benchmark", **kwargs:
                        simmod.SimSupport(ok=True, sim_repo=Path(tree["repos"]["sim_repo"])))

    def fake_orchestrate(**kw):
        calls.append(kw)
        from locomanilab.base import StageResult
        return StageResult(ok=True, message="captured")
    monkeypatch.setattr(simmod, "orchestrate", fake_orchestrate)
    return calls


@pytest.mark.parametrize("name", MODELS)
def test_every_plugin_forwards_every_client_flag(plugin, harness, name):
    p = plugin(name)
    res = p.eval(sim=True, prompt="hello", scene_id=401, inference_rate=2.0, robot="G1FloatingBody",
                 results_json="/tmp/r.json", genuine_min_steps=50, token_ema_alpha=0.5,
                 inference_backend="onnx-gpu", fixed_scene=True, scene_seed=7,
                 policy_config="/tmp/p.yaml", reset_jitter=0.05, reset_object_jitter=0.02, jitter_seed=123,
                 backend="sonic", episodes=3, **CKPT_KW[name])
    assert res.ok, res.message
    (kw,) = harness
    extra = kw["extra_client_args"]
    for flag, val in (("--inference-rate", "2.0"), ("--scene-id", "401"),
                      ("--robot", "G1FloatingBody"), ("--results-json", "/tmp/r.json"),
                      ("--genuine-min-steps", "50"), ("--ema-alpha", "0.5"),
                      ("--inference-backend", "onnx-gpu"), ("--scene-seed", "7"),
                      ("--policy-config", "/tmp/p.yaml"), ("--reset-jitter", "0.05"),
                      ("--reset-object-jitter", "0.02"), ("--jitter-seed", "123")):
        assert extra[extra.index(flag) + 1] == val, (name, flag, extra)
    assert "--fixed-scene" in extra
    assert kw["prompt"] == "hello" and kw["episodes"] == 3 and kw["port"] == 5550
    assert kw["server_argv"] and kw["server_cwd"]


def test_default_prompt_is_test_unless_model_says_otherwise(plugin, harness):
    plugin("act").eval(model_path="/tmp/c")
    plugin("gr00t").eval(model_path="/tmp/ckpt")
    assert harness[0]["prompt"] == "test"
    assert harness[1]["prompt"] == "pick up the mug and place it on the plate"


def test_gr00t_trt_bumps_client_rate_unless_user_set_it(plugin, harness):
    p = plugin("gr00t")
    p.eval(model_path="/tmp/ckpt", trt=True, trt_engine="/tmp/eng")
    p.eval(model_path="/tmp/ckpt", trt=True, trt_engine="/tmp/eng", inference_rate=4.0)
    p.eval(model_path="/tmp/ckpt")
    rates = [c["extra_client_args"] for c in harness]
    assert rates[0][rates[0].index("--inference-rate") + 1] == "10.0"
    assert rates[1][rates[1].index("--inference-rate") + 1] == "4.0"
    assert "--inference-rate" not in rates[2]


def test_sim_not_supported_is_refused_before_anything_else(plugin, monkeypatch):
    monkeypatch.setattr(simmod, "detect",
                        lambda ctx, conda_env="tele_benchmark", **kwargs: simmod.SimSupport(ok=False))
    adapter = plugin("fastwam")
    adapter.runner.dry_run = False
    res = adapter.eval()      # not even a checkpoint given
    assert not res.ok and "sim eval not supported" in res.message


@pytest.mark.parametrize("name", MODELS)
def test_infer_check_uses_the_same_server(plugin, name, monkeypatch, tmp_path):
    from locomanilab.common import infer_check as ic
    seen = {}

    def fake(**kw):
        seen.update(kw)
        from locomanilab.base import StageResult
        return StageResult(ok=True, message="captured")
    monkeypatch.setattr(ic, "orchestrate", fake)
    p = plugin(name)
    res = p.infer_check(dataset=tmp_path, episode=2, prompt="test", **CKPT_KW[name])
    assert res.ok, res.message
    assert seen["out"] == f"runs/infer_check/{name}_ep2.png"
    assert seen["prompt"] == "test" and seen["episode"] == 2
    srv = p._sim_server(stage="eval", prompt="test", port=5550, **CKPT_KW[name])
    assert seen["server_argv"] == srv.argv and seen["server_cwd"] == srv.cwd


@pytest.mark.parametrize("name,kw,flag", [
    ("psi0", dict(model_path="/tmp/run", ckpt_step=15000), "--ckpt-step"),
    ("gr00t", dict(model_path="/tmp/ckpt", t5_mode="cache"), "--t5-mode"),
    ("fastwam", dict(run_dir="/tmp/run", trt=True), "--trt"),
    ("act", dict(model_path="/tmp/c", config_name="x"), "--config-name"),
])
def test_model_specific_flags_are_refused_not_swallowed(plugin, harness, name, kw, flag):
    res = plugin(name).eval(**kw)
    assert not res.ok and flag in res.message and "does not use" in res.message
    assert harness == []                       # refused before the server was even built
    res = plugin(name).infer_check(dataset=Path("/tmp/ds"), **kw)
    assert not res.ok and flag in res.message


def test_refuse_unsupported_only_fires_on_values(plugin):
    p = plugin("psi0")
    assert p.refuse_unsupported("deploy", p.deploy, dict(model_path="/x", trt=False, t5_mode=None)) is None
    r = p.refuse_unsupported("deploy", p.deploy, dict(model_path="/x", trt=True, embodiment_tag="T"))
    assert r is not None and "--trt, --embodiment-tag" in r.message
