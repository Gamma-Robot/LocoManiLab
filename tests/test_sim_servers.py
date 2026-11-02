"""Every plugin's `_sim_server` hook: the server argv the sim harness will Popen.

These are golden strings. A change here must be a change you meant -- it is the
exact command a machine runs, and the sim/infer-check harness never goes through
`runner.run`, so this is the only place it is checked before a 15-minute eval.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from locomanilab.base import SimServer, StageResult
from tests.conftest import MODELS


def _sh(argv):
    assert argv[:2] == ["bash", "-lc"], argv
    return argv[2]


def _server(p, **kw):
    s = p._sim_server(stage="eval", prompt=kw.pop("prompt", None), port=kw.pop("port", 5550), **kw)
    assert isinstance(s, SimServer), getattr(s, "message", s)
    return s


@pytest.mark.parametrize("name", MODELS)
def test_every_plugin_has_a_sim_server_hook(plugin, name):
    from locomanilab.base import ModelPlugin
    p = plugin(name)
    assert type(p)._sim_server is not ModelPlugin._sim_server, f"{name} has no _sim_server hook"
    # calling with no checkpoint must refuse politely, never raise
    res = p._sim_server(stage="eval", prompt=None, port=5550)
    assert isinstance(res, StageResult) and not res.ok
    assert "eval" in res.message


def test_gr00t_server_is_uv_and_trt_sets_client_rate(plugin, tree):
    p = plugin("gr00t")
    s = _server(p, model_path="/tmp/ckpt")
    assert s.argv == ["uv", "run", "--no-sync", "python", "gr00t/eval/run_gr00t_server.py",
                      "--model-path", "/tmp/ckpt", "--embodiment-tag", "UNITREE_G1_SONIC",
                      "--device", "cuda:0", "--port", "5550"]
    assert s.cwd == Path(tree["repos"]["isaac_gr00t"])
    assert s.prompt == "pick up the mug and place it on the plate" and s.inference_rate is None
    s2 = _server(p, model_path="/tmp/ckpt", trt=True, trt_engine="/tmp/eng", action_space="smpl")
    assert s2.inference_rate == 10.0
    assert "--trt-engine-path" in s2.argv and "UNITREE_G1_SONIC" not in s2.argv


def test_gr00t_server_sources_setup_on_cluster(plugin, tree):
    p = plugin("gr00t", paths={"uv_setup": "/data/setup_env.sh"})
    s = _server(p, model_path="/tmp/ckpt")
    assert _sh(s.argv) == ("source /data/setup_env.sh && exec uv run --no-sync python "
                           "gr00t/eval/run_gr00t_server.py --model-path /tmp/ckpt "
                           "--embodiment-tag UNITREE_G1_SONIC --device cuda:0 --port 5550")


def test_fastwam_server_conda_with_path_fix(plugin, tree):
    p = plugin("fastwam")
    fw = tree["repos"]["fastwam"]
    s = _server(p, run_dir="/tmp/run", prompt="hello", ckpt_step=5, t5_mode="cache")
    assert _sh(s.argv) == (
        f"source {tree['conda_sh']} && conda activate fastwam && "
        'export PATH="$CONDA_PREFIX/bin:$PATH" && '
        f"exec python deployment/run_fastwam_server.py --run-dir /tmp/run "
        f"--base-model-path {fw}/checkpoints/Wan-AI/Wan2.2-TI2V-5B --port 5550 "
        f"--checkpoint /tmp/run/checkpoints/weights/step_000005.pt "
        f"--t5-cache /tmp/run/cache_t5 --initial-prompt hello")
    assert s.cwd == Path(fw) and s.prompt == "hello"


def test_fastwam_accepts_model_path_alias_and_env_override(plugin, tree):
    p = plugin("fastwam", conda_envs={"fastwam": "fastwam-cluster"})
    s = _server(p, model_path="/tmp/run2")
    assert "conda activate fastwam-cluster" in _sh(s.argv)
    assert "--run-dir /tmp/run2" in _sh(s.argv)


def test_starvla_and_prts_export_pythonpath_before_activation(plugin, tree):
    for name, repo_key, extra in (("starvla", "starvla", ""), ("prts", "prts",
                                  ' && export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"')):
        p = plugin(name)
        s = _server(p, model_path="/tmp/c", prompt="hello")
        sh = _sh(s.argv)
        repo = tree["repos"][repo_key]
        assert sh.startswith(f"export PYTHONPATH={repo}${{PYTHONPATH:+:${{PYTHONPATH}}}}{extra} && source ")
        assert 'export PATH="$CONDA_PREFIX/bin:$PATH" && exec python' in sh


def test_pi05_server_is_uv_with_cd_and_exports(plugin, tree):
    p = plugin("pi05", paths={"openpi_data_home": "/fake/home", "datasets_root": "/fake/ds"})
    s = _server(p, model_path="/tmp/c", prompt="hello", config_name="mycfg")
    repo = tree["repos"]["openpi"]
    assert _sh(s.argv) == (
        f"cd {repo} && export HF_LEROBOT_HOME=/fake/ds && export OPENPI_DATA_HOME=/fake/home && "
        "exec uv run --no-sync python deployment/run_openpi_server.py --config-name mycfg "
        "--ckpt-path /tmp/c --port 5550 --initial-prompt hello")


def test_psi0_server_uv_with_hf_offline(plugin, tree):
    p = plugin("psi0", paths={"psi0_home": "/fake/psi"})
    s = _server(p, model_path="/tmp/run/checkpoints/ckpt_500")
    repo = tree["repos"]["psi0"]
    assert _sh(s.argv) == (
        f"cd {repo} && export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1 && "
        "export HF_HOME=/fake/psi/hf_hub && export HF_HUB_CACHE=/fake/psi/hf_hub && "
        "exec uv run --no-sync python src/psi/deploy/run_psi0_server.py --run-dir /tmp/run "
        "--ckpt-step 500 --host 0.0.0.0 --port 5550 --num-inference-steps 8")


def test_lerobot_server_prefers_venv_python(plugin, tree):
    repo = Path(tree["repos"]["lerobot"])
    p = plugin("act")
    s = _server(p, model_path="/tmp/c")
    assert _sh(s.argv).startswith(f"cd {repo} && exec uv run --no-sync python ")
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python3").write_text("")
    s2 = _server(plugin("dp"), model_path="/tmp/c")
    assert _sh(s2.argv).startswith(f"cd {repo} && exec {repo}/.venv/bin/python3 ")


def test_dreamzero_server_is_conda_with_pythonpath_and_client_rate(plugin, tree):
    """DreamZero's server is the only one that needs a PYTHONPATH export (it imports
    `groot.*` from the checkout) and the only one that pins the client rate from the
    manifest -- a diffusion forward per call cannot answer at the sim's default rate."""
    p = plugin("dreamzero", paths={"dreamzero_tokenizer": "/data/umt5-xxl"})
    repo = tree["repos"]["dreamzero"]
    s = _server(p, model_path="/tmp/ckpt", prompt="walk to the toilet")
    # `pre` lands BEFORE activation (Runner.wrap), so the export survives whatever the
    # cluster's conda-env.sh does.
    assert _sh(s.argv) == (
        f"export PYTHONPATH={repo}${{PYTHONPATH:+:${{PYTHONPATH}}}} && "
        f"source {tree['conda_sh']} && conda activate dreamzero && "
        'export PATH="$CONDA_PREFIX/bin:$PATH" && '
        f"exec python deployment/run_dreamzero_server.py --model-path /tmp/ckpt "
        f"--port 5550 --master-port 29500 --dreamzero-repo {repo} "
        f"--embodiment-tag unitree_g1_sonic "
        f"--video-frames-per-chunk 8 --initial-prompt 'walk to the toilet' "
        f"--tokenizer-path /data/umt5-xxl")
    assert s.cwd == Path(repo)
    assert s.inference_rate == 2.0


def test_dreamzero_rendezvous_port_follows_the_serving_port(plugin):
    """Two DreamZero servers must be able to run at once.

    The server opens a SECOND port for torch.distributed's rendezvous (GrootSimPolicy
    needs a process group even at world_size=1). Pinned at the default 29500, a second
    server dies with EADDRINUSE before it loads anything -- and the harness reports only
    "policy server did not open :<port>", which points at the wrong thing entirely.
    """
    p = plugin("dreamzero")
    seen = {}
    for port in (5550, 5551, 5591, 5593):
        argv = _sh(_server(p, model_path="/tmp/c", port=port).argv)
        mp = int(argv.split("--master-port ")[1].split()[0])
        assert 20000 <= mp < 30000, mp
        seen[port] = mp
    assert seen[5550] == 29500, "the single-server case must keep the port it always had"
    assert len(set(seen.values())) == len(seen), f"rendezvous ports collide: {seen}"


def test_dreamzero_leaves_video_stride_to_the_checkpoint(plugin):
    """No --video-stride unless the manifest overrides it.

    The correct stride is training's action_horizon // 8, and the horizon lives in the
    CHECKPOINT (24 for runs before 2026-08-26, 48 after). A manifest default would
    mis-serve every checkpoint trained with the other one, silently.
    """
    p = plugin("dreamzero")
    assert "--video-stride" not in _sh(_server(p, model_path="/tmp/c").argv)


def _sonic_prepared(tmp_path):
    """A prepared dir the plugin will read as the SONIC controller lane."""
    import json
    ds = tmp_path / "prepared"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "embodiment.json").write_text('{"embodiment_tag": "unitree_g1_sonic"}')
    (ds / "meta" / "modality.json").write_text(json.dumps(
        {"state": {"joint_pos": {"original_key": "observation.state"}},
         "action": {"motion_token": {"original_key": "action.motion_token"},
                    "left_hand_joints": {"original_key": "teleop.left_hand_joints"},
                    "right_hand_joints": {"original_key": "teleop.right_hand_joints"}},
         "video": {"ego_view": {}}}))
    (ds / "meta" / "stats.json").write_text(json.dumps(
        {k: {} for k in ("observation.state", "action.motion_token",
                         "teleop.left_hand_joints", "teleop.right_hand_joints")}))
    return ds


def test_dreamzero_train_picks_the_backbone_and_puts_set_last(plugin, tree, tmp_path):
    """`--config-name` selects the data + action_head configs and the Wan weight paths,
    and `--set` lands AFTER the manifest defaults so a hydra dotlist override wins."""
    ds = _sonic_prepared(tmp_path)
    p = plugin("dreamzero", paths={"dreamzero_wan22": "/w/wan22", "dreamzero_wan21": "/w/wan21",
                                   "dreamzero_tokenizer": "/w/tok",
                                   "dreamzero_outputs": str(tmp_path / "out")})
    r = p.train(dataset=ds, gpus=1, gpu_ids="3", master_port=29601, task="run1",
                overrides=["max_steps=50"])
    assert r.ok, r.message
    cmd = r.command
    assert "torchrun --nproc_per_node=1 --standalone --master_port=29601" in cmd
    assert "data=dreamzero/unitree_g1_sonic_relative_wan22" in cmd
    assert "model/dreamzero/action_head=wan_flow_matching_action_tf_wan22" in cmd
    assert "num_views=1" in cmd                      # G1 SONIC has ONE camera
    # 48, not upstream's 24: the pretrained checkpoint's action register is built for 48.
    # The dataset must agree, which it does via dataset_kwargs.action_chunk_len.
    # 32 for G1 SONIC, per Psi0's reference g1_sonic_training_full.sh.
    assert "action_horizon=32" in cmd and "num_action_per_block=32" in cmd
    # full FT: the three coupled switches and the deepspeed stage all follow one knob.
    assert "train_architecture=full" in cmd and "save_lora_only=false" in cmd
    assert "deepspeed/zero3.json" in cmd
    assert "dit_version=/w/wan22" in cmd
    assert "vae_pretrained_path=/w/wan22/Wan2.2_VAE.pth" in cmd
    # Wan2.2 ships no CLIP -- the image encoder comes out of the Wan2.1 dir
    assert ("image_encoder_pretrained_path=/w/wan21/"
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth") in cmd
    assert cmd.rstrip("'").endswith("max_steps=50"), cmd     # --set is last
    # wan22 declares no pretrained_path_key: every released DreamZero checkpoint is a 14B
    # model, so pointing the 5B backbone at one would shape-mismatch the whole DiT and
    # train from scratch behind a "loaded pretrained weights" log line.
    assert "pretrained_model_path" not in cmd

    r21 = p.train(dataset=ds, gpus=2, config_name="wan21", task="run2")
    assert "data=dreamzero/unitree_g1_sonic_relative " in r21.command + " "
    assert "model/dreamzero/action_head=wan_flow_matching_action_tf " in r21.command + " "
    assert "dit_version=/w/wan21" in r21.command
    assert "vae_pretrained_path=/w/wan21/Wan2.1_VAE.pth" in r21.command
    # 220, not upstream's 880: frame_seqlen counts tokens per frame after the VAE and the
    # DiT patch embedding, and upstream's 880 assumes the 2x2 grid a 3-view clip is tiled
    # into. G1 SONIC has ONE camera, so there is no tiling and it is (176/8/2)*(320/8/2).
    assert "frame_seqlen=220" in r21.command and "image_resolution_height=176" in r21.command
    assert "pretrained_model_path" not in r21.command        # paths.dreamzero_pretrained unset


def test_dreamzero_pretrained_is_wan21_only(plugin, tmp_path):
    """A released DreamZero checkpoint may only be fed to the 14B backbone."""
    ds = _sonic_prepared(tmp_path)
    p = plugin("dreamzero", paths={"dreamzero_wan22": "/w/wan22", "dreamzero_wan21": "/w/wan21",
                                   "dreamzero_tokenizer": "/w/tok",
                                   "dreamzero_pretrained": "/w/DreamZero-AgiBot",
                                   "dreamzero_outputs": str(tmp_path / "out")})
    r21 = p.train(dataset=ds, gpus=1, config_name="wan21", task="a")
    assert "pretrained_model_path=/w/DreamZero-AgiBot" in r21.command
    assert "++action_head_cfg.config.skip_component_loading=true" in r21.command
    assert "++action_head_cfg.config.defer_lora_injection=false" in r21.command  # full FT
    r22 = p.train(dataset=ds, gpus=1, config_name="wan22", task="b")
    assert "pretrained_model_path" not in r22.command, "5B must not take a 14B checkpoint"


def test_dreamzero_train_refuses_an_unconverted_dataset(plugin, tmp_path):
    """No meta/embodiment.json => the loader cannot pick a modality config at all, and a
    stats.json without the action columns is a bare KeyError minutes into startup."""
    import json
    ds = tmp_path / "raw"
    (ds / "meta").mkdir(parents=True)
    p = plugin("dreamzero", paths={"dreamzero_wan22": "/w/wan22", "dreamzero_wan21": "/w/w21",
                                   "dreamzero_tokenizer": "/w/tok"})
    r = p.train(dataset=ds, gpus=1)
    assert not r.ok and "embodiment.json" in r.message and "lab convert" in r.message

    (ds / "meta" / "embodiment.json").write_text('{"embodiment_tag": "unitree_g1_sonic"}')
    (ds / "meta" / "modality.json").write_text(json.dumps(
        {"state": {"joint_pos": {"original_key": "observation.state"}},
         "action": {"motion_token": {"original_key": "action.motion_token"},
                    "left_hand_joints": {"original_key": "teleop.left_hand_joints"}},
         "video": {"ego_view": {}}}))
    (ds / "meta" / "stats.json").write_text('{"observation.state": {}}')
    r2 = p.train(dataset=ds, gpus=1)
    # The required columns come from the dataset's OWN modality.json, so this message
    # names whatever that lane reads -- not a hardcoded sonic list.
    assert not r2.ok and "teleop.left_hand_joints" in r2.message


# ---- InternVLA-A1.5 -------------------------------------------------------

def _internvla_paths(tmp_path):
    return {"internvla_a15_vlm": "/data/Qwen3.5-2B",
            "internvla_a15_base": "/data/InternVLA-A1.5-base-g1sonic",
            "internvla_a15_data_root": str(tmp_path / "ivla_data"),
            "internvla_a15_outputs": str(tmp_path / "ivla_outs")}


def test_internvla_a15_server_points_the_processor_at_the_local_vlm(plugin, tree, tmp_path):
    """The server owns the Qwen3-VL processor, and the checkpoint's config.json names the
    HUB id "Qwen/Qwen3.5-2B" -- unreachable under HF_HUB_OFFLINE. `--vlm-model-path` is
    what keeps the server startable on the cluster, so it is pinned here.

    PYTHONPATH carries BOTH the repo root (for `evaluation.*`, whose backend does the
    inference) and repo/src (for `lerobot.*`).
    """
    p = plugin("internvla_a15", paths=_internvla_paths(tmp_path))
    repo = tree["repos"]["internvla_a15"]
    s = _server(p, model_path="/tmp/ckpt", prompt="walk over to the toilet")
    assert _sh(s.argv) == (
        f"export PYTHONPATH={repo}:{repo}/src${{PYTHONPATH:+:${{PYTHONPATH}}}} && "
        f"export TOKENIZERS_PARALLELISM=false && "
        f"source {tree['conda_sh']} && conda activate internvla_a1_5 && "
        'export PATH="$CONDA_PREFIX/bin:$PATH" && '
        f"exec python deployment/run_internvla_server.py --ckpt-path /tmp/ckpt "
        f"--port 5550 --robot-type unitree_g1_sonic --resize-size 224 "
        f"--max-prompt-length 650 --inference-backend standard "
        f"--vlm-model-path /data/Qwen3.5-2B --initial-prompt 'walk over to the toilet'")
    assert s.cwd == Path(repo)


def test_internvla_a15_train_pins_the_g1_widths_and_the_silent_traps(plugin, tmp_path):
    """Four flags on this line are load-bearing and all fail SILENTLY if dropped.

    * `--policy.max_action_dim=78` / `--policy.max_state_dim=43`: the released base is
      32/32, so these must match the widened checkpoint or nothing loads.
    * `tokenize_state` on BOTH configs: the policy default is True and the dataset
      default is False, and the disagreement trains a model with no proprioception at
      all while the loss curve looks normal.
    * `--dataset.use_imagenet_stats=false`: True is the default and KeyErrors on locomanilab
      data (no image entry in episodes_stats).
    * `--policy.enable_vqa_loss=false`: True materializes ~650x250k fp32 logits per
      sample for a loss that is identically zero.
    """
    p = plugin("internvla_a15", paths=_internvla_paths(tmp_path))
    res = p.train(dataset=Path("/data/datasets/all31_merged"), gpus=2, gpu_ids="0,1",
                  master_port=29511, task="smoke")
    cmd = res.command
    for flag in ("--policy.max_action_dim=78", "--policy.max_state_dim=43",
                 "--policy.tokenize_state=true", "--dataset.tokenize_state=true",
                 "--dataset.max_state_dim=43", "--dataset.max_action_dim=78",
                 "--dataset.use_imagenet_stats=false",
                 "--policy.enable_vqa_loss=false",
                 "--policy.action_loss_only=true",
                 "--dataset.use_fast_action_tokens=false",
                 "--dataset.action_mode=abs",
                 "--dataset.repo_id=v30/all31_merged",
                 "--policy.pretrained_path=/data/InternVLA-A1.5-base-g1sonic",
                 "--policy.vlm_model_name_or_path=/data/Qwen3.5-2B",
                 "--dataset.vlm_model_name_or_path=/data/Qwen3.5-2B"):
        assert flag in cmd, flag
    # `--dataset.root` must NOT appear: the fork resolves a dataset dir two different
    # ways and only `HF_LEROBOT_HOME` + repo_id satisfies both.
    assert "--dataset.root" not in cmd
    assert f"HF_LEROBOT_HOME={tmp_path / 'ivla_data'}" in cmd
    assert "--main_process_port=29511" in cmd and "--multi_gpu" in cmd


def test_internvla_a15_set_replaces_the_default_instead_of_appending(plugin, tmp_path):
    """`--set steps=2000` must REPLACE `--steps=60000`, not trail after it.

    draccus builds an argparse parser, so a duplicate flag is probably last-wins --
    but a 60k-step run is not the place to find out, and the doubled flag makes the
    logged command unreadable.
    """
    p = plugin("internvla_a15", paths=_internvla_paths(tmp_path))
    cmd = p.train(dataset=Path("/data/ds"), gpus=1, task="t",
                  overrides=["steps=2000", "save_freq=1000",
                             "policy.optimizer_lr=1e-5"]).command
    assert "--steps=2000" in cmd and "--steps=60000" not in cmd
    assert "--save_freq=1000" in cmd and "--save_freq=5000" not in cmd
    assert "--policy.optimizer_lr=1e-5" in cmd and "--policy.optimizer_lr=5e-05" not in cmd


def test_internvla_a15_set_refuses_to_move_an_accelerate_flag(plugin, tmp_path):
    """`--set num_processes=4` would be silently dropped: the flag lives BEFORE the
    training script on the line, and folding it in would move it after, where draccus
    sees it instead of accelerate."""
    p = plugin("internvla_a15", paths=_internvla_paths(tmp_path))
    res = p.train(dataset=Path("/data/ds"), gpus=1, task="t", overrides=["num_processes=4"])
    assert not res.ok and "accelerate launch" in res.message
