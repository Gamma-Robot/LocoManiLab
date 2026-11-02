"""starVLA plugin.

Maps locomanilab's uniform stages onto starVLA's native scripts:
  convert -> validate the GR00T LeRobot dataset against the registered DataConfig (no rewrite)
  train   -> accelerate launch starVLA/training/train_starvla.py --config_yaml ... (OmegaConf overrides)
  deploy  -> python -m deployment.model_server.run_starvla_server (gr00t-compat :5550/get_action,
             the default) | --native: deployment/model_server/server_policy.py (starVLA's own
             websocket server, :10093, for the LIBERO / SimplerEnv clients)
  eval / infer-check -> the gr00t-compat server + the shared sim / offline client

starVLA is a general Qwen-VL VLA codebase (LIBERO / RoboTwin / SimplerEnv / OXE). It reads the
GR00T LeRobot dataset directly through its gr00t_lerobot dataloader, so collect/prepare are not
its business and convert only validates.

Everything runs locally in the `starVLA` conda env on whatever machine invokes it.
See the starVLA repo README and examples/LIBERO for the underlying scripts.
"""
from __future__ import annotations

import os

import time
from pathlib import Path
from typing import Any

from locomanilab.plugin import ModelPlugin, SimServer, StageResult, register


@register("starvla")
class StarVLAPlugin(ModelPlugin):

    # ---- convert -------------------------------------------------------
    def _variant(self, dataset: Path) -> dict[str, Any]:
        """Per-controller settings (data mix, robot_type, action/state dims).

        starVLA reads the GR00T LeRobot dataset directly, so the controller decides which
        registered DataConfig can read it at all — detect it from the dataset's own
        meta/modality.json rather than asking the caller to remember a flag.
        """
        from locomanilab.common import variants as variantmod
        found = variantmod.detect(dataset, self.manifest.raw.get("variants") or {})
        if isinstance(found, str):
            print(f"[locomanilab:starvla] {found}\n"
                  f"[locomanilab:starvla] falling back to the manifest defaults (sonic layout)")
            return {}
        name, spec = found
        print(f"[locomanilab:starvla] dataset schema: {name} -> data_mix={spec.get('data_mix')} "
              f"action_dim={spec.get('action_dim')} state_dim={spec.get('state_dim')}")
        return spec

    def convert(self, *, dataset: Path, **opts: Any) -> StageResult:
        """No dataset rewrite: starVLA reads GR00T LeRobot data directly via its
        gr00t_lerobot dataloader. This validates that the dataset exposes the keys the
        registered UnitreeG1SonicDataConfig / sonic_g1 mixture expects, so training does
        not fail late, and prints the ready-to-run train + dataloader smoke-test commands.
        """
        dataset = self.ctx.input_path(dataset)
        from locomanilab.common import variants as variantmod

        cv = self.manifest.convert
        variant = self._variant(dataset)
        data_mix = variant.get("data_mix") or cv.get("data_mix", "sonic_g1")
        robot_type = variant.get("robot_type") or cv.get("robot_type", "unitree_g1_sonic")
        require = variant.get("require_modality") or cv.get("require_modality", {})

        problem = variantmod.check_modality(dataset, require, what=f"starVLA '{robot_type}' config")
        if problem:
            return StageResult(ok=False, message=problem)

        repo = self.repo(self.manifest.repo_key)
        smoke = (f"python -c \"from omegaconf import OmegaConf; "
                 f"from starVLA.dataloader.lerobot_datasets import get_vla_dataset; "
                 f"c=OmegaConf.create({{'data_root_dir':'{dataset}','data_mix':'{data_mix}',"
                 f"'video_backend':'torchvision_av'}}); d=get_vla_dataset(data_cfg=c); "
                 f"print('samples=',len(d)); print(type(d[0]))\"")
        return StageResult(
            ok=True,
            message=(f"OK — {dataset} matches starVLA data_mix='{data_mix}' "
                     f"(robot_type={robot_type}, no rewrite needed).\n"
                     f"Train:\n  lab train --model starvla --dataset {dataset} --gpus N\n"
                     f"Optional dataloader smoke-test (in the starVLA conda env, cwd={repo}):\n"
                     f"  {smoke}"))

    # ---- train ---------------------------------------------------------
    def train(self, *, dataset: Path, gpus: int, gpu_ids: str | None = None,
              master_port: int | None = None, task: str | None = None,
              base_model: str | None = None, output_dir: str | None = None,
              overrides: list[str] | None = None, **opts: Any) -> StageResult:
        dataset = self.ctx.input_path(dataset)
        repo = self.repo(self.manifest.repo_key)          # e.g. external/starVLA
        env = self.manifest.env("train")
        tr = self.manifest.train

        variant = self._variant(dataset)
        framework = self.manifest.raw.get("default_framework", "QwenGR00T")
        base_vlm = str(self.ctx.input_path(base_model)) if base_model else tr.get("base_vlm", "StarVLA/Qwen3-VL-4B-Instruct-Action")
        data_mix = variant.get("data_mix") or tr.get("default_data_mix", "bridge_rt_1")
        config_yaml = tr.get("config_yaml", "starVLA/config/training/starvla_cotrain_oxe.yaml")
        accel_cfg = tr.get("accelerate_config", "starVLA/config/deepseeds/deepspeed_zero2.yaml")
        entrypoint = tr.get("entrypoint", "starVLA/training/train_starvla.py")
        run_root = str(self.ctx.anchor(output_dir) if output_dir else (
            self.ctx.path("starvla_outputs") or self.ctx.workspace_root / "checkpoints" / "starvla"))
        # Per-run id so repeat runs don't overwrite each other's checkpoints.
        # --task pins an explicit, reusable run_id.
        run_id = task or ("locomanilab_starvla_" + time.strftime("%Y%m%d_%H%M%S"))

        # honor --gpu-ids: accelerate starts from cuda:0 by default; on a shared machine that
        # collides with other trainings -> pin visible devices and match num_processes to the count.
        extra_env: dict[str, str] = {}
        gpus = self.resolve_gpus(gpus, gpu_ids, extra_env)
        # WANDB offline by default: the shipped config's wandb_entity is a placeholder that would
        # crash on init; offline logs locally under the run dir. Set WANDB_MODE=online to override.
        if tr.get("wandb_offline", True):
            extra_env.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))
        # The base VLM is already in the local HF cache, but transformers still probes the
        # Hub for it. On a node whose egress goes through the unavailable network proxy that probe
        # hangs in DNS — the run sits at "VLA Training :: Warming Up" forever, no CPU, no IO,
        # no error (observed 2026-08-07: 12 min in, 22 s of CPU). Offline is also the right
        # default: a training that silently re-downloads its backbone can change under you.
        extra_env.setdefault("HF_HUB_OFFLINE", os.environ.get("HF_HUB_OFFLINE", "1"))
        extra_env.setdefault("TRANSFORMERS_OFFLINE", os.environ.get("TRANSFORMERS_OFFLINE", "1"))
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments corrupts training to NaN here
        # (deepspeed/bf16 interaction). The note alone was not enough — an export in the
        # launching shell still poisoned runs on 2026-08-07 — so strip it from the child.
        self.drop_expandable_segments(extra_env, "starvla")

        cmd = ["accelerate", "launch",
               "--config_file", accel_cfg,
               "--num_processes", str(gpus)]
        # distinct rendezvous port for parallel runs (default clashes with other torchrun/accelerate).
        if master_port is not None:
            cmd += ["--main_process_port", str(master_port)]
        cmd += [entrypoint,
                "--config_yaml", config_yaml,
                "--framework.name", framework,
                "--framework.qwenvl.base_vlm", base_vlm,
                "--datasets.vla_data.data_root_dir", str(dataset),
                "--datasets.vla_data.data_mix", data_mix,
                "--run_root_dir", run_root,
                "--run_id", run_id]
        # G1 embodiment action/state dims for the action head (config default is 7-DOF OXE):
        # action = motion_token(64)+hands(7+7)=78 (deploy parity with gr00t/fastwam),
        # state = 43-D G1 proprioception. Appended before user --set so `--set` can still override.
        adim = variant.get("action_dim", tr.get("action_dim"))
        sdim = variant.get("state_dim", tr.get("state_dim"))
        if adim is not None:
            cmd.append(f"--framework.action_model.action_dim={adim}")
        if sdim is not None:
            cmd.append(f"--framework.action_model.state_dim={sdim}")
        # --set k=v passthrough: starVLA reads dotted OmegaConf overrides as `--k=v`
        # (normalize_dotlist_args); later duplicates win, so user --set overrides the above.
        for kv in (overrides or []):
            cmd.append(f"--{kv}" if not kv.startswith("--") else kv)

        return self.runner.run(cmd, cwd=repo, env_kind=env.kind, env_name=env.name,
                               setup=self.env_setup(env.kind),
                               extra_env=extra_env or None, stage="train")

    # ---- deploy --------------------------------------------------------
    def deploy(self, *, model_path: str | None = None, run_dir: str | None = None,
               prompt: str | None = None, port: int = 5550,
               native: bool = False, **opts: Any) -> StageResult:
        """Start a gr00t-compatible policy server (tcp://0.0.0.0:<port>/get_action) for the
        given starVLA checkpoint, so the real G1 robot loop (gear_sonic run_vla_inference.py)
        and the tele sim drive it exactly like gr00t/fastwam. The framework
        (QwenGR00T/QwenPI/QwenOFT) is auto-detected from the checkpoint config.

        Checkpoint = a single .pt file (steps_<N>_pytorch_model.pt). Point at it with
        --model-path (preferred) / --run-dir, or set paths.starvla_ckpt in env.local.yaml.
        Pass native=True to instead start starVLA's OWN websocket server (:10093), used by
        the LIBERO / SimplerEnv eval clients.
        """
        ckpt = self.resolve_ckpt(model_path, run_dir, env_key="starvla_ckpt", stage="deploy",
                                 hint=self.CKPT_HINT)
        if isinstance(ckpt, StageResult):
            return ckpt
        ckpt = str(ckpt)

        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("deploy")
        dep = self.manifest.deploy
        extra_env = {"PYTHONPATH": str(repo)}

        if native:  # starVLA websocket server (:10093) — for LIBERO/SimplerEnv clients
            server = dep.get("native_server", "deployment/model_server/server_policy.py")
            srv_port = port if port != 5550 else int(dep.get("native_port", 10093))
            cmd = ["python", server, "--ckpt_path", str(ckpt), "--port", str(srv_port)]
            if dep.get("use_bf16", True):
                cmd.append("--use_bf16")
        else:       # gr00t-compat server (:5550 / get_action) — real robot + tele sim
            cmd = self._gr00t_server_cmd(ckpt, port, prompt)

        return self.runner.run(cmd, cwd=repo, env_kind=env.kind, env_name=env.name,
                               setup=self.env_setup(env.kind),
                               extra_env=extra_env, stage="deploy")

    # ---- sim harness (eval + infer-check share this) --------------------
    CKPT_HINT = "--model-path (checkpoint .pt), --run-dir, or paths.starvla_ckpt in env.local.yaml"

    def _sim_server(self, *, stage: str, prompt: str | None, port: int,
                    model_path: str | None = None, run_dir: str | None = None,
                    **opts: Any) -> SimServer | StageResult:
        """starVLA gr00t-compat server (conda starVLA, repo on PYTHONPATH). starVLA outputs
        latent motion_token, same as fastwam."""
        ckpt = self.resolve_ckpt(model_path, run_dir, env_key="starvla_ckpt", stage=stage,
                                 hint=self.CKPT_HINT)
        if isinstance(ckpt, StageResult):
            return ckpt
        prompt = prompt or self.manifest.deploy.get("prompt_fixed")
        return SimServer(argv=self._server_argv(str(ckpt), port, prompt),
                         cwd=self.repo(self.manifest.repo_key), prompt=prompt)

    # ---- deploy helpers ------------------------------------------------
    def _gr00t_server_cmd(self, ckpt: str, port: int, prompt: str | None) -> list[str]:
        """`python -m <module> --ckpt-path .. --port ..` for the gr00t-compat server."""
        dep = self.manifest.deploy
        module = dep.get("module", "deployment.model_server.run_starvla_server")
        cmd = ["python", "-m", module, "--ckpt-path", str(ckpt), "--port", str(port)]
        prompt = prompt or dep.get("prompt_fixed")
        if prompt:
            cmd += ["--initial-prompt", str(prompt)]
        if dep.get("use_bf16", True):
            cmd.append("--use-bf16")
        if dep.get("state_mode"):
            cmd += ["--state-mode", str(dep["state_mode"])]
        return cmd

    def _server_argv(self, ckpt: str, port: int, prompt: str | None) -> list[str]:
        """Env-wrapped gr00t-compat server for the sim harness (Popen'd directly, not via
        runner.run): the repo goes on PYTHONPATH the way deploy's extra_env puts it."""
        import shlex
        repo = self.repo(self.manifest.repo_key)
        pre = [f"export PYTHONPATH={shlex.quote(str(repo))}${{PYTHONPATH:+:${{PYTHONPATH}}}}"]
        return self.server_argv(self._gr00t_server_cmd(ckpt, port, prompt), pre=pre)
