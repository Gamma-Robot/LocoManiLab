"""GR00T-Sonic plugin.

  convert -> none (gr00t consumes the shared prepare_dataset output directly)
  train   -> uv run [torchrun] gr00t/experiment/launch_finetune.py ...   (Isaac-GR00T, uv)
  deploy  -> start_inference.sh --model-path --prompt
             (GR00T-WholeBodyControl, .venv_inference; optional TRT)
  eval / infer-check -> gr00t/eval/run_gr00t_server.py (Isaac-GR00T uv venv, :5550)
             + the shared sim / offline client (see ModelPlugin._sim_server)

Base model for finetuning: nvidia/GR00T-N1.7-3B (see docs/training.md).
"""
from __future__ import annotations

import time
import re
from pathlib import Path
from typing import Any

from locomanilab.plugin import ModelPlugin, SimServer, StageResult, register


@register("gr00t")
class GR00TPlugin(ModelPlugin):

    # ---- convert -------------------------------------------------------
    def convert(self, *, dataset: Path, **opts: Any) -> StageResult:
        dataset = self.ctx.input_path(dataset)
        return StageResult(
            ok=True,
            message=("gr00t needs no conversion: it trains directly on the prepare_dataset "
                     "output (LeRobot + meta/modality.json). Pass this dataset to `lab train`."),
            artifacts={"dataset": Path(dataset)},
        )

    # embodiment tag encodes the action space: motion_token vs smpl (reference-motion).
    # explicit --embodiment-tag wins; else pick per --action-space; else the manifest default.
    def _tag(self, action_space: str | None, override: str | None,
             dataset: Path | str | None = None) -> str:
        if override:
            return override
        # A heft/amo dataset can only be trained under its own embodiment, and
        # --action-space (motion_token|smpl) only ever meant sonic's two targets — so
        # when the data says heft/amo, that decides. sonic data keeps the old behaviour.
        if dataset is not None:
            from locomanilab.common import variants as variantmod
            found = variantmod.detect(dataset, self.manifest.raw.get("variants") or {})
            if not isinstance(found, str) and found[0] != "sonic":
                name, spec = found
                tag = spec.get("embodiment_tag")
                if tag:
                    print(f"[locomanilab:gr00t] dataset schema: {name} -> --embodiment-tag {tag}")
                    return tag
        return (self.manifest.action_space(action_space).get("embodiment_tag")
                or self.manifest.train.get("embodiment_tag", "UNITREE_G1_SONIC"))

    # ---- train ---------------------------------------------------------
    def train(self, *, dataset: Path, gpus: int, task: str | None = None,
              base_model: str | None = None, output_dir: str | None = None,
              gpu_ids: str | None = None, master_port: int | None = None,
              per_gpu_batch: int | None = None, action_space: str | None = None,
              overrides: list[str] | None = None, **opts: Any) -> StageResult:
        dataset = self.ctx.input_path(dataset)
        repo = self.repo(self.manifest.repo_key)          # Isaac-GR00T
        env = self.manifest.env("train")
        tcfg = self.manifest.train

        base = str(self.ctx.input_path(base_model)) if base_model else self._base_model_path()
        if base is None:
            return StageResult(ok=False, message=(
                "gr00t train needs the base model. Download nvidia/GR00T-N1.7-3B and set "
                "paths.gr00t_base in env.local.yaml, or pass --base-model. See docs/training.md"))

        extra_env: dict[str, str] = {}
        gpus = self.resolve_gpus(gpus, gpu_ids, extra_env)

        name = task or time.strftime("gr00t_%Y%m%d_%H%M%S")
        output_dir = str(self.ctx.anchor(output_dir) if output_dir else (
            self.ctx.path("gr00t_outputs") or self.ctx.workspace_root / "checkpoints" / "gr00t") / name)

        d = tcfg.get("defaults", {})
        launcher = tcfg.get("launcher", "gr00t/experiment/launch_finetune.py")
        # convention: global batch = per_gpu_batch * num_gpus (override with --set global_batch_size / --per-gpu-batch)
        pgb = int(per_gpu_batch or d.get("per_gpu_batch", 256))
        global_batch = d.get("global_batch_size") or pgb * gpus
        args = [
            launcher,
            "--base-model-path", str(base),
            "--dataset-path", str(dataset),
            "--embodiment-tag", self._tag(action_space, None, dataset),
            "--modality-config-path", tcfg.get("modality_config", "gr00t/configs/data/embodiment_configs.py"),
            "--num-gpus", str(gpus),
            "--output-dir", output_dir,
            "--save-total-limit", str(d.get("save_total_limit", 5)),
            "--save-steps", str(d.get("save_steps", 20000)),
            "--max-steps", str(d.get("max_steps", 100000)),
            "--global-batch-size", str(global_batch),
            "--dataloader-num-workers", str(d.get("dataloader_num_workers", 4)),
        ]
        args += list(tcfg.get("extra_args", []))          # e.g. --color-jitter-params ...
        for kv in (overrides or []):
            # --set max_steps=100000 -> --max-steps 100000 ; --set no_wandb -> --no-wandb
            # (a bare flag used to come out as `--no-wandb ""`, an empty positional the
            # launcher then choked on)
            k, has_value, v = kv.partition("=")
            args.append(f"--{k.replace('_', '-')}")
            if has_value:
                args.append(v)

        # multi-gpu -> torchrun; single-gpu -> python
        if gpus > 1:
            tr = ["torchrun", f"--nproc_per_node={gpus}"]
            if master_port is not None:
                tr.append(f"--master-port={master_port}")
            cmd = [*tr, *args]
        else:
            cmd = ["python", *args]
        return self.runner.run(cmd, cwd=repo, env_kind=env.kind,
                               setup=self.env_setup(env.kind),
                               extra_env=extra_env or None, stage="train")

    def _base_model_path(self) -> str | None:
        p = self.ctx.path("gr00t_base")
        return str(p) if p else None

    def _resolve_trt_engine(self, trt_engine: str | None,
                            model_path: str | None) -> Path | None:
        """TRT engines dir: explicit --trt-engine > env.local > conventional next to model.

        Only returns a dir that actually exists, so a stale/placeholder env.local value
        falls through to the convention instead of being passed to the launcher verbatim.
        The conventional layout for locally exported TensorRT engines is:
            <isaac>/gr00t_model/<name>       -> <isaac>/gr00t_model_trt/<name>/engines

        Always absolute: the launcher runs with cwd=<gr00t_wbc>, so a relative path
        (e.g. derived from a relative --model-path) would resolve against the wrong dir.
        """
        if trt_engine:                      # explicit wins, even if missing (surfaces typos)
            return self.ctx.input_path(trt_engine)
        p = self.ctx.path("gr00t_trt_engine")
        if p and p.is_dir():
            return p.absolute()
        if model_path:                      # convention: sibling *_trt tree next to the model
            mp = self.ctx.input_path(model_path)
            cand = mp.parent.parent / "gr00t_model_trt" / mp.name / "engines"
            if cand.is_dir():
                return cand
        return None

    # ---- deploy --------------------------------------------------------
    def deploy(self, *, prompt: str | None = None,
               model_path: str | None = None, trt: bool = False,
               trt_engine: str | None = None,
               server_only: bool = False, port: int = 5550,
               embodiment_tag: str | None = None, action_space: str | None = None,
               policy_config: str | None = None,
               **opts: Any) -> StageResult:
        if opts.get("fetch") or opts.get("step") is not None:
            return StageResult(ok=False, message=(
                "Checkpoint transfer options were removed. Provide a local checkpoint with --model-path."))
        if model_path is None:
            mp = self.ctx.path("gr00t_model")
            if mp is None:
                return StageResult(ok=False, message=(
                    "gr00t deploy needs --model-path (a local checkpoint directory), "
                    "or set paths.gr00t_model in env.local.yaml"))
            model_path = str(mp)
        # absolute: start_inference.sh runs with cwd=<gr00t_wbc>, so a relative
        # --model-path (or a repo-root-relative env.local value) would break there.
        model_path = str(self.ctx.input_path(model_path))

        # resolve TRT engines dir (explicit > env.local > conventional next to model)
        engine = None
        if trt:
            engine = self._resolve_trt_engine(trt_engine, model_path)
            if engine is None:
                return StageResult(ok=False, message=(
                    "--trt given but no TRT engines dir found. Pass --trt-engine <dir>, "
                    "or set paths.gr00t_trt_engine in env.local.yaml. Build engines locally "
                    "using the model backend's documented TensorRT export. Drop --trt for PyTorch mode."))

        # --- server-only: just the gr00t PolicyServer (:5550), for sim / remote clients ---
        # (no camera/tmux; used by humanoid_benchmark sim eval and any get_action client)
        if server_only:
            if policy_config:
                return StageResult(ok=False, message=(
                    "--sonic-ckpt / --policy-config pick the SONIC controller, which "
                    "--server-only does not start (only the policy server; the sim client "
                    "or `lab eval --sonic-ckpt` chooses the controller there). Drop it."))
            isaac = self.repo("isaac_gr00t")
            # explicit --embodiment-tag wins; else pick per --action-space (smpl -> reference-motion
            # tag, encoder->decoder); else motion_token default. Reuses the same resolver as train.
            tag = self._server_tag(model_path, embodiment_tag, action_space, None)
            cmd = ["python", "gr00t/eval/run_gr00t_server.py",
                   "--model-path", model_path,
                   "--embodiment-tag", tag,
                   "--device", "cuda:0", "--port", str(port)]
            if engine:
                cmd += ["--trt-engine-path", str(engine),
                        "--trt-mode", self.manifest.deploy.get("trt_mode", "n17_full_pipeline")]
            return self.runner.run(cmd, cwd=isaac, env_kind="uv",
                                   setup=self.env_setup("uv"), stage="deploy")

        # --- full real-robot deploy: start_inference.sh (camera + inference loop + tmux) ---
        wbc = self.repo("gr00t_wbc")
        venv = str(wbc / ".venv_inference")
        if prompt is None:
            return StageResult(ok=False, message="gr00t deploy requires --prompt (training task string)")
        extra_env: dict[str, str] = {}
        if engine:
            extra_env["TRT_ENGINE_PATH"] = str(engine)
        # machine-specific roots for start_inference.sh — the WBC repo keeps neutral defaults;
        # Isaac-GR00T comes from repos.*, TRT/CUDA12 runtimes from optional env.local paths.
        extra_env["GROOT_REPO_ROOT"] = str(self.repo("isaac_gr00t"))
        for env_key, path_key in (("GROOT_TRT_ROOT", "gr00t_trt_root"),
                                  ("GROOT_CUDA12_ROOT", "gr00t_cuda12_root")):
            p = self.ctx.path(path_key)
            if p:
                extra_env[env_key] = str(p)
        for k in ("ACTION_HORIZON", "INFERENCE_RATE"):
            if k in opts and opts[k] is not None:
                extra_env[k] = str(opts[k])
        cmd = ["bash", "start_inference.sh", "--model-path", model_path, "--prompt", prompt]
        # Match the SONIC variant used for g1_backend collection: a policy trained
        # on v1_1 data needs the v1_1 decoder that produced its motion_tokens.
        # Without one the launcher default (low_latency on this checkout) -- so
        # low_latency=False keeps the flag list empty rather than forcing a preset.
        sonic_args = self.gr00t_wbc_sonic_args(wbc, policy_config, low_latency=False)
        if isinstance(sonic_args, StageResult):
            return sonic_args
        cmd += sonic_args
        return self.runner.run(cmd, cwd=wbc, env_kind="venv", venv_path=venv,
                               extra_env=extra_env or None, stage="deploy")


    # ---- sim harness (eval + infer-check share this) --------------------
    def _server_tag(self, model_path: str, override: str | None, action_space: str | None,
                    backend: str | None) -> str:
        """The embodiment tag the policy server is started with.

        Order: explicit --embodiment-tag; the tag the CHECKPOINT was trained with
        (experiment_cfg/config.yaml, written by launch_finetune); the controller the
        client is told to drive (--backend heft/amo/almi -> that variant's tag, since
        a heft checkpoint cannot be served under sonic's embodiment); else the
        sonic action-space default. Before this, eval always fell back to
        UNITREE_G1_SONIC unless --embodiment-tag was given -- an amo/almi/heft
        checkpoint then loaded under the wrong action head and failed at the
        first request. `deploy --server-only` resolves the same way, so the real
        robot and the sim start the same server for the same checkpoint.
        """
        if override:
            return override
        cfg = Path(model_path) / "experiment_cfg" / "config.yaml"
        if cfg.is_file():
            m = re.search(r"^\s*embodiment_tag:\s*['\"]?([A-Za-z0-9_]+)", cfg.read_text(), re.M)
            if m:
                tag = m.group(1)
                print(f"[locomanilab:gr00t] checkpoint was trained with --embodiment-tag {tag} ({cfg})")
                return tag
        if backend and backend != "sonic":
            spec = (self.manifest.raw.get("variants") or {}).get(backend) or {}
            tag = spec.get("embodiment_tag")
            if tag:
                print(f"[locomanilab:gr00t] --backend {backend} -> --embodiment-tag {tag}")
                return tag
        return self._tag(action_space, None)

    def _sim_server(self, *, stage: str, prompt: str | None, port: int,
                    model_path: str | None = None, trt: bool = False,
                    trt_engine: str | None = None, embodiment_tag: str | None = None,
                    action_space: str | None = None, **opts: Any) -> SimServer | StageResult:
        """gr00t PolicyServer (Isaac-GR00T uv venv) for the sim / offline client.

        The gr00t server takes no prompt: the client sends it with every request, so
        the default prompt here is the client's. Under --trt the client re-queries at
        the real-robot TRT rate (10 Hz) unless the user set --inference-rate.
        """
        ckpt = self.resolve_ckpt(model_path, env_key="gr00t_model", stage=stage,
                                 hint="--model-path or env.local paths.gr00t_model")
        if isinstance(ckpt, StageResult):
            return ckpt
        model_path = str(ckpt)              # absolute: the server runs with cwd=<isaac>
        engine = self._resolve_trt_engine(trt_engine, model_path) if trt else None
        tag = self._server_tag(model_path, embodiment_tag, action_space, opts.get("backend"))
        isaac = self.repo("isaac_gr00t")
        srv = ["python", "gr00t/eval/run_gr00t_server.py",
               "--model-path", model_path, "--embodiment-tag", tag,
               "--device", "cuda:0", "--port", str(port)]
        if engine:
            srv += ["--trt-engine-path", str(engine),
                    "--trt-mode", self.manifest.deploy.get("trt_mode", "n17_full_pipeline")]
        # sim eval uses the PyTorch server -> match the real-robot PyTorch inference
        # rate (start_inference.sh: 2.5Hz PyTorch / 10Hz TRT). The client default is
        # already 2.5; bump to 10 under --trt.
        return SimServer(argv=self.server_argv(srv, env_kind="uv"), cwd=isaac,
                         prompt=prompt or "pick up the mug and place it on the plate",
                         inference_rate=10.0 if trt else None)
