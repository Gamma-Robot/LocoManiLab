"""FastWAM plugin.

Maps locomanilab's uniform stages onto FastWAM's native scripts:
  convert -> scripts/preprocess_action_dit_backbone.py + scripts/precompute_text_embeds.py
  train   -> scripts/train_zero1.sh <ngpu> task=<task> [key=value ...]   (hydra)
  deploy  -> start_inference_fastwam.sh -p -d -m  (gr00t-compat server)
  eval / infer-check -> deployment/run_fastwam_server.py in the FastWAM repo (conda fastwam)
             + the shared sim / offline client (see ModelPlugin._sim_server)

Everything runs locally in the `fastwam` conda env on whatever machine invokes it.
See docs/data_collection.md and the FastWAM repo's FINETUNE_CUSTOM.md.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from locomanilab.plugin import ModelPlugin, SimServer, StageResult, register


@register("fastwam")
class FastWAMPlugin(ModelPlugin):

    # ---- controller variant --------------------------------------------
    def _variant(self, dataset: Path) -> dict[str, Any]:
        """Per-controller settings (data config + task recipe) for this dataset.

        Detected from the dataset's own meta/modality.json — a heft/amo dataset simply
        cannot be read by the sonic data config (different columns, different widths),
        so this is not a preference the caller should have to remember. Falls back to the
        manifest defaults (sonic) with a warning if the schema is unrecognised.
        """
        from locomanilab.common import variants as variantmod
        found = variantmod.detect(dataset, self.manifest.raw.get("variants") or {})
        if isinstance(found, str):
            print(f"[locomanilab:fastwam] {found}\n"
                  f"[locomanilab:fastwam] falling back to the manifest defaults (sonic layout)")
            return {}
        name, spec = found
        print(f"[locomanilab:fastwam] dataset schema: {name} -> data={spec.get('data_config_name')} "
              f"task={spec.get('task')}")
        return spec

    # ---- convert -------------------------------------------------------
    def convert(self, *, dataset: Path, **opts: Any) -> StageResult:
        dataset = self.ctx.input_path(dataset)
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("convert")
        variant = self._variant(dataset)
        data_cfg = variant.get("data_config_name") or self.manifest.convert.get(
            "data_config_name", "custom_1cam")
        model_cfg = self.manifest.convert.get("model_config", "configs/model/fastwam.yaml")
        backbone_out = self.manifest.convert.get(
            "backbone_out",
            "checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt")
        # DIFFSYNTH looks here for base weights (auto-downloaded on first run).
        extra_env = {"DIFFSYNTH_MODEL_BASE_PATH": str(repo / "checkpoints")}

        # Step A: build ActionDiT backbone from Wan2.2 (idempotent: skip if exists).
        if not (repo / backbone_out).exists():
            r = self.runner.run(
                ["python", "scripts/preprocess_action_dit_backbone.py",
                 "--model-config", model_cfg, "--output", backbone_out,
                 "--device", "cuda", "--dtype", "bfloat16"],
                cwd=repo, env_kind=env.kind, env_name=env.name, setup=self.env_setup(env.kind),
                extra_env=extra_env, stage="convert")
            if not r.ok:
                return r

        # Step B: precompute T5 text embeds for this dataset's tasks.jsonl.
        # Override BOTH dataset_dirs and text_embedding_cache_dir: the data config's
        # cache dir is hardcoded to an unrelated dataset, so without this the cache
        # lands outside our dataset (fragile / not portable). Keep it inside the
        # dataset (<dataset>/cache_t5) so data + its T5 cache travel together, and so
        # convert and train agree on the location (train sets the same override).
        cache_dir = f"{str(dataset).rstrip('/')}/cache_t5"
        return self.runner.run(
            ["python", "scripts/precompute_text_embeds.py",
             "--config-name", "train", f"data={data_cfg}", "model=fastwam",
             f"data.train.dataset_dirs=[{dataset}]",
             f"data.train.text_embedding_cache_dir={cache_dir}"],
            cwd=repo, env_kind=env.kind, env_name=env.name, setup=self.env_setup(env.kind),
            extra_env=extra_env, stage="convert")

    # ---- train ---------------------------------------------------------
    def train(self, *, dataset: Path, gpus: int, task: str | None = None,
              gpu_ids: str | None = None, master_port: int | None = None,
              action_space: str | None = None, output_dir: str | None = None,
              overrides: list[str] | None = None, **opts: Any) -> StageResult:
        dataset = self.ctx.input_path(dataset)
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("train")
        variant = self._variant(dataset)
        task = task or variant.get("task") or self.manifest.train.get(
            "default_task", "custom_1cam_1e-4")
        # honor --gpu-ids: train_zero1.sh -> accelerate starts from cuda:0 by default,
        # which on a shared machine collides with other trainings -> pin visible devices
        # and set NPROC (train_zero1.sh $1) to the id count.
        extra_env: dict[str, str] = {}
        gpus = self.resolve_gpus(gpus, gpu_ids, extra_env)
        # MASTER_PORT default (29500) can clash with another concurrent torchrun;
        # set a distinct port for parallel runs.
        if master_port is not None:
            extra_env["MASTER_PORT"] = str(master_port)
        self.drop_expandable_segments(extra_env, "fastwam")
        # point DiffSynth at repo/checkpoints (the shared base: Wan2.2 components + backbone)
        # so training reuses the locally provisioned base model.
        extra_env["DIFFSYNTH_MODEL_BASE_PATH"] = str(repo / "checkpoints")
        # inject --dataset as a hydra override; user --set overrides can still follow.
        # text_embedding_cache_dir must match what convert wrote (<dataset>/cache_t5),
        # else training reads T5 embeds from the config's hardcoded (unrelated) dir.
        cache_dir = f"{str(dataset).rstrip('/')}/cache_t5"
        output = self.ctx.anchor(output_dir) if output_dir else (
            self.ctx.path("fastwam_outputs") or self.ctx.workspace_root / "checkpoints" / "fastwam") / time.strftime("%Y%m%d_%H%M%S")
        cmd = ["bash", "scripts/train_zero1.sh", str(gpus), f"task={task}",
               f"output_dir={output}",
               f"data.train.dataset_dirs=[{dataset}]",
               f"data.train.text_embedding_cache_dir={cache_dir}"]
        # --action-space: append the variant's hydra overrides (target action key + output dim)
        cmd += list(self.manifest.action_space(action_space).get("overrides", []))
        for kv in (overrides or []):                       # hydra key=value passthrough
            cmd.append(kv)
        return self.runner.run(cmd, cwd=repo, env_kind=env.kind, env_name=env.name, setup=self.env_setup(env.kind),
                               extra_env=extra_env or None, stage="train")

    # ---- deploy --------------------------------------------------------
    def _t5_cache_dir(self, t5_mode: str, run_dir: str) -> tuple[str | None, StageResult | None]:
        """Resolve --t5-mode into a server cache dir. encode -> (None, None);
        cache -> (<run_dir>/cache_t5, None) or (None, error) if it is missing."""
        if t5_mode not in ("encode", "cache"):
            return None, StageResult(ok=False, message=f"--t5-mode must be encode|cache, got {t5_mode!r}")
        if t5_mode == "encode":
            return None, None
        cache = Path(run_dir).expanduser() / "cache_t5"
        if not cache.is_dir() and not self.runner.dry_run:
            return None, StageResult(ok=False, message=(
                f"--t5-mode cache: {cache} not found.\n"
                "Place the matching text embedding cache alongside the local checkpoint, "
                "or use --t5-mode encode."))
        return str(cache), None

    def _ckpt_for_step(self, run_dir: str, ckpt_step: int | None) -> tuple[str | None, StageResult | None]:
        """Resolve --ckpt-step into an absolute step_*.pt path under <run_dir>/checkpoints/weights.
        None -> (None, None): the server picks the newest checkpoint itself. The server resolves
        a relative --checkpoint against its own cwd (the wbc repo), so always hand it an
        absolute path."""
        if ckpt_step is None:
            return None, None
        weights = Path(run_dir).expanduser() / "checkpoints" / "weights"

        def _step(p: Path) -> int:          # numeric, so step_10 lists after step_9
            try:
                return int(p.stem.split("_")[1])
            except (IndexError, ValueError):
                return -1
        pts = sorted(weights.glob("step_*.pt"), key=_step)
        for p in pts:
            try:
                if int(p.stem.split("_")[1]) == ckpt_step:
                    return str(p.resolve()), None
            except (IndexError, ValueError):
                continue
        if self.runner.dry_run:                       # weights may not exist yet in a dry run
            return str(weights / f"step_{ckpt_step:06d}.pt"), None
        have = ", ".join(p.name for p in pts) or "none"
        return None, StageResult(ok=False, message=(
            f"--ckpt-step {ckpt_step}: no matching step_*.pt in {weights} (have: {have})"))

    def deploy(self, *, prompt: str | None = None,
               run_dir: str | None = None, base_model: str | None = None,
               action_space: str | None = None,
               t5_mode: str | None = None, ckpt_step: int | None = None,
               policy_config: str | None = None,
               **opts: Any) -> StageResult:
        if opts.get("fetch") or opts.get("step") is not None:
            return StageResult(ok=False, message=(
                "Checkpoint transfer options were removed. Provide a local checkpoint with --run-dir."))
        t5_mode = t5_mode or "encode"
        action_space = action_space or self.manifest.default_action_space
        self.manifest.action_space(action_space)  # validate the requested variant exists

        wbc = self.repo("gr00t_wbc")                       # start_inference_fastwam.sh lives here
        env = self.manifest.env("deploy")
        prompt = prompt or self.manifest.deploy.get("prompt_fixed")
        if prompt is None:
            return StageResult(ok=False, message="deploy requires --prompt (or deploy.prompt_fixed)")

        # run_dir: a local FastWAM run directory (already available on this machine).
        if run_dir is None:
            rd = self.ctx.path("fastwam_runs")
            if rd is None:
                return StageResult(
                    ok=False,
                    message="deploy needs --run-dir, or set paths.fastwam_runs in env.local.yaml")
            run_dir = str(rd)

        # base model (Wan2.2-TI2V-5B) — from the local FastWAM repo by default.
        if base_model is None:
            fastwam_repo = self.repo(self.manifest.repo_key)
            base_rel = self.manifest.deploy.get("base_model_rel", "Wan2.2-TI2V-5B")
            base_model = str(fastwam_repo / base_rel)

        run_dir = str(self.ctx.input_path(run_dir))
        base_model = str(self.ctx.input_path(base_model))
        t5_cache, err = self._t5_cache_dir(t5_mode, run_dir)
        if err is not None:
            return err
        ckpt, err = self._ckpt_for_step(run_dir, ckpt_step) 
        if err is not None:
            return err

        extra_env = {k: str(v) for k, v in self.manifest.deploy.get("env_overrides", {}).items()}
        cmd = ["bash", "start_inference_fastwam.sh",
               "-p", prompt, "-d", run_dir, "-m", base_model,
               "--action-space", action_space,
               "-r", str(self.repo(self.manifest.repo_key))]
        if ckpt:
            cmd += ["-c", ckpt]                                # launcher -> server --checkpoint
        if t5_cache:
            cmd += ["--t5-cache", t5_cache]                    # launcher then skips --evict-t5
        # Match the robot's SONIC controller to the variant used for g1_backend data. The
        # launcher's own default is the low_latency preset, so only a non-default
        # variant needs forwarding; `--low-latency` is what it does already.
        sonic_args = self.gr00t_wbc_sonic_args(wbc, policy_config, low_latency=False)
        if isinstance(sonic_args, StageResult):
            return sonic_args
        cmd += [a for a in sonic_args if a != "--low-latency"]
        return self.runner.run(cmd, cwd=wbc, env_kind=env.kind, env_name=env.name, setup=self.env_setup(env.kind),
                               extra_env=extra_env, stage="deploy")


    # ---- sim harness (eval + infer-check share this) --------------------
    def _sim_server(self, *, stage: str, prompt: str | None, port: int,
                    run_dir: str | None = None, model_path: str | None = None,
                    base_model: str | None = None, t5_mode: str | None = None,
                    ckpt_step: int | None = None, **opts: Any) -> SimServer | StageResult:
        """fastwam gr00t-compat server for the sim / offline client.

        fastwam identifies a checkpoint by its RUN DIR. --model-path (the flag every
        other model uses) is accepted as an alias rather than silently ignored:
        doing so once made a whole eval run score the env.local default checkpoint
        instead of the one on the command line -- 0/20 vs the expected 11/20, with
        nothing in the logs to suggest the wrong model was loaded.

        The server lives in the FastWAM repo (it imports fastwam.* and nothing else),
        so it runs there; only the real-robot launcher (cameras, tmux, the inference
        main loop) stays in gr00t_wbc.
        """
        rd = self.resolve_ckpt(run_dir, model_path, env_key="fastwam_runs", stage=stage,
                               hint="--run-dir or env.local paths.fastwam_runs")
        if isinstance(rd, StageResult):
            return rd
        run_dir = str(rd)                   # absolute: the server runs with cwd=<FastWAM>
        if base_model is None:
            base_model = str(self.repo(self.manifest.repo_key)
                             / self.manifest.deploy.get("base_model_rel", "Wan2.2-TI2V-5B"))
        base_model = str(self.ctx.input_path(base_model))
        prompt = prompt or self.manifest.deploy.get("prompt_fixed")
        fw = self.repo(self.manifest.repo_key)
        server = self.manifest.deploy.get("server", "deployment/run_fastwam_server.py")
        t5_cache, err = self._t5_cache_dir(t5_mode or "encode", run_dir)
        if err is not None:
            return err
        ckpt, err = self._ckpt_for_step(run_dir, ckpt_step)
        if err is not None:
            return err
        srv = ["python", server, "--run-dir", run_dir, "--base-model-path", base_model,
               "--port", str(port)]
        if ckpt:
            srv += ["--checkpoint", ckpt]
        srv += ["--t5-cache", t5_cache] if t5_cache else ["--evict-t5"]
        if prompt:
            srv += ["--initial-prompt", prompt]
        # sim eval uses the PyTorch fastwam server -> the client default (2.5 Hz) already
        # matches the real-robot PyTorch inference rate (start_inference_fastwam.sh).
        return SimServer(argv=self.server_argv(srv), cwd=fw, prompt=prompt)
