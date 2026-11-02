"""DreamZero plugin.

DreamZero (github.com/dreamzero0/dreamzero, NVIDIA GEAR Lab, Apache-2.0) = a Wan video
diffusion DiT that denoises the next video block and the next action chunk jointly. No
VLM backbone (`backbone=identity`); language goes through Wan's umt5-xxl text encoder and
the first frame through Wan's CLIP encoder, both inside the action head.

Maps locomanilab's stages onto DreamZero's native entrypoints:
  convert -> a PREPARED dataset dir (symlinked data/videos + a real meta/) + upstream's
             scripts/data/convert_lerobot_to_gear.py --keep-modality
  train   -> torchrun groot/vla/experiment/experiment.py <hydra dotlist>  (DeepSpeed ZeRO-2)
  deploy  -> deployment/run_dreamzero_server.py       (gr00t-compat :5550/get_action)
  eval / infer-check -> that same server + the shared sim / offline client

Two backbones, chosen with `--config-name wan21|wan22` (see `backbones:` in the manifest);
the choice is baked into the checkpoint, so only `train` takes the flag.

Never imports DreamZero code — pure shell-out. The one thing this plugin does on the
filesystem itself is build convert's prepared dir, because the alternative (upstream's
`--output-path`, a full `shutil.copytree`) would duplicate every video.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from locomanilab.plugin import ModelPlugin, SimServer, StageResult, register


@register("dreamzero")
class DreamZeroPlugin(ModelPlugin):

    # ---- controller variant --------------------------------------------
    def _variant(self, dataset: Path) -> dict[str, Any]:
        """Which controller produced this dataset, from its own meta/modality.json.

        sonic emits a 64-D latent `motion_token`; amo emits a 36-D explicit joint +
        locomotion command. They need different embodiment tags, data configs and action
        widths, and a dataset cannot be read by the other lane's config -- so this is not
        something the caller should have to remember.
        """
        from locomanilab.common import variants as variantmod
        found = variantmod.detect(dataset, self.manifest.raw.get("variants") or {})
        if isinstance(found, str):
            print(f"[locomanilab:dreamzero] {found}\n"
                  f"[locomanilab:dreamzero] falling back to the sonic lane")
            return dict((self.manifest.raw.get("variants") or {}).get("sonic") or {})
        name, spec = found
        print(f"[locomanilab:dreamzero] controller lane: {name} -> tag {spec.get('embodiment_tag')}, "
              f"{spec.get('data_config_base')}, action {spec.get('action_dim')}-D")
        return dict(spec or {})

    # ---- backbone ------------------------------------------------------
    def _backbone(self, name: str | None) -> tuple[str, dict[str, Any]]:
        """(name, spec) for the requested backbone, or the manifest default."""
        table = self.manifest.raw.get("backbones") or {}
        name = name or self.manifest.raw.get("default_backbone") or "wan22"
        if name not in table:
            raise KeyError(f"dreamzero: unknown --config-name {name!r}; "
                           f"known backbones: {sorted(table)}")
        return name, dict(table[name] or {})

    def _weight(self, spec: dict, path_key: str, filename: str | None = None) -> Path | None:
        """A backbone weight path from env.local `paths.<key>` (+ an optional filename)."""
        root = self.ctx.path(spec.get(path_key, ""))
        if root is None:
            return None
        return root / filename if filename else root

    # ---- convert -------------------------------------------------------
    def convert(self, *, dataset: Path, output_dir: str | None = None,
                **opts: Any) -> StageResult:
        """Build the GEAR metadata DreamZero needs, WITHOUT touching the source dataset.

        Our datasets are shared with gr00t / PRTS / pi05 and their `meta/stats.json` only
        carries a few columns, while DreamZero indexes it by original key
        (`le_statistics[original_key]`) for every state and action group — a missing column
        is a KeyError at step 0, and letting DreamZero recompute it in place would rewrite a
        file the other models read. So convert produces a PREPARED dir instead:

            <paths.dreamzero_data_root>/<dataset>/
                data/   -> symlink to the source dataset's data/
                videos/ -> symlink to the source dataset's videos/
                meta/      a real copy, with embodiment.json + a complete stats.json added

        Train and infer-check should then be pointed at the prepared dir.
        """
        dataset = self.ctx.input_path(dataset)
        from locomanilab.common import variants as variantmod

        cv = self.manifest.convert
        problem = variantmod.check_modality(dataset, cv.get("require_modality", {}),
                                            what="DreamZero (unitree_g1_sonic)")
        if problem:
            return StageResult(ok=False, message=problem)

        root_key = cv.get("data_root_key", "dreamzero_data_root")
        root = self.ctx.path(root_key)
        if root is None:
            return StageResult(ok=False, message=(
                f"set paths.{root_key} in env.local.yaml — where DreamZero's prepared "
                f"dataset dirs (symlinked data/videos + a real meta/) go"))
        out = self.ctx.anchor(output_dir) if output_dir else root / dataset.name

        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("convert")
        variant = self._variant(dataset)
        cmd = ["python", cv.get("converter", "scripts/data/convert_lerobot_to_gear.py"),
               "--dataset-path", str(out),
               "--embodiment-tag", variant.get("embodiment_tag")
               or cv.get("embodiment_tag", "unitree_g1_sonic"),
               "--keep-modality", "--keep-tasks", "--force"]
        stats_cols = cv.get("stats_columns") or []
        if stats_cols:
            cmd += ["--stats-columns", *stats_cols]

        if self.runner.dry_run:
            print(f"[locomanilab:dreamzero] would prepare {out} "
                  f"(data/ + videos/ symlinked from {dataset}, meta/ copied)")
        else:
            err = self._prepare_dir(dataset, out)
            if err:
                return StageResult(ok=False, message=err)

        r = self.runner.run(cmd, cwd=repo, env_kind=env.kind, env_name=env.name,
                            setup=self.env_setup(env.kind), stage="convert")
        if r.ok:
            r.artifacts["prepared"] = out
            r.message = (f"prepared DreamZero dataset at {out}\n"
                         f"  data/ and videos/ are symlinks — the source dataset was not "
                         f"modified.\n"
                         f"  train / infer-check with --dataset {out}")
        return r

    @staticmethod
    def _prepare_dir(src: Path, out: Path) -> str | None:
        """Create/refresh the prepared dir. Returns an error message, or None."""
        for sub in ("data", "videos", "meta"):
            if not (src / sub).exists():
                return f"{src} has no {sub}/ — not a locomanilab-prepared LeRobot dataset?"
        out.mkdir(parents=True, exist_ok=True)
        for sub in ("data", "videos"):
            link = out / sub
            if link.is_symlink():
                have = Path(os.readlink(link))
                if have == (src / sub):
                    continue
                # `lab convert` resolves --dataset through its symlinks, so the prepared
                # dir is named after the REAL directory. Two datasets can share that name
                # (bench_dataset/sonic/atom_task/X and .../atom_task_fixed/X both end in X),
                # and silently repointing would leave a meta/ describing the other dataset.
                return (f"{out} is already prepared from a DIFFERENT dataset:\n"
                        f"  {link} -> {have}\n  wanted -> {src / sub}\n"
                        f"Pass --output-dir to convert this one somewhere else, or remove "
                        f"{out} if it is stale.")
            elif link.exists():
                return (f"{link} exists and is not a symlink — refusing to replace it. "
                        f"Remove it, or pass --output-dir to convert somewhere else.")
            link.symlink_to(src / sub)
        # meta/ is a real copy: this is the only part convert rewrites, and the whole point
        # is that the rewrite does not land on the shared dataset.
        meta = out / "meta"
        if meta.is_symlink():
            meta.unlink()
        meta.mkdir(exist_ok=True)
        for f in sorted((src / "meta").iterdir()):
            if f.is_file():
                shutil.copy2(f, meta / f.name)
        return DreamZeroPlugin._fill_blob_original_keys(meta)

    #: Blob columns a null `original_key` refers to, per modality section. Every
    #: controller except sonic packs its groups into a couple of concatenated LeRobot
    #: columns and declares the byte ranges in modality.json (see
    #: `locomanilab/runtime/sim_client.py:_action_groups`).
    _BLOB_COLUMN = {"state": "observation.state", "action": "action"}

    @staticmethod
    def _fill_blob_original_keys(meta: Path) -> str | None:
        """Give every state/action group an explicit `original_key`.

        A blob-schema dataset (amo / heft / almi ...) leaves `original_key` null for the
        groups packed into `observation.state` and `action`. DreamZero has no notion of
        that: `_get_metadata` does a bare `le_statistics[original_key]`, so a null key is
        `KeyError: None` several minutes into startup. Filling them in the PREPARED copy
        keeps the shared dataset untouched and leaves modality.json otherwise verbatim.

        Verified against the widths rather than assumed: the groups' index ranges must
        tile the blob column exactly, and this refuses rather than guess if they do not.
        """
        import json
        mj = meta / "modality.json"
        info = meta / "info.json"
        if not mj.is_file():
            return f"{mj} not found"
        mod = json.loads(mj.read_text())
        feats = (json.loads(info.read_text()).get("features") or {}) if info.is_file() else {}
        changed = []
        for section, column in DreamZeroPlugin._BLOB_COLUMN.items():
            entries = mod.get(section) or {}
            nulls = [k for k, v in entries.items() if not v.get("original_key")]
            if not nulls:
                continue
            shape = (feats.get(column) or {}).get("shape") or []
            width = shape[0] if shape else None
            span = max((entries[k].get("end") or 0) for k in nulls)
            if width is not None and span != width:
                return (f"{mj}: {section} groups with no original_key span [0,{span}) but "
                        f"'{column}' is {width} wide — refusing to guess the mapping.")
            for k in nulls:
                entries[k]["original_key"] = column
            changed.append(f"{section}: {len(nulls)} groups -> {column}[0:{span}]")
        if changed:
            mj.write_text(json.dumps(mod, indent=4))
            print("[locomanilab:dreamzero] blob schema, filled original_key — " + "; ".join(changed))
        return None

    # ---- train ---------------------------------------------------------
    def train(self, *, dataset: Path, gpus: int, gpu_ids: str | None = None,
              master_port: int | None = None, task: str | None = None,
              config_name: str | None = None, output_dir: str | None = None,
              overrides: list[str] | None = None, **opts: Any) -> StageResult:
        dataset = self.ctx.input_path(dataset)
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("train")
        tr = self.manifest.train
        d = tr.get("defaults", {})
        try:
            bb_name, bb = self._backbone(config_name)
        except KeyError as e:
            return StageResult(ok=False, message=str(e))

        variant = self._variant(dataset)
        missing = self._missing_meta(dataset, variant)
        if missing:
            return StageResult(ok=False, message=missing)

        weights, err = self._weight_overrides(bb, bb_name)
        if err:
            return StageResult(ok=False, message=err)

        # The hydra data config is <lane base><backbone suffix>: the controller lane picks
        # the base (sonic's latent motion_token vs amo's explicit 36-D command) and the
        # backbone picks the suffix (Wan2.1 14B vs Wan2.2 5B). Checked on disk because a
        # missing combination is otherwise a hydra error 200 lines into a stack trace.
        base = (variant.get("data_config_base")
                or "dreamzero/unitree_g1_sonic_relative")
        data_config = base + (bb.get("data_config_suffix") or "")
        cfg_file = repo / "groot/vla/configs/data" / f"{data_config}.yaml"
        if not self.runner.dry_run and not cfg_file.is_file():
            return StageResult(ok=False, message=(
                f"no data config for lane '{variant.get('embodiment_tag', '?')}' on backbone "
                f"'{bb_name}': {cfg_file} does not exist. Write it, or train this dataset on "
                f"the other backbone."))

        # LoRA vs full changes three coupled switches and the deepspeed stage; deriving
        # them from one knob is what stops a half-LoRA/half-full run. `--set
        # train_architecture=...` still works because the derived values are emitted
        # BEFORE the overrides, so a user override of any single one still wins.
        arch = str(tr.get("train_architecture", "full")).lower()
        deepspeed = (tr.get("deepspeed_full", "groot/vla/configs/deepspeed/zero3.json")
                     if arch == "full"
                     else tr.get("deepspeed", "groot/vla/configs/deepspeed/zero2.json"))

        run_name = task or ("locomanilab_dreamzero_" + time.strftime("%Y%m%d_%H%M%S"))
        # run_name is derived from the output dir's BASENAME by the trainer
        # (base.py: `cfg.training_args.run_name = output_dir.split("/")[-1]`), so --task
        # only takes effect through the path.
        out = self.ctx.anchor(output_dir) if output_dir else None
        if out is None:
            root = self.ctx.path("dreamzero_outputs") or self.ctx.workspace_root / "checkpoints" / "dreamzero"
            out = root / run_name
        elif out.name != run_name and task:
            out = out / run_name

        extra_env: dict[str, str] = {k: str(v) for k, v in (tr.get("env") or {}).items()}
        self.drop_expandable_segments(extra_env, "dreamzero")
        gpus = self.resolve_gpus(gpus, gpu_ids, extra_env)
        # torchrun puts the ENTRYPOINT's directory on sys.path, not the cwd, so
        # `groot.*` resolves only through the `pip install -e .`. That install records an
        # absolute path, and the env here was built on a machine where the checkout lived
        # somewhere else — leaving a finder pointing at a path that does not exist and a
        # `ModuleNotFoundError: No module named 'groot'` two seconds into every run.
        # Exporting the repo makes the command work against any checkout, installed or not.
        extra_env["PYTHONPATH"] = (f"{repo}:{os.environ['PYTHONPATH']}"
                                   if os.environ.get("PYTHONPATH") else str(repo))

        cmd = ["torchrun", f"--nproc_per_node={gpus}", "--standalone"]
        if master_port is not None:
            cmd += [f"--master_port={master_port}"]
        cmd += [tr.get("entrypoint", "groot/vla/experiment/experiment.py")]

        hydra = [
            f"data={data_config}",
            f"model={tr.get('model', 'dreamzero/vla')}",
            f"model/dreamzero/action_head={bb['action_head']}",
            f"model/dreamzero/transform={tr.get('transform', 'dreamzero_cotrain')}",
            f"train_architecture={arch}",
            f"num_views={d.get('num_views', 1)}",
            f"num_frames={d.get('num_frames', 33)}",
            f"action_horizon={d.get('action_horizon', 24)}",
            f"num_frame_per_block={d.get('num_frame_per_block', 2)}",
            f"num_action_per_block={d.get('num_action_per_block', 24)}",
            f"num_state_per_block={d.get('num_state_per_block', 1)}",
            f"max_chunk_size={d.get('max_chunk_size', 2)}",
            f"max_state_dim={d.get('max_state_dim', 43)}",
            f"seed={d.get('seed', 42)}",
            f"training_args.learning_rate={d.get('learning_rate', 1e-5)}",
            f"training_args.warmup_ratio={d.get('warmup_ratio', 0.05)}",
            f"training_args.deepspeed={deepspeed}",
            f"output_dir={out}",
            f"per_device_train_batch_size={d.get('per_device_batch_size', 1)}",
            f"max_steps={d.get('max_train_steps', 20000)}",
            "save_strategy=steps",
            f"save_steps={d.get('save_steps', 2000)}",
            f"save_total_limit={d.get('save_total_limit', 10)}",
            f"weight_decay={d.get('weight_decay', 1e-5)}",
            "upload_checkpoints=false",
            "bf16=true", "tf32=true", "eval_bf16=true",
            "dataloader_pin_memory=false",
            f"dataloader_num_workers={d.get('dataloader_num_workers', 1)}",
            f"save_lora_only={'true' if arch == 'lora' else 'false'}",
            "report_to=wandb", "wandb_project=dreamzero",
            f"{variant.get('data_root_key') or tr.get('data_root_key', 'unitree_g1_sonic_data_root')}={dataset}",
        ]
        if bb.get("image_resolution"):
            h, w = bb["image_resolution"]
            hydra += [f"image_resolution_height={h}", f"image_resolution_width={w}"]
        if bb.get("frame_seqlen"):
            hydra.append(f"frame_seqlen={bb['frame_seqlen']}")
        hydra += weights
        # A released DreamZero checkpoint to LoRA-adapt from. The two ++ flags are what
        # upstream's new-embodiment scripts pass alongside it: the action head's components
        # are re-created for our 78-D action instead of being loaded, and LoRA is injected
        # after that. Without a pretrained path none of the three belong on the line.
        # Per BACKBONE, not per model: the released DreamZero checkpoints are all Wan2.1
        # 14B, so asking for one under the 5B backbone would shape-mismatch every DiT
        # tensor and quietly train from scratch. wan22 declares no pretrained_path_key.
        pre = self.ctx.path(bb["pretrained_path_key"]) if bb.get("pretrained_path_key") else None
        if pre is not None:
            hydra += [f"pretrained_model_path={pre}",
                      "++action_head_cfg.config.skip_component_loading=true",
                      f"++action_head_cfg.config.defer_lora_injection="
                      f"{'true' if arch == 'lora' else 'false'}"]
        # `--set k=v` last: hydra's dotlist takes the LAST occurrence of a key, so a user
        # override beats the manifest default above it.
        hydra += list(overrides or [])

        print(f"[locomanilab:dreamzero] backbone {bb_name} -> data={data_config} "
              f"action_head={bb['action_head']}"
              + (f"  pretrained={pre}" if pre is not None
                 else ("  (no pretrained_model_path: training from the Wan video prior with a "
                       "randomly-initialised action head" +
                       ("; this backbone cannot use one, see the manifest)"
                        if not bb.get("pretrained_path_key")
                        else "; set paths.dreamzero_pretrained to start from a released "
                             "DreamZero checkpoint)"))))
        return self.runner.run(cmd + hydra, cwd=repo, env_kind=env.kind, env_name=env.name,
                               setup=self.env_setup(env.kind),
                               extra_env=extra_env or None, stage="train")

    def _weight_overrides(self, bb: dict, bb_name: str) -> tuple[list[str], str | None]:
        """The four Wan weight paths as hydra overrides, or a message naming what is unset."""
        dit = self._weight(bb, "dit_path_key")
        img_root = self._weight(bb, "image_encoder_path_key")
        tok = self._weight(bb, "tokenizer_path_key")
        # dict, not a list: wan21 names the same dir for the DiT and the CLIP encoder, and
        # reporting `paths.dreamzero_wan21` twice would read like two separate settings.
        want = {bb.get("dit_path_key"): dit,
                bb.get("image_encoder_path_key"): img_root,
                bb.get("tokenizer_path_key"): tok}
        # A `None` key means the MANIFEST is missing that *_path_key, not env.local.
        missing_key = [k for k in ("dit_path_key", "image_encoder_path_key",
                                   "tokenizer_path_key") if not bb.get(k)]
        if missing_key:
            return [], (f"configs/dreamzero.yaml backbone {bb_name} is missing "
                        f"{', '.join(missing_key)}")
        unset = [k for k, v in want.items() if v is None]
        if unset:
            return [], (f"dreamzero backbone {bb_name}: set paths.{', paths.'.join(unset)} "
                        f"in env.local.yaml (Wan weights / umt5-xxl tokenizer)")
        return ([f"dit_version={dit}",
                 f"text_encoder_pretrained_path={dit / bb['text_encoder']}",
                 f"vae_pretrained_path={dit / bb['vae']}",
                 f"image_encoder_pretrained_path={img_root / bb['image_encoder']}",
                 f"tokenizer_path={tok}"], None)

    @staticmethod
    def _missing_meta(dataset: Path, variant: dict[str, Any] | None = None) -> str | None:
        """Refuse a dataset that has not been through `lab convert --model dreamzero`.

        Both files are produced by convert and neither has a useful fallback: without
        embodiment.json the loader cannot pick a modality config at all, and a stats.json
        that does not cover the action columns dies with a bare KeyError several minutes
        into startup — after the 5B/14B weights have been loaded.
        """
        meta = dataset / "meta"
        if not (meta / "embodiment.json").exists():
            return (f"{meta}/embodiment.json not found — run "
                    f"`lab convert --model dreamzero --dataset {dataset}` first "
                    f"and train on the prepared dir it prints.")
        try:
            stats = json.loads((meta / "stats.json").read_text())
        except Exception as e:  # noqa: BLE001
            return f"cannot read {meta}/stats.json: {e}"
        # The columns this LANE reads, taken from the dataset's own modality.json rather
        # than hardcoded: sonic reads action.motion_token + teleop.*_hand_joints, amo reads
        # the `action` blob + action.hands, and a future controller will read something
        # else again. DreamZero looks statistics up by original key for every state and
        # action group in the file, so all of them have to be present.
        try:
            mod = json.loads((meta / "modality.json").read_text())
        except Exception as e:  # noqa: BLE001
            return f"cannot read {meta}/modality.json: {e}"
        need = []
        for section in ("state", "action"):
            for entry in (mod.get(section) or {}).values():
                key = entry.get("original_key")
                if key and key not in need:
                    need.append(key)
        if not need:
            return (f"{meta}/modality.json declares no state/action original_key — "
                    f"re-run `lab convert --model dreamzero` (it fills blob-schema keys).")
        missing = [k for k in need if k not in stats]
        if missing:
            return (f"{meta}/stats.json is missing {', '.join(missing)} — DreamZero looks "
                    f"statistics up by original key, so this is a KeyError at step 0. "
                    f"Re-run `lab convert --model dreamzero`.")
        return None

    # ---- deploy --------------------------------------------------------
    CKPT_HINT = ("--model-path (a DreamZero checkpoint dir, e.g. "
                 "checkpoints/<run>/checkpoint-2000), --run-dir, or "
                 "paths.dreamzero_ckpt in env.local.yaml")

    def deploy(self, *, model_path: str | None = None, run_dir: str | None = None,
               prompt: str | None = None, port: int = 5550, **opts: Any) -> StageResult:
        """Start the gr00t-compatible policy server (:5550/get_action) for a checkpoint."""
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("deploy")
        ckpt = self.resolve_ckpt(model_path, run_dir, env_key="dreamzero_ckpt",
                                 stage="deploy", hint=self.CKPT_HINT)
        if isinstance(ckpt, StageResult):
            return ckpt
        return self.runner.run(self._server_cmd(str(ckpt), port, prompt), cwd=repo,
                               env_kind=env.kind, env_name=env.name,
                               setup=self.env_setup(env.kind), stage="deploy")

    # ---- sim harness (eval + infer-check share this) --------------------
    def _sim_server(self, *, stage: str, prompt: str | None, port: int,
                    model_path: str | None = None, run_dir: str | None = None,
                    **opts: Any) -> SimServer | StageResult:
        """DreamZero gr00t-compat server (conda dreamzero). Emits latent motion_token,
        like fastwam / starVLA / PRTS."""
        ckpt = self.resolve_ckpt(model_path, run_dir, env_key="dreamzero_ckpt", stage=stage,
                                 hint=self.CKPT_HINT)
        if isinstance(ckpt, StageResult):
            return ckpt
        dep = self.manifest.deploy
        prompt = prompt or dep.get("prompt_fixed")
        repo = self.repo(self.manifest.repo_key)
        # PYTHONPATH: the server imports `groot.*`; `pip install -e .` covers it, but a
        # checkout that was never installed would otherwise fail at import.
        import shlex
        pre = [f"export PYTHONPATH={shlex.quote(str(repo))}${{PYTHONPATH:+:${{PYTHONPATH}}}}"]
        return SimServer(argv=self.server_argv(self._server_cmd(str(ckpt), port, prompt),
                                               pre=pre),
                         cwd=repo, prompt=prompt,
                         inference_rate=dep.get("inference_rate"))

    @staticmethod
    def _rendezvous_port(port: int) -> int:
        """A torch.distributed rendezvous port unique to this serving port.

        Offset from the gr00t default (5550 -> 29500) so the usual single-server case keeps
        the port it always had, and wrapped into the ephemeral-safe 20000-29999 band so an
        unusual --port cannot land on something privileged or already-listening.
        """
        return 20000 + (29500 - 20000 + port - 5550) % 10000

    def _server_cmd(self, ckpt: str, port: int, prompt: str | None) -> list[str]:
        dep = self.manifest.deploy
        repo = self.repo(self.manifest.repo_key)
        # The server needs a torch.distributed process group even at world_size=1
        # (GrootSimPolicy.__init__ calls dist.get_rank()), and its rendezvous port is a
        # SECOND port that must also be unique. Leaving it at the default 29500 means two
        # DreamZero servers can never run at once: the second dies with
        # `DistNetworkError: ... port: 29500 ... EADDRINUSE` before loading anything, while
        # the harness just reports "policy server did not open :<port>". Parallel eval /
        # infer-check on several checkpoints is normal here, so derive it from the serving
        # port and keep the two in lockstep.
        cmd = ["python", dep.get("gr00t_server", "deployment/run_dreamzero_server.py"),
               "--model-path", str(ckpt), "--port", str(port),
               "--master-port", str(self._rendezvous_port(port)),
               "--dreamzero-repo", str(repo),
               "--embodiment-tag", str(dep.get("embodiment_tag", "unitree_g1_sonic")),
               "--video-frames-per-chunk", str(dep.get("video_frames_per_chunk", 8))]
        # Only when the manifest overrides it -- otherwise the server derives it from the
        # checkpoint, which is the only place the training horizon is actually recorded.
        if dep.get("video_stride"):
            cmd += ["--video-stride", str(dep["video_stride"])]
        prompt = prompt or dep.get("prompt_fixed")
        if prompt:
            cmd += ["--initial-prompt", str(prompt)]
        # The checkpoint's experiment_cfg/conf.yaml stores the TRAINING machine's absolute
        # tokenizer path; override it when this machine names one, so a checkpoint stays
        # portable (the same trap gr00t's Cosmos-Reason2 path has, see sim-eval-pipeline).
        tok = self.ctx.path("dreamzero_tokenizer")
        if tok is not None:
            cmd += ["--tokenizer-path", str(tok)]
        return cmd
