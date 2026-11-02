"""InternVLA-A1.5 plugin.

InternVLA-A1.5 (github.com/InternRobotics/InternVLA-A-series, CC BY-NC-SA 4.0) =
Qwen3.5-2B VLM backbone + a unified flow-matching action expert, plus a latent
*foresight* branch supervised by a frozen WAN2.2-TI2V-5B video model (dropped at
inference). The repo is a LeRobot fork: draccus dataclasses, `accelerate launch
src/lerobot/scripts/lerobot_train.py`.

Maps locomanilab's stages onto the fork's scripts:
  convert -> scripts/data/locomanilab_lerobot_v21_to_v30.py  (v2.1 -> v3.0 + robot_type stamp)
  train   -> accelerate launch src/lerobot/scripts/lerobot_train.py <--policy.* --dataset.*>
  deploy  -> deployment/run_internvla_server.py       (gr00t-compat :5550/get_action)
  eval / infer-check -> that same server + the shared sim / offline client

Everything machine-specific is in env.local.yaml (`repos.internvla_a15`,
`paths.internvla_a15_{data_root,base,vlm,outputs,ckpt}`); the flag semantics and
the traps behind them are documented in locomanilab/configs/internvla_a15.yaml.

Never imports InternVLA / lerobot code -- pure shell-out.
"""
from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import Any

from locomanilab.plugin import ModelPlugin, SimServer, StageResult, register
from locomanilab.models._legacy import converter_path


@register("internvla_a15")
class InternVLAA15Plugin(ModelPlugin):

    # ---- paths ---------------------------------------------------------
    def _data_root(self) -> Path | None:
        """Where v21/ staging links and v30/ products live.

        Also exported as HF_LEROBOT_HOME: the fork resolves a dataset dir two
        incompatible ways (LeRobotDatasetMetadata treats `--dataset.root` as the
        dataset dir, factory.find_info_json_path_for_repo as its parent), and
        `HF_LEROBOT_HOME / repo_id` with root unset is the one form both agree on.
        """
        return self.ctx.path("internvla_a15_data_root")

    def _v30_dir(self, dataset: Path) -> Path | None:
        root = self._data_root()
        return None if root is None else root / "v30" / Path(dataset).name

    def _stage_link(self, dataset: Path) -> Path | None:
        root = self._data_root()
        return None if root is None else root / "v21" / Path(dataset).name / "data"

    def _missing_data_root(self, stage: str) -> StageResult:
        return StageResult(
            ok=False,
            message=(f"{stage}: paths.internvla_a15_data_root is not set in env.local.yaml. "
                     f"It holds the v21/ staging links and the v30/ training copies, and is "
                     f"exported as HF_LEROBOT_HOME."))

    # ---- convert -------------------------------------------------------
    def convert(self, *, dataset: Path, **opts: Any) -> StageResult:
        """v2.1 -> v3.0 (the only layout this fork trains on) + robot_type stamp.

        Unlike prts/pi05 this is a real materialising convert, because the fork's
        dataloader cannot read a v2.1 dataset at all. The product shares nothing
        with the source but symlinks (`<root>/v21/<ds>/{data,videos,meta}`), so
        the source dataset -- shared with gr00t / PRTS / pi0.5 / DreamZero -- is
        never written to.
        """
        dataset = self.ctx.input_path(dataset)
        from locomanilab.common import variants as variantmod

        dataset = Path(dataset).expanduser()
        cv = self.manifest.convert
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("convert")

        problem = variantmod.check_modality(
            dataset, cv.get("require_modality", {}),
            what=f"InternVLA-A1.5 (robot_type={cv.get('robot_type')})")
        if problem:
            return StageResult(ok=False, message=problem)

        # An unknown robot_type does NOT raise in the fork: SchemaRegistry.get() logs a
        # warning and returns an identity mapping (observation.image -> image0), i.e. it
        # trains on the wrong keys in silence. If the schema file is not in the checkout,
        # stop here rather than discover it as a bad loss curve.
        schema = repo / cv.get("schema_file", "src/lerobot/dataset_schemas/configs/unitree_g1_sonic.yaml")
        if not self.runner.dry_run and not schema.is_file():
            return StageResult(
                ok=False,
                message=(f"convert: {schema} is missing -- the fork checkout is not on the "
                         f"maintained G1 integration revision, so robot_type "
                         f"{cv.get('robot_type')!r} would silently fall back to the "
                         f"registry's identity schema."))

        root = self._data_root()
        if root is None:
            return self._missing_data_root("convert")

        # Provenance: the staging symlink IS the record of what this product was built
        # from. The upstream converter silently `return`s when its output dir exists
        # (`if new_root.is_dir(): return`), so without this a rename or a second dataset
        # of the same basename would train on the wrong data and say nothing.
        link, out = self._stage_link(dataset), self._v30_dir(dataset)
        if not self.runner.dry_run and link.is_symlink():
            have = link.resolve()
            want = (dataset / "data").resolve()
            if have != want:
                return StageResult(
                    ok=False,
                    message=(f"convert: {out} was built from {have.parent}, not {dataset}. "
                             f"Delete {out} and {link.parent}, or convert under a different "
                             f"dataset name."))

        cmd = ["python", converter_path(repo, cv.get("converter", "scripts/data/locomanilab_lerobot_v21_to_v30.py")),
               "--src", str(dataset.absolute()),
               "--root", str(root),
               "--robot-type", str(cv.get("robot_type", "unitree_g1_sonic"))]
        res = self.runner.run(cmd, cwd=repo, env_kind=env.kind, env_name=env.name,
                              setup=self.env_setup(env.kind), stage="convert")
        if res.ok and out is not None:
            res.artifacts["v30_dataset"] = out
            res.message = (res.message or "") + (
                f"\nTrain on it with `--dataset --repo-id v30/{Path(dataset).name}`; "
                f"⚠️ `lab infer-check` still wants the ORIGINAL v2.1 dataset "
                f"({dataset}) -- it reads data/chunk-*/episode_*.parquet.")
        return res

    # ---- train ---------------------------------------------------------
    def _base_model(self, override: str | None) -> str | None:
        """The WIDENED base checkpoint (tools/widen_for_g1_sonic.py), not the released one.

        The release is max_action_dim=max_state_dim=32; loading it into a 78/43 model
        raises even with strict=False, because PyTorch reports size mismatches
        regardless of that flag.
        """
        if override:
            return str(self.ctx.input_path(override))
        p = self.ctx.path("internvla_a15_base")
        return None if p is None else str(p)

    def train(self, *, dataset: Path, gpus: int, gpu_ids: str | None = None,
              master_port: int | None = None, task: str | None = None,
              base_model: str | None = None, output_dir: str | None = None,
              overrides: list[str] | None = None, **opts: Any) -> StageResult:
        dataset = self.ctx.input_path(dataset)
        dataset = Path(dataset).expanduser()
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("train")
        tr = self.manifest.train
        d = tr.get("defaults", {})

        root = self._data_root()
        if root is None:
            return self._missing_data_root("train")
        v30 = self._v30_dir(dataset)
        if not self.runner.dry_run and not (v30 / "meta" / "info.json").is_file():
            return StageResult(
                ok=False,
                message=(f"train: {v30} does not exist. This fork trains only on LeRobot "
                         f"v3.0 -- run `lab convert --model {self.name} --dataset {dataset}` "
                         f"first."))

        base = self._base_model(base_model)
        if base is None:
            return StageResult(
                ok=False,
                message=("train: no base checkpoint. Set paths.internvla_a15_base in "
                         "env.local.yaml to the output of tools/widen_for_g1_sonic.py "
                         "(NOT the released InternVLA-A1.5-base, which is 32-D), or pass "
                         "--base-model."))
        vlm = self.ctx.path("internvla_a15_vlm")

        run_name = task or ("locomanilab_internvla_a15_" + time.strftime("%Y%m%d_%H%M%S"))
        if output_dir:
            out = self.ctx.anchor(output_dir)
        else:
            outs = self.ctx.path("internvla_a15_outputs") or self.ctx.workspace_root / "checkpoints" / "internvla_a15"
            if outs is None:
                return StageResult(
                    ok=False,
                    message="train: pass --output-dir or set paths.internvla_a15_outputs.")
            out = outs / run_name
        # TrainPipelineConfig.validate() raises FileExistsError on an existing
        # output_dir, so it must NOT be created here.

        extra_env: dict[str, str] = {k: str(v) for k, v in (tr.get("env") or {}).items()}
        extra_env["HF_LEROBOT_HOME"] = str(root)
        self.drop_expandable_segments(extra_env, self.name)
        gpus = self.resolve_gpus(gpus, gpu_ids, extra_env)
        self.raise_nofile(model=self.name)

        cmd = ["accelerate", "launch"]
        if gpus > 1:
            cmd += ["--multi_gpu"]
        cmd += [f"--num_processes={gpus}", "--num_machines=1", "--machine_rank=0"]
        if master_port is not None:
            cmd += [f"--main_process_port={master_port}"]
        entrypoint = tr.get("entrypoint", "src/lerobot/scripts/lerobot_train.py")
        cmd += [entrypoint]

        policy = tr.get("policy_type", "internvla_a1_5")
        cmd += [
            f"--policy.type={policy}",
            f"--policy.pretrained_path={base}",
            f"--policy.max_action_dim={d.get('max_action_dim', 78)}",
            f"--policy.max_state_dim={d.get('max_state_dim', 43)}",
            f"--policy.chunk_size={d.get('chunk_size', 50)}",
            f"--policy.n_action_steps={d.get('chunk_size', 50)}",
            f"--policy.action_loss_only={_b(d.get('action_loss_only', True))}",
            f"--policy.enable_vqa_loss={_b(d.get('enable_vqa_loss', False))}",
            # tokenize_state exists on BOTH configs with OPPOSITE defaults (policy True,
            # dataset False). Disagreeing means the model builds state_proj while the
            # prompt carries no State: block -- training with no proprioception, and a
            # loss curve that looks fine. Always emit both.
            f"--policy.tokenize_state={_b(d.get('tokenize_state', True))}",
            f"--policy.optimizer_lr={d.get('lr', 5.0e-5)}",
            f"--policy.scheduler_warmup_steps={d.get('warmup_steps', 2000)}",
            f"--policy.scheduler_decay_steps={d.get('decay_steps', 60000)}",
            f"--policy.scheduler_decay_lr={d.get('decay_lr', 5.0e-6)}",
            "--policy.dtype=bfloat16",
            "--policy.push_to_hub=false",
            f"--policy.repo_id=lerobot_lab/{policy}",
        ]
        if vlm is not None:
            cmd += [f"--policy.vlm_model_name_or_path={vlm}"]

        cmd += [
            f"--dataset.type={tr.get('dataset_type', policy)}",
            # No --dataset.root: HF_LEROBOT_HOME + repo_id is the only resolution both
            # code paths in the fork agree on (see the manifest's hazards).
            f"--dataset.repo_id=v30/{Path(dataset).name}",
            f"--dataset.action_mode={d.get('action_mode', 'abs')}",
            f"--dataset.tokenize_state={_b(d.get('tokenize_state', True))}",
            f"--dataset.max_state_dim={d.get('max_state_dim', 43)}",
            f"--dataset.max_action_dim={d.get('max_action_dim', 78)}",
            f"--dataset.chunk_size={d.get('chunk_size', 50)}",
            f"--dataset.use_fast_action_tokens={_b(d.get('use_fast_action_tokens', False))}",
            # TRUE would KeyError: factory.py assigns meta.stats[<camera key>][...] for
            # every camera, and locomanilab datasets carry no image entry in episodes_stats.
            "--dataset.use_imagenet_stats=false",
            "--dataset.use_external_stats=false",
            "--dataset.dist_loading=false",
        ]
        if vlm is not None:
            cmd += [f"--dataset.vlm_model_name_or_path={vlm}"]

        cmd += [
            f"--output_dir={out}",
            f"--job_name={run_name}",
            f"--seed={d.get('seed', 42)}",
            f"--batch_size={d.get('batch_size', 8)}",
            f"--steps={d.get('steps', 60000)}",
            f"--save_freq={d.get('save_freq', 5000)}",
            f"--log_freq={d.get('log_freq', 200)}",
            f"--num_workers={d.get('num_workers', 8)}",
            "--wandb.enable=false",
        ]
        # `--set k=v` REPLACES the flag we emitted rather than being appended after it.
        # draccus builds an argparse parser, so a duplicate flag would probably be
        # last-wins -- but "probably" is not a thing to bet a 60k-step run on, and a
        # command with `--steps=60000 ... --steps=2000` in it is unreadable in a log.
        folded = _apply_overrides(cmd, overrides or [], entrypoint)
        if isinstance(folded, str):
            return StageResult(ok=False, message=folded)
        cmd = folded

        res = self.runner.run(cmd, cwd=repo, env_kind=env.kind, env_name=env.name,
                              setup=self.env_setup(env.kind),
                              extra_env=extra_env or None, stage="train")
        res.artifacts["run_dir"] = out
        return res

    # ---- deploy --------------------------------------------------------
    def deploy(self, *, model_path: str | None = None, run_dir: str | None = None,
               prompt: str | None = None, port: int = 5550, **opts: Any) -> StageResult:
        """Start the gr00t-compatible policy server (:5550/get_action) for a checkpoint."""
        repo = self.repo(self.manifest.repo_key)
        env = self.manifest.env("deploy")
        ckpt = self.resolve_ckpt(model_path, run_dir, env_key="internvla_a15_ckpt",
                                 stage="deploy", hint=self.CKPT_HINT)
        if isinstance(ckpt, StageResult):
            return ckpt
        return self.runner.run(self._server_cmd(str(ckpt), port, prompt), cwd=repo,
                               env_kind=env.kind, env_name=env.name,
                               setup=self.env_setup(env.kind), stage="deploy")

    # ---- sim harness (eval + infer-check share this) --------------------
    CKPT_HINT = ("--model-path (a <run>/checkpoints/<step>/pretrained_model dir, or the "
                 "`last` symlink beside it), --run-dir, or paths.internvla_a15_ckpt in "
                 "env.local.yaml")

    def _sim_server(self, *, stage: str, prompt: str | None, port: int,
                    model_path: str | None = None, run_dir: str | None = None,
                    **opts: Any) -> SimServer | StageResult:
        """InternVLA-A1.5 gr00t-compat server (conda internvla_a1_5). Emits latent
        motion_token, like fastwam / starVLA / PRTS."""
        ckpt = self.resolve_ckpt(model_path, run_dir, env_key="internvla_a15_ckpt",
                                 stage=stage, hint=self.CKPT_HINT)
        if isinstance(ckpt, StageResult):
            return ckpt
        prompt = prompt or self.manifest.deploy.get("prompt_fixed")
        return SimServer(argv=self._server_argv(str(ckpt), port, prompt),
                         cwd=self.repo(self.manifest.repo_key), prompt=prompt,
                         inference_rate=self.manifest.deploy.get("inference_rate"))

    # ---- deploy helpers ------------------------------------------------
    def _server_cmd(self, ckpt: str, port: int, prompt: str | None) -> list[str]:
        dep = self.manifest.deploy
        cmd = ["python", dep.get("module", "deployment/run_internvla_server.py"),
               "--ckpt-path", str(ckpt), "--port", str(port),
               "--robot-type", str(dep.get("robot_type", "unitree_g1_sonic")),
               "--resize-size", str(dep.get("resize_size", 224)),
               "--max-prompt-length", str(dep.get("max_prompt_length", 650)),
               "--inference-backend", str(dep.get("inference_backend", "optimized"))]
        vlm = self.ctx.path("internvla_a15_vlm")
        if vlm is not None:
            # The checkpoint config names the hub id "Qwen/Qwen3.5-2B"; the server owns
            # the Qwen3-VL processor, so on an offline machine it needs the local copy.
            cmd += ["--vlm-model-path", str(vlm)]
        prompt = prompt or dep.get("prompt_fixed")
        if prompt:
            cmd += ["--initial-prompt", str(prompt)]
        return cmd

    def _server_argv(self, ckpt: str, port: int, prompt: str | None) -> list[str]:
        """Env-wrapped server for the sim / infer-check harness (Popen'd directly)."""
        repo = self.repo(self.manifest.repo_key)
        pre = [f"export PYTHONPATH={shlex.quote(str(repo))}:{shlex.quote(str(repo / 'src'))}"
               "${PYTHONPATH:+:${PYTHONPATH}}",
               "export TOKENIZERS_PARALLELISM=false"]
        return self.server_argv(self._server_cmd(ckpt, port, prompt), pre=pre)


def _b(v: Any) -> str:
    """draccus parses booleans from the literal strings 'true'/'false'."""
    return "true" if bool(v) else "false"


def _apply_overrides(cmd: list[str], overrides: list[str],
                     entrypoint: str) -> list[str] | str:
    """Fold `--set k=v` into an already-built draccus flag list, replacing in place.

    Only the TAIL after the training script is touched: the head is `accelerate
    launch`'s own flags, and moving one of those to the end of the line (past the
    script path) would hand it to draccus instead, where it is an unknown argument.

    A `k=v` with no `=` is an error worth refusing on rather than silently passing a
    bare `--k` to draccus, which would consume the NEXT token as its value and shift
    every remaining flag by one.

    Returns the folded argv, or an error message string.
    """
    i = cmd.index(entrypoint)
    head, tail = cmd[:i + 1], cmd[i + 1:]
    for kv in overrides:
        flag = kv if kv.startswith("--") else f"--{kv}"
        if "=" not in flag:
            return f"--set expects k=v, got {kv!r}"
        key = flag.split("=", 1)[0]
        if any(c.startswith(f"{key}=") for c in head):
            return (f"--set {kv!r} targets an `accelerate launch` flag, not a "
                    f"training option; use --gpu-ids / --master-port instead")
        tail = [c for c in tail if not c.startswith(f"{key}=")]
        tail.append(flag)
    return head + tail
