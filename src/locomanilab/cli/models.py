"""Models commands."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import typer
from locomanilab.base import StageResult
from locomanilab.registry import available_models
from .app import app
from .common import _ctx, _finish, _plugin, _resolve_sonic_ckpt, _stage

@app.command("list-models")
def list_models():
    """List registered models."""
    for name, m in available_models().items():
        typer.echo(f"  {name:12s} embodiment={m.embodiment or '-':18s} action_dim={m.action.get('dim', '?')}")


@app.command()
def convert(
    model: str = typer.Option(..., "--model", "-M"),
    dataset: Path = typer.Option(..., "--dataset", "-d"),
    config_name: Optional[str] = typer.Option(None, "--config-name", help="pi05 recipe or PRTS data configuration."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir",
                                      help="dreamzero: where the prepared dataset dir goes "
                                           "(default <paths.dreamzero_data_root>/<dataset>). Needed "
                                           "when two datasets resolve to the same directory NAME."),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Convert the shared dataset into <model>'s training format."""
    plugin, _ = _plugin(model, dry_run)
    # anchor to the invoking cwd — plugins run their scripts from the model repo
    from locomanilab.workspace import absolute_path
    dataset = absolute_path(dataset)
    if not dataset.is_dir():
        typer.secho(f"dataset dir not found: {dataset}", fg=typer.colors.RED)
        raise typer.Exit(1)
    _finish(_stage(plugin, "convert", dataset=dataset, output_dir=output_dir, config_name=config_name))


@app.command()
def train(
    model: str = typer.Option(..., "--model", "-M"),
    dataset: Path = typer.Option(..., "--dataset", "-d"),
    gpus: int = typer.Option(1, "--gpus", "-g", min=1),
    gpu_ids: Optional[str] = typer.Option(None, "--gpu-ids", help="pin GPUs, e.g. 6,7 (sets CUDA_VISIBLE_DEVICES)"),
    master_port: Optional[int] = typer.Option(None, "--master-port", help="torchrun master port (gr00t)"),
    task: Optional[str] = typer.Option(None, "--task"),
    base_model: Optional[str] = typer.Option(None, "--base-model", help="base model path (gr00t)"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir"),
    per_gpu_batch: Optional[int] = typer.Option(None, "--per-gpu-batch",
                                                help="gr00t: global batch = per_gpu_batch * num_gpus (default 256)"),
    action_space: Optional[str] = typer.Option(None, "--action-space",
                                      help="what the policy predicts: motion_token (SONIC latent) | smpl "
                                           "(SONIC encoder input; deploy runs encoder->decoder). Default "
                                           "motion_token; gr00t / fastwam only."),
    config_name: Optional[str] = typer.Option(None, "--config-name",
                                      help="Model recipe: pi05 config name, PRTS YAML, or DreamZero wan22/wan21."),
    set_: list[str] = typer.Option(None, "--set", help="passthrough override key=value (repeatable)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Launch <model> training locally (uses this machine's env + data)."""
    plugin, _ = _plugin(model, dry_run)
    from locomanilab.workspace import absolute_path
    dataset = absolute_path(dataset)
    if not dataset.is_dir():
        raise typer.BadParameter(f"dataset directory not found: {dataset}", param_hint="--dataset")
    _finish(_stage(plugin, "train", dataset=dataset, gpus=gpus, gpu_ids=gpu_ids, master_port=master_port,
                   task=task, base_model=base_model, output_dir=output_dir,
                   per_gpu_batch=per_gpu_batch, action_space=action_space,
                   config_name=config_name, overrides=set_ or []))


@app.command()
def deploy(
    model: str = typer.Option(..., "--model", "-M"),
    config_name: Optional[str] = typer.Option(None, "--config-name", help="Model serving recipe; use the recipe matching the checkpoint."),
    ckpt_step: Optional[int] = typer.Option(None, "--ckpt-step",
                                help="fastwam: load this training step's step_*.pt from "
                                     "<run-dir>/checkpoints/weights instead of the newest one."),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p"),
    run_dir: Optional[str] = typer.Option(None, "--run-dir"),
    model_path: Optional[str] = typer.Option(None, "--model-path", help="gr00t model dir"),
    base_model: Optional[str] = typer.Option(None, "--base-model"),
    trt: bool = typer.Option(False, "--trt", help="gr00t: use TRT engine"),
    trt_engine: Optional[str] = typer.Option(None, "--trt-engine", help="gr00t: TRT engines dir (overrides env.local paths.gr00t_trt_engine)"),
    server_only: bool = typer.Option(False, "--server-only", help="gr00t: start only the policy server (:5550) for sim/remote clients"),
    port: int = typer.Option(5550, "--port", help="policy server port"),
    embodiment_tag: Optional[str] = typer.Option(None, "--embodiment-tag", help="gr00t: override embodiment tag (checkpoint-specific)"),
    action_space: Optional[str] = typer.Option(None, "--action-space",
                                     help="which checkpoint to deploy: motion_token (feed tokens to SONIC decoder) | "
                                          "smpl (reference-motion, encoder->decoder). Must match how it was trained "
                                          "(default motion_token). gr00t: picks the reference-motion embodiment tag; "
                                          "gr00t / fastwam only."),
    t5_mode: Optional[str] = typer.Option(None, "--t5-mode",
                                help="fastwam only (default encode): encode = load T5 once at startup to encode --prompt (slow, ~11GB extra); "
                                     "cache = read the embedding from <run-dir>/cache_t5 (stored with the checkpoint; "
                                     "prompt must exactly match a training prompt; T5 never loaded)."),
    sonic_ckpt: Optional[str] = typer.Option(None, "--sonic-ckpt",
                                help="gr00t / fastwam real-robot deploy: which SONIC controller tracks the policy -- "
                                     "low_latency (launcher default) | v1_1. `lab collect` uses the equivalent "
                                     "--policy-config path; "
                                     "use the variant the training data was collected with (lane sonic -> low_latency, "
                                     "sonic_v1_1 -> v1_1). Not for --server-only."),
    policy_config: Optional[str] = typer.Option(None, "--policy-config",
                                help="gr00t / fastwam real-robot deploy: g1_backend policy.yaml naming the SONIC variant "
                                     "(what --sonic-ckpt expands to; mutually exclusive with it)."),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Deploy a local checkpoint using the model's own environment."""
    plugin, ctx = _plugin(model, dry_run)
    if sonic_ckpt is not None or policy_config:
        if model not in ("gr00t", "fastwam"):
            typer.secho(f"--sonic-ckpt / --policy-config are gr00t / fastwam deploy options, not {model!r}",
                        fg=typer.colors.RED)
            raise typer.Exit(1)
        if sonic_ckpt is not None:
            policy_config = _resolve_sonic_ckpt(ctx, sonic_ckpt, backend="sonic",
                                                policy_config=policy_config)
    _finish(_stage(plugin, "deploy", ckpt_step=ckpt_step, prompt=prompt, run_dir=run_dir,
                   model_path=model_path, base_model=base_model, trt=trt,
                   trt_engine=trt_engine,
                   server_only=server_only, port=port, embodiment_tag=embodiment_tag,
                   action_space=action_space, t5_mode=t5_mode, config_name=config_name,
                   policy_config=policy_config))


@app.command("eval")
def eval_(
    typer_ctx: typer.Context,
    model: str = typer.Option("gr00t", "--model", "-M"),
    sim: bool = typer.Option(False, "--sim", help="closed-loop sim eval (humanoid_benchmark)"),
    task: Optional[str] = typer.Option(None, "--task",
                                help="--sim: a benchmark task (`lab tasks`); fills env, scene, prompt, "
                                     "episodes, step budget and reset settings from the task table. "
                                     "Options given on the command line win. Output defaults to "
                                     "runs/eval/<model>/<task>/."),
    task_set: Optional[str] = typer.Option(None, "--task-set",
                                help="--sim: evaluate a selection of benchmark tasks one after another "
                                     "(all | atomic | composite | seen | unseen_task | unseen_scene | "
                                     "<type>/<split>, comma-separated). Each task writes "
                                     "<root>/<task>/results.json (root: --video-dir, default "
                                     "runs/eval/<model>); finished tasks are skipped on a rerun, and "
                                     "<root>/summary.json reports per-task and per-group success rates."),
    model_path: Optional[str] = typer.Option(None, "--model-path", help="gr00t model dir"),
    run_dir: Optional[str] = typer.Option(None, "--run-dir", help="fastwam run dir"),
    ckpt_step: Optional[int] = typer.Option(None, "--ckpt-step",
                                help="fastwam: load this training step's step_*.pt from "
                                     "<run-dir>/checkpoints/weights instead of the newest one."),
    base_model: Optional[str] = typer.Option(None, "--base-model", help="fastwam Wan2.2 base"),
    trt: bool = typer.Option(False, "--trt"),
    trt_engine: Optional[str] = typer.Option(None, "--trt-engine", help="gr00t: TRT engines dir (overrides env.local paths.gr00t_trt_engine)"),
    embodiment_tag: Optional[str] = typer.Option(None, "--embodiment-tag"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p"),
    env_name: str = typer.Option("TabletopPickPlace", "--env", help="sim task env"),
    episodes: int = typer.Option(1, "--episodes", "-n"),
    max_steps: int = typer.Option(60, "--max-steps"),
    action_horizon: int = typer.Option(64, "--action-horizon", "-a"),
    render: str = typer.Option("video", "--render", help="video (headless egl, saves mp4) | viewer (live on-screen window)"),
    render_scale: int = typer.Option(1, "--render-scale", min=1, help="supersample camera by N then area-average to 320x240 (OSMesa aliasing test)"),
    watch: bool = typer.Option(False, "--watch", help="live-visualize the rollout in an on-screen MuJoCo window instead of saving mp4 (shorthand for --render viewer; needs a display, run in foreground)"),
    video_dir: Optional[str] = typer.Option(None, "--video-dir"),
    port: int = typer.Option(5550, "--port"),
    sim_env: str = typer.Option("tele_benchmark", "--sim-env", help="conda env for the sim client"),
    action_space: Optional[str] = typer.Option(None, "--action-space",
                                      help="gr00t: which action the model was trained on: motion_token (latent) | "
                                           "smpl (reference-motion, encoder->decoder); picks the server's embodiment "
                                           "tag (default motion_token). Other models refuse the flag."),
    check: bool = typer.Option(False, "--check", help="only check sim support, don't run"),
    t5_mode: Optional[str] = typer.Option(None, "--t5-mode",
                                help="fastwam only (default encode): encode = T5 encodes the prompt at startup; "
                                     "cache = read embeds from <run-dir>/cache_t5 (T5 never loaded)."),
    inference_rate: Optional[float] = typer.Option(None, "--inference-rate",
                                help="policy re-query rate (Hz). Default matches the real-robot "
                                     "PyTorch server (2.5Hz); --trt bumps to 10Hz. Higher over-replans "
                                     "and the robot moves wrong — leave unset unless you know why."),
    scene_id: Optional[int] = typer.Option(None, "--scene-id",
                                help='--sim: pin an iTHOR FloorPlan (e.g. 401) -- the scene the data was collected in (the dataset-name suffix usually IS the FloorPlan). Unset = RANDOM scenes: rooms the policy never saw, far lower and not comparable to same-scene numbers.'),
    config_name: Optional[str] = typer.Option(None, "--config-name",
                                help="pi05: openpi TrainConfig name matching the checkpoint and dataset. "
                                     "Use the same recipe as conversion and training."),
    backend: str = typer.Option("sonic", "--backend",
                                help='--sim: whole-body controller: sonic | amo | heft | scalebfm | almi (as `lab collect --sim`). MUST be the backend the training data was collected through -- it fixes the action space (sonic=motion_token, heft/scalebfm=ref_qpos, amo=velocity_cmd, almi=twist_ee); a mismatch is caught on the first policy reply.'),
    robot: Optional[str] = typer.Option(None, "--robot",
                                help="--sim: robosuite robot class. Default is the client's "
                                     "(G1FloatingBody). Datasets collected before the two G1 "
                                     "models were merged record 'G1NewFloatingBody' — that name "
                                     "is gone; G1FloatingBody IS that model now."),
    results_json: Optional[str] = typer.Option(None, "--results-json",
                                help="--sim: write the per-episode table (success/steps/seconds) "
                                     "here, for batch sweeps and later comparison."),
    genuine_min_steps: Optional[int] = typer.Option(None, "--genuine-min-steps",
                                help="--sim: report successes shorter than this separately — they "
                                     "usually mean the episode spawned already in the goal state "
                                     "rather than the policy doing the task."),
    token_ema_alpha: Optional[float] = typer.Option(None, "--token-ema-alpha",
                                help="--sim: EMA weight on the first action of each new chunk "
                                     "(1.0 = snap, <1 smooths the seam between chunks)."),
    inference_backend: Optional[str] = typer.Option(None, "--inference-backend",
                                help='--sim: where the WHOLE-BODY CONTROLLER runs (not the VLA): onnx-cpu (default) | onnx-gpu | tensorrt | torch-jit. The controller steps at 50 Hz; on a loaded CPU it is the bottleneck, so onnx-gpu is worth it on a busy box.'),
    fixed_scene: bool = typer.Option(False, "--fixed-scene",
                                help='--sim: rebuild the SAME scene every episode from --scene-seed, as `lab collect --sim` does. --scene-id pins only the FloorPlan, not object sampling or where the robot starts; without this eval spawns the robot where the demos never did. Numbers: docs/sim_validation.md.'),
    scene_seed: Optional[int] = typer.Option(None, "--scene-seed",
                                help="--sim: seed for --fixed-scene. Match the value collection "
                                     "used (lab collect defaults to 0)."),
    reset_jitter: Optional[float] = typer.Option(None, "--reset-jitter",
                                help="--sim: XY jitter radius (m) re-applied to the ROBOT after every reset; use WITH --fixed-scene (alone it freezes the spawn and every episode is one rollout). Pass the collection manifest's jitter.g1_radius (0.05) to match the training start distribution. Why: docs/sim_validation.md."),
    reset_object_jitter: Optional[float] = typer.Option(None, "--reset-object-jitter",
                                help="--sim: same for task objects (manifest jitter.object_radius, "
                                     "typically 0.02)."),
    jitter_seed: Optional[int] = typer.Option(None, "--jitter-seed", min=0,
                                help="--sim: reset jitter uses this seed + zero-based episode index."),
    randomize: Optional[str] = typer.Option(None, "--randomize",
                                help="--sim: eval-time scene randomization for robustness sweeps. "
                                     "Presets off|mild|standard|stress, or comma-separated tokens "
                                     "(textures[=frac] | lights[=jitter] | camera | lens | pbr | rgba | "
                                     "robot | distractors[=N] | exclude=A|B | texture-split= | all); "
                                     "a preset expands in place and later tokens override it. "
                                     "--randomize help prints the full reference. Same syntax as "
                                     "replay_augment / preview_randomization. Some defaults resolve per "
                                     "scene family; the resolved values are echoed at startup and written "
                                     "to --results-json. Deterministic from --randomize-seed. "
                                     "distractors' baseline-restore guard needs --fixed-scene. "
                                     "Example: --randomize standard,distractors=2"),
    randomize_seed: int = typer.Option(0, "--randomize-seed",
                                help="--sim: seed for --randomize; same spec + seed reproduces the same scenes."),
    policy_config: Optional[str] = typer.Option(None, "--policy-config",
                                help="--sim: explicit sim2real policy.yaml for the whole-body controller (bridge runtime only), overriding --backend's default checkpoint. MUST be the variant the data was collected through: a motion_token only means something to its own decoder, and nothing at runtime checks (the run scores zero)."),
    sonic_ckpt: Optional[str] = typer.Option(None, "--sonic-ckpt",
                                help="--sim --backend sonic: which g1 export by name -- low_latency (default) | v1_1; shorthand for --policy-config <g1_backend>/checkpoints/sonic/<variant>/g1/policy.yaml. `lab collect` uses the explicit --policy-config path. Use the variant the dataset's lane names (outputs_*/sonic -> low_latency, outputs_*/sonic_v1_1 -> v1_1)."),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Evaluate <model> in closed loop (currently: --sim via humanoid_benchmark)."""
    from .benchmark import finished_episodes, plan_task_runs, summarize, task_row

    ctx = _ctx()
    opts = dict(model_path=model_path, run_dir=run_dir, ckpt_step=ckpt_step,
                base_model=base_model, trt=trt,
                trt_engine=trt_engine, embodiment_tag=embodiment_tag, prompt=prompt, env_name=env_name,
                episodes=episodes, max_steps=max_steps, action_horizon=action_horizon, render=render,
                render_scale=render_scale,
                video_dir=video_dir, port=port, sim_env=sim_env, action_space=action_space,
                t5_mode=t5_mode, inference_rate=inference_rate, scene_id=scene_id,
                backend=backend,
                robot=robot, results_json=results_json,
                genuine_min_steps=genuine_min_steps, token_ema_alpha=token_ema_alpha,
                inference_backend=inference_backend,
                fixed_scene=fixed_scene, scene_seed=scene_seed,
                reset_jitter=reset_jitter, reset_object_jitter=reset_object_jitter,
                jitter_seed=jitter_seed,
                randomize=randomize, randomize_seed=randomize_seed,
                policy_config=policy_config, sonic_ckpt=sonic_ckpt,
                config_name=config_name)
    runs = plan_task_runs(typer_ctx, model, task, task_set, opts)
    for _, kw in runs:
        ckpt_name = kw.pop("sonic_ckpt")
        if ckpt_name is not None:
            kw["policy_config"] = _resolve_sonic_ckpt(ctx, ckpt_name, backend=kw["backend"],
                                                      policy_config=kw["policy_config"])
    if check or not sim:
        from locomanilab.common import sim as simmod
        first = runs[0][1]
        support = simmod.detect(ctx, conda_env=sim_env, backend=first["backend"],
                                policy_config=first["policy_config"])
        typer.echo(support.summary())
        if check:
            raise typer.Exit(0 if support.ok else 1)
        if not sim:
            typer.secho("pass --sim to run closed-loop sim eval", fg=typer.colors.YELLOW)
            raise typer.Exit(1)
    if watch:                            # live on-screen viewer instead of saved mp4
        render = "viewer"
        disp = os.environ.get("DISPLAY")
        if not dry_run and not disp:
            typer.secho("--watch needs an on-screen display, but DISPLAY is unset.", fg=typer.colors.RED)
            typer.echo("  run it on the machine with a monitor/VNC (in that machine's own terminal),")
            typer.echo("  X11-forward with `ssh -X`, or drop --watch to save an mp4 (--render video --video-dir <dir>).")
            raise typer.Exit(2)
        typer.secho(f"[watch] a live MuJoCo window will open on DISPLAY={disp or '?'}; "
                    "run in the FOREGROUND (not nohup/&). Close the window or Ctrl-C to stop.",
                    fg=typer.colors.CYAN)
    plugin, _ = _plugin(model, dry_run)
    if not task_set:
        bench_task, kw = runs[0]
        kw["render"] = render
        if bench_task is not None:
            typer.echo(f"[task] {bench_task.name}: {bench_task.type}/{bench_task.split}, "
                       f"{kw['episodes']} episodes, results -> {kw['results_json']}")
        _finish(plugin.eval(**kw))
        return
    rows = []
    for i, (bench_task, kw) in enumerate(runs, 1):
        kw["render"] = render
        results = plugin.ctx.anchor(kw["results_json"])
        head = f"[task {i}/{len(runs)}] {bench_task.name}"
        if not dry_run and finished_episodes(results) >= kw["episodes"]:
            typer.echo(f"{head}: results exist, skipped ({results})")
            rows.append(task_row(bench_task, results, ok=True))
            continue
        typer.secho(f"{head}: {bench_task.type}/{bench_task.split}", fg=typer.colors.CYAN)
        res = plugin.eval(**kw)
        typer.secho(f"{head}: [{'OK' if res.ok else 'FAIL'}] {res.message}",
                    fg=typer.colors.GREEN if res.ok else typer.colors.RED)
        rows.append(task_row(bench_task, results, ok=res.ok, message=res.message))
    if dry_run:
        typer.secho(f"\n[OK] dry-run of {len(runs)} tasks", fg=typer.colors.GREEN)
        return
    root = plugin.ctx.anchor(video_dir or f"runs/eval/{model}")
    summary = summarize(rows, root / "summary.json")
    for name, g in summary["groups"].items():
        typer.echo(f"  {name:28s} tasks={g['tasks']:<4d} mean success rate={100 * g['mean_success_rate']:.1f}%")
    failed = [r["task"] for r in rows if not r["ok"]]
    _finish(StageResult(ok=not failed, message=(f"{len(rows)} tasks; summary -> {root / 'summary.json'}"
                                                + (f"; failed: {', '.join(failed)}" if failed else ""))))


@app.command("infer-check")
def infer_check(
    model: str = typer.Option(..., "--model", "-M"),
    config_name: Optional[str] = typer.Option(None, "--config-name", help="Model recipe matching the checkpoint and dataset."),
    dataset: Path = typer.Option(..., "--dataset", "-d", help="training LeRobot dataset to replay"),
    episode: int = typer.Option(0, "--episode", help="ordinal index into the dataset's episodes"),
    frames: Optional[int] = typer.Option(None, "--frames", help="cap frames (default: whole episode)"),
    out: Optional[str] = typer.Option(None, "--out", help="output plot png (default: runs/infer_check/<model>_ep<N>.png)"),
    prompt: str = typer.Option("test", "--prompt", "-p"),
    model_path: Optional[str] = typer.Option(None, "--model-path", help="gr00t model dir"),
    run_dir: Optional[str] = typer.Option(None, "--run-dir", help="model run/checkpoint directory"),
    ckpt_step: Optional[int] = typer.Option(None, "--ckpt-step",
                                help="fastwam: load this training step's step_*.pt from "
                                     "<run-dir>/checkpoints/weights instead of the newest one."),
    base_model: Optional[str] = typer.Option(None, "--base-model"),
    trt: bool = typer.Option(False, "--trt", help="gr00t: use TRT engine"),
    embodiment_tag: Optional[str] = typer.Option(None, "--embodiment-tag"),
    port: int = typer.Option(5550, "--port"),
    sim_env: str = typer.Option("tele_benchmark", "--sim-env", help="conda env for the offline client"),
    action_space: Optional[str] = typer.Option(None, "--action-space",
                                      help="gr00t: which action the model was trained on: motion_token | smpl "
                                           "(picks the server's embodiment tag, default motion_token; GT compared "
                                           "against the matching head)."),
    t5_mode: Optional[str] = typer.Option(None, "--t5-mode",
                                help="fastwam only (default encode): encode = T5 encodes the prompt at startup; "
                                     "cache = read embeds from <run-dir>/cache_t5 (T5 never loaded)."),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Offline open-loop check: run <model> on a training dataset and plot predicted vs GT actions.

    Starts the model's gr00t-compat server, replays dataset frames through get_action, and saves a
    per-dimension pred-vs-GT plot. Works for every model that has a sim eval server (any checkpoint)."""
    plugin, _ = _plugin(model, dry_run)
    _finish(plugin.infer_check(dataset=dataset, episode=episode, frames=frames, out=out,
                               prompt=prompt, model_path=model_path, run_dir=run_dir,
                               ckpt_step=ckpt_step,
                               base_model=base_model, trt=trt, embodiment_tag=embodiment_tag,
                               port=port, sim_env=sim_env, action_space=action_space,
                               t5_mode=t5_mode, config_name=config_name))
