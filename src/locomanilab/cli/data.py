"""Data commands."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import typer
from locomanilab.common import dataset as ds
from .app import app
from .common import _ctx, _default_collect_out, _finish, _range_datasets, _replay_sources, _resolve_sonic_ckpt, _workflow


@app.command("init")
def init_workspace(
    directory: Path = typer.Argument(Path("."), help="Directory for a local workspace."),
) -> None:
    """Create a minimal workspace configuration without overwriting existing files."""
    from locomanilab.workspace import absolute_path
    import yaml

    root = absolute_path(directory)
    target = root / "env.local.yaml"
    try:
        root.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            stream.write("# Local configuration. Keep this file out of version control.\n")
            yaml.safe_dump({"schema_version": 2, "role": ["data"], "repos": {},
                            "paths": {}, "deploy": {}, "conda_envs": {}}, stream, sort_keys=False)
    except FileExistsError:
        raise typer.BadParameter(f"configuration already exists: {target}; it was left unchanged")
    except OSError as exc:
        raise typer.BadParameter(f"cannot create workspace {root}: {exc}")
    typer.echo(f"Created {target}\nNext: cd {root}\nThen: lab sample-data --out datasets/example")


@app.command("sample-data")
def sample_data(
    out: Path = typer.Option(Path("datasets/example"), "--out", help="New synthetic dataset directory."),
    episodes: int = typer.Option(2, "--episodes", min=1, max=20),
    frames: int = typer.Option(20, "--frames", min=2, max=500),
) -> None:
    """Generate a tiny synthetic dataset for checking the data pipeline (CPU only)."""
    from locomanilab.data.sample import create_sample

    destination = _ctx().anchor(out)
    try:
        create_sample(destination, episodes=episodes, frames=frames)
    except (OSError, ValueError, RuntimeError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.echo(f"Created synthetic data: {destination}\n"
               f"{episodes} episodes, {episodes * frames} frames. "
               "This sample checks file processing; it is not a robot demonstration.")

@app.command()
def validate(dataset: Path):
    """Validate a LeRobot dataset against the shared contract (exit 1 when it fails).

    Missing meta files / the state column are errors, not warnings: a dataset without
    modality.json or tasks.jsonl cannot be read by any model's loader, so a scripted
    pipeline (or CI) must be able to stop on it."""
    report = ds.validate(dataset)
    typer.echo(report.summary())
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def collect(
    task_prompt: str = typer.Option("test", "--task-prompt"),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help='dataset output dir (default outputs_sim/<lane>/<env>[_fixed_scene<id>]_<timestamp>, or outputs_real/<lane>/real_<...>; _2, _3 appended if taken). <lane> = --backend, except sonic v1_1 uses sonic_v1_1 (the collection default) while low_latency keeps the historical sonic lane; see variants.collection_lane.',
    ),
    camera_host: Optional[str] = typer.Option(None, "--camera-host"),
    sim: bool = typer.Option(False, "--sim",
                             help="collect INSIDE the humanoid_benchmark sim (external "
                                  "submodule) instead of on the real robot"),
    sim_env: str = typer.Option(
        "EmptyEnv",
        "--env",
        help="--sim: environment id (default: asset-free native MuJoCo ground)",
    ),
    backend: str = typer.Option("sonic", "--backend",
                                help="whole-body policy driving the G1 — sonic | amo | "
                                     "heft | scalebfm | almi. All collection uses "
                                     "g1_backend's teleoperation and recording layers"),
    pico_source: str = typer.Option("mock", "--pico-source",
                                    help="mock (no hardware) | real (PICO headset)"),
    real_runtime: str = typer.Option("auto", "--real-runtime",
                                     help="collection runtime — auto | g1_backend. Both select "
                                          "the shared G1 runtime for hardware and simulation"),
    robot_io: Optional[str] = typer.Option(None, "--robot-io",
                                 help="--real-runtime g1_backend: inline (default: this "
                                      "process opens the robot's DDS, no bridge tick of "
                                      "latency) | zmq (a separate scripts/g1/real_bridge.py "
                                      "process; adds 20-40 ms of drifting delay and was "
                                      "measured to make ALMI's gait jitter -- keep for A/B "
                                      "runs only)"),
    dex3: str = typer.Option("double", "--dex3",
                             help="--real-runtime g1_backend: in-process Dex3 hand control — "
                                  "none | left | right | double"),
    camera_connect: Optional[str] = typer.Option(None, "--camera-connect",
                                                 help="--real-runtime g1_backend: sim2real camera "
                                                      "server address (default derived from "
                                                      "--camera-host, port 5594)"),
    camera_port: Optional[int] = typer.Option(None, "--camera-port",
                                              help="--real-runtime g1_backend: camera server port "
                                                   "for --camera-host (default env.local "
                                                   "deploy.camera_port, else 5594; the G1 onboard "
                                                   "server publishes on 5555)"),
    ego_size: Optional[str] = typer.Option(None, "--ego-size",
                                           help="--real-runtime g1_backend: WxH to store ego "
                                                "frames at, or 'native' (default 640x480, the "
                                                "camera's own size; --sim records 320x240)"),
    policy_config: Optional[str] = typer.Option(None, "--policy-config",
                                                help="explicit sim2real policy.yaml, overriding "
                                                     "--backend's default checkpoint"),
    sonic_ckpt: Optional[str] = typer.Option(
        None,
        "--sonic-ckpt",
        help="--backend sonic: checkpoint name (v1_1, the collection default, or "
             "low_latency); shorthand for its g1_backend policy.yaml. Cannot be "
             "combined with --policy-config.",
    ),
    inference_backend: Optional[str] = typer.Option(None, "--inference-backend",
                                                    help="onnx-cpu | onnx-gpu | tensorrt | torch-jit"),
    gmr_viewer: bool = typer.Option(False, "--gmr-viewer",
                                    help="heft/scalebfm: keep the GMR subprocess's viser viewer"),
    headset_source: str = typer.Option("x11", "--headset-source",
                                       help="--headset-video: where the sender picks up the "
                                            "picture. x11 (default) grabs the sim's ego preview "
                                            "window, waiting for it to appear; sim takes the "
                                            "frames straight out of the collector over ZMQ "
                                            "(nothing to keep unobscured, one resampling step "
                                            "fewer)"),
    headset_video: bool = typer.Option(False, "--headset-video",
                                       help="--sim: stream the sim's first-person view to the "
                                            "PICO, so the operator sees what the robot sees "
                                            "without looking away from the headset. Starts "
                                            "the packaged video sender alongside the "
                                            "collector; it speaks the same protocol the robot's "
                                            "OrinVideoSender does, so only the video-source IP in "
                                            "the PICO app changes (point it at this machine). No "
                                            "robot needed"),
    set_: list[str] = typer.Option(None, "--set",
                                   help="--real-runtime g1_backend: override one bridge-spec key "
                                        "(repeatable), e.g. --set sim2real_pose_blend_s=1.0"),
    tmux: bool = typer.Option(True, "--tmux/--no-tmux",
                              help="--real-runtime g1_backend: start everything in one tmux "
                                   "session (window <backend> with controller / GMR teleop / "
                                   "recording monitor / keyboard panes, plus bridge and camera "
                                   "windows) — the default. --no-tmux runs just the collector "
                                   "here and leaves the bridge and camera to you"),
    robot_interface: Optional[str] = typer.Option(None, "--robot-interface",
                                                  help="robot network interface -- the NIC on the "
                                                       "G1's 192.168.123.0/24 (--tmux bridge pane, "
                                                       "--robot-io inline, and Dex3 DDS). Default: "
                                                       "env.local deploy.robot_interface, else eth0. "
                                                       "A wrong one fails in the bridge pane, not here"),
    camera_source: str = typer.Option("realsense", "--camera-source",
                                      help="--tmux: camera server source (realsense | cv2)"),
    camera_runtime: Optional[str] = typer.Option(
        None, "--camera-runtime",
        help="--real-runtime g1_backend, remote camera: which publisher to put on the "
             "robot's camera port before collecting. Two programs can serve it and only "
             "one can hold the ZED, so this both starts one and stops the other. "
             "orin_sender = OrinVideoSender waiting for the headset (default; it opens "
             "the camera only once the headset connects, so the port stays silent until "
             "then) | orin_headless = OrinVideoSender in its direct mode, capturing with "
             "no headset attached (no robot-eye view in the headset) | composed_camera = "
             "the sonic-era composed_camera_server.service, which captures on its own but "
             "opens the ZED by the serial number written in its unit | none = leave the "
             "robot alone. Default: env.local deploy.camera_runtime, else orin_sender"),
    amo_arm_source: Optional[str] = typer.Option(None, "--amo-arm-source",
                                                 help="--sim --backend amo: gmr (retarget your PICO "
                                                      "body skeleton — the default with --pico-source "
                                                      "real) | ik (3-point head+controllers solve)"),
    max_episodes: Optional[int] = typer.Option(None, "--max-episodes",
                                               help="stop after N episodes (default: no limit on "
                                                    "the real robot, 1000 in --sim)"),
    fixed_scene: bool = typer.Option(False, "--fixed-scene", help="--sim: pin the scene layout"),
    scene_seed: Optional[int] = typer.Option(None, "--scene-seed", help="--sim: scene seed"),
    scene_id: Optional[int] = typer.Option(None, "--scene-id",
                                           help="--sim: FloorPlan id for the iTHOR room envs "
                                                "(e.g. 13 kitchen / 209 living room / 329 bedroom "
                                                "/ 401 bathroom); ignored by envs that don't take one"),
    fixed_task: bool = typer.Option(False, "--fixed-task",
                                    help="--sim: collect through robocasa.scripts.collect_fixed_task "
                                         "instead of collect_demos_g1 — scene AND initial task state "
                                         "stay fixed, each reset only jitters the G1 root and the "
                                         "task objects. Writes fixed_task_manifest.json + "
                                         "fixed_task_resets.jsonl next to meta/"),
    jitter_g1: Optional[float] = typer.Option(None, "--jitter-g1",
                                              help="--fixed-task: G1 root XY jitter radius in m "
                                                   "(default 0.05; 0 disables)"),
    jitter_objects: Optional[float] = typer.Option(None, "--jitter-objects",
                                                   help="--fixed-task: task-object XY jitter radius "
                                                        "in m (default 0.02; 0 disables)"),
    jitter_seed: Optional[int] = typer.Option(None, "--jitter-seed",
                                              help="--fixed-task: seed for the reset-time jitter "
                                                   "(--scene-seed does not control it)"),
    sim_conda_env: str = typer.Option("tele_benchmark", "--sim-conda-env", help="--sim: conda env for the sim"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Collect a dataset — on the real robot (default) or in sim (`--sim`).

    Both paths emit the same LeRobot contract, so `lab prepare` / `convert` / `train`
    consume either unchanged: sim runs humanoid_benchmark, and real collection
    runs g1_backend's own collector — the two share the whole teleop
    layer (`g1_bridge`), so the recorded columns are the same by construction.

    The real g1_backend path expects two other processes you start yourself: the
    camera server (`python -m sim2real.camera.server`) and, with `--robot-io zmq`,
    the robot bridge (`python scripts/g1/real_bridge.py`).

    There is no `--model`: what varies here is the controller (`--backend`), not
    the VLA. Datasets land in
    `outputs_sim/<lane>/<env>[_fixed_scene<scene-id>]_<timestamp>`, or
    `outputs_real/<lane>/real_<timestamp>` on hardware, where <lane> is the backend
    plus the checkpoint variant when it is not the historical low_latency lane
    (`sonic_v1_1`); logs in
    `runs/collect/<backend>-<sim|real>/`."""
    from locomanilab.workflows.collection import CollectionWorkflow
    if problem := CollectionWorkflow.collection_runtime_error(real_runtime):
        typer.secho(problem, fg=typer.colors.RED)
        raise typer.Exit(1)
    plugin, ctx = _workflow(dry_run)
    if sonic_ckpt is not None:
        policy_config = _resolve_sonic_ckpt(
            ctx, sonic_ckpt, backend=backend, policy_config=policy_config
        )
    elif backend == "sonic" and policy_config is None:
        from locomanilab.common.variants import SONIC_COLLECT_DEFAULT_VARIANT

        policy_config = _resolve_sonic_ckpt(
            ctx,
            SONIC_COLLECT_DEFAULT_VARIANT,
            backend=backend,
            policy_config=None,
        )
    if fixed_task:
        # The fixed-task collector pins the scene by construction and takes neither of
        # these flags; passing them would be silently dropped rather than honoured.
        if not sim:
            typer.secho("--fixed-task is a --sim collector", fg=typer.colors.RED)
            raise typer.Exit(1)
        for flag, given in (("--fixed-scene", fixed_scene),
                            ("--amo-arm-source", amo_arm_source is not None)):
            if given:
                typer.secho(f"--fixed-task does not accept {flag} "
                            f"(collect_fixed_task has no such option)", fg=typer.colors.RED)
                raise typer.Exit(1)
    if headset_source not in ("x11", "sim"):
        typer.secho(f"--headset-source must be x11 or sim, not {headset_source!r}",
                    fg=typer.colors.RED)
        raise typer.Exit(1)
    if headset_video and not sim:
        # On hardware the headset already gets the ZED through the Orin
        # (--camera-runtime orin_sender); a second sender on this machine would
        # just be a port the headset is not dialling.
        typer.secho("--headset-video is a --sim option; on the real robot the headset "
                    "video comes from the Orin (see --camera-runtime)",
                    fg=typer.colors.RED)
        raise typer.Exit(1)
    if out is None:
        out = _default_collect_out(ctx, backend=backend, sim=sim,
                                   name=sim_env if sim else "real",
                                   fixed_task=fixed_task, scene_id=scene_id,
                                   policy_config=policy_config)
        typer.echo(f"out: {out}")
    _finish(plugin.collect(task_prompt=task_prompt, out_dir=out, camera_host=camera_host,
                           sim=sim, sim_env=sim_env, backend=backend, pico_source=pico_source,
                           amo_arm_source=amo_arm_source,
                           max_episodes=max_episodes, fixed_scene=fixed_scene,
                           scene_seed=scene_seed, scene_id=scene_id, fixed_task=fixed_task,
                           jitter_g1=jitter_g1, jitter_objects=jitter_objects,
                           jitter_seed=jitter_seed, sim_conda_env=sim_conda_env,
                           real_runtime=real_runtime, robot_io=robot_io, dex3=dex3,
                           camera_connect=camera_connect, camera_port=camera_port, ego_size=ego_size,
                           policy_config=policy_config, inference_backend=inference_backend,
                           gmr_viewer=gmr_viewer, headset_video=headset_video,
                           headset_source=headset_source,
                           overrides=set_ or [], tmux=tmux,
                           robot_interface=robot_interface, camera_source=camera_source,
                           camera_runtime=camera_runtime))


@app.command()
def replay(
    source_runs: Optional[list[Path]] = typer.Argument(None,
                                    help="run dirs to replay — original collections OR "
                                         "runs `lab augment` produced (both carry the "
                                         "fixed-task sidecars this needs)"),
    start: Optional[str] = typer.Option(None, "--start",
                                        help="first run of a range, by TIMESTAMP "
                                             "(same selection as `lab prepare --start`)"),
    end: Optional[str] = typer.Option(None, "--end", help="last run of the range, inclusive"),
    view: bool = typer.Option(False, "--view",
                              help="watch it in the MuJoCo window (needs a display). "
                                   "Implies --no-write unless you pass --write"),
    write: bool = typer.Option(False, "--write",
                               help="ALSO write an output run. Off by default — replay "
                                    "is for checking, not producing, and a replay of an "
                                    "un-randomized scene is just the source again. Only "
                                    "meaningful together with --randomize; use "
                                    "`lab augment` when producing data is the point"),
    randomize: str = typer.Option("off", "--randomize",
                                  help="'off' (default) replays the scene as recorded. "
                                       "Give a spec to replay it in a randomized scene "
                                       "instead — same syntax as `lab augment` / "
                                       "`lab eval`, which is what `lab augment` is: "
                                       "this command with --randomize standard"),
    episodes: str = typer.Option("all", "--episodes", help="'all' or a comma list of indices"),
    reroll: str = typer.Option("0", "--reroll", metavar="N",
                               help="re-roll the texture/light randomization every N "
                                      "EPISODES: 0 (default) = never, one coherent room for "
                                      "the whole run; 1 = every episode; N = every N. Camera "
                                      "/ lens / robot tint / distractor placement are "
                                      "per-episode regardless"),
    variants: int = typer.Option(1, "--variants",
                                 help="only meaningful with --randomize: output runs per source"),
    seed: int = typer.Option(0, "--seed"),
    out_base: Optional[Path] = typer.Option(None, "--out-base",
                                            help="parent dir for the output runs "
                                                 "(default: the sibling '<lane>_replay' lane)"),
    keep_failed: bool = typer.Option(False, "--keep-failed",
                                     help="ALSO write episodes whose replay did not reach "
                                          "success (debug only — their video no longer "
                                          "matches the copied labels)"),
    object_textures: bool = typer.Option(False, "--randomize-object-textures",
                                         help="ALSO re-texture the task objects. Off by "
                                              "default: the instruction still names the "
                                              "original object, so this trades "
                                              "instruction-appearance grounding for colour "
                                              "robustness"),
    distractor_groups: Optional[str] = typer.Option(None, "--distractor-groups",
                                                    help="comma list the `distractors` token "
                                                         "draws from"),
    preroll_ticks: Optional[int] = typer.Option(None, "--preroll-ticks"),
    model: Optional[str] = typer.Option(None, "--model", "-M", hidden=True),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Replay a collected run — to check it, or to generate augmented data.

    Feeds each recorded episode's actions back through the backend that collected
    it. Plain, it answers "does this run still reach success?" after a controller
    or scene change; episodes that no longer reach `_check_success` are reported
    (and dropped from any output run).

    Nothing is written by default — a replay answers a question. Add
    `--randomize <spec> --write` to turn it into DATA GENERATION: the same
    episodes replayed in randomized scenes, labels preserved, written to a
    sibling `<lane>_replay` lane. `--view` watches it happen in the MuJoCo
    window instead."""
    plugin, ctx = _workflow(dry_run)
    if model:
        typer.echo("Note: replay is model-independent; --model is deprecated.")
    if write and randomize.strip() in ("", "off"):
        typer.secho("--write with --randomize off would write a copy of the source run "
                    "(same labels, re-rendered videos). Give a spec — e.g. "
                    "--randomize standard --write — to make the output worth having.",
                    fg=typer.colors.RED)
        raise typer.Exit(1)
    runs = _replay_sources(ctx, source_runs, start, end)
    _finish(plugin.replay(source_runs=runs, randomize=randomize,
                   variants=variants, seed=seed, episodes=episodes,
                   reroll=reroll, out_base=out_base,
                   object_textures=object_textures,
                   distractor_groups=distractor_groups,
                   preroll_ticks=preroll_ticks, keep_failed=keep_failed,
                   viewer=view, write=write))


@app.command()
def prepare(
    datasets: Optional[list[Path]] = typer.Argument(None),
    start: Optional[str] = typer.Option(None, "--start",
                                        help="first dataset of a range, by TIMESTAMP (dir name, "
                                             "bare timestamp or its prefix, or a path — a path "
                                             "scans its parent, otherwise outputs_sim/, "
                                             "outputs_real/ and their per-backend subdirs)"),
    end: Optional[str] = typer.Option(None, "--end",
                                      help="last dataset dir of the range, inclusive (name, prefix, or path)"),
    out: Optional[Path] = typer.Option(None, "--out"),
    merge_only: bool = typer.Option(False, "--merge-only",
                                    help="skip discard/SMPL/hand/smooth: merge already-prepared "
                                         "datasets bit-identically (re-smoothing would alter data)"),
    terminal_hold: bool = typer.Option(False, "--terminal-hold",
                                       help="append frozen terminal frames to every episode; "
                                            "disabled by default"),
    hold_frames: int = typer.Option(64, "--hold-frames", min=1,
                                    help="frames appended per episode when --terminal-hold is set"),
    tasks: Optional[list[str]] = typer.Option(None, "--task",
                                              help="override each source dataset's task prompt at merge "
                                                   "time (repeat once per dataset, in dataset order; "
                                                   "sources are never modified)"),
    model: Optional[str] = typer.Option(None, "--model", "-M", hidden=True),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Preprocess / merge raw collected datasets (locomanilab/data/prepare.py).

    Works for every backend: which stages apply (stale-frame cleaning, hand open/close
    labels, smoothing) is read from each dataset's own meta/modality.json, and merging
    is refused across backends (meta/info.json `collection_info`). The output carries
    only the columns all sources share.

    Pass dataset dirs explicitly, or use --start/--end to merge every timestamped
    dir in the range (either bound may be omitted to leave that side open)."""
    plugin, ctx = _workflow(dry_run)
    if model:
        typer.echo("Note: prepare is model-independent; --model is deprecated.")
    if (start or end) and datasets:
        typer.secho("pass either explicit dataset dirs or --start/--end, not both", fg=typer.colors.RED)
        raise typer.Exit(1)
    if start or end:
        datasets = _range_datasets(ctx, start, end)
        typer.echo(f"selected {len(datasets)} dataset(s):")
        for d in datasets:
            typer.echo(f"  {d}")
    if not datasets:
        typer.secho("no datasets given: pass dirs or --start/--end", fg=typer.colors.RED)
        raise typer.Exit(1)
    # Dataset inputs are relative to the caller, independent of the helper's cwd.
    datasets = [ctx.input_path(d) for d in datasets]
    missing = [d for d in datasets if not d.is_dir()]
    if missing:
        typer.secho("dataset dir(s) not found: " + " ".join(map(str, missing)), fg=typer.colors.RED)
        raise typer.Exit(1)
    if tasks and len(tasks) != len(datasets):
        typer.secho(f"--task count ({len(tasks)}) must equal dataset count ({len(datasets)})",
                    fg=typer.colors.RED)
        raise typer.Exit(1)
    if out is not None:
        out = ctx.anchor(out) if ctx.schema_version >= 2 else ctx.input_path(out)
    _finish(plugin.prepare(datasets=datasets, out=out, merge_only=merge_only, tasks=tasks,
                           hold_frames=hold_frames if terminal_hold else 0))

