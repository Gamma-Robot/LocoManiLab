"""Model-independent simulator and robot collection orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from locomanilab.plugin import StageResult
from locomanilab.common.sim import runtime_checkout_error, runtime_pythonpath
from locomanilab.workflows.devices import DeviceWorkflow


class CollectionWorkflow(DeviceWorkflow):
    @staticmethod
    def collection_runtime_error(real_runtime: str) -> str | None:
        if real_runtime == "gr00t_wbc":
            return ("--real-runtime gr00t_wbc no longer supports collection. "
                    "Use --real-runtime auto or g1_backend; collection uses g1_backend "
                    "for both simulation and hardware. GR00T-WBC is supported only "
                    "for real-robot deployment.")
        if real_runtime not in {"auto", "g1_backend"}:
            return f"unknown --real-runtime {real_runtime!r}; choose auto or g1_backend"
        return None

    def collect(self, *, task_prompt: str, out_dir: Path, camera_host: str | None = None,
                low_latency: bool = True, sim: bool = False, sim_env: str = "EmptyEnv",
                sim_conda_env: str = "tele_benchmark", backend: str = "sonic",
                pico_source: str = "mock",
                max_episodes: int | None = None, real_runtime: str = "auto",
                **opts: Any) -> StageResult:
        if problem := self.collection_runtime_error(real_runtime):
            return StageResult(ok=False, message=problem)
        if opts.get("policy_config") is not None:
            opts["policy_config"] = str(self.ctx.input_path(opts["policy_config"]))
        if sim:
            return self._collect_sim(task_prompt=task_prompt, out_dir=out_dir,
                                     sim_env=sim_env, sim_conda_env=sim_conda_env,
                                     backend=backend, pico_source=pico_source,
                                     max_episodes=max_episodes, **opts)
        return self._collect_real_g1_backend(
            task_prompt=task_prompt, out_dir=out_dir, camera_host=camera_host,
            backend=backend, pico_source=pico_source, max_episodes=max_episodes,
            **opts)

    def _collect_real_g1_backend(self, *, task_prompt: str, out_dir: Path,
                                 camera_host: str | None, backend: str,
                                 pico_source: str, max_episodes: int | None,
                                 robot_io: str | None = None, dex3: str = "double",
                                 camera_connect: str | None = None,
                                 camera_port: int | None = None,
                                 ego_size: str | None = None,
                                 policy_config: str | None = None,
                                 inference_backend: str | None = None,
                                 gmr_viewer: bool = False,
                                 amo_arm_source: str | None = None,
                                 overrides: list[str] | None = None,
                                 tmux: bool = False, robot_interface: str | None = None,
                                 camera_source: str = "realsense",
                                 camera_runtime: str | None = None,
                                 orin_ssh_host: str = "",
                                 orin_listen: str = "",
                                 **opts: Any) -> StageResult:
        """Real-robot teleop collection through g1_backend's own collector.

        Same whole-body controller, same reference/mode-manager path and the same
        LeRobot schema as `--sim` (both hosts share `g1_bridge`), so a real dataset
        and a sim one differ only in the columns a simulator alone can fill.

        locomanilab stays thin: this shells out `python -m g1_bridge.collect_real` and
        does not start the operator's other processes (the camera server, and with
        `--robot-io zmq` the robot bridge) -- those are separate terminals by
        design, since each has its own failure mode to watch.
        """
        repo = self.ctx.repo_path("g1_backend")
        if not repo.exists():
            return StageResult(ok=False, message=(
                "real collection needs the g1_backend submodule: run "
                f"`git submodule update --init external/g1_backend` ({repo})"))
        if not self.runner.dry_run and (problem := runtime_checkout_error(repo)):
            return StageResult(ok=False, message=problem)
        # Inline Unitree DDS avoids an additional unsynchronized bridge loop.
        # Keep --robot-io zmq available when a separate bridge is required.
        if robot_io is None:
            robot_io = "inline"
        out_dir = self.ctx.anchor(out_dir).resolve()   # the collector requires an ABSOLUTE --out
        connect = camera_connect or self._camera_connect(camera_host, camera_port)
        # The NIC facing the robot is a property of the machine, not of the run:
        # eth0 on the original collection box, a USB dongle's enx… name elsewhere.
        # Nothing in the collector can check it up front -- the bridge pane just
        # fails to open DDS -- so it belongs in env.local with the camera address.
        robot_interface = (robot_interface or self.ctx.deploy.get("robot_interface")
                           or "eth0")
        if not self.runner.dry_run:
            # Existence alone does not mean the NIC can carry DDS: a USB dongle
            # that is present but unplugged, or a box rewired onto another
            # subnet, passes that test and then fails in the bridge pane long
            # after the operator has stopped watching this terminal.
            from locomanilab.common import netif
            problem = netif.check(robot_interface)
            if problem:
                return StageResult(ok=False, message=problem)
        if pico_source == "real":
            # The headset reaches the collector only through the XRoboToolkit PC
            # service. A service that is up but not listening on :63901 (proxy env
            # from the launching shell leaked into it) looks fine to xrt.init() and
            # fails fifteen seconds later as a body-data timeout -- check it here.
            from locomanilab.common import xrt_service
            ok, msg = xrt_service.ensure_service(dry_run=self.runner.dry_run)
            print(f"[locomanilab] {msg}")
            if not ok:
                return StageResult(ok=False, message=msg)
        if camera_runtime is None:
            # Unset on the command line -> the machine decides. Which publisher a
            # robot runs is a property of that robot, and `none` belongs to a box
            # whose camera is already managed elsewhere -- neither is worth
            # retyping every run.
            camera_runtime = str(self.ctx.deploy.get("camera_runtime", "orin_sender"))
        if camera_runtime not in self.CAMERA_RUNTIMES and camera_runtime != "none":
            return StageResult(ok=False, message=(
                f"unknown camera runtime {camera_runtime!r}; expected "
                f"{sorted(self.CAMERA_RUNTIMES) + ['none']} "
                f"(--camera-runtime, or env.local deploy.camera_runtime)"))
        if camera_runtime != "none" and not self._camera_is_local(connect):
            self._bring_up_robot_camera(repo, connect, camera_runtime,
                                        orin_ssh_host, orin_listen)
        cmd = ["python", "-m", "g1_bridge.collect_real"]
        if tmux:
            # The launcher builds the tmux session (bridge / camera / GMR panes)
            # and runs the collector in pane 0; everything after `--` is the
            # collector's own argv, so this stays one argument list.
            cmd = ["python", "-m", "g1_bridge.launch_collect_real",
                   "--interface", robot_interface]
            if robot_io == "inline":
                cmd.append("--no-bridge")   # inline mode has no separate bridge process
            if connect:
                cmd += ["--camera-connect", connect]
            if self._camera_is_local(connect):
                # Only serve when the camera is attached here. Remote (the usual
                # case: the server runs on the robot) leaves the camera window as
                # what it is meant to be -- a viewer on the link.
                cmd += ["--camera-server", "--camera-source", camera_source]
            cmd.append("--")               # everything after this is the collector's argv
        cmd += ["--backend", backend, "--pico-source", pico_source,
                "--task-prompt", task_prompt, "--out", str(out_dir),
                "--robot-io", robot_io, "--robot-interface", robot_interface,
                "--dex3", dex3]
        if connect:
            cmd += ["--camera-connect", connect]
        if ego_size:
            cmd += ["--ego-size", ego_size]
        if max_episodes is not None:
            cmd += ["--max-episodes", str(max_episodes)]
        if policy_config:
            cmd += ["--policy-config", policy_config]
        if inference_backend:
            cmd += ["--inference-backend", inference_backend]
        if gmr_viewer:
            cmd.append("--gmr-viewer")
        if amo_arm_source:
            cmd += ["--amo-arm-source", amo_arm_source]
        for override in overrides or []:
            cmd += ["--set", override]
        # Which interpreter has sim2real + g1_bridge + lerobot differs per machine:
        # a robot/onboard box uses g1_backend's own uv project (upstream's `uv run`),
        # while a box that also runs the sim already has them in that conda env.
        # env.local `conda_envs.g1_backend` picks the latter.
        conda_env = self.ctx.conda_envs.get("g1_backend")
        if conda_env:
            return self.runner.run(cmd, cwd=repo, env_kind="conda", env_name=conda_env,
                                   setup=self.env_setup("conda"),
                                   label="collect", stage=f"{backend}-real")
        return self.runner.run(cmd, cwd=repo, env_kind="uv",
                               setup=self.env_setup("uv"),
                               label="collect", stage=f"{backend}-real")

    def _collect_sim(self, *, task_prompt: str, out_dir: Path, sim_env: str,
                     sim_conda_env: str, backend: str, pico_source: str,
                     max_episodes: int | None,
                     scene_seed: int | None = None, fixed_scene: bool = False,
                     scene_id: int | None = None, fixed_task: bool = False,
                     jitter_g1: float | None = None, jitter_objects: float | None = None,
                     jitter_seed: int | None = None,
                     amo_arm_source: str | None = None,
                     policy_config: str | None = None,
                     inference_backend: str | None = None,
                     gmr_viewer: bool = False,
                     headset_video: bool = False,
                     headset_source: str = "x11",
                     **opts: Any) -> StageResult:
        """Collect demos INSIDE the humanoid_benchmark sim (external submodule).

        Same LeRobot output contract as real-robot collection, so `lab prepare` / `convert` /
        `train` consume it unchanged. `backend` picks the whole-body policy driving the G1
        (sonic/amo, or heft/scalebfm which additionally need the GMR venv from
        `scripts/setup.sh g1_backend`); `pico_source` is `mock` (no hardware) or
        `real` (PICO headset). The sim's own default is `real`, so always pass it through.

        `amo_arm_source` (amo only) picks what drives the arms: `gmr` retargets the
        operator's PICO body skeleton, `ik` solves 3-point head+controllers. Left as
        None the sim decides — gmr with `--pico-source real`, ik otherwise.

        `policy_config` / `inference_backend` / `gmr_viewer` are forwarded the same way
        the real path forwards them. They used to be swallowed here, which silently
        pinned sim collection to each backend's default checkpoint on CPU inference —
        fine while every backend had exactly one, not once a family ships variants or
        wants the GPU.

        `headset_video` puts the ego view in the operator's PICO: locomanilab runs
        the packaged headset-video sender beside the collector to encode the
        first-person frames to the headset. No robot is involved -- the
        real-robot equivalent is the Orin's OrinVideoSender, which this replaces
        for sim sessions. `headset_source` picks x11 (grab the ego preview
        window; the sender waits for it to appear) or sim (frames straight out
        of the collector over LOCOMANILAB_EGO_STREAM).

        `fixed_task` swaps the collector for `robocasa.scripts.collect_fixed_task`:
        same teleop layer and same LeRobot output, but the scene AND the initial task
        state are pinned, and every reset only jitters the G1 root / task objects
        instead of resampling the scene. It writes two sidecars (fixed_task_manifest
        .json, fixed_task_resets.jsonl) into the dataset root. `jitter_g1` /
        `jitter_objects` are radii in metres, where 0 means "disable that jitter" —
        the collector spells those as separate --no-*-jitter flags.
        """
        sim_repo = self.ctx.sim_repo()
        if not sim_repo.exists():
            return StageResult(ok=False, message=(
                "sim collect needs the humanoid_benchmark submodule: "
                f"run `git submodule update --init external/humanoid_benchmark` ({sim_repo})"))
        runtime = self.ctx.repo_path("g1_backend")
        if not self.runner.dry_run and (problem := runtime_checkout_error(runtime)):
            return StageResult(ok=False, message=problem)
        out_dir = self.ctx.anchor(out_dir).resolve()   # the script requires an ABSOLUTE --out
        script = ("robocasa.scripts.collect_fixed_task" if fixed_task
                  else "robocasa.scripts.collect_demos_g1")
        cmd = ["python", "-m", script,
               "--env", sim_env, "--out", str(out_dir),
               "--task-prompt", task_prompt,
               "--backend", backend, "--pico-source", pico_source]
        if max_episodes is not None:
            cmd += ["--max-episodes", str(max_episodes)]
        if scene_seed is not None:
            cmd += ["--scene-seed", str(scene_seed)]
        if scene_id is not None:
            cmd += ["--scene-id", str(scene_id)]
        if fixed_task:
            # radius 0 is not a radius the collector accepts — it is its own flag.
            if jitter_g1 is not None:
                cmd += ["--no-g1-jitter"] if jitter_g1 == 0 else \
                       ["--g1-jitter-radius", str(jitter_g1)]
            if jitter_objects is not None:
                cmd += ["--no-object-jitter"] if jitter_objects == 0 else \
                       ["--object-jitter-radius", str(jitter_objects)]
            if jitter_seed is not None:
                cmd += ["--g1-jitter-seed", str(jitter_seed)]
        else:
            if fixed_scene:
                cmd.append("--fixed-scene")
            if amo_arm_source is not None:
                cmd += ["--amo-arm-source", amo_arm_source]
        if policy_config:
            cmd += ["--policy-config", policy_config]
        if inference_backend:
            cmd += ["--inference-backend", inference_backend]
        if gmr_viewer:
            cmd.append("--gmr-viewer")
        extra_env = {"MUJOCO_GL": "egl", "PYTHONPATH": runtime_pythonpath(
            sim_repo=sim_repo, runtime_repo=runtime)}
        sender = None
        if headset_video:
            if headset_source not in self.HEADSET_SOURCES:
                return StageResult(ok=False, message=(
                    f"unknown --headset-source {headset_source!r} "
                    f"({' | '.join(self.HEADSET_SOURCES)})"))
            endpoint = str(self.ctx.deploy.get("ego_stream") or self.EGO_STREAM_DEFAULT)
            # Only the zmq source needs the in-process publisher; leaving the env
            # var unset for x11 keeps that socket from being bound for nothing.
            if headset_source == "sim":
                extra_env["LOCOMANILAB_EGO_STREAM"] = endpoint
                # Existing user-owned simulator checkouts only read the legacy name.
                extra_env["SOMA_EGO_STREAM"] = endpoint
            sender = self._start_headset_video(headset_source, endpoint, sim_conda_env)
        try:
            return self.runner.run(cmd, cwd=sim_repo, env_kind="conda", env_name=sim_conda_env,
                                   setup=self.env_setup("conda"),
                                   extra_env=extra_env,
                                   label="collect",
                                   stage=f"{backend}-sim{'-fixed' if fixed_task else ''}")
        finally:
            self._stop_headset_video(sender)
