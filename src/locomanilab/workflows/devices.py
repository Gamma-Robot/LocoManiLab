"""Camera and headset process helpers used by collection and local deployment."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from locomanilab.plugin import StageResult
from locomanilab.workflows.context import WorkflowContext


class DeviceWorkflow(WorkflowContext):
    CAMERA_RUNTIMES = {"orin_sender": "listen", "orin_headless": "headless", "composed_camera": "composed"}
    EGO_STREAM_DEFAULT = "tcp://127.0.0.1:5557"
    HEADSET_SOURCES = ("x11", "sim")

    def gr00t_wbc_sonic_args(self, wbc: Path, policy_config: str | None, *,
                             low_latency: bool = True) -> "list[str] | StageResult":
        """Flags that pick the SONIC controller for GR00T-WBC hardware deployment.

        `policy_config` is the g1_backend manifest `--sonic-ckpt` / `--policy-config`
        resolved to; here only its variant matters, mapped onto GR00T-WBC's own
        deploy assets (variants.GR00T_WBC_DEPLOY_ASSETS). GR00T and FastWAM's
        real-robot launchers take the same `--deploy-checkpoint /
        --deploy-obs-config` pair. Choose the variant that produced the training
        data collected through g1_backend. Without a manifest, preserve the
        launcher behaviour: `--low-latency` when `low_latency` (the launcher preset),
        else nothing (the C++ deploy's own default, which on this checkout is
        low_latency too).
        """
        from locomanilab.common import variants as variantmod
        variant = variantmod.sonic_ckpt_variant(policy_config) if policy_config else None
        if policy_config and variant not in variantmod.GR00T_WBC_DEPLOY_ASSETS:
            return StageResult(ok=False, message=(
                f"the gr00t_wbc runtime only knows the named sonic exports "
                f"{sorted(variantmod.GR00T_WBC_DEPLOY_ASSETS)}; got {policy_config!r} "
                f"(variant {variant!r}). Use --sonic-ckpt <name> matching the "
                "controller variant that produced the training data."))
        if variant is None:
            return ["--low-latency"] if low_latency else []
        if variant == "low_latency":
            return ["--low-latency"]
        files = variantmod.gr00t_wbc_deploy_files(wbc, variant)
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            return StageResult(ok=False, message=(
                f"gr00t_wbc deploy files for sonic {variant} are missing:\n  "
                + "\n  ".join(missing)
                + f"\nfetch them with `python download_from_hf.py --sonic-v1-1 --no-planner` "
                f"in {wbc} (or copy model_encoder/model_decoder.onnx from "
                "external/g1_backend/checkpoints/sonic/<variant>/g1/)"))
        ckpt, obs = variantmod.GR00T_WBC_DEPLOY_ASSETS[variant]
        print(f"[locomanilab] gr00t_wbc sonic variant: {variant} -> {ckpt}_{{encoder,decoder}}.onnx + {obs} "
              "(first run builds the TensorRT engines next to the onnx files; allow a few minutes)")
        return ["--deploy-checkpoint", ckpt, "--deploy-obs-config", obs]

    def _camera_connect(self, camera_host: str | None, camera_port: int | None) -> str | None:
        """ZMQ address of the camera server, or None to let the collector default.

        The port is not one number across the fleet: sim2real's own server binds
        5594, while the G1's onboard server (the one locomanilab has always pointed
        `deploy.camera_host` at) publishes on 5555 -- same msgpack protocol, so the
        only thing that has to travel with the host is its port. `deploy.camera_connect`
        in env.local pins the whole address when a machine has a fixed camera.
        """
        if not camera_host:
            return self.ctx.deploy.get("camera_connect")
        if "://" in camera_host:
            return camera_host
        if ":" in camera_host:                       # host:port
            return f"tcp://{camera_host}"
        port = camera_port or int(self.ctx.deploy.get("camera_port", 5594))
        return f"tcp://{camera_host}:{port}"

    @staticmethod
    def _camera_is_local(connect: str | None) -> bool:
        """Whether a camera address points at this machine."""
        if not connect:
            return True                      # the collector's own default is localhost
        host = connect.split("://")[-1].rsplit(":", 1)[0].strip("[]")
        return host in ("127.0.0.1", "localhost", "*", "0.0.0.0", "::1")

    def _bring_up_robot_camera(self, repo: Path, connect: str | None, runtime: str,
                               ssh_host: str = "", listen: str = "") -> None:
        """Put a camera publisher on the robot's camera port before collecting.

        Two programs on the G1 can publish there and only one can have the ZED:
        `composed_camera_server.service` (the sonic-era publisher, `enabled`, so
        it returns on every Orin reboot -- "I stopped it yesterday" is not a
        state that survives) and OrinVideoSender. `runtime` picks one; the script
        stops whatever else holds the camera, idempotently.

        Deliberately best-effort: a failure here is reported and collection
        proceeds. The operator may already have the camera up in another window,
        and the collector's own camera pane shows a dead link plainly. Refusing
        to launch over it would cost more sessions than it saves.

        env.local `deploy.orin_ssh_host` selects the robot; `deploy.orin_listen`
        selects the headset endpoint. The sudo password is read from
        $G1_SSH_PASSWORD by the script
        itself and never passes through locomanilab.
        """
        mode = self.CAMERA_RUNTIMES[runtime]
        script = repo / "scripts" / "g1" / "start_orin_camera.sh"
        if not script.is_file():
            print(f"[locomanilab] NOTE: {script} not found -- start the camera yourself")
            return
        host = ssh_host or self.ctx.deploy.get("orin_ssh_host")
        if not host:
            print("[locomanilab] NOTE: set deploy.orin_ssh_host in env.local.yaml "
                  "to start the robot camera; no camera SSH command was run")
            return
        listen_addr = listen or self.ctx.deploy.get("orin_listen")
        if mode == "listen" and not listen_addr:
            print("[locomanilab] NOTE: set deploy.orin_listen in env.local.yaml "
                  "to the robot's address:port on the headset network; "
                  "no camera SSH command was run")
            return
        # The sender binds a port; `connect` is an address to dial. Reuse only the
        # port number so a bind endpoint never inherits the robot's host.
        port = 5555
        if connect:
            tail = connect.rsplit(":", 1)[-1]
            if tail.isdigit():
                port = int(tail)
        cmd = [str(script), "--mode", mode, "--host", host,
               "--zmq", f"tcp://*:{port}"]
        if listen_addr:
            cmd += ["--listen", listen_addr]
        if self.runner.dry_run:
            # --dry-run means "show me what would happen"; sshing into the robot
            # to stop a service and spawn a process is not that.
            print(f"[locomanilab] would run: {' '.join(cmd)}")
            return
        try:
            rc = subprocess.run(cmd).returncode
        except OSError as exc:                      # not executable, missing shell
            print(f"[locomanilab] WARNING: camera bring-up could not run: {exc}")
            return
        if rc != 0:
            print("[locomanilab] WARNING: camera bring-up failed -- collection will have "
                  "no images unless something else publishes on that port")

    def _start_headset_video(self, source: str, endpoint: str, sim_conda_env: str):
        """Put the sim's first-person view in the PICO, with no robot involved.

        The real robot streams the ZED to the headset from the Orin
        (OrinVideoSender). Nothing did that for the sim, so the operator had to
        look away from the headset at a Tk window on the desktop. This starts
        locomanilab's own sender, which speaks the same protocol to the headset.

        `source` picks where the pixels come from:
          * `x11` grabs the sim's own ego preview window. Nothing has to exist
            yet -- the sender waits for the window, which is the point: it must
            be listening before the sim starts, and that window does not appear
            until the operator steps the first frame.
          * `sim` takes the frames out of the collector's process over ZMQ
            (`endpoint`, which is why `_collect_sim` sets LOCOMANILAB_EGO_STREAM). No
            window to keep unobscured, and one resampling step fewer.

        Runs in the sim conda env: the `sim` source needs pyzmq + msgpack and
        the locomanilab CLI env has neither. Best-effort like the robot camera
        bring-up -- a failure here costs the headset view, not the session.
        Returns the Popen (or None), which the caller must terminate.
        """
        from locomanilab.runtime import script_path
        script = script_path("headset_video.py")
        if not script.is_file():
            print(f"[locomanilab] NOTE: {script} not found -- no headset video")
            return None
        listen = str(self.ctx.deploy.get("headset_listen") or "13579")
        spec = "x11:auto" if source == "x11" else f"zmq:{endpoint}"
        if source == "x11" and not os.environ.get("DISPLAY"):
            print("[locomanilab] WARNING: --headset-source x11 needs a DISPLAY; "
                  "run locomanilab from a desktop session or use --headset-source sim")
        cmd = ["python", str(script), "--source", spec, "--listen", listen]
        argv = self.server_argv(cmd, env_kind="conda", env_name=sim_conda_env)
        if self.runner.dry_run:
            print(f"[locomanilab] would run: {' '.join(argv)}")
            return None
        print(f"[locomanilab] headset video sender: {' '.join(cmd)}")
        try:
            return subprocess.Popen(argv)
        except OSError as exc:
            print(f"[locomanilab] WARNING: headset video sender could not start: {exc}")
            return None

    @staticmethod
    def _stop_headset_video(proc) -> None:
        """SIGTERM, then SIGKILL. The sender holds a listening port, so leaving
        it behind makes the NEXT session fail to bind rather than this one."""
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
