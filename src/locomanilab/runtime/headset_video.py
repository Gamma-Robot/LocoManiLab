#!/usr/bin/env python3
"""Stream a local image source to a PICO headset, in place of the robot's OrinVideoSender.

The headset's XRoboToolkit client does not care where its video comes from: it
opens ONE TCP control connection to whatever address is configured in the app,
sends ``OPEN_CAMERA`` carrying the resolution/bitrate it wants plus its own
receive address, and then reads length-prefixed H.264 access units on a second
connection that the sender opens back to it. The robot's Orin binary
(``~/XRoboToolkit-Orin-Video-Sender``, ``main_zed_tcp.cpp``) is just one
implementation of that contract, welded to the ZED SDK and to Jetson's
``nvv4l2h264enc``.

This is the same contract re-implemented for the collection PC, so the headset
can show:

  * the sim's first-person window during ``lab collect --sim`` -- no robot
    needs to be powered on at all; or
  * any X11 window on the desktop.

Point the PICO app's video source at THIS machine and run this instead. The
protocol is byte-for-byte the one the Orin speaks (see PROTOCOL below), so
nothing changes on the headset except the IP.

Two sources:

    --source x11:...   grab a desktop window (zero changes anywhere else, but
                       an occluded or minimised window grabs the wrong pixels)
    --source zmq:...   subscribe to the sim's ego-frame publisher, which speaks
                       sim2real.camera.messages (the robot camera wire format),
                       so `g1_bridge.camera_monitor --connect <same endpoint>`
                       can watch the very same feed

Scaling, letterboxing and the side-by-side duplication are all done by ffmpeg's
filtergraph -- this file never touches a pixel, it only speaks the protocol and
cuts the encoder's output into whole frames.

PROTOCOL (reverse-engineered from main_zed_tcp.cpp / network_helper.hpp)
-----------------------------------------------------------------------
Control, headset -> us, TCP (port only; the Orin binds INADDR_ANY and ignores
the IP in its own --listen argument, so we do too)::

    [4B big-endian bodyLength][body]
    body = [int32 LE cmdLen][cmd][int32 LE dataLen][data]
    cmd  = "OPEN_CAMERA" | "CLOSE_CAMERA"
    data = CA FE                         magic
           01                            protocol version
           int32 LE x 7                  width height fps bitrate
                                         enableMvHevc renderMode port
           [uint8 len][bytes]            camera   (the Unity client sends "ZEDMINI")
           [uint8 len][bytes]            ip       (where the headset wants the video)

Video, us -> headset, TCP to (ip, port) from that payload::

    [4B big-endian length][one H.264 Annex-B access unit]      repeated

Usage::

    python -m locomanilab.runtime.headset_video --source x11:auto
    python -m locomanilab.runtime.headset_video --source 'zmq:tcp://127.0.0.1:5557'
    python -m locomanilab.runtime.headset_video --source x11:auto --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

# The port the headset dials. Same default as the Orin's --listen.
DEFAULT_PORT = 13579
# Fallbacks for a headset that sends a payload we cannot parse. These are the
# Orin's own --send defaults (main_zed_tcp.cpp:920), i.e. what its stereo ZED
# stream looks like: 2560x720 is TWO 1280x720 eyes side by side.
DEFAULT_SIZE = (2560, 720)
DEFAULT_FPS = 30
DEFAULT_BITRATE = 4_000_000
# The sim's ego window title (demo_empty_teleop.py: EgoCameraPreview). Its live
# title gets " -- <n> FPS" appended, so `--source x11:auto` matches a substring.
EGO_WINDOW_NEEDLE = "G1 ego view"

_MAGIC = b"\xca\xfe"
_AUD = b"\x00\x00\x01\x09"          # access unit delimiter, our frame boundary

#: Set by the signal handler. Waiting loops poll it so Ctrl-C is never swallowed
#: by a long wait (notably X11Source waiting for a window that never appears).
SHUTDOWN = threading.Event()


def log(msg: str) -> None:
    print(f"[headset-video] {msg}", flush=True)


# --------------------------------------------------------------------------
# Protocol
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraRequest:
    """The headset's OPEN_CAMERA payload."""

    width: int
    height: int
    fps: int
    bitrate: int
    enable_mv_hevc: int
    render_mode: int
    port: int
    camera: str
    ip: str

    def sanitised(self, args) -> "CameraRequest":
        """Replace nonsense fields with the CLI fallbacks.

        The headset is the authority on what it wants, but a zero width would
        build an ffmpeg command that cannot run, and a blank ip has nowhere to
        connect -- better a working stream at the default size than none.
        """
        w, h = (self.width, self.height)
        if w <= 0 or h <= 0:
            w, h = args.size
        return CameraRequest(
            width=w, height=h,
            fps=self.fps if self.fps > 0 else args.fps,
            bitrate=self.bitrate if self.bitrate > 0 else args.bitrate,
            enable_mv_hevc=self.enable_mv_hevc,
            render_mode=self.render_mode,
            port=self.port, camera=self.camera, ip=self.ip,
        )


def _read_compact_string(data: bytes, off: int) -> tuple[str, int]:
    """One [uint8 length][bytes] string, as C# BinaryWriter-lite writes them."""
    if off >= len(data):
        raise ValueError("truncated string length")
    n = data[off]
    off += 1
    if n == 0:
        return "", off
    if off + n > len(data):
        raise ValueError("truncated string body")
    return data[off:off + n].decode("utf-8", "replace"), off + n


def parse_camera_request(data: bytes) -> CameraRequest:
    if len(data) < 10:
        raise ValueError(f"payload too small ({len(data)} bytes)")
    if data[:2] != _MAGIC:
        raise ValueError(f"bad magic {data[:2].hex()} (want cafe)")
    version = data[2]
    if version != 1:
        raise ValueError(f"unsupported protocol version {version}")
    if len(data) < 31:
        raise ValueError("payload too small for the integer block")
    ints = struct.unpack_from("<7i", data, 3)
    camera, off = _read_compact_string(data, 3 + 28)
    ip, _ = _read_compact_string(data, off)
    return CameraRequest(*ints, camera=camera, ip=ip)


def parse_protocol(body: bytes) -> tuple[str, bytes]:
    """``[int32 LE cmdLen][cmd][int32 LE dataLen][data]`` -> (command, data)."""
    if len(body) < 8:
        raise ValueError(f"body too small ({len(body)} bytes)")
    (cmd_len,) = struct.unpack_from("<i", body, 0)
    if cmd_len < 0 or 4 + cmd_len > len(body):
        raise ValueError(f"bad command length {cmd_len}")
    command = body[4:4 + cmd_len].decode("utf-8", "replace").split("\0", 1)[0]
    off = 4 + cmd_len
    if off + 4 > len(body):
        raise ValueError("body too small for the data length")
    (data_len,) = struct.unpack_from("<i", body, off)
    off += 4
    if data_len < 0 or off + data_len > len(body):
        raise ValueError(f"bad data length {data_len}")
    return command, body[off:off + data_len]


# --------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------

class Source:
    """Where the pixels come from. Only ever produces ffmpeg input arguments."""

    needs_stdin = False

    def input_args(self, req: CameraRequest) -> list[str]:
        raise NotImplementedError

    def attach(self, stdin, stop: threading.Event) -> None:
        """Called once per session for sources that push into ffmpeg's stdin."""

    def describe(self) -> str:
        return self.__class__.__name__


def _window_tree(display: str) -> str:
    return subprocess.run(
        ["xwininfo", "-display", display, "-root", "-tree"],
        capture_output=True, text=True, timeout=15, check=True,
    ).stdout


# `  0x2343060 "title": ("class" "Class")  800x600+94+54  +94+54`
_TREE_LINE = re.compile(
    r'^\s*(0x[0-9a-fA-F]+)\s+"([^"]*)".*?\s(\d+)x(\d+)\+-?\d+\+-?\d+')

#: Below this, a "window" is a placeholder, not something worth streaming --
#: gnome-shell parks several 1x1 windows on the root, and every real window has
#: a 1x1 child of its own in the tree.
_MIN_WINDOW_PX = 32


def find_window(display: str, needle: str) -> tuple[int, str] | None:
    """Best mapped window whose title contains ``needle``, or None.

    ``xwininfo -name`` needs the title verbatim, which is no use here: the ego
    preview appends a live FPS readout to its own title. So walk the tree.
    (xdotool would be the obvious tool and is NOT installed on this box.)

    Returns None rather than raising: the caller usually wants to WAIT, because
    the window it is after does not exist yet (see wait_for_window).
    """
    try:
        tree = _window_tree(display)
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"WARNING: could not list windows on {display}: {exc}")
        return None
    matches = []
    for line in tree.splitlines():
        m = _TREE_LINE.match(line)
        if not m or needle.lower() not in m.group(2).lower():
            continue
        if int(m.group(3)) < _MIN_WINDOW_PX or int(m.group(4)) < _MIN_WINDOW_PX:
            continue
        matches.append((int(m.group(1), 16), m.group(2)))
    if not matches:
        return None
    # A title that STARTS with the needle beats one that merely contains it:
    # the ego window is "G1 ego view (robot0_head_camera) -- 12.3 FPS", while
    # the terminal you launched this from may well have the same words in the
    # middle of its own title.
    matches.sort(key=lambda m: not m[1].lower().startswith(needle.lower()))
    if len(matches) > 1:
        log(f"NOTE: {len(matches)} windows match {needle!r}; using the first")
        for wid, title in matches:
            log(f"    {wid:#x}  {title}")
    return matches[0]


def wait_for_window(display: str, needle: str, timeout: float) -> tuple[int, str]:
    """Block until a window matching ``needle`` exists.

    This is the whole point of `--source x11:auto` being usable from locomanilab: the
    sender is started BEFORE the simulator (it has to be listening when the
    headset dials in), and the sim's ego window does not exist until the first
    stepped frame -- which is after the operator's first keypress. Failing at
    startup, which is what this used to do, meant x11:auto could never be
    automated.

    ``timeout`` of 0 waits forever. Ctrl-C always gets out (SHUTDOWN).
    """
    found = find_window(display, needle)
    if found:
        return found
    log(f"waiting for a window matching {needle!r} on {display} "
        f"({'no timeout' if not timeout else f'up to {timeout:.0f}s'}) ...")
    deadline = time.monotonic() + timeout if timeout else None
    last_note = time.monotonic()
    while not SHUTDOWN.is_set():
        if SHUTDOWN.wait(2.0):
            break
        found = find_window(display, needle)
        if found:
            log(f"window appeared: {found[0]:#x} {found[1]!r}")
            return found
        now = time.monotonic()
        if now - last_note >= 30:
            log(f"still waiting for {needle!r} ...")
            last_note = now
        if deadline and now >= deadline:
            raise LookupError(
                f"no window matching {needle!r} on {display} after "
                f"{timeout:.0f}s. Is the sim running, and has it started "
                f"stepping? (the ego window opens on the first frame)")
    raise LookupError("shutting down")


class X11Source(Source):
    """Grab a desktop window with ffmpeg's x11grab.

    Caveat worth knowing before you debug a black headset: X11 hands out what is
    ON SCREEN. A window that is minimised, or covered by another window, grabs
    the wrong pixels. Keep it visible, or use the zmq source.
    """

    def __init__(self, spec: str, display: str, wait: float = 0.0):
        self.display = display
        self.wait = wait
        self.window_id: int | None = None
        self.region: str | None = None
        self.needle: str | None = None      # set -> resolved per session, not now
        self.title = ""
        if spec in ("auto", ""):
            self.needle = EGO_WINDOW_NEEDLE
        elif spec == "root":
            pass                                        # whole screen
        elif spec.startswith("name="):
            self.needle = spec[5:]
        elif spec.startswith("region="):
            self.region = spec[7:]                      # WxH+X+Y
        else:
            self.window_id = int(spec, 16 if spec.lower().startswith("0x") else 10)

    def resolve(self) -> None:
        """Find the window, waiting for it if necessary.

        Deliberately NOT done in __init__: the sender outlives any one sim run,
        and a window id does not. Re-resolving per session means restarting the
        simulator does not require restarting this.
        """
        if self.needle is None:
            return
        self.window_id, self.title = wait_for_window(
            self.display, self.needle, self.wait)

    def input_args(self, req: CameraRequest) -> list[str]:
        self.resolve()
        args = ["-f", "x11grab", "-draw_mouse", "0",
                "-framerate", str(req.fps)]
        if self.window_id is not None:
            args += ["-window_id", hex(self.window_id)]
        grab = self.display
        if self.region:
            m = re.fullmatch(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)", self.region)
            if not m:
                raise SystemExit("[headset-video] --source x11:region= wants WxH+X+Y")
            w, h, x, y = m.groups()
            args += ["-video_size", f"{w}x{h}"]
            grab = f"{self.display}+{x},{y}"
        args += ["-i", grab]
        return args

    def describe(self) -> str:
        if self.window_id is not None:
            return f"x11 window {self.window_id:#x} {self.title!r} on {self.display}"
        if self.needle is not None:
            return (f"x11 window titled ~{self.needle!r} on {self.display} "
                    f"(looked up when the headset opens the camera)")
        if self.region:
            return f"x11 region {self.region} on {self.display}"
        return f"x11 root window on {self.display}"


class ZmqSource(Source):
    """Subscribe to a camera PUB socket and pace its JPEGs into ffmpeg.

    The wire format is sim2real.camera.messages: one msgpack map per tick,
    ``{"timestamps": {name: float}, "images": {name: <JPEG bytes>}}``. That is
    what the sim's EgoFramePublisher sends and what the robot's own camera
    server sends, so the same endpoint can be watched with
    ``python -m g1_bridge.camera_monitor --connect <endpoint>``.

    Frames are REPEATED at the requested rate when the publisher is slower (the
    sim runs at 20 Hz, the headset usually asks for 30): a constant feed keeps
    the encoder emitting, which keeps the periodic IDR coming, which is what
    lets a headset that reconnects mid-session get a picture straight away.
    """

    needs_stdin = True

    def __init__(self, endpoint: str, key: str):
        self.endpoint = endpoint
        self.key = key
        self._latest: bytes | None = None
        self._lock = threading.Lock()
        self._frames = 0

    def input_args(self, req: CameraRequest) -> list[str]:
        return ["-f", "image2pipe", "-c:v", "mjpeg",
                "-framerate", str(req.fps), "-i", "pipe:0"]

    def attach(self, stdin, stop: threading.Event) -> None:
        threading.Thread(target=self._subscribe, args=(stop,),
                         name="zmq-sub", daemon=True).start()
        threading.Thread(target=self._pace, args=(stdin, stop),
                         name="zmq-pace", daemon=True).start()

    def _subscribe(self, stop: threading.Event) -> None:
        # Imported here, not at module scope: the x11 source must keep working
        # in any interpreter, and pyzmq/msgpack only live in the sim env.
        try:
            import msgpack
            import zmq
        except ImportError as exc:                       # pragma: no cover
            log(f"ERROR: the zmq source needs pyzmq + msgpack ({exc}). "
                f"Run this script with the sim conda env's python.")
            stop.set()
            return
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVHWM, 2)                   # keep only fresh frames
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(self.endpoint)
        log(f"subscribed to {self.endpoint} (key {self.key!r})")
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        try:
            while not stop.is_set():
                if not poller.poll(200):
                    continue
                payload = msgpack.unpackb(sock.recv(), raw=False)
                image = payload.get("images", {}).get(self.key)
                if isinstance(image, (bytes, bytearray)):
                    with self._lock:
                        self._latest = bytes(image)
                        self._frames += 1
        except Exception as exc:                         # pragma: no cover
            log(f"zmq subscriber stopped: {type(exc).__name__}: {exc}")
        finally:
            sock.close(0)

    def _pace(self, stdin, stop: threading.Event) -> None:
        """Write the newest frame at a steady rate, repeating when starved."""
        period = 1.0 / 30.0
        next_tick = time.monotonic()
        warned = False
        while not stop.is_set():
            with self._lock:
                frame = self._latest
            if frame is None:
                if not warned:
                    log("waiting for the first frame from the publisher ...")
                    warned = True
            else:
                if warned:
                    log("first frame received")
                    warned = False
                try:
                    stdin.write(frame)
                    stdin.flush()
                except (BrokenPipeError, ValueError, OSError):
                    return
            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()

    def describe(self) -> str:
        return f"zmq {self.endpoint} key={self.key!r}"


def build_source(spec: str, args) -> Source:
    kind, _, rest = spec.partition(":")
    if kind == "x11":
        return X11Source(rest, args.display, wait=args.wait_window)
    if kind == "zmq":
        return ZmqSource(rest or "tcp://127.0.0.1:5557", args.key)
    raise SystemExit(f"[headset-video] unknown --source {spec!r}; "
                     f"want x11:... or zmq:tcp://host:port")


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------

def build_ffmpeg_cmd(source: Source, req: CameraRequest, args) -> list[str]:
    """Mirror the Orin's GStreamer pipeline (buildPipelineString) with ffmpeg.

    Orin                                   here
    ----                                   ----
    appsrc BGRA WxH                        the source's input args
    (ZED already delivers side-by-side)    scale + pad + split/hstack
    nvv4l2h264enc bitrate=<payload>        h264_nvenc -rc cbr -b:v <payload>
    idrinterval=15                         -g 15 -forced-idr 1
    insert-sps-pps=true                    -bsf dump_extra=freq=keyframe
    h264parse ! appsink (one buf = one AU) -bsf h264_metadata=aud=insert, then
                                           we cut on the AUDs ourselves
    """
    eye_w = max(2, req.width // 2)
    # force_original_aspect_ratio + pad: the ego view is 4:3 and each eye is
    # 16:9, so it gets letterboxed rather than stretched into a funhouse mirror.
    vf = (f"scale={eye_w}:{req.height}:force_original_aspect_ratio=decrease,"
          f"pad={eye_w}:{req.height}:-1:-1:color=black,"
          f"split=2[a][b];[a][b]hstack=inputs=2")

    hevc = bool(req.enable_mv_hevc)
    codec = args.encoder
    if codec == "auto":
        codec = ("hevc_nvenc" if hevc else "h264_nvenc") if args.nvenc else \
                ("libx265" if hevc else "libx264")
    bsf = "hevc_metadata=aud=insert" if hevc else "h264_metadata=aud=insert"

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", args.ffmpeg_loglevel, "-nostdin"]
    cmd += source.input_args(req)
    cmd += ["-vf", vf, "-c:v", codec]
    if codec.endswith("_nvenc"):
        # p1/ll/zerolatency: the operator is steering a robot by this picture,
        # so latency beats quality at equal bitrate.
        cmd += ["-preset", "p1", "-tune", "ll", "-zerolatency", "1", "-delay", "0"]
    else:
        cmd += ["-preset", "ultrafast", "-tune", "zerolatency"]
    cmd += ["-rc", "cbr"] if codec.endswith("_nvenc") else []
    cmd += ["-b:v", str(req.bitrate), "-maxrate", str(req.bitrate),
            "-bufsize", str(max(req.bitrate // 2, 1)),
            "-g", "15", "-forced-idr", "1", "-bf", "0",
            "-pix_fmt", "yuv420p",
            "-bsf:v", f"{bsf},dump_extra=freq=keyframe",
            "-f", "hevc" if hevc else "h264", "pipe:1"]
    return cmd


class AUFramer:
    """Cut an Annex-B byte stream into access units on the AUD (NAL type 9).

    Every AU we emit starts at a start code immediately followed by an AUD, so
    the headset gets exactly one frame per length-prefixed packet -- the same
    granularity the Orin's appsink callback produces.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    @staticmethod
    def _aud_starts(buf: bytes) -> list[int]:
        starts, i = [], 0
        while (j := buf.find(_AUD, i)) >= 0:
            # A 4-byte start code (00 00 00 01) means the AU begins one earlier.
            starts.append(j - 1 if j > 0 and buf[j - 1] == 0 else j)
            i = j + len(_AUD)
        return starts

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buf += chunk
        starts = self._aud_starts(self._buf)
        if len(starts) < 2:
            return []
        out = [bytes(self._buf[a:b]) for a, b in zip(starts, starts[1:])]
        del self._buf[:starts[-1]]
        return out


class Session:
    """One OPEN_CAMERA..CLOSE_CAMERA lifetime: encoder + socket to the headset."""

    def __init__(self, source: Source, req: CameraRequest, args) -> None:
        self.source, self.req, self.args = source, req, args
        self.stop = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.sock: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._sent = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> bool:
        # Resolve the source BEFORE dialling the headset: an x11 source may sit
        # here for minutes waiting for the sim's window, and holding an open
        # video socket that sends nothing all that time just looks like a hang
        # at the headset end.
        try:
            cmd = build_ffmpeg_cmd(self.source, self.req, self.args)
        except LookupError as exc:
            log(f"cannot start this session: {exc}")
            return False
        if not self._connect():
            return False
        log("ffmpeg: " + " ".join(cmd))
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if self.source.needs_stdin else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=None,
            )
        except OSError as exc:
            log(f"ERROR: could not start ffmpeg: {exc}")
            self.close()
            return False
        if self.source.needs_stdin:
            self.source.attach(self.proc.stdin, self.stop)
        t = threading.Thread(target=self._pump, name="pump", daemon=True)
        t.start()
        self._threads.append(t)
        return True

    def _connect(self) -> bool:
        """Dial the headset, retrying like the Orin's initialize_sender()."""
        for attempt in range(1, 11):
            if self.stop.is_set():
                return False
            try:
                self.sock = socket.create_connection(
                    (self.req.ip, self.req.port), timeout=5)
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock.settimeout(None)
                log(f"video connection up -> {self.req.ip}:{self.req.port}")
                return True
            except OSError as exc:
                log(f"connect to {self.req.ip}:{self.req.port} failed "
                    f"({attempt}/10): {exc}")
                time.sleep(1.0)
        log("giving up on the headset's video port")
        return False

    def _pump(self) -> None:
        framer = AUFramer()
        save = open(self.args.save, "wb") if self.args.save else None
        assert self.proc is not None and self.proc.stdout is not None
        try:
            while not self.stop.is_set():
                chunk = self.proc.stdout.read(65536)
                if not chunk:
                    log("ffmpeg output ended")
                    break
                for au in framer.feed(chunk):
                    if save:
                        save.write(au)
                    self._send(au)
                    self._sent += 1
                    if self._sent % 300 == 0:
                        log(f"sent {self._sent} frames")
        except Exception as exc:                         # pragma: no cover
            # A teardown we asked for closes the socket under this thread; that
            # is not news. Only an unexpected break is worth a line.
            if not self.stop.is_set():
                log(f"streaming stopped: {type(exc).__name__}: {exc}")
        finally:
            if save:
                save.close()
            # Never take the whole process down with the stream: the Orin
            # catches send errors per frame and keeps its control server up, so
            # the operator can just re-open the camera from the headset.
            self.stop.set()

    def _send(self, au: bytes) -> None:
        assert self.sock is not None
        self.sock.sendall(struct.pack(">I", len(au)) + au)

    def close(self) -> None:
        self.stop.set()
        if self.proc is not None:
            # Terminate BEFORE closing the pipes: close them first and ffmpeg
            # spends its dying breath printing "Broken pipe" at the operator.
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()
            for pipe in (self.proc.stdin, self.proc.stdout):
                try:
                    if pipe:
                        pipe.close()
                except OSError:
                    pass
            self.proc = None
        if self.sock is not None:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
            self.sock = None
        for t in self._threads:
            t.join(timeout=2)
        self._threads.clear()
        log(f"session closed ({self._sent} frames sent)")


# --------------------------------------------------------------------------
# Control server
# --------------------------------------------------------------------------

class ControlServer:
    """Accept the headset's control connection and drive sessions from it."""

    def __init__(self, port: int, source: Source, args) -> None:
        self.port, self.source, self.args = port, source, args
        self.session: Session | None = None
        self._running = True

    def serve_forever(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # INADDR_ANY, exactly like the Orin's TCPServer -- which is why the IP
        # half of its --listen argument has never actually done anything.
        try:
            srv.bind(("0.0.0.0", self.port))
        except OSError as exc:
            srv.close()
            # By far the usual cause: an earlier sender outlived its session
            # (a killed `lab collect` cannot run its own cleanup). Say so,
            # with the command that finds it -- a bare traceback here just
            # looks like the script is broken.
            raise SystemExit(
                f"[headset-video] cannot listen on port {self.port}: {exc}\n"
                f"  Something already holds it -- most likely a sender left over "
                f"from an earlier session.\n"
                f"  Find it:  ss -ltnp | grep :{self.port}\n"
                f"  Stop the process holding this port before restarting.\n"
                f"  Or pick another port with --listen <port> (the PICO app must "
                f"then dial that port too)."
            ) from None
        srv.listen(1)
        srv.settimeout(1.0)
        log(f"listening for the headset on 0.0.0.0:{self.port}")
        log(f"source: {self.source.describe()}")
        try:
            while self._running:
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                log(f"headset control connection from {addr[0]}:{addr[1]}")
                try:
                    self._handle(conn)
                finally:
                    conn.close()
                    self._stop_session("control connection closed")
        finally:
            srv.close()

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(1.0)
        buf = bytearray()
        while self._running:
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                continue
            except OSError as exc:
                log(f"control connection error: {exc}")
                return
            if not chunk:
                log("headset disconnected")
                return
            buf += chunk
            # The Orin reads one recv() and assumes it holds a whole message.
            # Do it properly instead: a length-prefixed stream reader costs ten
            # lines and cannot be split by TCP at an inconvenient boundary.
            while len(buf) >= 4:
                (body_len,) = struct.unpack_from(">I", buf, 0)
                if body_len > 1 << 20:
                    log(f"absurd body length {body_len}; dropping the connection")
                    return
                if len(buf) < 4 + body_len:
                    break
                body = bytes(buf[4:4 + body_len])
                del buf[:4 + body_len]
                self._dispatch(body)

    def _dispatch(self, body: bytes) -> None:
        try:
            command, data = parse_protocol(body)
        except ValueError as exc:
            log(f"undecodable control message ({exc}): {body[:32].hex()}")
            return
        if command == "OPEN_CAMERA":
            self._open(data)
        elif command == "CLOSE_CAMERA":
            self._stop_session("CLOSE_CAMERA")
        else:
            log(f"ignoring unknown command {command!r}")

    def _open(self, data: bytes) -> None:
        try:
            req = parse_camera_request(data).sanitised(self.args)
        except ValueError as exc:
            log(f"bad OPEN_CAMERA payload ({exc}); using CLI fallbacks")
            req = CameraRequest(*self.args.size, self.args.fps, self.args.bitrate,
                                0, 0, 0, "unknown", "")
            if not req.ip:
                log("ERROR: no headset address in the payload and none to guess")
                return
        log(f"OPEN_CAMERA {req.width}x{req.height} @{req.fps} "
            f"{req.bitrate / 1e6:.1f} Mbps hevc={req.enable_mv_hevc} "
            f"render={req.render_mode} camera={req.camera!r} "
            f"-> {req.ip}:{req.port}")
        # The Orin rejects anything whose camera name does not start with "ZED"
        # (main_zed_tcp.cpp:690). We are not a ZED, and neither is the sim, so
        # that check would only ever reject ourselves. Log and carry on.
        self._stop_session("re-opening")
        session = Session(self.source, req, self.args)
        if session.start():
            self.session = session
        else:
            session.close()

    def _stop_session(self, why: str) -> None:
        if self.session is not None:
            log(f"stopping session ({why})")
            self.session.close()
            self.session = None

    def shutdown(self) -> None:
        self._running = False
        self._stop_session("shutting down")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_size(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)x(\d+)", text)
    if not m:
        raise argparse.ArgumentTypeError("want WxH, e.g. 2560x720")
    return int(m.group(1)), int(m.group(2))


def _parse_listen(text: str) -> int:
    """Accept `PORT` or `IP:PORT`; the IP is decorative (see ControlServer)."""
    return int(text.rsplit(":", 1)[-1])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m locomanilab.runtime.headset_video",
        description=__doc__.split("PROTOCOL")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", default="x11:auto",
                   help="x11:auto | x11:0x<id> | x11:name=<substring> | "
                        "x11:region=WxH+X+Y | x11:root | zmq:tcp://host:port "
                        "(default: x11:auto = the sim's ego window)")
    p.add_argument("--listen", type=_parse_listen, default=DEFAULT_PORT,
                   metavar="[IP:]PORT",
                   help=f"control port the headset dials (default {DEFAULT_PORT}). "
                        f"The IP half is accepted and ignored, as on the Orin.")
    p.add_argument("--display", default=os.environ.get("DISPLAY", ":0"),
                   help="X display for the x11 source (default: $DISPLAY)")
    p.add_argument("--key", default="ego_view",
                   help="image key inside the zmq payload (default: ego_view)")
    p.add_argument("--size", type=_parse_size, default=DEFAULT_SIZE, metavar="WxH",
                   help=f"fallback frame size when the headset does not say "
                        f"(default {DEFAULT_SIZE[0]}x{DEFAULT_SIZE[1]}, "
                        f"= two 1280x720 eyes side by side)")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS, help="fallback frame rate")
    p.add_argument("--bitrate", type=int, default=DEFAULT_BITRATE,
                   help="fallback bitrate in bits/s")
    p.add_argument("--encoder", default="auto",
                   help="auto | h264_nvenc | libx264 | hevc_nvenc | libx265")
    p.add_argument("--no-nvenc", dest="nvenc", action="store_false",
                   help="use the software encoder even if NVENC is available")
    p.add_argument("--save", metavar="FILE",
                   help="also write the encoded stream to FILE (debugging: "
                        "`ffplay -f h264 FILE` should play it)")
    p.add_argument("--wait-window", type=float, default=0.0, metavar="SECONDS",
                   help="x11:auto / x11:name=: how long to wait for the window "
                        "when the headset opens the camera. 0 (default) waits "
                        "forever -- the sim's ego window only exists once the "
                        "operator starts stepping, which can be any time after "
                        "this starts")
    p.add_argument("--ffmpeg-loglevel", default="error")
    p.add_argument("--dry-run", action="store_true",
                   help="print the ffmpeg command for a synthetic OPEN_CAMERA "
                        "and exit, without listening or encoding")
    return p


def check_nvenc() -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True).stdout
    return "h264_nvenc" in out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = build_source(args.source, args)

    if args.dry_run:
        req = CameraRequest(*args.size, args.fps, args.bitrate, 0, 0, 12345,
                            "ZEDMINI", "<headset>")
        log(f"source: {source.describe()}")
        log("static plan only: ffmpeg, encoder and display readiness were not checked")
        if isinstance(source, X11Source) and source.needle is not None:
            # Window discovery starts an external program. A static plan uses
            # a placeholder even if a matching window is currently available.
            log(f"(window matching {source.needle!r} resolves at run time; "
                "showing a placeholder id)")
            source.window_id = 0x0
            source.needle = None
        log(f"would answer OPEN_CAMERA {req.width}x{req.height}@{req.fps} with:")
        print(" ".join(build_ffmpeg_cmd(source, req, args)))
        return 0

    if shutil.which("ffmpeg") is None:
        raise SystemExit("[headset-video] ffmpeg is not on PATH")
    if args.nvenc and args.encoder == "auto" and not check_nvenc():
        log("NVENC not available; falling back to libx264")
        args.nvenc = False

    # Tell the operator up front whether the window is there, without making it
    # a precondition: this process is usually started BEFORE the sim.
    if isinstance(source, X11Source) and source.needle is not None:
        found = find_window(source.display, source.needle)
        if found:
            log(f"window {found[0]:#x} {found[1]!r} is up (re-checked per session)")
        else:
            log(f"no window matching {source.needle!r} yet -- will wait for one "
                f"when the headset opens the camera")

    server = ControlServer(args.listen, source, args)

    def _sigint(_sig, _frm):
        log("interrupt; shutting down")
        SHUTDOWN.set()                       # wakes any wait_for_window loop
        server.shutdown()

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)
    try:
        server.serve_forever()
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
