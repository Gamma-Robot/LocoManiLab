"""Health check + (re)start for the XRoboToolkit PC service (``RoboticsServiceProcess``).

``--pico-source real`` reads the headset through this service: the collector's
``xrobotoolkit_sdk`` talks gRPC to it on 127.0.0.1:60061, and the headset
connects to it on TCP :63901 (the service also broadcasts discovery from that
worker). ``xrt.init()`` therefore succeeds even when no headset can ever
connect -- the only symptom is ``Timed out ... waiting for
xrt.is_body_data_available()`` fifteen seconds later.

The one failure seen in practice: the service inherits ``http_proxy`` /
``all_proxy`` from the launching shell (``~/.zshrc``), Qt's ``QTcpServer``
then resolves a proxy and ``listen`` fails before ``bind`` -- the service log
says ``tcp server listen failed. port: 63901`` and never broadcasts. So the
service must be started with the proxy variables stripped, and a running
instance that is not listening on :63901 must be restarted the same way.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from pathlib import Path

SERVICE_SCRIPT = Path("/opt/apps/roboticsservice/runService.sh")
#: ``comm`` is truncated to 15 chars, so ``pgrep -x RoboticsServiceProcess`` never matches.
SERVICE_COMM = "RoboticsService"
HEADSET_PORT = 63901
SDK_PORT = 60061
PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy", "no_proxy",
              "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def service_pids() -> list[int]:
    try:
        out = subprocess.run(["pgrep", "-x", SERVICE_COMM], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [int(p) for p in out.split()]


def proxy_free_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in PROXY_VARS}


def _stop(pids: list[int], grace_s: float = 3.0) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and service_pids():
        time.sleep(0.2)
    for pid in service_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _start(wait_s: float = 8.0) -> bool:
    subprocess.Popen(["bash", str(SERVICE_SCRIPT)], env=proxy_free_env(),
                     cwd=str(SERVICE_SCRIPT.parent), start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if port_open(HEADSET_PORT) and port_open(SDK_PORT):
            return True
        time.sleep(0.25)
    return False


def ensure_service(*, dry_run: bool = False, echo=print) -> tuple[bool, str]:
    """Make sure the service is up AND listening for the headset on :63901.

    Returns (ok, message). Never touches a healthy service. With ``dry_run``
    only reports what it would do.
    """
    if not SERVICE_SCRIPT.exists():
        return True, f"XRoboToolkit service script not found ({SERVICE_SCRIPT}); skipping check"
    pids = service_pids()
    if pids and port_open(HEADSET_PORT):
        return True, f"XRoboToolkit service ok (pid {pids[0]}, :{HEADSET_PORT} listening)"
    if pids:
        action = (f"XRoboToolkit service (pid {pids[0]}) is running but NOT listening on "
                  f":{HEADSET_PORT} -- the headset cannot connect; restarting it without "
                  "proxy env vars")
    else:
        action = "XRoboToolkit service not running; starting it without proxy env vars"
    if dry_run:
        return True, f"[dry-run] {action}"
    echo(f"[locomanilab] {action}")
    if pids:
        _stop(pids)
    if _start():
        return True, f"XRoboToolkit service started, :{HEADSET_PORT} listening"
    return False, (
        f"XRoboToolkit service did not open :{HEADSET_PORT}; see "
        "~/.local/share/PICOBusinessSuitData/log/<date>.txt ('tcp server listen "
        f"failed' = proxy env leaked in). Manual fix: kill `pgrep -x {SERVICE_COMM}` "
        f"and run `env -u http_proxy -u https_proxy -u all_proxy bash {SERVICE_SCRIPT}`")
