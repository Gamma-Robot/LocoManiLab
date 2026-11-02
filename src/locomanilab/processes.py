"""Stop only subprocess groups created by locomanilab with start_new_session=True."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from contextlib import contextmanager


def stop_process_group(proc: subprocess.Popen, *, grace: float = 3) -> None:
    # A leader can exit before its workers. Its PID is still the group's ID, so
    # do not return early merely because poll()/wait() reports the leader's exit.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


@contextmanager
def managed_process(proc: subprocess.Popen):
    """Clean up owned children on normal exit, errors, Ctrl-C, TERM, and HUP."""
    previous = {}
    if threading.current_thread() is threading.main_thread():
        def terminated(signum, _frame):
            raise SystemExit(128 + signum)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, terminated)
    try:
        yield proc
    finally:
        try:
            stop_process_group(proc)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
