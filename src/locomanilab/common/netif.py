"""Link-state facts about a local network interface.

`lab collect` hands a NIC name to the robot bridge, which opens DDS on it. A
name that merely *exists* is not enough: a USB dongle left unplugged, or a box
rewired onto a different subnet, leaves the interface present in
`/sys/class/net` with the link down and no address. That failure surfaces
minutes later inside a tmux pane, with the collector already up and the
operator watching a different window -- so it is worth catching here.

Linux-only by construction (`/sys/class/net`, SIOCGIFADDR); so is everything
else that talks to the robot.
"""

from __future__ import annotations

import fcntl
import socket
import struct
from pathlib import Path

_SIOCGIFADDR = 0x8915
_NET = Path("/sys/class/net")


def exists(nic: str) -> bool:
    return (_NET / nic).exists()


def operstate(nic: str) -> str:
    """`up` / `down` / `unknown` -- the kernel's own view of the link."""
    try:
        return (_NET / nic / "operstate").read_text().strip()
    except OSError:
        return "unknown"


def ipv4(nic: str) -> str | None:
    """The interface's IPv4 address, or None when it has none."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            packed = fcntl.ioctl(sock.fileno(), _SIOCGIFADDR,
                                 struct.pack("256s", nic.encode()[:15]))
        except OSError:                      # no address, or no such device
            return None
    return socket.inet_ntoa(packed[20:24])


def candidates() -> list[tuple[str, str]]:
    """(name, ipv4) for every addressed interface, loopback and docker aside.

    Ordered as the kernel lists them. Meant for an error message: the operator
    knows which subnet the robot is on, so showing the addresses is enough to
    pick the right NIC without a second command.
    """
    out: list[tuple[str, str]] = []
    for path in sorted(_NET.iterdir()) if _NET.is_dir() else []:
        name = path.name
        if name == "lo" or name.startswith(("docker", "br-", "veth")):
            continue
        addr = ipv4(name)
        if addr:
            out.append((name, addr))
    return out


def check(nic: str) -> str | None:
    """None when `nic` can carry robot traffic, else why it cannot."""
    if not exists(nic):
        problem = "does not exist on this machine"
    elif operstate(nic) == "down":
        problem = "exists but its link is DOWN (cable/dongle unplugged?)"
    elif ipv4(nic) is None:
        problem = f"is {operstate(nic)} but has no IPv4 address"
    else:
        return None
    usable = candidates()
    listing = ("\n  ".join(f"{name}\t{addr}" for name, addr in usable)
               if usable else "(none -- this machine has no addressed NIC)")
    return (f"robot interface {nic!r} {problem}.\n"
            f"Interfaces with an address here:\n  {listing}\n"
            f"Pass --robot-interface <nic> or set env.local deploy.robot_interface "
            f"to the one on the robot's subnet.")
