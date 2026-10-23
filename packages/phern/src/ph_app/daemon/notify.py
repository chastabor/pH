"""`sd_notify`, the four lines of it this daemon speaks (P13-12).

A `Type=notify` unit waits for `READY=1` before it calls the service started, ends
it when `WATCHDOG=1` stops arriving inside `WatchdogSec`, and reads `STOPPING=1` as
an orderly exit beginning. All three are a datagram on the socket `NOTIFY_SOCKET`
names, and that is the whole protocol — so this is stdlib rather than a dependency,
and a no-op anywhere the variable is unset, which is every start that is not
systemd's.

**It never raises.** A watchdog that could not be fed because the socket went away
is systemd's to notice, and the daemon has its own work to do.

@module ph_app.daemon.notify
"""

from __future__ import annotations

import logging
import os
import socket

__all__ = ["notify", "watchdog_interval"]

log = logging.getLogger("ph_app.daemon")


def notify(state: str) -> bool:
    """Send one `sd_notify` state. `False` when nothing is listening or it failed."""
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return False
    # An abstract-namespace socket is spelled with a leading `@` in the variable and
    # a leading NUL on the wire.
    target = "\0" + address[1:] if address.startswith("@") else address
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(target)
            sock.sendall(state.encode("utf-8"))
    except OSError:
        log.warning("ph_app.daemon: could not notify %s of %r", address, state, exc_info=True)
        return False
    return True


def watchdog_interval() -> float | None:
    """Seconds between `WATCHDOG=1`s, or `None` when systemd asked for none.

    Half of `WATCHDOG_USEC`, which is what systemd's own documentation recommends,
    so one missed beat is not the end. `WATCHDOG_PID`, when set, names the process
    expected to feed it; a daemon that is not it stays quiet.
    """
    usec = os.environ.get("WATCHDOG_USEC")
    if not usec:
        return None
    expected = os.environ.get("WATCHDOG_PID")
    if expected and expected != str(os.getpid()):
        return None
    try:
        return int(usec) / 2_000_000
    except ValueError:
        return None
