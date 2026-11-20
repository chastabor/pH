"""`sd_notify`: the three states of it that this daemon sends (P13-12).

A `Type=notify` unit waits for `READY=1` before it treats the service as started.
It ends the service when `WATCHDOG=1` stops arriving inside `WatchdogSec`. It
reads `STOPPING=1` as the start of an orderly exit. All three are a datagram on
the socket that `NOTIFY_SOCKET` names, and that is the whole protocol. This
module therefore uses the standard library and not a dependency. It does nothing
where the variable is unset, which is every start that systemd did not make.

**It never raises.** If the watchdog could not be fed because the socket went
away, systemd is the one to notice, and the daemon has its own work to do.

@module ph_app.daemon.notify
"""

from __future__ import annotations

import logging
import os
import socket

__all__ = ["notify", "watchdog_interval"]

log = logging.getLogger("ph_app.daemon")


def notify(state: str) -> bool:
    """Send one `sd_notify` state. Return `False` when nothing is listening or the
    send failed."""
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
    """Return the seconds between `WATCHDOG=1` messages, or `None` when systemd
    asked for none.

    The interval is half of `WATCHDOG_USEC`, which the documentation of systemd
    recommends, so that one missed beat does not end the service. `WATCHDOG_PID`,
    when set, names the process that is expected to feed the watchdog. A daemon
    that is not that process stays quiet.
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
