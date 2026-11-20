"""The wire limits of the daemon, on top of the JSONL of `ph.framing` (P5-01).

The framing itself is one JSON object on each line, bounded by `MAX_LINE`. It is
in `ph.framing`, in core, because the tool commands of the session use it too.
This module holds what only the wire of the daemon needs: how large an
attachment one frame may carry, derived from that bound.

@module ph_app.daemon.framing
"""

from __future__ import annotations

__all__ = ["MAX_ATTACHMENT_BYTES", "MAX_ATTACHMENT_SIZE"]

MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
"""The largest attachment that one frame may carry.

The limit is under `ph.framing.MAX_LINE` (8 MiB) with room to spare, because the
frame also carries the method, the session id and the 4/3 expansion of base64. A
limit set at the frame size would refuse *after* the client had already sent the
frame, as a framing error with no name.

The limit is here and not in `server.py` because it is a property of the
**wire**. It is derived from that bound, and both ends of the wire need it: the
daemon to refuse a frame, and a front end to refuse an upload before it builds a
frame. A front end that imported it from the server module would name the module
of the supervisor only to read an integer."""

MAX_ATTACHMENT_SIZE = f"{MAX_ATTACHMENT_BYTES // (1024 * 1024)} MiB"
"""The limit as the daemon and the web server both state it to the user.

The division floors. If the limit stopped being a whole number of MiB, those two
places would otherwise each round it down silently, in two messages.
"""
