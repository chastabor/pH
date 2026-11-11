"""The daemon's wire limits, over `ph.framing`'s JSONL (P5-01).

The framing itself — one JSON object per line, bounded by `MAX_LINE` — is
`ph.framing`, in core, because the session's tool commands speak it too. What is
here is what only the daemon's wire needs: how large an attachment one frame may
carry, derived from that bound.

@module ph_app.daemon.framing
"""

from __future__ import annotations

__all__ = ["MAX_ATTACHMENT_BYTES", "MAX_ATTACHMENT_SIZE"]

MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
"""The largest attachment one frame may carry.

Under `ph.framing.MAX_LINE` (8 MiB) with room to spare, because the frame also carries the
method, the session id and base64's own 4/3 expansion — a limit set at the frame
size would refuse *after* the client had already sent it, as a framing error with
no name.

Here rather than in `server.py` because it is a property of the **wire**: it is
derived from that bound, and both ends of that wire need it — the daemon
to refuse a frame, and a front end to refuse an upload before it builds one. A
front end importing it from the server module would name the supervisor's home to
read an integer."""

MAX_ATTACHMENT_SIZE = f"{MAX_ATTACHMENT_BYTES // (1024 * 1024)} MiB"
"""The limit as both doors say it to a person.

The division floors, so a limit that stops being a whole number of MiB would
otherwise have two doors quietly rounding it down in two sentences.
"""
