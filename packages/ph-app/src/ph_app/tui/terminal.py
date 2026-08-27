"""The terminal's window title.

The bell is `App.bell()` — Textual writes it through its own driver, which is
also headless-safe. The title has no such helper, so it lives here, with the
one subtlety that matters: Textual redirects `sys.stdout` while it drives the
screen, so a control sequence written there lands in the app's log (and, under a
test harness, in the captured output as literal `]2;pH` noise). These bytes are
instructions, not output; they go to `sys.__stdout__`, and only when that is a
terminal that will interpret them.

@module ph_app.tui.terminal
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

__all__ = ["TerminalTitle"]


def _terminal() -> TextIO | None:
    stream = sys.__stdout__
    try:
        return stream if stream is not None and stream.isatty() else None
    except (AttributeError, ValueError):
        return None


@dataclass(slots=True)
class TerminalTitle:
    """Sets the window title, and only when it actually changed."""

    _current: str = ""

    def set(self, detail: str = "") -> None:
        title = f"pH — {detail}" if detail else "pH"
        if title == self._current:
            return
        self._current = title
        stream = _terminal()
        if stream is None:
            return
        stream.write(f"\033]2;{title}\007")  # OSC 2
        stream.flush()

    def clear(self) -> None:
        self.set("")
