"""The two failures a cell can see from a governed call, and the line between them.

`ToolFailed` is the program's to handle — a timeout, a bad argument, a file that
was not there. Catching it and trying something else is exactly right.

`RunStopped` is not. It derives from `BaseException` so that `except Exception`
does not swallow it, and the reason is C3: a program that can catch a refusal can
route around it — retry with a different path, fall back to `subprocess`. The
refusal ends the run, the model sees it in context, and partial state is bounded
to one cell. A budget (C4) ends the run for the same reason: governance is per
call, but attention is per turn.

@module ph_runtime.errors
"""

from __future__ import annotations

__all__ = ["RunStopped", "ToolFailed"]


class ToolFailed(Exception):
    """A governed call failed. Yours to handle."""

    def __init__(self, name: str, message: str) -> None:
        super().__init__(f"{name} failed: {message}")
        self.tool_name = name


class RunStopped(BaseException):
    """The run is over — refused, or out of budget. Not yours to handle.

    `BaseException` so `except Exception` does not swallow it — but note where
    the *enforcement* is: a cell can still write `except BaseException`, so what
    actually ends a refused run is the host firing its abort ladder (C3). This
    class is the courtesy that lets a well-behaved cell unwind with a readable
    message first.
    """
