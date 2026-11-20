"""The two failures that a cell can see from a governed call, and how they differ.

`ToolFailed` is for the program to handle: a timeout, a bad argument, or a file
that was not there. To catch it and try something else is correct.

`RunStopped` is not for the program to handle. It derives from `BaseException` so
that `except Exception` does not catch it. The reason is C3: a program that can
catch a refusal can go around it, for example by a retry with a different path or
a fallback to `subprocess`. The refusal ends the run, the model sees it in
context, and partial state is limited to one cell. A budget (C4) ends the run for
the same reason: governance is per call, but attention is per turn.

@module ph_runtime.errors
"""

from __future__ import annotations

__all__ = ["RunStopped", "ToolFailed"]


class ToolFailed(Exception):
    """A governed call failed. The program may handle this."""

    def __init__(self, name: str, message: str) -> None:
        super().__init__(f"{name} failed: {message}")
        self.tool_name = name


class RunStopped(BaseException):
    """The run is over, because it was refused or it is out of budget. The program
    must not handle this.

    The class derives from `BaseException` so that `except Exception` does not
    catch it. But the *enforcement* is elsewhere. A cell can still write `except
    BaseException`, so what really ends a refused run is the abort ladder of the
    host (C3). This class is a courtesy: it lets a well-behaved cell unwind with
    a readable message first.
    """
