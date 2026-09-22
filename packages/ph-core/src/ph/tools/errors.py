"""Tool-pipeline error vocabulary.

Each carries a stable `code`, because a failure's routing matters as much as its
message: retry policy, the sandbox layer and replay all branch on the code, and
a string match would break the moment the wording improved.

@module ph.tools.errors
"""

from __future__ import annotations

from typing import Literal, TypeAlias

__all__ = [
    "TOOL_ABORTED",
    "TOOL_ABORTED_BEFORE_DISPATCH",
    "TOOL_BUDGET_SPENT",
    "TOOL_DENIED",
    "TOOL_TURN_CONCLUDED",
    "HarnessError",
    "ToolNotFoundError",
    "ToolOutputError",
]

TOOL_ABORTED = "ABORTED"
"""Cancellation after the tool body was entered."""

TOOL_ABORTED_BEFORE_DISPATCH = "ABORTED_BEFORE_DISPATCH"
"""Cancellation before the body ran — the call had no effect."""

TOOL_DENIED = "TOOL_DENIED"
"""Policy refused the call: a `deny` decision, or a monotonic guard.

Routable on purpose. A refusal and a failure look identical to a model reading
content, but they are different facts and different code has to branch on them:
Code Mode fails the whole run on a refusal and lets the program handle a failure
(C3), which is impossible if the two are indistinguishable.
"""


TOOL_BUDGET_SPENT = "TOOL_BUDGET_SPENT"
"""A configured ceiling stopped the call: the budget is spent (D7).

Routable for `TOOL_DENIED`'s reason and kept apart from it for `FailureKind`'s.
A denial is policy refusing *this* call and a model may reasonably ask for
permission or try another way; a spent budget refuses every later call too, and
there is nothing to ask for — the ceiling does not move within a turn. The two
read identically as content and are different facts, which is the distinction
this module exists to keep.

Carried with `kind="failed"` rather than `"denied"`, because no policy judged
the call: `code_mode.CodeRunFailure` already separates `budget` from `denied`
on exactly this line, and `fs.FileTooLarge` from `fs.FsDenied` on the same one.
"""


TOOL_TURN_CONCLUDED = "TOOL_TURN_CONCLUDED"
"""The turn ended before this call ran (C13).

`aborted` in kind, because that is what happened to the call: it never
dispatched, it had no effect, and it is safe to retry. Its own code rather than
`TOOL_ABORTED_BEFORE_DISPATCH` for the reason that pair is itself split — a
person's interrupt and a spent budget end a call at the same moment and are not
the same fact, and a model that reads "aborted before dispatch" will conclude
somebody cancelled it.
"""


FailureKind: TypeAlias = Literal["denied", "failed", "aborted"]
"""What kind of non-success one tool call was.

The fact every consumer branches on and none may infer: policy **denied** the
call, the tool **failed**, or cancellation **aborted** it. Code Mode ends the run
on the first, lets the program handle the second, and a UI colors each
differently.

Declared here, in the lowest module of the tools package, because it is what an
*error* says about itself — `definition.py` re-exports it for the callers that
know it as part of the tool contract."""


class HarnessError(Exception):
    """An error carrying a machine-routable code.

    `failure_kind` is what raising this means to a consumer, declared by the
    error that knows rather than inferred downstream from a list of codes. A
    class attribute rather than a `ClassVar`, because one error type can carry
    more than one kind — `CodeRunFailure` maps its three — and an instance that
    knows sets it in `__init__`.
    """

    failure_kind: FailureKind = "failed"

    concludes_turn: bool = False
    """Whether raising this also ends the turn (C13).

    Declared by the error that knows, for `failure_kind`'s reason and travelling
    the same way: `registry._failure` reads both off the class rather than
    inferring either from the code. A raise that ends a turn had no way to say
    so, so a ceiling reached inside a Code Mode program stopped the program and
    left the loop running — the one result the loop reads came back with the
    flag unset, because the exception it was built from could not carry it.
    """

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ToolNotFoundError(HarnessError):
    """The model asked for a tool that is not registered, or not callable this way.

    `reachable_from` names the route back when the tool *is* visible and only the
    presentation forbids calling it directly — under Code Mode a native call is
    refused with the SDK path in the denial, so the model can correct itself
    instead of guessing (C6).
    """

    failure_kind: FailureKind = "denied"

    def __init__(self, tool_name: str, reachable_from: str | None = None) -> None:
        detail = f'unknown tool "{tool_name}"'
        if reachable_from is not None:
            detail = f"{detail}: {reachable_from}"
        super().__init__(detail, "UNKNOWN_TOOL")
        self.tool_name = tool_name


class ToolOutputError(HarnessError):
    """A tool body or post-policy value violated the tool's declared output."""

    def __init__(self, tool_name: str, violations: list[str]) -> None:
        super().__init__(
            f'tool "{tool_name}" returned invalid output: {"; ".join(violations)}',
            "INVALID_TOOL_OUTPUT",
        )
        self.violations = violations


def error_message(error: object) -> str:
    """A human-readable message from an arbitrary thrown value.

    Total by construction: a hostile value can break `isinstance`, attribute
    access *and* `str()`, and error normalization is the outermost safety
    boundary — so its fallback cannot itself raise.
    """
    try:
        if isinstance(error, BaseException):
            return str(error) or type(error).__name__
        message = getattr(error, "message", None)
        if isinstance(message, str):
            return message
        return str(error)
    except Exception:
        return "<unprintable raised value>"


def error_info(error: object) -> dict[str, str] | None:
    """`{name, code}` for a coded harness error, else `None`."""
    try:
        if isinstance(error, HarnessError):
            return {"name": type(error).__name__, "code": error.code}
    except Exception:
        return None
    return None
