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
    "TOOL_DENIED",
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


FailureKind: TypeAlias = Literal["denied", "failed", "aborted"]
"""What kind of non-success one tool call was.

The fact every consumer branches on and none may infer: policy **denied** the
call, the tool **failed**, or cancellation **aborted** it. Code Mode ends the run
on the first, lets the program handle the second, and a UI colours each
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
