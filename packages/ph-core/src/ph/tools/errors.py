"""The error vocabulary of the tool pipeline.

Each error carries a stable `code`, because the routing of a failure matters as
much as its message. Retry policy, the sandbox layer and replay all branch on
the code. A string match would break as soon as the wording was improved.

@module ph.tools.errors
"""

from __future__ import annotations

from typing import Literal, TypeAlias

__all__ = [
    "SPAWN_REFUSED",
    "TOOL_ABORTED",
    "TOOL_ABORTED_BEFORE_DISPATCH",
    "TOOL_BUDGET_SPENT",
    "TOOL_DENIED",
    "TOOL_EFFECT_IN_FLIGHT",
    "TOOL_TIMEOUT",
    "TOOL_TURN_CONCLUDED",
    "HarnessError",
    "ToolNotFoundError",
    "ToolOutputError",
]

TOOL_ABORTED = "ABORTED"
"""Cancellation after the tool body was entered."""

TOOL_ABORTED_BEFORE_DISPATCH = "ABORTED_BEFORE_DISPATCH"
"""Cancellation before the body ran. The call had no effect."""

TOOL_TIMEOUT = "TIMEOUT"
"""The call ran past its `timeout_ms` and was cut off. The kind is `failed`,
because the tool did not finish. Its effect may still have happened. The
registry, and not this code, records that fact for a keyed call
(`ToolRunContext._cut_off`)."""

TOOL_DENIED = "TOOL_DENIED"
"""Policy refused the call: a `deny` decision, or a monotonic guard.

The code is routable on purpose. A refusal and a failure look identical to a
model that reads the content. But they are different facts, and different code
must branch on them. Code Mode fails the whole run on a refusal and lets the
program handle a failure (C3). That is impossible if the two cannot be told
apart.
"""


SPAWN_REFUSED = "SPAWN_REFUSED"
"""A delegation was refused before the child existed (D15).

This is different from a failure of the call that carried the delegation.
Nothing ran, and no session and no log were created, so a caller may retry with
different arguments. A spawn that a *budget* refused carries `TOOL_BUDGET_SPENT`
instead, for the reason that the code exists: the ceiling does not move within
the turn, so a retry is the one thing that cannot help.
"""


TOOL_BUDGET_SPENT = "TOOL_BUDGET_SPENT"
"""A configured ceiling stopped the call: the budget is spent (D7).

The code is routable for the same reason as `TOOL_DENIED`, and it is separate
from `TOOL_DENIED` for the reason that `FailureKind` gives. A denial is policy
refusing *this* call, and a model may reasonably ask for permission or try
another way. A spent budget also refuses every later call, and there is nothing
to ask for, because the ceiling does not move within a turn. The two read
identically as content and are different facts. This module exists to keep that
distinction.

The code is carried with `kind="failed"` and not `"denied"`, because no policy
judged the call. `code_mode.CodeRunFailure` already separates `budget` from
`denied` in the same way, and `fs.FileTooLarge` is separate from `fs.FsDenied`
in the same way.
"""


TOOL_EFFECT_IN_FLIGHT = "TOOL_EFFECT_IN_FLIGHT"
"""Another call in this process is already carrying out the same effect.

A tool that names its effect (`ToolDefinition.idempotency_key`) runs one time
for each effect. A second call that asks for the effect while the first call is
still running is refused and does not run. Running it would be the repeat that
the key exists to prevent, and opening a second intent underneath the first
would displace the settle of the first. The kind is `failed`, because nothing
ran. The call is safe to retry after the first call has returned, and the retry
is then answered from the log.
"""

TOOL_TURN_CONCLUDED = "TOOL_TURN_CONCLUDED"
"""The turn ended before this call ran (C13).

The kind is `aborted`, because that is what happened to the call: it was never
dispatched, it had no effect, and it is safe to retry. It has its own code and
does not use `TOOL_ABORTED_BEFORE_DISPATCH`, for the reason that pair is itself
split: an interrupt by a person and a spent budget end a call at the same moment
and are not the same fact. A model that reads "aborted before dispatch" will
conclude that somebody canceled the call.
"""


FailureKind: TypeAlias = Literal["denied", "failed", "aborted"]
"""The kind of non-success that one tool call was.

Every consumer branches on this fact, and none may infer it: policy **denied**
the call, the tool **failed**, or cancellation **aborted** it. Code Mode ends
the run on the first and lets the program handle the second, and a UI colors
each one differently.

It is declared here, in the lowest module of the tools package, because it is
what an *error* says about itself. `definition.py` exports it again for the
callers that know it as part of the tool contract."""


class HarnessError(Exception):
    """An error that carries a machine-routable code.

    `failure_kind` is what raising this error means to a consumer. The error
    that knows the kind declares it, and nothing downstream infers it from a
    list of codes. It is a class attribute and not a `ClassVar`, because one
    error type can carry more than one kind. `CodeRunFailure` maps its three
    kinds, and an instance that knows its kind sets it in `__init__`.
    """

    failure_kind: FailureKind = "failed"

    concludes_turn: bool = False
    """Whether raising this error also ends the turn (C13).

    The error that knows declares it, for the same reason as `failure_kind`, and
    it travels in the same way: `registry._failure` reads both from the error
    and infers neither from the code. Before this field, a raise that ended a
    turn could not say so. A ceiling that was reached inside a Code Mode program
    stopped the program and left the loop running. The one result that the loop
    reads came back with the flag unset, because the exception that it was built
    from could not carry the flag.
    """

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class ToolNotFoundError(HarnessError):
    """The model asked for a tool that is not registered, or that is not callable
    in this way.

    `reachable_from` names the way back when the tool *is* visible and only the
    presentation forbids a direct call. Under Code Mode, a native call is
    refused with the SDK path in the denial, so the model can correct itself and
    does not have to guess (C6).
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
    """A human-readable message from an arbitrary raised value.

    The function is total by construction. A hostile value can break
    `isinstance`, attribute access *and* `str()`. Error normalization is the
    outermost safety boundary, so its fallback must not raise.
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
