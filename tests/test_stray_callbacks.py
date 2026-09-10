"""The attribution hook in `conftest.py`, held to what it promises.

A callback that raises after its test has finished is collected by anyio's
session-wide `TestRunner` and re-raised inside an unrelated test, with a
traceback holding no application frames — `self = None`, one line of
`asyncio/events.py`, nothing else. Twice that has sent someone investigating a
bystander (issue 58, and the earlier instance `daemon_helpers.close_clients`
was written for), so `pytest_configure` attaches what the traceback lacks.

**Tested here because a guard nobody tests is a guard somebody deletes while
tidying imports** — the argument `test_the_human_door_needs_no_daemon_at_all`
makes about its own `TYPE_CHECKING` import. The handler is called directly
rather than by provoking a real stray: a test that leaves one would fail by
design, and a suite that expects a failure somewhere is worse than the flake.

Issue 58 is root-caused but not yet fixed, so this still fires — and the note
tells that reader the answer is already written down rather than sending them
after it a third time.
"""

from __future__ import annotations

from typing import Any

from anyio._backends._asyncio import TestRunner as _Runner_

"""Aliased: pytest tries to *collect* any imported class named `Test*`,
and warns that it cannot because `TestRunner` takes constructor arguments."""


def _fire(context: dict[str, Any]) -> BaseException:
    """Put one loop-callback failure through the installed handler."""
    error = context["exception"]
    collected: list[BaseException] = []

    class _Runner:
        def __init__(self) -> None:
            self._exceptions = collected

    _Runner_._exception_handler(_Runner(), None, context)  # type: ignore[arg-type]
    assert isinstance(error, BaseException)
    return error


def test_a_stray_callback_names_its_own_callback() -> None:
    """The `context:` note is the whole point: it is the one line that says
    *which* callback raised, where the traceback says only that one did."""
    error = _fire(
        {
            "exception": ValueError("invalid state"),
            "message": "Exception in callback Future.set_result(2)",
            "handle": "<Handle Future.set_result(2)>",
        }
    )

    notes = "\n".join(getattr(error, "__notes__", []))
    assert "Future.set_result(2)" in notes, "the callback is not named"
    assert "running:" in notes, "and neither is the test it fired during"
    assert "very likely not the cause" in notes, "nor the warning against chasing it"


def test_the_handler_still_collects_so_a_stray_keeps_failing() -> None:
    """Attribution, not suppression. A stray callback is a real defect, and a
    hook that swallowed it would trade a misattributed failure for none at
    all — which is the worse of the two."""
    error = ValueError("boom")
    collected: list[BaseException] = []

    class _Runner:
        def __init__(self) -> None:
            self._exceptions = collected

    _Runner_._exception_handler(_Runner(), None, {"exception": error})  # type: ignore[arg-type]

    assert collected == [error], "anyio must still re-raise it"


def test_a_context_with_no_exception_is_left_alone() -> None:
    """Not every loop report carries one — a transport warning does not, and
    annotating `None` would be an `AttributeError` inside the handler that
    reports errors."""

    class _Runner:
        def __init__(self) -> None:
            self._exceptions: list[BaseException] = []

    # No `exception` key at all: the handler must fall through to anyio's own,
    # which hands it to the loop's default handler rather than raising here.
    _Runner_._exception_handler(_Runner(), _Loop(), {"message": "socket closed"})  # type: ignore[arg-type]


class _Loop:
    """Just enough loop for anyio's fallback path."""

    def default_exception_handler(self, context: dict[str, Any]) -> None:
        return None
