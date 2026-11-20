"""The start that did not happen: reported one time, kept, and answerable.

A terminal has nothing behind its prompt when its daemon would not start, or
when the daemon refused its session. The report of that used to be a toast, and
a toast lasts five seconds. It left a screen with its headings drawn and nothing
behind them, which looks like a hang. This modal is the report that waits until
the user reads it. It also offers the answers that a user can give:

- Run the whole open again, for the case where the fix is in another terminal.
- Leave.
- Stay, on the verbs that belong to this terminal.

@module ph_app.tui.modals.start
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeAlias

from ph.wire import literal_lookup

from .base import Action, ConfirmModal

__all__ = ["StartDecision", "could_not_start_modal", "start_decision"]

StartDecision: TypeAlias = Literal["retry", "quit", "stay"]
"""What to do about a start that failed."""

_DECISIONS: Mapping[str, StartDecision] = literal_lookup(StartDecision)


def could_not_start_modal(reason: str) -> ConfirmModal:
    """Show why there is no session, and ask what to do about it.

    The last paragraph of the body is for Esc, which is the one answer that has
    no button."""
    return ConfirmModal(
        title="pH could not start",
        body=(
            f"{reason}\n\nEsc keeps this terminal open on its own verbs, such as "
            "/theme and /quit; anything that needs a session waits for a daemon."
        ),
        actions=[
            Action("retry", "Try again", "primary"),
            Action("quit", "Quit", "error"),
        ],
        cancel_value="stay",
    )


def start_decision(answer: str | None) -> StartDecision:
    """The answer of the modal, narrowed. Anything except the two buttons means stay."""
    return _DECISIONS.get(answer or "", "stay")
