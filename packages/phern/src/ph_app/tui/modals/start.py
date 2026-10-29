"""The start that did not happen: said once, kept, and answerable.

A terminal whose daemon would not start, or whose session the daemon refused, has
nothing behind its prompt. The report of that used to be a toast, and a toast
lasts five seconds; what it left was a screen with its headings drawn and nothing
behind them, which reads as a hang. This modal is the report that waits to be
read, and the answers a person can give it: run the whole open again, for the
case where the fix is in another terminal; leave; or stay, on this terminal's
own verbs.

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

    The last paragraph is for Esc, the one answer with no button."""
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
    """The modal's answer, narrowed: anything but the two buttons is staying."""
    return _DECISIONS.get(answer or "", "stay")
