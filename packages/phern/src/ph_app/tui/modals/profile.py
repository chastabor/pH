"""The named profile of a session changed since it started: take the new version?

The question is asked when a session starts and the profile file of the user
has moved since the version that the session runs on (session profiles, S6).
The daemon holds the session and puts the change to whoever is attached. The
body is the account that the daemon gives of the change: each setting with its
old value and its new value, and whose setting it is. Every front end therefore
shows the same list, and the choice belongs to the user.

@module ph_app.tui.modals.profile
"""

from __future__ import annotations

from collections.abc import Mapping

from ph.wire import literal_lookup

from ...payloads import ProfileDecision
from .base import Action, ConfirmModal

__all__ = ["decision_of", "profile_changed_modal"]

_DECISIONS: Mapping[str, ProfileDecision] = literal_lookup(ProfileDecision)


def profile_changed_modal(name: str, listing: str) -> ConfirmModal:
    """Show what changed in `name`, and ask which version the session runs on."""
    return ConfirmModal(
        title=f"{name or 'this profile'} has changed",
        body=listing,
        actions=[
            Action("adopt", "Use the new version", "success"),
            Action("keep", "Keep this session's", "primary"),
            Action("later", "Ask me next time", "default"),
        ],
        cancel_value="later",
    )


def decision_of(answer: str | None) -> ProfileDecision:
    """The answer of the modal as a decision. A dismissal is `later`."""
    return _DECISIONS.get(answer or "", "later")
