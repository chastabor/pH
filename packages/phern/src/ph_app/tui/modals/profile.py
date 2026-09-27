"""A session's named profile changed since it started — take the new version?

Asked when a session starts and the person's own profile file moved since the
version the session runs on (session profiles, S6): the daemon holds the session
and puts the change to whoever is attached. The body is the daemon's account of
it — each setting with its old and new value, and whose it is — so every front
end shows the same list, and the choice is the person's.

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
    """The modal's answer as a decision; a dismissal is `later`."""
    return _DECISIONS.get(answer or "", "later")
