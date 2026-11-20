"""Project trust: one decision, as a `ConfirmModal`.

The decision is "read this, then choose". It needs a modal for this reason: pH
will run the `AGENTS.md` of this project, its hooks and its configured plugins,
and the user has not read the configuration of a directory that they have just
cloned. "Trust this project" is remembered under `$PH_HOME`, so the prompt then
does not appear again for that project root. "This session only" is not
remembered.

@module ph_app.tui.modals.trust
"""

from __future__ import annotations

from pathlib import Path

from .base import Action, ConfirmModal

__all__ = ["project_trust_modal"]


def project_trust_modal(root: Path) -> ConfirmModal:
    """Ask whether to load this project's configuration."""
    body = (
        f"{root}\n\n"
        "pH loads this project's AGENTS.md, hooks and plugins. Those run on your\n"
        "machine with your permissions. Trust it only if you would read its\n"
        "configuration before running it."
    )
    return ConfirmModal(
        title="trust this project?",
        body=body,
        actions=[
            Action("always", "Trust this project", "success"),
            Action("once", "This session only", "primary"),
            Action("no", "Don't load it", "error"),
        ],
        cancel_value="no",
        # Enter on an unread prompt picks the answer that ends with the session.
        # To trust a directory permanently should be a choice that the user
        # moved to.
        initial="once",
    )
