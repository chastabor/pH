"""Project trust and plan review — two decisions, one shape.

Both are "read this, then choose", so both are `ConfirmModal`. What makes trust
worth a modal at all: pH will run this project's `AGENTS.md`, its hooks and its
configured plugins, and a directory the user has just cloned is a directory
whose configuration they have not read. The prompt is once per project root,
and the answer is remembered under `$PH_HOME`.

@module ph_app.tui.modals.trust
"""

from __future__ import annotations

from pathlib import Path

from .base import Action, ConfirmModal

__all__ = ["plan_review_modal", "project_trust_modal"]


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
    )


def plan_review_modal(plan: str) -> ConfirmModal:
    """Show a proposed plan and ask what to do with it.

    Opened by Phase 4's `plan/mode` exit; the dialog is Phase 2's (P2-04).
    """
    return ConfirmModal(
        title="review the plan",
        body=plan,
        actions=[
            Action("approve", "Go ahead", "success"),
            Action("edit", "Let me revise it", "primary"),
            Action("reject", "No, stop here", "error"),
        ],
        cancel_value="reject",
    )
