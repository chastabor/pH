"""Project trust and plan review — two decisions, one shape.

Both are "read this, then choose", so both are `ConfirmModal`. What makes trust
worth a modal at all: pH will run this project's `AGENTS.md`, its hooks and its
configured plugins, and a directory the user has just cloned is a directory
whose configuration they have not read. The prompt is once per project root,
and the answer is remembered under `$PH_HOME`.

@module ph_app.tui.modals.trust
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ph.paths import write_text_under

from .base import Action, ConfirmModal

__all__ = ["TrustStore", "plan_review_modal", "project_trust_modal"]


@dataclass(slots=True)
class TrustStore:
    """Which project roots the user has trusted, kept outside the project.

    Deliberately not stored in the project: a file inside the repository could
    declare the repository trustworthy, which is the one thing the prompt is
    supposed to prevent.
    """

    path: Path

    def trusted(self, root: Path) -> bool:
        return str(root.resolve()) in self._load()

    def trust(self, root: Path) -> None:
        roots = self._load()
        roots.add(str(root.resolve()))
        write_text_under(self.path, json.dumps({"trusted": sorted(roots)}, indent=2) + "\n")

    def _load(self) -> set[str]:
        try:
            record = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        trusted = record.get("trusted")
        return (
            {item for item in trusted if isinstance(item, str)}
            if isinstance(trusted, list)
            else set()
        )


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
            Action("trust", "Trust this project", "success"),
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
