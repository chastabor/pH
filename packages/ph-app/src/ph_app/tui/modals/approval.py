"""The approval modal, and the answerer that puts it on screen.

This is the seam's front-end half. `ApprovalService.request` records
`approval/asked`, waterfalls to whoever registered an answerer, and records
`approval/decided` — so the modal's only job is to return an outcome. It does
not append anything itself; a front-end writing its own approval events would
give the log two authors for one decision.

Two outcomes reach the model: `allowed-once` and `rejected`. A rejection may
carry a reason, and that reason is worth collecting — "no, use the existing
helper" redirects a turn, where a bare denial only stops it. Dismissing the
modal is a rejection: absence is not consent.

@module ph_app.tui.modals.approval
"""

from __future__ import annotations

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Button, Input, Static

from ph.seams.approval import ApprovalOutcome, ApprovalRequest

from .base import PhModal

__all__ = ["ApprovalDecision", "ApprovalModal"]


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """What the human chose."""

    outcome: ApprovalOutcome
    reason: str = ""


class ApprovalModal(PhModal[ApprovalDecision]):
    """Ask whether one tool call may run."""

    DEFAULT_CSS = """
    ApprovalModal { align: center middle; background: $background 60%; }
    ApprovalModal > #approval {
        width: 80; max-width: 90%; height: auto;
        background: $ph-panel; border: round $ph-warning; padding: 1 2;
    }
    ApprovalModal #approval-title { color: $ph-warning; }
    ApprovalModal #approval-reason { color: $ph-muted; margin: 1 0; }
    ApprovalModal #approval-why { border: none; background: $ph-surface; display: none; }
    ApprovalModal.-explaining #approval-why { display: block; }
    ApprovalModal #approval-actions { height: auto; }
    ApprovalModal Button { margin-right: 1; }
    """

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request
        self.cancel_value = ApprovalDecision(outcome="rejected")

    def compose(self) -> ComposeResult:
        with Vertical(id="approval"):
            yield Static(
                Content.from_markup("[b]$tool[/b] wants to run", tool=self.request.tool_name),
                id="approval-title",
            )
            yield Static(Content(self.request.reason or ""), id="approval-reason")
            yield Input(placeholder="why not? (optional)", id="approval-why")
            with Vertical(id="approval-actions"):
                yield Button("Allow once", id="approval-allow", variant="success")
                yield Button("Reject", id="approval-reject", variant="error")

    def on_mount(self) -> None:
        self.query_one("#approval-allow", Button).focus()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "approval-allow":
            self.dismiss(ApprovalDecision(outcome="allowed-once"))
            return
        # First press on Reject reveals the reason box; second submits it. The
        # reason is optional, so the same button has to mean "reject now" once
        # the box is already open.
        if self.has_class("-explaining"):
            reason = self.query_one("#approval-why", Input).value.strip()
            self.dismiss(ApprovalDecision(outcome="rejected", reason=reason))
            return
        self.add_class("-explaining")
        self.query_one("#approval-why", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.dismiss(ApprovalDecision(outcome="rejected", reason=event.value.strip()))
