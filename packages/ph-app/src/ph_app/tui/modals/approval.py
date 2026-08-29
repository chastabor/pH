"""The approval modal, and the answerer that puts it on screen.

This is the seam's front-end half. `ApprovalService.request` records
`approval/asked`, waterfalls to whoever registered an answerer, and records
`approval/decided` — so the modal's only job is to return an outcome. It does
not append anything itself; a front-end writing its own approval events would
give the log two authors for one decision.

Four decisions reach the model (P4-05), and each is a different sentence a
person wants to say. *Allow* and *Reject* stop or start the call; a rejection may
carry a reason, because "no, use the existing helper" redirects a turn where a
bare denial only stops it. *Edit* corrects the call rather than refusing it, and
*Respond* answers in the tool's own voice without running it — both exist
because stopping a turn to say "wrong path" or "you don't need that" costs a
round trip that saying it in place does not.

Which of the four are offered is the *asking row's* call, carried on the request
as `allowed_decisions`: a deployment may withhold `edit` for a tool whose
arguments must not be hand-written. Dismissing the modal is a rejection:
absence is not consent.

@module ph_app.tui.modals.approval
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Button, Input, Static
from textual.widgets.button import ButtonVariant

from ph.seams.approval import (
    ApprovalAnswer,
    ApprovalDecisionName,
    ApprovalRequest,
    Edited,
    Responded,
)

from .base import PhModal

__all__ = ["ApprovalDecision", "ApprovalModal"]


@dataclass(frozen=True, slots=True)
class _Offer:
    """One button, and what pressing it opens.

    The four decisions were spelled in three parallel tables — button id to
    decision, decision to placeholder, decision to prefill — which is three
    places that have to agree every time a fifth is added. One row per decision
    instead.
    """

    decision: ApprovalDecisionName
    label: str
    placeholder: str
    variant: ButtonVariant = "default"
    prefills_arguments: bool = False


_OFFERS: tuple[_Offer, ...] = (
    _Offer("approve", "Allow once", "", variant="success"),
    _Offer("edit", "Edit arguments", "arguments as JSON", prefills_arguments=True),
    _Offer("respond", "Answer instead", "what should the model be told instead?"),
    _Offer("reject", "Reject", "why not? (optional)", variant="error"),
)

_BY_DECISION: dict[str, _Offer] = {offer.decision: offer for offer in _OFFERS}
_REJECT = _BY_DECISION["reject"]


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """What the human chose.

    `answer` is what the seam is handed; `reason` is the aside a rejection may
    carry, which the front end steers in as user input rather than putting on
    the decision — it is a message to the model, not part of the verdict.
    """

    answer: ApprovalAnswer
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
        self._pending = _REJECT
        self.cancel_value = ApprovalDecision(answer="rejected")

    def compose(self) -> ComposeResult:
        with Vertical(id="approval"):
            yield Static(
                Content.from_markup("[b]$tool[/b] wants to run", tool=self.request.tool_name),
                id="approval-title",
            )
            yield Static(Content(self.request.reason or ""), id="approval-reason")
            yield Input(id="approval-why")
            with Vertical(id="approval-actions"):
                for offer in _OFFERS:
                    if self._offers(offer.decision):
                        yield Button(
                            offer.label, id=f"approval-{offer.decision}", variant=offer.variant
                        )

    def _offers(self, decision: ApprovalDecisionName) -> bool:
        """Whether the asking row will accept this one. Empty means all four."""
        allowed = self.request.allowed_decisions
        return not allowed or decision in allowed

    def on_mount(self) -> None:
        first = next(iter(self.query(Button)), None)
        if first is not None:
            first.focus()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "approval-approve":
            self.dismiss(ApprovalDecision(answer="allowed-once"))
            return
        # The three that collect text share one box, revealed on the first press
        # and submitted on the second — the same button has to mean "do it now"
        # once the box is open, because every one of these inputs is optional or
        # prefilled.
        if self.has_class("-explaining"):
            self._submit(self.query_one("#approval-why", Input).value)
            return
        name = str(event.button.id).removeprefix("approval-")
        self._pending = _BY_DECISION.get(name, _REJECT)
        box = self.query_one("#approval-why", Input)
        box.placeholder = self._pending.placeholder
        box.value = _as_json(self.request.arguments) if self._pending.prefills_arguments else ""
        self.add_class("-explaining")
        box.focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._submit(event.value)

    def _submit(self, text: str) -> None:
        """Turn what was typed into the answer the seam expects."""
        typed = text.strip()
        if self._pending.decision == "respond" and typed:
            self.dismiss(ApprovalDecision(answer=Responded(message=typed)))
            return
        if self._pending.decision == "edit" and typed:
            try:
                arguments = json.loads(typed)
            except ValueError:
                # Not a refusal — the person meant to edit and mistyped, so the
                # box stays open with what they wrote rather than the call being
                # silently rejected on a stray comma.
                self.query_one("#approval-why", Input).placeholder = "not valid JSON — try again"
                return
            self.dismiss(ApprovalDecision(answer=Edited(arguments=arguments)))
            return
        self.dismiss(ApprovalDecision(answer="rejected", reason=typed))


def _as_json(arguments: Any) -> str:
    """The edit box opens on the call as it stands, when it can be rendered."""
    try:
        return json.dumps(arguments)
    except (TypeError, ValueError):
        return ""
