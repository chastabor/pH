"""The approval modal, and the decision that it returns.

This is the front-end half of the seam. `ApprovalService.request` records
`approval/asked`, runs the waterfall to whoever registered an answerer, and
records `approval/decided`. The only task of the modal is therefore to return an
outcome. It appends nothing itself. If a front end wrote its own approval
events, the log would have two authors for one decision.

Four decisions reach the model (P4-05), and each one is a different sentence
that a person wants to say. *Allow* starts the call and *Reject* stops it. A
rejection may carry a reason, because "no, use the existing helper" redirects a
turn, and a bare denial only stops it. *Edit* corrects the call and does not
refuse it. *Respond* answers in place of the tool and does not run it. Both
exist because stopping a turn to say "wrong path" or "you don't need that" costs
a round trip, and saying it in place does not.

The *asking row* decides which of the four are offered, and the request carries
that as `allowed_decisions`. A deployment may withhold `edit` for a tool whose
arguments must not be written by hand. Dismissing the modal is a rejection,
because absence is not consent.

@module ph_app.tui.modals.approval
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Button, Input, Static
from textual.widgets.button import ButtonVariant

from ph.json import JsonValue
from ph.seams.approval import (
    ApprovalAnswer,
    ApprovalDecisionName,
    ApprovalRequest,
    Edited,
    Responded,
)

from .base import Answers, PhModal

__all__ = ["ApprovalDecision", "ApprovalModal"]


@dataclass(frozen=True, slots=True)
class _Offer:
    """One button, and what a press on it opens.

    The four decisions were written in three parallel tables: button id to
    decision, decision to placeholder, and decision to prefill. Those are three
    places that must agree every time a fifth decision is added. There is now
    one row for each decision.
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
    """What the user chose.

    `answer` is what the seam receives. `reason` is the remark that a rejection
    may carry. The front end steers it in as user input and does not put it on
    the decision, because it is a message to the model and not part of the
    verdict.
    """

    answer: ApprovalAnswer
    reason: str = ""


class ApprovalModal(PhModal[ApprovalDecision]):
    """Ask whether one tool call may run.

    The decisions are an `Answers` column, the same as the answers of every
    other dialog. They are centered, and the arrow keys move between them as
    well as Tab. The arrows move among the decisions only, so after the reason
    box opens, they stay in the box.
    """

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
            yield Answers(
                *(
                    Button(offer.label, id=f"approval-{offer.decision}", variant=offer.variant)
                    for offer in _OFFERS
                    if self._offers(offer.decision)
                ),
                id="approval-actions",
            )

    def _offers(self, decision: ApprovalDecisionName) -> bool:
        """Whether the asking row accepts this decision. An empty list means all four."""
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
        # The three decisions that collect text share one box. The first press
        # shows the box and the second press submits it. The same button must
        # mean "do it now" after the box is open, because every one of these
        # inputs is optional or prefilled.
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
        """Turn the typed text into the answer that the seam expects."""
        typed = text.strip()
        if self._pending.decision == "respond" and typed:
            self.dismiss(ApprovalDecision(answer=Responded(message=typed)))
            return
        if self._pending.decision == "edit" and typed:
            try:
                arguments = json.loads(typed)
            except ValueError:
                # This is not a refusal. The user meant to edit and mistyped, so
                # the box stays open with the typed text. A stray comma must not
                # reject the call silently.
                self.query_one("#approval-why", Input).placeholder = "not valid JSON — try again"
                return
            self.dismiss(ApprovalDecision(answer=Edited(arguments=arguments)))
            return
        self.dismiss(ApprovalDecision(answer="rejected", reason=typed))


def _as_json(arguments: JsonValue) -> str:
    """The edit box opens with the call as it stands, when the call can be rendered."""
    try:
        return json.dumps(arguments)
    except (TypeError, ValueError):
        return ""
