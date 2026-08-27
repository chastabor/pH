"""`ph.agent` — the agent handle, its inbox, and the registry."""

from __future__ import annotations

from .inbox import Inbox, InboxNotifications, InboxTarget
from .registry import AgentRegistry
from .types import (
    AgentCancelCause,
    AgentOptions,
    AgentStatus,
    PreStepDecision,
    PreStepRequest,
    RequestErrorAction,
    RequestFailure,
    RequestProposal,
    TurnEndReason,
)

__all__ = [
    "AgentCancelCause",
    "AgentOptions",
    "AgentRegistry",
    "AgentStatus",
    "Inbox",
    "InboxNotifications",
    "InboxTarget",
    "PreStepDecision",
    "PreStepRequest",
    "RequestErrorAction",
    "RequestFailure",
    "RequestProposal",
    "TurnEndReason",
]
