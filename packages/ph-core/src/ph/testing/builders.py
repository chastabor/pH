"""Event-payload builders for tests and fixtures.

The three surface types have a shape every test needs and none should retype:
a user message, an assistant message inside its `assistant/message` payload,
and a tool result inside its `tool/result` payload. Ids are explicit so a test
can cite them.

@module ph.testing.builders
"""

from __future__ import annotations

from typing import Any

__all__ = ["assistant_payload", "tool_result_payload", "user_payload"]


def user_payload(text: str, message_id: str = "m1") -> dict[str, Any]:
    """A `user/message` payload for typed human text."""
    return {
        "id": message_id,
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "source": {"kind": "user"},
    }


def assistant_payload(
    text: str, message_id: str, *, turn: int = 1, step: int = 1, provider: str = "fake"
) -> dict[str, Any]:
    """An `assistant/message` payload; empty `text` gives an empty-content message."""
    return {
        "turn": turn,
        "step": step,
        "message": {
            "id": message_id,
            "role": "assistant",
            "content": [{"type": "text", "text": text}] if text else [],
            "source": {"kind": "model", "provider": provider, "model": "m"},
        },
    }


def tool_result_payload(
    text: str, message_id: str, call_id: str = "c1", *, turn: int = 1, step: int = 1
) -> dict[str, Any]:
    """A `tool/result` payload carrying one text block."""
    return {
        "turn": turn,
        "step": step,
        "message": {
            "id": message_id,
            "role": "user",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": call_id,
                    "content": [{"type": "text", "text": text}],
                    "isError": False,
                }
            ],
            "source": {"kind": "tool", "callId": call_id},
        },
    }
