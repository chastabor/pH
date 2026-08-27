"""`ph.testing` — the fake adapter, payload builders, and the loop test harness."""

from __future__ import annotations

from .builders import assistant_payload, tool_result_payload, user_payload
from .fake_adapter import FakeAdapter, text_script

__all__ = ["FakeAdapter", "assistant_payload", "text_script", "tool_result_payload", "user_payload"]
