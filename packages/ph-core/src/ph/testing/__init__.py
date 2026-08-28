"""`ph.testing` — fake and replay adapters, builders, a stub runtime."""

from __future__ import annotations

from .builders import (
    FAKE_OPTIONS,
    StubAgent,
    assistant_payload,
    plugin_payload,
    raising,
    run_tool,
    simple_tool,
    tool_result_payload,
    tool_runtime,
    user_payload,
)
from .fake_adapter import FakeAdapter, text_script
from .replay_adapter import RecordedStep, ReplayAdapter, recorded_steps
from .stub_runtime import StubCodeRuntime

__all__ = [
    "FAKE_OPTIONS",
    "FakeAdapter",
    "RecordedStep",
    "ReplayAdapter",
    "StubAgent",
    "StubCodeRuntime",
    "assistant_payload",
    "plugin_payload",
    "raising",
    "recorded_steps",
    "run_tool",
    "simple_tool",
    "text_script",
    "tool_result_payload",
    "tool_runtime",
    "user_payload",
]
