"""`ph.testing` — fake and replay adapters, builders, a stub runtime and tier."""

from __future__ import annotations

from .anthropic_wire import anthropic_reply
from .builders import (
    FAKE_OPTIONS,
    StubAgent,
    assistant_payload,
    boundary_for,
    parked_gate,
    plugin_payload,
    raising,
    reference_fork,
    run_tool,
    simple_tool,
    skill_service,
    stored_log,
    tool_result_payload,
    tool_runtime,
    user_payload,
    workspace_acquired,
    workspace_disposed,
    workspace_log,
    workspace_retained,
    workspace_seam,
    write_reference_fork,
)
from .diagnostics import report_section
from .fake_adapter import FakeAdapter, text_script
from .git import WORKTREE_ROWS, git, git_repo, needs_git, worktree_agent
from .replay_adapter import (
    REPLAY_ROW,
    RecordedStep,
    ReplayAdapter,
    recorded_steps,
    shared_prefix,
    text_chunks,
    tool_call_chunks,
)
from .skills import skill, write_skill
from .stub_runtime import StubCodeRuntime
from .stub_sandbox import StubSandboxProvider
from .stub_subagent import StubSubagentProvider
from .stub_workspace import StubWorkspaceProvider, acquire_for_role

__all__ = [
    "FAKE_OPTIONS",
    "REPLAY_ROW",
    "WORKTREE_ROWS",
    "FakeAdapter",
    "RecordedStep",
    "ReplayAdapter",
    "StubAgent",
    "StubCodeRuntime",
    "StubSandboxProvider",
    "StubSubagentProvider",
    "StubWorkspaceProvider",
    "acquire_for_role",
    "anthropic_reply",
    "assistant_payload",
    "boundary_for",
    "git",
    "git_repo",
    "needs_git",
    "parked_gate",
    "plugin_payload",
    "raising",
    "recorded_steps",
    "reference_fork",
    "report_section",
    "run_tool",
    "shared_prefix",
    "simple_tool",
    "skill",
    "skill_service",
    "stored_log",
    "text_chunks",
    "text_script",
    "tool_call_chunks",
    "tool_result_payload",
    "tool_runtime",
    "user_payload",
    "workspace_acquired",
    "workspace_disposed",
    "workspace_log",
    "workspace_retained",
    "workspace_seam",
    "worktree_agent",
    "write_reference_fork",
    "write_skill",
]
