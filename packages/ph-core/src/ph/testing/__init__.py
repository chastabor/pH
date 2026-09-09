"""`ph.testing` — the builders, stubs and fixtures a test stands a profile up with.

**Nothing here is mounted by a plugin row.** The fake adapter, the replay adapter
and the stub code runtime are rows a deployment can mount by name, so they live
with the seams they implement (`ph.llm.fake`, `ph.llm.replay`,
`ph.seams.code_runtime_stub`) and are re-exported below for tests. What is left
is scaffolding: no shipped module imports it, which is what lets it depend on
whatever a test needs.

`.git` and `.jj` are the exception to the re-export, and deliberately: they drive
real binaries, their fixtures are only useful to a test that carries the matching
`needs_git`/`needs_jj` marker, and a name that reads `git` here would shadow the
submodule it came from. A caller names them — `from ph.testing.git import
git_repo` — the way `ph-rlm`'s suite already did.

@module ph.testing
"""

from __future__ import annotations

from ..llm.fake import FakeAdapter, text_script
from ..llm.replay import (
    REPLAY_ROW,
    RecordedStep,
    ReplayAdapter,
    recorded_steps,
    shared_prefix,
    text_chunks,
    tool_call_chunks,
)
from ..seams.code_runtime_stub import StubCodeRuntime
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
from .folds import VerifyingFoldCache, assert_fold_laws, check_fold_laws, prefix_of
from .skills import skill, write_skill
from .stub_sandbox import StubSandboxProvider
from .stub_subagent import StubSubagentProvider
from .stub_workspace import (
    StubCheckpointingProvider,
    StubWorkspaceProvider,
    acquire_for_role,
)

__all__ = [
    "FAKE_OPTIONS",
    "REPLAY_ROW",
    "FakeAdapter",
    "RecordedStep",
    "ReplayAdapter",
    "StubAgent",
    "StubCheckpointingProvider",
    "StubCodeRuntime",
    "StubSandboxProvider",
    "StubSubagentProvider",
    "StubWorkspaceProvider",
    "VerifyingFoldCache",
    "acquire_for_role",
    "anthropic_reply",
    "assert_fold_laws",
    "assistant_payload",
    "boundary_for",
    "check_fold_laws",
    "parked_gate",
    "plugin_payload",
    "prefix_of",
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
    "write_reference_fork",
    "write_skill",
]
