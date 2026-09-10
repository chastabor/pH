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

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ..cordis import Context

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
    as_kind,
    assistant_payload,
    block_text,
    boundary_for,
    not_none,
    noted,
    noting,
    parked_gate,
    plugin_payload,
    raising,
    reference_fork,
    run_tool,
    session_of,
    simple_tool,
    skill_service,
    store_root,
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

MountProfile: TypeAlias = "Callable[..., Awaitable[Context]]"
"""`await mount(*rows)` → a mounted root, disposed when the test ends.

The type of the root conftest's `mount` fixture, declared **here** rather than
there so a test can annotate its parameter without `from conftest import
MountProfile`. That import is the shape issue 32 removed: `conftest` resolved to
whichever tree won the name under full collection.

It is also the lever that closed the row. 673 test signatures said
`mount: Any`, so `await mount()` was `Any`, so every `ctx.<service>` read off it
was invisible to the checker — 1,581 of them, which is what issue 32 counted.
Naming the type is what let mypy find them.
"""

__all__ = [
    "FAKE_OPTIONS",
    "REPLAY_ROW",
    "FakeAdapter",
    "MountProfile",
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
    "as_kind",
    "assert_fold_laws",
    "assistant_payload",
    "block_text",
    "boundary_for",
    "check_fold_laws",
    "not_none",
    "noted",
    "noting",
    "parked_gate",
    "plugin_payload",
    "prefix_of",
    "raising",
    "recorded_steps",
    "reference_fork",
    "report_section",
    "run_tool",
    "session_of",
    "shared_prefix",
    "simple_tool",
    "skill",
    "skill_service",
    "store_root",
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
