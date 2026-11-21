"""`ph.testing`: the builders, stubs and fixtures that a test uses to set up a profile.

**No plugin row mounts anything here.** The fake adapter, the replay adapter
and the stub code runtime are rows that a deployment can mount by name. They
live with the seams that they implement (`ph.llm.fake`, `ph.llm.replay`,
`ph.seams.code_runtime_stub`), and this module re-exports them for tests. The
rest is scaffolding. No shipped module imports it, so it can depend on whatever
a test needs.

`.git` and `.jj` are deliberately not re-exported, for three reasons. They
drive real binaries. Their fixtures are useful only to a test that carries the
matching `needs_git` or `needs_jj` marker. A name `git` here would shadow the
submodule that it came from. A caller imports from the submodule, as in
`from ph.testing.git import git_repo`. The `ph-rlm` suite already did this.

@module ph.testing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

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
    EMOJI_FAMILY,
    FAKE_OPTIONS,
    SCAFFOLDING,
    FarSide,
    StoppingAgent,
    StoredLogs,
    StubAgent,
    admitted_child,
    as_kind,
    assistant_payload,
    block_text,
    boundary_for,
    code_mode_stub,
    delivered_ids,
    disk_fills_midway,
    external_tool,
    full_disk_batch,
    hold_session,
    isolated_intent_kinds,
    isolated_log_types,
    live_settles,
    llama_refusal,
    log_event,
    log_interrupted_call,
    logged_events,
    models_overlay,
    new_agent,
    not_none,
    noted,
    noting,
    open_fds,
    parked_gate,
    plugin_payload,
    raising,
    reconciled_call,
    reference_fork,
    result_text,
    resume_stored,
    revision_record,
    run_tool,
    searches_into,
    session_of,
    settle_background,
    settled,
    simple_tool,
    skill_service,
    splice_line,
    store_root,
    stored_events,
    stored_log,
    stored_types,
    streamed,
    tool_result_payload,
    tool_runtime,
    unwritten,
    user_payload,
    workspace_acquired,
    workspace_acquiring,
    workspace_disposed,
    workspace_log,
    workspace_retained,
    workspace_seam,
    write_host_config,
    write_profile,
    write_reference_fork,
    write_stored_log,
)
from .diagnostics import report_section
from .folds import VerifyingFoldCache, assert_fold_laws, check_fold_laws, prefix_of
from .skills import KIND_OUTPUT, skill, write_skill
from .stub_compaction import StubCompactionEngine
from .stub_decider import StubDecider
from .stub_sandbox import StubSandboxProvider
from .stub_subagent import HoldingProvider, RefusingProvider, StubSubagentProvider
from .stub_workspace import (
    StubCheckpointingProvider,
    StubWorkspaceProvider,
    acquire_for_role,
)

MountProfile: TypeAlias = "Callable[..., Awaitable[Context]]"
"""`await mount(*rows)` returns a mounted root, which is disposed when the test ends.

This is the type of the `mount` fixture of the root conftest. It is declared
**here** so that a test can annotate its parameter without
`from conftest import MountProfile`. Issue 32 removed that form of import,
because `conftest` resolved to whichever tree won the name under full
collection.

The alias also let issue 32 be closed. 673 test signatures said `mount: Any`,
so `await mount()` was `Any`, and the checker could not see any `ctx.<service>`
read from it. Issue 32 counted 1,581 such reads. With the type named, mypy
found them.
"""

ReapedHost: TypeAlias = "Callable[..., Path]"
"""`reaped_host(...)` returns a `$PH_HOME` whose daemon is reaped when the test ends.

The alias is here for the same reason as `MountProfile`, and it had spread
across the same five files. The root conftest declares the fixture, and four
suites had each written the alias again to avoid `from conftest import`, which
is the import that issue 32 removed.
"""

__all__ = [
    "EMOJI_FAMILY",
    "FAKE_OPTIONS",
    "KIND_OUTPUT",
    "REPLAY_ROW",
    "SCAFFOLDING",
    "FakeAdapter",
    "FarSide",
    "HoldingProvider",
    "MountProfile",
    "ReapedHost",
    "RecordedStep",
    "RefusingProvider",
    "ReplayAdapter",
    "StoppingAgent",
    "StoredLogs",
    "StubAgent",
    "StubCheckpointingProvider",
    "StubCodeRuntime",
    "StubCompactionEngine",
    "StubDecider",
    "StubSandboxProvider",
    "StubSubagentProvider",
    "StubWorkspaceProvider",
    "VerifyingFoldCache",
    "acquire_for_role",
    "admitted_child",
    "anthropic_reply",
    "as_kind",
    "assert_fold_laws",
    "assistant_payload",
    "block_text",
    "boundary_for",
    "check_fold_laws",
    "code_mode_stub",
    "delivered_ids",
    "disk_fills_midway",
    "external_tool",
    "full_disk_batch",
    "hold_session",
    "isolated_intent_kinds",
    "isolated_log_types",
    "live_settles",
    "llama_refusal",
    "log_event",
    "log_interrupted_call",
    "logged_events",
    "models_overlay",
    "new_agent",
    "not_none",
    "noted",
    "noting",
    "open_fds",
    "parked_gate",
    "plugin_payload",
    "prefix_of",
    "raising",
    "reconciled_call",
    "recorded_steps",
    "reference_fork",
    "report_section",
    "result_text",
    "resume_stored",
    "revision_record",
    "run_tool",
    "searches_into",
    "session_of",
    "settle_background",
    "settled",
    "shared_prefix",
    "simple_tool",
    "skill",
    "skill_service",
    "splice_line",
    "store_root",
    "stored_events",
    "stored_log",
    "stored_types",
    "streamed",
    "text_chunks",
    "text_script",
    "tool_call_chunks",
    "tool_result_payload",
    "tool_runtime",
    "unwritten",
    "user_payload",
    "workspace_acquired",
    "workspace_acquiring",
    "workspace_disposed",
    "workspace_log",
    "workspace_retained",
    "workspace_seam",
    "write_host_config",
    "write_profile",
    "write_reference_fork",
    "write_skill",
    "write_stored_log",
]
