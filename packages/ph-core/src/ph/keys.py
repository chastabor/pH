"""Every service `ph-core` provides, as a typed key.

One key per `ctx.provide(...)` in this package — `test_keys` holds the two lists
against each other — so a row names what it needs (`inject=[LLM, SESSIONS]`),
provides what it has (`ctx.provide(FS, service)`) and reads what it was given
(`ctx.require(TOOLS)`) through a symbol the checker follows, instead of a string
it cannot.

**Why one module.** The obvious home for a key is beside the service it names,
and it is the wrong one for the reason `ph.seams._registry` gives about
ownership: a *name* is the registry's vocabulary, not the service's property.
`ctx.provide("fs", …)` is a fact about cordis's namespace that `FsService` never
mentions — so the concern decides the home, and `ph.seams._names` completes it:
shared names live in one module precisely so two copies cannot disagree, which
`test_keys` then makes enforceable in both directions.

The import graph agrees rather than deciding it. Keys beside their services
measures at **124 new runtime edges**, eighteen of them seam→seam
(`workspace → fs`, `subprocess → shell`) that today's layering never needed.
Here the service types are imported under `TYPE_CHECKING` only: the annotation
is a string mypy reads and the value is a `ServiceKey("name")`, so this module
adds no edge a consumer did not already have. It is not edge-*free* — `MOUNT` and
`PROJECT_ROOT` are re-exported from `.cordis.loader`, which `ph.cordis` already
imports, so the measured cost is zero. Those two are the exception in spelling
only: both are cordis's own, declared beside the `Profile.mount` that provides
them, and listed here so a reader has one list.

The downstream packages declare their own the same way: `ph_rlm.keys` for the
five the RLM bundle provides, and beside the service where a package has one.

@module ph.keys
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cordis.key import ServiceKey
from .cordis.loader import MOUNT, PROJECT_ROOT

if TYPE_CHECKING:
    from .agent.registry import AgentRegistry
    from .agent.types import AgentDriver
    from .llm.adapter import LlmRuntime
    from .llm.fake import FakeAdapter
    from .llm.replay import ReplayAdapter
    from .persistence.protocol import SessionPersistence
    from .seams.approval import ApprovalService
    from .seams.attachments import AttachmentStore
    from .seams.code_runtime import CodeRuntimeSeam
    from .seams.code_runtime_stub import StubCodeRuntime
    from .seams.commands import CommandRegistry
    from .seams.compaction import CompactionSeam
    from .seams.containment import ContainmentService
    from .seams.credentials import CredentialService
    from .seams.diagnostics import DiagnosticsRegistry
    from .seams.fs import FsService
    from .seams.goals import GoalService
    from .seams.invariants import InvariantRegistry
    from .seams.jobs import JobService
    from .seams.permission_presets import PermissionPresetService
    from .seams.sandbox import SandboxSeam
    from .seams.schedule import ScheduleService
    from .seams.settings import SettingsService
    from .seams.shell import ShellService
    from .seams.skills import SkillService
    from .seams.spill import SpillStore
    from .seams.subagents import SubagentPresetService, SubagentService
    from .seams.subprocess import SubprocessService
    from .seams.telemetry import SessionTelemetry
    from .seams.token_meter import TokenMeter
    from .seams.tui_screens import TuiScreenRegistry
    from .seams.tui_status import TuiStatusRegistry
    from .seams.uploads import UploadRegistry
    from .seams.user_questions import UserQuestionService
    from .seams.workspace import WorkspaceSeam
    from .session.store import SessionStore
    from .system_prompt.assembly import SystemPromptService
    from .tools.registry import ToolRuntime

__all__ = [
    "AGENT",
    "AGENTS",
    "APPROVAL",
    "ATTACHMENTS",
    "CODE_RUNTIME",
    "CODE_RUNTIME_STUB",
    "COMMANDS",
    "COMPACTION",
    "CONTAINMENT",
    "CREDENTIALS",
    "DIAGNOSTICS",
    "FS",
    "GOALS",
    "INVARIANTS",
    "JOBS",
    "LLM",
    "LLM_FAKE",
    "LLM_REPLAY",
    "MOUNT",
    "PERMISSION_PRESETS",
    "PROJECT_ROOT",
    "SANDBOX",
    "SCHEDULE",
    "SESSIONS",
    "SESSION_PERSISTENCE",
    "SESSION_TELEMETRY",
    "SETTINGS",
    "SHELL",
    "SKILLS",
    "SPILL_STORE",
    "SUBAGENTS",
    "SUBAGENT_PRESETS",
    "SUBPROCESS",
    "SYSTEM_PROMPT",
    "TOKEN_METER",
    "TOOLS",
    "TUI_SCREENS",
    "TUI_STATUS",
    "UPLOADS",
    "USER_QUESTIONS",
    "WORKSPACE",
]

AGENT: ServiceKey[AgentDriver] = ServiceKey("agent")
AGENTS: ServiceKey[AgentRegistry] = ServiceKey("agents")
APPROVAL: ServiceKey[ApprovalService] = ServiceKey("approval")
ATTACHMENTS: ServiceKey[AttachmentStore] = ServiceKey("attachments")
CODE_RUNTIME: ServiceKey[CodeRuntimeSeam] = ServiceKey("code_runtime")
CODE_RUNTIME_STUB: ServiceKey[StubCodeRuntime] = ServiceKey("code_runtime_stub")
COMMANDS: ServiceKey[CommandRegistry] = ServiceKey("commands")
COMPACTION: ServiceKey[CompactionSeam] = ServiceKey("compaction")
CONTAINMENT: ServiceKey[ContainmentService] = ServiceKey("containment")
CREDENTIALS: ServiceKey[CredentialService] = ServiceKey("credentials")
DIAGNOSTICS: ServiceKey[DiagnosticsRegistry] = ServiceKey("diagnostics")
FS: ServiceKey[FsService] = ServiceKey("fs")
GOALS: ServiceKey[GoalService] = ServiceKey("goals")
INVARIANTS: ServiceKey[InvariantRegistry] = ServiceKey("invariants")
JOBS: ServiceKey[JobService] = ServiceKey("jobs")
LLM: ServiceKey[LlmRuntime] = ServiceKey("llm")
LLM_FAKE: ServiceKey[FakeAdapter] = ServiceKey("llm_fake")
LLM_REPLAY: ServiceKey[ReplayAdapter] = ServiceKey("llm_replay")
PERMISSION_PRESETS: ServiceKey[PermissionPresetService] = ServiceKey("permission_presets")
SANDBOX: ServiceKey[SandboxSeam] = ServiceKey("sandbox")
SCHEDULE: ServiceKey[ScheduleService] = ServiceKey("schedule")
SESSION_PERSISTENCE: ServiceKey[SessionPersistence] = ServiceKey("session_persistence")
SESSION_TELEMETRY: ServiceKey[SessionTelemetry] = ServiceKey("session_telemetry")
SESSIONS: ServiceKey[SessionStore] = ServiceKey("sessions")
SETTINGS: ServiceKey[SettingsService] = ServiceKey("settings")
SHELL: ServiceKey[ShellService] = ServiceKey("shell")
SKILLS: ServiceKey[SkillService] = ServiceKey("skills")
SPILL_STORE: ServiceKey[SpillStore] = ServiceKey("spill_store")
SUBAGENT_PRESETS: ServiceKey[SubagentPresetService] = ServiceKey("subagent_presets")
SUBAGENTS: ServiceKey[SubagentService] = ServiceKey("subagents")
SUBPROCESS: ServiceKey[SubprocessService] = ServiceKey("subprocess")
SYSTEM_PROMPT: ServiceKey[SystemPromptService] = ServiceKey("system_prompt")
TOKEN_METER: ServiceKey[TokenMeter] = ServiceKey("token_meter")
TOOLS: ServiceKey[ToolRuntime] = ServiceKey("tools")
TUI_SCREENS: ServiceKey[TuiScreenRegistry] = ServiceKey("tui_screens")
TUI_STATUS: ServiceKey[TuiStatusRegistry] = ServiceKey("tui_status")
UPLOADS: ServiceKey[UploadRegistry] = ServiceKey("uploads")
USER_QUESTIONS: ServiceKey[UserQuestionService] = ServiceKey("user_questions")
WORKSPACE: ServiceKey[WorkspaceSeam] = ServiceKey("workspace")
