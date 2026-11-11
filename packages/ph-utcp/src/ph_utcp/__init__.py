"""`utcp` — tools published as UTCP manuals, as tools a pH agent can call.

One row. Its config lists manuals — each a UTCP *manual call template*, the
object any UTCP harness registers — and another row can add one at mount through
`ctx.utcp.add`. For each manual the row runs discovery, and every tool the manual
names becomes a registered pH tool: in the prompt, through the pipeline, in the
log, and in Code Mode's listing like any other.

## A UTCP tool sees what its agent sees

A `cli` tool's command is started by `ctx.shell` for the agent making the call
(`_library.PhCliProtocol`): in that agent's workspace, under that agent's
sandbox, ended by the call's own cancellation. So a UTCP tool is bounded the way
`bash` is bounded — by the sandbox, and not by `permissions-fs`, which governs
calls through `ctx.fs` and says so of itself. A manual whose tool keeps state
outside the workspace names those paths in `writable`, or under confinement the
tool cannot write them.

## Read-only bars the workspace, not a tool's working state

A tool that builds something *about* a tree — an index, a graph — changes no
file of it. A manual names a directory for that (`state`), and each call is told
where it is for the agent making it: `$PH_TOOL_STATE`, a directory of that
agent's scratch. The scratch is writable under every sandbox mode and the
workspace only under `workspace-write`, so the tool works in a read-only session
and cannot write the repository there; and it is per session and agent, so an
agent that restarts finds what it built. `writable` is for what agents share — a
grammar cache, a model's weights — and a grant that overlaps the workspace is
refused (`ph.seams.sandbox.grants_apart`), so no manual reopens the tree. A
confined command's temporary files go in the scratch too (`TMPDIR`, set by the
shell seam), for the same reason.

## A host, for a manual whose tools are slow to start

A manual can name a **host** (`Manual.host`): a command that stays up and answers
the manual's tools on a socket, so what they load — an embedding model — is
loaded once, and a long piece of work can go on past the call that asked for it.
The row keeps one running for each tree an agent works in, started as every
command is: by `ctx.shell`, for that agent, in its workspace and under its
sandbox. It is a job of that agent's (`ctx.jobs`), so it ends when the agent
does and when the manual's row unmounts; it leaves by itself when idle, and the
next call starts another. Each call is told the socket in `$PH_TOOL_HOST`, and a
command that finds nobody there runs by itself — a host is never required.

While a host works in the background the footer says how far it has got, and
when it finishes the agent it works for is told, with a note that waits for that
agent's next step and starts no turn.

**One host a tree, not one an agent.** A child builds its own only when it has a
tree it writes for keeps — a worktree, an overlay. One that works in its
parent's checkout, or was granted only to read the project, uses its parent's
host and its parent's state (`UtcpSeam.owner_of`): the index is of the tree, and
a copy for every child of a read-only fan-out would be the same documents
embedded again and a model held in memory for each.

## What UTCP cannot say, the row's config says

A manual carries a name, a description and two schemas. It has no word for "two
of these may overlap" or "this changes nothing", and pH's scheduler and policy
rows read exactly those. Each defaults to the careful answer — not safe to
overlap, has effects — and a manual's `tools` table states otherwise per tool.

## A missing provider is a missing capability, not a failed start

A manual whose discovery fails — the command is not installed, the endpoint is
down, the protocol is not one this deployment allows — is **skipped**: a warning
in the log, a line in `phern doctor`, and a note in the system prompt so the
model is not left guessing why a skill names a tool it cannot find. The mount
goes ahead with the manuals that answered.

## Not stored

A template's `${VAR}` references are resolved by the UTCP client from the
environment when a call is made. Nothing this row holds is a credential.

@module ph_utcp
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import time
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import Field

from ph.agent.types import AgentHandle, PreStepDecision, PreStepRequest
from ph.cancel import Canceled, Cancellation
from ph.cordis import Context, Disposer, Next, ServiceKey, plugin
from ph.json import as_obj, as_str, dumps
from ph.keys import AGENTS, FS, JOBS, SHELL, SYSTEM_PROMPT, TOOLS, TUI_STATUS
from ph.llm.types import CONTEXT_SUMMARY_MAX_CHARS, PluginSource, create_user_message
from ph.paths import RuntimeDirError, file_stamp, resolve_roots
from ph.seams._registry import contribute_item
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.jobs import UNSETTLED, Job
from ph.seams.sandbox import allowance_path, grants_apart
from ph.seams.shell import ShellResult
from ph.seams.tui_status import StatusField, StatusReading
from ph.seams.workspace import project_access, scratch_of, workspace_of
from ph.session import Session
from ph.system_prompt.assembly import ORDER_TOOL_GUIDANCE, AssembleContext, PromptSection
from ph.text import count_of
from ph.toolcli import (
    HOST_ENV,
    PROGRESS_FILE,
    STATE_ENV,
    Progress,
    read_envelope,
    read_progress,
    value_schema,
)
from ph.tools.definition import (
    ToolDefinition,
    ToolRunContext,
    define_tool,
    text_content,
)
from ph.tools.errors import HarnessError
from ph.tools.json_schema import validate_json_schema_value
from ph.tools.presentation import CardKind, simple_views
from ph.wire import WireModel

from ._library import (
    Call,
    CallTemplateSerializer,
    Tool,
    UtcpClient,
    calling,
    install,
    require_ran,
)

__all__ = [
    "BUNDLE",
    "UTCP",
    "Config",
    "Host",
    "Manual",
    "ManualState",
    "ToolTraits",
    "UtcpSeam",
    "apply",
    "command_said",
]

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""This distribution's profile layer, for the `ph.bundles` group."""

log = logging.getLogger("ph_utcp")

UTCP_FAILED = "UTCP_FAILED"
UTCP_BAD_ARGUMENTS = "UTCP_BAD_ARGUMENTS"
UTCP_BAD_ANSWER = "UTCP_BAD_ANSWER"

_NOT_A_WORD = re.compile(r"\W")
_CLIP = 600
"""How much of an answer that was not what the tool declared is quoted back."""

_HOST_STARTS_S = 15.0
"""How long a call waits for a host it started to listen, before running alone."""

_WATCH_EVERY_S = 0.5
"""How often a running host's progress is looked at while a run works. Kept
here, in memory, so the footer — drawn far more often — never reads a file."""

_WATCH_IDLE_S = 2.0
"""How often it is looked at between runs: a new one is noticed this late."""


# ------------------------------------------------------------------- config ----


class ToolView(WireModel):
    """How a front end draws one tool's card: a title over one salient argument."""

    card: CardKind = "generic"
    title: str
    argument: str


class ToolTraits(WireModel):
    """What a manual cannot say about one of its tools. Every default is the
    careful answer."""

    concurrency_safe: bool = False
    """Two calls may overlap in one batch."""
    effect_free: bool = False
    """A call changes nothing."""
    self_limits: bool = False
    """The tool bounds its own output and offers a way to page."""
    arguments_disposable: bool = False
    """The arguments are a payload delivered, not something the model refers back to."""
    timeout_ms: int | None = None
    """This tool's budget, over the row's."""
    view: ToolView | None = None


class Host(WireModel):
    """A command that stays up and answers a manual's tools: `ph.toolcli`'s `serve`.

    The row adds `--socket`, `--root` and `--idle` to `command`, and `--warm` for
    each of `warm` when it starts the host ahead of any call; the state directory
    it is told as each call is, in `$PH_TOOL_STATE`."""

    command: list[str]
    """The command, up to and including `serve`."""
    label: str
    """What the footer and the completion note call it: `text index`."""
    warm: list[str] = Field(default_factory=list)
    """Tools the host calls once when it is started `eager`ly: an index tool,
    which in a host answers at once and works in the background."""
    eager: bool = True
    """Start it for a root agent at that agent's first step, and not at its first
    call — so an index is being built before anything asks for it. A child that
    works in a tree of its own gets its host at its first call either way: a
    throwaway worktree may never ask."""
    idle_s: float = 900.0
    """How long the host stays up with nothing to do."""


class Manual(WireModel):
    """One UTCP manual, and how pH runs what it names."""

    template: dict[str, Any]
    """The manual call template, as any UTCP harness would register it. Its `name`
    is the manual's name."""
    names: Literal["prefixed", "bare"] = "prefixed"
    """How its tools are named in pH. `prefixed` is `<manual>_<tool>`, which cannot
    collide with a tool pH or another manual registers. `bare` is the tool's own
    name, for a manual that is the only source of those tools."""
    envelope: bool = False
    """The tool answers `{ok, value, text, error}` — `ph.toolcli`'s envelope.

    Then the pH tool's value is the envelope's `value`, its content is the
    envelope's `text`, and a failure is raised with the envelope's code: the tool
    reads to an agent exactly as it would registered in-process. Stated rather
    than detected, because an object with an `ok` key is not a promise."""
    state: str = Field("", pattern=r"^([A-Za-z0-9][A-Za-z0-9._-]*)?$")
    """A directory name for this manual's working state, or empty for none.

    Each call's command is then told `$PH_TOOL_STATE=<the calling agent's
    scratch>/<state>`: writable under every sandbox mode, apart from the
    workspace, and the same directory for the same agent after a restart. One
    name, not a path — where it goes is the agent's, and not the manual's to say."""
    writable: list[str] = Field(default_factory=list)
    """Paths outside the workspace this manual's commands may write: what agents
    share. Honored under every sandbox mode; one that overlaps the workspace is
    refused."""
    env: dict[str, str] = Field(default_factory=dict)
    """Variables set on this manual's commands, over what its templates say."""
    pass_env: list[str] = Field(default_factory=list)
    """Variables this manual's commands are handed from the harness's own
    environment, by name — a credential.

    A command's environment starts as the deployment's scrubbed one, which drops
    every variable whose name looks like a credential, and a template cannot
    name its way past that. This is how a deployment says one may pass: named in
    a profile, read each time a command is started, and held nowhere."""
    host: Host | None = None
    """A command that stays up and answers this manual's tools, or none."""
    tools: dict[str, ToolTraits] = Field(default_factory=dict)
    """Per tool, by the tool's own name."""


class Config(WireModel):
    """Row config for `utcp`."""

    manuals: list[Manual] = Field(default_factory=list)
    protocols: list[str] = Field(default_factory=lambda: ["cli"])
    """The call template types a manual may use.

    `cli` alone by default, because it is the one this row runs through pH's own
    doors. Another type — `http`, `mcp` — runs inside the harness process, by the
    UTCP library's own client, outside the sandbox and its egress rules; naming
    it here is how a deployment says it means that."""
    timeout_ms: int = 120_000
    """How long one call's command may run, unless the tool's own entry says."""
    discovery_timeout_ms: int = 60_000
    """How long a manual's discovery command may run."""


# --------------------------------------------------------------------- seam ----


UTCP: ServiceKey[UtcpSeam] = ServiceKey("utcp")
"""The UTCP manuals this mount holds, and the door another row adds one through."""


@dataclass(frozen=True, slots=True)
class ManualState:
    """What became of one manual: the tools it gave, or why it gave none."""

    tools: tuple[str, ...] = ()
    """The pH names registered."""
    problems: tuple[str, ...] = ()
    """Why the manual, or one of its tools, is not available."""

    @property
    def available(self) -> bool:
        return bool(self.tools)

    def said(self) -> str:
        """The tools, as a row's `phern doctor` line says them, and why any are
        missing."""
        if self.available and not self.problems:
            return f"{', '.join(self.tools)} (through UTCP)"
        reasons = "; ".join(self.problems) or "the manual named no tools"
        return f"{', '.join(self.tools) or 'unavailable'} — {reasons}"


def command_said(program: str, verb: str, result: ShellResult) -> str:
    """What a manual's own command said, for a person: its output, or that it
    failed and how."""
    said = result.stdout.strip() or result.stderr.strip()
    if result.exit_code != 0:
        return f"{program} {verb} failed (exit {result.exit_code}): {said}"
    return said


@dataclass(slots=True)
class _Hosted:
    """One running host: whose tree it serves, where it listens, how far it is."""

    name: str
    """The manual it answers for."""
    host: Host
    owner: AgentHandle
    socket: Path
    home: Path | None
    """Its state directory, where it reports progress."""
    job: Job | None = None
    progress: Progress | None = None
    """What it last reported, as `_watch` read it: nothing until it reports,
    since a report an earlier host left in the directory is not its own."""
    since: float = field(default_factory=time.time)
    """When it was started: a report of a run that began before is one an
    earlier host left in the directory."""
    told: set[float] = field(default_factory=set)
    """The runs, by when each began, whose end its agent has been told of."""
    listening: bool = False
    """Its socket has been seen. Asked once, and a field after."""

    @property
    def running(self) -> bool:
        return self.job is not None and self.job.state in UNSETTLED


@dataclass(slots=True)
class UtcpSeam:
    """The service published as `ctx.utcp`."""

    ctx: Context
    config: Config
    client: Any
    """The UTCP client. `Any` because the library ships no types."""
    states: dict[str, ManualState] = field(default_factory=dict)
    manuals: dict[str, Manual] = field(default_factory=dict)
    """The manuals that gave tools, by name — what a host is started from."""
    hosts: dict[tuple[str, str], _Hosted] = field(default_factory=dict)
    """Running hosts, by manual and the id of the agent whose tree each serves."""
    socket_dir: Path | None = None
    """Where hosts — and anything else a manual's command asks over a socket —
    listen: `$PH_RUNTIME/utcp`, made once at mount. `None` on a deployment with no
    usable runtime directory, which then runs every call alone."""
    warmed: set[tuple[str, str]] = field(default_factory=set)
    """The trees, by manual and owner, whose host has been started unasked once.
    Once: a host that leaves when idle is started again by the next call, and
    not by the next step, which would index the tree again for nobody."""

    # ----------------------------------------------------------- registering --

    async def add(self, manual: Manual, *, scope: Context | None = None) -> ManualState:
        """Discover `manual` and register its tools on `scope`. Never raises for a
        manual that cannot be reached: the state says why, and so does the log.

        `scope` is who the manual belongs to, so a row that adds its own gets its
        tools unregistered, and the manual forgotten, when it unmounts. Adding a
        name that is already held and gave tools changes nothing and answers with
        the state there is — the tools are registered, and a second discovery
        would only find the registry refusing each of them by name.
        """
        name = _NOT_A_WORD.sub("_", as_str(manual.template.get("name")))
        if not name:
            state = ManualState(problems=("its template has no `name`",))
            log.warning("ph_utcp: a manual was skipped: its template has no `name`")
            return state
        held = self.states.get(name)
        if held is not None and held.available:
            return held
        made, unmade = await anyio.to_thread.run_sync(
            _made, manual.writable, allowance_path(str(self.ctx.require(FS).root_for(None)))
        )
        manual = manual.model_copy(update={"writable": made})
        state = await self._discovered(name, manual, scope)
        if unmade:
            state = ManualState(tools=state.tools, problems=(*state.problems, *unmade))
        self.states[name] = state
        if state.available:
            self.manuals[name] = manual
        for problem in state.problems:
            log.warning("ph_utcp: %s: %s", name, problem)

        async def forget() -> None:
            """What the owning scope's unwinding takes back: the state, so the name
            can be added again, the hosts it started, and the client's own record
            of the manual."""
            if self.states.get(name) is state:
                del self.states[name]
                self.manuals.pop(name, None)
                for key in [key for key in self.hosts if key[0] == name]:
                    job = self.hosts.pop(key).job
                    if job is not None:
                        job.cancel()
            try:
                await self.client.deregister_manual(name)
            except Exception:
                log.debug("ph_utcp: %s was not deregistered", name, exc_info=True)

        await (scope or self.ctx).effect(lambda: forget, label=f"utcp-manual({name})")
        return state

    async def _discovered(self, name: str, manual: Manual, scope: Context | None) -> ManualState:
        template = manual.template
        protocol = as_str(template.get("call_template_type"))
        if protocol not in self.config.protocols:
            return ManualState(
                problems=(
                    f'it uses the "{protocol}" protocol, which this deployment does not '
                    f"allow (the `utcp` row's `protocols` is {self.config.protocols})",
                )
            )
        try:
            call_template = CallTemplateSerializer().validate_dict({**template, "name": name})
            if await self.client.config.tool_repository.get_manual(name) is not None:
                await self.client.deregister_manual(name)
            call = self._call(manual, timeout_ms=self.config.discovery_timeout_ms)
            with calling(call):
                result = await self.client.register_manual(call_template)
            if protocol == "cli":
                require_ran(call, f"discovery of {name}")
        except Exception as error:
            log.debug("ph_utcp: %s was not discovered", name, exc_info=True)
            return ManualState(problems=(f"discovery failed: {error}",))
        if not result.success:
            reasons = "; ".join(str(one) for one in result.errors) or "no reason given"
            return ManualState(problems=(f"discovery failed: {reasons}",))

        registered: list[str] = []
        problems: list[str] = []
        for tool in result.manual.tools:
            definition = self._definition(name, manual, tool)
            try:
                self.ctx.require(TOOLS).register(definition, scope=scope)
            except ValueError as error:
                problems.append(f"{definition.name} was not registered: {error}")
            else:
                registered.append(definition.name)
        if not registered and not problems:
            problems.append("its manual names no tools")
        return ManualState(tools=tuple(registered), problems=tuple(problems))

    def _call(
        self,
        manual: Manual,
        *,
        timeout_ms: int,
        agent: AgentHandle | None = None,
        owner: AgentHandle | None = None,
        signal: Cancellation | None = None,
        host: Path | None = None,
    ) -> Call:
        """How one of `manual`'s commands runs for `agent`, whose tree's owner is
        `owner`: asked once by the caller, because each of these needs it."""
        env, writable = self._surroundings(manual, agent, owner)
        if host is not None:
            env = {**env, HOST_ENV: str(host)}
        return Call(
            shell=self.ctx.require(SHELL),
            root=self._tree(manual, agent, owner),
            timeout_ms=timeout_ms,
            env=env,
            writable=writable,
            agent=agent,
            signal=signal,
        )

    def _surroundings(
        self, manual: Manual, agent: AgentHandle | None, owner: AgentHandle | None
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        """What a command of this manual run for `agent` is given: its variables,
        and the paths outside the workspace it may write. One derivation, for a
        tool call, the host that answers it and a person's provisioning command."""
        env = dict(manual.env)
        for name in manual.pass_env:
            # From the harness's own environment and not the scrubbed one, which
            # is the point: read now, set on this one command, kept nowhere.
            value = os.environ.get(name)
            if value:
                env[name] = value
        writable = list(manual.writable)
        state = self._state(manual, owner)
        if state is not None:
            env[STATE_ENV] = str(state)
            held = workspace_of(self.ctx, agent)
            if held is not None and not state.is_relative_to(held.scratch):
                # Its tree's owner's state, and not in its own scratch: granted,
                # for the call that finds no host and does the work itself.
                writable.append(str(state))
        if manual.host is not None and self.socket_dir is not None:
            # Where a host listens. A caller connects there and a host binds
            # there, and both are confined.
            writable.append(str(self.socket_dir))
        return env, tuple(writable)

    def _tree(self, manual: Manual, agent: AgentHandle | None, owner: AgentHandle | None) -> Path:
        """Where `manual`'s command runs for `agent`: the tree its state is about —
        the owner's — for a manual that keeps state, and the agent's own root for
        one that does not.

        The working directory is how a command is told its tree (`ph.toolcli`'s
        default `--root`), so a child that shares its parent's index works on its
        parent's tree rather than indexing its own empty directory into that
        index and sweeping every document out as gone."""
        fs = self.ctx.require(FS)
        return fs.root_for(owner if manual.state else agent)

    def run_command(
        self, manual: Manual, argv: Sequence[str], agent: AgentHandle | None, *, timeout_ms: int
    ) -> Awaitable[ShellResult]:
        """Run one of `manual`'s own commands for `agent`, as a person asked —
        `install`, `status` — with what a call of its tools is given: the
        variables, the state directory, the tree.

        Unconfined, as a provisioning command a person types is: `install` needs
        the network, and nobody's model chose it."""
        owner = None if agent is None else self.owner_of(agent)
        env, _ = self._surroundings(manual, agent, owner)
        return self.ctx.require(SHELL).run_argv(
            tuple(argv),
            cwd=self._tree(manual, agent, owner),
            timeout_ms=timeout_ms,
            extra_env=env,
        )

    def owner_of(self, agent: AgentHandle) -> AgentHandle:
        """The agent whose tree `agent`'s tools are about: itself, or the nearest
        ancestor whose tree it shares.

        What a tool builds about a tree belongs to the tree, so the question is
        whether a child has a tree of its own to build about:

        * **It works in its parent's checkout** (`shared`, the same root). One
          tree, so one index and one host.
        * **It was granted only to read the project** — `readonly-scratch`, whose
          root is an empty directory beside a repository it cannot write, and the
          two ephemeral kinds, whose writes reach nobody. It has nothing of its
          own worth indexing, and a fan-out of such children is the case this
          rule is for: they ask their parent's index, and none of them embeds a
          document or holds a model.
        * **It has a tree it writes for keeps** — a `worktree`, an `overlay`. Its
          files diverge from its parent's, so it is the owner of that tree and
          builds its own.

        `project_access` is the seam's own reading of a kind, the one a spawn
        records as what the child was granted."""
        own = workspace_of(self.ctx, agent)
        while own is not None:
            parent = self._parent(agent)
            theirs = None if parent is None else workspace_of(self.ctx, parent)
            if parent is None or theirs is None:
                break
            if theirs.root != own.root and project_access(own.kind) == "write":
                break
            agent, own = parent, theirs
        return agent

    def _parent(self, agent: AgentHandle) -> AgentHandle | None:
        """The agent that delegated to `agent`, if it is a sub-agent and that
        agent is still here. An agent's id is its session's."""
        session = agent.session
        wanted = None if session is None else session.header.delegating_parent
        registry = self.ctx.get(AGENTS)
        return None if wanted is None or registry is None else registry.get(wanted)

    def _state(self, manual: Manual, owner: AgentHandle | None) -> Path | None:
        """The directory `manual` named, in `owner`'s scratch.

        `None` when the manual names none, and when there is no agent to have a
        scratch — discovery, or a call made for the deployment. The command then
        uses its own default, and nothing confines it: `ctx.shell` confines a
        command only for an agent that holds a workspace, and that agent has a
        scratch. The directory is the command's to make; the scratch is there."""
        if not manual.state or owner is None:
            return None
        session = owner.session
        if session is not None:
            return scratch_of(self.ctx, session.id, owner) / manual.state
        held = workspace_of(self.ctx, owner)
        return None if held is None else held.scratch / manual.state

    # ---------------------------------------------------------------- hosts --

    async def _host_for(
        self, name: str, owner: AgentHandle | None, *, warm: bool = False, wait: bool = True
    ) -> Path | None:
        """The socket of the host answering manual `name` for `owner`'s tree,
        started if there is none. `None` when there is no host to be had — the
        manual names none, there is no agent, no job runner or no runtime
        directory, or the host did not come up — and the call then runs alone.
        `wait=False` starts it and answers at once, for a caller that will not
        use it.

        Never raises: a host is how a call is made faster, and a call that could
        have been answered without one must not fail for want of it.
        """
        manual = self.manuals.get(name)
        jobs = self.ctx.get(JOBS)
        if manual is None or manual.host is None or owner is None:
            return None
        if jobs is None or self.socket_dir is None:
            return None
        key = (name, owner.id)
        hosted = self.hosts.get(key)
        if hosted is None or not hosted.running:
            hosted = self.hosts[key] = _Hosted(
                name=name,
                host=manual.host,
                owner=owner,
                # Short, because a socket's address is at most 108 bytes, and
                # random, because two roots of one daemon share this directory.
                socket=self.socket_dir / f"{secrets.token_hex(6)}.sock",
                home=self._state(manual, owner),
            )
            try:
                hosted.job = await jobs.start(
                    kind="utcp-host",
                    label=f"{manual.host.label} for {owner.id}",
                    run=partial(self._hosting, hosted, manual, warm),
                    # The owner's: the host ends when the agent whose tree it
                    # serves does. Stopped as well when the manual's row unmounts.
                    scope=owner.ctx,
                )
            except Exception:
                log.warning("ph_utcp: no host for %s was started", name, exc_info=True)
                self.hosts.pop(key, None)
                return None
        if not wait:
            return None
        return hosted.socket if await self._listening(hosted) else None

    async def _listening(self, hosted: _Hosted) -> bool:
        """Wait, briefly, for `hosted` to listen. A host is mounting its rows for
        its first fraction of a second; a call that did not wait would run alone
        beside it and load what the host is about to."""
        if hosted.listening:
            return hosted.running
        with anyio.move_on_after(_HOST_STARTS_S):
            while hosted.running:
                if await anyio.to_thread.run_sync(hosted.socket.exists):
                    hosted.listening = True
                    return True
                await anyio.sleep(0.02)
        return False

    async def _hosting(self, hosted: _Hosted, manual: Manual, warm: bool, job: Job) -> None:
        """One host's life: the command, run for its tree's owner until it leaves
        or its job is canceled, and its progress read while it runs."""
        host, owner = hosted.host, hosted.owner
        root = self._tree(manual, owner, owner)
        env, writable = self._surroundings(manual, owner, owner)
        argv = [*host.command, "--socket", str(hosted.socket), "--root", str(root)]
        argv += ["--idle", f"{host.idle_s:g}"]
        if warm:
            for tool in host.warm:
                argv += ["--warm", tool]
        inherited = manual.template.get("inherit_env_vars")
        try:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(self._watch, hosted)
                result = await self.ctx.require(SHELL).run_argv(
                    tuple(argv),
                    cwd=root,
                    agent=owner,
                    # No limit: it leaves when it is idle, and its job ends it.
                    timeout_ms=None,
                    signal=job.token,
                    keep_env=inherited if isinstance(inherited, list) else None,
                    extra_env=env,
                    writable_extra=writable,
                )
                tasks.cancel_scope.cancel()
            if result.exit_code != 0 and not result.canceled:
                said = command_said(f"the {hosted.name} host", "serve", result)
                log.warning("ph_utcp: %s", said[:_CLIP])
        finally:
            if self.hosts.get((hosted.name, owner.id)) is hosted:
                del self.hosts[(hosted.name, owner.id)]
            # The host removes its socket when it leaves by itself. One that was
            # ended under a sandbox is killed with it and removes nothing.
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(partial(hosted.socket.unlink, missing_ok=True))

    async def _watch(self, hosted: _Hosted) -> None:
        """Read `hosted`'s progress as it changes, and tell its agent when a piece
        of background work ends.

        Polled rather than watched: one small file, read when its stat says it
        changed — twice a second while a run works, and less often between runs —
        and the reading is kept here, so the footer never reads it. The note is
        `inject`ed: it waits for the agent's next step, starts no turn, and is in
        that agent's log.
        """
        home = hosted.home
        if home is None:
            return
        seen: object = None
        while True:
            stamp = await anyio.to_thread.run_sync(file_stamp, home / PROGRESS_FILE)
            if stamp == seen:
                # Quickly while a run works, and before the first report: a host
                # just started is about to begin one.
                last = hosted.progress
                working = last is None or last.state == "working"
                await anyio.sleep(_WATCH_EVERY_S if working else _WATCH_IDLE_S)
                continue
            seen = stamp
            found = await anyio.to_thread.run_sync(read_progress, home)
            if found is not None and found.started < hosted.since:
                # An earlier host's last report: this one has said nothing yet.
                found = None
            hosted.progress = found
            if (
                found is not None
                and found.state != "working"
                # The run says whether its end is news: a call was answered
                # "still going", or the host began it unasked. One whose caller
                # got its counts back has been told already.
                and found.owed
                and found.started not in hosted.told
            ):
                hosted.told.add(found.started)
                self._tell(hosted, found)

    def _tell(self, hosted: _Hosted, progress: Progress) -> None:
        label = hosted.host.label
        if progress.state == "failed":
            summary = f"The {label} stopped: {progress.failure}"
            text = (
                f"{summary}. It got through {progress.done} of {progress.total} "
                f"{progress.what}; what it finished is kept, and running its index "
                "tool again goes on from there."
            )
        else:
            summary = f"The {label} is finished: {progress.said()}"
            text = f"{summary}. Searches now cover all of it."
        registry = self.ctx.get(AGENTS)
        # Asked of the roster, which holds the driver: a handle is what a reader
        # is given, and has no inbox to put a message in.
        driver = None if registry is None else registry.get(hosted.owner.id)
        if driver is None:
            # An agent that has gone. The reading and the tool both still say it.
            return
        try:
            driver.inject(
                create_user_message(
                    content=[{"type": "text", "text": text}],
                    source=PluginSource(
                        plugin="ph_utcp",
                        form="notice",
                        summary=summary[:CONTEXT_SUMMARY_MAX_CHARS],
                    ),
                )
            )
        except Exception:
            log.debug("ph_utcp: %s was not told its %s ended", driver.id, label, exc_info=True)

    def reading(self, session: Session) -> StatusReading | None:
        """The footer's phrase for this session: what its hosts are building.
        From memory — `_watch` reads the files."""
        said: list[str] = []
        failed = False
        for hosted in self.hosts.values():
            owned = hosted.owner.session
            progress = hosted.progress
            if owned is None or owned.id != session.id or progress is None:
                continue
            if progress.state == "done" or not hosted.running:
                continue
            failed = failed or progress.state == "failed"
            said.append(f"{hosted.host.label} {progress.said()}")
        if not said:
            return None
        return StatusReading(text=", ".join(said), level="warning" if failed else "normal")

    async def eager(self, request: PreStepRequest, next_: Next[PreStepDecision]) -> PreStepDecision:
        """`agent/pre-step`: before a root agent's step, see that each manual
        whose host is `eager` has one — so its index is being built before the
        model asks for it. After the workspace row's listener, which is what
        gives the agent the tree the host is started in."""
        if not request.session.header.is_subagent:
            agent = request.agent
            for name, manual in list(self.manuals.items()):
                if manual.host is None or not manual.host.eager:
                    continue
                if (name, agent.id) in self.warmed:
                    continue
                self.warmed.add((name, agent.id))
                # Started, not waited for: the step is not the host's to hold up.
                await self._host_for(name, agent, warm=True, wait=False)
        return await next_()

    # ------------------------------------------------------------- the tool --

    def _definition(self, name: str, manual: Manual, tool: Tool) -> ToolDefinition:
        """One UTCP tool as a pH tool."""
        utcp_name: str = tool.name
        own = utcp_name.removeprefix(f"{name}.")
        ph_name = own if manual.names == "bare" else _NOT_A_WORD.sub("_", utcp_name)
        traits = manual.tools.get(own, ToolTraits())
        budget = traits.timeout_ms or self.config.timeout_ms
        inputs = _object_schema(tool.inputs.model_dump(by_alias=True, exclude_none=True))
        outputs: dict[str, Any] = tool.outputs.model_dump(by_alias=True, exclude_none=True)
        declared = value_schema(outputs) if manual.envelope else outputs
        confined = tool.tool_call_template.call_template_type == "cli"

        async def execute(args: object, run: ToolRunContext) -> object:
            arguments = dict(as_obj(args))
            # The raw-schema path hands the model's arguments through unchecked;
            # the tool owns validation, and this is the tool.
            violations = validate_json_schema_value(inputs, arguments)
            if violations:
                raise HarnessError("; ".join(violations), UTCP_BAD_ARGUMENTS)
            owner = None if run.agent is None else self.owner_of(run.agent)
            call = self._call(
                manual,
                timeout_ms=budget,
                agent=run.agent,
                owner=owner,
                signal=run.signal,
                host=await self._host_for(name, owner),
            )
            try:
                with calling(call):
                    answer = await self.client.call_tool(utcp_name, arguments)
                if confined:
                    require_ran(call, utcp_name)
            except (HarnessError, Canceled):
                raise
            except Exception as error:
                raise HarnessError(f"{utcp_name} failed: {error}", UTCP_FAILED) from error
            if manual.envelope:
                return self._unwrapped(utcp_name, answer, run, elsewhere=call.root)
            mismatches = validate_json_schema_value(declared, answer)
            if mismatches:
                # The usual cause is a failure: the `cli` protocol ignores exit
                # codes and returns stderr as the result, so a tool that declares
                # an object answers with a sentence. The sentence is what the
                # model needs, so it is the error rather than a schema complaint.
                raise HarnessError(
                    f"{utcp_name} did not answer in its declared shape: {_clip(answer)}",
                    UTCP_BAD_ANSWER,
                )
            return answer

        view = traits.view
        views = simple_views(view.card, view.title, view.argument) if view is not None else None
        return define_tool(
            ph_name,
            tool.description,
            parameters=inputs,
            output=declared,
            execute=execute,
            timeout_ms=budget,
            self_limits=traits.self_limits,
            arguments_disposable=traits.arguments_disposable,
            effect_free=traits.effect_free,
            is_concurrency_safe=traits.concurrency_safe,
            present_call=views["present_call"] if views is not None else None,
            present_result=views["present_result"] if views is not None else None,
        )

    def _unwrapped(
        self, name: str, answer: object, run: ToolRunContext, *, elsewhere: Path
    ) -> object:
        """An envelope's `value`, presented with its `text` as the content.

        The text was written by the tool's own `render` in its own process, from
        the arguments and the value — the projection a session would have made had
        the tool been registered here — so it is what the registry shows for this
        value (`ToolRunContext.present`), and what every post-execute row sees and
        edits. A failure is raised with the tool's own code
        (`ph.toolcli.read_envelope`).

        `elsewhere` is where the command ran. When that is not the asking agent's
        own root — a child asking the index of the tree it shares — the text says
        so, because the paths in it start there."""
        try:
            value, text = read_envelope(answer)
        except ValueError as error:
            raise HarnessError(
                f"{name} did not answer with an envelope: {_clip(answer)}", UTCP_BAD_ANSWER
            ) from error
        if text is not None:
            if elsewhere != self.ctx.require(FS).root_for(run.agent):
                text += f"\n[Paths here are relative to {elsewhere}, the tree this was built from.]"
            run.present(value, text_content(text))
        return value

    # ------------------------------------------------------------ reporting --

    def report(self) -> list[tuple[str, str]]:
        """`phern doctor`'s section: each manual, and what it gave."""
        rows: list[tuple[str, str]] = []
        for name in sorted(self.states):
            state = self.states[name]
            said = count_of(len(state.tools), "tool") if state.tools else "unavailable"
            if state.problems:
                said += " — " + "; ".join(state.problems)
            rows.append((name, said))
        return rows

    def unavailable(self) -> str:
        """The note the model reads: which manuals gave no tools, and why."""
        missing = [
            f"- `{name}`: {'; '.join(state.problems)}"
            for name, state in sorted(self.states.items())
            if not state.available
        ]
        if not missing:
            return ""
        return (
            "# Tools that are not available\n\n"
            "These tool providers were configured for this session and could not be "
            "reached, so their tools are not in your tool list. Do not call them; say "
            "so if a task needs one.\n\n" + "\n".join(missing)
        )


# ------------------------------------------------------------------ schemas ----


def _object_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """A manual's `inputs` as the object schema a provider is sent. A tool that
    declares no inputs takes none."""
    if schema.get("type") == "object":
        return schema
    return {"type": "object", "properties": {}, **schema}


def _sockets() -> Path | None:
    """Where hosts listen: a directory of `$PH_RUNTIME`, made owner-only — or
    `None` on a deployment with no usable runtime directory, which then runs
    every call alone. Under the runtime root because a socket's address is short
    and this path is, and because what is there does not outlive a reboot."""
    try:
        directory = resolve_roots().ensure_runtime().runtime / "utcp"
        directory.mkdir(mode=0o700, exist_ok=True)
    except (RuntimeDirError, OSError):
        return None
    return directory


def _made(paths: Sequence[str], root: str) -> tuple[list[str], list[str]]:
    """Make each writable path a directory that exists: those that do, and why the
    rest do not.

    A path that overlaps the session's workspace `root` is not made and not
    granted (`grants_apart`), and said here once, where a person reads it —
    `ctx.shell` applies the same rule to every call, against the calling agent's
    own root, and says nothing.

    A sandbox backend grants a path by binding it, and a path that is not there
    cannot be bound — so one missing directory would fail every call the manual's
    tools make, with the backend's own words for it. Made once, here, and a path
    that cannot be made is left out of the grant and said.

    Spelled as the deployment's own allowances are (`allowance_path`): `~`
    expanded and every symlink resolved, so a profile's `~/.cache/x` is that
    directory and not one named `~` under the harness's working directory, and
    the path the backend binds is the one the kernel will see.
    """
    made: list[str] = []
    apart, overlapping = grants_apart(root, paths)
    unmade = [
        f"{path} cannot be granted as writable: it overlaps the workspace {root}"
        for path in overlapping
    ]
    for spelled in apart:
        try:
            Path(spelled).mkdir(parents=True, exist_ok=True)
        except OSError as error:
            unmade.append(f"{spelled} cannot be granted as writable: {error.strerror}")
        else:
            made.append(spelled)
    return made, unmade


def _clip(answer: object) -> str:
    text = answer if isinstance(answer, str) else dumps(answer)
    return text if len(text) <= _CLIP else f"{text[:_CLIP]}…"


# ---------------------------------------------------------------------- row ----


@plugin("utcp", affects="environment", inject=[TOOLS, SHELL, FS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the seam, and register the tools of every manual that answers."""
    install()
    held: list[Any] = []

    async def client() -> Disposer:
        held.append(await UtcpClient.create(root_dir=str(ctx.require(FS).root_for(None))))

        async def close() -> None:
            await held[0].close()

        return close

    await ctx.effect(client, label="utcp-client")
    seam = UtcpSeam(ctx=ctx, config=config, client=held[0], socket_dir=_sockets())
    ctx.provide(UTCP, seam)
    for manual in config.manuals:
        await seam.add(manual)

    prompt = ctx.get(SYSTEM_PROMPT)
    if prompt is not None:

        def note(_request: AssembleContext) -> str:
            return seam.unavailable()

        prompt.section(
            PromptSection(name="utcp:unavailable", order=ORDER_TOOL_GUIDANCE + 1, text=note)
        )
    contribute(ctx, Diagnostic(id="utcp", title="UTCP tools", order=62, read=seam.report))
    ctx.on("agent/pre-step", seam.eager)
    contribute_item(
        ctx, TUI_STATUS, StatusField(id="utcp-hosts", read=seam.reading, order=70), label="utcp"
    )
