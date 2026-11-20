"""`utcp`: use tools published as UTCP manuals as tools a pH agent can call.

This package has one row. The config of the row lists manuals. Each manual is a
UTCP *manual call template*, which is the object that any UTCP harness
registers. Another row can also add a manual at mount through `ctx.utcp.add`.
The row runs discovery for each manual. Each tool that the manual names then
becomes a registered pH tool. Like any other tool, it appears in the prompt,
goes through the tool pipeline, is recorded in the log, and is listed in Code
Mode.

## A UTCP tool sees what its agent sees

`ctx.shell` starts the command of a `cli` tool for the agent that makes the call
(`_library.PhCliProtocol`). The command runs in the workspace of that agent and
under its sandbox. If the call is canceled, the command stops. A UTCP tool
therefore has the same limits as `bash`. The sandbox sets those limits.
`permissions-fs` does not, because it applies to calls through `ctx.fs`, as its
own documentation says. If a tool keeps state outside the workspace, its manual
names those paths in `writable`. Without that grant, a confined tool cannot
write them.

## A read-only session protects the workspace, not the working state of a tool

Some tools build data about a tree, such as an index or a graph. A tree is the
directory of project files that a tool works on. Such a tool changes no file in
the tree. A manual names a directory for that data (`state`). For each call, the
row tells the command where that directory is for the calling agent:
`$PH_TOOL_STATE`, a directory in the scratch directory of that agent. The
scratch directory is writable in every sandbox mode. The workspace is writable
only in `workspace-write`. The tool therefore works in a read-only session, and
it cannot write the repository there. Each session and agent has its own scratch
directory, so an agent that restarts finds what it built. `writable` is for data
that agents share, such as a grammar cache or model weights. The row refuses a
grant that overlaps the workspace (`ph.seams.sandbox.grants_apart`), so no
manual can make the tree writable again. The temporary files of a confined
command also go in the scratch directory (`TMPDIR`, set by the shell seam), for
the same reason.

## Background runs, for a tool whose work is long

Each call is one command. No process stays running between calls. A manual can
name tools whose work is long (`Manual.background`), such as an indexing run.
The row then starts the command of such a tool itself. One such execution is a
*background run*. The row starts it as it starts every command: through
`ctx.shell`, for the agent that owns the tree the run works on, in the workspace
of that agent and under its sandbox. The run is a job of that agent
(`ctx.jobs`). It stops when the agent ends, and when the row of the manual
unmounts.

The row runs the command with `--progress` (`ph.toolcli`). The command reports
on its stderr in JSON Lines: one line at a time while it works, and one last
line. This stream is the only information the row has about the run. The footer
shows the progress of the run. When the run ends, the row tells the agent that
the run works for. It uses a note that waits for the next step of that agent and
does not start a turn. If the stream ends with no last line, the run died, and
the note says so.

The row starts a run without a request before the first step of a root agent, so
an index is in progress before a call asks for it. A call of the tool also
starts a run. That call waits up to `wait_s` for the run that it started, and
answers with the result of the run. After `wait_s`, it answers with what the
command says while another run holds its work: `state: indexing`. The run
continues.

**One background run for each tree at a time, not one for each agent.** A child
agent builds its own state only when it has a tree that it writes to
permanently: a worktree or an overlay. A child that works in the checkout of its
parent, or that can only read the project, shares the state of its parent
(`UtcpSeam.owner_of`). The index describes the tree. A copy for each child of a
read-only fan-out would embed the same documents again.

## The row config says what UTCP cannot say

A manual gives a name, a description, and two schemas. It cannot say "two calls
of this tool may overlap" or "this tool changes nothing". The scheduler and the
policy rows of pH read exactly those two facts. Each defaults to the careful
answer: calls cannot overlap, and a call has effects. The `tools` table of a
manual can state otherwise for each tool.

## A missing provider is a missing capability, not a failed start

Discovery of a manual can fail: the command is not installed, the endpoint is
down, or this deployment does not allow the protocol. The row then **skips** the
manual. It writes a warning to the log, `phern doctor` shows a line, and the
system prompt gets a note. The note tells the model why a skill names a tool
that it cannot find. The mount continues with the manuals whose discovery
succeeded.

## No credential is stored

A template can contain `${VAR}` references. The UTCP client resolves them from
the environment when a call is made. This row holds no credential.

@module ph_utcp
"""

from __future__ import annotations

import logging
import os
import re
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import Field

from ph.agent.types import AgentHandle, PreStepDecision, PreStepRequest
from ph.cancel import Canceled, Cancellation
from ph.characters import head
from ph.cordis import Context, Disposer, Next, Releaser, ServiceKey, plugin
from ph.json import JsonObject, as_obj, as_str, dumps, loads
from ph.keys import AGENTS, FS, JOBS, SHELL, SUBPROCESS, SYSTEM_PROMPT, TOOLS, TUI_STATUS
from ph.llm.types import PluginSource, create_user_message
from ph.seams._registry import contribute_item
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.jobs import Job
from ph.seams.sandbox import allowance_path, grants_apart
from ph.seams.shell import ShellResult
from ph.seams.subprocess import SubprocessService
from ph.seams.tui_status import StatusField, StatusReading
from ph.seams.workspace import project_access, scratch_of, workspace_of
from ph.session import Session
from ph.system_prompt.assembly import ORDER_TOOL_GUIDANCE, AssembleContext, PromptSection
from ph.text import count_of, elide
from ph.toolcli import STATE_ENV, Progress, read_envelope, value_schema
from ph.tools.definition import (
    ToolDefinition,
    ToolRunContext,
    define_tool,
    text_content,
)
from ph.tools.errors import HarnessError, error_message
from ph.tools.json_schema import validate_json_schema_value
from ph.tools.presentation import CardKind, simple_views
from ph.wire import WireModel

from ._library import (
    UTCP_TIMEOUT,
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
    "Background",
    "Config",
    "Manual",
    "ManualState",
    "ToolTraits",
    "UtcpSeam",
    "apply",
    "command_said",
]

BUNDLE = Path(__file__).parent / "bundle.yaml"
"""The profile layer of this distribution, for the `ph.bundles` group."""

log = logging.getLogger("ph_utcp")

UTCP_FAILED = "UTCP_FAILED"
UTCP_BAD_ARGUMENTS = "UTCP_BAD_ARGUMENTS"
UTCP_BAD_ANSWER = "UTCP_BAD_ANSWER"

_NOT_A_WORD = re.compile(r"\W")
_CLIP = 600
"""The length limit for quoting back an answer that did not match what the tool
declared."""

_STRAY_LINES = 5
"""The number of stderr lines of a run, other than progress lines, that are
kept. They are the last output of a command that died."""


# ------------------------------------------------------------------- config ----


class ToolView(WireModel):
    """How a front end draws the card of one tool: a title above one important
    argument."""

    card: CardKind = "generic"
    title: str
    argument: str


class ToolTraits(WireModel):
    """The facts about one tool that a manual cannot give. Each default is the
    careful answer."""

    concurrency_safe: bool = False
    """Two calls may overlap in one batch."""
    effect_free: bool = False
    """A call changes nothing."""
    self_limits: bool = False
    """The tool limits its own output and offers a way to page through it."""
    arguments_disposable: bool = False
    """The arguments are a payload to deliver. The model does not refer back to them."""
    timeout_ms: int | None = None
    """The time limit for this tool. It overrides the limit of the row."""
    view: ToolView | None = None


class Background(WireModel):
    """The tools of a manual whose work is long. The row runs them where the
    session can monitor them, with the `call --progress` command of `ph.toolcli`.

    The row adds the tool name, `--progress`, and `--args-json` to `command`. It
    reads the progress of the run from stderr. The command gets the state
    directory in `$PH_TOOL_STATE`, as each call does."""

    command: list[str]
    """The command, up to and including `call`."""
    label: str
    """The name of a run in the footer and in the completion note: `text index`."""
    tools: list[str]
    """The tools that run this way, by their own names. An index tool is one."""
    eager: bool = True
    """Run each of the `tools` one time for a root agent, with no arguments,
    before the first step of that agent. An index is then in progress before a
    call asks for it. A child agent that works in its own tree gets a run at its
    first call instead, because a temporary worktree may never call the tool."""
    wait_s: float = 5.0
    """How long a call waits for the run that it started. After this time, the
    call answers that the run continues."""


class Manual(WireModel):
    """One UTCP manual, and how pH runs the tools that it names."""

    template: dict[str, Any]
    """The manual call template, as any UTCP harness registers it. Its `name` is
    the name of the manual."""
    names: Literal["prefixed", "bare"] = "prefixed"
    """How pH names the tools of the manual. `prefixed` gives `<manual>_<tool>`,
    which cannot collide with a tool that pH or another manual registers. `bare`
    gives the tool's own name. Use `bare` for a manual that is the only source of
    those tools."""
    envelope: bool = False
    """The tool answers with `{ok, value, text, error}`, the envelope of
    `ph.toolcli`.

    The value of the pH tool is then the `value` of the envelope, and its content
    is the `text` of the envelope. A failure is raised with the code from the
    envelope. The tool then looks the same to an agent as a tool registered
    in-process. The manual must state this setting. The row does not detect an
    envelope, because an object with an `ok` key is not always an envelope."""
    state: str = Field("", pattern=r"^([A-Za-z0-9][A-Za-z0-9._-]*)?$")
    """A directory name for the working state of this manual, or empty for none.

    For each call, the row then sets `$PH_TOOL_STATE=<the calling agent's
    scratch directory>/<state>` on the command. That directory is writable in
    every sandbox mode, is separate from the workspace, and is the same directory
    for the same agent after a restart. The value is one name, not a path. The
    agent decides where the directory goes, not the manual."""
    writable: list[str] = Field(default_factory=list)
    """Paths outside the workspace that the commands of this manual may write.
    They are for data that agents share. A grant applies in every sandbox mode.
    The row refuses a path that overlaps the workspace."""
    env: dict[str, str] = Field(default_factory=dict)
    """Variables set on the commands of this manual. They override the values in
    its templates."""
    pass_env: list[str] = Field(default_factory=list)
    """The names of variables that the row passes to the commands of this manual
    from the harness's own environment. Use this for a credential.

    The environment of a command starts as the scrubbed environment of the
    deployment. The scrub removes every variable whose name looks like a
    credential, and a template cannot get such a variable by naming it. This
    setting is how a deployment lets one pass. The name is in a profile. The row
    reads the value each time it starts a command, and stores it nowhere.

    Every command of the manual gets the values: its discovery (`<cli>
    manual`), each tool call and background run, and the `install` and `status`
    that a person runs. Discovery and a command of a person run unconfined, so
    the values also reach a command that no sandbox holds."""
    background: Background | None = None
    """The tools of this manual that run in the background, or none."""
    tools: dict[str, ToolTraits] = Field(default_factory=dict)
    """The traits of each tool, keyed by the tool's own name."""


class Config(WireModel):
    """The row config for `utcp`."""

    manuals: list[Manual] = Field(default_factory=list)
    protocols: list[str] = Field(default_factory=lambda: ["cli"])
    """The call template types that a manual may use.

    The default is `cli` only. `cli` is the only type whose commands this row
    runs through the pH shell service and its sandbox. Other types, such as
    `http` and `mcp`, run inside the harness process. The client of the UTCP
    library runs them, outside the sandbox and its egress rules. A deployment
    names such a type here to accept that."""
    timeout_ms: int = 120_000
    """How long the command of one call may run, unless the tool's own entry sets
    a limit."""
    discovery_timeout_ms: int = 60_000
    """How long the discovery command of a manual may run."""


# --------------------------------------------------------------------- seam ----


UTCP: ServiceKey[UtcpSeam] = ServiceKey("utcp")
"""The UTCP manuals that this mount holds, and the service through which another
row adds a manual."""


@dataclass(frozen=True, slots=True)
class ManualState:
    """The result of adding one manual: the tools that it gave, or why it gave
    none."""

    tools: tuple[str, ...] = ()
    """The pH names that were registered."""
    problems: tuple[str, ...] = ()
    """Why the manual, or one of its tools, is not available."""

    @property
    def available(self) -> bool:
        return bool(self.tools)

    def said(self) -> str:
        """The text for a row's line in `phern doctor`: the tools, and why any
        are missing."""
        if self.available and not self.problems:
            return f"{', '.join(self.tools)} (through UTCP)"
        reasons = "; ".join(self.problems) or "the manual named no tools"
        return f"{', '.join(self.tools) or 'unavailable'} — {reasons}"


def command_said(program: str, verb: str, result: ShellResult) -> str:
    """The output of a manual's own command, as text for a person. If the command
    failed, the text says so and gives the exit code."""
    said = result.stdout.strip() or result.stderr.strip()
    if result.exit_code != 0:
        return f"{program} {verb} failed (exit {result.exit_code}): {said}"
    return said


@dataclass(slots=True)
class _Run:
    """One background run: the owner of the tree that it works on, the request,
    its progress, and how it ended."""

    name: str
    """The manual whose tool this run executes."""
    background: Background
    owner: AgentHandle
    tool: str
    arguments: JsonObject
    job: Job | None = None
    progress: Progress | None = None
    """The latest line of its report (`ph.toolcli.Progress`)."""
    stray: deque[str] = field(default_factory=lambda: deque(maxlen=_STRAY_LINES))
    """The last stderr lines that were not progress lines."""
    result: ShellResult | None = None
    error: Exception | None = None
    """Why the command could not be started, if it could not. The run then has no
    result, and no stderr line gives the reason. The field holds the error
    itself, so that a refusal keeps its own code."""
    ended: anyio.Event = field(default_factory=anyio.Event)
    waiting: int = 0
    """The number of calls that wait to answer with the result of this run. A
    call that still waits when the run ends takes the result, and the row sends
    no note."""
    owed: bool = False
    """A call got the "still going" answer. The row owes the agent of that call a
    note when the run ends."""

    @property
    def last(self) -> Progress | None:
        """The final line of the run, if the run wrote one."""
        progress = self.progress
        return progress if progress is not None and progress.state != "working" else None


@dataclass(slots=True)
class UtcpSeam:
    """The service published as `ctx.utcp`."""

    ctx: Context
    config: Config
    client: Any
    """The UTCP client. The type is `Any` because the library ships no types."""
    states: dict[str, ManualState] = field(default_factory=dict)
    _forgets: dict[str, Releaser] = field(default_factory=dict)
    """The `forget` of each held state, registered on the scope that added the
    state. When another add replaces the state, the old `forget` is dropped
    without running. Each name then has one teardown, which belongs to its
    current holder."""
    manuals: dict[str, Manual] = field(default_factory=dict)
    """The manuals that gave tools, by name. The row starts a run from these."""
    runs: dict[tuple[str, str], _Run] = field(default_factory=dict)
    """The run in progress for each tree, keyed by manual and by the id of the
    owner of the tree."""
    warmed: dict[tuple[str, str], Releaser] = field(default_factory=dict)
    """The trees whose tools the row has already run one time without a request,
    keyed by manual and owner. Each value takes the mark back when its owner ends
    (`_warm`). The row does this one time only, because at the next step it would
    index the tree again for no caller. When a manual is removed, its marks are
    removed, and the values that would have taken them back are dropped (the
    `forget` in `add`)."""

    # ----------------------------------------------------------- registering --

    async def add(self, manual: Manual, *, scope: Context | None = None) -> ManualState:
        """Discover `manual` and register its tools on `scope`. This method never
        raises for a manual that cannot be reached. The state gives the reason,
        and so does the log.

        `scope` is the owner of the manual. When a row adds its own manual and
        then unmounts, the tools are unregistered and the manual is forgotten. If
        the name is already held and gave tools, this method changes nothing and
        returns the existing state. The tools are already registered, and a
        second discovery would only make the registry refuse each tool by name.

        If the name is held with no tools, the manual is discovered again. The
        name then belongs to this add, and the teardown of the earlier add is
        dropped from its scope without running. Before this rule, the earlier
        teardown stayed on its scope. When that scope ended, the teardown removed
        the client's record of the manual, and every later call of the tools from
        this add failed.
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
        said = (*unmade, *_unscrubbed(manual.pass_env, self.ctx.get(SUBPROCESS)))
        if said:
            state = ManualState(tools=state.tools, problems=(*state.problems, *said))

        async def forget() -> None:
            """Undo this add when the owning scope unwinds. This removes the state,
            so that the name can be added again. It removes the runs that the
            manual started and the marks that record them. It removes the client's
            own record of the manual. Only the current holder runs this. An add
            that replaces the state drops it without running it."""
            del self.states[name]
            del self._forgets[name]
            self.manuals.pop(name, None)
            for key in [key for key in self.warmed if key[0] == name]:
                self.warmed.pop(key).drop()
            for key in [key for key in self.runs if key[0] == name]:
                job = self.runs.pop(key).job
                if job is not None:
                    job.cancel()
            try:
                await self.client.deregister_manual(name)
            except Exception:
                log.debug("ph_utcp: %s was not deregistered", name, exc_info=True)

        # Register the teardown first, and do not suspend between here and the
        # return. A scope that is already down then refuses before the name is
        # held. No other add can run between the write of the state and the
        # registration of the teardown that removes it.
        holding = (scope or self.ctx).add_disposer(forget, label=f"utcp-manual({name})")
        replaced = self._forgets.pop(name, None)
        if replaced is not None:
            replaced.drop()
        self.states[name], self._forgets[name] = state, holding
        if state.available:
            self.manuals[name] = manual
        for problem in state.problems:
            log.warning("ph_utcp: %s: %s", name, problem)
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
    ) -> Call:
        """Build the `Call` for one command of `manual`, run for `agent`. `owner`
        is the owner of the agent's tree. The caller finds the owner one time and
        passes it, because each helper here needs it."""
        env, writable = self._surroundings(manual, agent, owner)
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
        """The variables that a command of this manual gets when it runs for
        `agent`, and the paths outside the workspace that it may write.
        Discovery, a tool call, a background run, and a provisioning command that
        a person runs all use this one derivation."""
        env = dict(manual.env)
        for name in manual.pass_env:
            # Read from the harness's own environment, not the scrubbed one. That
            # is the purpose of `pass_env`: the value is read now, set on this one
            # command, and stored nowhere.
            value = os.environ.get(name)
            if value:
                env[name] = value
        writable = list(manual.writable)
        state = self._state(manual, owner)
        if state is not None:
            env[STATE_ENV] = str(state)
            held = workspace_of(self.ctx, agent)
            if held is not None and not state.is_relative_to(held.scratch):
                # The state belongs to the owner of the tree, and it is not in
                # this agent's own scratch directory. Grant write access to it.
                writable.append(str(state))
        return env, tuple(writable)

    def _tree(self, manual: Manual, agent: AgentHandle | None, owner: AgentHandle | None) -> Path:
        """The directory where a command of `manual` runs for `agent`. For a
        manual that keeps state, this is the tree that the state describes, which
        is the tree of `owner`. For a manual that keeps no state, it is the
        agent's own root.

        The working directory is how a command learns its tree (the default
        `--root` of `ph.toolcli`). A child that shares the index of its parent
        must therefore work on the tree of its parent. If it ran in its own empty
        directory, it would index that directory into the shared index and remove
        every document as gone."""
        fs = self.ctx.require(FS)
        return fs.root_for(owner if manual.state else agent)

    def _command(
        self,
        manual: Manual,
        argv: Sequence[str],
        *,
        agent: AgentHandle,
        owner: AgentHandle,
        timeout_ms: int | None,
        signal: Cancellation | None,
        on_stderr_line: Callable[[str], None] | None = None,
    ) -> Awaitable[ShellResult]:
        """Run one `ph.toolcli` command of `manual` for `agent`. The row starts
        this command itself and does not use the UTCP client. The command runs in
        the tree, under the sandbox, with the variables and grants that a call of
        the manual's tools gets."""
        env, writable = self._surroundings(manual, agent, owner)
        inherited = manual.template.get("inherit_env_vars")
        return self.ctx.require(SHELL).run_argv(
            tuple(argv),
            cwd=self._tree(manual, agent, owner),
            agent=agent,
            timeout_ms=timeout_ms,
            signal=signal,
            keep_env=inherited if isinstance(inherited, list) else None,
            extra_env=env,
            writable_extra=writable,
            on_stderr_line=on_stderr_line,
        )

    def run_command(
        self, manual: Manual, argv: Sequence[str], agent: AgentHandle | None, *, timeout_ms: int
    ) -> Awaitable[ShellResult]:
        """Run one of the manual's own commands for `agent`, at the request of a
        person. Examples are `install` and `status`. The command gets what a call
        of the manual's tools gets: the variables, the state directory, and the
        tree.

        The command is not confined, like any provisioning command that a person
        types. `install` needs the network, and no model chose the command."""
        owner = None if agent is None else self.owner_of(agent)
        env, _ = self._surroundings(manual, agent, owner)
        return self.ctx.require(SHELL).run_argv(
            tuple(argv),
            cwd=self._tree(manual, agent, owner),
            timeout_ms=timeout_ms,
            extra_env=env,
        )

    def owner_of(self, agent: AgentHandle) -> AgentHandle:
        """The agent that owns the tree that `agent`'s tools work on: the agent
        itself, or the nearest ancestor whose tree it shares.

        The data that a tool builds about a tree belongs to the tree. The question
        is therefore whether a child has its own tree to build data about:

        * **The child works in the checkout of its parent** (`shared`, the same
          root). There is one tree, so there is one index and one run at a time.
        * **The child can only read the project.** This is `readonly-scratch`,
          whose root is an empty directory beside a repository that it cannot
          write, and the two ephemeral kinds, whose writes reach nobody. Such a
          child has nothing of its own to index. A fan-out of such children is
          the case that this rule exists for. The children query the index of
          their parent, and none of them embeds a document or holds a model.
        * **The child has a tree that it writes to permanently** (a `worktree` or
          an `overlay`). Its files diverge from the files of its parent, so the
          child owns that tree and builds its own data.

        `project_access` is the workspace seam's own classification of a
        workspace kind. A spawn records its result as the access that the child
        was granted."""
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
        """The agent that delegated to `agent`, if `agent` is a sub-agent and the
        delegating agent still exists. The id of an agent is the id of its
        session."""
        session = agent.session
        wanted = None if session is None else session.header.delegating_parent
        registry = self.ctx.get(AGENTS)
        return None if wanted is None or registry is None else registry.get(wanted)

    def _state(self, manual: Manual, owner: AgentHandle | None) -> Path | None:
        """The directory that `manual` named, in the scratch directory of `owner`.

        Returns `None` when the manual names no directory. Also returns `None`
        when no agent exists to have a scratch directory: during discovery, or
        for a call made for the deployment. The command then uses its own
        default, and nothing confines it. `ctx.shell` confines a command only for
        an agent that holds a workspace, and such an agent has a scratch
        directory. The command creates the state directory itself. The scratch
        directory already exists."""
        if not manual.state or owner is None:
            return None
        session = owner.session
        if session is not None:
            return scratch_of(self.ctx, session.id, owner) / manual.state
        held = workspace_of(self.ctx, owner)
        return None if held is None else held.scratch / manual.state

    # ----------------------------------------------------------------- runs --

    async def _start(
        self,
        name: str,
        manual: Manual,
        owner: AgentHandle,
        asked: Sequence[tuple[str, JsonObject]],
    ) -> _Run | None:
        """Start `asked` in the background, one after the other, for the tree of
        `owner`. `asked` holds tools of `manual` and their arguments. Returns the
        first run. Returns `None` when a run cannot be started: there is no job
        runner, or the manual names no background.

        This method never raises. A background run exists to make a long call
        tolerable. A call that could have run in the ordinary way must not fail
        because a background run is not available.
        """
        background = manual.background
        jobs = self.ctx.get(JOBS)
        if background is None or jobs is None or not asked:
            return None
        key = (name, owner.id)
        runs = [
            _Run(
                name=name,
                background=background,
                owner=owner,
                tool=tool,
                arguments=arguments,
            )
            for tool, arguments in asked
        ]
        # Put the run on the table before the job starts. A call that arrives in
        # the meantime then joins this run and does not start a second run.
        self.runs[key] = runs[0]
        try:
            job = await jobs.start(
                kind="utcp-run",
                label=f"{background.label} for {owner.id}",
                run=partial(self._running, manual, runs),
                # Use the owner's scope. The run then ends when the agent that
                # owns the tree ends. It also stops when the manual's row unmounts.
                scope=owner.ctx,
                on_settled=lambda _job: self._unreached(runs),
            )
        except Exception:
            log.warning("ph_utcp: no %s run was started", name, exc_info=True)
            # Take the run off the table and also mark it ended. A call that
            # joined it in the meantime then asks the command itself, and does
            # not wait for `wait_s` on a run that never started.
            self._take_off(runs[0])
            return None
        for run in runs:
            run.job = job
        return runs[0]

    async def _running(self, manual: Manual, runs: Sequence[_Run], job: Job) -> None:
        """The body of one job. It runs the command of each run in turn, for the
        owner of its tree, and reads the progress from stderr as the command
        writes it."""
        for run in runs:
            owner = run.owner
            key = (run.name, owner.id)
            self.runs[key] = run
            argv = [*run.background.command, run.tool, "--progress"]
            try:
                run.result = await self._command(
                    manual,
                    [*argv, "--args-json", dumps(run.arguments)],
                    agent=owner,
                    owner=owner,
                    # No time limit. A run takes as long as its work takes, and
                    # its job ends it.
                    timeout_ms=None,
                    signal=job.token,
                    on_stderr_line=partial(_read_line, run),
                )
            except Exception as error:
                # The command never ran. For example, the sandbox refused it or
                # the program is missing. No stderr line will give the reason.
                run.error = error
                log.warning("ph_utcp: the %s could not be run", run.background.label, exc_info=True)
                raise
            finally:
                self._take_off(run)
                # Call `_ended` in the same step that sets `ended`. A waiting
                # call has not resumed yet, so `waiting` still shows correctly
                # whether a call takes the result.
                self._ended(run)
            if job.token.canceled:
                return

    def _unreached(self, runs: Sequence[_Run]) -> None:
        """End every run of a settled job that the job body never reached (S6).

        The first run goes on the table before its job starts, so that a call in
        the meantime joins it. A run is marked ended only inside the job body. A
        job can be canceled before its body runs, or stopped after an earlier run
        and before this one. Without this method, such a job left a run whose
        `ended` was never set. Every later call joined that run and waited for
        `wait_s`, and no new run was started. This method ends such a run with no
        result and no error. A joined call reads that as nothing to take, and
        asks the command itself.
        """
        for run in runs:
            if not run.ended.is_set():
                self._take_off(run)

    def _take_off(self, run: _Run) -> None:
        """Remove `run` from the table if it is still the run there, and mark it
        ended. This is the only way a run leaves the table, whether it ran, failed
        to start, or was never reached."""
        key = (run.name, run.owner.id)
        if self.runs.get(key) is run:
            del self.runs[key]
        run.ended.set()

    def _ended(self, run: _Run) -> None:
        """Tell the agent of `run` how the run ended. Send nothing if a call takes
        the result of the run, or if there is nothing to tell.

        A run can end in five ways:

        * The command could not be started. The run has no result, and nothing is
          on its stderr. The note gives the reason.
        * The last line says `done`. The agent gets a note when a call got the
          "still going" answer, or when a run that nobody requested changed
          something.
        * The last line says `busy`. Another run held what this run builds, so
          this run did nothing. The agent gets a note only when a call got the
          "still going" answer.
        * The last line says `failed`. The note gives the tool's own sentence.
        * There is no last line. The command was killed, ran out of memory, or
          crashed below Python. The note says so, and gives the last output of
          the command and its progress."""
        result, job = run.result, run.job
        if run.waiting or (result is not None and result.canceled):
            return
        if job is not None and job.token.canceled:
            # The run was stopped because its owner ended, or because the
            # manual's row unmounted.
            return
        label = run.background.label
        if result is None:
            if run.error is None:
                return
            summary = f"The {label} could not be started: {error_message(run.error)}"
            text = f"{summary}. Nothing was built; calling its tool tries again."
            self._tell(run.owner, summary, text)
            return
        last, progress = run.last, run.progress
        if last is not None and last.state == "busy":
            if not run.owed:
                return
            summary = f"The {label} did not run: another run holds what it builds"
            text = f"{summary}. Call its tool again once that run has finished."
        elif last is not None and last.state == "done":
            if not run.owed and last.changed is not True:
                return
            summary = f"The {label} is finished: {last.said()}"
            text = f"{summary}. Searches now cover all of it."
            if last.note:
                # What the tool result would have told a caller that waited.
                text = f"{text}\n{last.note}"
        else:
            if last is not None:
                why = last.failure or "it gave no reason"
            else:
                said = "; ".join(run.stray) or "it wrote nothing more"
                why = f"its command ended (exit {result.exit_code}) without finishing: {said}"
            summary = f"The {label} stopped: {why}"
            got = (
                ""
                if progress is None
                else f" It got through {progress.done} of {progress.total} {progress.what}."
            )
            text = (
                f"{summary}.{got} What it finished is kept, and running its tool "
                "again goes on from there."
            )
            log.warning("ph_utcp: %s", head(summary, _CLIP))
        self._tell(run.owner, summary, text)

    def _tell(self, owner: AgentHandle, summary: str, text: str) -> None:
        """Send a note to `owner` with `inject`. The note waits for the next step
        of the agent, does not start a turn, and is recorded in the log of that
        agent."""
        registry = self.ctx.get(AGENTS)
        # Get the driver from the agent registry. A handle is what a reader of
        # the agent gets, and it has no inbox for a message.
        driver = None if registry is None else registry.get(owner.id)
        if driver is None:
            # The agent has ended, so there is nobody to tell. The tool still
            # reports the outcome.
            return
        try:
            driver.inject(
                create_user_message(
                    content=[{"type": "text", "text": text}],
                    source=PluginSource(
                        plugin="ph_utcp",
                        form="notice",
                        summary=summary,
                    ),
                )
            )
        except Exception:
            log.debug("ph_utcp: %s was not told: %s", driver.id, summary, exc_info=True)

    async def _in_background(
        self,
        name: str,
        manual: Manual,
        tool: str,
        arguments: JsonObject,
        run: ToolRunContext,
        owner: AgentHandle,
        *,
        timeout_ms: int,
    ) -> object:
        """Answer a call of a background tool. The answer is the command's own
        envelope.

        The call starts a run, or joins the run in progress for its tree, and
        waits up to `wait_s` for it. If the run was for this call's request and
        ended in time, the call answers with the result of the run. Otherwise the
        row asks the command directly with `--wait 0`. The command then gives the
        answer that it gives while another run holds its work: `state: indexing`,
        in the tool's own shape. The run continues, and the row tells its agent
        when it ends. The same request made again joins its run and does not
        start another run. A different request is not queued."""
        background = manual.background
        assert background is not None
        key = (name, owner.id)
        with anyio.move_on_after(background.wait_s):
            while True:
                current = self.runs.get(key)
                if current is None:
                    current = await self._start(name, manual, owner, [(tool, arguments)])
                    if current is None:
                        break
                mine = current.tool == tool and current.arguments == arguments
                current.waiting += mine
                try:
                    await current.ended.wait()
                finally:
                    current.waiting -= mine
                    if not current.ended.is_set():
                        # This call gets the "still going" answer below. The row
                        # owes its agent a note when the run ends.
                        current.owed = True
                if not mine:
                    # The run of another request ended within the wait. Start
                    # the run for this request.
                    continue
                if isinstance(current.error, HarnessError):
                    # A refusal that has its own error code, for example from
                    # the sandbox.
                    raise current.error
                if current.error is not None:
                    raise HarnessError(
                        f"{name}.{tool} could not be started: {error_message(current.error)}",
                        UTCP_FAILED,
                    )
                if current.result is not None and not current.result.canceled:
                    return _answer_of(current.result, f"{name}.{tool}")
                break
        assert run.agent is not None
        result = await self._command(
            manual,
            [*background.command, tool, "--wait", "0", "--args-json", dumps(arguments)],
            agent=run.agent,
            owner=owner,
            timeout_ms=timeout_ms,
            signal=run.signal,
        )
        return _answer_of(result, f"{name}.{tool}")

    def working(self, name: str, owner: AgentHandle | None) -> str | None:
        """The text to add to an answer of the tools of manual `name` while a run
        for the tree of `owner` is in progress: the progress of the run. The
        command knows only that a run holds its work. This row has read the
        count."""
        run = None if owner is None else self.runs.get((name, owner.id))
        if run is None or run.progress is None or run.progress.state != "working":
            return None
        return f"[The {run.background.label} is still working: {run.progress.said()}.]"

    def reading(self, session: Session) -> StatusReading | None:
        """The footer text for this session: what its runs are building. The text
        comes from memory, because the row reads the stderr of each run as the
        command writes it."""
        said: list[str] = []
        for run in self.runs.values():
            owned = run.owner.session
            progress = run.progress
            if owned is None or owned.id != session.id or progress is None:
                continue
            if progress.state == "working":
                said.append(f"{run.background.label} {progress.said()}")
        return StatusReading(text=", ".join(said), level="normal") if said else None

    async def eager(self, request: PreStepRequest, next_: Next[PreStepDecision]) -> PreStepDecision:
        """Handle `agent/pre-step`. Before a step of a root agent, start the
        `eager` background tools of each manual, one time. The index is then in
        progress before the model asks for it. This listener runs after the
        listener of the workspace row, which gives the agent the tree that the
        run works on. The run is started and not awaited, because a run must not
        delay the step."""
        if not request.session.header.is_subagent:
            agent = request.agent
            for name, manual in list(self.manuals.items()):
                background = manual.background
                if background is None or not background.eager:
                    continue
                if (name, agent.id) in self.warmed:
                    continue
                self._warm(name, agent)
                if (name, agent.id) not in self.runs:
                    await self._start(
                        name, manual, agent, [(tool, {}) for tool in background.tools]
                    )
        return await next_()

    def _warm(self, name: str, agent: AgentHandle) -> None:
        """Mark a tree as warmed for as long as its owner lives.

        The mark belongs to the agent and is removed with it. If marks were kept
        longer, the table would hold one mark for every root agent that the
        process ever stepped. This is one holding (Holdings rule 1): the take-back
        is registered first, and then the mark is written. If the manual is
        removed first, it drops the take-back without running it.
        """
        key = (name, agent.id)

        def take_back() -> None:
            del self.warmed[key]

        self.warmed[key] = agent.ctx.add_disposer(take_back, label="utcp-warmed")

    # ------------------------------------------------------------- the tool --

    def _definition(self, name: str, manual: Manual, tool: Tool) -> ToolDefinition:
        """Build the pH tool for one UTCP tool."""
        utcp_name: str = tool.name
        own = utcp_name.removeprefix(f"{name}.")
        ph_name = own if manual.names == "bare" else _NOT_A_WORD.sub("_", utcp_name)
        traits = manual.tools.get(own, ToolTraits())
        budget = traits.timeout_ms or self.config.timeout_ms
        inputs = _object_schema(tool.inputs.model_dump(by_alias=True, exclude_none=True))
        outputs: dict[str, Any] = tool.outputs.model_dump(by_alias=True, exclude_none=True)
        declared = value_schema(outputs) if manual.envelope else outputs
        confined = tool.tool_call_template.call_template_type == "cli"
        background = manual.background
        # A background tool answers in the `ph.toolcli` envelope. The row runs its
        # command itself and reads what `call` prints.
        in_background = background is not None and own in background.tools and manual.envelope

        async def execute(args: object, run: ToolRunContext) -> object:
            arguments = dict(as_obj(args))
            # The raw-schema path passes the model's arguments through without a
            # check. The tool is responsible for validation, and this is the tool.
            violations = validate_json_schema_value(inputs, arguments)
            if violations:
                raise HarnessError("; ".join(violations), UTCP_BAD_ARGUMENTS)
            owner = None if run.agent is None else self.owner_of(run.agent)
            if in_background and owner is not None:
                answer = await self._in_background(
                    name, manual, own, arguments, run, owner, timeout_ms=budget
                )
                return self._unwrapped(utcp_name, name, manual, answer, run, owner)
            call = self._call(
                manual, timeout_ms=budget, agent=run.agent, owner=owner, signal=run.signal
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
                return self._unwrapped(utcp_name, name, manual, answer, run, owner)
            mismatches = validate_json_schema_value(declared, answer)
            if mismatches:
                # The usual cause is a failed command. The `cli` protocol ignores
                # exit codes and returns stderr as the result, so a tool that
                # declares an object answers with a sentence. The model needs
                # that sentence, so the error quotes it and does not list the
                # schema violations.
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
        self,
        name: str,
        manual_name: str,
        manual: Manual,
        answer: object,
        run: ToolRunContext,
        owner: AgentHandle | None,
    ) -> object:
        """Return the `value` of an envelope, and present its `text` as the
        content.

        The tool's own `render` wrote the text in the tool's own process, from
        the arguments and the value. It is the projection that a session would
        make if the tool were registered here. The text is therefore what the
        registry shows for this value (`ToolRunContext.present`), and what every
        post-execute row sees and edits. A failure is raised with the tool's own
        code (`ph.toolcli.read_envelope`).

        The command ran in the tree of `owner`. That tree is not always the asking
        agent's own root, for example when a child queries the index of the tree
        that it shares. The text then says so, because the paths in the text
        start at that tree. While a run for that tree is in progress, the text
        also gives the progress of the run (`working`)."""
        try:
            value, text = read_envelope(answer)
        except ValueError as error:
            raise HarnessError(
                f"{name} did not answer with an envelope: {_clip(answer)}", UTCP_BAD_ANSWER
            ) from error
        if text is not None:
            elsewhere = self._tree(manual, run.agent, owner)
            if elsewhere != self.ctx.require(FS).root_for(run.agent):
                text += f"\n[Paths here are relative to {elsewhere}, the tree this was built from.]"
            working = self.working(manual_name, owner)
            if working is not None:
                text += f"\n{working}"
            run.present(value, text_content(text))
        return value

    # ------------------------------------------------------------ reporting --

    def report(self) -> list[tuple[str, str]]:
        """The section for `phern doctor`: each manual, and what it gave."""
        rows: list[tuple[str, str]] = []
        for name in sorted(self.states):
            state = self.states[name]
            said = count_of(len(state.tools), "tool") if state.tools else "unavailable"
            if state.problems:
                said += " — " + "; ".join(state.problems)
            rows.append((name, said))
        return rows

    def unavailable(self) -> str:
        """The note that the model reads: the manuals that gave no tools, and why."""
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
    """Return the `inputs` of a manual as the object schema that is sent to a
    model provider. A tool that declares no inputs takes none."""
    if schema.get("type") == "object":
        return schema
    return {"type": "object", "properties": {}, **schema}


def _read_line(run: _Run, line: str) -> None:
    """Handle one stderr line of a run. A progress line updates the progress of
    the run. Any other line is kept, in case it is part of the run's last
    output."""
    try:
        run.progress = Progress.model_validate_json(line)
    except ValueError:
        if line.strip():
            said = head(line.strip(), _CLIP)
            run.stray.append(said)
            log.debug("ph_utcp: %s wrote: %s", run.name, said)


def _answer_of(result: ShellResult, what: str) -> object:
    """Return the one object that a `ph.toolcli` command printed, for a command
    that this row ran itself. A command that printed anything else failed before
    it could answer."""
    if result.canceled:
        raise Canceled("the UTCP command was canceled")
    if result.timed_out:
        raise HarnessError("the UTCP command ran past its time and was stopped", UTCP_TIMEOUT)
    try:
        answer: object = loads(result.stdout)
    except ValueError as error:
        raise HarnessError(
            f"{what} did not answer (exit {result.exit_code}): "
            f"{_clip(result.stdout.strip() or result.stderr.strip() or 'it printed nothing')}",
            UTCP_BAD_ANSWER,
        ) from error
    return answer


def _unscrubbed(names: Sequence[str], subprocess: SubprocessService | None) -> list[str]:
    """Return a warning for each variable that a manual passes by name and that
    the environment scrub of the deployment does not remove.

    `pass_env` gives a credential to the commands of one manual and to no other
    command. Every other command, such as an agent's `bash`, starts from the
    scrubbed environment. But the scrub matches on the name of a variable. A
    variable whose name the scrub does not match is already in the environment of
    every command. Passing it protects nothing, and a person should know that."""
    if subprocess is None:
        return []
    marks = ", ".join(subprocess.scrub.marks)
    return [
        f"${name} is passed to its commands by name, and the deployment's environment "
        "scrub does not drop that name — so every command an agent runs has it already. "
        f"Name it so the scrub matches it ({marks}), or add a mark for it to "
        "`subprocess-local`'s `scrub`"
        for name in names
        if not subprocess.scrub.drops(name)
    ]


def _made(paths: Sequence[str], root: str) -> tuple[list[str], list[str]]:
    """Make each writable path a directory that exists. Returns the paths that now
    exist, and a reason for each path that does not.

    A path that overlaps the session's workspace `root` is not created and not
    granted (`grants_apart`). This function reports it one time, where a person
    reads it. `ctx.shell` applies the same rule to every call, against the
    calling agent's own root, and reports nothing.

    A sandbox backend grants a path by binding it, and it cannot bind a path that
    does not exist. One missing directory would therefore fail every call of the
    manual's tools, with the backend's own error text. This function creates each
    directory one time. A path that cannot be created is left out of the grant
    and reported.

    Paths are spelled as the deployment's own allowances are (`allowance_path`):
    `~` is expanded and every symlink is resolved. A profile's `~/.cache/x` is
    then that directory, and not a directory named `~` under the working
    directory of the harness. The path that the backend binds is also the path
    that the kernel sees.
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
    return elide(text, _CLIP)


# ---------------------------------------------------------------------- row ----


@plugin("utcp", affects="environment", inject=[TOOLS, SHELL, FS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the seam, and register the tools of every manual whose discovery
    succeeds."""
    install()
    held: list[Any] = []

    async def client() -> Disposer:
        held.append(await UtcpClient.create(root_dir=str(ctx.require(FS).root_for(None))))

        async def close() -> None:
            await held[0].close()

        return close

    await ctx.effect(client, label="utcp-client")
    seam = UtcpSeam(ctx=ctx, config=config, client=held[0])
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
        ctx, TUI_STATUS, StatusField(id="utcp-runs", read=seam.reading, order=70), label="utcp"
    )
