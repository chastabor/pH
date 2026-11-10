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
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import anyio
from pydantic import Field

from ph.agent.types import AgentHandle
from ph.cancel import Canceled, Cancellation
from ph.cordis import Context, Disposer, ServiceKey, plugin
from ph.json import as_obj, as_str, dumps
from ph.keys import FS, SHELL, SYSTEM_PROMPT, TOOLS
from ph.llm.types import ContentBlock
from ph.seams.diagnostics import Diagnostic, contribute
from ph.seams.sandbox import allowance_path
from ph.system_prompt.assembly import ORDER_TOOL_GUIDANCE, AssembleContext, PromptSection
from ph.text import count_of
from ph.toolcli import read_envelope, value_schema
from ph.tools.definition import (
    ToolDefinition,
    ToolExecution,
    ToolExecutionResult,
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
    "Manual",
    "ManualState",
    "ToolTraits",
    "UtcpSeam",
    "apply",
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
    writable: list[str] = Field(default_factory=list)
    """Paths outside the workspace this manual's commands may write."""
    env: dict[str, str] = Field(default_factory=dict)
    """Variables set on this manual's commands, over what its templates say."""
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


@dataclass(slots=True)
class UtcpSeam:
    """The service published as `ctx.utcp`."""

    ctx: Context
    config: Config
    client: Any
    """The UTCP client. `Any` because the library ships no types."""
    states: dict[str, ManualState] = field(default_factory=dict)
    _texts: dict[str, str] = field(default_factory=dict)
    """Per call id, the rendering an envelope carried — held between `execute`,
    which reads it, and `finalize_content`, which shows it."""

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
        made, unmade = await anyio.to_thread.run_sync(_made, manual.writable)
        manual = manual.model_copy(update={"writable": made})
        state = await self._discovered(name, manual, scope)
        if unmade:
            state = ManualState(tools=state.tools, problems=(*state.problems, *unmade))
        self.states[name] = state
        for problem in state.problems:
            log.warning("ph_utcp: %s: %s", name, problem)

        async def forget() -> None:
            """What the owning scope's unwinding takes back: the state, so the name
            can be added again, and the client's own record of the manual."""
            if self.states.get(name) is state:
                del self.states[name]
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
        signal: Cancellation | None = None,
    ) -> Call:
        return Call(
            shell=self.ctx.require(SHELL),
            root=self.ctx.require(FS).root_for(agent),
            timeout_ms=timeout_ms,
            env=manual.env,
            writable=tuple(manual.writable),
            agent=agent,
            signal=signal,
        )

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
            call = self._call(manual, timeout_ms=budget, agent=run.agent, signal=run.signal)
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
                return self._unwrapped(utcp_name, answer, run)
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
            finalize_content=self._shown if manual.envelope else None,
            timeout_ms=budget,
            self_limits=traits.self_limits,
            arguments_disposable=traits.arguments_disposable,
            effect_free=traits.effect_free,
            is_concurrency_safe=traits.concurrency_safe,
            present_call=views["present_call"] if views is not None else None,
            present_result=views["present_result"] if views is not None else None,
        )

    def _unwrapped(self, name: str, answer: object, run: ToolRunContext) -> object:
        """An envelope's `value`, with its `text` kept for the content. A failure
        is raised with the tool's own code (`ph.toolcli.read_envelope`)."""
        try:
            value, text = read_envelope(answer)
        except ValueError as error:
            raise HarnessError(
                f"{name} did not answer with an envelope: {_clip(answer)}", UTCP_BAD_ANSWER
            ) from error
        if text is not None:
            self._texts[run.call_id] = text
        return value

    def _shown(
        self, execution: ToolExecution, result: ToolExecutionResult
    ) -> Sequence[ContentBlock] | None:
        """The tool's own rendering, in place of pH's default one for the value.

        The text was written by the tool's `render` in its own process, from the
        arguments and the value — the projection a session would have made had the
        tool been registered here. Total, as a finalizer must be: no text held, or
        a failed call, leaves the content as it is.
        """
        text = self._texts.pop(execution.call_id, None)
        if text is None or result.is_error:
            return None
        return text_content(text)

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


def _made(paths: Sequence[str]) -> tuple[list[str], list[str]]:
    """Make each writable path a directory that exists: those that do, and why the
    rest do not.

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
    unmade: list[str] = []
    for path in paths:
        spelled = allowance_path(path)
        try:
            Path(spelled).mkdir(parents=True, exist_ok=True)
        except OSError as error:
            unmade.append(f"{path} cannot be granted as writable: {error.strerror}")
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
