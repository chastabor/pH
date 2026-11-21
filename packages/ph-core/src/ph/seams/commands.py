"""`ctx.commands`: slash commands of the user that spend no model turn.

A command is not a tool. `/compact`, `/revert` and `/refine` are things that
the *user* asks the harness to do. A route through a model turn would be slower
and also false: the log would show the model deciding something that the user
decided. So a command dispatches directly, records `command/run` and
`command/done`, and never opens a `turn/*`.

@module ph.seams.commands
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Literal, TypeAlias

from ..agent.types import AgentHandle
from ..cordis import (
    Context,
    Disposer,
    MaybeAwaitable,
    Running,
    events,
    maybe_await,
    plugin,
    running,
    settled_or_none,
)
from ..json import JsonObject
from ..keys import COMMANDS
from ..session import Claim, IntentNotDurable, Session, intents_of
from ..session.kinds import COMMAND_RUN, command_done
from ..wire import WireModel, declarable
from ._registry import claim_key

__all__ = [
    "CommandBody",
    "CommandContext",
    "CommandDefinition",
    "CommandRecord",
    "CommandRegistry",
    "CommandSchema",
    "CommandVerb",
    "Provisioning",
    "Verbs",
    "apply",
    "install_or_status",
    "parse_command_line",
]

log = logging.getLogger("ph.seams.commands")

events.declare(
    "commands/change",
    "emit",
    owner="ph.seams.commands",
    doc="The command set changed; consumers re-read list().",
)


class CommandSchema(WireModel):
    """What a front end needs to *offer* a command: everything except its body.

    This is the declarable half of a `CommandDefinition`. It is split from the
    callable in the way that `ToolSchema` is split from `ToolDefinition`
    (P7-11). A palette and a completer need these three fields and nothing
    else. The body runs where the profile is mounted, and over a socket that is
    not where the palette is.
    """

    name: str
    summary: str
    argument_hint: str = ""


CommandBody: TypeAlias = Callable[[str, "CommandContext"], "MaybeAwaitable[str | None]"]
"""What a slash command *is*: the line that the user typed goes in, and what to
show them comes out.

A body may be sync or async, because it arrives through the entry point of a
plugin row, and no checker between that row and `dispatch` sees it. With
`maybe_await`, a row written either way works and does not fail on an `await`.
Most bodies in this tree are `async def`. A few are plain functions, such as
the usage verb of `install_or_status` and `/refine --show`. `None` is a command
that did its work and has nothing to say.

This alias is a declaration. Before it, the contract was prose,
`run(argument, ctx) -> str | None`, written under a `Callable[..., Any]`. The
alias is the only thing between the signature of a row and the dispatch that
calls it.
"""


@dataclass(frozen=True, slots=True)
class CommandDefinition:
    """One slash command."""

    name: str
    summary: str
    run: CommandBody
    """The body. It returns the line to show the user, or nothing. A command with
    verbs gets a `Verbs` table here. The table is how the registry tells a verb
    that only asks from a verb that acts."""
    argument_hint: str = ""

    def schema(self) -> CommandSchema:
        """The wire-facing half. Nothing else about the command reaches a front end."""
        return CommandSchema.model_validate(declarable(self))


CommandRecord: TypeAlias = Literal["before", "with-act", "after"]
"""When the `command/run` of a verb reaches disk. This is the one thing that a
verb says about its record. It is one value and not two flags, because a verb
that only asks has no barrier to place.

* `before`: the own barrier of `COMMAND_RUN`, ahead of the body (S16). This is
  the default, and it is what a verb says when unsure.
* `with-act`: the barrier of the durable intent that the body opens before it
  acts. `/revert <seq>` restores through `WORKSPACE_RESTORE`. The record is
  appended before the body and written by that flush
  (`IntentJournal.open_deferred`). That is one fsync where two in a row gained
  nothing more. It is only for a body whose every act is behind such an intent.
* `after`: a verb that only asks. It changes no tree, no setting, and no record
  except the own pair of the command. It has no barrier, and it is recorded
  when it answers, as one pair. So the question that someone asks on a full
  disk, "what state are things in?", is still answered.

A wrong `before` costs an fsync, or a refused listing on a full disk. A wrong
`after` is an act with no record ahead of it. A wrong `with-act` is an act whose
record was still in memory. Each of those two is the failure that S16 closed."""


@dataclass(frozen=True, slots=True)
class CommandVerb:
    """One verb of a command: the body that answers it, and when its record is
    written."""

    run: CommandBody
    """It is given what follows the verb, stripped. The table reads the verb itself."""
    record: CommandRecord = "before"


@dataclass(frozen=True, slots=True)
class Verbs:
    """A command body that dispatches on the first word of its argument. It is
    the one parse of a verb, and the one place where a verb says when its record
    is written.

    **The record is set for each verb, not for each command**, because every
    command with a question to answer also has a verb that acts:
    `/workspaces list` beside `remove`, bare `/revert` beside `/revert <seq>`.
    A tool can declare `effect_free` as a whole, and a command cannot. The
    verbs are in one table, because a command that parsed its own verbs and
    declared the reading verbs beside them kept two lists in step by hand. One
    wrong entry there runs an act with no record ahead of it.

    `""` is the bare command. Verbs match without regard to case.
    """

    table: Mapping[str, CommandVerb]
    otherwise: CommandVerb | str
    """For an argument whose first word names no verb. It is either a usage line,
    which is answered as it is, or a verb that is given the whole argument. For
    `/revert <seq>`, that verb is the one that the argument is data for."""
    refused: tuple[type[Exception], ...] = ()
    """What a verb raises to refuse. The refusal is answered with its text and is
    not raised further. Each command states its own refusal one time here, and
    does not catch it in every verb."""

    def lookup(self, argument: str) -> tuple[CommandVerb | str, str]:
        """The verb that `argument` names, and what follows it. When it names no
        verb, the result is `otherwise` and the whole argument."""
        word, _, rest = argument.partition(" ")
        verb = self.table.get(word.casefold())
        return (self.otherwise, argument) if verb is None else (verb, rest.strip())

    def record(self, argument: str) -> CommandRecord:
        """When the record of what `argument` asks for is written. A usage line acts on
        nothing, so it is answered as a question is."""
        verb, _ = self.lookup(argument)
        return "after" if isinstance(verb, str) else verb.record

    async def __call__(self, argument: str, invocation: CommandContext) -> str | None:
        verb, given = self.lookup(argument)
        if isinstance(verb, str):
            return verb
        try:
            return await maybe_await(verb.run(given, invocation))
        except self.refused as refusal:
            return str(refusal)


Provisioning: TypeAlias = Literal["install", "status"]


def install_or_status(
    name: str,
    summary: str,
    answer: Callable[[Provisioning, CommandContext], Awaitable[str]],
) -> CommandDefinition:
    """The command that a plugin offers when it provisions something. `install`
    acts. `status` only asks, and so does the bare command.

    `answer` is handed the invocation as well as the verb, because "is this
    ready" is asked by someone. A plugin whose state is per agent, such as an
    index in the scratch directory of the agent, answers for the agent and the
    session that typed the command.

    Every such plugin has this one shape, because "is this ready" should be
    asked of each in the same way, and its verbs should be sorted into asking
    and acting in the same way. The function returns the whole definition, so
    that the hint that names the verbs comes from the table that owns them.
    """
    status = CommandVerb(lambda _rest, invocation: answer("status", invocation), record="after")
    install = CommandVerb(lambda _rest, invocation: answer("install", invocation))
    return CommandDefinition(
        name=name,
        summary=summary,
        argument_hint="[install|status]",
        run=Verbs(
            {"": status, "status": status, "install": install},
            otherwise=CommandVerb(
                lambda argument, _invocation: (
                    f"/{name} takes `install` or `status`, not {argument!r}."
                ),
                record="after",
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class CommandContext:
    """What a command body is given."""

    ctx: Context
    session: Session | None
    scope: Context
    """The boundary that this dispatch was told to run in (P6-24).

    `dispatch` used to resolve the boundary, use it for the binding, and stop
    there. A body that asked a *scoped* question then had only `agent`, and
    derived a boundary again, one frame below a caller that had just stated
    it. `ph.commands.revert` was the case. It read the tool table to decide
    what a revert covers, which is a policy read. It now reads the answer that
    each dispatch recorded (`restoreCovered`), and no body in this tree reads
    this field today. The boundary is stated at the top of the dispatch, so the
    bottom has no reason to guess."""
    agent: AgentHandle | None = None
    line: str = ""
    """The command as it was typed, with `/name` included. A record of the
    command reports this as what was asked. It is here because a verb is handed
    only what follows it (`Verbs`), and a body that records the line should not
    rebuild it."""


@dataclass(frozen=True, slots=True)
class _Registered:
    """A command and who registered it (P6-29).

    The two are in the table and not beside it. The identity-checked release of
    `claim_key` then takes both away in one step, and they can never be removed
    out of step. A parallel `dict[str, Running]` would have made that failure
    silent.
    """

    definition: CommandDefinition
    by: Running


def parse_command_line(line: str) -> tuple[str, str]:
    """`/name the rest` → `("name", "the rest")`. This is the one spelling of the
    split.

    It is exported because a front end that routes a line *before* it
    dispatches the line has to tokenize in the same way as this registry. The
    TUI over a socket is such a front end: it decides which end owns a verb.
    Otherwise a leading space and a doubled space read as two different
    commands on the two ends.
    """
    name, _, argument = line.strip().lstrip("/").partition(" ")
    return name, argument.strip()


@dataclass(slots=True)
class CommandRegistry:
    """The service published as `ctx.commands`."""

    ctx: Context
    _commands: dict[str, _Registered] = field(default_factory=dict)

    def register(self, definition: CommandDefinition, *, scope: Context | None = None) -> Disposer:
        """Register one command. Emits `commands/change` on both edges.

        The announcement is the `then` of `claim_key`, and not a wrapper around
        the disposer that `claim_key` returns. A scope that unwinds runs what
        `add_disposer` was given. With a wrapper, the unload of a row takes its
        verb away and tells nobody, and that is the edge that a remote palette
        most needs to hear about.
        """
        by = self.ctx.running_for(scope)
        changed = partial(self.ctx.emit, "commands/change")
        released = claim_key(
            by.owner,
            self._commands,
            definition.name,
            _Registered(definition, by),
            label="command",
            then=changed,
        )
        changed()
        return released

    def list(self) -> list[CommandDefinition]:
        return [self._commands[name].definition for name in sorted(self._commands)]

    def get(self, name: str) -> CommandDefinition | None:
        entry = self._commands.get(name)
        return entry.definition if entry is not None else None

    async def dispatch(
        self,
        line: str,
        *,
        scope: Context | None = None,
        session: Session | None = None,
        agent: AgentHandle | None = None,
    ) -> str | None:
        """Run one `/name argument` line. Returns the text to show the user.

        Both halves are recorded even when the body fails. A command that broke
        is something that the user did, and the log is where they will look
        for it.

        **`scope=` is the boundary that the body runs for** (P6-24). It mirrors
        `ToolExecutionInput.scope` on the tools side. Before this parameter
        existed, the binding below had to derive a boundary from `agent`. That
        value is documented as the target of approval routing and not as the
        policy boundary. `scope=` exists to remove that guess. The two differ
        for a subagent whose driver holds a child ctx.

        **This fallback is not the fail-open fallback that `ctx.fs` had.**
        There, an unreadable `agent.ctx` fell back to the *mount*, which no
        agent-scoped registration reaches, so a scoped policy row silently
        applied to nobody. P6-24 answered that with a refusal. P6-32 then
        deleted the derivation outright, so `ctx.fs` requires a stated
        `Boundary` and has no fallback at all.
        `SubagentService._delegating_boundary` is the refuser that remains. Its
        handle arrives on a payload that the caller builds, so the broken case
        stays reachable there. Here the fallback is the *own layer of the
        registration*. A dispatch with no agent lands there anyway, and that
        layer is exactly as wide as the command already was. A failure to read
        an agent widens nothing, so nothing justifies the refusal of a slash
        command of the user.
        """
        name, argument = parse_command_line(line)
        entry = self._commands.get(name)
        if entry is None:
            raise KeyError(f'unknown command "/{name}"')
        definition = entry.definition
        journal = intents_of(self.ctx)
        opening: JsonObject = {"name": name, "argument": argument}
        held: Claim | None = None
        run = definition.run
        # A body that is not a verb table said nothing about its record. It gets
        # the barrier that anything unsure gets.
        record = run.record(argument) if isinstance(run, Verbs) else "before"
        if session is not None and record == "before":
            # **On disk before the body runs** (S16). This is the barrier of
            # `COMMAND_RUN`. The effects of a command used to come before any
            # durable record of it. `/revert` rewrote the tree while
            # `command/run` was still in memory, and a crash in the body left an
            # unpaired record that nothing ever closed. A command whose record
            # cannot be written does not run.
            try:
                held = await journal.open(session, COMMAND_RUN, opening)
            except IntentNotDurable:
                return f"refusing: /{name} was not run, because the log could not record it"
        elif session is not None and record == "with-act":
            # This record is also appended before the body, and it is on disk
            # before the act of the body. The barrier of the intent that the act
            # opens writes it, and no second barrier is placed here.
            held = journal.open_deferred(session, COMMAND_RUN, opening)
        outcome = "ok"
        detail: str | None = None
        try:
            # **As the row that registered it, for the agent that typed it**
            # (P6-26, P6-29). The `ctx=self.ctx` that is handed to the body is
            # the context of the *registry*. That is the same
            # `scope or self.ctx` shape that P6-12 named one table over, so a
            # command that registered anything did it globally and forever.
            #
            # The owner is the registering row, exactly as `_invoke` binds the
            # own scope of a listener and not the scope of the emitter. The
            # reason is the same: whose code this is does not depend on who
            # triggered it. So the case with no agent no longer needs special
            # treatment. P6-26 bound nothing at all in that case, to avoid
            # restating the fallback of `owner_for` and deleting its second
            # branch. The recorded pair is a better answer than either, and it
            # is available unconditionally.
            #
            # Only the *layer* still comes from the agent, and it reads the
            # declared `ctx`. The `getattr` probe that this used to be fell
            # back to the own layer of the registration, which is the global
            # layer for every command. The layer of the agent is narrower than
            # the seam. It is stated here because a per-agent command would
            # make this the line that decides who sees what the command
            # registers. `None` past this line means "no boundary was stated,
            # and there is no agent to read one from". The two consumers below
            # each state their own fallback: bind nothing, and hand the body
            # the layer of the registration.
            resolved = scope if scope is not None else (agent.ctx if agent is not None else None)
            with running(entry.by, resolved):
                result = await maybe_await(
                    run(
                        argument,
                        CommandContext(
                            ctx=self.ctx,
                            session=session,
                            scope=resolved if resolved is not None else entry.by.layer,
                            agent=agent,
                            line=f"/{name} {argument}".rstrip(),
                        ),
                    )
                )
            # **Checked, not coerced.** A body arrives through the entry point
            # of a plugin row, so `CommandBody` is a claim that no checker
            # between that row and here has seen. This line used to read
            # `result if isinstance(result, str) else None`. That silently
            # substitutes a default, which is what `settled_or_none` was
            # written to end. The check also protects the `finally` below: a
            # detail that is not JSON would raise out of the append of
            # `command/done` *there*, and mask the own exception of the body.
            detail = settled_or_none(f"/{name}", result, str)
            return detail
        except Exception as error:
            outcome = "error"
            detail = str(error)
            raise
        except BaseException as error:
            # **Cancellation is not a failure, and it is not an `Exception`.**
            # `asyncio.CancelledError` is a `BaseException`. A `/compact` that
            # the user interrupted, or one whose task group was torn down,
            # therefore unwound past the branch above, and the `finally`
            # recorded the value that `outcome` was initialized with: `ok`, with
            # no detail. The log then said that a command which never finished
            # had succeeded. That is the one question that `command/done`
            # exists to answer. The same reasoning covers `KeyboardInterrupt`
            # and `SystemExit`: the record says that the body did not complete.
            outcome = "canceled"
            detail = str(error) or type(error).__name__
            raise
        finally:
            if session is not None and held is not None:
                journal.settle(
                    session,
                    held,
                    command_done(
                        name=name, run_seq=held.opened.seq, outcome=outcome, detail=detail
                    ),
                )
            elif session is not None:
                # A question (`after`). It is recorded when it is answered, as
                # one pair and with no barrier. It changed nothing that a record
                # had to precede, and a disk that cannot take a write is exactly
                # when a user asks what state things are in.
                journal.open_settled(
                    session,
                    COMMAND_RUN,
                    opening,
                    lambda opened: command_done(
                        name=name, run_seq=opened.seq, outcome=outcome, detail=detail
                    ),
                )


@plugin("commands", affects="environment")
async def apply(ctx: Context, config: None) -> None:
    """Mount the command registry."""
    ctx.provide(COMMANDS, CommandRegistry(ctx=ctx))
