"""`ctx.commands` — human slash commands that spend no model turn.

A command is not a tool. `/compact`, `/revert`, `/refine` are things the *human*
asks the harness to do, and routing them through a model turn would be both
slower and dishonest: the log would show the model deciding something the user
decided. So a command dispatches directly, records `command/run` and
`command/done`, and never opens a `turn/*`.

@module ph.seams.commands
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..cordis import Context, Disposer, Running, maybe_await, plugin, running
from ..session import Session
from ._registry import claim_key

__all__ = ["CommandDefinition", "CommandRegistry", "apply"]

log = logging.getLogger("ph.seams.commands")


@dataclass(frozen=True, slots=True)
class CommandDefinition:
    """One slash command."""

    name: str
    summary: str
    run: Callable[..., Any]
    """`run(argument: str, ctx: CommandContext) -> str | None` — the line to show."""
    argument_hint: str = ""


@dataclass(frozen=True, slots=True)
class CommandContext:
    """What a command body is given."""

    ctx: Context
    session: Session | None
    agent: Any = None


@dataclass(frozen=True, slots=True)
class _Registered:
    """A command and who registered it (P6-29).

    In the table rather than beside it, so `claim_key`'s identity-checked release
    takes both away in one step and the two can never be popped out of step —
    the failure a parallel `dict[str, Running]` would have made silent.
    """

    definition: CommandDefinition
    by: Running


@dataclass(slots=True)
class CommandRegistry:
    """The service published as `ctx.commands`."""

    ctx: Context
    _commands: dict[str, _Registered] = field(default_factory=dict)

    def register(self, definition: CommandDefinition, *, scope: Context | None = None) -> Disposer:
        by = self.ctx.running_for(scope)
        return claim_key(
            by.owner,
            self._commands,
            definition.name,
            _Registered(definition, by),
            label="command",
        )

    def list(self) -> list[CommandDefinition]:
        return [self._commands[name].definition for name in sorted(self._commands)]

    def get(self, name: str) -> CommandDefinition | None:
        entry = self._commands.get(name)
        return entry.definition if entry is not None else None

    async def dispatch(
        self, line: str, *, session: Session | None = None, agent: Any = None
    ) -> str | None:
        """Run one `/name argument` line. Returns the text to show the human.

        Both halves are recorded even when the body fails: a command that broke
        is something the user did, and the log is where they will look for it.
        """
        name, _, argument = line.lstrip("/").partition(" ")
        entry = self._commands.get(name)
        if entry is None:
            raise KeyError(f'unknown command "/{name}"')
        definition = entry.definition
        if session is not None:
            session.append("command/run", {"name": name, "argument": argument.strip()})
        outcome = "ok"
        detail: str | None = None
        try:
            # **As the row that registered it, for the agent that typed it**
            # (P6-26, P6-29). The `ctx=self.ctx` handed to the body is the
            # *registry's* — the same `scope or self.ctx` shape P6-12 named one
            # table over — so a command that registered anything did it globally
            # and forever.
            #
            # The owner is the registering row, exactly as `_invoke` binds a
            # listener's own scope rather than the emitter's, and for the same
            # reason: whose code this is does not depend on who triggered it. So
            # there is no longer a no-agent case to special-case — P6-26 bound
            # nothing at all there, to avoid restating `owner_for`'s fallback and
            # deleting its second branch. The recorded pair is a better answer
            # than either, and it is available unconditionally.
            #
            # Only the *layer* still comes from the agent, and only that half
            # uses P6-24's `getattr` idiom. It fails open to the registration's
            # own layer, which for every command in the tree is the global one —
            # narrower than the seam, and stated here because a per-agent command
            # would make this the line that decides who sees what it registers.
            scope = getattr(agent, "ctx", None)
            with running(entry.by, scope if isinstance(scope, Context) else None):
                result = await maybe_await(
                    definition.run(
                        argument.strip(),
                        CommandContext(ctx=self.ctx, session=session, agent=agent),
                    )
                )
            detail = result if isinstance(result, str) else None
            return detail
        except Exception as error:
            outcome = "error"
            detail = str(error)
            raise
        finally:
            if session is not None:
                data: dict[str, Any] = {"name": name, "outcome": outcome}
                if detail is not None:
                    data["detail"] = detail
                session.append("command/done", data)


@plugin("commands")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the command registry."""
    ctx.provide("commands", CommandRegistry(ctx=ctx))
