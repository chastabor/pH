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

from ..cordis import Context, Disposer, maybe_await, plugin
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


@dataclass(slots=True)
class CommandRegistry:
    """The service published as `ctx.commands`."""

    ctx: Context
    _commands: dict[str, CommandDefinition] = field(default_factory=dict)

    def register(self, definition: CommandDefinition, *, scope: Context | None = None) -> Disposer:
        return claim_key(
            self.ctx.owner_for(scope), self._commands, definition.name, definition, label="command"
        )

    def list(self) -> list[CommandDefinition]:
        return [self._commands[name] for name in sorted(self._commands)]

    def get(self, name: str) -> CommandDefinition | None:
        return self._commands.get(name)

    async def dispatch(
        self, line: str, *, session: Session | None = None, agent: Any = None
    ) -> str | None:
        """Run one `/name argument` line. Returns the text to show the human.

        Both halves are recorded even when the body fails: a command that broke
        is something the user did, and the log is where they will look for it.
        """
        name, _, argument = line.lstrip("/").partition(" ")
        definition = self._commands.get(name)
        if definition is None:
            raise KeyError(f'unknown command "/{name}"')
        if session is not None:
            session.append("command/run", {"name": name, "argument": argument.strip()})
        outcome = "ok"
        detail: str | None = None
        try:
            result = await maybe_await(
                definition.run(
                    argument.strip(), CommandContext(ctx=self.ctx, session=session, agent=agent)
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
