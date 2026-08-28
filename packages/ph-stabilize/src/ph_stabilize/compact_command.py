"""`command-compact` — `/compact`, over the backend-independent seam (P4-03).

A row of its own, exactly as dsh splits `command-compact` from
`compaction-basic`: the human verb and the summarization policy are different
decisions, and a deployment that wants automatic compaction without a slash
command — or a slash command with a different engine behind it — should not have
to fork either. This row injects `commands` and `compaction` and knows nothing
about summaries.

**A command is not a turn.** `/compact` is something the *person* asked the
harness to do, so it dispatches directly, records `command/run` and
`command/done`, and never opens a `turn/*`. The one thing it does open is a
surface replacement, which is why it insists the agent is idle first — a
compaction that landed mid-turn would move the surface underneath a request the
loop had already derived.

**Every failure is a sentence, not a traceback.** `CompactionError.code` is a
closed set precisely so a front end can phrase each one; the mapping lives here
because this is the only place that has a person to say it to.

@module ph_stabilize.compact_command
"""

from __future__ import annotations

from typing import Any

from ph.cordis import Context, plugin
from ph.seams.commands import CommandDefinition
from ph.seams.compaction import CompactionError
from ph.text import count_of

__all__ = ["FAILURE_TEXT", "apply"]

FAILURE_TEXT: dict[str, str] = {
    "busy": (
        "Compaction needs an idle session: this one is working, or a compaction "
        "is already running. The conversation is unchanged."
    ),
    "summary": (
        "Compaction could not produce a usable summary. The conversation is "
        "unchanged; the attempt is recorded in the session log."
    ),
    "unavailable": "This profile has no compaction engine, so there is nothing to compact with.",
}
"""One line per `CompactionError.code`. dsh's `expectedFailure`, narrowed to the
codes pH's seam actually defines — a phrase for a code nothing raises is a
promise nobody can check."""


@plugin("command-compact", inject=["commands", "compaction"])
async def apply(ctx: Context, config: Any) -> None:
    """Register `/compact`."""

    async def compact(argument: str, invocation: Any) -> str:
        agent = invocation.agent
        if agent is None:
            return "refusing: /compact needs an agent whose session to compact"
        try:
            # Anything typed after the verb is the person saying what they are
            # about to work on. dsh refuses arguments here and deepagents'
            # compact tool takes none — but the moment someone compacts on
            # purpose is usually the moment they are changing subject, and they
            # know something about what comes next that the summarizer cannot
            # read off the conversation. The engine decides what to do with it;
            # this row only passes it on and lets the log record it.
            result = await ctx.compaction.compact_now(agent, instructions=argument.strip())
        except CompactionError as error:
            # `.get` rather than a lookup: a future code without a phrase should
            # still say something true, and the exception's own message is the
            # truest thing available.
            return FAILURE_TEXT.get(error.code, str(error))
        if result is None:
            return "no compactable history yet"
        return (
            f"compacted {count_of(len(result.shadowed_seqs), 'message')} "
            f"(~{result.shadowed_tokens} tokens)"
        )

    ctx.commands.register(
        CommandDefinition(
            name="compact",
            summary="Replace older conversation history with a summary.",
            argument_hint="[what you are about to work on]",
            run=compact,
        )
    )
