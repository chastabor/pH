"""`command-compact`: `/compact`, over the backend-independent seam (P4-03).

This is a row of its own, exactly as dsh splits `command-compact` from
`compaction-basic`. The human verb and the summarization policy are different
decisions. A deployment that wants automatic compaction without a slash command,
or a slash command with a different engine behind it, should not have to fork
either row. This row injects `commands` and `compaction` and knows nothing about
summaries.

**A command is not a turn.** `/compact` is something that the *user* asked the
harness to do. It therefore dispatches directly, records `command/run` and
`command/done`, and never opens a `turn/*`. The one thing that it does open is a
surface replacement, which is why it requires the agent to be idle first. A
compaction that landed mid-turn would move the surface under a request that the
loop had already derived.

**Every failure is a sentence, not a traceback.** `CompactionError.code` is a
closed set so that a front end can phrase each code. The mapping is here because
this is the only place that has a user to tell.

@module ph_stabilize.compact_command
"""

from __future__ import annotations

from ph.cordis import Context, plugin
from ph.keys import COMMANDS, COMPACTION
from ph.seams.commands import CommandContext, CommandDefinition
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
"""One line for each `CompactionError.code`. This is the `expectedFailure` of
dsh, narrowed to the codes that the seam of pH actually defines. A phrase for a
code that nothing raises is a promise that nobody can check."""


@plugin("command-compact", affects="environment", inject=[COMMANDS, COMPACTION])
async def apply(ctx: Context, config: None) -> None:
    """Register `/compact`."""

    async def compact(argument: str, invocation: CommandContext) -> str:
        agent = invocation.agent
        if agent is None:
            return "refusing: /compact needs an agent whose session to compact"
        try:
            # Anything typed after the verb is the user's statement of what they
            # are about to work on. dsh refuses arguments here, and the compact
            # tool of deepagents takes none. But the moment when someone
            # compacts on purpose is usually the moment when they change
            # subject, and they know something about what comes next that the
            # summarizer cannot read from the conversation. The engine decides
            # what to do with the text. This row only passes it on and lets the
            # log record it.
            result = await ctx.require(COMPACTION).compact_now(agent, instructions=argument.strip())
        except CompactionError as error:
            # `.get` and not a lookup: a future code without a phrase should
            # still say something true, and the message of the exception itself
            # is the most accurate text available.
            return FAILURE_TEXT.get(error.code, str(error))
        if result is None:
            return "no compactable history yet"
        return (
            f"compacted {count_of(len(result.shadowed_seqs), 'message')} "
            f"(~{result.shadowed_tokens} tokens)"
        )

    ctx.require(COMMANDS).register(
        CommandDefinition(
            name="compact",
            summary="Replace older conversation history with a summary.",
            argument_hint="[what you are about to work on]",
            run=compact,
        )
    )
