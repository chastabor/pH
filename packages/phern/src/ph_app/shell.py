"""`!<command>` and `!!<command>`: the user's own shell, run where the session
lives (P7-10).

This module is the counterpart to `attach.py`, and it exists for the same
reason: **the rule must have one author.** It was written when two front ends
ran the rule. In process the TUI held the session, and over a socket a daemon
root did. The in-process front end is deleted, and a daemon root is now the one
caller of `run_shell`. The rule is: refuse if nothing can run the command,
append what is about to happen, run it, append what happened. That rule is the
whole feature, and a copy for each transport is a copy that drifts. The first
draft had one, and it had already diverged inside a single increment.

**Two events, and the log comes first** (§5 rule 2). `shell/command` is appended
*before* the child starts. A command that hangs, or that takes the daemon down
with it, therefore still shows in the log what was started, and that is exactly
the command worth knowing about. One event on completion would lose it. The
pair is an intent (`ph.session.kinds.SHELL_COMMAND`, P10-08). `ctx.intents` puts
the command on disk before the child starts. Repair settles a command that the
daemon died during, so no reader is told forever that a command is still
running.

**Both forms are logged in full, and only `!` reaches the model.** Neither event
type is surface-eligible (see `SURFACE_EVENT_TYPES` in `ph.session.events`), so
no `shell/*` event can reach the model because it was logged. That makes `!!`
private by construction, and not by a filter that somebody has to remember. A
filter could not express it. `!` therefore cannot widen that whitelist to
deliver its output. It appends a *separate* user message, at its own budget
(`SURFACE_OUTPUT`), so the one path to the model is visible on the log as a
path.

**The event carries facts, not a rendering.** `stdout`, `stderr` and `exitCode`
stay apart on the log, as `BashValue` keeps them apart for the model. To join
them is a presentation choice: a front end that wants stderr in red can show it
so, and one that wants a single column can join them. A view travels *beside* an
event and never inside it. `daemon/cards.py` says the same thing from the other
side.

@module ph_app.shell
"""

from __future__ import annotations

from collections.abc import Mapping

from ph.agent.types import AgentDriver
from ph.characters import head_bytes, utf8_size
from ph.cordis import Context
from ph.json import JsonObject, as_int, as_str
from ph.keys import SHELL
from ph.llm.types import PluginSource, TextBlock, create_user_message
from ph.seams.shell import SHELL_OUTPUT, ShellResult, ShellService
from ph.session import Session, Unsettled, unsettled_why
from ph.text import NO_OUTPUT, truncation_marker
from ph.tools.builtin.bash_tool import TIMED_OUT

from .protocol import SeamAbsent

__all__ = [
    "INTERRUPTED",
    "SHELL_OUTPUT",
    "SURFACE_OUTPUT",
    "run_shell",
    "shell_body",
    "shell_message",
    "shell_of",
]

SURFACE_OUTPUT = 4 * 1024
"""How much of the output of a command a `!` puts in front of the model.

It is much smaller than `SHELL_OUTPUT`, because the two bounds protect different
costs. The bound of the log guards a file that is written one time. This bound
guards a **prompt prefix that is sent again on every step until compaction**. A
single `!find /` at the cap of the log would be ~32k tokens paid again each
turn, for output that the user ran in order to read it themselves. The full
text is on the log in both cases: `!` chooses what the model reads, never what
is kept, and a user who wants the rest can scroll.
"""


INTERRUPTED: Mapping[Unsettled, str] = {
    "outcome-unknown": (
        "[ph: the harness stopped before this command's result was recorded; "
        "what it did is unknown]"
    ),
    "not-started": "[ph: not run — the log could not record the command first]",
}
"""What a card says for a result that nobody saw the command write, keyed by the
`unsettled` reason (`ph.session.unsettled_why`). The card says it because a card
that settled with no output and no exit code reads as a command that printed
nothing and succeeded."""


def shell_body(data: JsonObject) -> str:
    """The streams of a `shell/result` as one column, as a terminal shows them.

    **It is rendered here, from the event, and not stored on the event.** The
    log keeps the two streams apart so that a front end can color them apart.
    This is the default for a front end that wants one column, and it is shared
    so that the terminal and the browser cannot disagree. `[stderr]`,
    `[exit N]`, the timeout line and the truncation sentence all follow the
    renderer of `tool-bash`, so `!!make` and a `bash("make")` of the model read
    alike in one transcript. The last two follow it by calling the same
    functions, after P7-13 added them to one side only.
    """
    parts: list[str] = []
    stdout = as_str(data.get("stdout")).rstrip()
    stderr = as_str(data.get("stderr")).rstrip()
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    if data.get("timedOut"):
        parts.append(TIMED_OUT)
    if data.get("dropped"):
        parts.append(truncation_marker(as_int(data["dropped"]), as_int(data.get("cap"))).strip())
    if data.get("clipped"):
        parts.append(f"[ph: the log keeps {SHELL_OUTPUT} bytes of each stream]")
    code = data.get("exitCode")
    if code:
        parts.append(f"[exit {code}]")
    why = unsettled_why(data)
    if why is not None:
        parts.append(INTERRUPTED[why])
    return "\n".join(parts)


def shell_message(command: str, data: JsonObject) -> str:
    """What a `!` tells the model: the command, then what it printed.

    It is composed here, beside `shell_body` and from the same event, so the
    spliced message cannot describe the command differently from the card. The
    command line is the half that cannot be recovered from the streams. Output
    alone, spliced into a conversation, is text with no origin.

    The body is clipped against `SURFACE_OUTPUT`, and the one sentence of
    `truncation_marker` *says so*. A model that reads a prefix must be told that
    it is a prefix, or it will reason from a file whose end it thinks it has
    seen.
    """
    body = shell_body(data) or NO_OUTPUT
    size = utf8_size(body)
    if size > SURFACE_OUTPUT:
        kept = head_bytes(body, SURFACE_OUTPUT)
        body = kept + truncation_marker(size - utf8_size(kept), SURFACE_OUTPUT)
    return f"$ {command}\n{body}"


def shell_of(ctx: Context) -> ShellService:
    """The shell seam, or the refusal for its absence.

    The *caller* resolves it, and `run_shell` does not. A transport that orders
    validation before an effect can then refuse in its validating half. Under
    the daemon the idempotence key is claimed between the two halves, and a
    refusal after the claim would use up a retry that the client still needs.
    """
    shell = ctx.get(SHELL)
    if shell is None:
        raise SeamAbsent("this deployment mounts no shell")
    return shell


async def run_shell(
    shell: ShellService,
    session: Session,
    agent: AgentDriver,
    command: str,
    *,
    surface: bool = False,
) -> ShellResult:
    """Run a command of the user as a recorded pair (`ShellService.run_recorded`).
    Returns what ran, for a caller that must reply.

    **`surface` is the difference between `!` and `!!`.** Both run the command
    of the user in the workspace of the session and log it in full. Only `!`
    puts the result into the conversation. The result is spliced as a **user
    message**, which `ph.session.events` prescribes: a `tool/result` with no
    `tool_use` block to pair with is an orphan that several providers reject.
    `inject` delivers it, so it waits for the next step of the agent and does
    not start a turn. A user who runs a command is telling the model something,
    and is not asking it to act.

    The parameter is an `AgentDriver` and not the handle, because the splice is
    composed from the settle that `run_recorded` returns, and so cannot disagree
    with it. The caller holds the driver too, and could do the splice there. The
    price would be to derive the rendering again from `ShellResult`. That is the
    same second derivation that the `cwd` paragraph below refuses, for the same
    reason.

    `cwd` comes back *from the seam* and is not derived here. `run` resolves the
    working directory from the agent and honors a workspace redirection, so a
    second derivation could disagree with the fact that it claims to record. It
    is written on the result event, after the child has actually run somewhere.
    """
    result, settled = await shell.run_recorded(
        command, agent=agent, session=session, surface=surface
    )
    if surface:
        agent.inject(
            create_user_message(
                content=[TextBlock(text=shell_message(command, settled.data))],
                # `relay`, never an untagged user message. The user typed
                # `!make`, and did not type 4 KiB of compiler output.
                # `tool-attach` states the rule. `ph_app.tui.trajectory` and
                # `transcript_mode` both route on `source`. If the message were
                # untagged, the stdout of a machine would be replayed as the
                # user's own words.
                source=PluginSource(plugin="ph-app.shell", form="relay"),
            )
        )
    return result
