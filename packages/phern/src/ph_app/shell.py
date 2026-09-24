"""`!<command>` and `!!<command>` — a person's own shell, run where the session
lives (P7-10).

The counterpart to `attach.py`, and here for the same reason that one exists:
**two front ends run this and there must be one author.** In process the TUI
holds the session; over a socket a daemon root does. The rule they share — refuse
if nothing can run it, append what is about to happen, run it, append what did —
is the whole feature, and a copy per transport is a copy that drifts. The first
draft had one, and it had already diverged inside a single increment.

**Two events, log first** (§5 rule 2). `shell/command` is appended *before* the
child starts, so a command that hangs — or that takes the daemon down with it —
still says in the log what was started, which is exactly the command worth
knowing about. One event on completion would lose it.

**Both forms log in full; only `!` is spoken.** Neither event type is
surface-eligible — see `SURFACE_EVENT_TYPES` in `ph.session.events` — so no
`shell/*` event can reach the model by having been logged. That is what makes
`!!` private by construction rather than by a filter anybody has to remember,
and it is not expressible as one. `!` therefore cannot widen that whitelist to
say its piece: it appends a *separate* user message, at its own budget
(`SURFACE_OUTPUT`), which makes the one door visible on the log as a door.

**The event carries facts, not a rendering.** `stdout`, `stderr` and `exitCode`
stay apart on the log the way `BashValue` keeps them apart for the model, because
joining them is a presentation choice — a front end that wants stderr in red can
make it, and one that wants a single column can join. A view rides *beside* an
event and never inside it; `daemon/cards.py` says the same thing from the other
side.

@module ph_app.shell
"""

from __future__ import annotations

from ph.agent.types import AgentDriver
from ph.cordis import Context
from ph.json import JsonObject, as_int, as_str
from ph.keys import SESSIONS, SHELL
from ph.llm.types import PluginSource, TextBlock, create_user_message
from ph.seams.shell import ShellResult, ShellService
from ph.session import Session
from ph.text import NO_OUTPUT, truncation_marker
from ph.tools.builtin.bash_tool import TIMED_OUT

from .protocol import SeamAbsent

__all__ = ["SHELL_OUTPUT", "SURFACE_OUTPUT", "run_shell", "shell_body", "shell_message", "shell_of"]

SHELL_OUTPUT = 64 * 1024
"""How much of one stream the log keeps.

Bounded because the log is durable and `!!find /` is one keystroke; generous
because the reason a person runs a command is to read what it said. Applied per
stream, and truncation is recorded on the event rather than left to be inferred
from a suspiciously round length.

**A display clip on top of a bound that already held.** `ctx.subprocess` caps
what it *keeps* from a child (P7-13), so by the time this runs the string is
bounded; this decides how much of it a durable log should carry, which is a much
smaller number and a different question. Both can apply to one command, and the
event records them apart: `dropped` is what the seam threw away, `clipped` is
what this kept back.
"""


SURFACE_OUTPUT = 4 * 1024
"""How much of that a `!` puts in front of the model.

Much smaller than `SHELL_OUTPUT`, because the two bounds protect different
payers. The log's is a file written once; this one guards a **prompt prefix
re-sent on every step until compaction**, so a single `!find /` at the log's cap
would be ~32k tokens paid again each turn — for output the person ran to read
themselves. The full text is on the log either way: `!` chooses what the model
reads, never what is kept, and a person who wants the rest can scroll.
"""


def shell_body(data: JsonObject) -> str:
    """A `shell/result`'s streams as one column, the way a terminal shows them.

    **Rendered here, from the event, and not stored on it.** The log keeps the
    two streams apart so a front end can color them apart; this is the default
    a front end that wants one column uses, and it is shared so the terminal and
    the browser cannot disagree. `[stderr]`, `[exit N]`, the timeout line and the
    truncation sentence all follow `tool-bash`'s renderer, so `!!make` and a
    model's `bash("make")` read alike in one transcript — the last two by
    calling the same functions, after P7-13 added them to one side only.
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
    return "\n".join(parts)


def shell_message(command: str, data: JsonObject) -> str:
    """What a `!` says to the model: the command, then what it printed.

    Composed here, beside `shell_body` and from the same event, so the splice
    cannot describe the command differently than the card does. The command line
    is the half that is not recoverable from the streams — output alone, spliced
    into a conversation, is text from nowhere.

    Clipped against `SURFACE_OUTPUT` and *said so*, in `truncation_marker`'s one
    sentence: a model reading a prefix has to be told it is a prefix or it will
    reason from a file it thinks it has seen the end of.
    """
    body = shell_body(data) or NO_OUTPUT
    if len(body) > SURFACE_OUTPUT:
        body = body[:SURFACE_OUTPUT] + truncation_marker(len(body) - SURFACE_OUTPUT, SURFACE_OUTPUT)
    return f"$ {command}\n{body}"


def shell_of(ctx: Context) -> ShellService:
    """The shell seam, or the refusal for its absence.

    Resolved by the *caller* rather than inside `run_shell`, so a transport that
    orders validation before an effect can refuse in its validating half: under
    the daemon the idempotence key is claimed between the two, and a refusal that
    happened after the claim would burn a retry the client still needs.
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
    """Append, run, append. Returns what ran, for a caller that must reply.

    **`surface` is the difference between `!` and `!!`.** Both run the person's
    command in the session's workspace and log it in full; only `!` puts the
    result into the conversation. It is spliced as a **user message**, which
    `ph.session.events` prescribes — a `tool/result` with no `tool_use` block to
    pair with is an orphan several providers reject — and delivered by `inject`,
    so it waits for the agent's next step rather than starting a turn: a person
    running a command is telling the model something, not asking it to act.

    An `AgentDriver` rather than the handle, because the splice is composed from
    the event *this function just appended* and so cannot disagree with it. The
    caller holds the driver too, and could do it there — at the price of
    re-deriving the rendering from `ShellResult`, which is the same second
    derivation the `cwd` paragraph below refuses for the same reason.

    `cwd` comes back *from the seam* rather than being derived here: `run`
    resolves the working directory from the agent and honors a workspace
    redirection, so a second derivation could disagree with the fact it claims
    to record. It is written on the result event, once the child has actually
    run somewhere.
    """
    # `surface` on the *command* event, not the result: it is what the person
    # asked for, it is known before the child starts, and a front end draws the
    # card from this event. Without it `!make` and `!!make` render identically
    # and nobody can see which one is about to put output in front of the model.
    #
    # Written on every command, `False` included, rather than only when true.
    # `InboxSplice` argues the opposite for `removedCount` — but that argument is
    # about not changing the shape of logs already written, and it does not reach
    # here: one shape for every `shell/command` means no reader has to know that
    # an absent key encodes the quiet half.
    started = session.append("shell/command", {"command": command, "surface": surface})
    # **On disk before the child starts** (F9) — the whole reason this is two
    # events: a command that hangs, or takes the daemon down with it, still shows
    # in the log what was started. Appended alone it showed nothing of the kind,
    # because nothing flushed until the next model request. Fail-closed like the
    # checkpoint policy's barriers: a command whose record could not be written
    # does not run.
    await agent.ctx.require(SESSIONS).flush(session)
    result = await shell.run(command, agent=agent)
    settled = session.append(
        "shell/result",
        {
            # The command this settles, so a fold can pair them and a front end
            # need not assume only one is ever in flight — two attached UIs can
            # each be running one, and the log is what tells them apart.
            "commandSeq": started.seq,
            "exitCode": result.exit_code,
            "ok": result.exit_code == 0,
            "cwd": result.cwd,
            "confinedBy": result.confined_by,
            "stdout": result.stdout[:SHELL_OUTPUT],
            "stderr": result.stderr[:SHELL_OUTPUT],
            # **Two bounds, two keys.** A fold that wants to know why a person is
            # looking at a prefix wants to tell "the seam threw 37 MB away" from
            # "we kept 64 KiB of what survived" — one `truncated` bool lost that.
            # Compared on the originals, which is O(1) and *before* anything is
            # concatenated: the first draft joined both streams in full to keep
            # 64 KiB, which on a 50 MB output was three full-size copies and
            # ~100 ms of memcpy inside the daemon's event loop.
            "dropped": result.dropped,
            "cap": result.cap,
            "timedOut": result.timed_out,
            "clipped": len(result.stdout) > SHELL_OUTPUT or len(result.stderr) > SHELL_OUTPUT,
        },
    )
    if surface:
        agent.inject(
            create_user_message(
                content=[TextBlock(text=shell_message(command, settled.data))],
                # `relay`, never an untagged user message: the person typed
                # `!make`, they did not type 4 KiB of compiler output.
                # `tool-attach` states the rule, and `ph_app.tui.trajectory` and
                # `transcript_mode` both route on `source` — untagged, a
                # machine's stdout is replayed as the person's own words.
                source=PluginSource(plugin="ph-app.shell", form="relay"),
            )
        )
    return result
