"""`!!<command>` — a person's own shell, run where the session lives (P7-10).

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

**Neither type is surface-eligible**, which is what keeps `!!` out of the model's
context: see `SURFACE_EVENT_TYPES` in `ph.session.events`. It is not a filter
anybody has to remember, and it is not expressible as one.

**The event carries facts, not a rendering.** `stdout`, `stderr` and `exitCode`
stay apart on the log the way `BashValue` keeps them apart for the model, because
joining them is a presentation choice — a front end that wants stderr in red can
make it, and one that wants a single column can join. A view rides *beside* an
event and never inside it; `daemon/cards.py` says the same thing from the other
side.

@module ph_app.shell
"""

from __future__ import annotations

from typing import Any

from ph.seams.shell import ShellResult

from .protocol import SeamAbsent

__all__ = ["SHELL_OUTPUT", "run_shell", "shell_body", "shell_of"]

SHELL_OUTPUT = 64 * 1024
"""How much of one stream the log keeps.

Bounded because the log is durable and `!!find /` is one keystroke; generous
because the reason a person runs a command is to read what it said. Applied per
stream, and truncation is recorded on the event rather than left to be inferred
from a suspiciously round length.

**Not the only bound that should exist.** `ctx.subprocess` drains a child into an
unbounded buffer and decodes it whole, so this clip is applied to something
already in memory twice over — the cap belongs in the drain, where `tool-bash`
would get one too. Recorded as P7-13 rather than half-fixed here.
"""


def shell_body(data: Any) -> str:
    """A `shell/result`'s streams as one column, the way a terminal shows them.

    **Rendered here, from the event, and not stored on it.** The log keeps the
    two streams apart so a front end can colour them apart; this is the default
    a front end that wants one column uses, and it is shared so the terminal and
    the browser cannot disagree. `[stderr]` and `[exit N]` follow `tool-bash`'s
    renderer, so `!!make` and a model's `bash("make")` read alike in one
    transcript.
    """
    parts: list[str] = []
    stdout = str(data.get("stdout", "")).rstrip()
    stderr = str(data.get("stderr", "")).rstrip()
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    if data.get("truncated"):
        parts.append("[output truncated]")
    code = data.get("exitCode")
    if code:
        parts.append(f"[exit {code}]")
    return "\n".join(parts)


def shell_of(ctx: Any) -> Any:
    """The shell seam, or the refusal for its absence.

    Resolved by the *caller* rather than inside `run_shell`, so a transport that
    orders validation before an effect can refuse in its validating half: under
    the daemon the idempotence key is claimed between the two, and a refusal that
    happened after the claim would burn a retry the client still needs.
    """
    shell = ctx.get("shell")
    if shell is None:
        raise SeamAbsent("this deployment mounts no shell")
    return shell


async def run_shell(shell: Any, session: Any, agent: Any, command: str) -> ShellResult:
    """Append, run, append. Returns what ran, for a caller that must reply.

    `cwd` comes back *from the seam* rather than being derived here: `run`
    resolves the working directory from the agent and honours a workspace
    redirection, so a second derivation could disagree with the fact it claims
    to record. It is written on the result event, once the child has actually
    run somewhere.
    """
    started = session.append("shell/command", {"command": command})
    result: ShellResult = await shell.run(command, agent=agent)
    session.append(
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
            # Compared on the originals, which is O(1), and *before* anything is
            # concatenated: the first draft joined both streams in full to keep
            # 64 KiB, which on a 50 MB output was three full-size copies and
            # ~100 ms of memcpy inside the daemon's event loop.
            "truncated": len(result.stdout) > SHELL_OUTPUT or len(result.stderr) > SHELL_OUTPUT,
        },
    )
    return result
