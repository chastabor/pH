"""`/sandbox`: see what a confined command may reach, and change it without a restart.

This command is the human side of `sandbox-allow`. A refused boundary appears in
the transcript as a `sandbox/denied` notice. The notice names this command and
the line that lifts the refusal, and this command is where that line goes. The
command shows the posture in force: the backend, the network mode, the hosts,
the extra directories, and the refusals of this session. It also edits the
allowances.

**An edit is an override of the environment of this session** (session profiles,
S4). It goes through one function, `ph.session_profile.override`. A
`profile/override` record is appended to the log of the session and the log is
written. Only then is the `sandbox-allow` row applied again with the new config.
That releases one slot on the seam and fills it again. No provider is swapped,
no probe runs again and no proxy restarts. The agent whose command is running
notices nothing until the new statement bounds its next command. A change whose
record cannot be written is not made. The next start of this session applies the
change again from the log. An edit used to be a drop-in beside the profile,
which applied to every session on that profile at once.

**The command refuses `/`.** To allow the root directory is
`danger-full-access` written to look like an allowlist. Every narrower
directory that exists is the decision of the user. The docstring of the row
says what belongs there and what it costs.

@module ph.commands.sandbox
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..cordis import Context, Row, plugin
from ..keys import COMMANDS, MOUNT, SANDBOX, TUI_STATUS
from ..seams._registry import contribute_item
from ..seams.commands import CommandContext, CommandDefinition, CommandVerb, Verbs
from ..seams.invariants import contribute_fold_cache
from ..seams.sandbox import DENIED, NETWORK_MODES, Allowances, NetworkAllowance
from ..seams.sandbox_allow import describe
from ..seams.tui_status import StatusField, StatusReading
from ..session import Session, SessionEvent, SessionFoldCache
from ..session_profile import OverrideNotRecorded, override
from ..text import count_of

__all__ = ["apply"]


ROW_NAME = "sandbox-allow"
"""The row that this command edits. The command finds it by *name*, because a
profile addresses rows by id and the profile chooses the id."""

USAGE = (
    "usage: /sandbox [allow host <host[:port]> | allow path <dir> | "
    "revoke host <host> | revoke path <dir> | network off|allowlist|full]"
)
HINT = USAGE.removeprefix("usage: /sandbox ")

_HOST = re.compile(r"^(\*\.)?[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*(:\d{1,5})?$")
"""What `allow host` accepts: a host, with an optional `*.` prefix and an optional
`:port`. It accepts no scheme and no path. People paste those, but the proxy
matches hosts."""


def _denials_in(session: Session) -> tuple[SessionEvent, ...]:
    """Every `sandbox/denied` record in a session, oldest first."""
    return session.select(DENIED)


def denial_count(session: Session) -> int:
    """The number of refused boundaries that this log records. This is a fold.

    It was a lambda, and it now has a name for the reason that every sibling
    seam names its fold. Nobody can import an unnamed fold, so nobody can test
    it against the laws that `SessionFoldCache` requires of it and cannot check
    itself.
    """
    return len(_denials_in(session))


def extend_denial_count(previous: int, session: Session, from_seq: int) -> int:
    """`denial_count`, continued from a prefix that is already counted.

    A refusal is rare and `session.seq` moves on every chunk, so the cache
    misses on nearly every read while a model streams. Because only the new
    slice is counted, the miss costs nothing.
    """
    return previous + sum(1 for event in session.events_from(from_seq) if event.type == DENIED)


class _Refused(Exception):
    """A refusal that is raised where it must escape a caller. This is the
    `/workspaces` idiom."""


@dataclass(frozen=True, slots=True)
class _Sandbox:
    """The view that one dispatch has of the sandbox posture."""

    ctx: Context
    session: Session | None
    line: str
    """The command as it was typed. An override record stores it as what asked
    for the change."""

    # -------------------------------------------------------------- reading --

    def show(self) -> str:
        seam = self.ctx.require(SANDBOX)
        lines: list[str] = []
        if seam.provider is None:
            lines.append(
                "confinement: none — no sandbox backend is mounted, so commands run unconfined"
            )
        else:
            lines.append(
                f"confinement: {type(seam.provider).__name__} (enforces {seam.enforcement})"
            )
        # `describe` is the rendering of the row itself, which `phern doctor`
        # prints. Its docstring already claimed that this command shared it. The
        # command did not: it derived the same three rows again, and its wording
        # for an empty host list had drifted.
        lines += [f"{label}: {value}" for label, value in describe(seam)]
        if self.session is not None:
            denied = _denials_in(self.session)
            if denied:
                lines.append(f"denied this session: {len(denied)}")
                lines += [f"  · {event.data.get('message')}" for event in denied[-3:]]
        lines.append(USAGE)
        return "\n".join(lines)

    # -------------------------------------------------------------- editing --

    async def edit(self, rest: str, *, adding: bool) -> str:
        """`allow` and `revoke`, which differ by a verb and a set operation and by
        nothing else.

        **One path through validation serves both directions.** That is the
        reason to write it one time. Before, `allow host` normalized through
        `_valid_host`, which lowercases, and `revoke host` compared the raw
        argument. So `/sandbox allow host GitHub.com` stored `github.com`, and
        `/sandbox revoke host GitHub.com` answered that it was never on the
        list. Whether an entry is normalized is a property of the *kind* of
        entry, and it is now written one time for each kind.
        """
        kind, _, value = rest.strip().partition(" ")
        entry = _KINDS.get(kind)
        if entry is None or not value.strip():
            return USAGE
        wanted = entry.validate(value.strip())
        verb = "now" if adding else "no longer"
        return await self._change(
            lambda current: entry.write(current, _with(entry.read(current), wanted, adding)),
            f"{wanted} is {verb} {entry.said}",
            f"{wanted} was {'already' if adding else 'not'} on the allowlist",
        )

    async def network(self, mode: str) -> str:
        # The mode is read through `NETWORK_MODES`, so the config model cannot
        # accept a fourth `NetworkMode` that this command silently refuses.
        # `resolve_mode` already uses this idiom. The narrowing of the lookup
        # removed the `type: ignore[assignment]` that this had, and
        # `literal_lookup` says why.
        chosen = NETWORK_MODES.get(mode)
        if chosen is None:
            return USAGE
        return await self._change(
            lambda current: current.model_copy(
                update={"network": current.network.model_copy(update={"mode": chosen})}
            ),
            f"network is now {mode}",
            f"network was already {mode}",
        )

    async def _change(
        self, mutate: Callable[[Allowances], Allowances], said: str, unchanged: str
    ) -> str:
        """Make an edit an override of this session: recorded first, then applied."""
        row = self._row()
        current = self.ctx.require(SANDBOX).allowances or Allowances()
        updated = mutate(current)
        if updated == current:
            return unchanged
        if self.session is None:
            return "Not changed: there is no session here to record the change in."
        try:
            await override(
                self.ctx,
                self.session,
                row.id,
                updated.to_wire(),
                source="command",
                command=self.line,
            )
        except OverrideNotRecorded as error:
            return str(error)
        posture = self.ctx.require(SANDBOX).network_posture()
        return f"{said}; {posture}. Kept in this session's log, so it holds whenever it starts."

    def _row(self) -> Row:
        rows = [
            row
            for row in self.ctx.require(MOUNT).profile.rows
            if row.name == ROW_NAME and not row.disabled
        ]
        if not rows:
            raise _Refused(
                f"this profile mounts no {ROW_NAME} row, so there is nothing to change; "
                f"add `- id: {ROW_NAME}` / `  name: {ROW_NAME}` to the profile"
            )
        return rows[0]


@dataclass(frozen=True, slots=True)
class _Kind:
    """One thing that `/sandbox allow` and `/sandbox revoke` can name.

    This is a table and not four branches. Validation, reading and writing are
    each stated one time for each kind, and both directions must share them.
    """

    validate: Callable[[str], str]
    read: Callable[[Allowances], list[str]]
    write: Callable[[Allowances, list[str]], Allowances]
    said: str
    """What the answer calls it: "<entry> is now <said>"."""


def _with(entries: list[str], entry: str, adding: bool) -> list[str]:
    """`entries` with `entry` added or removed. The list is returned unchanged
    when there is nothing to do. `_change` then sees no change and says so, and
    does not record an override."""
    if adding:
        return entries if entry in entries else [*entries, entry]
    return [one for one in entries if one != entry]


def _with_hosts(current: Allowances, hosts: list[str]) -> Allowances:
    network: NetworkAllowance = current.network.model_copy(update={"hosts": hosts})
    return current.model_copy(update={"network": network})


def _valid_host(value: str) -> str:
    if not _HOST.match(value):
        raise _Refused(
            f"{value!r} is not a host: write `example.com`, `*.example.com` or "
            "`example.com:443`, with no scheme or path"
        )
    return value.lower()


def _valid_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise _Refused(f"{value!r} must be an absolute path or start with `~`")
    if path == Path("/"):
        raise _Refused(
            "refusing to allow `/`: that is danger-full-access spelled as an allowlist; "
            "use /mode for that posture"
        )
    if not path.is_dir():
        raise _Refused(
            f"{value} is not a directory; bwrap refuses to start when a bound path is "
            "missing, so it is not allowed until it exists"
        )
    return value


_KINDS: dict[str, _Kind] = {
    "host": _Kind(
        validate=lambda value: _valid_host(value),
        read=lambda current: current.network.hosts,
        write=_with_hosts,
        said="reachable",
    ),
    "path": _Kind(
        validate=lambda value: _valid_path(value),
        read=lambda current: current.paths,
        write=lambda current, paths: current.model_copy(update={"paths": paths}),
        said="writable beyond the workspace",
    ),
}


@dataclass(slots=True)
class _Denials:
    """The count of refusals for the footer, folded at most one time for each
    appended event."""

    cache: SessionFoldCache[int] = field(
        default_factory=lambda: SessionFoldCache(denial_count, extend=extend_denial_count)
    )

    def reading(self, session: Session) -> StatusReading | None:
        denied = self.cache.read(session)
        if denied == 0:
            return None
        return StatusReading(text=f"sandbox: {count_of(denied, 'refusal')}", level="warning")


@plugin("sandbox-commands", affects="environment", inject=[COMMANDS, SANDBOX, MOUNT])
async def apply(ctx: Context, config: None) -> None:
    """Register `/sandbox`, and the footer reading that says refusals happened."""

    def view(invocation: CommandContext) -> _Sandbox:
        return _Sandbox(ctx=ctx, session=invocation.session, line=invocation.line)

    showing = CommandVerb(lambda _rest, invocation: view(invocation).show(), record="after")
    sandbox = Verbs(
        {
            "": showing,
            "show": showing,
            "list": showing,
            "allow": CommandVerb(lambda rest, invocation: view(invocation).edit(rest, adding=True)),
            "revoke": CommandVerb(
                lambda rest, invocation: view(invocation).edit(rest, adding=False)
            ),
            "network": CommandVerb(lambda rest, invocation: view(invocation).network(rest)),
        },
        otherwise=USAGE,
        refused=(_Refused,),
    )

    ctx.require(COMMANDS).register(
        CommandDefinition(
            name="sandbox",
            summary="Show what confined commands may reach, and change it without a restart.",
            argument_hint=HINT,
            run=sandbox,
        ),
        scope=ctx,
    )
    denials = _Denials()
    contribute_item(
        ctx,
        TUI_STATUS,
        StatusField(id="sandbox", read=denials.reading, order=12),
        label="sandbox(status)",
    )
    # This is the lightest of the folds that declare a cache, and it is still
    # worth polling. The number is the account in the footer of how much the
    # sandbox refused, and a count that drifts down drifts in the reassuring
    # direction. The declared count also goes with its session, and a footer
    # that is drawn one more time for a session that has gone files no entry.
    contribute_fold_cache(
        ctx, id="sandbox-fold-cache", subject="sandbox refusal count", stale=denials.cache.stale
    )
