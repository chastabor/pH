"""`/sandbox` — see what a confined command may reach, and change it without a restart.

The human half of `sandbox-allow`. A refused boundary lands in the transcript as a
`sandbox/denied` notice that names this command and the line that lifts it; this
is where that line goes. It shows the posture in force — backend, network mode,
hosts, extra directories, the refusals this session — and edits the allowances.

**An edit is two things, in this order: applied, then kept.** `Mount.reconfigure`
re-applies the `sandbox-allow` row with the new config, which releases one slot on
the seam and fills it again — no provider swapped, no probe rerun, no proxy
restarted, and the agent whose command is running notices nothing until its next
command is bounded by the new statement. Then the row is written to a **drop-in**
under the profile's directory, `$PH_HOME/profiles/<name>.d/sandbox.yaml`, which the
next start composes after the hand-written overlay. A drop-in rather than an edit
of `<name>.yaml`, because that file is the user's and a YAML rewrite drops every
comment in it; this one is pH's, says so at the top, and holds one row.

**Refuses `/`**, and only that. Allowing the root directory is `danger-full-access`
spelled to look like an allowlist. Everything narrower is the user's call — the
row's docstring says what belongs there and what it costs.

@module ph.commands.sandbox
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

import anyio
import yaml

from ..cordis import Context, plugin
from ..keys import COMMANDS, MOUNT, SANDBOX, TUI_STATUS
from ..paths import resolve_roots, write_text_under
from ..seams._registry import contribute_item
from ..seams.commands import CommandDefinition
from ..seams.invariants import contribute_fold_cache
from ..seams.sandbox import DENIED, Allowances, NetworkAllowance, NetworkMode
from ..seams.sandbox_allow import describe
from ..seams.tui_status import StatusField, StatusReading
from ..session import Session, SessionFoldCache
from ..text import count_of

__all__ = ["apply"]

log = logging.getLogger("ph.commands.sandbox")

ROW_NAME = "sandbox-allow"
"""The row this command edits — found by *name*, because a profile addresses rows
by id and the id is the profile's to choose."""

DROPIN = "sandbox.yaml"
"""The file under `<profile>.d/` this command owns."""

USAGE = (
    "usage: /sandbox [allow host <host[:port]> | allow path <dir> | "
    "revoke host <host> | revoke path <dir> | network off|allowlist|full]"
)
HINT = USAGE.removeprefix("usage: /sandbox ")

HEADER = (
    "# Written by /sandbox. pH rewrites this file whenever the allowlists change from\n"
    "# the TUI; hand edits belong in the profile's own .yaml, which this layers over.\n"
    "#\n"
    "# A row's config is replaced whole rather than merged, so the `hosts` list below\n"
    "# is now the entire allowlist for this profile: hosts added to pH's own defaults\n"
    "# in a later release will not appear until this file is deleted or edited.\n"
)

_HOST = re.compile(r"^(\*\.)?[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*(:\d{1,5})?$")
"""What `allow host` accepts: a host, optionally `*.`-prefixed, optionally `:port`.
No scheme and no path — those are what people paste, and the proxy matches hosts."""


def _denials_in(session: Session) -> tuple[Any, ...]:
    """Every `sandbox/denied` record in a session, oldest first."""
    return session.select(DENIED)


def denial_count(session: Session) -> int:
    """How many boundaries this log records being refused — a fold.

    Named, rather than the lambda it was, for the reason every sibling seam names
    its fold: a fold nobody can import is one nobody can hold to the laws
    `SessionFoldCache` requires of it and cannot itself check.
    """
    return len(_denials_in(session))


def extend_denial_count(previous: int, session: Session, from_seq: int) -> int:
    """`denial_count`, resumed from an already-counted prefix.

    A refusal is rare and `session.seq` moves on every chunk, so the cache misses
    on nearly every read while a model streams; counting the new slice is what
    makes the miss cost nothing.
    """
    return previous + sum(1 for event in session.events_from(from_seq) if event.type == DENIED)


class _Refused(Exception):
    """A refusal raised where it has to escape a caller — the `/workspaces` idiom."""


@dataclass(frozen=True, slots=True)
class _Sandbox:
    """One dispatch's view of the sandbox posture."""

    ctx: Context
    session: Session | None

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
        # `describe` is the row's own rendering, which `ph doctor` prints — and
        # which its docstring already claimed this command shared. It did not: this
        # re-derived the same three rows and had drifted on the empty-host wording.
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
        """`allow`/`revoke`, which differ by a verb and a set operation and nothing else.

        **One path through validation, for both directions**, which is the point of
        writing it once: `allow host` normalised through `_valid_host` (which
        lowercases) while `revoke host` compared the raw argument, so
        `/sandbox allow host GitHub.com` stored `github.com` and
        `/sandbox revoke host GitHub.com` answered that it was never on the list.
        Whether an entry is normalised is a property of the *kind* of entry, and now
        it is spelled once per kind.
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
        # `get_args`, so a fourth `NetworkMode` cannot be accepted by the config
        # model and silently refused here — the idiom `resolve_mode` already uses.
        if mode not in get_args(NetworkMode):
            return USAGE
        chosen: NetworkMode = mode  # type: ignore[assignment]
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
        """Apply an edit live, then keep it — and say which of the two happened."""
        row = self._row()
        current = self.ctx.require(SANDBOX).allowances or Allowances()
        updated = mutate(current)
        if updated == current:
            return unchanged
        config = updated.to_wire()
        await self.ctx.require(MOUNT).reconfigure(row.id, config)
        kept = await self._persist(row.id, config)
        return f"{said}; {self.ctx.require(SANDBOX).network_posture()}. {kept}"

    def _row(self) -> Any:
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

    async def _persist(self, row_id: str, config: dict[str, Any]) -> str:
        """Write the drop-in, or say why the change lives only in this process."""
        name = self.ctx.require(MOUNT).profile.name
        if not name:
            return (
                "Not saved: this deployment runs a profile file rather than a named profile; "
                "add the row to that file to keep the change."
            )
        path = resolve_roots().profile_dropins(name) / DROPIN
        text = HEADER + yaml.safe_dump([{"id": row_id, "config": config}], sort_keys=False)
        await anyio.to_thread.run_sync(write_text_under, path, text)
        return f"Saved to {path}; it applies now and on the next start."


@dataclass(frozen=True, slots=True)
class _Kind:
    """One thing `/sandbox allow` and `/sandbox revoke` can name.

    A table rather than four branches, so validation, reading and writing are each
    stated once per kind and both directions are obliged to share them.
    """

    validate: Callable[[str], str]
    read: Callable[[Allowances], list[str]]
    write: Callable[[Allowances, list[str]], Allowances]
    said: str
    """What the answer calls it: "<entry> is now <said>"."""


def _with(entries: list[str], entry: str, adding: bool) -> list[str]:
    """`entries` with `entry` added or removed. Unchanged when there is nothing to
    do, so `_change` sees no change and says so rather than rewriting the file."""
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
    """The footer's count of refusals, folded at most once per appended event."""

    cache: SessionFoldCache[int] = field(
        default_factory=lambda: SessionFoldCache(denial_count, extend=extend_denial_count)
    )

    def stale_folds(self, sessions: Iterable[Session]) -> list[str]:
        """Cached refusal counts that no longer equal their fold (I6).

        The lightest of the six folds and still worth polling: this number is the
        footer's account of how much the sandbox refused, and a count that drifted
        down is the reassuring direction to drift.
        """
        return self.cache.stale(sessions)

    def reading(self, session: Session) -> StatusReading | None:
        denied = self.cache.read(session)
        if denied == 0:
            return None
        return StatusReading(text=f"sandbox: {count_of(denied, 'refusal')}", level="warning")


@plugin("sandbox-commands", inject=[COMMANDS, SANDBOX, MOUNT])
async def apply(ctx: Context, _config: Any) -> None:
    """Register `/sandbox`, and the footer reading that says refusals happened."""

    async def sandbox(argument: str, invocation: Any) -> str:
        verb, _, rest = argument.strip().partition(" ")
        view = _Sandbox(ctx=ctx, session=invocation.session)
        try:
            if verb in ("", "show", "list"):
                return view.show()
            if verb in ("allow", "revoke"):
                return await view.edit(rest, adding=verb == "allow")
            if verb == "network":
                return await view.network(rest.strip())
        except _Refused as refusal:
            return str(refusal)
        return USAGE

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
    contribute_fold_cache(
        ctx, id="sandbox-fold-cache", subject="sandbox refusal count", stale=denials.stale_folds
    )
