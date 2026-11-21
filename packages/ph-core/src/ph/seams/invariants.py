"""`ctx.invariants`: which invariants this deployment enforces, and whether they hold.

Rows enforce the invariants of pH, and a row is optional. So "I3 holds" was
never a property of pH. It was a property of a profile that happened to mount
`agent-loop-invariant`, and nothing said which profiles those were. A person who
read `DESIGN.md` learned what pH promises, ran a profile that promised less, and
had no way to find out. **This registry closes that gap.** An invariant that is
enforced says so in the report. An invariant that nobody mounted is absent from
the report, and is not silently assumed.

**There are two kinds, and the difference matters.** Some invariants are
enforced *inline*: I3 checks every request as it is built, and refuses. An
inline invariant cannot be polled, because no state exists between requests to
poll. The answer to "does it hold" is "every request so far was checked". Other
invariants are *pollable*: a projection either equals its fold now or it does
not. A pollable invariant carries a `check`. To report the two kinds in the same
way would be the overstatement that E1 forbids, in both directions. An inline
invariant reported as "holds" claims a check that did not run. A pollable
invariant reported as "enforced" claims a guarantee about a file that nobody
read.

**A check that raises is a violation, and its row is not dropped.** Here the
seam differs from `ctx.diagnostics`, which drops a failing section so that the
report survives. An invariant is the thing that says that something is wrong. A
check that cannot complete is at best "unknown" and at worst the failure itself.
If the seam swallowed it, the quietest report would be the one where the most is
broken.

**There is no severity, for the reason that `diagnostics` gives** (§12 Q10). A
violated invariant is a sentence that a person reads, not a color that they
learn to skip.

**Not enforced (§5 rule 6): a poll has content only in a live process.**
`ctx.diagnostics.report()` has two callers today: `phern doctor`, and the
`status` verb of a package command line (`ph.toolcli`). Each mounts its rows
fresh for the one command: no session is created, no view is cached and no scope
is disposed. So "holds" there says that the checks run, and not that the live
state of the deployment passed them. The daemon is where the same checks have
something to look at. It checks each root when the root settles after a write
(`ph_app.daemon.supervisor.Supervisor.verifier`).

@module ph.seams.invariants
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..cordis import Context, Disposer, Running, plugin, running
from ..keys import INVARIANTS, SESSIONS
from ._names import require_slug
from ._registry import claim_key, contribute_item
from .diagnostics import ORDER_SELF_ASSESSMENT, Diagnostic
from .diagnostics import contribute as contribute_diagnostic

if TYPE_CHECKING:
    from ..session import Session


__all__ = [
    "ID_MAX",
    "ORDER_FOLD_CACHE",
    "Invariant",
    "InvariantRegistry",
    "Violation",
    "apply",
    "contribute",
    "contribute_fold_cache",
]

log = logging.getLogger("ph.seams.invariants")

ID_MAX = 32
"""How long an invariant id may be. `diagnostics`, `tui_status` and `tui_screens`
use the same bound."""


@dataclass(frozen=True, slots=True)
class Invariant:
    """One thing a row promises is true, and the way to find out."""

    id: str
    statement: str
    """What must hold, in one sentence, phrased so that a violation contradicts it.

    The report prints it next to the answer. "A11 holds" tells a person nothing
    that they can act on. "every projection equals its fold: holds" tells them
    what was checked."""
    check: Callable[[], Sequence[str]] | None = None
    """The violations found now. Empty when the invariant holds.

    `None` for an invariant that is enforced *inline*: its rule runs on the path
    that it governs and refuses there. In that case there is nothing to poll. A
    `check` that returned `[]` would be a false reassurance. It would report
    "holds" on a deployment where the enforcing listener had never been reached
    at all."""
    order: int = 0


@dataclass(frozen=True, slots=True)
class Violation:
    """One way an invariant is not holding, with the invariant that names it."""

    invariant: str
    detail: str


@dataclass(frozen=True, slots=True)
class _Registered:
    """An invariant and who registered it (P6-29). See `ph.seams.diagnostics`."""

    invariant: Invariant
    by: Running


@dataclass(slots=True)
class InvariantRegistry:
    """The service published as `ctx.invariants`."""

    ctx: Context
    _entries: dict[str, _Registered] = field(default_factory=dict)

    def register(self, invariant: Invariant, *, scope: Context | None = None) -> Disposer:
        """Declare an invariant that this deployment enforces.

        `scope=` means what it means everywhere else (P6-12, P6-25). A
        registration from the `apply` of a row already unwinds with the row, so
        pass `scope=` only to register on the lifetime of someone else. **The
        purpose is that an invariant unwinds with its enforcer.** A row that is
        unloaded stops promising, and the report stops claiming.
        """
        require_slug(invariant.id, maximum=ID_MAX, kind="invariant id")
        by = self.ctx.running_for(scope)
        return claim_key(
            by.owner, self._entries, invariant.id, _Registered(invariant, by), label="invariant"
        )

    def _checked(self) -> list[tuple[Invariant, list[str] | None]]:
        """Every invariant in `order` then id order, with what its check found.

        The result is `None` for an inline invariant, which has nothing to poll,
        and an empty list for a pollable invariant that holds. Both readers
        project from this one pass, so the order that a report prints and the
        order in which the checks run cannot drift apart.

        A check that raises becomes a finding that carries its exception. It is
        not dropped. This seam exists to report that something is wrong, and a
        failure to determine that is not evidence of health.
        """
        checked: list[tuple[Invariant, list[str] | None]] = []
        for entry in self._ordered():
            invariant = entry.invariant
            if invariant.check is None:
                checked.append((invariant, None))
                continue
            try:
                # The check runs as the row that registered it (P6-29), so a
                # check that reads a scoped registry sees what its own row would
                # see.
                with running(entry.by):
                    found = list(invariant.check())
            except Exception as error:
                log.warning(
                    "ph.seams.invariants: %r could not be checked", invariant.id, exc_info=True
                )
                found = [f"the check itself failed: {error!r}"]
            checked.append((invariant, found))
        return checked

    def enforced(self) -> list[Invariant]:
        """Every invariant registered, in `order` then id order. Runs nothing."""
        return [entry.invariant for entry in self._ordered()]

    def _ordered(self) -> list[_Registered]:
        return sorted(
            self._entries.values(), key=lambda one: (one.invariant.order, one.invariant.id)
        )

    def verify(self) -> list[Violation]:
        """Run every pollable check and collect what is not holding.

        An empty result means that every *pollable* invariant holds. It never
        means that every invariant holds, which no poll can establish.
        `enforced()` says which invariants were in the scope of that claim.
        """
        return [
            Violation(invariant.id, detail)
            for invariant, found in self._checked()
            if found
            for detail in found
        ]

    def describe(self) -> list[tuple[str, str]]:
        """The rows for `phern doctor`: one for each invariant, with the answer first.

        The checks run here and are not cached from an earlier time, because a
        person who types `phern doctor` asks about now. `diagnostics` permits
        the cost. That is the distinction that it draws against `tui_status`,
        whose fields are read often and must be cheap.
        """
        rows: list[tuple[str, str]] = []
        for invariant, found in self._checked():
            if found:
                answer = "VIOLATED · " + "; ".join(found)
            elif found is None:
                answer = f"enforced inline · {invariant.statement}"
            else:
                answer = f"holds · {invariant.statement}"
            rows.append((invariant.id, answer))
        return rows


def contribute(ctx: Context, invariant: Invariant) -> None:
    """Declare an invariant from the `apply` of a row, whether or not this seam is mounted.

    `contribute_via` carries the rule and the reason. The reason is sharper
    here. To a reader, an invariant that nobody can see is the same as one that
    nobody enforces. A hard `inject=[INVARIANTS]` would make *declaring* an
    invariant a precondition for *enforcing* it, which inverts the two.
    """
    contribute_item(ctx, INVARIANTS, invariant, label=f"invariant({invariant.id})")


ORDER_FOLD_CACHE = 15
"""Where a fold-cache invariant sits: just after `session-log`, which it extends.

`session-log` asks whether the projections of a `Session` itself equal its
folds. These invariants ask the same about the caches that consumers keep *over*
that log. They are adjacent in the report because a person who reads one wants
the other. They come after `session-log` because a drifting log explains a
drifting fold over it, and the reverse is not true.
"""


def contribute_fold_cache(
    ctx: Context, *, id: str, subject: str, stale: Callable[[Iterable[Session]], Sequence[str]]
) -> None:
    """Declare that the `SessionFoldCache` of one consumer still equals its fold (I6).

    `stale` is the delegate of the owning service to `SessionFoldCache.stale`,
    and it is given the sessions to check. The store lookup happens here, so
    nine callers do not each write it. The statement is also written one time.
    Nine consumers cache a fold over the session log, and nine hand-written
    sentences would drift into nine slightly different promises about one
    property.

    **Each cache gets its own row, and one row does not cover all of them.** The
    reason is the one that `skills-invariant` gives for not folding itself into
    `tools-invariant`: they are different folds with different failure stories,
    and a person can act on a report that names which cache drifted. It also
    keeps each declaration inside the row that owns the cache. A profile that
    drops the seam then drops the claim with it, and does not report `holds`
    about a cache that nobody mounted.
    """

    def check() -> Sequence[str]:
        # The store is read at poll time and never captured at `apply` time,
        # because the store may mount after this row. With no store there is
        # nothing to enumerate, and no cached answer that could be served to
        # anybody. So `[]` is the correct answer and not a reassuring one.
        # `HarnessService.stale_projections` reads its sessions the same way,
        # for the same reason.
        sessions = ctx.get(SESSIONS)
        return [] if sessions is None else stale(sessions.list())

    contribute(
        ctx,
        Invariant(
            id=id,
            statement=f"every cached {subject} equals the fold of the log it projects",
            check=check,
            order=ORDER_FOLD_CACHE,
        ),
    )


@plugin("invariants", affects="deployment")
async def apply(ctx: Context, config: None) -> None:
    """Mount the registration seam. This row itself registers no invariant."""
    registry = InvariantRegistry(ctx=ctx)
    ctx.provide(INVARIANTS, registry)
    contribute_diagnostic(
        ctx,
        Diagnostic(
            id="invariants",
            title="Invariants",
            read=registry.describe,
            # Last: a person who scans `phern doctor` for what is wrong should
            # read the account that the deployment gives of itself before its
            # self-assessment.
            order=ORDER_SELF_ASSESSMENT,
        ),
    )
