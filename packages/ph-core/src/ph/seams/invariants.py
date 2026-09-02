"""`ctx.invariants` — which invariants this deployment enforces, and whether they hold.

pH's invariants are enforced by rows, and a row is optional. So "I3 holds" was
never a property of pH; it was a property of a profile that happened to mount
`agent-loop-invariant` — and nothing said which profiles those were. A person
reading `DESIGN.md` learned what pH promises, ran a profile that promised less,
and had no way to find out. **That gap is what this registry closes**: an
invariant that is enforced says so out loud, and one that nobody mounted is
absent from the report rather than silently assumed.

**Two kinds, and the difference is load-bearing.** Some invariants are enforced
*inline* — I3 checks every request as it is built, and refuses. Those cannot be
polled, because there is no state between requests to poll: the answer to "does
it hold" is "every request so far was checked". Others are *pollable* — a
projection either equals its fold right now or it does not — and those carry a
`check`. Reporting the two identically would be the overstatement E1 forbids in
both directions: an inline invariant reported as "holds" claims a check that did
not run, and a pollable one reported as "enforced" claims a guarantee about a
file nobody read.

**A check that raises is a violation, not a dropped row.** This is where the
seam parts company with `ctx.diagnostics`, which drops a failing section so the
report survives. An invariant is the thing that says something is wrong; a check
that cannot complete is at best "unknown" and at worst the failure itself, and
swallowing it would make the quietest possible report the one where the most is
broken.

**No severity, for `diagnostics`' reason** (§12 Q10): a violated invariant is a
sentence a person reads, not a colour they learn to skip.

**Not enforced (§5 rule 6): a poll has content only in a live process.** The one
caller of `ctx.diagnostics.report()` today is `ph doctor`, which mounts a fresh
profile — no session created, no view cached, no scope disposed — so "holds"
there says the checks run, not that the deployment's live state passed them. The
daemon is where the same report has something to look at, and wiring the poll
into it is a separate row.

@module ph.seams.invariants
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..cordis import Context, Disposer, Running, plugin, running
from ._names import require_slug
from ._registry import claim_key, contribute_via
from .diagnostics import ORDER_SELF_ASSESSMENT, Diagnostic
from .diagnostics import contribute as contribute_diagnostic

__all__ = [
    "ID_MAX",
    "Invariant",
    "InvariantRegistry",
    "Violation",
    "apply",
    "contribute",
]

log = logging.getLogger("ph.seams.invariants")

ID_MAX = 32
"""How long an invariant id may be. Matches the two sibling registries."""


@dataclass(frozen=True, slots=True)
class Invariant:
    """One thing a row promises is true, and the way to find out."""

    id: str
    statement: str
    """What must hold, in one sentence, phrased so a violation contradicts it.

    Read aloud in the report next to the answer, because "A11 holds" tells a
    person nothing they can act on and "every projection equals its fold: holds"
    tells them what was actually checked."""
    check: Callable[[], Sequence[str]] | None = None
    """Violations found right now, empty when it holds.

    `None` for an invariant enforced *inline* — one whose rule runs on the path
    it governs and refuses there. There is nothing to poll in that case, and a
    `check` returning `[]` would be a lie shaped like reassurance: it would
    report "holds" on a deployment where the enforcing listener had never been
    reached at all."""
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
        """Declare an invariant this deployment enforces.

        `scope=` means what it means everywhere else (P6-12, P6-25): a
        registration from a row's own `apply` already unwinds with the row, so
        pass it only to register on someone else's lifetime. **An invariant that
        unwinds with its enforcer is the point** — a row that is unloaded stops
        promising, and the report stops claiming.
        """
        require_slug(invariant.id, maximum=ID_MAX, kind="invariant id")
        by = self.ctx.running_for(scope)
        return claim_key(
            by.owner, self._entries, invariant.id, _Registered(invariant, by), label="invariant"
        )

    def _checked(self) -> list[tuple[Invariant, list[str] | None]]:
        """Every invariant in `order` then id order, with what its check found.

        `None` for an inline one, which has nothing to poll; an empty list for a
        pollable one that holds. The one pass both readers project from, so the
        order a report prints and the order checks run cannot drift apart.

        A check that raises becomes a finding carrying its exception. It is not
        dropped: this seam exists to report that something is wrong, and a
        failure to determine that is not evidence of health.
        """
        checked: list[tuple[Invariant, list[str] | None]] = []
        for entry in self._ordered():
            invariant = entry.invariant
            if invariant.check is None:
                checked.append((invariant, None))
                continue
            try:
                # As the row that registered it (P6-29), so a check that reads a
                # scoped registry sees what its own row would see.
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

        Empty means every *pollable* invariant holds — never that every invariant
        holds, which no poll can establish. `enforced()` is what says which ones
        were in scope of that claim.
        """
        return [
            Violation(invariant.id, detail)
            for invariant, found in self._checked()
            if found
            for detail in found
        ]

    def describe(self) -> list[tuple[str, str]]:
        """The `ph doctor` rows: one per invariant, answer first.

        Checks run here rather than being cached from somewhere earlier, because
        the question a person is asking by typing `ph doctor` is about now.
        `diagnostics` permits the cost — that is the whole distinction it draws
        against `tui_status`, which is read on every spinner frame.
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
    """Declare an invariant from a row's `apply`, whether or not this seam is mounted.

    `contribute_via` carries the rule and the reason. The sharper form of it here:
    an invariant nobody can see is the same, to a reader, as one nobody enforces —
    and a hard `inject=["invariants"]` would make *declaring* an invariant a
    precondition for *enforcing* it, which inverts the two.
    """
    contribute_via(ctx, "invariants", invariant, label=f"invariant({invariant.id})")


@plugin("invariants")
async def apply(ctx: Context, _config: Any) -> None:
    """Mount the registration seam. No invariant ships in `ph-base` from here."""
    registry = InvariantRegistry(ctx=ctx)
    ctx.provide("invariants", registry)
    contribute_diagnostic(
        ctx,
        Diagnostic(
            id="invariants",
            title="Invariants",
            read=registry.describe,
            # Last: a person scanning `ph doctor` for what is wrong should meet
            # the deployment's own account of itself before its self-assessment.
            order=ORDER_SELF_ASSESSMENT,
        ),
    )
