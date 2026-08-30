"""`ctx.diagnostics` — what a row wants `ph doctor` to say about it.

Four rows in three packages arrived at this one by one: the containment tier
(P4-11), the workspace kind and `repo_writable` per agent (P4-07), a permission
row's honest reach (P4-06, which said out loud that it was "the fourth row
wanting to hand `ph doctor` a reading... which is `ctx.tui_status`' problem
again"), and the worker model. `ph-app` cannot import `ph-stabilize` or
`ph-rlm`, so without a seam each of them lands as a bespoke `ctx.<name>` the
consumer has to know by heart — and the consumer is the one command whose entire
job is to be complete.

**This is `ctx.tui_status` minus the `Session`.** Same registration shape, same
`scope=`, same drop-a-raising-contributor rule, same order-then-id sort. The
difference is what it is read *for*: a footer field answers "where am I now" on
every spinner frame, so it must be cheap; a diagnostic answers "what is this
deployment" once, at a person's request, so it may spawn a subprocess or stat a
tree. Two seams rather than one parameterised seam, because a field that quietly
became expensive would take the footer down with it.

**Rows, not a sentence.** `read()` returns `(label, value)` pairs — the shape
`PathRoots.describe()` already returns and `doctor` already prints — so a
contributor that has several things to say (a kind and a `repo_writable` per
agent) says them as rows rather than by inventing a delimiter the printer has to
learn.

**No severity.** §12 Q10 is explicit that `ph doctor` prints the tier's three
columns "rather than a severity colour", because a colour invites a reader to
skip the sentence — and the sentence is the entire point of E1. So there is no
`level` here, deliberately, where `tui_status` has one.

@module ph.seams.diagnostics
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..cordis import Context, Disposer, plugin
from ._names import require_slug
from ._registry import claim_key

__all__ = ["ID_MAX", "Diagnostic", "DiagnosticsRegistry", "apply", "contribute"]

log = logging.getLogger("ph.seams.diagnostics")

ID_MAX = 32
"""How long a section id may be.

Per-seam, as `_names` intends — it parameterises the bound precisely so each
registry states its own. 32 is generous for a heading a person reads down the
page, and matching the two sibling registries costs nothing here."""


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A row's contribution to `ph doctor`.

    `read` returns no rows when there is nothing to report, and the section is
    omitted rather than printed empty — the same reason `StatusField.read`
    returns `None`: a report that always shows every section is one where the
    section that matters cannot be found.
    """

    id: str
    read: Callable[[], Sequence[tuple[str, str]]]
    title: str = ""
    """The heading, defaulting to the id. Spelled separately so a section can be
    called "Containment" while its id stays a slug."""
    order: int = 0


@dataclass(slots=True)
class DiagnosticsRegistry:
    """The service published as `ctx.diagnostics`."""

    ctx: Context
    _sections: dict[str, Diagnostic] = field(default_factory=dict)

    def register(self, diagnostic: Diagnostic, *, scope: Context | None = None) -> Disposer:
        """Contribute a section.

        `scope=` is no longer needed for the ordinary case (P6-12, P6-25):
        a registration made from a row's `apply` — or from a listener that row
        wrote — already unwinds with the row. Pass it to register on *someone
        else's* lifetime, which is what it now means and all it now means.
        """
        require_slug(diagnostic.id, maximum=ID_MAX, kind="diagnostic id")
        owner = self.ctx.owner_for(scope)
        return claim_key(owner, self._sections, diagnostic.id, diagnostic, label="diagnostic")

    def report(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """Every section that has something to say, in `order` then id order.

        A contributor that raises is dropped with its traceback rather than
        taking the report down: `ph doctor` is what a person runs *because*
        something is wrong, and the one section that fails is the least
        acceptable moment to lose the other five.
        """
        report: list[tuple[str, list[tuple[str, str]]]] = []
        for diagnostic in sorted(self._sections.values(), key=lambda one: (one.order, one.id)):
            try:
                rows = list(diagnostic.read())
            except Exception:
                log.warning(
                    "ph.seams.diagnostics: section %r failed to read", diagnostic.id, exc_info=True
                )
                rows = [("(this section failed)", "see the log for the traceback")]
            if rows:
                report.append((diagnostic.title or diagnostic.id, rows))
        return report


def contribute(ctx: Context, diagnostic: Diagnostic) -> None:
    """Offer a section from a row's `apply`, whether or not this seam is mounted.

    **Through `ctx.inject` rather than a `ctx.get` at `apply` time**, which is
    the difference between a contribution that works and one that works if the
    rows happen to be in the right order. A `ctx.get` here reads whatever has
    been provided *so far*, so a contributor mounted above `diagnostics` finds
    nothing, contributes nothing, and reports that nowhere — for the one command
    whose whole job is to be complete. `inject` waits for the key through the
    loader's reconcile fixpoint instead, so the two rows may sit in either
    order, which is what `base.yaml` promises about every other row in it.

    Optional, still: a hard `inject=["diagnostics"]` on the row itself would
    make a *report* a precondition for the thing being reported on. `ph_rlm`'s
    kernel-snapshot row made exactly this call for exactly this reason — "the
    compaction seam is a nice to have here, and a deployment that removed the
    compaction row must not thereby lose kernel snapshots".

    The unwind comes free: `fn` is handed a child scope that disposes when the
    key goes away or the row unloads, so the section leaves with whatever
    answers it and no caller has to remember `scope=`.
    """

    def register(scope: Context) -> None:
        scope.diagnostics.register(diagnostic, scope=scope)

    ctx.inject(["diagnostics"], register, label=f"diagnostic({diagnostic.id})")


@plugin("diagnostics")
async def apply(ctx: Context, _config: Any) -> None:
    """Mount the registration seam. No section ships in `ph-base`."""
    ctx.provide("diagnostics", DiagnosticsRegistry(ctx=ctx))
