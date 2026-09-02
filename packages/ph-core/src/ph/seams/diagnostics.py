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

from ..cordis import Context, Disposer, Running, plugin, running
from ._names import require_slug
from ._registry import claim_key, contribute_via

__all__ = [
    "ID_MAX",
    "ORDER_SELF_ASSESSMENT",
    "Diagnostic",
    "DiagnosticsRegistry",
    "apply",
    "contribute",
]

log = logging.getLogger("ph.seams.diagnostics")

ID_MAX = 32
"""How long a section id may be.

Per-seam, as `_names` intends — it parameterises the bound precisely so each
registry states its own. 32 is generous for a heading a person reads down the
page, and matching the two sibling registries costs nothing here."""

ORDER_SELF_ASSESSMENT = 100
"""Where the deployment's judgement of itself sits — `invariants`' section, last.
A reading about the deployment orders below it; a section placing itself relative
to that boundary names the constant rather than a number two files away."""


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


@dataclass(frozen=True, slots=True)
class _Registered:
    """A section and who registered it (P6-29). See `ph.seams.tui_status`."""

    diagnostic: Diagnostic
    by: Running


@dataclass(slots=True)
class DiagnosticsRegistry:
    """The service published as `ctx.diagnostics`."""

    ctx: Context
    _sections: dict[str, _Registered] = field(default_factory=dict)

    def register(self, diagnostic: Diagnostic, *, scope: Context | None = None) -> Disposer:
        """Contribute a section.

        `scope=` is no longer needed for the ordinary case (P6-12, P6-25):
        a registration made from a row's `apply` — or from a listener that row
        wrote — already unwinds with the row. Pass it to register on *someone
        else's* lifetime, which is what it now means and all it now means.
        """
        require_slug(diagnostic.id, maximum=ID_MAX, kind="diagnostic id")
        by = self.ctx.running_for(scope)
        return claim_key(
            by.owner, self._sections, diagnostic.id, _Registered(diagnostic, by), label="diagnostic"
        )

    def report(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """Every section that has something to say, in `order` then id order.

        A contributor that raises is dropped with its traceback rather than
        taking the report down: `ph doctor` is what a person runs *because*
        something is wrong, and the one section that fails is the least
        acceptable moment to lose the other five.
        """
        report: list[tuple[str, list[tuple[str, str]]]] = []
        ordered = sorted(
            self._sections.values(), key=lambda one: (one.diagnostic.order, one.diagnostic.id)
        )
        for entry in ordered:
            diagnostic = entry.diagnostic
            try:
                # As the row that contributed it (P6-29); no target, for the same
                # reason `tui_status.readings` has none — `ph doctor` describes a
                # deployment, not an agent.
                with running(entry.by):
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

    `contribute_via` carries the rule and the reason. `ph_rlm`'s kernel-snapshot
    row made the same call for the same reason — "the compaction seam is a nice
    to have here, and a deployment that removed the compaction row must not
    thereby lose kernel snapshots".
    """
    contribute_via(ctx, "diagnostics", diagnostic, label=f"diagnostic({diagnostic.id})")


@plugin("diagnostics")
async def apply(ctx: Context, _config: Any) -> None:
    """Mount the registration seam. No section ships in `ph-base`."""
    ctx.provide("diagnostics", DiagnosticsRegistry(ctx=ctx))
