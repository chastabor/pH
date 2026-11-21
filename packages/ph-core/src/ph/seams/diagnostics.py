"""`ctx.diagnostics`: what a row wants `phern doctor` to say about it.

Four rows in three packages came to need this seam, one after another: the
containment tier (P4-11), the workspace kind and `repo_writable` of each agent
(P4-07), the reach that a permission row reports for itself (P4-06, which noted
that it was the fourth row that wanted to hand `phern doctor` a reading, and
that this was the problem of `ctx.tui_status` again), and the worker model.
`ph_app` may not import `ph_stabilize` or `ph_rlm`. So without a seam, each
reading becomes a separate `ctx.<name>` that the consumer must know by heart,
and the consumer is the one command whose only job is to be complete.

**This is `ctx.tui_status` without the `Session`.** It has the same
registration shape, the same `scope=` and the same sort by order and then by
id. Both seams contain a contributor that raises: the footer drops the field,
and this seam prints one row that says that the section failed. The difference
is what each seam is read *for*. A footer field answers "where am I now" each
time the footer is sent, so it must be cheap. A diagnostic answers "what is this
deployment" one time, when a user asks, so it may spawn a subprocess or stat a
tree. These are two seams and not one seam with a parameter, because a field
that quietly became expensive would take the footer down with it.

**Rows, not a sentence.** `read()` returns `(label, value)` pairs.
`PathRoots.describe()` already returns that shape, and `doctor` already prints
it. So a contributor with several things to report (a kind and a
`repo_writable` for each agent) reports them as rows, and does not invent a
delimiter that the printer must learn.

**No severity.** §12 Q10 states that `phern doctor` prints the three columns of
the tier and not a severity color. A color invites a reader to skip the
sentence, and the sentence is the purpose of E1. So this seam deliberately has
no `level`, where `tui_status` has one.

@module ph.seams.diagnostics
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..cordis import Context, Disposer, Running, plugin, running
from ..keys import DIAGNOSTICS
from ._names import require_slug
from ._registry import claim_key, contribute_item

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

The bound is per seam, as `_names` intends: `_names` takes the bound as a
parameter so that each registry states its own. 32 is generous for a heading
that a person reads down the page, and to match the two sibling registries
costs nothing here."""

ORDER_SELF_ASSESSMENT = 100
"""Where the judgement of the deployment about itself sits: the section of
`invariants`, which is last. A reading about the deployment takes a lower order.
A section that places itself relative to that boundary names the constant, and
not a number two files away."""


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """The contribution of a row to `phern doctor`.

    `read` returns no rows when there is nothing to report, and the section is
    then omitted and not printed empty. `StatusField.read` returns `None` for
    the same reason: in a report that always shows every section, the section
    that matters cannot be found.
    """

    id: str
    read: Callable[[], Sequence[tuple[str, str]]]
    title: str = ""
    """The heading. It defaults to the id. It is a separate field so that a
    section can be called "Containment" while its id stays a slug."""
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

        The ordinary case no longer needs `scope=` (P6-12, P6-25). A
        registration made from the `apply` of a row, or from a listener that
        the row wrote, already unwinds with the row. Pass `scope=` to register
        on the lifetime of *another* owner. That is now its only meaning.
        """
        require_slug(diagnostic.id, maximum=ID_MAX, kind="diagnostic id")
        by = self.ctx.running_for(scope)
        return claim_key(
            by.owner, self._sections, diagnostic.id, _Registered(diagnostic, by), label="diagnostic"
        )

    def report(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """Every section that has something to report, by `order` and then by id.

        A contributor that raises does not stop the report. Its section prints
        one row that says that the section failed, and the traceback goes to
        the log. `phern doctor` is what a user runs *because* something is
        wrong, and the moment when one section fails is the worst moment to
        lose the other sections.
        """
        report: list[tuple[str, list[tuple[str, str]]]] = []
        ordered = sorted(
            self._sections.values(), key=lambda one: (one.diagnostic.order, one.diagnostic.id)
        )
        for entry in ordered:
            diagnostic = entry.diagnostic
            try:
                # Run as the row that contributed the section (P6-29). There is
                # no target, for the same reason that `tui_status.readings` has
                # none: `phern doctor` describes a deployment, not an agent.
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
    """Offer a section from the `apply` of a row, whether or not this seam is mounted.

    `contribute_via` carries the rule and the reason. The kernel-snapshot row
    of `ph_rlm` makes the same choice for the same reason: the compaction seam
    is a nice-to-have for that row, and a deployment that removed the
    compaction row must not lose kernel snapshots because of it.
    """
    contribute_item(ctx, DIAGNOSTICS, diagnostic, label=f"diagnostic({diagnostic.id})")


@plugin("diagnostics", affects="deployment")
async def apply(ctx: Context, config: None) -> None:
    """Mount the registration seam. No section ships in `ph-base`."""
    ctx.provide(DIAGNOSTICS, DiagnosticsRegistry(ctx=ctx))
