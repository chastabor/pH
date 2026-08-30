"""`ctx.tui_status` — a live reading in the footer, contributed by a row.

The sibling of `ctx.tui_screens`, for the other thing a row wants from a front
end: not a whole screen, but one short reading beside the model name and the
context gauge. The gauge is the shape being generalized — *"the number a user
needs to see coming is the one where the harness will act"* — and a limit is
exactly that number for a different mechanism.

**A reading, not a notice.** The two answer different questions and both are
worth having: a notice in the transcript says why something *happened*, after
the fact; a reading says how close you are to it happening, before. A budget
that only announces itself on the step it stops you is a budget you cannot plan
around.

**Semantic level, never a colour.** `level` says a reading is `warning`; what
that looks like is the front end's business — the same rule `ContextForm` and
`CardKind` are held to, and the reason ph-core can own this seam without knowing
what a terminal is.

**A field is read on every redraw**, which for a running agent is every spinner
frame. It must therefore be cheap: fold through `SessionFoldCache` or read
`Session.latest`, never scan the log. A field that raises is dropped with its
traceback rather than taking the footer down with it.

@module ph.seams.tui_status
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from ..cordis import Context, Disposer, plugin
from ..session import Session
from ._names import require_slug
from ._registry import claim_key

__all__ = [
    "ID_MAX",
    "ReadingLevel",
    "StatusField",
    "StatusReading",
    "TuiStatusRegistry",
    "apply",
]

log = logging.getLogger("ph.seams.tui_status")

ID_MAX = 32
"""How long a field id may be — `tui_screens`' bound, for the same reason: it
shares a line with everything else the footer says."""


ReadingLevel: TypeAlias = Literal["normal", "warning"]
"""How much attention a reading is asking for.

Named rather than inlined, and semantic rather than visual — the rule
`ContextForm` and `CardKind` are held to. Two values because two is what a
footer can express without becoming a legend; a third would be a breaking
change for every front end, which is the right amount of friction."""


@dataclass(frozen=True, slots=True)
class StatusReading:
    """What one field currently says."""

    text: str
    level: ReadingLevel = "normal"


@dataclass(frozen=True, slots=True)
class StatusField:
    """A row's contribution to the footer.

    `read` returns `None` when the field has nothing to say — an unset limit, a
    counter at zero — and the footer shows nothing rather than a placeholder. A
    line that always carries every field is a line where the one that matters
    cannot be seen.
    """

    id: str
    read: Callable[[Session], StatusReading | None]
    order: int = 0


@dataclass(slots=True)
class TuiStatusRegistry:
    """The service published as `ctx.tui_status`."""

    ctx: Context
    _fields: dict[str, StatusField] = field(default_factory=dict)

    def register(self, status_field: StatusField, *, scope: Context | None = None) -> Disposer:
        """Contribute a field.

        `scope=` is no longer needed for the ordinary case (P6-12, P6-25):
        a registration made from a row's `apply` — or from a listener that row
        wrote — already unwinds with the row. Pass it to register on *someone
        else's* lifetime, which is what it now means and all it now means.
        """
        require_slug(status_field.id, maximum=ID_MAX, kind="status field id")
        owner = self.ctx.owner_for(scope)
        return claim_key(owner, self._fields, status_field.id, status_field, label="status-field")

    def readings(self, session: Session) -> list[StatusReading]:
        """Every field that has something to say, in `order` then id order."""
        ordered = sorted(self._fields.values(), key=lambda one: (one.order, one.id))
        readings: list[StatusReading] = []
        for status_field in ordered:
            try:
                reading = status_field.read(session)
            except Exception:
                log.warning(
                    "ph.seams.tui_status: field %r failed to read", status_field.id, exc_info=True
                )
                continue
            if reading is not None:
                readings.append(reading)
        return readings


@plugin("tui-status")
async def apply(ctx: Context, config: object) -> None:
    """Mount the footer's registration seam. No field ships in `ph-base`."""
    ctx.provide("tui_status", TuiStatusRegistry(ctx=ctx))
