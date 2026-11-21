"""`ctx.tui_status`: a live reading in the footer, contributed by a row.

This seam is the sibling of `ctx.tui_screens`. It serves the other thing that a
row wants from a front end: not a whole screen, but one short reading beside the
model name and the context gauge. The seam generalizes the gauge. The gauge
shows the number where the harness will act, because that is the number that a
user needs to see coming. A limit is exactly that number for a different
mechanism.

**A reading, not a notice.** The two answer different questions, and both are
worth having. A notice in the transcript says why something *happened*, after
the fact. A reading says how close you are to it, before it happens. A budget
that announces itself only on the step where it stops you is a budget that you
cannot plan around.

**A semantic level, never a color.** `level` says that a reading is `warning`.
What that looks like is the business of the front end. `ContextForm` and
`CardKind` are held to the same rule. It is also the reason why ph-core can own
this seam without knowing what a terminal is.

**A field is read often, so it must be cheap.** The daemon reads every field
each time it sends the footer: when a client attaches, and with each
`session.status` (`ph_app.daemon.projections.readings_of`). Fold through
`SessionFoldCache` or read `Session.latest`, and never scan the log. A field
that raises is dropped and its traceback is logged, so it does not take the
footer down with it.

@module ph.seams.tui_status
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from ..cordis import Context, Disposer, Running, plugin, running
from ..keys import TUI_STATUS
from ..session import Session
from ..wire import WireModel
from ._names import require_slug
from ._registry import claim_key

__all__ = [
    "ID_MAX",
    "ReadingLevel",
    "ReadingSlot",
    "StatusField",
    "StatusReading",
    "TuiStatusRegistry",
    "apply",
]

log = logging.getLogger("ph.seams.tui_status")

ID_MAX = 32
"""How long a field id may be. It is the bound of `tui_screens`, for the same
reason: it shares a line with everything else that the footer says."""


ReadingLevel: TypeAlias = Literal["normal", "warning"]
"""How much attention a reading asks for.

The type is named and not inlined, and it is semantic and not visual. That is
the rule that `ContextForm` and `CardKind` are held to. There are two values
because two is what a footer can express without becoming a legend. A third
value would be a breaking change for every front end, which is the right amount
of friction."""


ReadingSlot: TypeAlias = Literal["line", "session", "model", "context"]
"""Where a reading belongs, semantically.

* `line`: the footer line, in `order`. This is the default.
* `session`: the session panel, beside the directory that a session is bound to.
* `model`: qualifies the model name that the line opens with. One model at two
  efforts is two costs and two answers, so the effort is drawn *onto* the name
  (`flash/high`) and not somewhere further down the line.
* `context`: qualifies the context gauge. A cache rate is the other half of
  "how much of the window is in use", so it is drawn onto the gauge
  (`context 12%/cache 28.8k(98%)`).

The two joined slots say *what figure a reading is about*, and not how the join
is drawn. The slash belongs to the front end, as the colors do. A front end with
no gauge to draw still draws a `context` reading, on its own.

The set is closed and small for the same reason as `ReadingLevel`. Each value is
a placement that every front end must be able to express. `StatusReading`
validates the value, so a client that was built before a new value refuses that
value (`PROTOCOL_VERSION` says so)."""


class StatusReading(WireModel):
    """What one field currently says.

    This is a `WireModel`, so it crosses a socket in both directions with
    nothing written by hand at the edge (P7-11): `to_wire()` out, and
    `model_validate` back. A field that is added here reaches a browser tab with
    no projection edited and no reader edited. A hand-written dict would have
    dropped the field silently on one side or the other. Every other type that a
    remote front end rebuilds uses the same mechanism, so there is one extras
    policy and not two.
    """

    id: str = ""
    """Which field this reading came from. The registry stamps it, and the row
    does not write it (see `readings`).

    A reading used to be an anonymous string on one line, so a front end could
    only render all of them in order. With a name, a *particular* reading can be
    placed. The sandbox posture belongs in the session panel of the sidebar,
    beside the directory that it bounds. The fact then does not travel as a
    special field on every status frame. A client that does not recognize an id
    still renders the reading on the line, which keeps a new reading additive."""
    slot: ReadingSlot = "line"
    """The `slot` of the contributing field. The registry stamps it beside `id`."""
    text: str
    """One compact phrase: `high`, `cache 28.8k(98%)`, `read-only accepted`.

    A front end puts its own marks between readings, and between a reading and
    the figure that its slot names. So a reading carries no separator of its
    own. A row that wrote one would be drawn as two readings where it meant
    one."""
    level: ReadingLevel = "normal"


@dataclass(frozen=True, slots=True)
class StatusField:
    """The contribution of a row to the footer.

    `read` returns `None` when the field has nothing to say: an unset limit, or
    a counter at zero. The footer then shows nothing, and not a placeholder. On
    a line that always carries every field, the one that matters cannot be seen.
    """

    id: str
    read: Callable[[Session], StatusReading | None]
    order: int = 0
    slot: ReadingSlot = "line"
    """Where this belongs on screen, semantically. It does not say which widget
    draws it.

    It has the shape of `level`, for the same reason: the meaning belongs to the
    contributing row and the *looks* belong to the front end. The sandbox
    posture is a fact about the session, because it bounds the directory that
    the session panel already names. A budget is a live figure that belongs on
    the line with the others. The reasoning effort qualifies the model name
    (`model`) and the cache rate qualifies the context gauge (`context`). Each
    is half of a figure that the footer already draws, and is drawn onto it.

    A front end that draws only one region ignores this and renders everything.
    The alternative that this replaced was a hardcoded set of ids in the status
    widget of `ph_app`. That put a placement decision about a ph-core row in the
    consumer, where a renamed id would silently relocate the reading."""


@dataclass(frozen=True, slots=True)
class _Registered:
    """A field and who registered it (P6-29).

    The two are in the table and not beside it, so the identity-checked release
    of `claim_key` takes both away in one step. `ph.seams.commands` holds one for
    the same reason."""

    status_field: StatusField
    by: Running


@dataclass(slots=True)
class TuiStatusRegistry:
    """The service published as `ctx.tui_status`."""

    ctx: Context
    _fields: dict[str, _Registered] = field(default_factory=dict)

    def register(self, status_field: StatusField, *, scope: Context | None = None) -> Disposer:
        """Contribute a field.

        The ordinary case needs no `scope=` (P6-12, P6-25). A registration made
        from the `apply` of a row, or from a listener that the row wrote,
        already unwinds with the row. Pass `scope=` to register on the lifetime
        of *someone else*. That is its only meaning now.
        """
        require_slug(status_field.id, maximum=ID_MAX, kind="status field id")
        by = self.ctx.running_for(scope)
        return claim_key(
            by.owner,
            self._fields,
            status_field.id,
            _Registered(status_field, by),
            label="status-field",
        )

    def readings(self, session: Session) -> list[StatusReading]:
        """Every field that has something to say, in `order` then id order.

        **`read` runs as the row that contributed it** (P6-29). It is the body
        of a row that this registry invokes, the same category as the `execute`
        of a tool. It used to run unbound, so anything that a field registered
        landed on the seam. There is no *target* here, as `compaction.notes` and
        `system_prompt.assemble` have one. The footer is one line for one
        session and not a per-agent view. So both halves are what registration
        recorded, and a reading is contained to wherever the field was declared.
        """
        ordered = sorted(
            self._fields.values(), key=lambda one: (one.status_field.order, one.status_field.id)
        )
        readings: list[StatusReading] = []
        for entry in ordered:
            status_field = entry.status_field
            try:
                with running(entry.by):
                    reading = status_field.read(session)
            except Exception:
                log.warning(
                    "ph.seams.tui_status: field %r failed to read", status_field.id, exc_info=True
                )
                continue
            if reading is not None:
                # The id and the slot are stamped here and not trusted from the
                # row. The id is the key of the registry itself. A field that
                # was free to answer with the id of another field would let the
                # reading of one row be placed as the reading of another.
                readings.append(
                    reading.model_copy(update={"id": status_field.id, "slot": status_field.slot})
                )
        return readings


@plugin("tui-status", affects="presentation")
async def apply(ctx: Context, config: None) -> None:
    """Mount the registration seam of the footer. This row itself registers no field."""
    ctx.provide(TUI_STATUS, TuiStatusRegistry(ctx=ctx))
