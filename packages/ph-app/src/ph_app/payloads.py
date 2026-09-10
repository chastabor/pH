"""What the daemon *emits*, as models — the other half of P8-07 (P8-08).

P8-07 typed what the daemon **receives**: one `WireModel` per method's params,
checked at the edge by `parse_params`. What it answers and announces stayed
`dict[str, Any]`, built by hand at 30-odd sites and read by hand at as many
more — so `Peer.send`'s rule ("what goes out is ours to get right, and the type
is what checks it") held for the four envelope shapes and for nothing they
carry.

The cost was not hypothetical. **`session.status` had three payload shapes and
one reader.** `supervisor.py` published `{sessionId, status}` for a retry or a
passivation, `{sessionId, status, lastTurn, readings}` when the agent moved, and
`session/attach` replied with `describe()` plus the footer — which
`follow.py.seed` deliberately feeds to the same sink. `tui/remote.py._status`
absorbed all three with `str(params.get("provider") or self.state.provider)` and
a comment calling the shapes "honestly different". That comment was a union type
written as prose, and this module is it written as a type: `StatusFacts` has
every field optional because a frame states *what changed*, and `None` means
"not stated here, keep what you have" rather than "cleared".

**Here rather than in `daemon/`, and not in `protocol.py`.** A front end reads
these and must not import `ph_app.daemon.server` (`test_app_layering`), so they
cannot live beside the handlers that build them. `protocol.py` is the envelope
and the two shapes both transports share (`Cursor`, `NoParams`, `SessionParams`);
these are payloads, and `--mode rpc` emits two of them, so a module of their own
beside `protocol.py` is where both ends can reach them without either importing
the other's machinery.

**What stays a `dict[str, Any]`, on purpose.** `SessionEventNotice.event` is a
session event's own wire envelope — `SessionEvent.from_wire` is what validates
it, and re-declaring the log's envelope here would be the second spelling
`ph.session.events` exists to prevent. `presentation` is a rendered card whose
shape is the tool registry's. Both are handed on to something that knows them.

@module ph_app.payloads
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import Field

from ph.llm.types import AttachmentRef, ToolSchema
from ph.seams.approval import ApprovalRequest
from ph.seams.commands import CommandSchema
from ph.seams.tui_screens import ScreenSchema
from ph.seams.tui_status import StatusReading
from ph.seams.user_questions import UserQuestion
from ph.wire import WireModel, wire_alias

from .protocol import Cursor
from .sessions import SessionSummary

__all__ = [
    "NOTICES",
    "ApprovalAsk",
    "ApprovalAskReply",
    "AskSettledNotice",
    "AttachReply",
    "MutationRepeated",
    "QuestionAsk",
    "QuestionAskReply",
    "RootDescription",
    "RootDetail",
    "RootListing",
    "RootStatusReply",
    "SessionAsk",
    "SessionBrowse",
    "SessionCommandsNotice",
    "SessionEventNotice",
    "SessionNotice",
    "SessionReadingsReply",
    "SessionScoped",
    "SessionScreensNotice",
    "SessionStagedNotice",
    "SessionStatusNotice",
    "SessionToolsReply",
    "SnapshotPage",
    "StatusFacts",
    "notice_of",
]


class _CarriesJson(WireModel):
    """A payload with fields that are **already** wire JSON.

    `event`, `events`, `presentations` and `schedules` are typed `dict[str, Any]`
    / `list[dict[str, Any]]` because they are somebody else's shape — a session
    event's own envelope, a rendered card, a schedule row. Pydantic cannot know
    that, so `model_dump` walks them and rebuilds every nested container on the
    way out. That is a deep copy of a tree whose only destination is `dumps`.

    **Measured, and the reason this class exists:** a 200-block
    `assistant/message` notice dumped in 21.8 µs where the dict it replaced cost
    0.049 µs, and a 2048-event `session/snapshot` page cost 1.3 ms — per page,
    on top of the encode that has to happen anyway. `session.event` is the
    hottest frame the daemon sends, one per appended event per watcher.

    So the named fields are excluded from the dump and re-attached **by
    reference**. The output is byte-identical (`test_payloads` pins that); what
    changes is that a tree already in wire form is not rebuilt to prove it.
    """

    BY_REFERENCE: ClassVar[frozenset[str]] = frozenset()

    def to_wire(self) -> dict[str, Any]:
        wire = self.model_dump(by_alias=True, exclude_none=True, exclude=set(self.BY_REFERENCE))
        for name in self.BY_REFERENCE:
            value = getattr(self, name)
            if value is not None:
                wire[wire_alias(name)] = value
        return wire


# ------------------------------------------------------------------ status --


class StatusFacts(WireModel):
    """What a frame may say about where a root is, all of it optional.

    The union `tui/remote.py` was absorbing in prose. Three frames reach one
    reader and each states only what it knows: a `passivated` notice has a
    status and no footer, an `announce` has the footer too, and the attach reply
    has the route as well because a front end that has just connected has a
    footer to draw *now*. `None` therefore means "this frame does not say",
    never "cleared" — a reader keeps what it had, which is what every one of
    them already did by hand.

    `status` defaults to `""` rather than `None` for the same reason it is not a
    `Literal`: it is `str(agent.status)` plus the three the supervisor derives,
    and a closed set here would refuse a phase the driver added.
    """

    status: str = ""
    last_turn: str | None = None
    readings: list[StatusReading] | None = None
    provider: str | None = None
    model: str | None = None


# ----------------------------------------------------------------- replies --


class RootDescription(WireModel):
    """`Root.describe()` — what a client is told about one root.

    One name per fact: `rootId` and `sessionId` were the same string (a root
    *is* its session here) and `events` was `cursor.sequence`, which left a
    client picking which of two spellings was authoritative.
    """

    session_id: str
    status: str
    last_turn: str | None = None
    watchers: int
    cursor: Cursor
    provider: str = ""
    model: str = ""

    def facts(self) -> StatusFacts:
        """This description as the status half of it.

        `follow.py` feeds the attach reply to the same sink `session.status`
        goes to — the server's own docstring says they are the same shape — and
        this is that claim as a conversion rather than as a sentence.
        """
        return StatusFacts(
            status=self.status,
            last_turn=self.last_turn,
            provider=self.provider,
            model=self.model,
        )


class MutationRepeated(RootDescription):
    """A mutation whose idempotence key had already been claimed.

    One shape for every verb, so a client branches on one field: `MUTATIONS`'
    docstring states that rule, and this is it as a type rather than a
    `{**describe(), "repeated": True}` spread at the wrapper.
    """

    repeated: bool = True


class RootDetail(RootDescription):
    """`Root.detail()` — `describe`, plus what a client asking about *one* root
    wants. `status` collapses the retry ladder to `retrying` or `failed`; these
    are the rungs behind that answer."""

    attempts: int
    failed: bool


class RootStatusReply(RootDetail, _CarriesJson):
    """`session/status` — one root in detail, with what is still going to fire."""

    BY_REFERENCE: ClassVar[frozenset[str]] = frozenset({"schedules"})
    schedules: list[dict[str, Any]] = Field(default_factory=list)
    """`ph.seams.schedule.state_to_wire`'s hand-built rows. A model for them is a
    ph-core change; until then they pass through as the JSON they already are."""


class AttachReply(RootDescription):
    """`session/attach` — the description, with the footer beside it.

    The footer rides along because it belongs to the status it describes: a
    client that has just attached draws a complete one without a second call.
    """

    readings: list[StatusReading] = Field(default_factory=list)

    def facts(self) -> StatusFacts:
        """The description's status half, plus the footer this reply carries."""
        return super().facts().model_copy(update={"readings": self.readings})


class SnapshotPage(_CarriesJson):
    """`session/snapshot` — one bounded page of history, and the next cursor.

    2048 events a page, so the rebuild `_CarriesJson` avoids is the largest
    single one in the daemon.
    """

    BY_REFERENCE: ClassVar[frozenset[str]] = frozenset({"events", "presentations"})
    session_id: str
    events: list[dict[str, Any]] = Field(default_factory=list)
    """Session-event wire envelopes, validated by `SessionEvent.from_wire` where
    they are folded. Not re-declared here — see the module docstring."""
    presentations: dict[str, Any] = Field(default_factory=dict)
    """Rendered cards, keyed by `seq` as a string and **sparse**: a page is 2048
    events and a turn contributes a handful."""
    started_at: int = Field(default=0, validation_alias="from", serialization_alias="from")
    """Where the read actually began, which is not always where the cursor
    asked: `resume_at` answers a cursor from another incarnation of the log with
    0.

    The one field in this module whose wire name is not `wire_alias`'s answer:
    `from` is a Python keyword, so the attribute cannot be called that. Spelled
    as the two directional aliases rather than the single `alias=`, because
    without the pydantic mypy plugin a bare `alias` becomes the *parameter*
    name in the synthesized `__init__` — and `SnapshotPage(from=...)` is a
    syntax error, which makes the constructor unreachable by name."""
    cursor: Cursor
    more: bool = False


class RootListing(WireModel):
    """`sessions/list` — the roots this daemon is holding."""

    sessions: list[RootDescription] = Field(default_factory=list)


class SessionBrowse(WireModel):
    """`sessions/browse` — every session a person could open, stored and live.

    A separate model from `RootListing` despite the shared field name: the rows
    are `SessionSummary`, not `RootDescription`, and the two have disjoint
    fields. One model over both could only have typed the rows as `dict`, which
    is what it did — a name with no checking behind it, and a `.to_wire()` loop
    at each producer to feed it. Nested models dump recursively, so naming the
    row type is what deletes those loops.
    """

    sessions: list[SessionSummary] = Field(default_factory=list)


# **`daemon/status` is deliberately not modelled.** It is a diagnostics blob
# with one producer and one reader (`ph agents doctor`'s table), and it carries
# `DiagnosticsRegistry.report()`'s nested section shape verbatim — so a model
# here would be a second declaration of that, for a reply nothing else reads.
# The shapes worth a type are the ones with several producers or several
# readers, which is what made `session.status` worth one.


# ----------------------------------------------------------- notifications --


class SessionScoped(WireModel):
    """A frame about one root, in either direction.

    `METHOD` is the name this payload travels under, declared on the payload
    rather than passed beside it: `Root.publish` takes the notice and reads the
    name off it, so publishing a command list as `session.status` stops being a
    thing anyone can write. It was two arguments that had to agree, at eight
    call sites, with nothing checking that they did.

    The base exists so that the two directions are **siblings rather than one
    inheriting the other**. An ask is not a notification — it expects a reply —
    and while `ApprovalAsk` subclassed `SessionNotice` the only thing keeping it
    out of the notice table was an author remembering to omit it, plus a
    denylist in the test saying so a second time. A reader that found
    `approval/ask` among the notices would treat a question as an event and
    never answer it; now `Mapping[str, type[SessionNotice]]` cannot hold one.
    """

    METHOD: ClassVar[str] = ""
    session_id: str


class SessionNotice(SessionScoped):
    """A frame the daemon *announces*: no reply, no id, watch or ignore."""


class SessionAsk(SessionScoped):
    """A frame the daemon *asks*: it expects a typed answer back.

    `ask_id` lives here rather than on each ask — it was declared twice, and
    `AskDesk` keys its pending table on it either way.
    """

    ask_id: str


class SessionEventNotice(SessionNotice, _CarriesJson):
    """`session.event` — one appended event, with its card when it has one.

    The hottest frame the daemon sends: one per appended event, per watcher,
    for every streamed chunk. `_CarriesJson` is why both payload fields leave
    by reference rather than being rebuilt.
    """

    METHOD: ClassVar[str] = "session.event"
    BY_REFERENCE: ClassVar[frozenset[str]] = frozenset({"event", "presentation"})
    event: dict[str, Any]
    presentation: dict[str, Any] | None = None


class SessionStatusNotice(SessionNotice, StatusFacts):
    """`session.status` — where the root is now, as much of it as changed.

    Both bases on purpose: it is a `StatusFacts` (which is what the reader
    wants) that also names its root (which is what the dispatcher wants).
    """

    METHOD: ClassVar[str] = "session.status"


class SessionCommandsNotice(SessionNotice):
    """`session.commands` — the whole palette, not a delta."""

    METHOD: ClassVar[str] = "session.commands"
    commands: list[CommandSchema] = Field(default_factory=list)


class SessionScreensNotice(SessionNotice):
    """`session.screens` — the whole screen list, for the reason `commands` is
    whole: `ScreenDefinition.build` cannot travel, so a client keeps only the
    ids it can draw and re-reads the projection rather than applying a delta."""

    METHOD: ClassVar[str] = "session.screens"
    screens: list[ScreenSchema] = Field(default_factory=list)


class SessionReadingsReply(SessionNotice):
    """`session/readings` — the footer, asked for rather than pushed.

    A reply, not a notice, so it declares no `METHOD`: the footer *is* pushed,
    but it rides `session.status` beside the status it belongs to rather than
    travelling under a name of its own.
    """

    readings: list[StatusReading] = Field(default_factory=list)


class SessionToolsReply(SessionNotice):
    """`tools/list` — what the model may call in this deployment."""

    tools: list[ToolSchema] = Field(default_factory=list)


class SessionStagedNotice(SessionNotice):
    """`session.staged` — the composer's tray, which is shared: a chip only the
    uploader can see is a composer nobody else can reason about."""

    METHOD: ClassVar[str] = "session.staged"
    staged: list[AttachmentRef] = Field(default_factory=list)


class AskSettledNotice(SessionNotice):
    """`ask.settled` — somebody else answered; take the modal down."""

    METHOD: ClassVar[str] = "ask.settled"
    ask_id: str


NOTICES: Mapping[str, type[SessionNotice]] = {
    one.METHOD: one
    for one in (
        SessionEventNotice,
        SessionStatusNotice,
        SessionCommandsNotice,
        SessionScreensNotice,
        SessionStagedNotice,
        AskSettledNotice,
    )
}
"""Wire method → the payload that travels under it, for a client reading frames.

The client's half of the daemon's `METHODS`: the daemon has a table saying what
each name *takes*, and this says what each name *carries*. Both dispatchers had
the pairing written out branch by branch — `if method == X.METHOD: X.model_validate(...)`
— which is the mechanism, once per reader, with nothing checking that a notice
the daemon publishes has a reader that knows its model. `test_payloads` holds
this against every `SessionNotice` subclass that declares a `METHOD`, so a
seventh notice cannot be added without appearing here.

The asks cannot appear here — `SessionAsk` is a sibling of `SessionNotice`,
not a subclass — which is the point of the split: a reader that found
`approval/ask` among the notices would treat a question as an event and never
answer it.
"""


FED: frozenset[str] = frozenset({SessionEventNotice.METHOD, SessionStatusNotice.METHOD})
"""The two a *feed* reads: the transcript's events and where the root is.

Named once because two readers gate on it and both were spelling it inline —
and because a class-attribute load off a pydantic model never specializes
(`ModelMetaclass` defines `__getattr__`), so the tuple those branches built per
notification measured 0.105 µs against 0.021 for this. The other four notices
are palette and tray snapshots a feed has no use for; validating them to find
that out measured **129x** the string test it replaced, on a `session.commands`
carrying a 25-command palette."""


def notice_of(method: str, params: dict[str, Any]) -> SessionNotice | None:
    """One inbound frame as the payload its method declares, or `None`.

    `None` for a method this build has no model for — a daemon newer than this
    client, which a reader drops rather than crashes on. That is the one thing
    the branch-by-branch form got right by accident and this states: an unknown
    notification is not an error, it is a feature the client does not have.

    A payload that does *not* parse still raises, and that is deliberate now
    that it is survivable: `Peer._read` guards the notification body, so a bad
    frame costs a frame and says so in the log rather than being swallowed
    here — a dropped `session.event` is a hole in a transcript, which is worth
    a line. `ph_app.wire.view_of` swallows for the opposite reason: a card that
    will not parse costs a card, and the generic one drawn instead is plain
    rather than wrong.
    """
    model = NOTICES.get(method)
    return None if model is None else model.model_validate(params)


# -------------------------------------------------------------------- asks --
# The daemon → client direction, which is the only place the daemon is the one
# building *request* params. Modelled here rather than in `ph_app.params`
# because the client is what validates them.


class ApprovalAsk(SessionAsk):
    """`approval/ask` — put this to the person and tell me what they said."""

    METHOD: ClassVar[str] = "approval/ask"
    request: ApprovalRequest


class ApprovalAskReply(WireModel):
    """What a front end answers an `approval/ask` with.

    `reason` travels rather than being steered client-side: the daemon holds the
    agent, and a client steering a turn it does not own would be writing into
    somebody else's session.
    """

    answer: dict[str, Any] | str
    """Through `ph.seams.approval`'s own encoder — `Edited` and `Responded` are
    frozen dataclasses, and one in a frame unencoded is a `TypeError` inside the
    task group that answers the ask."""
    reason: str = ""


class QuestionAsk(SessionAsk):
    """`question/ask` — the ask-user modal, over the socket."""

    METHOD: ClassVar[str] = "question/ask"
    question: UserQuestion


class QuestionAskReply(WireModel):
    """What a front end answers a `question/ask` with. `None` is a person who
    dismissed the modal rather than one who typed nothing."""

    answer: str | None = None
