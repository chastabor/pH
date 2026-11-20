"""What the daemon *emits*, as models. This is the other half of P8-07 (P8-08).

P8-07 typed what the daemon **receives**: one `WireModel` for the params of each
method, checked at the edge by `parse_params`. What the daemon answers and
announces stayed `dict[str, Any]`. More than 30 sites built those dicts by hand,
and as many more read them by hand. `Peer.send` has a rule: this side must get
right what goes out, and the type checks it. That rule held for the four
envelope shapes and for nothing that they carry.

The cost was real. **`session.status` had three payload shapes and one reader.**
`supervisor.py` published `{sessionId, status}` for a retry or a passivation,
and `{sessionId, status, lastTurn, readings}` when the agent moved.
`session/attach` replied with `describe()` plus the footer, and `follow.py.seed`
deliberately feeds that reply to the same sink. `tui/remote.py._status` absorbed
all three with `str(params.get("provider") or self.state.provider)` and a
comment that called the shapes "honestly different". That comment was a union
type written as prose, and this module writes it as a type. Every field of
`StatusFacts` is optional because a frame states *what changed*. `None` means
"not stated here, keep what you have", and not "cleared". The one exception is
the name of a route, which `StatusFacts` describes.

**The models are here, and not in `daemon/` or in `protocol.py`.** A front end
reads them and must not import `ph_app.daemon.server` (`test_app_layering`), so
they cannot be beside the handlers that build them. `protocol.py` holds the
envelope and the shapes that both transports share (`Cursor`, `NoParams`,
`SessionParams`). The models here are payloads, and `--mode rpc` emits two of
them. In a module of their own beside `protocol.py`, both ends can reach them,
and neither end imports the machinery of the other.

**What stays a `dict[str, Any]`, on purpose.** `SessionEventNotice.event` is the
wire envelope of a session event. `SessionEvent.from_wire` validates it, and a
second declaration of the log envelope here would be the second spelling that
`ph.session.events` exists to prevent. `presentation` is a rendered card, and
its shape belongs to the tool registry. Both are handed on to code that knows
their shape.

@module ph_app.payloads
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal, NamedTuple, TypeAlias

from pydantic import Field

from ph.json import JsonObject, JsonValue
from ph.llm.types import AttachmentRef, ToolSchema
from ph.seams.approval import ApprovalRequest
from ph.seams.commands import CommandSchema
from ph.seams.live_index import Hold as RootHold
from ph.seams.live_index import LiveEntry
from ph.seams.models import ModelEntry, ModelRoute
from ph.seams.permission_presets import PresetSchema
from ph.seams.skills import Skill
from ph.seams.subagents import (
    Access,
    ChildState,
    DowngradeReason,
    StatusCause,
    SubagentStatus,
)
from ph.seams.tui_screens import ScreenSchema
from ph.seams.tui_status import StatusReading
from ph.seams.user_questions import UserQuestion
from ph.session import SessionHeader
from ph.wire import WireModel, wire_alias

from .protocol import CapabilityBlock, Cursor, ProbeName
from .sessions import SessionSummary
from .wire import named, stated

__all__ = [
    "FED",
    "NOTICES",
    "ApprovalAsk",
    "ApprovalAskReply",
    "AskKey",
    "AskSettledNotice",
    "AttachReply",
    "AttachmentStored",
    "ChildRow",
    "CommandShown",
    "CompactionFacts",
    "ConfigRow",
    "CredentialStored",
    "CredentialsHeldReply",
    "DaemonLifetime",
    "DaemonMode",
    "DaemonNotice",
    "DaemonStatusReply",
    "DiagnosticRow",
    "DiagnosticSection",
    "Hold",
    "LeftOnDisk",
    "MutationRepeated",
    "NewSessionReply",
    "OwedListing",
    "OwedRoot",
    "PresetApplied",
    "ProbeReply",
    "ProfileAsk",
    "ProfileAskReply",
    "ProfileDecision",
    "QuestionAsk",
    "QuestionAskReply",
    "RepeatOutcome",
    "ResumeOutcome",
    "ResumeReport",
    "ResumedRoot",
    "RootDescription",
    "RootDetail",
    "RootListing",
    "RootStatusReply",
    "ScheduleCanceled",
    "SessionAsk",
    "SessionBrowse",
    "SessionChildrenNotice",
    "SessionCommandsNotice",
    "SessionDetached",
    "SessionEventNotice",
    "SessionModelsReply",
    "SessionNotice",
    "SessionPresetsReply",
    "SessionReadingsReply",
    "SessionSchedulesReply",
    "SessionScoped",
    "SessionScreensNotice",
    "SessionSkillsReply",
    "SessionStagedNotice",
    "SessionStatusNotice",
    "SessionToolsReply",
    "ShellReply",
    "SnapshotPage",
    "StartingRoute",
    "StatusFacts",
    "notice_of",
]


class _CarriesJson(WireModel):
    """A payload with fields that are **already** wire JSON.

    `event`, `presentation`, `events`, `presentations` and `schedules` are typed
    `dict[str, Any]` or `list[dict[str, Any]]`, because their shape belongs to
    other code: the envelope of a session event, a rendered card, a schedule
    row. Pydantic cannot know that, so `model_dump` walks them and rebuilds
    every nested container on the way out. That is a deep copy of a tree whose
    only destination is `dumps`.

    **This class exists because of a measurement.** A 200-block
    `assistant/message` notice dumped in 21.8 µs, where the dict that it
    replaced cost 0.049 µs. A 2048-event `session/snapshot` page cost 1.3 ms
    for each page, in addition to the encode that must happen anyway.
    `session.event` is the hottest frame that the daemon sends: one for each
    appended event, for each watcher.

    The named fields are therefore excluded from the dump and attached again
    **by reference**. The output is byte-identical, and `test_payloads` checks
    that. The difference is that a tree that is already in wire form is not
    rebuilt to prove it.

    **`dict[str, Any]` is the right type here, and `JsonObject` is not.** The
    last paragraph of `ph.json.JsonValue` gives the reason. The abstract alias
    cannot be a pydantic field at all. `pydantic.JsonValue` brings back, on the
    validate side, the same walk that this class removes from the dump side.
    """

    BY_REFERENCE: ClassVar[frozenset[str]] = frozenset()

    def to_wire(self) -> dict[str, JsonValue]:
        wire = self.model_dump(by_alias=True, exclude_none=True, exclude=set(self.BY_REFERENCE))
        for name in self.BY_REFERENCE:
            value = getattr(self, name)
            if value is not None:
                wire[wire_alias(name)] = value
        return wire


# ------------------------------------------------------------------ status --


class CompactionFacts(WireModel):
    """What a front end is told about automatic compaction on a root."""

    trigger_fraction: float | None = None
    """The share of the context window at which the root compacts on its own, or
    `None` when nothing compacts on pressure. It is the answer of the compaction
    seam of the root (`CompactionSeam.trigger_fraction`), so a front end warns at
    the share that the profile of the root configures, and not at a copy of the
    default."""


class StatusFacts(WireModel):
    """What a frame may say about where a root is. Every field is optional.

    This is the union that `tui/remote.py` absorbed in prose. Three frames reach
    one reader, and each states only what it knows:

    - A `passivated` notice has a status and no footer.
    - An `announce` has the footer too.
    - The attach reply has the route as well, because a front end that has just
      connected has a footer to draw *now*.

    `None` therefore means "this frame does not say", and never "cleared". A
    reader keeps what it had, which is what every reader already did by hand.
    The name of a route is the one exception.

    **The route of a root is one fact.** `provider`, `model` and `model_key`
    describe one route, so a reader takes them together. A frame states the
    route when its `model` is present. The `provider` and the `model_key` of
    that frame then belong to that route. A `model_key` of `None` in such a
    frame is an answer: the profile lists no name for the route. A frame with
    no `model` does not state the route, and a reader keeps all three as it had
    them.

    `status` is a `str` and not a `Literal`. It is `str(agent.status)` plus the
    values that the supervisor derives: `waiting`, `retrying`, `failed`,
    `passivated`, `needs-credential` and `needs-profile-decision`. A closed set
    here would refuse a phase that the driver added.
    """

    status: str | None = None
    last_turn: str | None = None
    readings: list[StatusReading] | None = None
    provider: str | None = None
    model: str | None = None
    """The model of the route in force. Its presence is what says that this frame
    states the route."""
    model_key: str | None = None
    """The name that the session gives the route. It is read only in a frame that
    states the route (`model`). `None` there means that the list of the session
    does not hold the route."""
    compaction: CompactionFacts | None = None
    """When the root compacts on its own. `None` here means that this frame does
    not say. A frame that says "nothing compacts on pressure" carries a
    `CompactionFacts` with no `trigger_fraction`."""


# ----------------------------------------------------------------- replies --


class SessionScoped(WireModel):
    """A frame about one root, in either direction.

    `METHOD` is the name that this payload travels under. It is declared on the
    payload and is not passed beside it. `Root.publish` takes the notice and
    reads the name from it, so nobody can publish a command list as
    `session.status`. The name and the payload used to be two arguments that had
    to agree, at eight call sites, and nothing checked that they did.

    The base exists so that the two directions are **siblings, and neither
    inherits the other**. An ask is not a notification, because it expects a
    reply. While `ApprovalAsk` was a subclass of `SessionNotice`, only two
    things kept it out of the notice table: an author who remembered to omit
    it, and a denylist in the test that said so a second time. A reader that
    found `approval/ask` among the notices would treat a question as an event
    and never answer it. Now `Mapping[str, type[SessionNotice]]` cannot hold
    one.

    `METHOD` stays `""` on the base, where it means "no name". The replies
    below declare none, because a reply travels under the id of its request.
    `test_payloads` reads `METHOD` on every `SessionNotice` subclass to find
    the ones that are announced, so the base has to answer.
    """

    METHOD: ClassVar[str] = ""
    session_id: str


class RootDescription(SessionScoped):
    """`Root.describe()`: what a client is told about one root.

    Each fact has one name. `rootId` and `sessionId` used to be the same string
    (a root *is* its session here), and `events` was `cursor.sequence`. A client
    then had to choose which of two names was authoritative.
    """

    status: str
    last_turn: str | None = None
    watchers: int
    cursor: Cursor
    provider: str | None = None
    model: str | None = None
    """The route that this root is on. Both are `None` for a root that has no
    route."""
    model_key: str | None = None
    """Which of the models of the session this is, or `None` on a profile with no
    `models` row. A route that a person named is listed under a name that is
    made from the route (S4)."""
    profile: str | None = None
    """The named profile that this root runs on, which is the profile in its log
    (S5), or `None` for a profile file."""
    profile_changes: int = 0
    """The number of settings by which that named profile, as it composes now,
    differs from the version that this root started on (S6). It is `0` when the
    profile has not moved. A root runs on its own version until one is adopted,
    and `phern profiles diff` lists the differences."""
    profile_note: str | None = None
    """What a person who attaches is told about how that named profile moved. It
    is the one wording (`ph_app.profiles.kept_note`, which names `/profile`
    commands). It says whose the settings are, or why a start took the new
    version back. It is `None` when the profile has not moved, and while the
    root is held on that same question (`profile/ask`)."""
    compaction: CompactionFacts = Field(default_factory=CompactionFacts)
    """When this root compacts on its own, as its compaction seam answers it."""

    def facts(self) -> StatusFacts:
        """This description as its status half.

        `follow.py` feeds the attach reply to the same sink that `session.status`
        goes to. The docstring of `Followed.seed` says that the two have the
        same shape. This method is that claim as a conversion, and not as a
        sentence.

        The three fields of the route go across as they are. A description with
        a `model` therefore states the route, and its `model_key` of `None`
        says that the route has no listed name (`StatusFacts`).
        """
        return StatusFacts(
            status=self.status,
            last_turn=self.last_turn,
            provider=self.provider,
            model=self.model,
            model_key=self.model_key,
            compaction=self.compaction,
        )


RepeatOutcome: TypeAlias = Literal["settled", "unknown"]
"""What became of the first attempt that a repeat names (P10-10, decision 4)."""


class MutationRepeated(RootDescription):
    """A mutation whose idempotence key had already been claimed.

    One shape serves every verb, so a client branches on one field. The
    docstring of `MUTATIONS` states that rule. This class is the rule as a type,
    and not as a `{**describe(), "repeated": True}` spread at the wrapper.
    """

    repeated: bool = True
    outcome: RepeatOutcome
    """Whether the first attempt finished. The field is **required** and has no
    default. The daemon and the front ends ship together (`PROTOCOL_VERSION` 4),
    and a default would be the one field that silently claims `settled` for a
    verb that nobody saw finish.

    - `settled`: the first attempt ran and answered, and this is the same
      request again.
    - `unknown`: the key was claimed and no settle followed. The daemon died
      during the act and repair closed it, or the act raised, or the act is
      still running in this process.

    A client that is told `unknown` should ask the user, and not assume either
    answer."""


class RootDetail(RootDescription):
    """`Root.detail()`: `describe`, plus what a client that asks about *one* root
    needs. `status` reduces the retry ladder to `retrying` or `failed`, and these
    are the ladder fields behind that answer."""

    attempts: int
    failed: bool


class _CarriesSchedules(_CarriesJson):
    """The `schedules` field, its by-reference rule and the reason, stated one time.

    Two replies carry what is still going to fire on a root: `session/status`
    with the rest of the detail, and `schedule/list` alone. Both feed the same
    consumer (`agents._print_schedules`). When the field was declared twice, the
    two declarations had the same field, the same `BY_REFERENCE` and the same
    paragraph, and the prose of the second copy pointed at the first.

    The rows are the hand-built rows of `ph.seams.schedule.state_to_wire`. A
    model for them is a ph-core change. Until then they pass through as the JSON
    that they already are, which is the purpose of `BY_REFERENCE`.
    """

    BY_REFERENCE: ClassVar[frozenset[str]] = frozenset({"schedules"})
    schedules: list[dict[str, Any]] = Field(default_factory=list)


class RootStatusReply(RootDetail, _CarriesSchedules):
    """`session/status`: one root in detail, with what is still going to fire."""


class NewSessionReply(RootDescription):
    """`session/new`: the description, with the header of the root.

    A front end builds its mirror of the log of the root on this header
    (`DaemonSession.header`). The header is here and not on every description.
    The other replies that carry a description are about status, and a client
    builds its mirror one time, from this reply.
    """

    header: SessionHeader


class AttachReply(RootDescription):
    """`session/attach`: the description, with the footer beside it.

    The footer comes with the status that it belongs to. A client that has just
    attached then draws a complete footer without a second call.
    """

    readings: list[StatusReading] = Field(default_factory=list)

    def facts(self) -> StatusFacts:
        """The status half of the description, plus the footer that this reply carries."""
        return super().facts().model_copy(update={"readings": self.readings})


class SnapshotPage(SessionScoped, _CarriesJson):
    """`session/snapshot`: one bounded page of history, and the next cursor.

    A page is 2048 events, so the rebuild that `_CarriesJson` avoids is the
    largest single rebuild in the daemon.
    """

    BY_REFERENCE: ClassVar[frozenset[str]] = frozenset({"events", "presentations"})
    events: list[dict[str, Any]] = Field(default_factory=list)
    """The wire envelopes of session events. `SessionEvent.from_wire` validates
    them where they are folded. The module docstring says why they are not
    declared again here."""
    presentations: dict[str, Any] = Field(default_factory=dict)
    """Rendered cards, keyed by `seq` as a string. The map is **sparse**: a page
    is 2048 events, and a turn contributes only a few cards."""
    started_at: int = Field(validation_alias="from", serialization_alias="from")
    """Where the read actually began, which is not always where the cursor asked.
    `resume_at` answers a cursor from another log with 0. It answers a cursor
    from an earlier incarnation of this log with no more than the store kept
    (S17).

    The field is required. Zero is a real index, the start of the log, so a
    default of zero would be a claim that nobody made.

    This is the one field in this module whose wire name is not the answer of
    `wire_alias`. `from` is a Python keyword, so the attribute cannot have that
    name. The field uses the two directional aliases and not the single
    `alias=`. Without the pydantic mypy plugin, a bare `alias` becomes the
    *parameter* name in the synthesized `__init__`. `SnapshotPage(from=...)` is
    a syntax error, so nobody could call the constructor by name."""
    cursor: Cursor
    more: bool = False


class RootListing(WireModel):
    """`sessions/list`: the roots that this daemon holds."""

    sessions: list[RootDescription] = Field(default_factory=list)


class SessionBrowse(WireModel):
    """`sessions/browse`: every session that a person could open, stored and live.

    This is a separate model from `RootListing`, although the field name is the
    same. The rows are `SessionSummary` and not `RootDescription`, and the two
    have disjoint fields. One model over both could type the rows only as
    `dict`, and that is what it did. The name had no checking behind it, and
    each producer needed a `.to_wire()` loop to feed it. Nested models dump
    recursively, so naming the row type deletes those loops.
    """

    sessions: list[SessionSummary] = Field(default_factory=list)


class DiagnosticRow(WireModel):
    """One `(label, value)` of a diagnostic section.

    The row has two named keys and is not a two-element array. This is the rule
    that `RootDescription` states about one name for each fact. An array of
    pairs on the wire has positional meaning, and a reader gets positional
    meaning wrong."""

    label: str
    value: str


class DiagnosticSection(WireModel):
    """One titled block of `DiagnosticsRegistry.report()`, on the wire."""

    title: str
    rows: list[DiagnosticRow] = Field(default_factory=list)


class StartingRoute(WireModel):
    """The route that a new root starts on, under the name that the profile lists
    it by.

    This is `ModelEntry` with one difference. The seam spells a route that the
    profile does not list as a `key` of `""`. Here that route has no `key`.
    `of` converts it with `stated`, which every such conversion goes through.
    """

    key: str | None = None
    """The name that the profile lists the route by, or `None` for a route that
    the daemon was started on and the profile does not list."""
    route: ModelRoute
    default: bool = False
    """Whether this is the default of the profile."""

    @classmethod
    def of(cls, entry: ModelEntry) -> StartingRoute:
        """The entry of the seam as the wire carries it."""
        return cls(key=stated(entry.key), route=entry.route, default=entry.default)


class DaemonStatusReply(CapabilityBlock):
    """`daemon/status`: what the running process is, for `phern agents doctor`.

    **This reverses issue 67(a), which left the reply unmodeled on purpose.**
    The grounds were that the reply had one producer and one reader, and that it
    carried the nested shape of `report()` verbatim, so a model would be a
    second declaration of that shape. Two things changed.

    - The reply is now named by a `Verb`. A verb whose reply is `dict[str, Any]`
      is the `Verb[P, Any]` escape, and issue 74 said that this escape would be
      the thing to delete afterwards. The verb table therefore makes the model
      worth its cost.
    - The "second declaration" was not a second one. `report()` returns
      `list[tuple[str, list[tuple[str, str]]]]`, a *tuple* shape that cannot
      travel. `DiagnosticSection` is therefore the first declaration of the wire
      form, and the hand-written `as_obj`/`as_seq` decoder in `agents.py` was
      the second. Naming the wire form here deletes that decoder.
    """

    pid: int
    socket: str
    uptime_ms: int
    roots: int
    starts_on: StartingRoute | None = None
    """What a new root runs on: the `--provider/--model` of the daemon, or else
    the default of its profile. It is `None` when neither names a route. Each
    root may move off it with `/model`."""
    passivate_after: float | None = None
    """Seconds, or `None` for a daemon that never passivates. The field is
    optional because the absence is a *setting* and not a missing field.
    `exclude_none` drops it, so the reader defaults it back to `None` and prints
    "off"."""
    next_wake: int | None = None
    """When the scheduler next wakes, in epoch ms, or `None` while it has nothing
    planned. The scheduler sleeps until then and does not tick."""
    next_release: int | None = None
    """When the next quiet root may be released, in epoch ms, or `None` while
    none can be. Release sleeps until then and does not sweep (P12-01)."""
    socket_watch: str
    """How the path of the socket is watched (P5-11, P12-05): `inotify`, `kqueue`,
    `off`, or `unavailable: <why>`. In the last case the daemon checks the path
    whenever it is asked. This field used to be `watchEvery`, a thirty-second
    `lstat`."""
    check_invariants: bool
    """Whether the pollable invariants of a root are checked when it settles (I6,
    P12-04). They used to be checked on the five-minute cadence that
    `invariantsEvery` named."""
    unreachable_since: int | None = None
    """Set only after the socket stopped being the socket of this daemon (P5-11).
    Absent is the normal answer, so the row that it feeds is absent too."""
    unwritten: list[str] = Field(default_factory=list)
    """Roots whose log could not be written when this daemon last tried (P13-08):
    the record of the ladder, the flush of a passivation, the ack of a mutation.
    Each root stays listed until a later write of that root succeeds. Readiness
    is false while any root is listed."""
    probe: str | None = None
    """Where the probes answer over TCP, `host:port`, or `None` when
    `--probe-bind` was not given (P13-12)."""
    sections: list[DiagnosticSection] = Field(default_factory=list)


type ConfigRow = JsonObject
"""One row of a composed profile, in the shape of `Row.to_dump()`.

The daemon reads these rows to find the credentials that a profile names
(`ph_app.daemon.projections.credentials_named`). The rows do not cross the
socket. The `daemon/config` verb used to send the whole composed profile to
every front end, and it was removed when nothing asked for it any more.

It is `JsonObject` and not a hand-written `Mapping[str, JsonValue]`, which was
the third spelling of that in the tree."""


class CredentialsHeldReply(SessionScoped):
    """`credentials/held`: every credential that the profile of the root names,
    and whether each is held.

    **The names are the answer of the daemon, not the question of the client.**
    They used to be the question. A front end then had to know them, so the
    whole composed profile was sent to every client that might open a login
    picker. The daemon now reads them from the profile that the root is mounted
    on (`credentials_of` in `daemon/projections.py`), and adds any credential
    that this process was given and that no row names.

    The reply carries booleans and never values. A client asks the daemon, and
    does not read a store locally, so that the value never crosses the socket.
    Now the configuration does not cross it either."""

    held: dict[str, bool] = Field(default_factory=dict)


class CredentialStored(SessionScoped):
    """`credentials/store`: the name and the fact, never the value.

    `_Connection._credentials_store` in `daemon/server.py` states the rule that
    this shape enforces: the secret is used and not kept, never logged, and
    never echoed here."""

    name: str
    stored: bool = True


class SessionDetached(SessionScoped):
    """`session/detach`: whether this client had been attached.

    It is not an error when the client was not attached. A client detaches while
    it tidies up, often twice, and nobody can write a teardown path correctly if
    that path raises. The boolean is the correct answer to "did that do
    anything"."""

    detached: bool = False


class AttachmentStored(SessionScoped):
    """`attachment/put`: the reference that the client came for.

    The store is content-addressed, so a second put of the same bytes stores
    them one time, and is a cheap way to *learn* the reference. That is why this
    is not a `MUTATIONS` row: a `repeated` envelope would withhold the one field
    that the client needs."""

    attachment: AttachmentRef


class CommandShown(SessionScoped):
    """`session/command`: the text that a command wants shown to the user.

    `shown` is `None` for a command that shows nothing, which is what `dispatch`
    itself returns. The *record* of the command is in the session log, where
    every other attached UI reads it. This reply is only what the caller who
    typed the command should see."""

    shown: str | None = None


class ShellReply(SessionScoped):
    """`session/shell`: the exit code, and nothing else.

    The output travels as `shell/command` and `shell/result` **events**, so
    every attached UI reads it by the one route that they all watch already.

    **There is no `ok`, and that is a correction.** It used to be sent as a
    second wire key, on the grounds that "a front end branching on it should not
    have to know that zero means success". No front end branches on it. The only
    production caller discards this reply, and `tui/adapter.py` reads `ok` from
    the `shell/result` *event*, not from here. The key was therefore a second
    spelling of one fact with no consumer, and `ShellReply(exit_code=1, ok=True)`
    validated. That broke a rule that `daemon/server.py` states: each fact has
    one encoding, so nothing on the wire can disagree with itself. `ok` then
    became a property that nothing read, and the property is removed too.
    """

    exit_code: int


class PresetApplied(SessionScoped):
    """`session/preset`: the preset that is now in force, by name."""

    preset: str


class ScheduleCanceled(SessionScoped):
    """`schedule/cancel`: whether that schedule was still there to cancel."""

    schedule_id: str
    canceled: bool = False


class SessionSchedulesReply(SessionScoped, _CarriesSchedules):
    """`schedule/list`: what is still going to fire on this root, and nothing else."""


# ----------------------------------------------------------- notifications --


class SessionNotice(SessionScoped):
    """A frame that the daemon *announces*. It has no reply and no id, and a
    client may watch it or ignore it."""


class AskKey(NamedTuple):
    """An ask as the daemon names it. Its `ask_id` is unique within its root only.

    The desk of each root counts its own asks and coordinates with no other
    root, so a front end that is attached to two roots can be asked two
    `ask-1`s. The pair tells them apart. Both frames that name an ask give the
    pair as `key`, so a front end files an ask by what the wire says and does
    not rebuild the pair.
    """

    session_id: str
    ask_id: str


class SessionAsk(SessionScoped):
    """A frame that the daemon *asks*. It expects a typed answer back.

    `ask_id` is here and not on each ask. It used to be declared twice, and
    `AskDesk` keys its pending table on it in both cases. The desk creates the
    id, which is unique within its root only. A front end therefore files its
    modals by `key`, which is the root and the id together. The id is not the
    name that the log gives the ask.
    """

    ask_id: str

    @property
    def key(self) -> AskKey:
        return AskKey(self.session_id, self.ask_id)


class SessionEventNotice(SessionNotice, _CarriesJson):
    """`session.event`: one appended event, with its card when it has one.

    This is the hottest frame that the daemon sends: one for each appended
    event, for each watcher, for every streamed chunk. Because of
    `_CarriesJson`, both payload fields leave by reference and are not rebuilt.
    """

    METHOD: ClassVar[str] = "session.event"
    BY_REFERENCE: ClassVar[frozenset[str]] = frozenset({"event", "presentation"})
    event: dict[str, Any]
    presentation: dict[str, Any] | None = None


class SessionStatusNotice(SessionNotice, StatusFacts):
    """`session.status`: where the root is now, as much of it as changed.

    It has both bases on purpose. It is a `StatusFacts`, which is what the
    reader wants, and it also names its root, which is what the dispatcher
    wants.
    """

    METHOD: ClassVar[str] = "session.status"


class SessionCommandsNotice(SessionNotice):
    """`session.commands`: the whole palette, not a delta."""

    METHOD: ClassVar[str] = "session.commands"
    commands: list[CommandSchema] = Field(default_factory=list)


class SessionScreensNotice(SessionNotice):
    """`session.screens`: the whole screen list, for the reason that `commands`
    is whole. `ScreenDefinition.build` cannot travel, so a client keeps only the
    ids that it can draw, and reads the projection again instead of applying a
    delta."""

    METHOD: ClassVar[str] = "session.screens"
    screens: list[ScreenSchema] = Field(default_factory=list)


class ChildRow(WireModel):
    """One sub-agent beneath a root, as its own log tells it (P11-07).

    **The row is built from the log of the child and from nothing else**
    (`ChildState`, Phase 11). The log of a root no longer holds any record of
    its children, so a front end cannot fold them from the stream that it
    watches. The daemon folds the log of each child and sends this row. For the
    same reason, the panel draws the same states that the resume sweep, the
    passivation check and the budgets read.

    **The row narrows `ChildState.to_wire()` on purpose.** That is the row that
    a model reads from `list_subagents`. It carries the task prompt, the paths,
    the tools and the counts of the retry ladder. A panel that is pushed to
    every watcher on every change has no use for any of those, and a prompt is
    the one field that can be large. This row holds what a person who watches
    the fan-out reads: who, on what, allowed what, where it stands and why, what
    it cost, and whether it was revoked.

    The closed sets are the `Literal`s of the seam. A status or a cause that
    this build does not know is therefore refused at the edge, and is not drawn
    as a blank glyph. The daemon and the front ends ship together
    (`PROTOCOL_VERSION` 6).
    """

    run_id: str
    session_id: str
    parent_id: str
    """The agent that spawned it: the root for a child, a child for a grandchild.
    A list of a whole family is a tree, and this field is the edge that draws
    the tree."""
    name: str | None = None
    """The name that the spawn gave the child, or `None` when it gave none. A
    panel then draws the `run_id`."""
    model: str | None = None
    """The model that the child was admitted on, or `None` when its admission
    records none."""
    granted_access: Access = "read"
    downgrade_reason: DowngradeReason | None = None
    status: SubagentStatus = "queued"
    """`queued` until its first status: admitted, and not yet started."""
    cause: StatusCause | None = None
    detail: str | None = None
    deleted: bool = False
    """Revoked. A tombstone, not a removal: the row stays, and so does `status`."""
    deleted_reason: str | None = None
    awaiting: str | None = None
    """The credential it is held for, by name (T5), or `None`."""
    tokens: int = 0
    """What it spent on its answers and its compactions, counted by
    `TokenUsage.total`. This is the count that `/autonomous` charges to the
    `children` of a goal."""

    @classmethod
    def of(cls, state: ChildState) -> ChildRow:
        """The state of one child as the row that a front end draws. The fields are
        read from the state by the names that the two share, so a field that is
        added to both needs no third line here.

        The admission spells "no name" and "no model" as `""`. This is the one
        place that turns each into `None` for the wire."""
        row = cls.model_validate(state, from_attributes=True)
        return row.model_copy(update={"name": named(row.name), "model": named(row.model)})


class SessionChildrenNotice(SessionNotice):
    """`session.children`: every sub-agent beneath the root, whole (P11-07).

    The daemon pushes it when the log of a descendant moves in a way that its
    row shows. `session/children` answers with it for a client that has just
    attached. Those are the two routes that `session.screens` has, and the list
    is whole for the same reason: each frame is a snapshot, so it is correct in
    any order of arrival, and a client keeps no deltas.

    **A parent comes before its own children, and siblings are in admission
    order.** A reader therefore draws the tree in one pass. The list holds every
    level of the tree, read as the log of the root opens.
    """

    METHOD: ClassVar[str] = "session.children"
    children: list[ChildRow] = Field(default_factory=list)


class SessionReadingsReply(SessionNotice):
    """`session/readings`: the footer, asked for and not pushed.

    This is a reply and not a notice, so it declares no `METHOD`. The footer
    *is* pushed, but it travels in `session.status` beside the status that it
    belongs to, and not under a name of its own.
    """

    readings: list[StatusReading] = Field(default_factory=list)


class SessionToolsReply(SessionNotice):
    """`tools/list`: what the model may call in this deployment."""

    tools: list[ToolSchema] = Field(default_factory=list)


class SessionPresetsReply(SessionNotice):
    """`presets/list`: the postures, and which one is in force."""

    presets: list[PresetSchema] = Field(default_factory=list)


class SessionModelsReply(SessionNotice):
    """`models/list`: the models that the profile of this root lists, the default first."""

    models: list[ModelEntry] = Field(default_factory=list)


class SessionSkillsReply(SessionNotice):
    """`skills/list`: what is installed here, as the catalog states it.

    The reply carries `Skill` itself and not a projection of it. The model of
    the seam is already a `WireModel` that carries what a catalog entry is: the
    name, the description, where the skill came from, which revision, and the
    other fields of the entry. A second shape here would be a copy to keep in
    step, with no field gained. `body` is not on it. That is the purpose of
    progressive disclosure, and the reason why this listing is cheap.
    """

    skills: list[Skill] = Field(default_factory=list)


class SessionStagedNotice(SessionNotice):
    """`session.staged`: the tray of the composer, which is shared. If only the
    uploader could see a chip, nobody else could reason about the composer."""

    METHOD: ClassVar[str] = "session.staged"
    staged: list[AttachmentRef] = Field(default_factory=list)


class AskSettledNotice(SessionNotice):
    """`ask.settled`: somebody else answered, so take the modal down."""

    METHOD: ClassVar[str] = "ask.settled"
    ask_id: str

    @property
    def key(self) -> AskKey:
        """The ask it settles, as the frame that asked it named it."""
        return AskKey(self.session_id, self.ask_id)


NOTICES: Mapping[str, type[SessionNotice]] = {
    one.METHOD: one
    for one in (
        SessionEventNotice,
        SessionStatusNotice,
        SessionCommandsNotice,
        SessionScreensNotice,
        SessionChildrenNotice,
        SessionStagedNotice,
        AskSettledNotice,
    )
}
"""Wire method → the payload that travels under it, for a client that reads frames.

This is the client half of the `METHODS` of the daemon. The daemon has a table
that says what each name *takes*, and this table says what each name *carries*.
Both dispatchers used to write the pairing branch by branch, as
`if method == X.METHOD: X.model_validate(...)`. That is the mechanism, written
one time for each reader, and nothing checked that a notice that the daemon
publishes has a reader that knows its model. `test_payloads` compares this table
with every `SessionNotice` subclass that declares a `METHOD`, so nobody can add
an eighth notice without adding it here.

The asks cannot appear here, because `SessionAsk` is a sibling of
`SessionNotice` and not a subclass. That is the purpose of the split: a reader
that found `approval/ask` among the notices would treat a question as an event
and never answer it.
"""


FED: frozenset[str] = frozenset({SessionEventNotice.METHOD, SessionStatusNotice.METHOD})
"""The two notices that a *feed* reads: the events of the transcript, and where
the root is.

The set is named one time because two readers gate on it, and both spelled it
inline. There is also a cost reason. A class-attribute load from a pydantic
model never specializes (`ModelMetaclass` defines `__getattr__`), so the tuple
that those branches built for each notification measured 0.105 µs, against 0.021
for this set. The other five notices are palette, family and tray snapshots,
and a feed has no use for them. Validating them to find that out measured
**129x** the string test that replaced it, on a `session.commands` that carried
a 25-command palette."""


def notice_of(method: str, params: dict[str, Any]) -> SessionNotice | None:
    """One inbound frame as the payload that its method declares, or `None`.

    The result is `None` for a method that this build has no model for. That is
    a daemon newer than this client, and a reader drops the frame and does not
    crash on it. The branch-by-branch form got this right by accident, and this
    function states it: an unknown notification is not an error. It is a feature
    that the client does not have.

    A payload that does *not* parse still raises. That is deliberate, now that
    the reader survives it. `Peer._read` guards the notification body, so a bad
    frame costs one frame and is reported in the log, and is not swallowed
    here. A dropped `session.event` is a hole in a transcript, which is worth a
    log line. `ph_app.wire.view_of` swallows the error for the opposite reason:
    a card that does not parse costs one card, and the generic card that is
    drawn in its place is plain and not wrong.
    """
    model = NOTICES.get(method)
    return None if model is None else model.model_validate(params)


# -------------------------------------------------- the daemon's own frames --


class DaemonNotice(WireModel):
    """A frame about the *process*, not about a root.

    These frames are a family of their own. `SessionScoped` carries a
    `sessionId`, and every reader gates on it before parsing. That is correct
    for a client that watches one root among several, and wrong for a fact about
    the daemon that they all share. A lifetime notice with a `sessionId` would
    have to invent one, and a reader would drop the notice because it belonged
    to somebody else.

    `METHOD` is declared on the payload for the reason that `SessionScoped`
    states: the name and the model were two values that had to agree.

    **This is a type, not a table.** `NOTICES` exists because two readers each
    spelled `if method == X.METHOD: X.model_validate(...)` for six names. One
    reader and one name do not justify a second mapping, and the only consumer
    of that mapping narrowed straight back to the concrete type that it had just
    looked up. The base does justify the exclusion.
    `Mapping[str, type[SessionNotice]]` structurally cannot hold one of these
    frames, so the oversight of an author can never dispatch a process-level
    frame behind the `sessionId` filter. `SessionAsk` makes the same argument
    for being a sibling and not a subclass. A second member is the moment to
    write the table.

    The base declares `METHOD` and gives it no value. Every frame of this
    family travels under a name, so each subclass must state one, and nothing
    reads the name from the base.
    """

    METHOD: ClassVar[str]


DaemonMode: TypeAlias = Literal["service", "ephemeral"]
"""Who decided the life of this daemon: a person who typed `phern daemon`, or a
UI that spawned a daemon because the socket was absent (P7-08).

This is a `Literal` and not a `str`, for the reason of `TrustAnswer`: it is a
closed set that the *daemon* writes, so the type stops a typo before it reaches
a client that would render it verbatim. The cost is that a front end older than
a third mode refuses the frame and does not ignore it. That is the skew that
`PROTOCOL_VERSION` documents and that `phern agents shutdown` fixes."""

Hold: TypeAlias = Literal["client", "task", "schedule", "keep-alive"]
"""One reason why a daemon is still running. This is the vocabulary of
`DaemonServer.holds`.

The set is closed, and typed as closed, because both ends use it. The server
appends these names, and the sidebar renders them into a sentence that a person
acts on. A fifth reason that is spelled at one end and not at the other is a row
that reads `held ·` and nothing else. With a `str`, that would be a surprise at
run time in a terminal, and not an error where it was written."""


class DaemonLifetime(DaemonNotice):
    """`daemon/lifetime` and `daemon.lifetime`: why this daemon is still here.

    One model goes through two routes, like `CapabilityBlock`. A client reads it
    one time at attach, because a client that connected in the middle of a turn
    would otherwise draw nothing until something moved. The client then receives
    it again whenever the answer changes.

    **There is no `exits_at`.** The plan had one, for a countdown in the
    sidebar, and it cannot be correct. The keep-alive is armed when the last
    connection closes, and the first frame of the next connection clears it. A
    client is therefore never connected while a deadline is running. A field
    whose only value on the wire is `null` is a setting that is connected to
    nothing. A countdown that is drawn from a deadline that the arrival of this
    client has already canceled is worse than no countdown. `keep_alive_ms` is
    what a connected client can say truthfully: not "it leaves at 9:41" but "it
    leaves five minutes after you do".
    """

    METHOD: ClassVar[str] = "daemon.lifetime"
    mode: DaemonMode
    """`"ephemeral"` is a daemon that a front end spawned, which leaves when
    nothing needs it. `"service"` is a daemon that somebody chose to run, which
    stays."""

    holds: list[Hold] = Field(default_factory=list)
    """Why it is still up, in the order of `DaemonServer.holds`. An empty list on
    a service daemon is the ordinary answer and not a contradiction, because
    `mode` is what says that it stays anyway."""

    clients: int = 0
    """How many front ends are connected now, this one included.

    The count is on the wire because the reader needs it and cannot derive it.
    `holds` says `client` whenever *anybody* is connected. A sidebar that
    subtracted its own connection would promise `exits on detach` to two
    terminals at the same time, and be wrong for whichever closed first. A count
    is the smallest fact that makes the subtraction correct. It costs the server
    nothing, because `holds` already reads the same set."""

    keep_alive_ms: int = 0
    """How long it stays after the last client leaves, or `0` for "at once".

    This is what the user typed as `--keep-alive 5m`, in the unit that the wire
    uses everywhere else. A service daemon also reports zero, and there it means
    nothing at all. `mode` is the field that carries that fact."""


# -------------------------------------------------------------------- asks --
# The daemon → client direction. This is the only place where the daemon builds
# *request* params. They are modeled here and not in `ph_app.params`, because
# the client validates them.


class ApprovalAsk(SessionAsk):
    """`approval/ask`: put this request to the user, and answer with what the user said."""

    METHOD: ClassVar[str] = "approval/ask"
    request: ApprovalRequest


class ApprovalAskReply(WireModel):
    """What a front end answers an `approval/ask` with.

    `reason` travels in the reply, and the client does not steer with it. The
    daemon holds the agent, and a client that steered a turn that it does not
    own would write into the session of somebody else.
    """

    answer: dict[str, Any] | str
    """Encoded by the encoder of `ph.seams.approval`. `Edited` and `Responded` are
    frozen dataclasses, and an unencoded one in a frame is a `TypeError` inside
    the task group that answers the ask."""
    reason: str = ""


class QuestionAsk(SessionAsk):
    """`question/ask`: the ask-user modal, over the socket."""

    METHOD: ClassVar[str] = "question/ask"
    question: UserQuestion


ProfileDecision: TypeAlias = Literal["adopt", "keep", "later"]
"""What a person says about a named profile that moved since the session started
(S6):

- `adopt`: start on the new version.
- `keep`: keep the version of this session, and do not ask about that version
  again.
- `later`: keep it for now, and ask at the next start."""


class ProfileAsk(SessionAsk):
    """`profile/ask`: the named profile of the session changed since it started (S6).

    `listing` is the change as the user reads it (`ph.session_profile.listing`):
    each setting with its old and new value and whose it is, then the overrides
    of the session that still apply over a changed row. The daemon renders it,
    so every front end shows the same account of the change."""

    METHOD: ClassVar[str] = "profile/ask"
    name: str
    listing: str


class ProfileAskReply(WireModel):
    """What a front end answers a `profile/ask` with."""

    decision: ProfileDecision = "later"


class QuestionAskReply(WireModel):
    """What a front end answers a `question/ask` with. `None` means that the
    user dismissed the modal, and not that the user typed nothing."""

    answer: str | None = None


LeftOnDisk: TypeAlias = Literal["spent", "damaged"]
"""Why a resume pass leaves an owed root unmounted (`OwedRoot.left_by_a_pass`)."""


class OwedRoot(WireModel):
    """One root the live-roots index says still owes work (P13-02)."""

    session_id: str
    holds: list[RootHold] = Field(default_factory=list)
    updated: int | None = None
    """When the host last said so, epoch ms, or `None` when the index entry
    carries no time."""
    mounted: bool
    """Whether this daemon holds it now."""
    status: str
    """The root's own status when mounted (`Root.status`), else `stored`."""
    cwd: str | None = None
    """The working directory that the header of the root recorded, or `None` when
    it recorded none."""
    profile: str | None = None
    """The named profile that the root ran on, or `None` when the host did not
    say. A rebuild of the index from a log does not say."""
    failed: bool = False
    """Its ladder was spent: it owes work and a resumer must not loop on it."""
    held: bool = False
    """Mounted and waiting for a person: for a credential, or for a decision about
    its profile (`Root.held`). Nothing that this daemon can do moves it. This is
    a fact that the daemon states, and not a reading of `status`, because the
    order of precedence of `status` can put another word first."""
    damaged: list[str] = Field(default_factory=list)
    """The logs beneath it that cannot be read, by session id (`LiveEntry.damaged`,
    P13-09). What they owe is unknown, and `phern doctor sessions --repair` is
    the fix."""

    def left_by_a_pass(self, *, force: bool = False, named: bool = False) -> LeftOnDisk | None:
        """What a resume pass leaves this root as, when the pass does not mount it
        (P13-03).

        - `spent`: its ladder is spent, and only `force` mounts it. Mounting it
          rings nothing, and a resumer that runs every minute would mount it
          every minute.
        - `damaged`: it is listed for a log beneath it that cannot be read, and
          it holds nothing that a mount would resume. A repair moves it and a
          mount does not. The exception is a root that a person `named`, which
          means that this person wants it mounted.

        The pass of the daemon (`Supervisor.resume_owed`) and the verdict of the
        service client (`Observed.of`) both ask this. What the client expects a
        pass to mount is therefore what the pass mounts.
        """
        if self.failed and not force:
            return "spent"
        if self.damaged and not self.holds and not named:
            return "damaged"
        return None

    @classmethod
    def of(cls, entry: LiveEntry, *, mounted: bool, status: str, held: bool = False) -> OwedRoot:
        """The row for one index entry, with what only the daemon knows beside it.

        This is the one place where the fields of the entry are carried onto the
        wire. A field that is added to `LiveEntry` is added here one time, and
        not at every site that lists an entry. `family` stays off the wire. It
        is where the log is filed, and a client never opens that.

        The entry spells "none" as `0` for its time and as `""` for its two
        texts. This is also the one place that turns each into `None`.
        """
        return cls(
            session_id=entry.session_id,
            holds=list(entry.holds),
            updated=entry.updated or None,
            cwd=stated(entry.cwd),
            profile=stated(entry.profile),
            failed=entry.failed,
            damaged=list(entry.damaged),
            mounted=mounted,
            status=status,
            held=held,
        )


class OwedListing(WireModel):
    """`sessions/owed`: every root that owes work, according to the index."""

    sessions: list[OwedRoot] = Field(default_factory=list)
    trusted: bool
    """Whether the index could vouch for itself when read (`LiveSurvey.trusted`)."""
    rebuilt: bool
    """Whether this reply rebuilt the index from the logs first."""


ResumeOutcome: TypeAlias = Literal["resumed", "mounted", "held", "spent", "damaged", "refused"]
"""What `sessions/resume` did with one root (P13-03).

- `resumed`: this pass mounted it.
- `mounted`: this daemon already held it.
- `held`: it is mounted, and it waits for a credential or a profile decision
  before anything runs.
- `spent`: its ladder is spent and `force` was not passed, so it was left on
  disk.
- `damaged`: it holds nothing that a mount would resume, and it is listed for a
  log beneath it that cannot be read. It was left on disk for
  `phern doctor sessions --repair` (P13-09).
- `refused`: the mount raised, and `detail` says why. Examples are a root that
  another process holds, a session that is not a root, and a log that cannot be
  read."""


class ResumedRoot(WireModel):
    """One root a resume pass was asked about."""

    session_id: str
    outcome: ResumeOutcome
    status: str | None = None
    """`Root.status` when the root is mounted, else `None`."""
    detail: str | None = None
    """Why, for `held`, `spent`, `damaged` and `refused`. `None` for the other
    outcomes."""


class ResumeReport(WireModel):
    """`sessions/resume`: what became of each root that was asked for."""

    sessions: list[ResumedRoot] = Field(default_factory=list)
    trusted: bool
    """Whether the index that the pass read could vouch for itself
    (`OwedListing.trusted`). An `all_owed` pass over an unvouched index may have
    missed a root."""


class ProbeReply(WireModel):
    """`daemon/probe`: one of the three answers, with the reasons when it is no."""

    name: ProbeName
    ok: bool
    reasons: list[str] = Field(default_factory=list)
    """Why not, one sentence each. Empty when `ok`."""
    checked_at: int
