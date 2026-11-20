"""The wire vocabulary that both transports speak (P5-02, I-7).

pH answers on two transports. `--mode rpc` over stdio is for a caller that owns
the process. `$PH_RUNTIME/daemon.sock` is for a caller that does not. **They are
the same protocol**, and this module is what makes that a fact that can be
checked, and not only an intention.

**The names are the names of dsh**, deliberately: `initialize`,
`session/prompt`, `session.event`, `session.status`. dsh already ships a Python
client for this shape. A second vocabulary would make "use the client you have"
false in exactly the deployment that Phase 5 exists for.

**The envelope is here, and the methods are in each server.** The parts that are
truly independent of the transport are the shaping of a request, a reply and an
error, and the version. What a transport serves is its own: one session over a
pipe, or many supervised roots over a socket. So this module owns `respond`,
`notification` and the capability block, and each server owns its method table.
The request params that both servers check against are in `ph_app.params`, and
what the daemon emits is in `ph_app.payloads`.

**There are two kinds of frame, and they are typed in two ways (P8-07).** A
frame that this side *builds* is one of the `TypedDict`s below. `notification`,
`request` and `respond` return them, and the outbox carries them. A builder that
forgot `jsonrpc` or misspelled `params` is a type error, and not a parse error at
the peer. A frame that the peer *sent* is a `dict[str, Any]` and stays one. It
is a claim, and it is read with `.get`. To give it one of these names would
assert what has not been checked. `ph_rlm.kernel.codec` draws the same line for
the fd-3 channel (C10). What *is* checked about an inbound request is its
`params`, and that check is `parse_params`: one `WireModel` for each method,
`extra="forbid"`, and a refusal with a named reason.

@module ph_app.protocol
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, NotRequired, TypeAlias, TypedDict, get_args

from pydantic import AfterValidator, Field, StringConstraints, ValidationError

from ph.json import as_str
from ph.persistence import resumption_of, resumptions
from ph.session import Session, valid_session_id
from ph.wire import WireModel, validation_errors

from .wire import named

__all__ = [
    "PROBES",
    "PROTOCOL_VERSION",
    "SNAPSHOT_EVENTS",
    "CapabilityBlock",
    "Cursor",
    "DaemonError",
    "DaemonGone",
    "Dispatch",
    "ErrorBody",
    "ErrorFrame",
    "Frame",
    "InvalidParams",
    "MethodResult",
    "NoParams",
    "NotificationFrame",
    "Notify",
    "ProbeName",
    "Refusal",
    "ReplyFrame",
    "RequestFrame",
    "ResultFrame",
    "SeamAbsent",
    "SessionId",
    "SessionParams",
    "Stated",
    "UnknownMethod",
    "Verb",
    "capabilities",
    "cursor_of",
    "cursor_text",
    "is_number",
    "notification",
    "parse_cursor",
    "parse_params",
    "request",
    "respond",
    "result_of",
    "resume_at",
    "served",
]

ProbeName: TypeAlias = Literal["startup", "ready", "live"]
"""The three questions that an orchestrator asks of a daemon (P13-12), in the
order in which it asks them: has it come up, may it be sent work, is it still
alive. The set is closed. A probe name that a client sends and the daemon does
not answer is therefore `invalid_params`, and not a 404 that somebody reads as
"not ready"."""

PROBES: tuple[ProbeName, ...] = get_args(ProbeName)

PROTOCOL_VERSION = 11
"""One number, in one place.

It was declared twice, one time for each transport. That is how two servers come
to claim the same version for two different vocabularies. Do not confuse it with
`ph_rlm.kernel.protocol.PROTOCOL_VERSION`, which numbers the fd-3 kernel
protocol and moves independently.

**2 (0.2.0): `cancelled` became `canceled` on the wire.** The `schedule/cancel`
reply carries `canceled` where it carried `cancelled`. Every status value that a
client folds (of a job, of a subagent, of an approval) is spelled in the
American way. A 0.1.x client that reads a 0.2.x daemon sees an absent field and
an unknown status, and not an error.

The same number covers the *additions* of 0.2.0: `daemon/lifetime` and the
`daemon.lifetime` notice (P9-07). They are not breaking in that direction, and
they are breaking in the other. An older client never asks for them. A 0.2.x
client that attaches to a daemon still running from before them fails at the
attach read. Every verb added since 0.1.0 already does that, and a restart of
the daemon fixes it.

**3 (0.3.0): a `sessionId` must be usable as a path component** (K9). Every
`session/*` method on both transports takes it through one model. That model
now refuses an id that is not alphanumerics, dots, dashes and underscores
starting with a letter or a digit, because an id becomes a directory name in the
spill store, the archive and the workspace scratch root. A 0.2.x client that
minted any other id is refused where it used to be served. This number was
released beside the move of the fd-3 protocol of the kernel from 1 to 4
(`ph_runtime.protocol.PROTOCOL_VERSION`, its own number), which is why every
package moved to 0.3.0 together.

**4 (0.4.0): a repeated mutation says whether the first attempt finished**
(P10-10). `MutationRepeated` carries a required `outcome`. It is `settled`, or
`unknown` when the key was claimed and no settle followed (a crash mid-act, an
act that raised). A client that is told `unknown` can therefore ask the user and
need not assume. A 0.3.x client that reads a 0.4.x repeat ignores the field. A
0.4.x client that reads a 0.3.x repeat refuses it, because the field is
required, and a restart of the daemon fixes that. This number was released with
format 2 of the session log (P10-15), so each number moves one time.

The same number covers **`credentials/store` taking no idempotence key** (T5).
The verb left `MUTATIONS`, so its params refuse `clientId`/`commandId`. A 0.3.x
client stamps both on every call that it knew as a mutation, and a 0.4.x daemon
refuses it with `invalid_params`. A 0.4.x client sends neither, and a 0.3.x
daemon serves that unguarded. The store was idempotent anyway, because one value
that is stored twice is stored one time. Its key also outlived the value that it
named, because the value lives in daemon memory.

**5 (0.5.0): a root runs on a model that its profile lists** (session profiles,
S1). The route of a root comes from the `models` row of the profile, and no
longer from the `--provider`/`--model` of the daemon alone. The changes:

- `RootDescription` and `StatusFacts` gain `modelKey`, the name that the profile
  gives to the route in force.
- `daemon/status` replaces its `provider`/`model` with `startsOn`, the entry
  that a new root runs on.
- `session/new` takes a `choice` and a `profile`, so the `--model` and
  `--profile` of a terminal belong to its sessions and not to the daemon.
- `RootDescription` gains `profile`, the profile that a root runs on. When the
  root has a log, it is the profile of that log.
- Two verbs are new. `session/model` is a mutation: run this root on a listed
  key or a whole route from its next request. It is answered with the
  description of the root. `models/list` answers the listed models, default
  first.

Then S6:

- A root whose named profile moved since the root started says how far behind it
  is (`profileChanges`) and what a user who attaches is told (`profileNote`).
  The note says why when a start took the new version back because it would not
  mount.
- A root that is held on a decision of the user about it reports
  `needs-profile-decision`.
- The daemon asks for that decision with `profile/ask` (answered `adopt`, `keep`
  or `later`).
- `session/adopt` records a version for the next start of a held root.

A 0.4.x client that reads a 5 daemon refuses the `modelKey` that it does not
know, because every model is `extra="forbid"`. A restart of the daemon on the
same release fixes that.

**6 (0.6.0): each session owns its log** (Phase 11). The stream of a root no
longer carries anything about its children. `subagent/admitted`,
`subagent/status` and `subagent/deleted` are in the own log of each child, and
`subagent/usage-attributed` is gone. A client that folded its panel from the
`session.event` frames of the root therefore sees an empty family. The family
comes from the daemon instead, which reads it from the own logs of the children.
It arrives by two routes with one payload: `session/children` answers it for a
client that has just attached, and a `session.children` notice pushes it whole
whenever a row moves. Each row (`ChildRow`) carries `runId`, `sessionId`,
`parentId`, `name`, `model`, `grantedAccess`, `downgradeReason`, `status`,
`cause`, `detail`, `deleted`, `deletedReason`, `awaiting` and `tokens`. The
`parentId` is the root for a child and a child for a grandchild, because the
list is the whole family, with a parent before its own children. The log of a
sub-agent is refused as a root (P11-08): a `session/attach` or `session/new`
that names one answers `not_a_root` and names the root whose mount writes it.
`sessions/browse` rows carry `origin` (`subagent`), so a picker shows a child
and does not offer it. A 0.5.x client refuses the `origin` that it does not know
and never asks for the family. A restart of the daemon on the same release fixes
that. This number was released with format 3 of the session log, so each number
moves one time.

The same number covers the scheduler that sleeps until something is due and does
not tick (S18). `daemon/status` drops `tickEvery` for `nextWake`, the time when
it next wakes, which is absent while nothing is planned. `session/new` answers
with a `NewSessionReply`: the description of the root plus its `header`, on
which a front end builds its mirror of the log of the root. A 0.5.x client
refuses the `nextWake` and `header` that it does not know. A restart of the
daemon on the same release fixes that.

**7 (0.7.0): the daemon writes nothing because time passed** (Phase 12,
`plans/Event_Driven_Daemon_Plan.md`). `daemon/status` drops `heartbeatEvery`,
and a root with a live schedule no longer appends `schedule/heartbeat` every
five minutes. Whether a daemon is alive is a question that a monitor asks it,
through this socket or the OpenTelemetry sink. It is not a record left in the
log. Logs that already hold heartbeats still open, because each heartbeat was
written `ignorable`. A 0.6.x client refuses this reply because it requires
`heartbeatEvery`. A 0.7 client that reads a 0.6 daemon refuses the
`heartbeatEvery` that it no longer knows. A restart of the daemon on the same
release fixes both. The release of roots and the keep-alive of the daemon itself
changed in the same way (P12-01, P12-02): `daemon/status` drops `sweepEvery` for
`nextRelease`, the time when the next quiet root may be released, which is
absent while none can be. Invariants are checked when a root settles (P12-04):
`invariantsEvery` becomes `checkInvariants`, a flag. The kernel watches the
socket (P12-05): `watchEvery` becomes `socketWatch`, which says how (`inotify`,
`kqueue`, `off`, or `unavailable: <why>`). That is the whole change of Phase 12
to this reply. The parent watch of the guest (P12-07) moved to kqueue and did
not touch the wire.

**8 (0.8.0): `session/new` may omit `sessionId`, and the daemon names the
session.** The reply carries the id, as it always has. An id that the daemon
makes is new, so the store of the daemon is not searched for it. An id that a
client made and sent was searched for twice, one time for its start and one time
by the open, for a log that could not be there. Every other method about a root
still requires its id (decision 55): only a session that does not exist yet has
no id to give. A 0.7 client always sends one and is unaffected. A client that
omits it against a 0.7 daemon is refused `invalid_params`, and a restart of the
daemon on the same release fixes that.

**9 (0.9.0): the sampling of a route is one value, and a skill says what its
child owes.** No method changed. Three replies embed core models that did
change. A `ModelRoute`, in `models/list` and in the `startsOn` of
`daemon/status`, carries `sampling: {temperature, topP, ...}` where it carried
`temperature`, and may carry `efforts`. A `Skill`, in `skills/list`, always
carries `effort`, `accepts` and `output`. And the `header` of `session/new` is a
format-4 session header. At attach, a 0.8 client refuses the header and the
skill list of a 0.9 daemon, and a route that sets either new field. A 0.9 client
refuses the header of a 0.8 daemon. A restart of the daemon on the same release
fixes both. This number was released with format 4 of the session log, so each
number moves one time.

**10 (0.10.0): a daemon answers for the work that it owes and the logs that it
could not write.** `daemon/status` carries `unwritten`, the roots with a log
that the daemon could not write (P13-08), and `probe`, the place where its
probes answer over TCP (P13-12). A 0.9 client therefore refuses the status of a
0.10 daemon. Three methods are new: `sessions/owed` (P13-02), `sessions/resume`
(P13-03) and `daemon/probe` (P13-12). Each is `unknown_method` to a 0.9 daemon.
A restart of the daemon on the same release fixes both. The session log stays
format 4: no record type changed.

The same number covers **`StatusReading.slot` gaining `model` and `context`**
(the TUI chrome plan, T1). The token meter places its `reasoning` reading beside
the model name and its `cache` reading beside the context gauge, and spells the
latter `cache 28.8k(98%)`. `slot` is a closed literal, so a client that was
built before this refuses the readings that a newer daemon stamps with either
value. A restart of the daemon on the same release fixes that.

**11 (0.13.0): the status of a root says when the root compacts, and
`daemon/config` is gone.** `RootDescription` and `StatusFacts` gain `compaction`
(`CompactionFacts`). Its `triggerFraction` is the share of the context window at
which the root compacts on its own, and it is absent when nothing compacts on
pressure. A front end draws its context gauge in the warning color from that
share, and no longer from a copy of the default. A client that was built before
this refuses the `compaction` that it does not know, because every model is
`extra="forbid"`. Restarting the daemon on the same release fixes that.
`daemon/config` sent the whole composed profile to a front end. Nothing asked for
it after `credentials/held` answered the one question that it was read for, so
the verb is removed, and an 11 daemon answers it with `unknown_method`.

The same number covers **a value that is absent being left out of its frame**,
where it used to travel as an empty string or as zero. A root description
(`sessions/list`, `session/new`, `session/attach`, `session/status`, and the
reply to a mutation) omits `provider`, `model`, `modelKey`, `profile` and
`profileNote`. A `session.status` that does not say where the root is omits
`status`. A frame states the route when it carries `model`, and its `provider`
and `modelKey` are read with it, so no `modelKey` there means that the profile
lists no name for the route. `daemon/status` omits `probe`, and the `key` of
`startsOn`. A `sessions/owed` row omits `cwd`, `profile` and `updated`. A
`sessions/resume` row omits `status` and `detail`. A `session.children` row
omits `name` and `model`. A `sessions/browse` row omits `title`, `cwd`, `family`
and `kind`, and a row for a live root that the disk has not seen omits
`modified` and `size` too. In the other direction, a client omits `clientId`
and `commandId` from an unkeyed mutation, `trust`, `profile` and `cwd` from
`session/new`, and `cwd` from `sessions/browse`. An 11 daemon refuses an empty
string in any of those as `invalid_params`. It also refuses an empty string in
the `name` and `mime` of `attachment/put`, in the `scheduleId` of
`schedule/create` and `schedule/cancel`, in an item of the `sessions` of
`sessions/resume`, and in an item of the `capabilities` of `initialize`. It
refuses a `commandId` sent without a `clientId`, because a command id is unique
only per client, and two clients without one shared a key. A 0.12 client sends
`trust: ""` from a trusted directory and `cwd: ""` for an unfiltered browse, so
both calls are refused. It also requires `updated`, `modified`, `size` and the
`key` of `startsOn`, and refuses a reply that omits one. A 0.13 client that
reads a 0.12 daemon takes each empty string as a value. It reads the
`status: ""` of a footer frame as a status, shows an empty profile note at
attach, and refuses the `kind: ""` of a `sessions/browse` row. Restarting the
daemon on the same release fixes both.

One value still travels with empty strings: the `choice` of `session/new` and
`session/model` (`ModelChoice`, whose `key`, `provider` and `model` are `""`
when not chosen). It is a type of `ph-core`, and it changes with the other
`ph-core` types that spell absence as `""`.

**Nothing refuses on this number, and the place that declares the number should
say so.** The number is reported in the capability block of `daemon/hello` and
printed by `phern agents doctor`. No client compares it and hangs up. So it
documents a skew and does not prevent one. The skew is reachable: `ensure_daemon`
connects to whatever is already listening, so an upgraded client can meet a
daemon that is still running from before the upgrade. The fix for that is to
restart the daemon, which `phern agents shutdown` does. A refusal here would be
a policy decision and not a number, and this row did not take that decision.
"""

SNAPSHOT_EVENTS = 2048
"""How many events one `session/snapshot` reply carries.

A resumed root can hold hundreds of thousands of events. A client that asked for
its history must not receive a frame that exceeds the `MAX_LINE` of the
transport itself, or a reply that it must buffer whole before it renders a line.
The cursor in each reply is what asks for the next page.

**It is a count, not a byte budget.** To fill a byte page, each event would be
measured with its own `dumps`. That costs more than the encode that it exists to
bound, and all of it is discarded. A count needs no measuring pass, and the
`MAX_LINE` of the transport is the real protection against an oversized frame.
"""

MethodResult: TypeAlias = WireModel | dict[str, Any] | None
"""What one method returns: its reply model, the object that the model would
dump to, or nothing at all.

It is a model *and* an object, because `respond` is the single point where a
result becomes a frame. A handler that answers with a model never writes
`.to_wire()` at its own `return`, and a handler whose reply has no model returns
the object directly. It is nothing wider. `result_of` reads the far end of this
contract as `dict[str, Any]`, so a method that answered with a bare list or
string would arrive as `{}`.

`None` is what a `Notify` answers. See `Announced` in `daemon/server.py`. Its
frame carries no id, so `respond` computes nothing and writes nothing. The
alternative was an `{"ok": True}` literal that existed only to give the handler
something to return.
"""

Dispatch = Callable[[str, dict[str, Any]], Awaitable[MethodResult]]
"""The method table of a server: `(method, params) -> result`. It raises to
refuse.

`params` arrives as the peer sent it. The table narrows it to the model of the
method as its first step, with `parse_params`. This signature does not narrow
it, because the model depends on the method and the envelope does not know the
vocabulary."""


# ---------------------------------------------------------------- envelopes --
# The four shapes of JSON-RPC 2.0, as the `TypedDict`s that this side builds.
# Each has its own `Literal["2.0"]`, and there is no shared base. A `TypedDict`
# base with only `jsonrpc` would let a reader accept "any frame", where every
# reader here wants one direction.


class NotificationFrame(TypedDict):
    """A frame with no `id`, which is what makes it a notification."""

    jsonrpc: Literal["2.0"]
    method: str
    params: dict[str, Any]


class RequestFrame(TypedDict):
    """A frame that expects a reply. `id` is `int | str` because both ends mint
    ids. `request` says why."""

    jsonrpc: Literal["2.0"]
    id: int | str
    method: str
    params: dict[str, Any]


class ErrorData(TypedDict):
    """The one member pH puts under `error.data`: the refusal's name."""

    reason: str


class ErrorBody(TypedDict):
    """The error object of JSON-RPC. `data` is present exactly when the refusal
    named itself. `respond` says why the code stays generic and the name does
    not."""

    code: int
    message: str
    data: NotRequired[ErrorData]


class ResultFrame(TypedDict):
    jsonrpc: Literal["2.0"]
    id: int | str
    result: dict[str, Any] | None
    """`None` for a `Notify` that was answered with an id. It is an
    acknowledgement that carries nothing, and that is intended. `shutdown` has
    no reply model, and to coerce this to `{}` would bring back the invented
    `{"ok": true}` that `test_an_rpc_round_trip_in_the_sdk_shape` exists to keep
    out. `result_of` reads either value back as `{}`."""


class ErrorFrame(TypedDict):
    jsonrpc: Literal["2.0"]
    id: int | str
    error: ErrorBody


ReplyFrame: TypeAlias = ResultFrame | ErrorFrame
"""What `respond` returns for a request that wanted an answer."""

Frame: TypeAlias = NotificationFrame | RequestFrame | ResultFrame | ErrorFrame
"""Every frame that this side builds: what the outbox carries and what
`write_frame` takes."""


class Refusal(Exception):
    """A refusal that names itself, for a server to raise.

    `code` is a class attribute here because these refusals are *kinds*: an
    unknown method is one thing, whatever the method was. An error that computes
    its code for each instance sets `self.code` instead. The coded errors of
    `ph-core` do that (`HarnessError`, `SessionForkError`, `CompactionError`),
    and `respond` reads that attribute, so both shapes reach the wire through
    one path.

    It is declared and not duck-typed, for the reason that the seams give for
    their provider Protocols. A probe finds whatever happens to be called
    `code`, and an error whose attribute drifted would go out unnamed, with
    nothing to report it.

    The base class names no kind, so its `code` is `None`. A bare `Refusal`
    goes out with its message and no `data.reason`.
    """

    code: str | None = None


class SeamAbsent(Refusal):
    """This deployment did not mount the seam that the method needs.

    It has its own code, because `unknown_method` says something different. That
    one means "this daemon is older than you think", and a client responds by
    disabling the feature everywhere. This one means "this deployment does not
    do that". That is a fact about one root, and the right response is to gray
    out one button.

    The read-side projections answer absence with an empty list, for the same
    reason seen from the other side. See `projections.py`. So the two halves
    agree that a missing seam is a fact about the profile, not a fault.

    **It is here and not in the daemon.** It was put here because the in-process
    front end refused for the same reason, and could not import from `daemon`
    without a cycle. That front end is deleted. `ph_app.shell` still raises this
    from outside the daemon package, and `daemon/server.py` imports
    `ph_app.shell`. A fourth refusal class per absent seam is the accumulation
    that this one was generalized to stop.
    """

    code = "seam_absent"


class UnknownMethod(Refusal):
    """This server does not serve that name.

    It is here and not in either server, because both raise it, and a client
    that branches on `unknown_method` must not need to know which transport
    answered. It lived in `daemon/server.py`, which a front end may not import
    (`test_app_layering`). So the stdio transport had grown a byte-identical
    second copy: one refusal code, two definitions, and nothing that compared
    them.
    """

    code = "unknown_method"


class InvalidParams(Refusal):
    """The request named a method this server serves, with params it does not take.

    It has its own code because the next step of the client differs from the
    one for `unknown_method`. That one means "this daemon is older than you
    think". This one means "you spelled the call wrong": a missing `sessionId`,
    a field that the method does not take, a cursor that is not a cursor. Before
    P8-07 the first of those was a `KeyError: 'sessionId'` rendered as the error
    message, which named the field and nothing else. The second was silently
    ignored.
    """

    code = "invalid_params"


class DaemonError(RuntimeError):
    """A refusal that the server sent back, with its name where it had one.

    This is the other end of the `data.reason` of `respond`. Without it, a
    client that wanted to tell `session_already_active` (I-5) from a mistyped
    method had to match on the message text. That is a contract that nobody
    wrote down and that every rewording breaks. `reason` is `None` when the
    server did not name one.
    """

    def __init__(self, message: str, reason: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason

    @classmethod
    def of(cls, error: dict[str, Any]) -> DaemonError:
        """Rebuild the refusal from an error frame. A frame with no `data.reason`,
        or with an empty one, names no reason."""
        data = error.get("data")
        reason = named(data.get("reason")) if isinstance(data, dict) else None
        return cls(as_str(error.get("message"), "the daemon refused"), reason)


class DaemonGone(DaemonError):
    """The connection ended without an answer. Nobody refused anything.

    It is a subclass, so a caller that only wants "the call did not succeed"
    still catches `DaemonError`. It is a distinct type, because the two are
    opposite diagnoses and a client renders them differently. When this was
    reported as a refusal, the message read "the daemon refused: the daemon
    closed the connection". That is precisely the confusion that the split
    between an absent socket and a stale socket in `phern agents` exists to
    prevent.
    """

    def __init__(self, message: str = "the daemon closed the connection") -> None:
        super().__init__(message, "connection_closed")


class CapabilityBlock(WireModel):
    """`initialize` and `daemon/hello`: the version, and what this end serves.

    One handler answers both names, so both verbs name this one reply. The
    values are all `True` by construction. `protocol.capabilities` says why a
    capability is present or absent and is not a flag. That makes
    `dict[str, bool]` the correct declaration, and not a field for each name.
    """

    protocol_version: int
    capabilities: dict[str, bool] = Field(default_factory=dict)


def capabilities(*names: str) -> CapabilityBlock:
    """The `initialize` reply, with whatever this transport adds.

    `sessions` and `streaming` are true of both transports. A transport that
    supervises adds names such as `roots`, `attach`, `cursors` and `snapshots`,
    and one that does not supervise omits them. A client reads the block and
    does not infer from which socket it happened to open. The arguments are
    names and not `**kwargs`, because every value is `True` by construction: a
    capability is present or absent.
    """
    return CapabilityBlock(protocol_version=PROTOCOL_VERSION, capabilities=served(*names))


def served(*names: str) -> dict[str, bool]:
    """The capability map alone, for a reply that *is* a `CapabilityBlock`.

    `DaemonStatusReply` subclasses the block. `daemon/status` used to build a
    whole `CapabilityBlock` and unpack two fields out of it. That named both
    fields a second time, so a third field added to the block would reach
    `initialize` and silently not reach `daemon/status`. Here the only thing
    that is written twice is `PROTOCOL_VERSION`, which is a reference to a
    constant and cannot disagree with itself. The map is split out, and
    `model_dump()` is not spread, because a `**` spread would take `pid`,
    `socket` and the settings of the daemon with it, and mypy would stop
    checking any of them.
    """
    return {"sessions": True, "streaming": True, **dict.fromkeys(names, True)}


def notification(method: str, params: dict[str, Any]) -> NotificationFrame:
    """A frame with no id, which is what makes it a notification."""
    return {"jsonrpc": "2.0", "method": method, "params": params}


def request(request_id: int | str, method: str, params: dict[str, Any]) -> RequestFrame:
    """A frame that expects a reply, from either side.

    The id is `int | str` because both ends mint ids now (P5-13). A client
    counts its own calls, and the daemon mints `"s<n>"` for the questions that
    it puts *to* a client. Strings and ints cannot collide, so a reader of a
    frame log can tell at a glance which side asked.

    **So `id` cannot say in which direction a frame is going, and `method`
    can.** A frame that carries a `method` is somebody asking, and a frame
    without one is an answer. Both pumps route on that. Both were written on the
    opposite assumption, which is why it is stated here one time and not twice
    over there.
    """
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


def result_of(frame: dict[str, Any]) -> dict[str, Any]:
    """One reply frame as its result. Raises whatever the peer refused.

    Both directions unwrap a reply in the same way, so this is the one place
    that knows that an `error` member outranks a `result`. When this was written
    twice, the two copies disagreed about what an empty frame meant.
    """
    if "error" in frame:
        raise DaemonError.of(frame["error"])
    result: dict[str, Any] = frame.get("result") or {}
    return result


class Cursor(WireModel):
    """Where a reader has got to: `{generation, sequence}`.

    A sequence alone has meaning only against the log that counted it. So it
    travels with the identity of that log **as it ran then**. `generation` is
    the time when that incarnation of the log began (`generation_of`), because a
    resume does not always continue from where a reader left off. `resume_at`
    says why, and how far the count of an older incarnation still holds (S17).

    It is a model and not the dict that it was, so the shape has one spelling.
    `cursor_of` builds it here, a method sends it back inside its params, and
    `resume_at` reads it off the model and not through two `isinstance` checks.
    A client sends it as the dict that `to_wire()` produced, and the params
    model of the method rebuilds it. A cursor that is not a cursor is therefore
    refused as `invalid_params` at the edge. A *stale* cursor (the right shape,
    the generation of another log) is still answered by `resume_at` with 0.
    """

    generation: str
    sequence: int


def cursor_of(session: Session, sequence: int | None = None) -> Cursor:
    """The position of a session as a `Cursor`. `to_wire()` puts it in a reply.

    It is here and not on the `Root` of the daemon, because it is a fact about a
    *session*. The stdio transport serves the same protocol, and would otherwise
    have to derive it again.
    """
    return Cursor(
        generation=generation_of(session),
        sequence=session.seq if sequence is None else sequence,
    )


def generation_of(session: Session) -> str:
    """Which incarnation of the log of `session` is running, named by when it
    began: the creation of the log, or its latest own resume (`resumption_of`).
    The value is an integer, as `createdAt` alone was, so the spelling of a
    cursor did not change."""
    resumed = resumption_of(session)
    return str(resumed.time if resumed is not None else session.header.created_at)


def cursor_text(cursor: Cursor) -> str:
    """A cursor as `GENERATION:SEQ`, the form that a person or a script passes back.

    It is beside `cursor_of` for the reason that `cursor_of` itself states. The
    shape is a fact about a session, not about one transport or one command, and
    the printed form is the same fact spelled for a terminal. It lived as an
    f-string in the status table of the CLI and a `rpartition` in its option
    parser, 350 lines apart, with nothing that tied the two.

    **It takes the model, and the parameter used to be `Any`.** The body was a
    `dict`-or-nothing test whose `else` branch returned `":"`. That was written
    when every caller held a reply dict, and it went wrong the moment one held a
    `Cursor`. `phern agents status` printed `--since :` and nothing complained,
    because `Any` accepts a model and the `isinstance(cursor, dict)` test
    quietly failed. This is the lesson of issue 74, and it arrived inside issue
    74: a widened parameter is a deleted check, and the deletion shows at the
    call site that changes.
    """
    return f"{cursor.generation}:{cursor.sequence}"


def is_number(text: str) -> bool:
    """Whether `text` is a run of ASCII digits (E6).

    `str.isdigit()` alone is true for `²`, `٣` and the rest of the numeric
    characters of Unicode, and `int()` raises on most of them. That would be a
    traceback out of a field that a *peer* sent, on the receive path, where the
    contract is to answer `None` and not to fail. `isascii()` beside it is the
    narrow test that matches what a cursor can legitimately be.

    It was made public because the generation was parsed in two places: here,
    and in the attach reply reader of the TUI, which had the bare `isdigit()`
    that this fixes. `parse_cursor` is now the only caller.
    """
    return text.isascii() and text.isdigit()


def parse_cursor(text: str, current: object) -> Cursor | None:
    """`GENERATION:SEQ` or a bare `SEQ`, as a cursor, or `None` if it is neither.

    There are two spellings, and only one of them can be checked. The full form
    is a cursor that a reader kept from an earlier read. It is passed through
    intact, so that `resume_at` can refuse it when the log has since been forked
    or rebuilt. A bare sequence is stamped with the generation of `current`.
    That is the only thing that gives a typed number a meaning, and it is
    exactly what makes the number impossible to verify.

    `generation` is an integer of epoch milliseconds (`generation_of`), so the
    split is unambiguous. The function returns `None` and does not raise,
    because the caller decides what to do about a cursor that does not parse.
    The CLI exits 2. A front end that reads a stored position would start from
    the beginning before it would fail to open.

    Returns the model, not its dict. A caller that puts it on the wire writes
    `.to_wire()` there, which is the one place where the shape becomes JSON.
    """
    generation, separator, sequence = text.rpartition(":")
    if not separator:
        fields = current if isinstance(current, dict) else {}
        generation, sequence = as_str(fields.get("generation")), text
    # `str.isdigit()` is true for `²`, `٣` and every other Unicode digit, and
    # `int()` then raises on most of them (E6). That would be a traceback out of
    # a cursor that a client sent, on the read path, where the whole contract of
    # this function is to answer `None` and not to fail. `str.isascii()` beside
    # it is the narrow test that matches what `int` will actually take.
    if not (is_number(sequence) and is_number(generation)):
        return None
    return Cursor(generation=generation, sequence=int(sequence))


def resume_at(session: Session, cursor: Cursor | None) -> int:
    """The index that a cursor asks to resume from, or 0 when it cannot say.

    **A cursor from an earlier incarnation of this log holds only as far as
    every resume since then kept** (S17). Events reach a reader before they are
    durable, so a crash can take back events that the reader saw. The resume
    that follows writes its repair closers and `session/resumed` at those very
    seqs. When the cursor was honored whole, it skipped them and kept events
    that the log no longer has. Each resume records how much of the log it began
    with on disk. So the reader resumes at the least of those counts and its own
    place, and the reply that says where is what tells it to drop the rest.

    A cursor from another log altogether is neither honored nor refused. To
    honor it would skip events that the client never saw. To refuse it would
    strand a client that did nothing wrong. So it reads as "you have seen
    nothing of *this* log", and the reply says where the read actually started.

    The shape is the business of the params model, which is why there is no
    `isinstance` here any more: a cursor that is not a cursor never reaches this
    function.
    """
    if cursor is None:
        return 0
    if cursor.generation == generation_of(session):
        # This is the common case, and the one that a catch-up pages through. So
        # it does not walk the log for resumes to bound it by: there are none
        # since.
        return max(0, min(cursor.sequence, session.seq))
    resumed = resumptions(session)
    generations = [str(session.header.created_at), *(str(one.time) for one in resumed)]
    if cursor.generation not in generations:
        return 0
    since = resumed[generations.index(cursor.generation) :]
    return max(0, min(cursor.sequence, session.seq, *(one.events for one in since)))


def parse_params[P: WireModel](method: str, model: type[P], params: object) -> P:
    """A request's params as the method's model, or the refusal that names why.

    There is one place for every server, so the sentence that a client reads for
    a bad call is the same over the socket and over stdio. `extra="forbid"` is
    inherited from `WireModel`. A field that the method does not take is refused
    and not ignored, because an ignored field was a client that believed it had
    said something. The error text of Pydantic is kept, because it names the
    field and the shape. It goes under the name of the method, which is what the
    reader is missing.
    """
    try:
        return model.model_validate(params)
    except ValidationError as error:
        raise InvalidParams(
            f"{method}: {'; '.join(validation_errors(error, root='params'))}"
        ) from error


@dataclass(frozen=True, slots=True)
class _Spoken[P: WireModel]:
    """A wire method name bound to the params model that it carries.

    This is the half that `Verb` and `Notify` share. It is private because
    neither client method takes it: `DaemonClient.call` wants a `Verb` and
    `notify` wants a `Notify`. The purpose of two types is that the wrong one is
    refused. A base that they *both* satisfy would be a third type to accept,
    and it would admit either.
    """

    name: str
    params: type[P]

    def parse(self, params: object) -> P:
        """The wire params as this method's model, or the refusal that names why.

        It is on the method and not at the dispatch, because the method is what
        knows the model. The caller used to pass both, and these types exist so
        that the two are no longer a separate pair.
        """
        return parse_params(self.name, self.params, params)


@dataclass(frozen=True, slots=True)
class Notify[P: WireModel](_Spoken[P]):
    """A method with **no reply**: a name, its params, and nothing that comes back.

    `shutdown` is the one such method, and its hazard is why this type exists in
    place of a `Verb` with an unread reply model. A request with a reply would
    leave the caller waiting for a frame that the daemon is losing the ability
    to write at the same moment. So "stop" is not a question and does not get an
    id. But while `shutdown` was a `Verb`, `client.call(SHUTDOWN, NoParams())`
    type-checked. Every one of the 19 call sites happened to use `notify`, and
    nothing made that a rule. Now `call` cannot take this type and `notify`
    cannot take a `Verb`. The hang cannot be written at all, where before it was
    only avoided.

    The type also deletes a model. `ShutdownAck` existed because a `Verb` must
    name a reply type and the handler must return *something*. It was for a
    frame that `respond` computes and then drops because the id is `None`: eight
    lines that described a reply nobody reads. A handler for a `Notify` returns
    `None`, which is the correct shape.

    **It is not a subtype of `Verb`, and `Verb` is not a subtype of it.** If one
    extended the other, `notify(SESSION_STATUS, …)` would pass. That call runs a
    method on the daemon and silently discards its answer, which is a different
    way to be wrong about the same distinction.
    """


@dataclass(frozen=True, slots=True)
class Verb[P: WireModel, R: WireModel](_Spoken[P]):
    """One method as one value: its name, what it takes, and what it answers.

    **The three used to be three.** A name was a string literal at the call
    site. The params model was named again in the table of the server. The reply
    was whatever the handler happened to return. So
    `client.call("session/status", SessionParams(session_id=s))` type-checked,
    and so did `client.call("sessions/list", SessionParams(...))`. P8-09 gave
    checked field spelling inside the params. It could not check that the params
    belonged to the method, because nothing held the two together. A `Verb`
    holds them together. `client.call(SESSION_STATUS, SessionParams(...))`
    cannot be given the params of another verb, and its reply arrives as
    `RootStatusReply` and not as a dict that the caller narrows again by hand.

    Verbs are declared in `ph_app.verbs`, which both ends import. That is the
    purpose, and it is the reason that this class lives here and not there.
    `protocol.py` is what a transport may depend on, and a verb table that
    imported the handlers of the server could not be read by a client.

    **Notifications are not verbs, and `METHOD` is not replaced.** A notice
    already binds its name to its payload on the payload itself
    (`SessionScoped.METHOD`). That is what let `Root.publish` take one argument
    and not two that had to agree. A `Verb` that owned those names would bring
    the two arguments back, so it does not own them. Verbs cover the
    request/reply half, and the emitted half keeps the binding that it has.
    """

    reply: type[R]

    def read(self, wire: dict[str, Any]) -> R:
        """A reply frame as the model of this verb.

        This retires `AttachReply.model_validate(await client.call(...))`. That
        was a dump and then a second parse, with a model that was chosen by hand
        at the call site. It could be the wrong model, with nothing to report
        it.
        """
        return self.reply.model_validate(wire)


class NoParams(WireModel):
    """A method that takes none.

    It is a model, and the parse is not skipped, so that a stray field on one of
    these methods is refused like a stray field anywhere else. The rule has no
    exceptions, and that is what makes it a rule. It is here and not in the
    table of either server for the same reason as `Cursor`: both transports have
    methods that take nothing, so the shape belongs to the protocol.
    """


def _usable_as_a_path_component(value: str) -> str:
    if not valid_session_id(value):
        raise ValueError(
            "sessionId must be alphanumerics, dots, dashes and underscores, "
            "starting with a letter or a digit"
        )
    return value


Stated = Annotated[str, StringConstraints(min_length=1)]
"""A string param that names something, so an empty string is refused.

A param that a client may leave out is `Stated | None`: it is absent from the
frame, or it names something. An empty string is neither, and it is refused as
`invalid_params`. It is not read as "absent", because two spellings of one fact
are how a client and a daemon come to disagree. `test_every_optional_text_param_is_stated`
holds every params model to this rule."""

SessionId = Annotated[str, AfterValidator(_usable_as_a_path_component)]
"""A session id on the wire, **checked here** (K9). An id becomes a directory
name in the spill store, the archive and the workspace scratch root. Several
reads of the daemon itself (`recorded_start`, the lease, `locate_session`) build
a path from it before any `Session` exists to refuse it. It is a type and not a
validator on one model, so that a method whose id may be absent (`session/new`)
is checked in the same way."""


class SessionParams(WireModel):
    """Every method about one session starts here.

    It belongs to the protocol, for the same reason: `sessionId` is what a
    method about a session is *about*, on either transport. What each server
    adds to it is its own: the idempotence key of the daemon, the optional id of
    stdio. That is where the two vocabularies really differ. Its id is a
    `SessionId`, so a handler cannot forget a check that the wire type already
    made.
    """

    session_id: SessionId


async def respond(request_frame: dict[str, Any], dispatch: Dispatch) -> ReplyFrame | None:
    """Run one request and shape its reply, or `None` if it wanted none.

    A method that fails is the failure of *this call*, not of the connection. An
    unknown method or a bad argument comes back as an error frame, and the peer
    keeps talking. Framing errors belong to the transport and end the stream,
    because after a bad frame there is no way to know where the next one starts.

    The body runs **before** the `id` check, not after. A frame with no id is a
    method whose answer nobody wants (`shutdown`), and its body still has to
    run.
    """
    request_id: int | str | None = request_frame.get("id")
    method = as_str(request_frame.get("method"))
    params = request_frame.get("params") or {}
    try:
        answer = await dispatch(method, params)
        # A handler that answers with a model is dumped **here**, at the one
        # point where a result becomes a frame. So every handler returns either
        # the type of its reply or a plain dict, and none of them writes
        # `.to_wire()` at the `return`. `dumps` cannot encode a model, so this
        # is also the only place that could have gone wrong.
        result = answer.to_wire() if isinstance(answer, WireModel) else answer
    except Exception as error:
        failure: ErrorBody = {"code": -32000, "message": str(error)}
        # A refusal that a client is expected to *branch* on carries a name, so
        # that the client need not match message text. `session_already_active`
        # (I-5) is a thing to retry elsewhere, and to tell it from a typo in a
        # method name should not mean a search of prose. The code of JSON-RPC
        # itself stays generic, because these names are the vocabulary of pH and
        # not of the transport.
        #
        # The name is read off the *instance*, which is what already carries a
        # code here. `HarnessError`, `SessionForkError` and the other coded
        # errors all set `self.code` in `__init__`, so a key on `type(error)`
        # would make every one of those families structurally invisible.
        #
        # An error that names nothing has no `code`, or a `code` of `None`
        # (`Refusal`). The test for an empty string stays for an error of another
        # package that spells "no code" that way, so the frame never carries a
        # reason that names nothing.
        reason = named(getattr(error, "code", None))
        if reason is not None:
            failure["data"] = {"reason": reason}
        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "error": failure}
    if request_id is None:
        return None
    return {"jsonrpc": "2.0", "id": request_id, "result": result}
