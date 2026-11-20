"""The method vocabulary of the daemon: one parameter model for each method (P8-07).

`_dispatch` used to read `params` as a `dict[str, Any]`. It called
`str(params["sessionId"])` at twenty-one sites. It handed `params.get("cursor")`
to a function that then asked `isinstance` twice. It silently ignored a field
that the method did not take. Each of those was a claim about the wire, made at
the point of use. mypy could check none of them, because a dict says nothing
about its keys.

Each model here is that claim, made one time, at the edge. The argument of
`Session._append` for typing a payload applies unchanged: a producer that can be
checked is checked, and the runtime gate stays for the producer that cannot. The
gate is `parse_params`, which refuses with `invalid_params`.

**`extra="forbid"`, inherited from `WireModel`, is a deliberate change of
policy.** A field that the method does not take used to be dropped. It is now a
refusal that names the field. The dropped field came from a client that believed
it had said something: `trust` on the wrong method, a cursor on
`session/status`. A daemon that agrees without listening is the failure that a
typed edge exists to end. Every field that every client in this repository sends
is declared below. The clients are `ph_app.tui.remote`, `ph_app.agents`,
`ph_app.attach`, `ph_app.service`, `ph_app.profiles_cli`,
`ph_app.daemon.follow`, and the idempotence pair of `DaemonClient.mutate`. The
tests of the daemon send nothing else.

**The models are beside `payloads.py`, not under `daemon/`.** To place them
with the table that reads them was the obvious choice, and it was wrong. Client
modules outside `ph_app.daemon` build request params: `attach.py`, `agents.py`
and `tui/remote.py`, and now `service.py` and `profiles_cli.py` too. `attach.py`
is the human route, and its layering test promises that it "needs no daemon at
all". A home that followed the owner therefore put a runtime edge from each of
those modules into `ph_app.daemon.*`. Direction was the distinction that was
needed: what a client *sends* is here, what the daemon *emits* is in
`payloads.py`, and the envelope that both share is in `protocol.py`. The
layering test of `attach.py` (`test_the_human_door_needs_no_daemon_at_all`) can
then name the whole `ph_app.daemon` subpackage, and does not list three modules.

**The camelCase comes from the alias function, not from the field.**
`session_id` is `sessionId` on the wire, `content_b64` is `contentB64`, and
`schedule_id` is `scheduleId`. `ph.wire.wire_alias` makes each of them. The
spelling that a client learns from the dsh SDK is therefore the spelling that
pydantic accepts, and no model here carries a string literal for a key.

**Which side owns a nested shape.** `Cursor` belongs to the protocol, because
both transports build one. It is in `ph_app.protocol` and is only *used* here.
`AttachmentRef` belongs to the log (`ph.llm.types`). A params model that names
it gets the nested validation of pydantic at no cost. A malformed reference is
refused as `invalid_params` before the store is asked whether it holds one, so
`attachment_unknown` means only what it says.

@module ph_app.params
"""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, model_validator

from ph.llm.types import AttachmentRef
from ph.seams.models import ModelChoice
from ph.seams.permission_presets import PresetName
from ph.seams.schedule import Schedule, ScheduleKind
from ph.wire import WireModel

from .protocol import Cursor, ProbeName, SessionId, SessionParams, Stated
from .trust import TrustAnswer

__all__ = [
    "AdoptParams",
    "BrowseParams",
    "CancelScheduleParams",
    "CommandParams",
    "CreateScheduleParams",
    "InitializeParams",
    "ModelParams",
    "MutationParams",
    "NewSessionParams",
    "OwedParams",
    "PresetParams",
    "ProbeParams",
    "PromptParams",
    "PutAttachmentParams",
    "ResumeParams",
    "ShellParams",
    "SnapshotParams",
    "StageParams",
    "StoreCredentialParams",
]


class InitializeParams(WireModel):
    """`initialize` / `daemon/hello`: what this *client* can do (P5-13)."""

    capabilities: list[Stated] = Field(default_factory=list)


class BrowseParams(WireModel):
    """`sessions/browse`: which sessions to list.

    **Every field has a default.** A client that sends `{}` gets every session,
    as it did before this model existed. When the model was added, the wire
    therefore did not move, and neither did `PROTOCOL_VERSION`.

    `cwd` is `None` for every session. A client with no directory to filter on
    omits the field. An empty string is not a directory, and it is refused.
    That refusal is new in protocol 11: a 0.12 client sent `cwd: ""` for an
    unfiltered browse (see `PROTOCOL_VERSION`).

    `cwd` is compared with the string that the header of a session recorded,
    with no `resolve()`. The daemon may be on another machine and may not have
    the path at all. Whether two spellings of one checkout are the same
    directory is for the user to decide, and not for the protocol. A mismatch
    costs a row in the picker, and an unfiltered browse still lists everything.

    **There is no `limit` field**, although the fold has one. Nothing would set
    it, because both clients send `cwd` alone. A declaration here would put `50`
    in a third place beside `browse_of` and `session_summaries`, and only one of
    those decides anything. Because every field has a default, the first client
    that wants `limit` can add it back, and the wire stays compatible in the way
    that the first paragraph describes.
    """

    cwd: Stated | None = None


class NewSessionParams(WireModel):
    """`session/new`. `cwd` is the directory of the client, because the daemon
    mounts.

    This is not a `SessionParams`. It is the one method about a root whose root
    may not exist yet, so its id may be absent. The daemon names a new session
    (protocol 8)."""

    session_id: SessionId | None = None
    cwd: Stated | None = None
    """The directory of the client, or `None` for a client that has none to name.
    An empty string is not a directory, and it is refused."""
    trust: TrustAnswer | None = None
    """What the user answered about this `cwd`, or `None` when nobody was asked.
    The daemon then checks the store (`ph_app.trust`)."""
    choice: ModelChoice = Field(default_factory=ModelChoice)
    """What this session runs on. Empty means the `--model` of the daemon, or
    else the default of the profile. A fresh root mounts on it. A root that is
    already running is moved to it, as `session/model` would move it."""
    profile: Stated | None = None
    """The named profile that a *new* session is created on (S5). `None` means
    the `--profile` of the daemon. A session that already has a log runs on
    what its log records. `/profile use` switches that, and this field does
    not."""


class SnapshotParams(SessionParams):
    """`session/snapshot`: a root, and where to start reading.

    `None` means "from the beginning", and so does a stale cursor. `resume_at`
    says why the two read the same. A client may not guess what a `Cursor` is: a
    dict of the wrong shape is refused here, and is not read as `None`.

    **`session/attach` does not take a cursor, although it looks as if it
    should.** It took a `cursor` for as long as it existed and never read it.
    Attach subscribes to what happens *next*, and the reply says where that
    starts, so from that point catch-up belongs to `session/snapshot`. A field
    that is accepted and ignored is what the `extra="forbid"` of this module
    exists to stop. Attach therefore declares `SessionParams`. A client that
    asks it to replay is now told, and is not quietly given a live-only
    subscription.
    """

    cursor: Cursor | None = None


class MutationParams(SessionParams):
    """Every method in `MUTATIONS`: a root, and the idempotence key that it acts
    under.

    Both halves of the key default to `None`. An unkeyed call is legal and
    simply unguarded, and `DaemonClient.mutate` exists to make that contract
    hard to fall into. `_command_key` turns the pair into the string that
    `journal.claim_once` claims (`_Connection._acted` in `daemon/server.py`).

    An empty string is not an id, and it is refused. A call that sent one as
    its `command_id` would otherwise claim a key that every other such call
    shares, and each call after the first would be answered as a repeat.

    **A `command_id` needs a `client_id`**, for the same reason. A command id is
    unique only within one client. Two clients that each sent `commandId: "1"`
    with no client id would claim one key, and the second prompt would be
    dropped as a repeat of the first. A `client_id` alone is legal: it names
    no command, so the call is unkeyed."""

    client_id: Stated | None = None
    command_id: Stated | None = None

    @model_validator(mode="after")
    def _keyed_by_a_client(self) -> Self:
        if self.command_id is not None and self.client_id is None:
            raise ValueError("commandId needs a clientId: a command id is unique only per client")
        return self


class PromptParams(MutationParams):
    """`session/prompt`. `attachments` are references the client already `put`."""

    prompt: str = ""
    attachments: list[AttachmentRef] = Field(default_factory=list)


class CommandParams(MutationParams):
    """`session/command`: one `/name argument` line, run in the root's context."""

    line: str = ""


class StageParams(MutationParams):
    """`session/stage`: one reference for the composer's tray."""

    attachment: AttachmentRef


class ShellParams(MutationParams):
    """`session/shell`: the user's own command, in the workspace of the session."""

    command: str = ""
    surface: bool = False
    """Whether the model is told. `!` sets it and `!!` does not. That is the one
    difference between the two. It is carried on the wire so that a browser tab
    and the terminal mean the same thing by each."""


class PresetParams(MutationParams):
    """`session/preset`. The field is a `PresetName`, so an unknown preset is an
    `invalid_params` that names the field, and not a `KeyError` that names the
    value."""

    preset: PresetName


class ModelParams(MutationParams):
    """`session/model`: a listed key or a whole route, resolved by the profile of the root."""

    choice: ModelChoice


class StoreCredentialParams(SessionParams):
    """`credentials/store`. **The value is used and not kept.** It is never
    logged and never echoed. This model holds it only for as long as the call
    takes. After the call, the value is in the memory of the daemon process, in
    the credential seam of the root, and nowhere else.

    This has not been a `MutationParams` since T5. The verb takes no idempotence
    key, because one value that is stored twice is stored one time, and a key
    would outlive the value that it named. A call that stamps a key anyway is
    refused by `extra="forbid"`. See `PROTOCOL_VERSION` 4 for the 0.3.x client
    that does so."""

    name: str
    value: str


class AdoptParams(SessionParams):
    """`session/adopt`: a version of a named profile, for the next start of this
    session (S6).

    `version` has the shape of a `profile/base` record. It is the whole version
    that the user was shown. That shape belongs to other code, so the field is a
    `dict` (`payloads.py` says why it is not a `JsonObject`). A running root
    keeps its base, and its next start makes this version its base."""

    version: dict[str, Any]


class PutAttachmentParams(SessionParams):
    """`attachment/put`: bytes the *client* read, base64 in one frame (I-9)."""

    content_b64: str = ""
    name: Stated | None = None
    mime: Stated | None = None


class CreateScheduleParams(SessionParams):
    """`schedule/create`. It mirrors `ph.seams.schedule.Schedule` field for field,
    under the `scheduleId` of the wire and not the `id` of the log."""

    schedule_id: Stated
    kind: ScheduleKind
    spec: str
    prompt: str

    def to_schedule(self) -> Schedule:
        """The model of the log, which differs only in the name of the id.

        The conversion is here and not at the handler, so that the field list is
        written one time. A field that was added to `Schedule` and forgotten in
        the copy used to be a value that was silently not scheduled. Under
        `extra="forbid"` it is now a client that the daemon refuses for sending
        what the seam asked for.
        """
        return Schedule(id=self.schedule_id, kind=self.kind, spec=self.spec, prompt=self.prompt)


class CancelScheduleParams(SessionParams):
    """`schedule/cancel`."""

    schedule_id: Stated


class OwedParams(WireModel):
    """`sessions/owed` (P13-02). `rebuild` reads the log of every stored root and
    does not trust the live-roots index. An orchestrator asks for that one time
    after a crash, when it does not trust the file to have survived."""

    rebuild: bool = False


class ProbeParams(WireModel):
    """`daemon/probe` (P13-12): which of the three questions to answer."""

    name: ProbeName


class ResumeParams(WireModel):
    """`sessions/resume` (P13-03): which stored roots to mount again.

    `sessions` names them. `all_owed` is every root that the live-roots index
    says owes work and that this daemon does not hold. A root whose ladder is
    spent is skipped unless `force` is set, because mounting it would be one
    more attempt for each restart."""

    sessions: list[SessionId] = Field(default_factory=list)
    all_owed: bool = False
    force: bool = False
