"""The daemon's method vocabulary, one parameter model per method (P8-07).

`_dispatch` used to read `params` as a `dict[str, Any]`: `str(params["sessionId"])`
at twenty-one sites, `params.get("cursor")` handed to a function that then asked
`isinstance` twice, and a field the method did not take silently ignored. Each of
those was a claim about the wire made at the point of use, and mypy could check
none of them because a dict says nothing about its keys.

Each model here is that claim, made once, at the edge. `Session.append`'s
argument for typing a payload applies unchanged: a producer that can be checked
is, and the runtime gate — `parse_params`, refusing with `invalid_params` —
stays for the one that cannot.

**`extra="forbid"`, inherited from `WireModel`, is a policy change and a
deliberate one.** A field the method does not take used to be dropped on the
floor; it is now a refusal that names the field. The dropped field was a client
that believed it had said something — `trust` on the wrong method, a cursor on
`session/status` — and the daemon agreeing without listening is the failure mode
a typed edge exists to end. Every field every client in this repository sends
is declared below (`ph_app.tui.remote`, `ph_app.agents`, `ph_app.attach`,
`ph_app.daemon.follow`, `DaemonClient.mutate`'s idempotence pair), and the
daemon's own tests send nothing else.

**Beside `payloads.py`, not under `daemon/`.** Placing these with the table
that reads them was the obvious choice and the wrong one: three front-end
modules build request params — `attach.py` (the human door, whose layering test
promises it "needs no daemon at all"), `agents.py` and `tui/remote.py` — so an
owner-based home put a runtime edge from each of them into `ph_app.daemon.*`.
Direction is the discriminator that was actually needed: what a client *sends*
here, what the daemon *emits* in `payloads.py`, and the envelope both share in
`protocol.py`. `FRONT_END_FORBIDS` can then name the whole `ph_app.daemon`
subpackage instead of listing three modules.

**The camelCase is the alias function's, not the field's.** `session_id` is
`sessionId` on the wire, `content_b64` is `contentB64`, `schedule_id` is
`scheduleId` — `ph.wire.wire_alias` in every case, so the spelling a client
learns from the dsh SDK is the spelling pydantic accepts, and no model here
carries a string literal for a key.

**Which side owns a nested shape.** `Cursor` is the protocol's — both transports
build one — so it lives in `ph_app.protocol` and is only *used* here.
`AttachmentRef` is the log's (`ph.llm.types`), and a params model that names it
gets pydantic's nested validation for free: a malformed reference is refused as
`invalid_params` before the store is asked whether it holds one, so
`attachment_unknown` is reserved for what it says.

@module ph_app.params
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import Field

from ph.llm.types import AttachmentRef
from ph.seams.permission_presets import PresetName
from ph.seams.schedule import Schedule, ScheduleKind
from ph.wire import WireModel

from .protocol import Cursor, SessionParams

__all__ = [
    "CancelScheduleParams",
    "CommandParams",
    "CreateScheduleParams",
    "HeldCredentialsParams",
    "InitializeParams",
    "MutationParams",
    "NewSessionParams",
    "PresetParams",
    "PromptParams",
    "PutAttachmentParams",
    "ShellParams",
    "SnapshotParams",
    "StageParams",
    "StoreCredentialParams",
    "TrustAnswer",
]

TrustAnswer: TypeAlias = Literal["", "once", "always"]
"""What a person said about mounting a `cwd` — `ph_app.trust` says what each
means. `""` is "nobody has been asked yet", which is the value a client that has
not shown the modal sends, and the one the daemon then checks the store for."""


class InitializeParams(WireModel):
    """`initialize` / `daemon/hello`: what this *client* can do (P5-13)."""

    capabilities: list[str] = Field(default_factory=list)


class NewSessionParams(SessionParams):
    """`session/new`. `cwd` is the client's directory, because the daemon mounts."""

    cwd: str | None = None
    trust: TrustAnswer = ""


class SnapshotParams(SessionParams):
    """`session/snapshot`: a root, and where to start reading.

    `None` is "from the beginning" and so is a stale cursor — `resume_at` says why
    the two read the same. What a `Cursor` is not, is a client's to guess: a dict
    of the wrong shape is refused here rather than read as `None`.

    **`session/attach` does not take one, though it looks like it should.** It
    took a `cursor` for as long as it existed and never read it — attach
    subscribes to what happens *next* and the reply says where that starts, so
    catch-up is `session/snapshot`'s from that point. A field accepted and
    ignored is the thing this module's `extra="forbid"` exists to stop, so
    attach declares `SessionParams` and a client that asks it to replay is now
    told rather than quietly given a live-only subscription.
    """

    cursor: Cursor | None = None


class MutationParams(SessionParams):
    """Every method in `MUTATIONS`: a root, and the idempotence key it acts under.

    Both halves of the key default to `""` — an unkeyed call is legal and simply
    unguarded, which is the contract `DaemonClient.mutate` exists to make hard to
    fall into. `_command_key` turns the pair into the string `Root.once` claims."""

    client_id: str = ""
    command_id: str = ""


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
    """`session/shell`: the person's own command, in the session's workspace."""

    command: str = ""


class PresetParams(MutationParams):
    """`session/preset`. A `PresetName`, so an unknown one is `invalid_params`
    naming the field rather than a `KeyError` naming the value."""

    preset: PresetName


class StoreCredentialParams(MutationParams):
    """`credentials/store`. **The value is used and not kept**: it is never
    logged, never echoed, and this model is the only thing that holds it —
    for exactly as long as the call takes."""

    name: str
    value: str


class HeldCredentialsParams(SessionParams):
    """`credentials/held`: which of these names the harness already has."""

    names: list[str] = Field(default_factory=list)


class PutAttachmentParams(SessionParams):
    """`attachment/put`: bytes the *client* read, base64 in one frame (I-9)."""

    content_b64: str = ""
    name: str | None = None
    mime: str | None = None


class CreateScheduleParams(SessionParams):
    """`schedule/create`. Mirrors `ph.seams.schedule.Schedule` field for field,
    under the wire's `scheduleId` rather than the log's `id`."""

    schedule_id: str
    kind: ScheduleKind
    spec: str
    prompt: str

    def to_schedule(self) -> Schedule:
        """The log's model, which differs only in what it calls the id.

        Here rather than at the handler so the field list is written once: a
        field added to `Schedule` and forgotten in the copy used to be a
        silently unscheduled value, and under `extra="forbid"` it is now a
        client refused by the daemon for sending what the seam asked for.
        """
        return Schedule(id=self.schedule_id, kind=self.kind, spec=self.spec, prompt=self.prompt)


class CancelScheduleParams(SessionParams):
    """`schedule/cancel`."""

    schedule_id: str
