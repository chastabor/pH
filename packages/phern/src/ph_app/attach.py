"""Files that a user named, turned into a message that the model can see (P7-01).

This is the **human door** of the split that I-9 draws. A user may attach
anything that they can already open. So this module reads paths directly, with
the permissions of the harness itself and no policy check. That is deliberately
not available to the model. A tool that could reach it would be an exfiltration
primitive: attach a private key as a "document" and let the provider OCR it. So
an attach that the model starts reads through `ctx.fs`, where `permissions-fs`
and the workspace tier bound it like every other read.

This is front-end code and not a seam method, because *which* files were named
is a question that only a front end is asked. The store below it takes bytes and
has no part in that question.

@module ph_app.attach
"""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ph.keys import ATTACHMENTS
from ph.llm.types import (
    AttachmentRef,
    ContentBlock,
    MediaBlock,
    Message,
    PluginSource,
    TextBlock,
    create_user_message,
)

if TYPE_CHECKING:  # pragma: no cover - a type, not a dependency
    from ph.cordis import Context

    from .daemon.client import DaemonClient
from . import verbs
from .params import PutAttachmentParams, StageParams

__all__ = ["AttachmentUnavailable", "Tray", "ingest", "prompt_message", "stage_bytes"]


class AttachmentUnavailable(RuntimeError):
    """Files were attached to a profile that mounts no attachment store.

    This is loud and not quiet. A user who passed `--attach` and got a plain
    text turn could not tell that their diagram was never sent. The purpose of
    this row is that such a thing is never silent.
    """


async def ingest(ctx: Context, paths: Sequence[Path | str]) -> tuple[AttachmentRef, ...]:
    """Store each named file and return the references that a message will carry.

    The order is the order that the user gave, because that is the order in
    which they will describe the files.
    """
    if not paths:
        return ()
    store = ctx.get(ATTACHMENTS)
    if store is None:
        raise AttachmentUnavailable(
            "this profile mounts no attachment store, so files cannot be attached"
        )
    return tuple([await store.save_path(path) for path in paths])


@dataclass(slots=True)
class Tray:
    """Attachments that wait for the next prompt: the tray of the composer.

    **It is keyed by digest, so one file that is staged twice is one chip.**
    That makes `session/stage` safe to retry without an idempotence key. A
    client that reconnects and sends again cannot double a file, for the same
    reason that `attachment/put` cannot double a blob: the identity is the
    content.

    One class serves the two places where a tray lives, so the stage and drain
    rule has one author. The first place is a daemon root, whose tray every
    attached front end shares. The second is `DaemonSession` in
    `ph_app.tui.remote`, which keeps a mirror of that tray. `take` drains: a
    staged attachment goes with *one* prompt, and to leave it would silently
    attach the same file again to every later turn.
    """

    _refs: dict[str, AttachmentRef] = field(default_factory=dict)

    def stage(self, ref: AttachmentRef) -> list[AttachmentRef]:
        self._refs[ref.attachment_id] = ref
        return self.refs

    def take(self) -> list[AttachmentRef]:
        refs = self.refs
        self._refs.clear()
        return refs

    @property
    def refs(self) -> list[AttachmentRef]:
        return list(self._refs.values())

    def __bool__(self) -> bool:
        return bool(self._refs)

    def __len__(self) -> int:
        return len(self._refs)


def prompt_message(
    text: str, attachments: Sequence[AttachmentRef] = (), *, source: PluginSource | None = None
) -> Message:
    """One `user/message` that carries what the user typed and what they attached.

    The text comes first, then the media. That is the order in which a person
    writes: "look at this" before the thing to look at. With nothing attached,
    the message is identical to what `agent.prompt` builds. A front end can
    therefore use this for every prompt, and need not branch on whether the user
    passed a file.

    `source` is for a prompt that the user did not just send, such as the prompt
    of a schedule (`scheduled_source`). Its words are the user's, and its moment
    is not.
    """
    content: list[ContentBlock] = [TextBlock(text=text)] if text else []
    content.extend(MediaBlock(attachment=one) for one in attachments)
    return create_user_message(
        content=content, source={"kind": "user"} if source is None else source
    )


async def stage_bytes(
    client: DaemonClient, session_id: str, name: str, mime: str, content: bytes | bytearray
) -> AttachmentRef:
    """Store these bytes on the daemon and put them on the tray of the session.

    This is two frames, and the pair belongs together. `attachment/put` is not a
    `MUTATIONS` row: it is content-addressed, so a retry is already a no-op and
    its reply *is* the reference. `session/stage` is keyed, because a tray is
    state. Neither half is useful alone, and both front ends that stage a file
    send the two frames back to back.

    It is here and not on `DaemonClient`, because the docstring of
    `daemon/client.py` says that every method there is one frame. It is also not
    a third daemon method. Such a method would have to collapse two verbs that
    the protocol keeps apart for the reasons above. It would also move the
    *name* and the *type* of a file onto the daemon, and those are the half of
    the decision that belongs to the human door (I-9).

    Returns the reference that the daemon minted. That is the same object that
    every other pH client reads, and not the one field that a particular caller
    happened to want, so a browser and the log describe one blob in the same
    way. The tray that it landed on reaches every front end as `session.staged`.

    `bytearray` is accepted because a streaming reader builds one. To copy it to
    `bytes` for a call that only base64-encodes it is a second 5 MiB that nobody
    asked for.
    """
    put = await client.call(
        verbs.ATTACHMENT_PUT,
        PutAttachmentParams(
            session_id=session_id,
            name=name,
            mime=mime,
            content_b64=b64encode(content).decode(),
        ),
    )
    await client.mutate(
        verbs.SESSION_STAGE, StageParams(session_id=session_id, attachment=put.attachment)
    )
    return put.attachment
