"""Files a person named, turned into a message the model can see (P7-01).

The **human door** of the split I-9 draws. A person may attach anything they can
already open, so this reads paths directly, with the harness's own permissions
and no policy check. That is deliberately not available to the model: a tool that
could reach it would be an exfiltration primitive — attach a private key as a
"document" and let the provider OCR it — so a model-initiated attach reads
through `ctx.fs`, where `permissions-fs` and the workspace tier bound it like
every other read.

Front-end code rather than a seam method, because *which* files were named is a
question only a front end is asked. The store below it takes bytes and stays out
of the question entirely.

@module ph_app.attach
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ph.llm.types import AttachmentRef, MediaBlock, Message, create_user_message

__all__ = ["AttachmentUnavailable", "Tray", "ingest", "prompt_message"]


class AttachmentUnavailable(RuntimeError):
    """Files were attached to a profile that mounts no attachment store.

    Loud rather than quiet: a person who passed `--attach` and got a plain text
    turn would have no way to tell their diagram was never sent, and the whole
    point of this row is that such a thing is never silent.
    """


async def ingest(ctx: Any, paths: Sequence[Path | str]) -> tuple[AttachmentRef, ...]:
    """Store each named file and return the references a message will carry.

    Order is the order the person gave, because that is the order they will
    describe them in.
    """
    if not paths:
        return ()
    store = ctx.get("attachments")
    if store is None:
        raise AttachmentUnavailable(
            "this profile mounts no attachment store, so files cannot be attached"
        )
    return tuple([await store.save_path(path) for path in paths])


@dataclass(slots=True)
class Tray:
    """Attachments waiting for the next prompt — the composer's tray.

    **Keyed by digest, so staging one file twice is one chip.** That is what
    makes `session/stage` safe to retry without an idempotence key: a client that
    reconnects and re-sends cannot double a file, for the same reason `attachment/
    put` cannot double a blob — the identity is the content.

    One class for the two places a tray lives — a daemon root, shared by every
    attached front end, and the in-process session that increment 2c deletes —
    so the stage/drain rule has one author. `take` drains: a staged attachment
    rides *one* prompt, and leaving it would silently re-attach the same file to
    every later turn.
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


def prompt_message(text: str, attachments: Sequence[AttachmentRef] = ()) -> Message:
    """One `user/message` carrying what the person typed and what they attached.

    The text first, then the media, which is the order a person writes in — "look
    at this" before the thing being looked at. Identical to what `agent.prompt`
    builds when nothing is attached, so a front end can use this uniformly rather
    than branching on whether the person passed a file.
    """
    content: list[Any] = [{"type": "text", "text": text}] if text else []
    content.extend(MediaBlock(attachment=one) for one in attachments)
    return create_user_message(content=content, source={"kind": "user"})
