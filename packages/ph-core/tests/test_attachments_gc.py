"""P7-01's collection rule — the fold behind `ph attachments gc`.

The rule was written into the plan before anything implemented it: **a blob any
stored log still references must not be collected, however old it is.** Age is
the obvious predicate and it is the wrong one, and what makes it wrong is what
P7-03 changed — after uploads, the local blob is the last copy, because a
provider's lives behind a handle that expires. So collecting a referenced
attachment does not degrade a session, it ends it.

The tests below are that rule from four sides: a reference found however it was
spelled, a reference in an *old* log still keeping a blob, a survey that could not
see the whole store refusing to remove anything, and a freshly written blob left
alone because a person may still be composing the prompt that will mention it.
"""

from __future__ import annotations

from pathlib import Path
from time import time
from typing import Any

import pytest

from ph.cordis import Context
from ph.llm.types import AttachmentRef, MediaBlock, create_user_message
from ph.seams.attachments import (
    LISTING_LIMIT,
    MIN_AGE,
    AttachmentStore,
    collect_attachments,
    digest_of,
    referenced_digests,
    survey_attachments,
)
from ph.seams.uploads import FileHandle, UploadRegistry
from ph.session import Session, SessionHeader, SurfaceIntent, is_surface_eligible_type

pytestmark = pytest.mark.anyio

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 32
OTHER = PNG + b"different"


class _Store:
    """A persistence stub: logs in memory, one of them optionally unreadable.

    Duck-typed against the two methods the fold uses, which is the whole reason
    the fold takes `Any` — a test that had to stand up a JSONL backend to assert a
    reference rule would be testing the backend.
    """

    def __init__(self, *sessions: Session, broken: str = "", truncate: bool = False) -> None:
        self.sessions = {one.id: one for one in sessions}
        self.broken = broken
        self.truncate = truncate

    def stored(self, *, limit: int = 50) -> list[Any]:
        rows = [_Row(one) for one in self.sessions]
        return rows * limit if self.truncate else rows

    def read_own(self, session_id: str, upto: Any = None, family: Any = None) -> Any:
        if session_id == self.broken:
            raise ValueError("torn log")
        session = self.sessions[session_id]
        return session.header, session.events


class _Row:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.modified = 0.0
        self.parent = None


def _session(session_id: str, *events: tuple[str, Any]) -> Session:
    """A log built the way one is written, surface markers included.

    `user/message` is surface-eligible, so appending one without an intent is
    refused — which is the log's own rule and not something a fixture should be
    able to route around, since the fold reads exactly what a stored log holds.
    """
    session = Session(session_id, header=SessionHeader(id=session_id, created_at=1))
    for kind, data in events:
        surface = SurfaceIntent("append") if is_surface_eligible_type(kind) else None
        session.append(kind, data, surface)
    return session


async def _stored(tmp_path: Path, *payloads: bytes) -> tuple[AttachmentStore, list[Any]]:
    store = AttachmentStore(ctx=None, root=tmp_path / "attachments")  # type: ignore[arg-type]
    refs = [await store.save_bytes(content=one, mime="image/png", name="a.png") for one in payloads]
    return store, refs


AGED = time() + 400 * 86_400.0
"""A clock far enough ahead that every blob written now is past `MIN_AGE`.

`survey_attachments(now=)` rather than back-dating each file with `os.utime`:
one mechanism for "pretend time has passed", and it is the one the production
signature already offers.
"""


# ------------------------------------------------------------ the references --


def test_a_reference_is_found_however_the_producer_spelled_it() -> None:
    """Three rows already write an attachment id three different ways.

    `attachment/uploaded` writes `attachmentId`, the media notices write
    `attachmentIds` as a list, and a `user/message` nests one inside a content
    block. Matching on the *value's* shape is what makes the fold independent of
    that — a key list here would be a fourth spelling maintained in a module none
    of them import, and the failure when it fell behind is a blob collected out
    from under a session.
    """
    digest = digest_of(PNG)
    block = MediaBlock(
        attachment=AttachmentRef(attachment_id=digest, mime="image/png", bytes=len(PNG))
    )
    session = _session(
        "s",
        ("user/message", create_user_message(content=[block], source={"kind": "user"}).to_wire()),
        ("attachment/uploaded", {"provider": "p", "attachmentId": digest}),
        ("attachment/degraded", {"provider": "p", "attachmentIds": [digest]}),
    )

    assert referenced_digests(session.events) == {digest}


def test_an_unknown_event_type_still_counts_as_a_reference() -> None:
    """The surface layer means a stored log can carry an event this build does
    not know (A3), and an unknown event that referenced a blob would otherwise
    read as no reference at all."""
    digest = digest_of(PNG)
    session = _session("s", ("some-future-row/kept", {"nested": [{"whatever": digest}]}))

    assert referenced_digests(session.events) == {digest}


# --------------------------------------------------------------- the survey --


async def test_a_referenced_blob_is_never_collected_however_old(tmp_path: Path) -> None:
    """The rule, stated at the age where the wrong predicate would fire.

    A year-old blob in a year-old conversation is exactly what an age-based sweep
    takes first, and it is content the session cannot be opened without.
    """
    store, (kept, dead) = await _stored(tmp_path, PNG, OTHER)
    logs = _Store(_session("old", ("user/message", {"content": [{"a": kept.attachment_id}]})))

    survey = survey_attachments(store, logs, now=AGED)

    assert [blob.digest for blob in survey.kept] == [kept.attachment_id]
    assert [blob.digest for blob in survey.collect] == [dead.attachment_id]
    assert survey.safe and survey.sessions == 1


async def test_a_new_blob_is_left_alone(tmp_path: Path) -> None:
    """The window a staged file lives in.

    A person drops a file on the composer and the bytes are on disk immediately;
    nothing references them until they send the prompt, which may be minutes or an
    afternoon later. `--min-age` never *authorises* collection — it only refuses —
    which is why it is not the age bound the rule forbids.
    """
    store, _refs = await _stored(tmp_path, PNG)

    survey = survey_attachments(store, _Store())

    assert not survey.collect and len(survey.recent) == 1
    assert MIN_AGE == 86_400.0


async def test_a_log_that_will_not_read_stops_the_collection(tmp_path: Path) -> None:
    """Fail closed, because the asymmetry is total.

    A log this build cannot parse may reference any digest in the store, so "no
    session mentions this blob" is a statement the fold is not entitled to make.
    Being wrong cautiously costs disk; being wrong the other way makes a
    conversation permanently unopenable.
    """
    store, _refs = await _stored(tmp_path, PNG)
    logs = _Store(_session("good"), _session("torn"), broken="torn")

    survey = survey_attachments(store, logs, now=AGED)

    assert survey.collect, "the blob is unreferenced by everything that read"
    assert not survey.safe and survey.unreadable == ("torn",)
    assert collect_attachments(survey) == (0, 0)
    assert store.root.joinpath(digest_of(PNG).partition(":")[2] + ".png").exists()


async def test_a_listing_that_was_cut_short_stops_the_collection(tmp_path: Path) -> None:
    """A bounded answer to "does anyone still need this" is not a smaller answer.

    The session below the cut is the old conversation nobody has opened lately,
    which is the one whose pictures are worth most. So the cap is checked rather
    than trusted.
    """
    store, _refs = await _stored(tmp_path, PNG)

    survey = survey_attachments(store, _Store(_session("one"), truncate=True), limit=8, now=AGED)

    assert survey.truncated and not survey.safe
    assert collect_attachments(survey) == (0, 0)
    assert LISTING_LIMIT > 8, "the shipped cap is not a page size"


async def test_a_store_that_cannot_be_listed_collects_nothing(tmp_path: Path) -> None:
    """ "No sessions" and "could not ask" are indistinguishable from here, and only
    one of them makes collection safe."""

    class _Broken:
        def stored(self, *, limit: int = 50) -> list[Any]:
            raise OSError("no store")

    store, _refs = await _stored(tmp_path, PNG)

    survey = survey_attachments(store, _Broken(), now=AGED)

    assert not survey.safe and survey.sessions == 0


async def test_collection_removes_the_blob_and_its_upload_handles(tmp_path: Path) -> None:
    """Both stores, one reference set — what `ctx.uploads` says this command owes.

    An entry is written per `(provider, digest)` ever uploaded and stays after the
    provider forgets the file, so that directory grows with every distinct
    attachment ever sent. Losing one costs an upload, not a conversation, which is
    why they are swept with the blobs and not aged like them.
    """
    store, (dead,) = await _stored(tmp_path, PNG)
    # The real registry rather than a stub with a `root`: the layout is that
    # seam's, and the memo it clears on the way out is the part an outside
    # deleter cannot see.
    uploads = UploadRegistry(ctx=Context(), root=tmp_path / "uploads")
    handle = FileHandle(
        provider="anthropic", attachment_id=dead.attachment_id, handle="file_1", uploaded_at=0
    )
    await uploads._store(handle)
    entry = uploads.path_for("anthropic", dead.attachment_id)

    survey = survey_attachments(store, _Store(_session("empty")), uploads=uploads, now=AGED)
    removed = collect_attachments(survey, uploads)

    assert removed == (1, 1)
    assert not entry.exists()
    assert not list(store.root.iterdir())
    assert uploads.cached("anthropic", dead.attachment_id) is None, "the memo went with it"


async def test_a_file_that_is_not_ours_is_left_alone(tmp_path: Path) -> None:
    """The name *is* the identity, so anything not named for a digest is somebody
    else's file sitting in the directory — a note, a partial copy — and not this
    command's to delete."""
    store, _refs = await _stored(tmp_path, PNG)
    stray = store.root / "README"
    stray.write_text("mine", encoding="utf-8")

    survey = survey_attachments(store, _Store(_session("empty")), now=AGED)
    collect_attachments(survey)

    assert stray.exists()
