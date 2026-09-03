"""P5-14/P7-01 — attaching a file to a harness that is somewhere else.

`ph -p --attach` reads a path with the harness's own permissions, which works
because the harness is in the same process as the person. Over a socket neither
half of that holds: the daemon's filesystem is not the person's, and a browser
tab has no filesystem at all. So the client reads the bytes and the daemon takes
content — I-9's human door, moved to the only side that can open it.

Two rules carry the file:

**A reference this deployment cannot resolve is refused, never dropped.** The
failure P7-01 exists to end is the quiet one: a person attaches a diagram, the
turn goes out as plain text, and nothing says the picture was never sent.

**Staging is shared and is not in the log.** A chip only the uploader can see is
a composer nobody else can reason about; an un-submitted attachment is no more an
act in the session than an un-submitted sentence is.
"""

from __future__ import annotations

from base64 import b64encode
from typing import Any

import pytest
from daemon_helpers import running, until

from ph.llm.types import AttachmentRef
from ph_app.protocol import DaemonError

pytestmark = pytest.mark.anyio

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000100000000f0802000000"
    "9a76829d0000000a49444154789c6360000002000100ff03cf0e0e0000"
    "000049454e44ae426082"
)
"""A real 16x15 PNG, built byte by byte for the reason `test_dimensions.py`
does: a fixture loaded from disk makes a wrong header read look like a missing
file, and the store measures dimensions out of these bytes."""


async def _put(client: Any, session_id: str, *, name: str = "diagram.png") -> dict[str, Any]:
    reply = await client.call(
        "attachment/put",
        sessionId=session_id,
        name=name,
        mime="image/png",
        contentB64=b64encode(PNG).decode(),
    )
    return dict(reply["attachment"])


# ------------------------------------------------------------------ storing --


async def test_the_client_sends_content_and_gets_back_a_reference(tmp_path: Any) -> None:
    """The daemon never learns a path, and answers with what a message carries.

    The reference is the whole point of the round trip: the client holds bytes,
    the deployment holds the blob, and what travels afterwards is a digest.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("attached")
        client = await daemon.client()

        wire = await _put(client, root.id)

        ref = AttachmentRef.model_validate(wire)
        assert ref.attachment_id.startswith("sha256:")
        assert ref.mime == "image/png" and ref.bytes == len(PNG)
        assert ref.name == "diagram.png"
        # Measured from the header by the store, with no image library involved.
        assert (ref.width, ref.height) == (16, 15)
        assert root.ctx.attachments.exists(ref)


async def test_the_same_file_twice_is_one_blob(tmp_path: Any) -> None:
    """Content-addressed, so a second put is a cheap way to learn the reference.

    Which is what makes two people dropping the same screenshot into one session
    cost one file, and what lets a client re-put rather than remember.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("twice")
        client = await daemon.client()

        first = await _put(client, root.id)
        second = await _put(client, root.id, name="same-picture.png")

        assert first["attachmentId"] == second["attachmentId"]
        stored = list(root.ctx.attachments.root.iterdir())
        assert len(stored) == 1, f"one blob expected, found {stored}"


async def test_a_file_too_large_for_a_frame_is_refused_by_name(tmp_path: Any) -> None:
    """Named, because the caller's next move is specific.

    Not "too big to attach" but "too big to attach *in one frame*" — and the
    limit is in the sentence, so a client can say so rather than guess. Chunked
    upload is the fix and it is not built (§5 rule 6); the alternative to this
    refusal is a framing error with no name, raised after the bytes were sent.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("oversized")
        client = await daemon.client()

        with pytest.raises(DaemonError) as refused:
            await client.call(
                "attachment/put",
                sessionId=root.id,
                name="huge.bin",
                mime="application/octet-stream",
                contentB64=b64encode(b"x" * (6 * 1024 * 1024)).decode(),
            )

        assert refused.value.reason == "attachment_too_large"
        assert "5 MiB" in str(refused.value)


# ------------------------------------------------------------- the prompting --


async def test_a_prompt_over_the_socket_carries_its_attachment(tmp_path: Any) -> None:
    """The gate this increment exists for: the model actually receives the file.

    Asserted on the `user/message` the log kept, because that is what
    `derive_messages` sends — a reply that merely said "accepted" would pass for
    a daemon that dropped the reference on the floor.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("carried")
        client = await daemon.client()
        wire = await _put(client, root.id)

        await client.call("session/prompt", sessionId=root.id, prompt="look", attachments=[wire])

        # `followup` queues; the root's own task is what drives the turn that
        # commits the message, so the log is where the wait belongs.
        await until(
            lambda: root.session.latest("user/message") is not None, what="the prompt to be logged"
        )
        message = root.session.latest("user/message")
        assert message is not None
        content = message.data["content"]
        assert [one["type"] for one in content] == ["text", "media"]
        assert content[1]["attachment"]["attachmentId"] == wire["attachmentId"]


async def test_an_attachment_this_deployment_never_stored_is_refused(tmp_path: Any) -> None:
    """Refused, not dropped — the silent failure P7-01 exists to end.

    A reference from another machine, or to a blob a `gc` took, would otherwise
    send the turn as plain text with nothing saying the picture never went.

    Sabotage: skip the `exists` check, and the prompt succeeds carrying a media
    block whose bytes nothing can resolve.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("stale")
        client = await daemon.client()
        elsewhere = AttachmentRef(
            attachment_id="sha256:" + "0" * 64, mime="image/png", bytes=10, name="ghost.png"
        )

        with pytest.raises(DaemonError) as refused:
            await client.call(
                "session/prompt",
                sessionId=root.id,
                prompt="look",
                attachments=[elsewhere.to_wire()],
            )

        assert refused.value.reason == "attachment_unknown"
        assert root.session.latest("user/message") is None, "nothing was sent"


# ---------------------------------------------------------------- the staging --


async def test_a_staged_attachment_rides_the_next_prompt_from_any_client(
    tmp_path: Any,
) -> None:
    """One tray, one conversation — the multiplex rule applied to the composer.

    The person who dropped the file and the person who pressed enter need not be
    the same person, because they are looking at one session. A per-connection
    tray would make the second one send a prompt whose chip they could see and
    whose file they could not attach.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("shared-tray")
        uploader = await daemon.client()
        sender = await daemon.client()
        wire = await _put(uploader, root.id)

        await uploader.call("session/stage", sessionId=root.id, attachment=wire)
        await sender.call("session/prompt", sessionId=root.id, prompt="what is this")

        await until(
            lambda: root.session.latest("user/message") is not None, what="the prompt to be logged"
        )
        message = root.session.latest("user/message")
        assert message is not None
        content = message.data["content"]
        assert content[1]["attachment"]["attachmentId"] == wire["attachmentId"]


async def test_staging_reaches_every_attached_front_end(tmp_path: Any) -> None:
    """A chip only the uploader can see is a composer nobody else can reason about.

    Sabotage: return the staged list without publishing it, and a second UI shows
    an empty composer for a session that is about to send a file.
    """
    async with running(tmp_path) as daemon:
        seen: list[dict[str, Any]] = []
        watcher = await daemon.client(
            on_notify=lambda method, params: (
                seen.append(params) if method == "session.staged" else None
            )
        )
        root = await daemon.root("broadcast")
        await watcher.call("session/attach", sessionId=root.id)
        uploader = await daemon.client()
        wire = await _put(uploader, root.id)

        await uploader.call("session/stage", sessionId=root.id, attachment=wire)

        await until(lambda: bool(seen), what="a session.staged notification")
        assert seen[0]["staged"][0]["attachmentId"] == wire["attachmentId"]


async def test_a_staged_attachment_rides_one_prompt_and_not_the_next(
    tmp_path: Any,
) -> None:
    """Drained, not copied.

    Leaving the tray filled would silently re-attach the same file to every later
    turn — which costs tokens on every step and makes the model answer about a
    picture nobody mentioned again.

    Sabotage: read `root.staged` without clearing it, and the second prompt
    carries the file too.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("drained")
        client = await daemon.client()
        wire = await _put(client, root.id)
        await client.call("session/stage", sessionId=root.id, attachment=wire)

        await client.call("session/prompt", sessionId=root.id, prompt="first")
        await client.call("session/prompt", sessionId=root.id, prompt="second")

        def logged() -> list[Any]:
            return [one for one in root.session.events if one.type == "user/message"]

        await until(lambda: len(logged()) == 2, what="both prompts to be logged")
        prompts = logged()
        kinds = [[block["type"] for block in one.data["content"]] for one in prompts]
        assert kinds == [["text", "media"], ["text"]]
        assert not root.staged, "the tray is empty after the prompt that took it"


async def test_the_tray_is_not_in_the_log(tmp_path: Any) -> None:
    """Un-submitted intent is not an act in the session (§5 rule 6).

    The same rule that keeps a half-typed prompt off the log. What it costs is
    stated where it is decided: a daemon that stops loses the staging while the
    *blob* remains, so the price is re-dropping the file rather than re-finding
    it.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("unlogged")
        client = await daemon.client()
        wire = await _put(client, root.id)

        await client.call("session/stage", sessionId=root.id, attachment=wire)

        assert root.staged, "it really is staged"
        assert [one.type for one in root.session.events] == [], "and the log says nothing"


async def test_staging_the_same_file_twice_is_one_chip(tmp_path: Any) -> None:
    """Idempotent by construction, for the same reason `attachment/put` is.

    `session/stage` takes no idempotence key, and a client that reconnects
    cannot know whether its last one landed. The tray is keyed by digest, so the
    retry is a no-op rather than a second copy of the same file on the prompt.

    Sabotage: make the tray a list, and the prompt carries the picture twice.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("retried")
        client = await daemon.client()
        wire = await _put(client, root.id)

        await client.call("session/stage", sessionId=root.id, attachment=wire)
        again = await client.call("session/stage", sessionId=root.id, attachment=wire)

        assert len(again["staged"]) == 1
        assert len(root.staged) == 1
