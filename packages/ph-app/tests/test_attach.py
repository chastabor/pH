"""P7-01 — attaching a file, end to end.

The row's gate, and until there was a producer it could not be run at all: *a
file attached from outside the repo reaches an accepting route and is
refused-with-a-pointer on one that cannot take it.* Everything below drives the
real one-shot path — `ph -p "…" --attach <file>` — rather than building a
`MediaBlock` by hand, because the half that was missing was precisely the half
that turns a path a person typed into one.

The second theme is that nothing about media is ever silent. A route that will
not take a file says so to the model (a pointer), to the log (`attachment/
degraded`) and to the person (a transcript row) — three readers, one fact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ph.llm.types import attachment_of, text_of
from ph.session import Session, thaw_json
from ph_app.attach import AttachmentUnavailable, ingest, prompt_message
from ph_app.modes import run_print
from ph_app.profiles import compose_profile

pytestmark = pytest.mark.anyio

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 64


def _attach(tmp_path: Path, name: str = "diagram.png", body: bytes = PNG) -> Path:
    """A file *outside* any repo the harness might be pointed at."""
    source = tmp_path / "elsewhere" / name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(body)
    return source


def _media_blocks(session: Session) -> list[Any]:
    return [
        attachment
        for message in session.derive_messages()
        for block in message.content
        if (attachment := attachment_of(block)) is not None
    ]


# ------------------------------------------------------------ the whole path --


async def test_the_one_shot_mode_accepts_an_attachment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ph -p "…" --attach <file>` runs, which is the entry point the row's gate
    names. What the model actually received is asserted below, where the mounted
    context is still reachable."""
    monkeypatch.setenv("PH_HOME", str(tmp_path))

    result = await run_print(
        compose_profile("headless"),
        "what is this?",
        provider="fake",
        model="fake-1",
        session_id="attached",
        attachments=[_attach(tmp_path)],
    )

    assert result.text


async def test_the_attachment_is_stored_and_carried_on_the_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two halves of one claim: the bytes are in the store, and the *message*
    the model was sent references them."""
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    from ph_app.runtime import prompted

    source = _attach(tmp_path)
    async with prompted(
        compose_profile("headless"),
        "what is this?",
        provider="fake",
        model="fake-1",
        session_id="carried",
        attachments=[source],
    ) as (ctx, session):
        (ref,) = _media_blocks(session)
        assert ref.name == "diagram.png"
        assert ref.mime == "image/png"
        assert await ctx.attachments.load_bytes(ref) == PNG
        # The *loop* built a message carrying the block — which is the half this
        # row adds. What the fake route then received is a pointer, because it
        # declares no media; that is `media-degrade`'s doing and is asserted
        # separately below.
        (request,) = [one for one in ctx.llm_fake.requests if one.is_loop_request]
        assert not any(attachment_of(block) is not None for block in request.messages[0].content)
        assert "was not sent" in text_of(request.messages[0].content)


async def test_the_same_file_attached_twice_stores_one_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Content addressing, through the front door this time."""
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    from ph_app.runtime import prompted

    first, second = _attach(tmp_path, "a.png"), _attach(tmp_path, "b.png")
    async with prompted(
        compose_profile("headless"),
        "these two",
        provider="fake",
        model="fake-1",
        session_id="twice",
        attachments=[first, second],
    ) as (ctx, _session):
        assert len(list(ctx.attachments.root.iterdir())) == 1


# ----------------------------------------------------------------- refusing --


async def test_a_route_that_cannot_read_it_says_so_three_ways(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never silent — to the model, to the log, and to the person.

    The fake adapter declares no media at all, which is the honest default for a
    route that has not said otherwise, so this is also the path any text-only
    provider takes.
    """
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    from ph_app.runtime import prompted

    source = _attach(tmp_path)
    async with prompted(
        compose_profile("headless"),
        "what is this?",
        provider="fake",
        model="fake-1",
        session_id="refused",
        attachments=[source],
    ) as (_ctx, session):
        (degraded,) = [one for one in session.events if one.type == "attachment/degraded"]
        assert list(degraded.data["attachmentIds"]) == [_media_blocks(session)[0].attachment_id]
        assert "does not accept" in json.dumps(thaw_json(degraded.data))
        # Ignorable: the model read a pointer, and that rides the `user/message`
        # it was already carried by.
        assert degraded.ignorable


async def test_the_refusal_is_recorded_once_not_once_per_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every request re-derives the same history, so a refused attachment is
    refused again on every step for the life of the session. One event."""
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    from ph.agent.types import AgentOptions
    from ph_app.runtime import mounted

    source = _attach(tmp_path)
    async with mounted(compose_profile("headless")) as ctx:
        session = ctx.sessions.create("repeat")
        refs = await ingest(ctx, [source])
        agent = ctx.agents.create(session, AgentOptions(provider="fake", model="fake-1"))
        agent.followup(prompt_message("look", refs))
        await agent.run()
        await agent.prompt("and again")
        await agent.prompt("and again")

        events = [one for one in session.events if one.type == "attachment/degraded"]
        assert len(events) == 1, "one notice per distinct refusal, not one per request"


# ------------------------------------------------------------------ the door --


async def test_a_profile_without_a_store_refuses_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A person who passed `--attach` and got a plain text turn would have no
    way to tell their file was never sent."""
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    from ph.cordis import Context

    with pytest.raises(AttachmentUnavailable):
        await ingest(Context(), [_attach(tmp_path)])


async def test_nothing_attached_builds_the_message_prompt_always_built(
    tmp_path: Path,
) -> None:
    """`prompted` uses this uniformly rather than branching, which is only safe
    if the no-attachment case is byte-for-byte what `agent.prompt` produces."""
    message = prompt_message("hello")

    assert [block.type for block in message.content] == ["text"]
    assert message.source.kind == "user"
