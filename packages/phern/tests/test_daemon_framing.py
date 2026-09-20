"""The wire's own limit, and that it does not depend on the kernel under it.

`MAX_LINE` is documented as "how long one frame may be", and both ends of the
wire derive `MAX_ATTACHMENT_BYTES` from it. It was not actually a cap: anyio's
`receive_until` searches for the delimiter *before* it tests the buffer against
`max_bytes`, so an over-length frame slipped through whenever the chunk carrying
its newline arrived while the buffer was still under the limit — which is a
property of `net.local.stream.recvspace`, 64 KiB on Linux and 8 KiB on macOS.

The same 6 MiB attachment was therefore accepted on Linux and closed the
connection on macOS, and a test written to pin the *named* refusal passed on one
kernel while proving the opposite on the other (P6-41). The chunk size is the
parameter here for exactly that reason: the refusal has to be the same on both.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import anyio
import pytest

from ph_app.daemon.duplex import Peer
from ph_app.daemon.framing import MAX_LINE, FramingError, read_frames

pytestmark = pytest.mark.anyio

LINUX_CHUNK = 64 * 1024
"""What a Linux unix socket hands over per read — the size that hid the defect."""

MACOS_CHUNK = 8 * 1024
"""What a macOS unix socket hands over. `net.local.stream.recvspace`, measured."""


class _Chunked:
    """A byte stream that delivers `payload` in fixed-size pieces.

    The kernel's chunking, made a parameter. A socketpair would exercise the real
    thing but not let a test *choose* the size, which is the one variable the
    defect turned on.
    """

    def __init__(self, payload: bytes, chunk: int) -> None:
        self._payload = payload
        self._chunk = chunk
        self._at = 0

    async def receive(self, max_bytes: int = 65536) -> bytes:
        if self._at >= len(self._payload):
            raise anyio.EndOfStream
        size = min(self._chunk, max_bytes)
        piece = self._payload[self._at : self._at + size]
        self._at += len(piece)
        return piece


async def _frames(payload: bytes, chunk: int) -> list[dict[str, object]]:
    stream: AsyncIterator[dict[str, object]] = read_frames(_Chunked(payload, chunk))  # type: ignore[arg-type]
    return [frame async for frame in stream]


@pytest.mark.parametrize("chunk", [MACOS_CHUNK, LINUX_CHUNK], ids=["macos", "linux"])
async def test_a_frame_over_the_limit_is_refused_whatever_the_chunk_size(chunk: int) -> None:
    """**The gate.** One oversized frame, two kernels' worth of chunking, one answer.

    The newline is present and arrives with the last chunk, which is the case
    `receive_until` alone lets through: it finds the delimiter and never asks
    whether the buffer went over. Under a Linux-sized chunk this used to yield the
    frame; now both raise, with the sentence that names the limit.
    """
    oversized = b"x" * (MAX_LINE + 1) + b"\n"

    with pytest.raises(FramingError) as refused:
        await _frames(oversized, chunk)

    assert str(MAX_LINE) in str(refused.value)


@pytest.mark.parametrize("chunk", [MACOS_CHUNK, LINUX_CHUNK], ids=["macos", "linux"])
async def test_a_frame_at_the_limit_still_arrives(chunk: int) -> None:
    """The other half, for `probe_sandbox`'s reason: a cap that refused everything
    would pass the test above while making the wire useless. The largest frame the
    limit permits is carried, on both chunk sizes."""
    room = MAX_LINE - len(b'{"type":"log","text":""}\n')
    frame = b'{"type":"log","text":"' + b"x" * room + b'"}\n'
    assert len(frame) <= MAX_LINE

    carried = await _frames(frame, chunk)

    assert len(carried) == 1
    assert carried[0]["type"] == "log"


async def test_a_peer_that_closes_mid_frame_ends_the_iteration() -> None:
    """Unchanged by the cap, and stated here because the two outcomes are easy to
    conflate: a truncated frame is not an oversized one. The peer did not send it,
    so there is nothing to refuse and nothing to act on."""
    assert await _frames(b'{"type":"log","text":"half', LINUX_CHUNK) == []


async def test_an_in_flight_slot_is_held_until_the_reply_is_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`IN_FLIGHT` bounds work *outstanding*, not work started (E4).

    The release was a `finally` around the dispatch alone, so the slot came back
    the moment the handler returned and the send that follows it ran unbounded.
    A peer that pipelines requests and stops reading then got every one of them
    dispatched: the outbox filled, each reply parked in `send`, and the tasks and
    frames piled up against a reader that was never going to drain them — which
    is the shape the bound exists to refuse.

    `send` is held open here rather than a real socket being filled, because what
    is under test is the accounting and not the kernel's buffer size.
    """
    released = anyio.Event()

    async def parked_send(self: Peer, frame: object) -> None:
        await released.wait()

    async def answer(method: str, params: dict[str, object]) -> dict[str, object]:
        return {"ok": True}

    monkeypatch.setattr(Peer, "send", parked_send)
    peer = Peer(stream=cast("Any", None), dispatch=answer)
    limit = anyio.Semaphore(1)
    # Taken by the reader before it starts a handler, as `_read` does: `_handle`
    # only ever gives a slot back, so a test that did not take one would be
    # counting the wrong direction.
    await limit.acquire()

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(
            peer._handle, {"jsonrpc": "2.0", "id": 1, "method": "anything", "params": {}}, limit
        )
        await anyio.sleep(0.05)

        assert limit.value == 0, "the slot came back before the reply was out"

        released.set()
    assert limit.value == 1, "and it comes back once the reply has gone"
