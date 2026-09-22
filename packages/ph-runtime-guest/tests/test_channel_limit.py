"""F8 — the guest's reader is sized by the host, not by a constant of its own.

The host derives its read ceiling from `maxSnapshotBytes` (`frame_cap`), because
one snapshotted variable must fit one frame. The guest kept a fixed 64 MiB, so a
deployment that raised the limit past ~48 MiB — base64 being four bytes per
three — lost the namespace on the way back in: the `restore` frame for a single
variable was a `ValueError` in the guest's reader, and the guest exited.

The limit crosses in the environment because the reader's size is fixed when the
connection opens, before any frame has been read.
"""

from __future__ import annotations

import socket

import pytest

from ph_runtime.channel import MAX_FRAME_BYTES, Channel, frame_limit
from ph_runtime.protocol import FRAME_BYTES_ENV

pytestmark = pytest.mark.anyio

_FRAME = b'{"type": "ping", "pad": "' + b"x" * 8_000 + b'"}\n'
"""A little over eight kilobytes: between the two limits the test sets."""


@pytest.mark.parametrize(("limit", "read"), [(4_096, False), (16_384, True)])
async def test_the_read_limit_is_the_hosts(
    monkeypatch: pytest.MonkeyPatch, limit: int, read: bool
) -> None:
    """Under the host's number a frame is read; over it, the channel ends.

    Sabotage: open the connection with `MAX_FRAME_BYTES` again and the small
    limit reads the frame too.
    """
    monkeypatch.setenv(FRAME_BYTES_ENV, str(limit))
    host, guest = socket.socketpair()
    # `detach`: the channel's socket owns the descriptor from here, and two
    # owners would close it twice.
    channel = await Channel.open(guest.detach())
    try:
        host.sendall(_FRAME)
        frame = await channel.receive()
        assert (frame is not None) is read
    finally:
        host.close()
        await channel.aclose()


@pytest.mark.parametrize("named", [None, "", "0", "-5", "lots"])
def test_without_a_usable_number_the_default_holds(
    monkeypatch: pytest.MonkeyPatch, named: str | None
) -> None:
    """An older host names none; a value that is not a positive integer is not
    trusted, because a zero limit would refuse every frame."""
    if named is None:
        monkeypatch.delenv(FRAME_BYTES_ENV, raising=False)
    else:
        monkeypatch.setenv(FRAME_BYTES_ENV, named)

    assert frame_limit() == MAX_FRAME_BYTES
