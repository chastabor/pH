"""F8: the host sizes the reader of the guest, not a constant in the guest.

The host derives its read limit from `maxSnapshotBytes` (`frame_cap`), because
one snapshotted variable must fit in one frame. The guest used to keep a fixed
64 MiB. A deployment that raised the limit past about 48 MiB then lost the
namespace on restore, because base64 uses four bytes for every three. The
`restore` frame for one variable raised `ValueError` in the reader of the guest,
and the guest exited.

The limit is passed in the environment, because the size of the reader is fixed
when the connection opens, before any frame is read.
"""

from __future__ import annotations

import socket

import pytest

from ph_runtime.channel import MAX_FRAME_BYTES, Channel, frame_limit
from ph_runtime.protocol import FRAME_BYTES_ENV

pytestmark = pytest.mark.anyio

_FRAME = b'{"type": "ping", "pad": "' + b"x" * 8_000 + b'"}\n'
"""A little over eight kilobytes, which is between the two limits that the test sets."""


@pytest.mark.parametrize(("limit", "read"), [(4_096, False), (16_384, True)])
async def test_the_read_limit_is_the_hosts(
    monkeypatch: pytest.MonkeyPatch, limit: int, read: bool
) -> None:
    """A frame under the number from the host is read. With a frame over it, the
    channel ends.

    Sabotage: open the connection with `MAX_FRAME_BYTES` again and the small
    limit reads the frame too.
    """
    monkeypatch.setenv(FRAME_BYTES_ENV, str(limit))
    host, guest = socket.socketpair()
    # `detach`: from here the socket of the channel owns the descriptor. Two
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
    """An older host gives no number. A value that is not a positive integer is
    not trusted, because a limit of zero would refuse every frame."""
    if named is None:
        monkeypatch.delenv(FRAME_BYTES_ENV, raising=False)
    else:
        monkeypatch.setenv(FRAME_BYTES_ENV, named)

    assert frame_limit() == MAX_FRAME_BYTES
