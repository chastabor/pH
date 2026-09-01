"""The mirror test (D7, D4): two protocol definitions that must not drift.

`ph_runtime.protocol` and `ph_rlm.kernel.protocol` are written independently on
purpose — the guest runs in a venv that must not contain the harness — so
nothing but this file stops them diverging. A field added on one side and
forgotten on the other would not raise anywhere: the frame would simply be
ignored, and the feature would be missing in a way no other test can see.

The truncation marker is compared byte for byte because a reader comparing a
transcript to a log must not find two different sentences for the same event.

## Why `FRAME_FIELDS` is the only declaration on each side

The guest module also carried a `TypedDict` per frame. Nothing consumed them at
runtime, and under `from __future__ import annotations` their `__required_keys__`
cannot see `NotRequired` — so they reported `namespaceId` as **required** while
`FRAME_FIELDS` had it optional. A third copy that no test compared had already
drifted, which is the argument against keeping one as documentation.
"""

from __future__ import annotations

import pytest

from ph_rlm.kernel import protocol as host
from ph_runtime import protocol as guest


def test_the_constants_agree() -> None:
    assert host.PROTOCOL_VERSION == guest.PROTOCOL_VERSION
    assert host.PROTOCOL_FD == guest.PROTOCOL_FD
    assert host.FD_ENV == guest.FD_ENV
    assert host.NAMESPACE_ENV == guest.NAMESPACE_ENV


def test_the_frame_vocabularies_agree() -> None:
    assert host.HOST_FRAMES == guest.HOST_FRAMES
    assert host.GUEST_FRAMES == guest.GUEST_FRAMES
    # Every named frame has a field set, and every field set names a frame.
    assert set(host.FRAME_FIELDS) == host.HOST_FRAMES | host.GUEST_FRAMES


@pytest.mark.parametrize("frame", sorted(guest.FRAME_FIELDS))
def test_each_frame_has_the_same_fields_on_both_sides(frame: str) -> None:
    assert frame in host.FRAME_FIELDS, f"the host does not know the frame {frame!r}"
    host_required, host_optional = host.FRAME_FIELDS[frame]
    guest_required, guest_optional = guest.FRAME_FIELDS[frame]
    assert host_required == guest_required, f"{frame}: required fields differ"
    assert host_optional == guest_optional, f"{frame}: optional fields differ"


def test_the_host_declares_no_frame_the_guest_does_not() -> None:
    assert set(host.FRAME_FIELDS) == set(guest.FRAME_FIELDS)


@pytest.mark.parametrize(
    ("dropped", "cap"), [(0, 65_536), (1, 65_536), (12_345, 65_536), (2, 8), (10**6, 10**6)]
)
def test_the_truncation_marker_is_byte_identical(dropped: int, cap: int) -> None:
    assert host.truncation_marker(dropped, cap) == guest.truncation_marker(dropped, cap)


def test_the_host_models_carry_every_declared_outbound_field() -> None:
    """The host's outbound field sets are derived from its models, not typed twice.

    So this asserts the derivation itself: a model whose `type` default went
    missing would silently register under the wrong key.
    """
    for model in (host.BootFrame, host.RunFrame, host.ReplyFrame, host.RestoreFrame):
        name = model.model_fields["type"].default
        assert name in host.FRAME_FIELDS
        required, optional = host.FRAME_FIELDS[str(name)]
        aliases = {info.alias or field for field, info in model.model_fields.items()}
        assert required | optional == aliases


def test_boot_requires_every_limit() -> None:
    """The host owns every default, so the guest has nothing to guess (D3).

    A limit that became optional here would mean two answers to "what is the
    cap", and which applied would depend on which side was older.
    """
    required, _ = host.FRAME_FIELDS["boot"]
    assert {
        "cpuSeconds",
        "addressSpaceBytes",
        "maxLogBytes",
        "maxValueBytes",
        "maxSnapshotBytes",
    } <= required
