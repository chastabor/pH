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

**The host now has `TypedDict`s again, and the argument still holds — because
they are not a copy.** Since P8-07 the host's inbound `FieldSpec` table is
*derived* from its frame types through `get_type_hints(..., include_extras=True)`
(which resolves the strings and keeps `NotRequired`, where `__required_keys__`
does not), so on the host there is one declaration and the table is a view of
it. `test_inbound_specs_are_the_frame_types_field_for_field` is what holds the
derivation to the guest's table: if it ever read `NotRequired` wrong, `done`'s
optional fields would come out required and the mirror would break here.
"""

from __future__ import annotations

import ast
from pathlib import Path

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


def test_the_vendored_json_is_byte_identical_to_ph_core_s() -> None:
    """Every definition in `ph_runtime/_json.py` against `ph/json.py`, character
    for character.

    The guest cannot import ph-core — `dill` is its only dependency and that
    module ships in the ph-core wheel — so it carries a copy. Identity rather
    than behaviour, which is the stronger assertion and no more work: a
    behavioural test pins the cases somebody thought to parametrize, and this
    pins every line, including the ones a future edit adds to `as_str` and
    forgets to bring across.

    That matters for the reason this file opens with: *"a third copy that no test
    compared had already drifted."*

    **Definition by definition, not file against file.** The copy holds only the
    narrowings the guest imports, so there is no whole-file comparison to make.
    Which names those are is read off the copy itself rather than declared here
    — paste a second one in and this picks it up — and one `ph.json` does not
    define is caught by the `<=` below.
    """
    from ph import json as host

    guest_path = Path(guest.__file__).with_name("_json.py")
    host_defs = _definitions(Path(host.__file__).read_text(encoding="utf-8"))
    guest_defs = _definitions(guest_path.read_text(encoding="utf-8"))
    fix = "copy it across from `ph.json` again"

    assert guest_defs, f"the vendored copy defines nothing — {fix}"
    assert guest_defs.keys() <= host_defs.keys(), (
        f"the vendored copy defines {sorted(guest_defs.keys() - host_defs.keys())}, "
        f"which `ph.json` does not — delete it, or {fix}"
    )
    for name, body in guest_defs.items():
        assert body == host_defs[name], f"the vendored `{name}` has drifted — {fix}"


def _definitions(source: str) -> dict[str, str]:
    """Each top-level function, by name, as the exact text that defines it.

    Parsed rather than split on a marker, because the copy is a subset: there is
    no line in `ph.json` that says "everything below here was taken", and a
    parser knows where one definition ends without being told.
    """
    tree = ast.parse(source)
    return {
        node.name: ast.get_source_segment(source, node) or ""
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


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


def test_inbound_specs_are_the_frame_types_field_for_field() -> None:
    """The host's `INBOUND` is derived from its `TypedDict`s; the guest's table is
    hand-written. Agreeing field for field — including which are optional — is
    the proof the derivation read `NotRequired`, which is exactly what a class's
    `__required_keys__` cannot do under postponed annotations."""
    for frame in sorted(host.GUEST_FRAMES):
        required, optional = guest.FRAME_FIELDS[frame]
        specs = host.INBOUND[frame]
        assert {spec.name for spec in specs if spec.required} == required, frame
        assert {spec.name for spec in specs if not spec.required} == optional, frame
        # Every frame's tag is declared, and it is the field the codec selects on.
        assert any(spec.name == "type" and spec.kind == "str" for spec in specs), frame


def test_the_functional_frame_carries_dshs_keyword_named_field() -> None:
    """`call` says `global`, which a class body cannot spell; the functional form
    must still put it in the spec, or the codec would strip every call's target."""
    assert "global" in {spec.name for spec in host.INBOUND["call"]}
    assert host.FRAME_FIELDS["call"] == guest.FRAME_FIELDS["call"]
