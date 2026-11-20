"""The mirror test (D7, D4): two protocol definitions that must not drift.

`ph_runtime.protocol` and `ph_rlm.kernel.protocol` are written independently
where they must be, because the guest runs in a venv that must not contain the
harness. For those parts, only this file stops the two from diverging. A field
that is added on one side and forgotten on the other would not raise anywhere.
The frame would be ignored, and the feature would be missing in a way that no
other test can see.

## What is no longer mirrored, and why the rest still is

**The five constants are one declaration.** `PROTOCOL_VERSION`, `PROTOCOL_FD`,
`FD_ENV`, `FRAME_BYTES_ENV` and `NAMESPACE_ENV` are declared in the guest and
*imported* by the host. The import is legal in that direction only, and it costs
nothing, because `ph-rlm` already depends on `ph-runtime-guest`. The constants
were pure data with no typing benefit, and a number that an assertion holds
equal is a number that can be edited on one side.

**The frame tables are not one declaration, and should not be.** Since P8-07
the host derives its `FieldSpec` table from `TypedDict`s, for `mypy`. A
`TypedDict` cannot be derived from a runtime dict, so the typed frames of the
host are a second declaration that cannot be removed. This file ties them to
the wire vocabulary. Without it, nothing would connect them to the frames that
they describe.

**The truncation marker is not one declaration either, for a different
reason.** The host re-exports `ph.text.truncation_marker`. The guest keeps a
deliberate copy, because it may not import ph-core at all. There are two
implementations, and they are compared byte for byte, because a reader who
compares a transcript with a log must not find two different sentences for the
same event. A shared declaration could never compare two *implementations*.

**None of this makes the two sides agree at runtime.** They import from
different venvs. The staleness marker and the refusal at `boot-ack` cover that
case.

## Why `FRAME_FIELDS` is the only declaration on each side

The guest module also carried a `TypedDict` for each frame. Nothing consumed
them at runtime. Under `from __future__ import annotations`, their
`__required_keys__` cannot see `NotRequired`, so they reported `namespaceId` as
**required** while `FRAME_FIELDS` had it as optional. A third copy that no test
compared had already drifted, and that is the argument against keeping one as
documentation.

**The host now has `TypedDict`s again, and the argument still holds, because
they are not a copy.** Since P8-07 the inbound `FieldSpec` table of the host is
*derived* from its frame types through
`get_type_hints(..., include_extras=True)`. That call resolves the strings and
keeps `NotRequired`, which `__required_keys__` does not. On the host there is
therefore one declaration, and the table is a view of it.
`test_inbound_specs_are_the_frame_types_field_for_field` holds the derivation
equal to the table of the guest. If the derivation ever read `NotRequired`
wrongly, the optional fields of `done` would come out as required, and the
mirror would break here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ph_rlm.kernel import protocol as host
from ph_rlm.kernel.manager import KernelLimits
from ph_runtime import protocol as guest

SHARED_CONSTANTS = (
    "PROTOCOL_VERSION",
    "PROTOCOL_FD",
    "FD_ENV",
    "FRAME_BYTES_ENV",
    "NAMESPACE_ENV",
)


def test_the_host_imports_the_guests_constants_rather_than_declaring_them() -> None:
    """The five constants are now one declaration, so this test checks the *route*.

    The test is stated **positively**: the import is there, and it names all
    five. The negative form is the wrong test for two reasons.

    First, a comparison of values cannot see the difference at all. CPython
    interns small ints and identifier-like strings, so a copied
    `PROTOCOL_FD = 3` or `FD_ENV = "PH_RUNTIME_FD"` satisfies both `==` and
    `is` against the constants of the guest.

    Second, the re-declaration that the source *could* show never reaches
    pytest. `ruff` refuses it as `F811 Redefinition of unused`, and `mypy`
    refuses it as "Cannot redefine an existing name as final" (the five in the
    guest are `Final`). Both run before the suite in `test.sh`.

    No gate catches one case: someone deletes the import and declares the five
    again in a clean module. Nothing conflicts and nothing is redefined. Only
    this test covers that drift, and an assertion on the import is the direct
    way to state it. The test reads `node.module`, so an import of the same
    names from another module fails too.

    Sabotage: delete the `from ph_runtime.protocol import ...` line and re-add
    `PROTOCOL_FD: Final = 3` beside `HOST_FRAMES`.
    """
    # Use `.body` and not `ast.walk`. The claim is about the namespace of this
    # module. A walk would also fail on a future *local* named `FD_ENV` inside
    # some function, which is the usual way that a negative test on source text
    # becomes stale.
    module = ast.parse(Path(host.__file__).read_text(encoding="utf-8"))
    imported = {
        alias.asname or alias.name
        for node in module.body
        if isinstance(node, ast.ImportFrom) and node.module == guest.__name__
        for alias in node.names
    }
    missing = sorted(set(SHARED_CONSTANTS) - imported)
    assert not missing, f"the host must import {missing} from {guest.__name__}, not declare them"
    # The host module also exposes them under the names that its importers use.
    for name in SHARED_CONSTANTS:
        assert getattr(host, name) == getattr(guest, name)


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
    ("dropped", "cap"),
    [(0, 65_536), (1, 65_536), (12_345, 65_536), (2, 8), (10**6, 10**6), (None, 65_536), (None, 8)],
)
def test_the_truncation_marker_is_byte_identical(dropped: int | None, cap: int) -> None:
    assert host.truncation_marker(dropped, cap) == guest.truncation_marker(dropped, cap)


def test_the_vendored_json_is_byte_identical_to_ph_core_s() -> None:
    """Compare every definition in `ph_runtime/_json.py` with `ph/json.py`,
    character for character.

    The guest cannot import ph-core: `dill` is its only dependency, and that
    module ships in the ph-core wheel. The guest therefore carries a copy. The
    test checks identity and not behavior. Identity is the stronger assertion
    and is no more work. A behavioral test checks the cases that someone thought
    to parametrize. This test checks every line, including the lines that a
    future edit adds to `as_str` and forgets to copy across.

    That matters for the reason that the docstring of this file gives: a third
    copy that no test compared had already drifted.

    **The comparison is definition by definition, not file against file.** The
    copy holds only the narrowings that the guest imports, so there is no
    whole-file comparison to make. The test reads the names from the copy itself
    and does not declare them here, so a second definition that is pasted in is
    picked up. The `<=` below catches a definition that `ph.json` does not have.
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


def test_the_vendored_characters_are_ph_core_s_word_for_word() -> None:
    """Compare `ph_runtime/_characters.py` with `ph/characters.py`: everything
    below the docstring of each module.

    The guest cuts what a cell prints at a byte cap. That cut must fall between
    characters by the same rule as the cuts of the host, so the guest carries a
    copy of the rule. The rule is tables as well as functions. The test
    therefore compares the whole module and does not use `_definitions`, because
    a table that drifted (a mark added to `_EXTENDING_LETTERS` on one side) is
    as much a drift as a function.
    """
    from ph import characters as host

    guest_path = Path(guest.__file__).with_name("_characters.py")
    host_code = _below_docstring(Path(host.__file__).read_text(encoding="utf-8"))
    guest_code = _below_docstring(guest_path.read_text(encoding="utf-8"))

    assert guest_code == host_code, (
        "the vendored `_characters.py` has drifted — copy `ph/characters.py` across "
        "again, below the guest's docstring"
    )


def _below_docstring(source: str) -> str:
    """The text of a module after its docstring."""
    first = ast.parse(source).body[0]
    return "".join(source.splitlines(keepends=True)[first.end_lineno :])


def _definitions(source: str) -> dict[str, str]:
    """Each top-level function, by name, as the exact text that defines it.

    The source is parsed and not split on a marker, because the copy is a
    subset. No line in `ph.json` says "everything below here was taken", and a
    parser knows where one definition ends without a marker.
    """
    tree = ast.parse(source)
    return {
        node.name: ast.get_source_segment(source, node) or ""
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def test_the_host_models_carry_every_declared_outbound_field() -> None:
    """The outbound field sets of the host are derived from its models, and are
    not typed twice.

    This test therefore asserts the derivation itself. A model that lost its
    `type` default would silently register under the wrong key.
    """
    for model in (host.BootFrame, host.RunFrame, host.ReplyFrame, host.RestoreFrame):
        name = model.model_fields["type"].default
        assert name in host.FRAME_FIELDS
        required, optional = host.FRAME_FIELDS[str(name)]
        aliases = {info.alias or field for field, info in model.model_fields.items()}
        assert required | optional == aliases


def test_boot_requires_every_limit() -> None:
    """The host owns every default, so the guest has nothing to guess (D3).

    If a limit became optional here, there would be two answers to "what is the
    cap", and the answer that applied would depend on which side was older.
    """
    required, _ = host.FRAME_FIELDS["boot"]
    # The set is derived from the model and not listed. This applies the
    # argument of `KernelLimits` to its test: a limit that is added there
    # reaches the row config and the `boot` frame, and is not typed again in
    # either. A hand-written set under `<=` is a copy that fails at nothing.
    # `idleCpuSeconds` was added to six places and missed here, and this
    # assertion continued to pass.
    declared = {info.alias or name for name, info in KernelLimits.model_fields.items()}
    assert declared <= required, sorted(declared - required)


def test_inbound_specs_are_the_frame_types_field_for_field() -> None:
    """The `INBOUND` of the host is derived from its `TypedDict`s, and the table
    of the guest is hand-written. If the two agree field for field, including
    which fields are optional, that proves that the derivation read
    `NotRequired`. The `__required_keys__` of a class cannot do that under
    postponed annotations."""
    for frame in sorted(host.GUEST_FRAMES):
        required, optional = guest.FRAME_FIELDS[frame]
        specs = host.INBOUND[frame]
        assert {spec.name for spec in specs if spec.required} == required, frame
        assert {spec.name for spec in specs if not spec.required} == optional, frame
        # The tag of every frame is declared, and it is the field that the codec
        # selects on.
        assert any(spec.name == "type" and spec.kind == "str" for spec in specs), frame


def test_the_functional_frame_carries_dshs_keyword_named_field() -> None:
    """`call` says `global`, which a class body cannot spell. The functional form
    must still put it in the spec, or the codec would strip the target of every
    call."""
    assert "global" in {spec.name for spec in host.INBOUND["call"]}
    assert host.FRAME_FIELDS["call"] == guest.FRAME_FIELDS["call"]
