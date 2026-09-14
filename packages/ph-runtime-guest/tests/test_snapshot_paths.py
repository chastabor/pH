"""The snapshotter's quiet paths: what it refuses, and what it can still read back.

`NamespaceSnapshotter.changed` is called after every cell, and what it emits is
the only account the host gets of a variable's fate. Two of its branches are
about *not* saying things, which is exactly the kind of behaviour a test has to
pin because nothing downstream fails when it breaks:

* a variable that cannot be pickled is reported once and then stays quiet, so a
  session with a database handle in it does not emit one identical refusal per
  cell forever;
* a variable that has gone is reported as `deleted`, because a restore that
  merely omitted it would leave the model to discover a `NameError` and read it
  as its own bug.

`restore` has the matching pair on the way back: standard pickle first, `dill`
only for what only `dill` could have written.

These were reached by `ph-rlm`'s kernel tests only incidentally — the suite's
cells hold ints and strings, which serialize on the first path every time.
"""

from __future__ import annotations

import base64
import pickle
from typing import Any

import pytest

from ph_runtime.snapshot import NamespaceSnapshotter, restore, serializable_names

pytestmark = pytest.mark.anyio

CAP = 1 << 20
"""A megabyte: large enough that nothing here trips the size refusal by accident."""


def records_for(namespace: dict[str, Any], snapshotter: NamespaceSnapshotter) -> dict[str, Any]:
    """`changed()` keyed by variable name, which is how every assertion reads it."""
    return {
        record["var"]: record
        for record in snapshotter.changed(namespace, protected=set(), max_value_bytes=CAP)
    }


def test_an_unpicklable_variable_is_reported_once_and_then_stays_quiet() -> None:
    """One refusal per *variable*, not one per cell.

    A socket, a file handle or a database connection sits in a namespace for the
    rest of the session. Reporting it every cell would be an unbounded stream of
    identical records about a variable that is not moving — and the host logs
    what it is told.
    """
    snapshotter = NamespaceSnapshotter()
    namespace: dict[str, Any] = {"handle": (item for item in range(3))}

    first = records_for(namespace, snapshotter)
    assert first["handle"]["skipped"], "the generator cannot be pickled and must be reported"

    assert "handle" not in records_for(namespace, snapshotter), "the same refusal repeated"
    assert "handle" not in records_for(namespace, snapshotter), "and again on the third cell"


def test_a_variable_that_becomes_unpicklable_for_a_new_reason_speaks_up_again() -> None:
    """Quiet is keyed to the reason, not to the name.

    The dedupe above must not swallow a *different* refusal for the same
    variable — that is the case where something actually changed and the host
    would otherwise never hear about it.
    """
    snapshotter = NamespaceSnapshotter()
    namespace: dict[str, Any] = {"value": (item for item in range(3))}
    assert records_for(namespace, snapshotter)["value"]["skipped"] == "unpicklable"

    # Picklable, but past the cap: a new reason for the same name.
    namespace["value"] = "x" * 4096
    again = snapshotter.changed(namespace, protected=set(), max_value_bytes=64)
    assert {record["var"]: record["skipped"] for record in again} == {"value": "too-large"}


def test_a_deleted_variable_is_reported_rather_than_omitted() -> None:
    """A restore has to be able to say a name is gone."""
    snapshotter = NamespaceSnapshotter()
    namespace: dict[str, Any] = {"kept": 1, "goes": 2}
    assert set(records_for(namespace, snapshotter)) == {"kept", "goes"}

    del namespace["goes"]
    after = records_for(namespace, snapshotter)

    assert after["goes"]["skipped"] == "deleted"
    assert "kept" not in after, "an unchanged immutable is not re-emitted"


def test_an_unchanged_immutable_is_not_re_emitted() -> None:
    """The identity shortcut, which is what keeps a steady namespace cheap."""
    snapshotter = NamespaceSnapshotter()
    namespace: dict[str, Any] = {"n": 42, "text": "stable"}

    assert set(records_for(namespace, snapshotter)) == {"n", "text"}
    assert records_for(namespace, snapshotter) == {}, "nothing moved, nothing reported"

    namespace["n"] = 43
    assert set(records_for(namespace, snapshotter)) == {"n"}


def test_restore_reads_a_standard_pickle() -> None:
    blob = base64.b64encode(pickle.dumps({"a": 1})).decode("ascii")

    namespace: dict[str, Any] = {}
    outcome = restore(namespace, [{"var": "mapping", "blob": blob}])

    assert outcome == {"restored": ["mapping"], "failed": []}
    assert namespace["mapping"] == {"a": 1}


def test_restore_falls_back_to_dill_for_what_only_dill_could_write() -> None:
    """A cell-defined function round-trips; the C pickler cannot read it back.

    This is the branch that makes a kernel snapshot survive a restart with the
    model's own helpers still defined — the case `dill` is in the guest's
    dependency list for at all.
    """
    dill = pytest.importorskip("dill", reason="the fallback only exists when dill is installed")

    source: dict[str, Any] = {}
    exec("def helper(value):\n    return value * 3", source)  # a cell-defined function
    blob = base64.b64encode(dill.dumps(source["helper"], recurse=True)).decode("ascii")

    namespace: dict[str, Any] = {}
    outcome = restore(namespace, [{"var": "helper", "blob": blob}])

    assert outcome == {"restored": ["helper"], "failed": []}
    assert namespace["helper"](2) == 6, "the restored function does not work"


def test_restore_reports_a_blob_neither_reader_can_take() -> None:
    """Corruption is per-variable, so the rest of the namespace still comes back."""
    good = base64.b64encode(pickle.dumps([1, 2])).decode("ascii")
    junk = base64.b64encode(b"not a pickle at all").decode("ascii")

    namespace: dict[str, Any] = {}
    outcome = restore(namespace, [{"var": "good", "blob": good}, {"var": "bad", "blob": junk}])

    assert outcome["restored"] == ["good"]
    assert outcome["failed"] == ["bad"]
    assert namespace == {"good": [1, 2]}


def test_restore_skips_a_record_that_is_not_shaped_like_one() -> None:
    """The wire is JSON from another process, so the shape is not a given."""
    namespace: dict[str, Any] = {}

    outcome = restore(
        namespace,
        [{"var": "nameless"}, {"blob": "x"}, {"var": 7, "blob": "x"}, {}],
    )

    assert outcome == {"restored": [], "failed": []}
    assert namespace == {}


def test_serializable_names_leaves_out_what_a_snapshot_has_no_business_carrying() -> None:
    """Modules, dunders and the runtime's own names, which are re-made on boot."""
    import json

    namespace: dict[str, Any] = {
        "keep": 1,
        "also_keep": 2,
        "_private": 3,
        "__builtins__": {},
        "json": json,
        "injected": 4,
    }

    assert serializable_names(namespace, protected={"injected"}) == ["also_keep", "keep"]
