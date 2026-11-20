"""The quiet paths of the snapshotter: what it refuses, and what it can still read back.

`NamespaceSnapshotter.changed` is called after every cell. What it emits is the
only account that the host gets of what happened to a variable. Two of its
branches are about *not* reporting something. A test must cover that kind of
behavior, because nothing downstream fails when it breaks:

* A variable that cannot be pickled is reported one time and is then quiet. A
  session that holds a database handle therefore does not emit one identical
  refusal for each cell, forever.
* A variable that is gone is reported as `deleted`. A restore that only omitted
  it would leave the model to find a `NameError` and read it as its own bug.

`restore` has the matching pair in the other direction: standard pickle first,
and `dill` only for bytes that only `dill` could have written.

The kernel tests of `ph-rlm` reached these paths only by accident. The cells of
that suite hold ints and strings, which serialize on the first path every time.
"""

from __future__ import annotations

import base64
import pickle
from typing import Any

import pytest

from ph_runtime.snapshot import NamespaceSnapshotter, restore, serializable_names

CAP = 1 << 20
"""One megabyte: large enough that nothing here causes the size refusal by accident."""


def records_for(
    namespace: dict[str, Any], snapshotter: NamespaceSnapshotter, *, cap: int = CAP
) -> dict[str, Any]:
    """`changed()` keyed by variable name, which is how every assertion reads it."""
    return {
        record["var"]: record
        for record in snapshotter.changed(namespace, protected=set(), max_value_bytes=cap)
    }


def test_an_unpicklable_variable_is_reported_once_and_then_stays_quiet() -> None:
    """One refusal for each *variable*, not one for each cell.

    A socket, a file handle, or a database connection stays in a namespace for
    the rest of the session. A report on every cell would be an endless stream of
    identical records about a variable that does not change, and the host logs
    what it is told.
    """
    snapshotter = NamespaceSnapshotter()
    namespace: dict[str, Any] = {"handle": (item for item in range(3))}

    first = records_for(namespace, snapshotter)
    assert first["handle"]["skipped"], "the generator cannot be pickled and must be reported"

    assert "handle" not in records_for(namespace, snapshotter), "the same refusal repeated"
    assert "handle" not in records_for(namespace, snapshotter), "and again on the third cell"


def test_a_variable_that_becomes_unpicklable_for_a_new_reason_speaks_up_again() -> None:
    """The quiet state is keyed to the reason, not to the name.

    The deduplication above must not hide a *different* refusal for the same
    variable. In that case something really changed, and without a report the
    host would never learn of it.
    """
    snapshotter = NamespaceSnapshotter()
    namespace: dict[str, Any] = {"value": (item for item in range(3))}
    assert records_for(namespace, snapshotter)["value"]["skipped"] == "unpicklable"

    # Picklable, but past the cap: a new reason for the same name.
    namespace["value"] = "x" * 4096
    again = records_for(namespace, snapshotter, cap=64)
    assert again["value"]["skipped"] == "too-large"


def test_a_deleted_variable_is_reported_rather_than_omitted() -> None:
    """A restore must be able to say that a name is gone."""
    snapshotter = NamespaceSnapshotter()
    namespace: dict[str, Any] = {"kept": 1, "goes": 2}
    assert set(records_for(namespace, snapshotter)) == {"kept", "goes"}

    del namespace["goes"]
    after = records_for(namespace, snapshotter)

    assert after["goes"]["skipped"] == "deleted"


def test_an_unchanged_immutable_is_not_re_emitted() -> None:
    """The identity shortcut, which keeps a steady namespace cheap."""
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
    """A function that a cell defined round-trips. The C pickler cannot read it
    back.

    This branch lets a kernel snapshot survive a restart with the helpers of the
    model still defined. It is the reason why `dill` is in the dependency list of
    the guest.
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
    """The wire is JSON from another process, so the shape is not guaranteed."""
    namespace: dict[str, Any] = {}

    outcome = restore(
        namespace,
        [{"var": "nameless"}, {"blob": "x"}, {"var": 7, "blob": "x"}, {}],
    )

    assert outcome == {"restored": [], "failed": []}
    assert namespace == {}


def test_serializable_names_leaves_out_what_a_snapshot_has_no_business_carrying() -> None:
    """Modules, dunder names, and the runtime's own names, which are made again at boot."""
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
