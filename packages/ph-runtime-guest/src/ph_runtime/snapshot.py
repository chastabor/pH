"""Snapshots of each variable, guest side (D17).

A snapshot is per *variable*, not per namespace, and that is the whole design. An
unchanged 200 MiB DataFrame emits nothing, because its digest did not change. A
snapshot of the namespace as one blob would append that DataFrame again on every
cell that changed anything. The log would then grow with the *size of the
namespace* and not with the size of the change.

The digest comparison alone saves only the *send*. It does not save the
serialization, which runs after every cell for every name. Three things make
"emits nothing" cheap in practice:

**The C pickler is tried first.** `dill.Pickler` subclasses `pickle._Pickler`,
the pure-Python pickler, so it would serialize a large container in Python
bytecode. The C pickler refuses some values: a function, a class, or a lambda
that a cell defined, because none of them can be imported from `__ph_cell__`.
Those values fall back to `dill`. `dill.loads` reads standard pickle bytes
unchanged. Each object takes the same path every time, so digests stay stable.

**A value that is certainly immutable is compared by identity.** If the object
under a name is the same `bytes`, `str`, or `int` object as after the last cell,
its content cannot have changed, so nothing is serialized. This is deliberately
*not* extended to tuples or frozensets. `t = ([1],)` keeps its identity while its
contents change, and a fast path that is wrong one time is worse than no fast
path.

**A variable over the cap stops at the cap.** The pickler writes into a sink that
raises when it passes `max_value_bytes`. The variable is then reported as
`too-large`, whichever pickler was writing.

Some things are deliberately **not** captured: the binding proxies, the imported
skill modules, and everything else that the bootstrap put in the namespace. They
are rebuilt at boot from the `boot` frame. Pickling them would store a stale copy
of the harness's own surface, and `dill` would serialize a proxy that holds a
closure over a dead channel without complaint.

`dill` is imported lazily. A runtime venv without `dill` still snapshots
everything that the C pickler accepts. That is a degraded session, not a failed
session.

@module ph_runtime.snapshot
"""

from __future__ import annotations

import base64
import hashlib
import pickle
from types import ModuleType
from typing import Any

from ._characters import head

__all__ = ["NamespaceSnapshotter", "restore", "serializable_names"]

_IMMUTABLE: tuple[type, ...] = (bytes, str, int, float, complex, bool, type(None))
"""The types whose contents cannot change while their identity stays the same.

The match is on the exact type, and the list is deliberately short. `tuple` and
`frozenset` are absent because they hold references, so their identity says
nothing about their contents.
"""


class _Overflow(Exception):
    """The payload passed the cap, so the rest of it was never built."""


class _BoundedSink:
    """A write target that stops when the payload exceeds `cap`."""

    __slots__ = ("cap", "chunks", "size")

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.size = 0
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.size += len(data)
        if self.size > self.cap:
            raise _Overflow
        self.chunks.append(data)
        return len(data)


def _dill() -> Any | None:  # noqa: ANN401
    try:
        # `dill` is optional in the runtime venv. Without it, each value that needs
        # it is reported as unpicklable.
        import dill  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:  # pragma: no cover — a venv without dill
        return None
    return dill


def serializable_names(namespace: dict[str, Any], protected: set[str]) -> list[str]:
    """The top-level names that are worth a snapshot attempt, in a stable order."""
    return sorted(
        name
        for name, value in namespace.items()
        if not name.startswith("_")
        if name not in protected
        if not isinstance(value, ModuleType)
    )


def _serialize(value: object, cap: int) -> tuple[bytes | None, str, str]:
    """Return `(payload, skipped, reason)`. Exactly one of `payload` and `skipped`
    is set."""
    sink = _BoundedSink(cap)
    try:
        pickle.Pickler(sink, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    except _Overflow:
        return None, "too-large", ""
    except Exception:
        # The C pickler cannot pickle this value: a function, a class, or a lambda
        # that a cell defined. `dill` handles those, and is used only for them.
        sink = _BoundedSink(cap)
        dill = _dill()
        if dill is None:  # pragma: no cover — a venv without dill
            return None, "unpicklable", "dill is not installed in the runtime venv"
        try:
            dill.Pickler(sink, recurse=True).dump(value)
        except _Overflow:
            return None, "too-large", ""
        except Exception as error:  # a reducer may raise anything at all
            return None, "unpicklable", head(str(error), 200)
    return b"".join(sink.chunks), "", ""


class NamespaceSnapshotter:
    """Remembers what it last sent, so that an unchanged variable costs nothing.

    The memo is here and not in the runner because it makes the capture cheap,
    and not only the send small. The identity fast path needs the previous
    *object*, not only its digest.
    """

    __slots__ = ("_digests", "_identities")

    def __init__(self) -> None:
        self._digests: dict[str, str] = {}
        self._identities: dict[str, tuple[int, Any]] = {}
        """Name → (id, a strong reference). The reference makes the id meaningful.
        Without it, the object could be freed and its id reused by a different
        value under the same name. The reference costs nothing, because the
        namespace holds the same object. It is dropped when the name is removed."""

    def changed(
        self, namespace: dict[str, Any], *, protected: set[str], max_value_bytes: int
    ) -> list[dict[str, Any]]:
        """Return the wire records for the variables that changed since the last
        call.

        The result includes a `deleted` record for each name that is gone. A
        restore must *say* that a name is absent. A model that finds a name
        undefined in the middle of a session reads that as its own bug, and uses
        a turn to work around it.
        """
        records: list[dict[str, Any]] = []
        present = serializable_names(namespace, protected)
        for name in present:
            record = self._record(name, namespace[name], max_value_bytes)
            if record is not None:
                records.append(record)

        for name in sorted(set(self._digests) - set(present)):
            del self._digests[name]
            self._identities.pop(name, None)
            records.append({"var": name, "skipped": "deleted"})
        return records

    def _record(self, name: str, value: object, cap: int) -> dict[str, Any] | None:
        known = self._identities.get(name)
        if known is not None and known[0] == id(value) and type(value) in _IMMUTABLE:
            return None

        payload, skipped, reason = _serialize(value, cap)
        if payload is None:
            self._identities.pop(name, None)
            # Report the refusal one time, then stay quiet. The same refusal on
            # every cell would be one log record per cell for a variable that is
            # not changing.
            if self._digests.get(name) == skipped:
                return None
            self._digests[name] = skipped
            record = {"var": name, "skipped": skipped}
            if reason:
                record["reason"] = reason
            return record

        digest = hashlib.sha256(payload).hexdigest()
        if type(value) in _IMMUTABLE:
            self._identities[name] = (id(value), value)
        if self._digests.get(name) == digest:
            return None
        self._digests[name] = digest
        # Encode to base64 only now, when the record is certain to go out. To
        # encode a large payload that is unchanged would be a whole pass per cell
        # over bytes that nobody sends.
        return {
            "var": name,
            "digest": digest,
            "bytes": len(payload),
            "blob": base64.b64encode(payload).decode("ascii"),
        }


def restore(namespace: dict[str, Any], variables: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Put the payloads back. Returns the names that were restored and the names
    that failed."""
    outcome: dict[str, list[str]] = {"restored": [], "failed": []}
    for record in variables:
        name = record.get("var")
        blob = record.get("blob")
        if not isinstance(name, str) or not isinstance(blob, str):
            continue
        try:
            namespace[name] = pickle.loads(base64.b64decode(blob))
        except Exception:
            # `dill` reads standard pickle bytes. The fallback is therefore needed
            # only for bytes that only `dill` could write.
            dill = _dill()
            if dill is None:  # pragma: no cover
                outcome["failed"].append(name)
                continue
            try:
                namespace[name] = dill.loads(base64.b64decode(blob))
            except Exception:
                outcome["failed"].append(name)
            else:
                outcome["restored"].append(name)
        else:
            outcome["restored"].append(name)
    return outcome
