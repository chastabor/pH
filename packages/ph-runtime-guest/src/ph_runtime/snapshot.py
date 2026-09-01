"""Per-variable snapshots, guest side (D17).

Per *variable*, not per namespace, and that is the whole design: an unchanged
200 MiB DataFrame emits nothing, because its digest did not move. Snapshotting the
namespace as one blob would append that DataFrame again on every cell that touched
anything at all, and the log would grow with the *size of the namespace* rather
than the size of the change.

Three things make "emits nothing" actually cheap, because the digest comparison
alone saves the *send* and not the serialization — which runs after every cell,
for every name:

**The C pickler is tried first.** `dill.Pickler` subclasses `pickle._Pickler`, the
pure-Python one, so a large container would be serialized in Python bytecode.
Anything the C pickler refuses — a cell-defined function, class or lambda, none of
which is importable from `__ph_cell__` — falls back to `dill`, and `dill.loads`
reads standard pickle bytes unchanged. Each object takes the same path every time,
so digests stay stable.

**A definitively-immutable value is compared by identity.** If the object under a
name is the same `bytes`/`str`/`int` object as last cell, its content cannot have
changed, so nothing is serialized. Deliberately *not* extended to tuples or
frozensets: `t = ([1],)` keeps its identity while its contents change, and a fast
path that is wrong once is worse than no fast path.

**An over-cap variable stops at the cap.** The pickler writes into a sink that
raises once it passes `max_value_bytes`, and reports `too-large` either way.

What is deliberately **not** captured: the binding proxies, the imported skill
modules, and anything else the bootstrap put there. Those are rebuilt at boot from
the `boot` frame, so pickling them would store a stale copy of the harness's own
surface — and `dill` would happily serialize a proxy holding a closure over a dead
channel.

`dill` is imported lazily. A runtime venv without it still snapshots everything
the C pickler accepts, which is a degraded session rather than no session.

@module ph_runtime.snapshot
"""

from __future__ import annotations

import base64
import hashlib
import pickle
from types import ModuleType
from typing import Any

__all__ = ["NamespaceSnapshotter", "restore", "serializable_names"]

_IMMUTABLE: tuple[type, ...] = (bytes, str, int, float, complex, bool, type(None))
"""Types whose contents cannot change while their identity does not.

Exact-type matched, and deliberately short. `tuple` and `frozenset` are absent
because they hold references: their identity says nothing about their contents.
"""


class _Overflow(Exception):
    """The payload passed the cap, so the rest of it was never built."""


class _BoundedSink:
    """A write target that gives up once the payload exceeds `cap`."""

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


def _dill() -> Any | None:
    try:
        import dill  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover — a venv without dill
        return None
    return dill


def serializable_names(namespace: dict[str, Any], protected: set[str]) -> list[str]:
    """Top-level names worth trying to snapshot, in a stable order."""
    return sorted(
        name
        for name, value in namespace.items()
        if not name.startswith("_")
        if name not in protected
        if not isinstance(value, ModuleType)
    )


def _serialize(value: Any, cap: int) -> tuple[bytes | None, str, str]:
    """`(payload, skipped, reason)` — exactly one of payload / skipped is set."""
    sink = _BoundedSink(cap)
    try:
        pickle.Pickler(sink, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    except _Overflow:
        return None, "too-large", ""
    except Exception:
        # Not picklable by the C pickler: a cell-defined function, class or
        # lambda. `dill` handles those, and is only reached for them.
        sink = _BoundedSink(cap)
        dill = _dill()
        if dill is None:  # pragma: no cover — a venv without dill
            return None, "unpicklable", "dill is not installed in the runtime venv"
        try:
            dill.Pickler(sink, recurse=True).dump(value)
        except _Overflow:
            return None, "too-large", ""
        except Exception as error:  # a reducer may raise anything at all
            return None, "unpicklable", str(error)[:200]
    return b"".join(sink.chunks), "", ""


class NamespaceSnapshotter:
    """Remembers what it last sent, so an unchanged variable costs nothing.

    The memo lives here rather than in the runner because it is what makes the
    capture cheap, not merely what makes the send small — the identity fast path
    needs the previous *object*, not just its digest.
    """

    __slots__ = ("_digests", "_identities")

    def __init__(self) -> None:
        self._digests: dict[str, str] = {}
        self._identities: dict[str, tuple[int, Any]] = {}
        """name → (id, a strong reference). The reference is what makes the id
        meaningful: without it the object could be freed and its id reused by a
        different value under the same name. It costs nothing — the namespace
        holds the same object — and is dropped when the name goes."""

    def changed(
        self, namespace: dict[str, Any], *, protected: set[str], max_value_bytes: int
    ) -> list[dict[str, Any]]:
        """Wire records for the variables that moved since the last call.

        Includes a `deleted` record per name that has gone, because a restore has
        to *say* a name is absent: a model that finds one undefined mid-session
        reads it as its own bug and spends a turn working around it.
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

    def _record(self, name: str, value: Any, cap: int) -> dict[str, Any] | None:
        known = self._identities.get(name)
        if known is not None and known[0] == id(value) and type(value) in _IMMUTABLE:
            return None

        payload, skipped, reason = _serialize(value, cap)
        if payload is None:
            self._identities.pop(name, None)
            # Reported once, then quiet: the same refusal every cell would be
            # one log record per cell for a variable that is not moving.
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
        # Base64 only now that the record is known to be going out: encoding a
        # large payload that turns out to be unchanged is a whole pass per cell
        # over bytes nobody will send.
        return {
            "var": name,
            "digest": digest,
            "bytes": len(payload),
            "blob": base64.b64encode(payload).decode("ascii"),
        }


def restore(namespace: dict[str, Any], variables: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Put payloads back. Reports what came back and what did not, per variable."""
    outcome: dict[str, list[str]] = {"restored": [], "failed": []}
    for record in variables:
        name = record.get("var")
        blob = record.get("blob")
        if not isinstance(name, str) or not isinstance(blob, str):
            continue
        try:
            namespace[name] = pickle.loads(base64.b64decode(blob))
        except Exception:
            # `dill` reads standard pickle bytes, so the fallback is only needed
            # for what only `dill` could write.
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
