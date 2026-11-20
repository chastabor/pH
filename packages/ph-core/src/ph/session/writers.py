"""Log writers: the only way to write a session log (T6).

**Nothing writes a session log except through a writer.** A writer writes only
the types for which its owner is the writer of record. `Session` has no public
`append`. A module that writes the log mints its own writer at import:

```python
_LOG = log_writer(__name__)

_LOG.append(session, "sandbox/mode", {"mode": mode})
```

The writer refuses a type that does not belong to its owner.
`known_event_types.WRITERS` states the rule. `test_log_writers.py` used to check
the rule only statically, and the writer now also enforces it at the write
(F12). Before that, a row that reached a session could write any type into it,
a posture record included, and nothing stopped the row at run time. Now the
right to write a type is an object that only the owner of the type holds.

**Where the right comes from.** For the vocabulary of ph-core, it comes from the
table of ph-core (`known_event_types._WRITTEN_BY`). For a type that a package
declares, it comes from `declare_log_type(..., owner=)`. `log_writer(owner)`
refuses to mint a writer for any module other than the module that calls it, so
a module cannot claim to be an owner that it is not. The journal writes the pair
of a kind through the writer of the leaf that declares the kind
(`IntentKind.writer`). That is how a kind proves that it owns both of its types.

**Rule 6: what this does not enforce.** Python cannot make a writer unforgeable.
A module can import the writer of another module, construct a `LogWriter`
directly, or call `Session._append`. Each of those is a deliberate act that a
reviewer can see, and `test_log_writers.py` turns each one into a failure. The
runtime check stops the accidental mistake: a row that appends a type that it
does not own. That is the mistake that happens in practice.

**`admit` is not a writer.** A replica, a seed and a fork each mirror a log that
some other writer wrote. They admit records, and they write none.

**`revises` is reserved.** A record that carries this key says which surface
replacements it explains, and `Session.records_of` pairs the record with those
replacements by the key. Only the revision module (`ph.session.revise`)
therefore stamps the key. `append` refuses a payload that carries it. The
revision module writes the record of a producer through `_append_revising`,
with the writer of the producer, because the type belongs to the producer.

@module ph.session.writers
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..json import JsonValue, as_str
from .events import SessionEvent, SurfaceIntent
from .known_event_types import declared_owner, written_by

if TYPE_CHECKING:
    from .session import Session, SessionBatch

__all__ = ["REVISES", "LogWriteError", "LogWriter", "log_writer", "scaffolding_writer"]

REVISES = "revises"
"""The payload key that the revision module stamps on the record of a producer. Its
value is the seqs of the replacements that the record explains. The key is reserved:
no other write may carry it."""


class LogWriteError(ValueError):
    """A write, or a writer, that no writer of record allows (T6)."""


@dataclass(frozen=True, slots=True, eq=False)
class LogWriter:
    """The right to write the types of one owner into a session log.

    `log_writer` mints a writer. Other code must not construct one. Writers are
    compared by identity: two writers for one owner are two objects, and neither
    is the other.
    """

    owner: str
    types: frozenset[str]
    """The ph-core types that the table of ph-core grants to `owner`. The types that
    `owner` declared are checked at the write and are not copied here, because a
    module may mint its writer before its declarations run."""

    def owns(self, event_type: str) -> bool:
        """Whether this writer may write `event_type`."""
        return event_type in self.types or declared_owner(event_type) == self.owner

    def append(
        self,
        log: Session | SessionBatch,
        event_type: str,
        data: Mapping[str, JsonValue],
        surface: SurfaceIntent | None = None,
    ) -> SessionEvent:
        """Append one event to `log`, which is a session or a batch that is open on one.

        The cost before the append itself is **one method call and a set lookup**.
        This method is on the hot path of the agent loop (`assistant/chunk`, one
        for each streamed token), and `test_log_writers.py` has a benchmark that
        checks the cost.

        :raises LogWriteError: when `event_type` does not belong to this writer, or
            `data` carries the reserved `revises`.
        """
        if event_type not in self.types and not self.owns(event_type):
            raise _not_the_writer(self.owner, event_type)
        try:
            stamped = REVISES in data
        except TypeError:
            # The session refuses a payload that is not an object, with an error
            # that says so.
            stamped = False
        if stamped:
            raise LogWriteError(
                f'"{REVISES}" pairs a record with the replacements it explains, so only the '
                "revision door writes it: hand the record to `ph.session.revise`"
            )
        return log._append(event_type, data, surface)

    def _append_revising(
        self,
        log: Session | SessionBatch,
        event_type: str,
        data: Mapping[str, JsonValue],
        revises: Sequence[int],
    ) -> SessionEvent:
        """Append `data`, stamped with the replacements that it explains. Only
        `ph.session.revise` calls this method."""
        if not self.owns(event_type):
            raise _not_the_writer(self.owner, event_type)
        return log._append(event_type, {**data, REVISES: list(revises)}, None)


def _not_the_writer(owner: str, event_type: str) -> LogWriteError:
    return LogWriteError(
        f'{owner} is not a writer of record for "{event_type}", so it cannot write one; the '
        "writers of record are `known_event_types.WRITERS`, and a package's own types are "
        "the ones it declared"
    )


def log_writer(owner: str) -> LogWriter:
    """Return the writer of the calling module: `_LOG = log_writer(__name__)`, at module top.

    **The function refuses any owner other than the caller.** It reads the caller
    from the calling frame. A module that named another module as its owner would
    claim the types of that module. A module that no table names, and that
    declared nothing, gets a writer that refuses every type. A third-party row
    meets that refusal at run time when it writes a type that it does not own.

    :raises LogWriteError: when `owner` is not the calling module.
    """
    caller = as_str(sys._getframe(1).f_globals.get("__name__"))
    if caller != owner:
        raise LogWriteError(
            f"{caller!r} cannot mint the writer of {owner!r}: a module mints its own, "
            "as `log_writer(__name__)`"
        )
    return LogWriter(owner=owner, types=written_by(owner))


def scaffolding_writer() -> LogWriter:
    """Return a writer for every type that this build reads. Only `ph.testing` may call it.

    Tests build logs by hand: a crashed turn, an ask that nobody answered, a log
    that a newer build wrote. A test is not a writer of record for anything, and
    a writer that could write only the types of its owner could build none of
    these logs. The function refuses a caller outside `ph.testing`, so a shipped
    module cannot use it.

    :raises LogWriteError: when called from outside `ph.testing`.
    """
    caller = as_str(sys._getframe(1).f_globals.get("__name__"))
    if caller != "ph.testing" and not caller.startswith("ph.testing."):
        raise LogWriteError(f"{caller!r} cannot mint the scaffolding writer; ph.testing may")
    return _Scaffolding(owner="ph.testing", types=frozenset())


@dataclass(frozen=True, slots=True, eq=False)
class _Scaffolding(LogWriter):
    """The writer that `scaffolding_writer` returns. It owns every type. The session
    still refuses a type that this build cannot read back (F11), and a test of that
    refusal needs to meet it."""

    def owns(self, event_type: str) -> bool:
        return True
