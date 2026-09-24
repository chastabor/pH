"""Log writers: the one door into a session log (T6).

**Nothing writes a session log except through a writer**, and a writer writes only the
types its owner is the writer of record for. `Session` has no public `append`: a
module that writes the log mints its own writer at import —

```python
_LOG = log_writer(__name__)

_LOG.append(session, "sandbox/mode", {"mode": mode})
```

— and the writer refuses a type that is not its owner's. So the rule
`known_event_types.WRITERS` states, and `test_log_writers.py` used to hold only
statically, is also held at the write (F12): a row that reached a session it was
handed could write any type into it, a posture record included, and nothing stopped
it at runtime. Now the right to write a type is an object only its owner holds.

**Where the right comes from.** ph-core's table (`known_event_types._WRITTEN_BY`)
for ph-core's vocabulary, and `declare_log_type(..., owner=)` for a type a package
declares. `log_writer(owner)` refuses to mint for a module other than the one calling
it, so the owner cannot be claimed by a module that is not it. A kind's pair is
written by the journal through the writer of the leaf that declares the kind
(`IntentKind.writer`), which is how a kind proves it owns both of its types.

**Rule 6 — what this does not enforce.** Python cannot make a writer unforgeable: a
module can import another's writer, or construct a `LogWriter` directly, or call
`Session._append`. Each of those is a deliberate, reviewable act, and
`test_log_writers.py` turns each into a failure. What the runtime holds is the
honest mistake — a row that appends a type it does not own — which is the one that
happens.

**Not a writer: `admit`.** A replica, a seed and a fork mirror a log somebody else
wrote; they admit records, and write none.

@module ph.session.writers
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..json import JsonValue, as_str
from .events import SessionEvent, SurfaceIntent
from .known_event_types import declared_owner, written_by

if TYPE_CHECKING:
    from .session import Session, SessionBatch

__all__ = ["LogWriteError", "LogWriter", "log_writer", "scaffolding_writer"]


class LogWriteError(ValueError):
    """A write, or a writer, that no writer of record allows (T6)."""


@dataclass(frozen=True, slots=True, eq=False)
class LogWriter:
    """The right to write one owner's types into a session log.

    Minted by `log_writer`, never constructed elsewhere. Compared by identity: two
    writers for one owner are two objects, and neither is the other's.
    """

    owner: str
    types: frozenset[str]
    """The ph-core types ph-core's table grants `owner`. Types `owner` declared are
    checked at the write rather than copied here, since a module may mint its writer
    before its declarations run."""

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
        """Append one event to `log` — a session, or a batch open on one.

        **One method call and a set lookup** on the way to the append itself, since
        this is on the agent loop's hot path (`assistant/chunk`, per streamed token);
        `test_log_writers.py` holds a benchmark to it.

        :raises LogWriteError: when `event_type` is not this writer's.
        """
        if event_type not in self.types and not self.owns(event_type):
            raise LogWriteError(
                f'{self.owner} is not a writer of record for "{event_type}", so it cannot '
                "write one; the writers of record are `known_event_types.WRITERS`, and a "
                "package's own types are the ones it declared"
            )
        return log._append(event_type, data, surface)


def log_writer(owner: str) -> LogWriter:
    """The calling module's writer: `_LOG = log_writer(__name__)`, at module top.

    **Refused for any owner but the caller**, read off the calling frame: a module
    naming another as its owner would claim that module's types. A module no table
    names and that declared nothing gets a writer that refuses every type — the
    runtime refusal a third-party row meets when it writes a type it does not own.

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
    """A writer for every type this build reads — for `ph.testing`, and nothing else.

    Tests build logs by hand: a crashed turn, an ask nobody answered, a log a newer
    build wrote. They are not writers of record for anything, and a writer that
    could write only its owner's types could build none of them. Refused outside
    `ph.testing`, so a shipped module cannot reach for it.

    :raises LogWriteError: when called from outside `ph.testing`.
    """
    caller = as_str(sys._getframe(1).f_globals.get("__name__"))
    if caller != "ph.testing" and not caller.startswith("ph.testing."):
        raise LogWriteError(f"{caller!r} cannot mint the scaffolding writer; ph.testing may")
    return _Scaffolding(owner="ph.testing", types=frozenset())


@dataclass(frozen=True, slots=True, eq=False)
class _Scaffolding(LogWriter):
    """`scaffolding_writer`'s: it owns every type, and the session still refuses one
    this build cannot read back (F11) — the refusal a test of that wants to meet."""

    def owns(self, event_type: str) -> bool:
        return True
