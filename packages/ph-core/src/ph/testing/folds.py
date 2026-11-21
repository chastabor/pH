"""Checks that a fold is a fold: the laws that `SessionFoldCache` cannot enforce.

`ph.session.folds` states the contract that a cached fold must meet. The fold
is a pure fold of the prefix, and its `extend` from any prefix equals the cold
fold. That module also says that nothing in it can check the contract ahead of
time. This module does check it ahead of time, from the process that built the
log. Only that process holds both the whole log and every prefix of it.

The laws are the ones that the `DeltaChannel` of LangGraph asks of its
reducers, in the shape of pH. That channel requires a reducer to be
*deterministic* and *batching-invariant*:

    reducer(reducer(state, xs), ys) == reducer(state, xs + ys)

Like `folds.py`, the channel enforces neither law. This module runs both, over
every sampled prefix of a real log:

* **Determinism.** Two folds of one log agree. The fold of a replica that was
  built by admitting the same events agrees with them too. So the fold depends
  on the events and the header, and on nothing else that the `Session` object
  holds.
* **No side effects.** The fold leaves the log at the length that it found. It
  also makes no call that reads the clock, the filesystem or a random source.
  That second check is the view that a profiler has of C calls, so it is a
  *heuristic*. It catches the common impurities by name and cannot prove their
  absence. For example, an environment read never reaches a C function that the
  check knows.
* **Batching invariance.** With `extend`, three things hold:

  - From every sampled prefix, extending to every later prefix equals folding
    the later prefix cold.
  - Walking the prefixes one step at a time never leaves the cold fold. The
    cache does this when it is read after every append.
  - Extending over an empty slice changes nothing.

  The first step that diverges is named with its event type, because the two
  paths handle that event differently.

Prefixes start at `first_live_seq` unless the caller says otherwise. A cache
holds no entry from before its session existed. So a cache is never asked to
extend from a prefix that is shorter than the seed of the constructor.

A finding is a sentence. `check_fold_laws` returns the findings.
`assert_fold_laws` raises with all of them. `VerifyingFoldCache` is a cache
that checks each answer at the moment when a running service reads it.

@module ph.testing.folds
"""

from __future__ import annotations

import reprlib
import sys
from collections.abc import Callable
from itertools import combinations, pairwise
from types import FrameType

from ..session import Session, SessionFoldCache

__all__ = ["VerifyingFoldCache", "assert_fold_laws", "check_fold_laws", "prefix_of"]

_MAX_DEFAULT_SPLITS = 64
"""Above this number of live events, the prefixes are sampled evenly and not all taken.

Every pair of sampled prefixes is folded. So the pair count is quadratic in the
sample, and the work is cubic in the log. A test log of a few hundred events
stays under a second. A real log would not.
"""

_PRINTER = reprlib.Repr()
_PRINTER.maxstring = 160
_PRINTER.maxother = 160
"""How a finding renders a folded value: bounded, and never built in full first."""


# C functions whose call inside a fold means that the fold read something other
# than the log. The key is what the profiler reports: the module and the
# qualified name. The module is `None` for a method of a built-in type. `posix`
# and `nt` are the two platform modules behind `os`. Both are listed so that the
# check means the same thing on either platform.
_CLOCK = (
    "time",
    "time_ns",
    "monotonic",
    "monotonic_ns",
    "perf_counter",
    "perf_counter_ns",
    "process_time",
    "process_time_ns",
    "localtime",
    "gmtime",
    "ctime",
)
_DATES = ("datetime.now", "datetime.utcnow", "datetime.today", "date.today")
_RANDOM = ("Random.random", "Random.getrandbits", "Random.randbytes")
_FILESYSTEM = ("stat", "lstat", "listdir", "scandir", "open", "read", "getcwd", "access")
_OS_MODULES = ("posix", "nt")

_IMPURE: dict[tuple[str | None, str], str] = {
    **{("time", name): "the clock" for name in _CLOCK},
    **{(None, name): "the clock" for name in _DATES},
    **{(None, name): "a random source" for name in _RANDOM},
    **{(module, "urandom"): "a random source" for module in _OS_MODULES},
    **{(module, name): "the filesystem" for module in _OS_MODULES for name in _FILESYSTEM},
    ("_io", "open"): "the filesystem",
}


def prefix_of(session: Session, length: int) -> Session:
    """The first `length` events of this log, as a `Session` of their own.

    The events go through `admit`, which keeps every event as it was (seq,
    time, payload). `admit` also accepts a prefix that ends inside a batch,
    which the seed path refuses as a torn write. The header is copied too,
    because some folds read `seed_length`.

    :raises ValueError: when the log holds a type that this build does not know
        and cannot ignore. This is the rule of the seed path, which `admit`
        shares.
    """
    prefix = Session(session.id, header=session.header)
    for event in session.events[:length]:
        prefix.admit(event)
    return prefix


def check_fold_laws[T](
    session: Session,
    compute: Callable[[Session], T],
    extend: Callable[[T, Session, int], T] | None = None,
    *,
    start: int | None = None,
) -> list[str]:
    """Every way in which `compute` and `extend` fail to be a pure fold of this log.

    `compute` and `extend` are what a `SessionFoldCache` is built from, in the
    shapes in which the cache calls them: `compute(log)` and
    `extend(previous, log, from_seq)`. The folded prefixes run from `start` to
    the end of the log. Every prefix is taken until the log is long, and then
    the prefixes are sampled evenly. `start` defaults to
    `session.first_live_seq`, the earliest seq at which a cache could have been
    read.

    An empty list means that every law held over the prefixes that were tried.
    It never means that the fold is pure, which no finite check establishes.
    """
    findings: list[str] = []
    first = session.first_live_seq if start is None else start
    points = _points(first, session.seq)
    prefixes = {k: prefix_of(session, k) for k in points}

    # A fold that grows the log has failed before any other law can be read.
    # The prefixes are built before this guard because `prefix_of` only admits.
    # It folds nothing.
    before = session.seq
    once = compute(session)
    if extend is not None:
        extend(compute(prefixes[first]), session, first)
    if session.seq != before:
        return [f"the fold appended to the log: seq went from {before} to {session.seq}"]

    # Determinism: the same log, twice, and the same events under another object.
    twice = compute(session)
    if once != twice:
        findings.append(
            "compute is not deterministic: two folds of one log gave "
            f"{_show(once)} and then {_show(twice)}"
        )
    replica = compute(prefixes[session.seq])
    if once != replica:
        findings.append(
            "compute reads more than the log: a replica admitting the same events under "
            f"the same header folded to {_show(replica)} where the original gave {_show(once)}"
        )

    # No side effects, as far as a profiler can see them.
    impure = _impure_calls(lambda: compute(session))
    if impure:
        findings.append("compute reads outside the log: " + ", ".join(impure))
    if extend is not None:
        base = compute(prefixes[first])
        impure = _impure_calls(lambda: extend(base, session, first))
        if impure:
            findings.append("extend reads outside the log: " + ", ".join(impure))

    if extend is not None:
        findings.extend(_batching_findings(session, compute, extend, prefixes))

    grown = [k for k in points if prefixes[k].seq != k]
    if grown:
        findings.append(
            f"the fold appended to a prefix: {len(grown)} of {len(points)} prefixes grew"
        )
    return findings


def _batching_findings[T](
    session: Session,
    compute: Callable[[Session], T],
    extend: Callable[[T, Session, int], T],
    prefixes: dict[int, Session],
) -> list[str]:
    """The `extend` laws: resume from anywhere, walk step by step, extend by nothing.

    `compute` is called again for every extension, and not one time per prefix.
    An `extend` may fold in place, as `extend_goals` does, and a value that was
    handed to it twice would carry the first extension into the second. The
    alternatives were measured. Reusing one cold value per prefix reports 26
    false findings of 55 on the goal log. Deep-copying the value is 2.3x slower
    than folding again.
    """
    findings: list[str] = []
    points = list(prefixes)
    cold = {k: compute(prefixes[k]) for k in points}

    # The disagreements are counted and not collected. Only the first one is
    # quoted, and a thoroughly broken fold would otherwise retain one folded
    # value per pair.
    disagreements = 0
    worst: tuple[int, int, T] | None = None
    for i, j in combinations(points, 2):
        got = extend(compute(prefixes[i]), prefixes[j], i)
        if got != cold[j]:
            disagreements += 1
            worst = worst or (i, j, got)
    if worst is not None:
        i, j, got = worst
        pairs = len(points) * (len(points) - 1) // 2
        findings.append(
            f"extend from prefix {i} to prefix {j} does not equal the cold fold of prefix {j} "
            f"({disagreements} of {pairs} prefix pairs disagree): extended {_show(got)}, "
            f"cold {_show(cold[j])}"
        )

    walked = compute(prefixes[points[0]])
    for a, b in pairwise(points):
        walked = extend(walked, prefixes[b], a)
        if walked != cold[b]:
            findings.append(
                f"extending step by step first leaves the cold fold over {_span(session, a, b)}: "
                f"extended {_show(walked)}, cold {_show(cold[b])}"
            )
            break

    idle = extend(compute(session), session, session.seq)
    if idle != cold[session.seq]:
        findings.append(
            f"extend over an empty slice changed the value: {_show(cold[session.seq])} "
            f"became {_show(idle)}"
        )
    return findings


def assert_fold_laws[T](
    session: Session,
    compute: Callable[[Session], T],
    extend: Callable[[T, Session, int], T] | None = None,
    *,
    start: int | None = None,
) -> None:
    """`check_fold_laws`, raising with every finding at once.

    :raises AssertionError: naming each law that failed and where.
    """
    findings = check_fold_laws(session, compute, extend, start=start)
    if findings:
        raise AssertionError("the fold breaks its laws:\n  - " + "\n  - ".join(findings))


class VerifyingFoldCache[T](SessionFoldCache[T]):
    """A `SessionFoldCache` that checks every answer against the cold fold.

    It is for a test that drives a real service over a real log. Give the
    service this cache in place of its own. Every `read` then checks the
    contract at the moment when the process reads the value. That includes the
    one failure that the law checker cannot see: a *reader* that mutated the
    value it was given, and so corrupted the entry for the next reader. A cold
    fold on every read is the cost that the cache exists to avoid, which is why
    this class is a test double.
    """

    def read(self, session: Session) -> T:
        value = super().read(session)
        cold = self._compute(session)
        if value != cold:
            raise AssertionError(
                f"session {session.id} at seq {session.seq}: the cached fold {_show(value)} "
                f"does not equal the cold fold {_show(cold)}"
            )
        return value


# ---------------------------------------------------------------- the helpers --


def _points(first: int, length: int) -> list[int]:
    """The prefix lengths to fold: every length up to the cap, an even sample past it.

    The sample size is clamped to the span. So a short log takes all of its
    prefixes, and a long log takes `_MAX_DEFAULT_SPLITS` sampled lengths and
    its full length, with the same arithmetic in both cases. A span of zero
    folds the whole log and nothing else.
    """
    span = length - first
    steps = min(span, _MAX_DEFAULT_SPLITS)
    return sorted({first + round(i * span / steps) for i in range(steps)} | {length})


def _impure_calls(run: Callable[[], object]) -> list[str]:
    """The impure C calls that `run` makes, each named one time, in the order first seen."""
    found: dict[str, None] = {}

    def hook(frame: FrameType, event: str, arg: object) -> None:
        if event != "c_call":
            return
        module = getattr(arg, "__module__", None)
        name = getattr(arg, "__qualname__", None)
        if not isinstance(name, str):
            return
        key = (module if isinstance(module, str) else None, name)
        reason = _IMPURE.get(key)
        if reason is None or _from_logging(frame):
            return
        found[f"{reason} through {name if key[0] is None else f'{key[0]}.{name}'}"] = None

    previous = sys.getprofile()
    sys.setprofile(hook)
    try:
        run()
    finally:
        sys.setprofile(previous)
    return list(found)


def _from_logging(frame: FrameType) -> bool:
    """Whether a call came from `logging` itself. `logging` stamps every record
    with the clock, and a fold that logs is noisy, not impure."""
    name = frame.f_globals.get("__name__", "")
    return isinstance(name, str) and name.split(".")[0] == "logging"


def _span(session: Session, start: int, stop: int) -> str:
    """For one event, `seq 4 (turn/end)`. For more, a range and the first few types."""
    events = session.events[start:stop]
    if len(events) == 1:
        return f"seq {start} ({events[0].type})"
    types = ", ".join(event.type for event in events[:4])
    more = "" if len(events) <= 4 else f", and {len(events) - 4} more"
    return f"seqs {start}..{stop - 1} ({types}{more})"


def _show(value: object) -> str:
    """A value, short enough to read in a finding.

    The value is rendered through `reprlib` and not through a slice of `repr`.
    Folds produce whole rosters and harness states, and a slice renders all of
    a value before it discards most of it. `reprlib` also elides the middle of
    a long container and not its tail, so the two sides of a disagreement stay
    comparable.
    """
    return _PRINTER.repr(value)
