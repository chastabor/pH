"""Checking that a fold is a fold: the laws `SessionFoldCache` cannot enforce.

`ph.session.folds` states the contract a cached fold must meet — a pure fold of
the prefix, whose `extend` from any prefix equals the cold fold — and says
plainly that nothing there can check it. This is where it is checked: ahead of
time, from the process that built the log, which is the one place the whole log
and every prefix of it are both to hand.

The laws are the ones LangGraph's `DeltaChannel` asks of its reducers, in pH's
shape. That channel requires a reducer to be *deterministic* and
*batching-invariant*:

    reducer(reducer(state, xs), ys) == reducer(state, xs + ys)

and, like `folds.py`, enforces neither. Here both are run, over every sampled
prefix of a real log:

* **Determinism.** Two folds of one log agree, and the fold of a replica built
  by admitting the same events agrees with them: the fold depends on the events
  and the header, not on anything else the `Session` object happens to hold.
* **No side effects.** The fold leaves the log the length it found it, and makes
  no call that reads the clock, the filesystem or a random source. That last is
  a profiler's view of C calls, so it is a *heuristic*: it catches the common
  impurities by name and cannot prove their absence — an environment read, say,
  never reaches a C function it knows.
* **Batching invariance.** With `extend`: from every sampled prefix, extending
  to every later prefix equals folding that prefix cold; walking the prefixes
  one step at a time — what the cache does when it is read after every append —
  never leaves the cold fold; and extending over an empty slice changes nothing.
  The first step that diverges is named with its event type, because that event
  is the one the two paths handle differently.

Prefixes start at `first_live_seq` unless told otherwise: a cache holds no entry
from before its session existed, so a prefix shorter than the constructor's seed
is one it can never be asked to extend from.

A finding is a sentence. `check_fold_laws` returns them, `assert_fold_laws`
raises with all of them, and `VerifyingFoldCache` is a cache that checks each
answer at the moment a running service reads it.

@module ph.testing.folds
"""

from __future__ import annotations

import reprlib
import sys
from collections.abc import Callable
from itertools import combinations, pairwise
from types import FrameType
from typing import Any

from ..session import Session, SessionFoldCache

__all__ = ["VerifyingFoldCache", "assert_fold_laws", "check_fold_laws", "prefix_of"]

_MAX_DEFAULT_SPLITS = 64
"""Above this many live events, prefixes are sampled evenly rather than taken all.

Every pair of sampled prefixes is folded, so the pair count is quadratic in the
sample and the work cubic in the log; a test log of a few hundred events stays
under a second, a real log would not.
"""

_PRINTER = reprlib.Repr()
_PRINTER.maxstring = 160
_PRINTER.maxother = 160
"""How a finding renders a folded value: bounded, and never built in full first."""


# C functions whose call inside a fold means it read something other than the log,
# keyed the way the profiler reports one: module and qualified name, the module
# `None` for a method of a built-in type. `posix` and `nt` are `os`'s two faces,
# both listed so the check means the same thing on either platform.
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
    """This log's first `length` events as a `Session` of their own.

    Through `admit`, which keeps every event as it was — seq, time, payload — and
    adds nothing: seeding would append a `session/end-seed` marker, which is right
    for a fork and wrong for a prefix. The header rides along, because
    `seed_length` is part of what some folds read.

    :raises ValueError: when the log holds a type this build does not know and
        cannot ignore — the seed path's rule, which `admit` shares.
    """
    prefix = Session(session.id, header=session.header)
    for event in session.events[:length]:
        prefix.admit(event)
    return prefix


def check_fold_laws[T](
    session: Session,
    compute: Callable[[Any], T],
    extend: Callable[[T, Any, int], T] | None = None,
    *,
    start: int | None = None,
) -> list[str]:
    """Every way `compute` and `extend` fail to be a pure fold of this log.

    `compute` and `extend` are what a `SessionFoldCache` is built from, in the
    shapes it calls them: `compute(log)` and `extend(previous, log, from_seq)`.
    The prefixes folded run from `start` to the end of the log, every one of them
    until the log is long and then evenly sampled. `start` defaults to
    `session.first_live_seq`, the earliest seq a cache could have been read at.

    Empty means every law held over the prefixes tried — never that the fold is
    pure, which no finite check establishes.
    """
    findings: list[str] = []
    first = session.first_live_seq if start is None else start
    points = _points(first, session.seq)
    prefixes = {k: prefix_of(session, k) for k in points}

    # A fold that grows the log has failed before any other law can be read.
    # Built before the guard because `prefix_of` only admits — it folds nothing.
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
    compute: Callable[[Any], T],
    extend: Callable[[T, Any, int], T],
    prefixes: dict[int, Session],
) -> list[str]:
    """The `extend` laws: resume from anywhere, walk step by step, extend by nothing.

    `compute` is called afresh for every extension rather than once per prefix,
    because an `extend` is allowed to fold in place — `extend_goals` does — and a
    value handed to it twice would carry the first extension into the second.
    Measured against the alternatives: reusing one cold value per prefix reports
    26 false findings of 55 on the goal log, and deep-copying it instead is 2.3x
    slower than folding afresh.
    """
    findings: list[str] = []
    points = list(prefixes)
    cold = {k: compute(prefixes[k]) for k in points}

    # Counted rather than collected: only the first disagreement is quoted, and a
    # thoroughly broken fold would otherwise retain one folded value per pair.
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
    compute: Callable[[Any], T],
    extend: Callable[[T, Any, int], T] | None = None,
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

    For a test that drives a real service over a real log: hand the service this
    in place of its cache and every `read` becomes a check of the contract at the
    moment the process actually reads it — including the one failure the law
    checker cannot see, a *reader* that mutated the value it was handed and so
    poisoned the entry for the next reader. The cold fold on every read is the
    cost the cache exists to avoid, which is why this is a test double.
    """

    def read(self, session: Any) -> T:  # noqa: ANN401
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
    """The prefix lengths to fold: every one up to the cap, evenly sampled past it.

    Clamping the sample size to the span is what makes a short log take all of its
    prefixes and a long one take `_MAX_DEFAULT_SPLITS` of them — the same
    arithmetic either way. A span of zero folds the whole log and nothing else.
    """
    span = length - first
    steps = min(span, _MAX_DEFAULT_SPLITS)
    return sorted({first + round(i * span / steps) for i in range(steps)} | {length})


def _impure_calls(run: Callable[[], object]) -> list[str]:
    """The impure C calls `run` makes, each named once, in the order first seen."""
    found: dict[str, None] = {}

    def hook(frame: FrameType, event: str, arg: Any) -> None:  # noqa: ANN401
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
    """Whether a call was `logging`'s own — it stamps every record with the clock,
    and a fold that logs is noisy, not impure."""
    name = frame.f_globals.get("__name__", "")
    return isinstance(name, str) and name.split(".")[0] == "logging"


def _span(session: Session, start: int, stop: int) -> str:
    """`seq 4 (turn/end)` for one event, a range and the first few types for more."""
    events = session.events[start:stop]
    if len(events) == 1:
        return f"seq {start} ({events[0].type})"
    types = ", ".join(event.type for event in events[:4])
    more = "" if len(events) <= 4 else f", and {len(events) - 4} more"
    return f"seqs {start}..{stop - 1} ({types}{more})"


def _show(value: object) -> str:
    """A value, short enough to read in a finding.

    Through `reprlib` rather than a slice of `repr`: the values folds produce are
    whole rosters and harness states, and slicing renders all of one before
    throwing most of it away. It also elides the middle of a long container
    rather than its tail, so the two sides of a disagreement stay comparable.
    """
    return _PRINTER.repr(value)
