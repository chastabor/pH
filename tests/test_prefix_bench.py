"""P6-03's guard — the report's claims, held against a live measurement.

The gate is "report checked in", and a checked-in report about performance is
the kind of document that goes quietly wrong: nothing breaks when it stops
being true. So each numbered claim in `docs/dev-notes/prefix-cache-benchmark.md`
has an assertion here, and they are **relationships rather than digits**.

Absolute token counts move with the length of the temporary directory — the
workspace path is rendered into the session context snapshot, so it is part of
what the model reads and therefore part of the count. Asserting 141,906 would
make the guard fail on a machine with a longer `TMPDIR`, which is a property of
the runner and not of pH. The committed `prefix_bench.json` records the run the
report's table was rendered from; its *shape* is checked here so the record
cannot stop describing the benchmark, while the claims are checked against
numbers measured now.

Unlike P3-23's replay, this needs no `sources/` — the workload is authored, so
the whole thing runs on a clean clone.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import anyio
import pytest
from prefix_bench import PROFILES, WINDOWS, Measurement, render, run_all

RECORD = Path(__file__).parent / "prefix_bench.json"
REPORT = Path(__file__).resolve().parents[1] / "docs" / "dev-notes" / "prefix-cache-benchmark.md"


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> Iterator[list[Measurement]]:
    """One measurement run for the whole module.

    Module-scoped because the claims are all about *one* run — asserting six
    properties of six independent runs would let them disagree while every
    assertion passed. It is also cheap: four mounts and twelve cells come to
    about a second, because `HOST_INTERPRETER` keeps the kernel off `uv`.

    `pytest.MonkeyPatch.context()` rather than the function-scoped fixture: the
    class is scope-independent, and it is the idiom the root conftest uses for
    this exact variable.
    """
    home = tmp_path_factory.mktemp("bench")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PH_HOME", str(home / "ph"))
        yield anyio.run(run_all, home)


def _row(rows: list[Measurement], profile: str, window: int) -> Measurement:
    return next(one for one in rows if one.profile == profile and one.context_window == window)


def test_the_committed_record_still_describes_this_benchmark() -> None:
    """The report's table is rendered from `prefix_bench.json`; this pins its shape.

    P3-23's lesson, applied: commit the reduction the assertions read, not the
    corpus. A record that drifted out of step with the profiles and windows
    actually measured would leave the report's table describing a run nobody can
    reproduce.
    """
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    assert [(one["profile"], one["contextWindow"]) for one in record] == [
        (profile, window) for window in WINDOWS for profile in PROFILES
    ]
    for one in record:
        assert len(one["cachedTokens"]) == len(one["requestTokens"])

    # And the report's table *is* this record, rendered. Compared byte-for-byte
    # against the serialiser rather than field by field, which is P3-23's rule:
    # a second encoding of the same reduction drifts from the first the moment
    # either changes.
    rows = [Measurement.from_wire(one) for one in record]
    assert render(rows) in REPORT.read_text(encoding="utf-8"), (
        "the report's table no longer matches the run it records; "
        "re-render with `uv run python tests/prefix_bench.py`"
    )


def test_without_stabilization_every_request_extends_the_last(
    measured: list[Measurement],
) -> None:
    """Claim 1 — A12 holds exactly, and that is the baseline the rest is read against.

    Each request's cacheable prefix is its predecessor *in full*: nothing is
    rewritten, so a provider re-reads only what the turn added. This is the
    property `test_prefix_stability.py` asserts structurally; here it is priced.
    """
    for window in WINDOWS:
        row = _row(measured, "rlm", window)
        for index in range(1, row.requests):
            assert row.cached_tokens[index] == row.request_tokens[index - 1], (
                f"rlm at {window} rewrote history before request {index}"
            )
        assert row.hit_rate > 0.70


def test_the_window_does_not_change_an_unstabilized_run(
    measured: list[Measurement],
) -> None:
    """Claim 2 — nothing in `rlm` reacts to the budget, which is the point of `stabilize`.

    Both `rlm` rows are byte-identical measurements. An unstabilized profile has
    no mechanism that consults the context window, so a session that outgrows it
    simply keeps growing — which is what the 8 192 row is: 141 906 tokens sent
    against an 8 192-token budget.
    """
    wide, narrow = _row(measured, "rlm", 200_000), _row(measured, "rlm", 8_192)

    assert wide.request_tokens == narrow.request_tokens
    assert narrow.request_tokens[-1] > 8_192 * 3, "the workload did not outgrow the small window"


def test_with_a_large_window_stabilization_costs_slightly_more(
    measured: list[Measurement],
) -> None:
    """Claim 3 — and the reason is claim 4: offload never engages.

    Under no pressure `stabilize` adds prompt sections and subtracts nothing, so
    it is a small net cost. Reporting stabilization as an unconditional saving
    would have been the comfortable answer and the wrong one.
    """
    plain, stable = _row(measured, "rlm", 200_000), _row(measured, "rlm-stable", 200_000)

    assert sum(stable.request_tokens) > sum(plain.request_tokens)
    assert sum(stable.request_tokens) < sum(plain.request_tokens) * 1.05, (
        "the gap is meant to be the added prompt sections, not a behaviour change"
    )
    assert stable.hit_rate > 0.70, "nothing rewrote history, so the prefix still extends"


def test_offload_cannot_reach_a_code_mode_result_at_shipped_defaults(
    measured: list[Measurement],
) -> None:
    """Claim 4 — two shipped thresholds that do not meet.

    `code-runtime-python` caps a cell's value at `maxValueBytes: 65536` (the
    `rlm` bundle sets it), and 64 KiB of text is ~17 000 tokens.
    `tool-result-offload` triggers at `TOOL_TOKEN_LIMIT_BEFORE_EVICT = 20_000`.
    So a Code Mode result is capped *below* the threshold that would offload it,
    and in these profiles almost nothing is native — `ipython` is the only
    callable tool — so offload has essentially nothing to act on.

    Asserted through the measurement rather than by reading the two constants:
    the largest single request under a window that never compacts is the one
    carrying a full cell result, and it stays under the trigger.
    """
    from ph_stabilize.offload import TOOL_TOKEN_LIMIT_BEFORE_EVICT

    stable = _row(measured, "rlm-stable", 200_000)
    largest_result = max(
        after - before
        for before, after in zip(stable.request_tokens, stable.request_tokens[1:], strict=False)
    )

    assert largest_result < TOOL_TOKEN_LIMIT_BEFORE_EVICT, (
        "a cell result reached the offload threshold, so claim 4 no longer holds"
    )


def test_under_pressure_stabilization_trades_the_prefix_for_headroom(
    measured: list[Measurement],
) -> None:
    """Claim 5 — the trade, and it is a real one in both directions.

    With the window exceeded every turn, `compaction-summarize` replaces the
    history with a summary. That is what cuts tokens per turn by two thirds, and
    it is also what costs the prefix: the summary lands at position 0, so every
    cached byte before it is invalidated (I4's surface `replace` against A12's
    prefix stability).
    """
    plain, stable = _row(measured, "rlm", 8_192), _row(measured, "rlm-stable", 8_192)

    assert stable.tokens_per_turn < plain.tokens_per_turn * 0.4, "compaction did not engage"
    assert stable.hit_rate < 0.30 < plain.hit_rate, "the prefix survived, so nothing was replaced"


def test_the_saving_is_almost_entirely_cached_tokens(
    measured: list[Measurement],
) -> None:
    """Claim 6 — the headline, and the one that changes what a person should expect.

    Uncached tokens are what a provider bills at full rate. They are within a
    few percent across **every** row: the 67 % reduction in tokens sent is
    almost entirely a reduction in tokens the prefix cache would have served
    anyway. Stabilization's value here is context-window headroom, not price.

    Stated as a bound rather than a ratio per row because the point is the
    absence of a difference, and a bound is what "no material difference" means.
    """
    uncached = [sum(row.request_tokens) - sum(row.cached_tokens) for row in measured]

    assert max(uncached) / min(uncached) < 1.05, (
        f"uncached totals diverged: {uncached} — the headline claim no longer holds"
    )
