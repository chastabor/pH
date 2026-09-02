"""P6-03 — prefix-cache hit rate and tokens/turn, measured through the shipped profiles.

A cache miss looks exactly like a cache hit, only on the invoice (A12). So the
two numbers a person actually pays for — how much of each request the provider
can serve from cache, and how many tokens each turn costs — are invisible unless
something measures them. This module is that measurement, and
`test_prefix_bench.py` is what keeps the report from rotting.

**The workload is a script over real tools, not a fabricated transcript.** Only
the model's *choices* are authored — which tool to call, with which arguments.
The tools then really run, over real files, and what lands in the log is their
own output. Fabricating tool *results* would have made every token count a
statement about the fabrication rather than about pH.

**Why the workload is authored at all** is the part that has to be said plainly:
there is no recorded RLM session to replay. P3-23 established that the
prime-agent fixtures are "the *coding* agent, not the RLM" — neither contains a
single `ipython` call — and dsh's own `python-sdk-single-exe` snapshots, which
*are* RLM-shaped, call `cordis_define`, `cordis_run`, `workflow` and
`cordis_undefine`: dsh-only tools with no pH counterpart, so replaying them
produces resolution failures rather than a trajectory. Recording a fresh one
needs a provider key. What this benchmark therefore licenses is a claim about
**pH's own prompt assembly under a representative shape**, not a claim about how
any particular model behaves.

@module tests.prefix_bench
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ph.agent.types import AgentOptions
from ph.cordis import DEPLOYMENT, Profile, ProfileDocument
from ph.llm.types import GenerateOptions
from ph.seams.token_meter import TokenMeter
from ph.testing import REPLAY_ROW, RecordedStep, shared_prefix, text_chunks, tool_call_chunks
from ph.wire import WireDataclass
from ph_app.profiles import profile_documents
from ph_app.runtime import mounted

__all__ = [
    "PROFILES",
    "WINDOWS",
    "Measurement",
    "authored_steps",
    "measure",
    "run_profile",
    "workload_files",
]

OPTIONS = AgentOptions(provider="replay", model="replay-1")
"""Routes to `llm-replay`, which serves the authored steps in order."""

PROFILES = ("rlm", "rlm-stable")
"""The A/B axis, and both are shipped profiles rather than hand-assembled row
sets: `rlm-stable` *is* `rlm` plus the `stabilize` bundle plus the row-flips that
profile exists for, so this measures what a person's `--profile` composes."""

HOST_INTERPRETER: dict[str, Any] = {"python": "host", "sweepOrphans": False}
"""`code-runtime-python`, pinned to this interpreter rather than a built venv.

A benchmark that spent a `uv` sync per profile would be measuring `uv`. Copied
from `ph-rlm`'s conftest, which states the rest of the reasoning."""

BIG_LINES = 2_000
"""Lines per large file, chosen to clear `TOOL_TOKEN_LIMIT_BEFORE_EVICT`.

Offload trips at 20 000 estimated tokens and `read`'s own default `limit` is
2 000 lines, so the file has to be large in *tokens* while staying inside what
one `read` returns — otherwise the benchmark reports that stabilization changes
nothing while measuring only that it never engaged. Two drafts measured exactly
that: 6 000-line files that `read` truncated at 2 000, and then lines short
enough that 2 000 of them came to 18 000 tokens, just under the threshold. The
line is wide for that reason, not for realism."""


WINDOWS = (200_000, 8_192)
"""The two context-window regimes, because the answer differs between them.

`ReplayAdapter` defaults to 8 192, and a first draft measured only that — where
each 18 000-token tool result is permanently over budget, so
`compaction-summarize` fires every single turn. That is a real regime and a
degenerate one, and reporting it alone would have described a session under
constant compaction as though it were the ordinary case. 200 000 is what a
current frontier window actually is, and there the same workload never compacts:
the comparison then isolates offload."""


@dataclass(frozen=True, slots=True)
class Measurement(WireDataclass):
    """What one profile's run cost, per request and in total."""

    profile: str
    context_window: int
    cached_tokens: tuple[int, ...]
    """Tokens of each request that a provider could serve from the previous
    request's cache: the longest common message prefix, and only when the system
    prompt is byte-identical. Zero for the first request, which has no
    predecessor to hit."""
    request_tokens: tuple[int, ...]
    """Total tokens each request carries — system prompt plus every message."""

    @property
    def requests(self) -> int:
        return len(self.request_tokens)

    @property
    def hit_rate(self) -> float:
        """Cacheable share of everything sent, across the whole run.

        Summed rather than averaged per request, because that is what the bill
        is: a mean over requests would weight a cheap first turn the same as an
        expensive last one."""
        total = sum(self.request_tokens)
        return 0.0 if total == 0 else sum(self.cached_tokens) / total

    @property
    def tokens_per_turn(self) -> float:
        return (
            0.0 if not self.request_tokens else sum(self.request_tokens) / len(self.request_tokens)
        )

    @classmethod
    def from_wire(cls, wire: dict[str, Any]) -> Measurement:
        return cls(
            profile=wire["profile"],
            context_window=wire["contextWindow"],
            cached_tokens=tuple(wire["cachedTokens"]),
            request_tokens=tuple(wire["requestTokens"]),
        )


def workload_files(root: Path) -> list[Path]:
    """Three real files the scripted turns read, two of them past the threshold.

    Written rather than shipped: the content is uninteresting — what matters is
    the size — and a checked-in 140 KB fixture would be 140 KB of nothing.
    """
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for index, lines in enumerate((BIG_LINES, BIG_LINES, 40)):
        path = root / f"module_{index}.py"
        path.write_text(
            "".join(
                f"VALUE_{index}_{line} = {line}  # a line of real content, wide enough "
                f"that two thousand of them clear the offload threshold\n"
                for line in range(lines)
            ),
            encoding="utf-8",
        )
        files.append(path)
    return files


def authored_steps(files: Sequence[Path], transport: str) -> list[RecordedStep]:
    """The model's side of the conversation: read each file in a cell, then answer.

    **Through Code Mode, because that is what the `rlm` bundle is.** `read` is
    not a native tool in these profiles — calling it directly is refused with
    "call it from inside ipython as `await tools.read(...)`" — so a workload of
    native `read` calls would have measured six identical error strings. That is
    what the first draft measured.

    `transport` is read off the *mounted* registry rather than written here:
    `rlm-presentation` renames the transport, and `ToolRuntime` refuses a call
    to `run_code` when the view presents it under another name.

    One step per cell plus a closing text step per turn, which is the shape the
    loop drives — a `tool-calls` finish reason continues the turn, `stop` ends
    it.
    """
    steps: list[RecordedStep] = []
    for turn, path in enumerate(files, start=1):
        program = f"found = await tools.read(path={str(path)!r})\nfound['text']"
        arguments = json.dumps({"program": program})
        steps += [
            RecordedStep(turn, 1, tool_call_chunks(f"call-{turn}", transport, arguments)),
            RecordedStep(turn, 2, text_chunks(f"Read {path.name}.")),
        ]
    return steps


def measure(
    profile: str, window: int, requests: Sequence[GenerateOptions], meter: TokenMeter
) -> Measurement:
    """Per-request totals and how much of each the previous request had already paid for.

    The cacheable prefix is the *longest common run of messages by id*, which is
    A12's own definition of stability — a provider's cache is keyed on a byte
    prefix, so the first position where two requests differ ends the hit. A
    changed system prompt ends it at zero, because the system prompt precedes
    every message.
    """
    # **Loop requests only**, which is the same boundary `agent-loop-invariant`
    # holds I3 at. A profile with the Continual Harness in it also makes `refine`
    # calls, and those carry their own system prompt for their own purpose — so
    # measuring a "prefix hit" from a loop step to a refine call and back is
    # comparing two unrelated prompts and calling the difference a cache miss.
    # The first draft did that and understated `rlm-stable` by two full
    # invalidations.
    loop = [one for one in requests if one.is_loop_request]
    cached: list[int] = []
    totals: list[int] = []
    for index, current in enumerate(loop):
        system_tokens = meter.measure_text(current.system or "")
        totals.append(system_tokens + meter.estimate_messages(current.messages))
        # `shared_prefix` is A12's own definition, shared with the structural
        # test — so the priced hit and the asserted hit cannot disagree.
        hits = None if index == 0 else shared_prefix(loop[index - 1], current)
        # `None` is a changed system prompt — nothing before the first message
        # survives. `0` is a matching prompt with no shared messages, and the
        # prompt itself is still the cached prefix.
        cached.append(
            0 if hits is None else system_tokens + meter.estimate_messages(current.messages[:hits])
        )
    return Measurement(
        profile=profile,
        context_window=window,
        cached_tokens=tuple(cached),
        request_tokens=tuple(totals),
    )


async def run_profile(
    profile: str, *, home: Path, layers: Sequence[ProfileDocument], context_window: int
) -> Measurement:
    """Drive the authored workload through one shipped profile and measure it.

    The root is disposed on the way out, so two profiles in one process do not
    share a `Context` — which would let the second inherit the first's caches
    and report a hit rate nobody could reproduce.
    """
    # A fixed-width slot, because the workspace path is rendered into the session
    # context snapshot and therefore into the token count: `rlm-8192` and
    # `rlm-200000` differed by nine tokens for no reason but their own names.
    # Absolute counts still move with the length of `home` itself, which is why
    # the guard asserts relationships rather than digits.
    work = home / f"{PROFILES.index(profile)}{WINDOWS.index(context_window)}"
    files = workload_files(work)
    # A `(name, document)` layer is what `--patch` composes as; composed here
    # like any other, and mounted through `ph_app.runtime` rather than a third
    # copy of compose/mount/unwind.
    bench: ProfileDocument = (
        "bench",
        [
            {"id": "fs", "config": {"root": str(work)}},
            {"id": "code-runtime-python", "config": dict(HOST_INTERPRETER)},
            REPLAY_ROW,
        ],
    )
    async with mounted(Profile.from_documents([*layers, bench])) as ctx:
        # The window a provider would report. Left at the adapter's 8 192
        # default, every request in this workload is over budget and the
        # measurement is of nothing but compaction.
        ctx.llm_replay.context_window = context_window
        transport = ctx.tools.view(DEPLOYMENT).transport_name
        ctx.llm_replay.steps = authored_steps(files, transport)
        session = ctx.sessions.create(f"bench-{profile}")
        agent = ctx.agents.create(session, OPTIONS)
        for path in files:
            await agent.prompt(f"Summarise {path.name}.")
        return measure(profile, context_window, ctx.llm_replay.requests, ctx.token_meter)


async def run_all(home: Path) -> list[Measurement]:
    """Every profile in every window regime, in a stable order.

    Imported by `test_prefix_bench.py` and by `__main__` below, so the report's
    table and the test's assertions read the same measurement rather than two
    that could drift.
    """
    return [
        await run_profile(
            profile,
            home=home,
            layers=profile_documents(profile),
            context_window=window,
        )
        for window in WINDOWS
        for profile in PROFILES
    ]


def render(rows: Sequence[Measurement]) -> str:
    """The report's table, generated from the measurement it describes."""
    lines = [
        "| context window | profile | requests | tokens/turn | sent | cacheable | uncached |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        sent, cached = sum(row.request_tokens), sum(row.cached_tokens)
        lines.append(
            f"| {row.context_window:,} | `{row.profile}` | {row.requests} | "
            f"{row.tokens_per_turn:,.0f} | {sent:,} | {row.hit_rate:.1%} | {sent - cached:,} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - the re-derivation entry point
    import os
    import sys
    import tempfile

    import anyio

    async def _main() -> None:
        home = Path(tempfile.mkdtemp(prefix="ph-bench-"))
        os.environ["PH_HOME"] = str(home / "ph")
        rows = await run_all(home)
        if "--json" in sys.argv:
            path = Path(__file__).parent / "prefix_bench.json"
            path.write_text(
                json.dumps([row.to_wire() for row in rows], indent=2) + "\n", encoding="utf-8"
            )
            print(f"wrote {path}")
        print(render(rows))

    anyio.run(_main)
