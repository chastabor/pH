"""P4-02 — `tool-result-offload`: a large result relocated, not lost (G2, C5).

The row's gates: *replaces at 80 001 chars and not at 80 000; excluded tools
untouched.*

The boundary test is the one to read, and it is written as a pair on purpose.
A threshold asserted only from the far side passes for any limit at or below
the value tested — 80 001 offloading proves nothing without 80 000 staying
inline, which is why upstream's own comparison is `>` and why this file spends
two tests on one number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from stabilize_helpers import PROFILE, blob, break_spill

from ph.cancel import CancelToken
from ph.cordis import DEPLOYMENT
from ph.llm.types import ToolCallBlock, text_of
from ph.session import Session, derive_event_message
from ph.session.known_event_types import (
    IGNORABLE_SESSION_EVENT_TYPES,
    KNOWN_SESSION_EVENT_TYPES,
)
from ph.testing import StubAgent, simple_tool
from ph.tools.batch import execute_tool_calls
from ph.tools.definition import Accept, text_content
from ph_stabilize.offload import (
    NUM_CHARS_PER_TOKEN,
    TOO_LARGE_TOOL_MSG,
    TOOL_TOKEN_LIMIT_BEFORE_EVICT,
    Config,
    content_preview,
    oversized,
)

pytestmark = pytest.mark.anyio

THRESHOLD = NUM_CHARS_PER_TOKEN * TOOL_TOKEN_LIMIT_BEFORE_EVICT
"""80 000. Derived from the two constants rather than written as a literal, so
the tests move with the policy instead of pinning a number twice."""

TOO_LARGE = TOO_LARGE_TOOL_MSG.partition(",")[0]
"""The replacement's opening words, from the constant — so a reworded upstream
message moves the assertions with it rather than leaving them green."""


async def _run(
    ctx: Any, session: Session, name: str, text: str, *, self_limits: bool = False
) -> Any:
    """Drive one real call through the whole pipeline, returning its event.

    The stub is registered on an *agent scope*, which shadows a global tool of
    the same name — the registry's own mechanism (B7), and the only way to test
    the excluded list, whose names (`read`, `glob`, …) belong to real tools the
    base profile already mounts. `ctx.scope()` and not `ctx`: `StubAgent` keeps
    whatever context it is handed, so passing the root would register globally
    and collide.
    """
    agent = StubAgent(ctx.scope("agent"), session)
    ctx.tools.register(
        simple_tool(name, lambda _args, _run: text, self_limits=self_limits), scope=agent.ctx
    )
    block = ToolCallBlock(id=f"call-{name}", name=name, arguments="{}")
    await execute_tool_calls(ctx, agent, 1, 1, [block], CancelToken(), lambda _c: None)
    return next(event for event in session.events if event.type == "tool/result")


def _call_id(event: Any) -> str:
    message = derive_event_message(event)
    assert message is not None
    return str(message.source.call_id)


def model_text(event: Any) -> str:
    """What the model actually read for one call.

    Through `derive_event_message`, which owns the `tool/result` payload shape —
    a third reader spelling it by hand is how they drift. Not `str(event.data)`:
    the payload's repr escapes newlines, so a multi-line original would never be
    found in it and the assertion would fail while the code was right.
    """
    message = derive_event_message(event)
    assert message is not None
    return "\n".join(text_of(block.content) for block in message.content)


# ------------------------------------------------------------- the boundary --


async def test_a_result_at_the_threshold_stays_inline(mount: Any) -> None:
    """80 000 characters is admitted — the limit is what the policy still allows."""
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("at-limit")

    event = await _run(ctx, session, "big", blob(THRESHOLD))

    assert TOO_LARGE not in model_text(event)
    assert not [e for e in session.events if e.type == "offload/spilled"]


async def test_one_character_over_the_threshold_is_offloaded(mount: Any) -> None:
    """80 001 is not. The row's gate, and the reason the comparison is `>`."""
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("over-limit")

    event = await _run(ctx, session, "big", blob(THRESHOLD + 1))

    assert TOO_LARGE in model_text(event)
    (spilled,) = [e for e in session.events if e.type == "offload/spilled"]
    assert spilled.data["callId"] == "call-big"
    assert Path(spilled.data["locator"]).is_file()


async def test_the_original_is_recoverable_from_the_path_the_model_was_given(
    mount: Any,
) -> None:
    """A relocation, not a deletion — the property the spill seam exists for.

    The path in the replacement must be the path that holds the text, or the
    harness has told the model something is retrievable when it is not.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("recoverable")
    original = blob(THRESHOLD + 1)

    event = await _run(ctx, session, "big", original)
    (spilled,) = [e for e in session.events if e.type == "offload/spilled"]

    assert spilled.data["locator"] in model_text(event), "the model was not told where it went"
    assert Path(spilled.data["locator"]).read_text(encoding="utf-8") == original


# ------------------------------------------------------------------ excluded --


async def test_a_self_limiting_tool_is_untouched(mount: Any) -> None:
    """The row's other gate, asked of the *tool* rather than of a name list.

    A tool that takes an offset and a limit has already told the model how to
    page, so offloading its result spends a file to teach it something its own
    contract said. `self_limits` is how it says that — and matching names
    instead would be this package keeping a list of another package's tools,
    which upstream can do because its tool set is closed and pH's is not.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("declared")

    event = await _run(ctx, session, "pager", blob(THRESHOLD * 2), self_limits=True)

    assert TOO_LARGE not in model_text(event)
    assert not [e for e in session.events if e.type == "offload/spilled"]


async def test_a_tool_that_does_not_declare_is_offloaded(mount: Any) -> None:
    """The other half of the pair: without the declaration the guard rail runs.

    Named after the first version of the test above, which registered a stub
    called `read` and so proved only that *name* matching worked — it passed
    whether or not the list matched any tool this harness registers.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("undeclared")

    event = await _run(ctx, session, "pager", blob(THRESHOLD * 2))

    assert TOO_LARGE in model_text(event)


async def test_exactly_the_paging_tools_declare_that_they_self_limit(mount: Any) -> None:
    """And the declaration is on the tools it is supposed to be on.

    Enumerated from the *registry* — every tool the profile actually mounts —
    rather than from a hand-written candidate list, so a new tool that forgets
    to declare, or one that declares when it should not, shows up here. The
    row's whole exclusion policy is these flags now: if `read` stopped
    declaring, a paged file read would start being spilled and nothing else in
    this file would notice.
    """
    ctx = await mount(profile=PROFILE)
    registered = {schema.name for schema in ctx.tools.schemas(scope=DEPLOYMENT)}
    declared = {name for name in registered if ctx.tools.get(name, scope=DEPLOYMENT).self_limits}

    assert declared == {"read", "write", "edit", "glob", "grep"}
    assert "bash" in registered and "bash" not in declared, (
        "a shell command has no offset and no limit to offer"
    )


# -------------------------------------------------------------- the preview --


def test_the_preview_shows_head_tail_and_what_it_left_out() -> None:
    """Upstream's shape: five numbered lines, a count, five more."""
    preview = content_preview("\n".join(f"line {n}" for n in range(1, 21)))

    assert "1  line 1" in preview
    assert "... [10 lines truncated] ..." in preview
    assert "16  line 16" in preview
    assert "line 8" not in preview, "the middle is what a preview omits"


def test_a_short_result_previews_whole() -> None:
    """Ten lines or fewer is the whole thing — no marker, nothing omitted."""
    preview = content_preview("\n".join(f"line {n}" for n in range(1, 5)))

    assert "truncated" not in preview
    assert preview.splitlines() == ["1  line 1", "2  line 2", "3  line 3", "4  line 4"]


def test_a_very_long_line_is_clipped_before_it_reaches_the_preview() -> None:
    """One 200 000-char line would otherwise make the *preview* the problem."""
    preview = content_preview("z" * 200_000 + "\nsecond")

    assert len(max(preview.splitlines(), key=len)) <= 1_000 + len("1  ")


# ---------------------------------------------------------------- fail open --


async def test_a_spill_that_fails_keeps_the_original_result(
    mount: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream's rule, and the one this file did not have a gate for.

    An offload that cannot store the content must not be the reason the model
    loses it. Written after a mutation that turned the `except` into a `raise`
    passed every other test here.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("no-disk")
    break_spill(monkeypatch)
    original = blob(THRESHOLD + 1)

    event = await _run(ctx, session, "big", original)

    assert TOO_LARGE not in model_text(event), "the result was replaced anyway"
    assert model_text(event) == original, "the model lost a result the disk could not hold"
    assert not [e for e in session.events if e.type == "offload/spilled"]


# ----------------------------------------------------------- one at a time --


async def test_only_the_oversized_sibling_is_replaced(mount: Any) -> None:
    """C5. Forty dispatches get forty answers, not one melted together.

    Every dispatch crosses this same waterfall, so the per-result decision is
    the pipeline's rather than something this row has to arrange — but that is
    a claim about the seam, and a claim is what a test is for.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("siblings")
    agent = StubAgent(ctx.scope("agent"), session)
    ctx.tools.register(
        simple_tool("huge", lambda _a, _r: blob(THRESHOLD + 1), safe=True), scope=agent.ctx
    )
    ctx.tools.register(simple_tool("tiny", lambda _a, _r: "small", safe=True), scope=agent.ctx)

    blocks = [
        ToolCallBlock(id="call-huge", name="huge", arguments="{}"),
        ToolCallBlock(id="call-tiny", name="tiny", arguments="{}"),
    ]
    await execute_tool_calls(ctx, agent, 1, 1, blocks, CancelToken(), lambda _c: None)

    results = {_call_id(e): model_text(e) for e in session.events if e.type == "tool/result"}
    assert TOO_LARGE in results["call-huge"]
    assert "small" in results["call-tiny"]
    assert TOO_LARGE not in results["call-tiny"]
    assert [e.data["callId"] for e in session.events if e.type == "offload/spilled"] == [
        "call-huge"
    ]


def test_the_event_type_is_in_the_vocabulary() -> None:
    """ph-core's `append(`-site walker sees only ph-core, so a producer in
    another package owes this proof through its own bundle's tests."""
    assert "offload/spilled" in KNOWN_SESSION_EVENT_TYPES
    assert "offload/spilled" in IGNORABLE_SESSION_EVENT_TYPES


def test_the_byte_threshold_is_a_second_way_to_trip() -> None:
    """`max_inline_bytes` is dsh's `spill-policy` knob, merged in as the plan
    asks. Off by default, so without this its branch would never have run."""
    small = Config(token_limit=None, max_inline_bytes=5)
    assert oversized("x" * 6, small)
    assert not oversized("x" * 5, small), "the limit is what is still allowed"
    # And a multi-byte character counts as its bytes, which is the point of
    # having a byte knob beside a character one.
    assert oversized("é" * 3, small)
    assert not oversized("é" * 3, Config(token_limit=None, max_inline_bytes=None))


async def test_it_measures_the_projection_another_listener_produced(mount: Any) -> None:
    """Composition: the guard rail measures what will be *sent*.

    A `tools/post-execute` row ahead of this one may rewrite the content — a
    redactor, a formatter, an RLM view shaper. Measuring the body's own output
    instead would let any such row switch offloading off just by touching the
    result, and would spill text the model was never going to see. This one
    grows a small result past the threshold; the offload must notice.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.sessions.create("composed")

    async def inflate(execution: Any, result: Any, next_: Any) -> Any:
        decision = await next_(execution, result)
        if execution.name != "small":
            return decision
        return Accept(content=text_content(blob(THRESHOLD + 1)))

    ctx.on("tools/post-execute", inflate)

    event = await _run(ctx, session, "small", "a short result")

    assert TOO_LARGE in model_text(event), "the inflated projection went unmeasured"
    (spilled,) = [e for e in session.events if e.type == "offload/spilled"]
    assert Path(spilled.data["locator"]).read_text(encoding="utf-8") == blob(THRESHOLD + 1)
