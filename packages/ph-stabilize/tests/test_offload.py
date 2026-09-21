"""P4-02 — `tool-result-offload`: a large result relocated, not lost (G2, C5).

The row's gates: *replaces at 80 001 chars and not at 80 000; excluded tools
untouched.*

The boundary test is the one to read, and it is written as a pair on purpose.
A threshold asserted only from the far side passes for any limit at or below
the value tested — 80 001 offloading proves nothing without 80 000 staying
inline, which is why upstream's own comparison is `>` and why this file spends
two tests on one number.

## Why `oversized` asks the character count first

The byte count is only asked when a deployment set `max_inline_bytes`: encoding a
**2 MB result** to answer a question the char count already answered would be
**70 µs per call**.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from stabilize_helpers import PROFILE, blob, break_spill, events_of

from ph.cancel import CancelToken
from ph.cordis import DEPLOYMENT, Context
from ph.keys import SESSIONS, SPILL_STORE, TOOLS
from ph.llm.types import ToolCallBlock, ToolResultBlock, ToolSource, text_of
from ph.session import Session, SessionEvent, derive_event_message
from ph.session.known_event_types import (
    IGNORABLE_SESSION_EVENT_TYPES,
    KNOWN_SESSION_EVENT_TYPES,
)
from ph.testing import (
    MountProfile,
    StubAgent,
    as_kind,
    not_none,
    run_tool,
    simple_tool,
    tool_runtime,
)
from ph.tools import ToolExecution
from ph.tools.batch import execute_tool_calls
from ph.tools.definition import (
    Accept,
    ToolOutput,
    define_tool,
    text_content,
)
from ph_stabilize.offload import (
    NUM_CHARS_PER_TOKEN,
    TOO_LARGE_TOOL_MSG,
    TOOL_TOKEN_LIMIT_BEFORE_EVICT,
    UPSTREAM_TOO_LARGE_TOOL_MSG,
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
    ctx: Context,
    session: Session,
    name: str,
    text: str,
    *,
    self_limits: bool = False,
) -> Any:  # noqa: ANN401
    """Drive one real call through the whole pipeline, returning its event.

    The stub is registered on an *agent scope*, which shadows a global tool of
    the same name — the registry's own mechanism (B7), and the only way to test
    the excluded list, whose names (`read`, `glob`, …) belong to real tools the
    base profile already mounts. `ctx.scope()` and not `ctx`: `StubAgent` keeps
    whatever context it is handed, so passing the root would register globally
    and collide.
    """
    agent = StubAgent(ctx.scope("agent"), session)
    ctx.require(TOOLS).register(
        simple_tool(name, lambda _args, _run: text, self_limits=self_limits), scope=agent.ctx
    )
    block = ToolCallBlock(id=f"call-{name}", name=name, arguments="{}")
    await execute_tool_calls(ctx, agent, 1, 1, [block], CancelToken(), lambda _c: None)
    return next(event for event in session.events if event.type == "tool/result")


def _replaces_value(name: str, value: Any) -> Callable[..., Awaitable[Any]]:  # noqa: ANN401
    """A `tools/post-execute` row that swaps one tool's value and leaves the rest.

    The shape D10 is about: a row that states a `value` and no `content`, so the
    content the model will see does not exist yet when the offload has to decide
    whether it is too large.
    """

    async def row(
        execution: ToolExecution,
        result: Any,  # noqa: ANN401
        next_: Callable[..., Awaitable[Any]],
    ) -> Any:  # noqa: ANN401
        decision = await next_(execution, result)
        return Accept(value=value, has_value=True) if execution.name == name else decision

    return row


def _call_id(event: SessionEvent) -> str:
    message = derive_event_message(event)
    assert message is not None
    return str(as_kind(message.source, ToolSource).call_id)


def model_text(event: SessionEvent) -> str:
    """What the model actually read for one call.

    Through `derive_event_message`, which owns the `tool/result` payload shape —
    a third reader spelling it by hand is how they drift. Not `str(event.data)`:
    the payload's repr escapes newlines, so a multi-line original would never be
    found in it and the assertion would fail while the code was right.
    """
    message = derive_event_message(event)
    assert message is not None
    return "\n".join(text_of(as_kind(block, ToolResultBlock).content) for block in message.content)


# ------------------------------------------------------------- the boundary --


async def test_a_result_at_the_threshold_stays_inline(mount: MountProfile) -> None:
    """80 000 characters is admitted — the limit is what the policy still allows."""
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("at-limit")

    event = await _run(ctx, session, "big", blob(THRESHOLD))

    assert TOO_LARGE not in model_text(event)
    assert not [e for e in session.events if e.type == "offload/spilled"]


async def test_one_character_over_the_threshold_is_offloaded(mount: MountProfile) -> None:
    """80 001 is not. The row's gate, and the reason the comparison is `>`."""
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("over-limit")

    event = await _run(ctx, session, "big", blob(THRESHOLD + 1))

    assert TOO_LARGE in model_text(event)
    (spilled,) = [e for e in session.events if e.type == "offload/spilled"]
    assert spilled.data["callId"] == "call-big"
    assert Path(str(spilled.data["locator"])).is_file()


async def test_the_original_is_recoverable_from_the_path_the_model_was_given(
    mount: MountProfile,
) -> None:
    """A relocation, not a deletion — the property the spill seam exists for.

    The path in the replacement must be the path that holds the text, or the
    harness has told the model something is retrievable when it is not.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("recoverable")
    original = blob(THRESHOLD + 1)

    event = await _run(ctx, session, "big", original)
    (spilled,) = [e for e in session.events if e.type == "offload/spilled"]

    assert str(spilled.data["locator"]) in model_text(event), "the model was not told where it went"
    assert Path(str(spilled.data["locator"])).read_text(encoding="utf-8") == original


# ------------------------------------------------------------------ excluded --


async def test_a_self_limiting_tool_is_untouched(mount: MountProfile) -> None:
    """The row's other gate, asked of the *tool* rather than of a name list.

    A tool that takes an offset and a limit has already told the model how to
    page, so offloading its result spends a file to teach it something its own
    contract said. `self_limits` is how it says that — and matching names
    instead would be this package keeping a list of another package's tools,
    which upstream can do because its tool set is closed and pH's is not.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("declared")

    event = await _run(ctx, session, "pager", blob(THRESHOLD * 2), self_limits=True)

    assert TOO_LARGE not in model_text(event)
    assert not [e for e in session.events if e.type == "offload/spilled"]


async def test_a_tool_that_does_not_declare_is_offloaded(mount: MountProfile) -> None:
    """The other half of the pair: without the declaration the guard rail runs.

    Named after the first version of the test above, which registered a stub
    called `read` and so proved only that *name* matching worked — it passed
    whether or not the list matched any tool this harness registers.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("undeclared")

    event = await _run(ctx, session, "pager", blob(THRESHOLD * 2))

    assert TOO_LARGE in model_text(event)


async def test_exactly_the_paging_tools_declare_that_they_self_limit(mount: MountProfile) -> None:
    """And the declaration is on the tools it is supposed to be on.

    Enumerated from the *registry* — every tool the profile actually mounts —
    rather than from a hand-written candidate list, so a new tool that forgets
    to declare, or one that declares when it should not, shows up here. The
    row's whole exclusion policy is these flags now: if `read` stopped
    declaring, a paged file read would start being spilled and nothing else in
    this file would notice.
    """
    ctx = await mount(profile=PROFILE)
    registered = {schema.name for schema in ctx.require(TOOLS).schemas(scope=DEPLOYMENT)}
    declared = {
        name
        for name in registered
        if not_none(ctx.require(TOOLS).get(name, scope=DEPLOYMENT)).self_limits
    }

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
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upstream's rule, and the one this file did not have a gate for.

    An offload that cannot store the content must not be the reason the model
    loses it. Written after a mutation that turned the `except` into a `raise`
    passed every other test here.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("no-disk")
    break_spill(monkeypatch)
    original = blob(THRESHOLD + 1)

    event = await _run(ctx, session, "big", original)

    assert TOO_LARGE not in model_text(event), "the result was replaced anyway"
    assert model_text(event) == original, "the model lost a result the disk could not hold"
    assert not [e for e in session.events if e.type == "offload/spilled"]


# ----------------------------------------------------------- one at a time --


async def test_only_the_oversized_sibling_is_replaced(mount: MountProfile) -> None:
    """C5. Forty dispatches get forty answers, not one melted together.

    Every dispatch crosses this same waterfall, so the per-result decision is
    the pipeline's rather than something this row has to arrange — but that is
    a claim about the seam, and a claim is what a test is for.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("siblings")
    agent = StubAgent(ctx.scope("agent"), session)
    ctx.require(TOOLS).register(
        simple_tool("huge", lambda _a, _r: blob(THRESHOLD + 1), safe=True), scope=agent.ctx
    )
    ctx.require(TOOLS).register(
        simple_tool("tiny", lambda _a, _r: "small", safe=True), scope=agent.ctx
    )

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


async def test_it_measures_the_projection_another_listener_produced(mount: MountProfile) -> None:
    """Composition: the guard rail measures what will be *sent*.

    A `tools/post-execute` row ahead of this one may rewrite the content — a
    redactor, a formatter, an RLM view shaper. Measuring the body's own output
    instead would let any such row switch offloading off just by touching the
    result, and would spill text the model was never going to see. This one
    grows a small result past the threshold; the offload must notice.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("composed")

    async def inflate(
        execution: ToolExecution,
        result: Any,  # noqa: ANN401
        next_: Callable[..., Awaitable[Any]],
    ) -> Any:  # noqa: ANN401
        decision = await next_(execution, result)
        if execution.name != "small":
            return decision
        return Accept(content=text_content(blob(THRESHOLD + 1)))

    ctx.on("tools/post-execute", inflate)

    event = await _run(ctx, session, "small", "a short result")

    assert TOO_LARGE in model_text(event), "the inflated projection went unmeasured"
    (spilled,) = [e for e in session.events if e.type == "offload/spilled"]
    assert Path(str(spilled.data["locator"])).read_text(encoding="utf-8") == blob(THRESHOLD + 1)


# ---------------------------------------------------------------- the sweep --


async def test_a_blob_whose_event_never_landed_is_swept_at_the_next_open(
    mount: MountProfile,
) -> None:
    """**P6-15's gate, through the producers that were never swept.** Both
    directions in one test, because a sweep that deleted *everything* would pass
    the orphan half while destroying the offload the model was told it could read
    back — which is what a per-producer sweep produces on a shared owner."""
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("swept")
    await _run(ctx, session, "big", blob(THRESHOLD + 1))
    (spilled,) = events_of(session, "offload/spilled")
    live = Path(spilled.data["locator"])

    orphan = await ctx.require(SPILL_STORE).save_text(
        owner=session.id, source="a crash", suggested_name="never-recorded.md", content="lost"
    )
    assert Path(orphan.locator).is_file(), "the crash-shaped file exists before the sweep"

    removed = await ctx.require(SPILL_STORE).sweep_session(session)

    assert removed == [orphan.locator], removed
    assert not Path(orphan.locator).exists()
    assert live.is_file(), "the offload the model was told to read back must survive"


async def test_the_sweep_is_wired_to_session_open(mount: MountProfile) -> None:
    """The seam mounts the listener, so it exists wherever the store does.

    Asserted separately from the fold because they fail differently: a fold that
    is wrong deletes the wrong files, and a fold nobody calls deletes nothing and
    looks exactly like a clean store. The second is what P6-15 was — the sweep
    was correct and its listener belonged to one producer.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("wired")
    orphan = await ctx.require(SPILL_STORE).save_text(
        owner=session.id, source="a crash", suggested_name="orphan.md", content="lost"
    )

    ctx.emit("session/created", session)
    await ctx.drain()

    assert not Path(orphan.locator).exists(), "session open did not sweep"


# ------------------------------------------------------ a value, not content --


async def test_a_structured_value_is_offloaded_by_its_render(mount: MountProfile) -> None:
    """D10 — `has_value` returned early, so a value was never offloaded at all.

    A row replacing the *value* leaves `content` alone and the registry renders
    it after this waterfall, so the content that will reach the model does not
    exist yet. `offload` read that as "nothing to measure" and returned — and a
    structured result went to the model whole however large it was, which is the
    one thing this row exists to prevent.

    The render is asked of `ToolRuntime.projected`, which is where the tool's own
    binding lives (P6-26): a renderer is the *tool's* row code, not this row's.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("valued")
    agent = StubAgent(ctx.scope("agent"), session)
    huge = blob(THRESHOLD + 1)

    ctx.on("tools/post-execute", _replaces_value("structured", huge))
    ctx.require(TOOLS).register(
        simple_tool("structured", lambda _args, _run: "small"), scope=agent.ctx
    )
    settled = await run_tool(
        ctx, "structured", agent=agent, session=agent.session, call_id="call-structured"
    )

    assert TOO_LARGE in text_of(settled.content), "a structured value went to the model whole"
    (spilled,) = [one for one in session.events if one.type == "offload/spilled"]
    assert Path(str(spilled.data["locator"])).read_text(encoding="utf-8") == huge


async def test_the_program_keeps_the_value_the_model_only_gets_a_pointer(
    mount: MountProfile,
) -> None:
    """D10's whole point, and the reason this is not "spill the value".

    `value` is what `bridge.call` hands the program under Code Mode, and the
    program already holds it in a variable — the context cost was never the
    value, it is the render that lands in the transcript. Replacing the value
    with a pointer would break the cell (`result["text"]` on a string) to save
    context that the cell was not spending.

    So the two projections deliberately differ, which is the pairing
    `_post_execute` used to refuse outright. One rule covers both transports:
    spill the render, leave the value alone. Under native tool calling nothing
    reads the value, so passing it on costs nothing there either.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("both")
    agent = StubAgent(ctx.scope("agent"), session)
    whole = {"rows": [blob(THRESHOLD + 1)], "count": 1}

    ctx.on("tools/post-execute", _replaces_value("rowset", whole))
    # `define_tool`, not `simple_tool`: this one needs an *object* output, and
    # rendering a value is the whole mechanism under test.
    ctx.require(TOOLS).register(
        define_tool(
            "rowset",
            "a structured result",
            parameters={"type": "object", "properties": {}},
            output=ToolOutput(
                schema={"type": "object"},
                render=lambda _args, value: text_content(str(value)),
            ),
            execute=lambda _args, _run: {"rows": ["small"], "count": 1},
        ),
        scope=agent.ctx,
    )

    settled = await run_tool(
        ctx, "rowset", agent=agent, session=agent.session, call_id="call-rowset"
    )

    assert settled.value == whole, "the program lost the object it was handed"
    assert TOO_LARGE in text_of(settled.content), "the model was sent the whole thing"


async def test_the_replacement_names_the_tools_this_deployment_actually_has(
    mount: MountProfile,
) -> None:
    """Two faults in one sentence, and both were pH's own words.

    The upstream paragraph describes paging as the only way through, so a model
    handed 40 MB reads it from the top; and it names `read_file`, which is not a
    tool pH has. The first fix appended a paragraph *retracting* the second —
    "pH's reader is `read`, not the `read_file` the paragraph above names" —
    which spends model attention correcting text pH itself emitted, and still
    hardcodes a name list in the one package whose `self_limits` docstring argues
    that a name list in another package cannot know what a deployment registered.

    So the name is localized instead: the reader is whichever visible tool
    declares `reads_paths`, the searchers are the ones declaring
    `searches_paths`, and the upstream literal stays byte-identical and unsent so
    an upgrade still produces a visible diff.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("hinted")

    event = await _run(ctx, session, "chatty", blob(THRESHOLD + 1))

    said = model_text(event)
    assert TOO_LARGE in said, "the fixture did not offload"
    assert "read_file" not in said, "the model was sent a tool pH does not register"
    assert "read tool" in said, "and was not told which one to use"
    assert "`grep`" in said and "`glob`" in said, "searching was still not offered"
    # The path is in it, and the hint points at that path rather than at a
    # vocabulary the model has to map onto its own tools.
    (spilled,) = [one for one in session.events if one.type == "offload/spilled"]
    assert str(spilled.data["locator"]) in said
    # The tracked literal is untouched, which is the whole reason it is separate.
    assert "read_file" in UPSTREAM_TOO_LARGE_TOOL_MSG


def test_the_tools_named_are_the_ones_that_declared_themselves() -> None:
    """The derivation, by declaration rather than by name.

    A name list in this package cannot know what a deployment registered —
    `ToolDefinition.reads_paths` says why — and the empty case is real: a
    profile can disable the fs tools, where "use the  tool" is worse than not
    naming one.

    Asked of the registry, and of the row only for the wording: the tiebreak
    ("which of several readers to name") is the row's, so the registry answers
    with both sets whole.
    """
    ctx, runtime = tool_runtime()
    tools = ctx.require(TOOLS)

    assert tools.path_tools(scope=DEPLOYMENT) == ((), ())

    runtime.register(simple_tool("peek", lambda _a, _r: "x", reads_paths=True))
    runtime.register(simple_tool("hunt", lambda _a, _r: "x", searches_paths=True))
    runtime.register(simple_tool("plain", lambda _a, _r: "x"))

    # None of these three is called `read`, `grep` or `glob`.
    assert tools.path_tools(scope=DEPLOYMENT) == (("peek",), ("hunt",))
