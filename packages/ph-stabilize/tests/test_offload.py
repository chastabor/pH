"""P4-02, `tool-result-offload`: a large result is relocated, not lost (G2, C5).

Gates:

* A result of 80 001 characters is replaced, and a result of 80 000 is not.
* Excluded tools are untouched.

The boundary test is the one to read. It is deliberately written as a pair. A
threshold that is asserted only from the far side passes for any limit at or
below the tested value: an offload at 80 001 proves nothing unless 80 000 stays
inline. That is why the comparison of upstream is `>`, and why this file uses
two tests for one number.

## Why `oversized` asks the character count first

The byte count is computed only when a deployment set `max_inline_bytes`. To
encode a **2 MB result** to answer a question that the character count already
answered would cost **70 µs per call**.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from stabilize_helpers import PROFILE, blob, break_spill, events_of, row

from ph.cancel import CancelToken
from ph.cordis import DEPLOYMENT, Context
from ph.keys import SESSIONS, SPILL_STORE, TOOLS
from ph.llm.types import ToolCallBlock, ToolResultBlock, ToolSource, text_of
from ph.session import Session, SessionEvent, derive_event_message
from ph.session.known_event_types import IGNORABLE_SESSION_EVENT_TYPES
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
"""80 000. The value is derived from the two constants and is not written as a
literal, so the tests move with the policy and do not state the number twice."""

TOO_LARGE = TOO_LARGE_TOOL_MSG.partition(",")[0]
"""The opening words of the replacement, taken from the constant. If the
upstream message is reworded, the assertions move with it. Without that, they
would stay green for the wrong reason."""


async def _run(
    ctx: Context,
    session: Session,
    name: str,
    text: str,
    *,
    self_limits: bool = False,
) -> Any:  # noqa: ANN401
    """Run one real call through the whole pipeline and return its event.

    The stub is registered on an *agent scope*. A tool on an agent scope shadows
    a global tool of the same name, which is the mechanism of the registry
    itself (B7). It was the only way to test the excluded list when that list
    held names (`read`, `glob`, …) that belong to real tools that the base
    profile already mounts. The scope is `ctx.scope()` and not `ctx`:
    `StubAgent` keeps the context that it is given, so the root context would
    register the stub globally and collide.
    """
    agent = StubAgent(ctx.scope("agent"), session)
    ctx.require(TOOLS).register(
        simple_tool(name, lambda _args, _run: text, self_limits=self_limits), scope=agent.ctx
    )
    block = ToolCallBlock(id=f"call-{name}", name=name, arguments="{}")
    await execute_tool_calls(ctx, agent, 1, 1, [block], CancelToken(), lambda _c: None)
    return next(event for event in session.events if event.type == "tool/result")


def _replaces_value(name: str, value: Any) -> Callable[..., Awaitable[Any]]:  # noqa: ANN401
    """A `tools/post-execute` row that replaces the value of one tool and leaves
    the other tools alone.

    This is the shape that D10 is about: a row that states a `value` and no
    `content`. The content that the model will see does not exist yet when the
    offload must decide whether it is too large.
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
    """The text that the model read for one call.

    The text is read through `derive_event_message`, which owns the shape of the
    `tool/result` payload. A third reader that spells the shape by hand is how
    the readers drift apart. `str(event.data)` is not used: the repr of the
    payload escapes newlines, so a multi-line original would never be found in
    it, and the assertion would fail while the code was correct.
    """
    message = derive_event_message(event)
    assert message is not None
    return "\n".join(text_of(as_kind(block, ToolResultBlock).content) for block in message.content)


# ------------------------------------------------------------- the boundary --


async def test_a_result_at_the_threshold_stays_inline(mount: MountProfile) -> None:
    """80 000 characters are admitted. The limit is the size that the policy still allows."""
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("at-limit")

    event = await _run(ctx, session, "big", blob(THRESHOLD))

    assert TOO_LARGE not in model_text(event)
    assert not [e for e in session.events if e.type == "offload/spilled"]


async def test_one_character_over_the_threshold_is_offloaded(mount: MountProfile) -> None:
    """80 001 characters are not admitted. This is the gate of the row, and the
    reason why the comparison is `>`."""
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
    """A relocation, not a deletion. The spill seam exists for this property.

    The path in the replacement must be the path that holds the text. If it is
    not, the harness has told the model that something is retrievable when it is
    not.
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
    """The other gate of the row. The question goes to the *tool* and not to a
    name list.

    A tool that takes an offset and a limit has already told the model how to
    page. To offload its result would spend a file to teach the model something
    that the contract of the tool already said. The tool says so with
    `self_limits`. If this package matched names instead, it would keep a list
    of the tools of another package. Upstream can do that because its tool set
    is closed, and the tool set of pH is not.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("declared")

    event = await _run(ctx, session, "pager", blob(THRESHOLD * 2), self_limits=True)

    assert TOO_LARGE not in model_text(event)
    assert not [e for e in session.events if e.type == "offload/spilled"]


async def test_a_tool_that_does_not_declare_is_offloaded(mount: MountProfile) -> None:
    """The other half of the pair: without the declaration, the guard rail runs.

    The name comes from the first version of the test above. That version
    registered a stub called `read`, so it proved only that *name* matching
    worked. It passed whether or not the list matched any tool that this harness
    registers.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("undeclared")

    event = await _run(ctx, session, "pager", blob(THRESHOLD * 2))

    assert TOO_LARGE in model_text(event)


async def test_an_excluded_tool_is_untouched_without_declaring(mount: MountProfile) -> None:
    """`excluded_tools` is the override for a tool whose author has not declared
    `self_limits`. The row treats a name in that list as self-limiting.

    Sabotage: drop the `excluded_tools` check from `_self_limiting`, and the
    result is offloaded.
    """
    ctx = await mount(row("tool-result-offload", excluded_tools=["pager"]), profile=PROFILE)
    session = ctx.require(SESSIONS).create("excluded")

    event = await _run(ctx, session, "pager", blob(THRESHOLD * 2))

    assert TOO_LARGE not in model_text(event)
    assert not [e for e in session.events if e.type == "offload/spilled"]


async def test_exactly_the_paging_tools_declare_that_they_self_limit(mount: MountProfile) -> None:
    """The declaration is on the tools that must have it.

    The tools are enumerated from the *registry*, which holds every tool that
    the profile mounts, and not from a hand-written candidate list. A new tool
    that forgets to declare, or one that declares when it should not, therefore
    shows up here. These flags are now the whole exclusion policy of the row. If
    `read` stopped declaring, a paged file read would start to be spilled, and
    nothing else in this file would notice.
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
    """A result of ten lines or fewer is shown whole: no marker, and nothing omitted."""
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
    """The rule of upstream, and the rule that this file had no gate for.

    An offload that cannot store the content must not be the reason that the
    model loses it. The test was written after a mutation that turned the
    `except` into a `raise` passed every other test here.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("no-disk")
    break_spill(monkeypatch)
    original = blob(THRESHOLD + 1)

    event = await _run(ctx, session, "big", original)

    assert TOO_LARGE not in model_text(event), "the result was replaced anyway"
    assert model_text(event) == original, "the model lost a result the disk could not hold"
    assert not [e for e in session.events if e.type == "offload/spilled"]


async def test_an_oversized_result_with_no_utf_8_form_stays_inline(
    mount: MountProfile,
) -> None:
    """A lone surrogate has no UTF-8 form, so the store cannot take the result at
    all. A `\\ud800` escape in a JSON tool result decodes to one. The result
    stays inline, as a result that the disk refused does. Before, the planner
    encoded the text itself, and the error left the middleware.

    Sabotage: encode in `plan_tool_result_spill` again, and the call raises.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("no-utf-8")
    original = "\ud800" + blob(THRESHOLD)

    event = await _run(ctx, session, "big", original)

    assert model_text(event) == original
    assert not [e for e in session.events if e.type == "offload/spilled"]


# ----------------------------------------------------------- one at a time --


async def test_only_the_oversized_sibling_is_replaced(mount: MountProfile) -> None:
    """C5. Forty dispatches get forty answers, not one merged answer.

    Every dispatch crosses this same waterfall. The decision for each result
    therefore belongs to the pipeline, and this row does not have to arrange it.
    But that is a claim about the seam, and a test exists to check a claim.
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


def test_the_event_type_is_ignorable() -> None:
    """A reader that skips the event keeps the conversation: the stand-in for the
    spilled body is in the result itself. (`test_log_writers.py` checks that the
    type is *known*.)"""
    assert "offload/spilled" in IGNORABLE_SESSION_EVENT_TYPES


def test_the_byte_threshold_is_a_second_way_to_trip() -> None:
    """`max_inline_bytes` is the `spill-policy` setting of dsh, merged in as the
    plan asks. It is off by default, so without this test its branch would never
    run."""
    small = Config(token_limit=None, max_inline_bytes=5)
    assert oversized("x" * 6, small)
    assert not oversized("x" * 5, small), "the limit is what is still allowed"
    # A multi-byte character counts as its bytes. That is the reason for a byte
    # setting beside a character setting.
    assert oversized("é" * 3, small)
    assert not oversized("é" * 3, Config(token_limit=None, max_inline_bytes=None))


async def test_it_measures_the_projection_another_listener_produced(mount: MountProfile) -> None:
    """Composition: the guard rail measures what will be *sent*.

    A `tools/post-execute` row ahead of this one may rewrite the content: a
    redactor, a formatter, an RLM view shaper. If the row measured the output of
    the body itself, any such row would switch offloading off just by touching
    the result, and the row would spill text that the model was never going to
    see. The listener here grows a small result past the threshold, and the
    offload must notice.
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
    """**The gate of P6-15, through the producers that were never swept.** One
    test covers both directions. A sweep that deleted *everything* would pass
    the orphan half, and it would destroy the offload that the model was told it
    could read back. A per-producer sweep on a shared owner does exactly that."""
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("swept")
    await _run(ctx, session, "big", blob(THRESHOLD + 1))
    (spilled,) = events_of(session, "offload/spilled")
    live = Path(spilled.data["locator"])

    orphan = not_none(
        await ctx.require(SPILL_STORE).try_save_text(
            owner=session.id, source="a crash", suggested_name="never-recorded.md", content="lost"
        )
    )
    assert Path(orphan.locator).is_file(), "the crash-shaped file exists before the sweep"

    removed = await ctx.require(SPILL_STORE).sweep_session(session)

    assert removed == [orphan.locator], removed
    assert not Path(orphan.locator).exists()
    assert live.is_file(), "the offload the model was told to read back must survive"


# ------------------------------------------------------ a value, not content --


async def test_a_structured_value_is_offloaded_by_its_render(mount: MountProfile) -> None:
    """D10: `has_value` returned early, so a value was never offloaded at all.

    A row that replaces the *value* leaves `content` alone, and the registry
    renders the value after this waterfall. The content that will reach the
    model therefore does not exist yet. `offload` read that as "nothing to
    measure" and returned. A structured result then went to the model whole,
    whatever its size, and that is the one thing that this row exists to
    prevent.

    The render is asked of `ToolRuntime.projected`, which is where the binding
    of the tool itself lives (P6-26). A renderer is the row code of the *tool*,
    not of this row.
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
    """The purpose of D10, and the reason why the rule is not "spill the value".

    `value` is what `bridge.call` gives the program under Code Mode, and the
    program already holds it in a variable. The context cost was never the
    value. The cost is the render that lands in the transcript. To replace the
    value with a pointer would break the cell (`result["text"]` on a string) to
    save context that the cell was not spending.

    The two projections therefore differ deliberately. `_post_execute` used to
    refuse that pairing outright. One rule covers both transports: spill the
    render, leave the value alone. Under native tool calling, nothing reads the
    value, so passing it on costs nothing there either.
    """
    ctx = await mount(profile=PROFILE)
    session = ctx.require(SESSIONS).create("both")
    agent = StubAgent(ctx.scope("agent"), session)
    whole = {"rows": [blob(THRESHOLD + 1)], "count": 1}

    ctx.on("tools/post-execute", _replaces_value("rowset", whole))
    # `define_tool`, not `simple_tool`: this tool needs an *object* output, and
    # the render of a value is the mechanism under test.
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
    """Two faults in one sentence, and pH itself sent both.

    The upstream paragraph describes paging as the only way through the file, so
    a model that is given 40 MB reads it from the top. The paragraph also names
    `read_file`, which is not a tool that pH has. The first fix appended a
    paragraph that *retracted* the second fault: "pH's reader is `read`, not the
    `read_file` the paragraph above names". That spends model attention to
    correct text that pH itself emitted. It also still hardcodes a name list, in
    the one package whose `self_limits` docstring argues that a name list in
    another package cannot know what a deployment registered.

    The name is therefore localized. The reader is whichever visible tool
    declares `reads_paths`, and the searchers are the tools that declare
    `searches_paths`. The upstream literal stays byte-identical and unsent, so
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
    # The path is in the text, and the hint points at that path. It does not
    # point at a vocabulary that the model must map onto its own tools.
    (spilled,) = [one for one in session.events if one.type == "offload/spilled"]
    assert str(spilled.data["locator"]) in said
    # The tracked literal is untouched. That is the reason why it is separate.
    assert "read_file" in UPSTREAM_TOO_LARGE_TOOL_MSG


def test_the_tools_named_are_the_ones_that_declared_themselves() -> None:
    """The derivation, by declaration and not by name.

    A name list in this package cannot know what a deployment registered, and
    `ToolDefinition.reads_paths` says why. The empty case is real: a profile can
    disable the fs tools, and then "use the  tool" is worse than no name.

    The registry is asked for the tools, and the row is asked only for the
    wording. The tiebreak ("which of several readers to name") belongs to the
    row, so the registry answers with both sets whole.
    """
    ctx, runtime = tool_runtime()
    tools = ctx.require(TOOLS)

    assert tools.path_tools(scope=DEPLOYMENT) == ((), ())

    runtime.register(simple_tool("peek", lambda _a, _r: "x", reads_paths=True))
    runtime.register(simple_tool("hunt", lambda _a, _r: "x", searches_paths=True))
    runtime.register(simple_tool("plain", lambda _a, _r: "x"))

    # None of these three is called `read`, `grep` or `glob`.
    assert tools.path_tools(scope=DEPLOYMENT) == (("peek",), ("hunt",))
