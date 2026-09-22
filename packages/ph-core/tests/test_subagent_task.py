"""P4-13 — `subagent-task`: delegation a plain tool-calling model can await.

`rlm_run` already delegates, and hands back a handle: the child replies later by
agent message, which needs an inbox, a roster, and a model that knows to keep
working and check back. This is the other shape — one call, one answer — for the
deployments that have none of that.

Two claims carry the file.

**It registers nothing without a provider.** A tool in every prompt that fails
on every call spends the context window teaching a capability the deployment
does not have, and the check runs after the profile is composed so a provider
layered *below* this row still counts.

**A child that did not finish is an error, not an answer.** `answer: ""` beside
`status: "error"` is a value a parent will read as "it found nothing", which is
the one misreading that turns a failed delegation into a wrong conclusion.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest

from ph.cancel import CancelToken
from ph.cordis import DEPLOYMENT, Context
from ph.keys import AGENTS, SESSIONS, SUBAGENTS, TOOLS
from ph.llm.types import text_of
from ph.seams.subagents import SubagentResult
from ph.testing import FAKE_OPTIONS, MountProfile, StubSubagentProvider, run_tool
from ph.tools.definition import Deny, ToolExecutionInput
from ph.tools.errors import SPAWN_REFUSED, TOOL_BUDGET_SPENT

pytestmark = pytest.mark.anyio

ROW: dict[str, Any] = {"id": "subagent-task"}
"""A patch of the row `ph-base` already carries, not a second copy of it —
addressing an existing id by name would mount the plugin twice."""


async def _mounted(
    mount: MountProfile,
    *providers: tuple[str, StubSubagentProvider],
    **config: object,
) -> Context:
    """A profile with these providers, composed the way a real one is.

    The providers land *after* the row's `apply` and before the composed
    moment — which is the sequence under test, and the one a profile that layers
    its provider below this row produces. There is no core row that registers a
    subagent provider, so the registration is by hand and `profile/mounted` is
    dispatched by hand after it.
    """
    ctx = await mount({**ROW, "config": config} if config else ROW)
    for name, provider in providers:
        ctx.require(SUBAGENTS).register_provider(name, provider)
    await ctx.serial("profile/mounted")
    return ctx


def _agent(ctx: Context) -> Any:  # noqa: ANN401
    return ctx.require(AGENTS).create(ctx.require(SESSIONS).create("s"), FAKE_OPTIONS)


async def test_no_provider_means_no_tool(mount: MountProfile) -> None:
    """`ph-base` mounts the seam and no provider, so the shipped profiles get
    the row and no `task` — the model is told about delegation exactly when the
    deployment can perform it."""
    ctx = await mount(ROW)

    assert ctx.require(TOOLS).get("task", scope=DEPLOYMENT) is None


async def test_a_provider_layered_anywhere_still_gets_the_tool(mount: MountProfile) -> None:
    """The reason the check is at `profile/mounted` and not at this row's own
    `apply`: a profile that layers its provider after this row would otherwise
    silently lose delegation, and nothing would report it."""
    ctx = await _mounted(mount, ("stub", StubSubagentProvider()))

    assert ctx.require(TOOLS).get("task", scope=DEPLOYMENT) is not None


async def test_the_answer_comes_back_as_the_result(mount: MountProfile) -> None:
    provider = StubSubagentProvider(answer="Found it in loader.py:88.")
    ctx = await _mounted(mount, ("stub", provider))

    result = await run_tool(
        ctx, "task", {"prompt": "find the loader", "name": "scout"}, agent=_agent(ctx)
    )

    assert not result.is_error
    assert "Found it in loader.py:88." in text_of(result.content)
    assert provider.last().prompt == "find the loader"
    assert provider.last().name == "scout"


async def test_read_is_what_a_child_gets_unless_asked(mount: MountProfile) -> None:
    """The seam's own default, restated here because a delegation tool that
    quietly asked for `write` would hand every child the parent's tree."""
    provider = StubSubagentProvider()
    ctx = await _mounted(mount, ("stub", provider))

    await run_tool(ctx, "task", {"prompt": "look at this"}, agent=_agent(ctx))

    assert provider.last().access == "read"


async def test_a_downgrade_is_reported_to_the_parent(mount: MountProfile) -> None:
    """A child that asked for `write` and got `read` will fail at its first edit,
    and the parent is the one that has to understand why."""
    provider = StubSubagentProvider(grants="read", downgrade_reason="workspace-not-mounted")
    ctx = await _mounted(mount, ("stub", provider))

    result = await run_tool(
        ctx, "task", {"prompt": "fix the bug", "access": "write"}, agent=_agent(ctx)
    )

    assert result.value["granted_access"] == "read"
    assert result.value["note"], "the downgrade reached the value with no sentence"
    assert result.value["note"] in text_of(result.content), "the model was not told"


async def test_a_child_that_failed_is_an_error_not_an_empty_answer(mount: MountProfile) -> None:
    """The misreading this prevents: `answer: ""` looks like "it found nothing",
    which is a conclusion — and a parent acting on it has been told something
    false by a delegation that never ran."""
    provider = StubSubagentProvider(status="error", answer="", error="the model refused")
    ctx = await _mounted(mount, ("stub", provider))

    result = await run_tool(ctx, "task", {"prompt": "do the thing"}, agent=_agent(ctx))

    assert result.is_error
    assert "the model refused" in text_of(result.content)


async def test_a_provider_that_cannot_be_waited_on_is_refused(mount: MountProfile) -> None:
    """`SubagentRun.result` is `None` for a provider whose children only reply by
    message. Blocking on one is not possible, so the call says so rather than
    returning an empty answer or hanging."""
    ctx = await _mounted(mount, ("stub", StubSubagentProvider(waitable=False)))

    result = await run_tool(ctx, "task", {"prompt": "delegate this"}, agent=_agent(ctx))

    assert result.is_error
    assert "cannot be waited on" in text_of(result.content)


async def test_a_refused_spawn_keeps_its_shape_on_the_way_to_the_model(
    mount: MountProfile,
) -> None:
    """D15 — the tool body flattened the refusal at the one boundary that had it.

    `SubagentSpawnError` was caught here and re-raised as a bare `ValueError`,
    so the code, the `failure_kind` and whether the turn should end were all
    dropped and the model read a generic failure. It is a `HarnessError` now and
    goes straight through, which `registry._failure` already knows how to read.

    A ceiling's refusal is the case that needs all three; an ordinary one — no
    such preset, a grant the parent does not hold — keeps the old reading, which
    is the default and is why nothing else in this file moved.

    Sabotage: catch it here again and re-raise as a `ValueError`; the code and
    the conclusion are gone.
    """
    ctx = await _mounted(mount, ("stub", StubSubagentProvider()))
    ctx.require(SUBAGENTS).guard(
        lambda _request: Deny(
            reason="child limit reached", concludes_turn=True, failure_kind="failed"
        )
    )

    result = await run_tool(ctx, "task", {"prompt": "delegate this"}, agent=_agent(ctx))

    assert result.is_error and result.error is not None
    assert result.error.kind == "failed"
    assert (result.error.info or {}).get("code") == TOOL_BUDGET_SPENT
    assert result.concludes_turn, "a spent child budget let the turn run on"


async def test_an_ordinary_refusal_still_reads_as_one_the_model_may_retry(
    mount: MountProfile,
) -> None:
    """The default half of D15: most refusals here are worth retrying.

    No such provider, no such preset, a grant the parent does not hold — a
    caller may fix the arguments and try again, which is why the turn does not
    end and the reading stays a plain failure.
    """
    ctx = await _mounted(mount, ("stub", StubSubagentProvider()))
    ctx.require(SUBAGENTS).guard(lambda _request: "not from here")

    result = await run_tool(ctx, "task", {"prompt": "delegate this"}, agent=_agent(ctx))

    assert result.is_error and result.error is not None
    assert result.error.kind == "failed"
    assert (result.error.info or {}).get("code") == SPAWN_REFUSED
    assert not result.concludes_turn


async def test_two_providers_and_no_choice_stands_the_row_down(mount: MountProfile) -> None:
    """ "Run a child agent" having two answers is why the seam names providers at
    all; picking one here would make that choice silently, in the row least
    entitled to make it."""
    ctx = await _mounted(mount, ("stub", StubSubagentProvider()), ("other", StubSubagentProvider()))

    assert ctx.require(TOOLS).get("task", scope=DEPLOYMENT) is None


async def test_a_named_provider_settles_the_ambiguity(mount: MountProfile) -> None:
    ctx = await _mounted(
        mount,
        ("stub", StubSubagentProvider(answer="wrong one")),
        ("other", StubSubagentProvider(answer="the right one")),
        provider="other",
    )

    result = await run_tool(ctx, "task", {"prompt": "go"}, agent=_agent(ctx))

    assert "the right one" in text_of(result.content)


async def test_a_canceled_wait_is_an_abort_rather_than_a_tool_failure(
    mount: MountProfile,
) -> None:
    """N3 — the cancel reached the child and then took the wrong exit.

    C7 gave `task` a cancel that releases the child; what it reported afterwards
    was a bare `ValueError`, so `registry._failure` took the `HarnessError`
    branch and told the model the tool had **failed**. `Canceled` has been mapped
    to `aborted_result` there all along — a person's own interrupt read as the
    harness breaking, and a model that reads breakage retries what it was just
    told to stop.

    The token is cancelled from *inside* the child's wait on purpose: cancelled
    beforehand, `prepare` answers `aborted` before the body ever runs, and the
    test would pass against the defect.

    Sabotage: raise `ValueError` again and the kind is `failed`.
    """
    released: list[bool] = [False]
    token = CancelToken()

    class _Parks(StubSubagentProvider):
        async def start(self, request: Any) -> Any:  # noqa: ANN401
            run = await super().start(request)

            async def park() -> SubagentResult:
                token.cancel("the person pressed stop")
                await anyio.sleep_forever()
                raise AssertionError("unreachable: the cancel wins")

            def release() -> None:
                released[0] = True

            run.result = park
            run.dispose = release
            return run

    ctx = await _mounted(mount, ("stub", _Parks()))
    agent = _agent(ctx)

    with anyio.fail_after(10):
        result = await ctx.require(TOOLS).execute(
            ToolExecutionInput(
                call_id="call-1",
                name="task",
                arguments={"prompt": "do the thing"},
                scope=agent.ctx,
                session=agent.session,
                agent=agent,
                cancel=token,
            )
        )

    assert released[0], "the child was left running"
    assert result.is_error
    assert result.error is not None
    assert result.error.kind == "aborted", (
        f"a person's interrupt was reported to the model as {result.error.kind}"
    )
