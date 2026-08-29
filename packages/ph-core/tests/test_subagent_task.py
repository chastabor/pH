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

import pytest

from ph.llm.types import text_of
from ph.testing import FAKE_OPTIONS, StubSubagentProvider, run_tool

pytestmark = pytest.mark.anyio

ROW: dict[str, Any] = {"id": "subagent-task"}
"""A patch of the row `ph-base` already carries, not a second copy of it —
addressing an existing id by name would mount the plugin twice."""


async def _mounted(mount: Any, *providers: tuple[str, StubSubagentProvider], **config: Any) -> Any:
    """A profile with these providers, composed the way a real one is.

    The providers land *after* the row's `apply` and before the composed
    moment — which is the sequence under test, and the one a profile that layers
    its provider below this row produces. There is no core row that registers a
    subagent provider, so the registration is by hand and `profile/mounted` is
    dispatched by hand after it.
    """
    ctx = await mount({**ROW, "config": config} if config else ROW)
    for name, provider in providers:
        ctx.subagents.register_provider(name, provider)
    await ctx.serial("profile/mounted")
    return ctx


def _agent(ctx: Any) -> Any:
    return ctx.agents.create(ctx.sessions.create("s"), FAKE_OPTIONS)


async def test_no_provider_means_no_tool(mount: Any) -> None:
    """`ph-base` mounts the seam and no provider, so the shipped profiles get
    the row and no `task` — the model is told about delegation exactly when the
    deployment can perform it."""
    ctx = await mount(ROW)

    assert ctx.tools.get("task") is None


async def test_a_provider_layered_anywhere_still_gets_the_tool(mount: Any) -> None:
    """The reason the check is at `profile/mounted` and not at this row's own
    `apply`: a profile that layers its provider after this row would otherwise
    silently lose delegation, and nothing would report it."""
    ctx = await _mounted(mount, ("stub", StubSubagentProvider()))

    assert ctx.tools.get("task") is not None


async def test_the_answer_comes_back_as_the_result(mount: Any) -> None:
    provider = StubSubagentProvider(answer="Found it in loader.py:88.")
    ctx = await _mounted(mount, ("stub", provider))

    result = await run_tool(
        ctx, "task", {"prompt": "find the loader", "name": "scout"}, agent=_agent(ctx)
    )

    assert not result.is_error
    assert "Found it in loader.py:88." in text_of(result.content)
    assert provider.last().prompt == "find the loader"
    assert provider.last().name == "scout"


async def test_read_is_what_a_child_gets_unless_asked(mount: Any) -> None:
    """The seam's own default, restated here because a delegation tool that
    quietly asked for `write` would hand every child the parent's tree."""
    provider = StubSubagentProvider()
    ctx = await _mounted(mount, ("stub", provider))

    await run_tool(ctx, "task", {"prompt": "look at this"}, agent=_agent(ctx))

    assert provider.last().access == "read"


async def test_a_downgrade_is_reported_to_the_parent(mount: Any) -> None:
    """A child that asked for `write` and got `read` will fail at its first edit,
    and the parent is the one that has to understand why."""
    provider = StubSubagentProvider(grants="read", downgrade_reason="no-workspace-tier")
    ctx = await _mounted(mount, ("stub", provider))

    result = await run_tool(
        ctx, "task", {"prompt": "fix the bug", "access": "write"}, agent=_agent(ctx)
    )

    assert result.value["granted_access"] == "read"
    assert result.value["note"], "the downgrade reached the value with no sentence"
    assert result.value["note"] in text_of(result.content), "the model was not told"


async def test_a_child_that_failed_is_an_error_not_an_empty_answer(mount: Any) -> None:
    """The misreading this prevents: `answer: ""` looks like "it found nothing",
    which is a conclusion — and a parent acting on it has been told something
    false by a delegation that never ran."""
    provider = StubSubagentProvider(status="error", answer="", error="the model refused")
    ctx = await _mounted(mount, ("stub", provider))

    result = await run_tool(ctx, "task", {"prompt": "do the thing"}, agent=_agent(ctx))

    assert result.is_error
    assert "the model refused" in text_of(result.content)


async def test_a_provider_that_cannot_be_waited_on_is_refused(mount: Any) -> None:
    """`SubagentRun.result` is `None` for a provider whose children only reply by
    message. Blocking on one is not possible, so the call says so rather than
    returning an empty answer or hanging."""
    ctx = await _mounted(mount, ("stub", StubSubagentProvider(waitable=False)))

    result = await run_tool(ctx, "task", {"prompt": "delegate this"}, agent=_agent(ctx))

    assert result.is_error
    assert "cannot be waited on" in text_of(result.content)


async def test_two_providers_and_no_choice_stands_the_row_down(mount: Any) -> None:
    """ "Run a child agent" having two answers is why the seam names providers at
    all; picking one here would make that choice silently, in the row least
    entitled to make it."""
    ctx = await _mounted(mount, ("stub", StubSubagentProvider()), ("other", StubSubagentProvider()))

    assert ctx.tools.get("task") is None


async def test_a_named_provider_settles_the_ambiguity(mount: Any) -> None:
    ctx = await _mounted(
        mount,
        ("stub", StubSubagentProvider(answer="wrong one")),
        ("other", StubSubagentProvider(answer="the right one")),
        provider="other",
    )

    result = await run_tool(ctx, "task", {"prompt": "go"}, agent=_agent(ctx))

    assert "the right one" in text_of(result.content)
