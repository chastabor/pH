"""P7-01's model door — `tool-attach`, and the gate it is not allowed to skip.

The row splits ingest by *who asked* (I-9). A person may attach anything they can
already open, and `--attach` reads it with the harness's own permissions. A model
may attach only what `ctx.fs` already lets it read, which is what makes this a
tool rather than an exfiltration primitive: the interesting test here is not that
a picture arrives, it is that a picture the policy refuses does **not**, and that
the refusal is the same one `read` would have given.

The other claim worth pinning is where the media lands. Both shipped wires flatten
a `tool-result` block to text, so an image returned as *result content* would be
dropped by the renderer on the way out — silently, which is the failure the whole
row exists to end. It rides a context message instead, appended after the result,
and the assertion below is against the next **request** rather than against the
result: what matters is that the model was shown the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_dimensions import png

from ph.agent.types import AgentOptions
from ph.cancel import CancelToken
from ph.cordis import Context
from ph.llm.adapter import ResolvedModel
from ph.llm.types import ToolCallBlock, attachment_of
from ph.session.json import dumps
from ph.testing import run_tool
from ph.tools.batch import execute_tool_calls
from ph.tools.builtin.attach_tool import MAX_ATTACH_BYTES

pytestmark = pytest.mark.anyio

OPTIONS = AgentOptions(provider="fake", model="fake-1")
PNG = png(1024, 768)


def _takes_images() -> ResolvedModel:
    return ResolvedModel(accepts=frozenset({"image/png"}), context_window=8192)


async def _agent(ctx: Context, name: str) -> Any:
    return ctx.agents.create(ctx.sessions.create(name), OPTIONS)


async def _dispatch(ctx: Context, agent: Any, name: str, arguments: Any) -> list[Any]:
    """One tool call through the batch, wired the way the driver wires it."""
    results: list[Any] = []

    def accept(context: Any) -> None:
        agent.inbox.append("next-step", context)

    def keep(_execution: Any, result: Any) -> None:
        results.append(result)

    ctx.on("tools/result", keep)
    await execute_tool_calls(
        ctx,
        agent,
        1,
        1,
        [ToolCallBlock(id="call-1", name=name, arguments=dumps(arguments))],
        CancelToken(),
        accept,
    )
    return results


def _media(request: Any) -> list[Any]:
    return [
        attachment
        for message in request.messages
        for block in message.content
        if (attachment := attachment_of(block)) is not None
    ]


async def test_a_file_the_model_attached_reaches_the_next_request(
    mount: Any, tmp_path: Path
) -> None:
    """The whole path, end to end, asserted where it counts.

    The result says what was attached; the *picture* arrives in the request after
    it, because a context message is spliced into the next-step inbox. Asserting
    on the request rather than on the result is deliberate — a build that put the
    media in the result content would pass a result-shaped assertion and still
    show the model nothing, since both wires drop a non-text block inside a tool
    result.
    """
    ctx: Context = await mount()
    ctx.llm_fake.route = _takes_images()
    (tmp_path / "shot.png").write_bytes(PNG)
    agent = await _agent(ctx, "attached")

    # Through the batch rather than through `run_tool`, because the splice is
    # what is under test: `execute_tool_calls` hands a deferred context to the
    # acceptor after the result is durable, and the driver's acceptor is this
    # one line. A test that appended the message itself would be asserting
    # against its own re-implementation of the loop.
    (result,) = await _dispatch(ctx, agent, "attach", {"path": "shot.png"})
    await agent.run()

    assert not result.is_error, result.content
    assert result.value["mime"] == "image/png"
    # Measured on the way in, by the store rather than by this tool (P7-03).
    assert (result.value["width"], result.value["height"]) == (1024, 768)
    assert "It follows this result." in str(result.content)

    (attached,) = _media(ctx.llm_fake.requests[-1])
    assert attached.attachment_id == result.value["attachment_id"]
    assert attached.name == "shot.png"
    assert ctx.attachments.exists(attached), "the bytes are in the store, not in the log"


async def test_the_media_is_not_in_the_tool_result(mount: Any, tmp_path: Path) -> None:
    """The drop this row is written against, pinned at the seam it would happen.

    A `MediaBlock` in the result content becomes a block inside `tool-result`, and
    Anthropic keeps only text blocks there while OpenAI's `tool` role takes a
    string — so the image would vanish between the tool and the wire with nothing
    failing. The result carries the account; the conversation carries the file.
    """
    ctx: Context = await mount()
    (tmp_path / "shot.png").write_bytes(PNG)
    agent = await _agent(ctx, "no-media-in-result")

    result = await run_tool(ctx, "attach", {"path": "shot.png"}, agent=agent)

    assert [block.type for block in result.content] == ["text"]
    assert len(result.additional_contexts) == 1
    assert result.additional_contexts[0].source.form == "relay", "a person did not attach this"


async def test_the_read_gate_bounds_what_a_model_may_attach(mount: Any, tmp_path: Path) -> None:
    """I-9's whole point, and the reason this is not `save_path`.

    A rule that refuses a read refuses the attach, with no second rule to write
    and nothing for `permissions-fs` to opt into — the tool goes through
    `fs/read-intent` exactly as `read` does. A tool that reached the human door
    instead would have handed a model a way to post a private key to a provider
    as a "document".
    """
    ctx: Context = await mount()
    secret = tmp_path / "id_rsa.png"  # media by name; forbidden by policy
    secret.write_bytes(PNG)
    agent = await _agent(ctx, "denied")

    async def refuse(intent: Any, next_: Any) -> Any:
        return "keys are not readable" if intent.path.name.startswith("id_rsa") else await next_()

    ctx.on("fs/read-intent", refuse)
    result = await run_tool(ctx, "attach", {"path": "id_rsa.png"}, agent=agent)

    assert result.is_error and result.error is not None
    assert result.error.kind == "denied", "a policy refusal must not read as a tool failure"
    assert "keys are not readable" in result.error.message
    assert not result.additional_contexts, "nothing was put in front of the model"
    assert not list(ctx.attachments.root.iterdir()) if ctx.attachments.root.exists() else True


async def test_a_file_no_provider_ingests_is_refused_with_the_way_out(
    mount: Any, tmp_path: Path
) -> None:
    """`MediaBlock` is not a general file block, and this is where a model finds out.

    Attaching an archive would otherwise store it, send it, have the route refuse
    it and degrade it to a pointer — four steps to say what one sentence says
    here, and the sentence names `read`, which is the model's actual next move.
    """
    ctx: Context = await mount()
    (tmp_path / "dump.dat").write_bytes(b"not media")
    (tmp_path / "notes.txt").write_text("plain")
    agent = await _agent(ctx, "not-media")

    unknown = await run_tool(ctx, "attach", {"path": "dump.dat"}, agent=agent)
    text = await run_tool(ctx, "attach", {"path": "notes.txt"}, agent=agent, call_id="call-2")

    for result in (unknown, text):
        assert result.is_error and result.error is not None
        assert result.error.kind == "failed", "nothing refused it; it is the wrong kind of file"
        assert "Use read" in result.error.message
    # An extension nothing classifies reads as "no recognisable type" rather than
    # as the literal `application/octet-stream`, which names a MIME the model
    # might reasonably try to fix by renaming the file.
    assert "no recognisable type" in unknown.error.message  # type: ignore[union-attr]
    assert "text/plain" in text.error.message  # type: ignore[union-attr]


async def test_an_oversized_file_is_refused_before_it_is_read(mount: Any, tmp_path: Path) -> None:
    """The cap is the harness's, not the route's, and it is answered by `stat`.

    A route's ceiling is `media-degrade`'s to apply, per request, with a pointer —
    a second copy of it here would refuse for whichever provider the session
    happens to be on. What this bounds is memory: the bytes are read, hashed and
    written, so a model that names a 2 GB file must not be able to make the
    process swap to find that out.
    """
    ctx: Context = await mount({"id": "tool-attach", "config": {"maxBytes": 64}})
    (tmp_path / "big.png").write_bytes(PNG + b"\x00" * 400)
    agent = await _agent(ctx, "too-big")

    result = await run_tool(ctx, "attach", {"path": "big.png"}, agent=agent)

    assert result.is_error and result.error is not None
    assert result.error.kind == "failed" and "over the 64-byte limit" in result.error.message
    assert not result.additional_contexts


async def test_the_tool_is_absent_without_a_store(mount: Any) -> None:
    """A capability the deployment does not have is not advertised.

    `inject` names `attachments`, so a profile that mounts no store never
    activates the row — the model is never offered the tool, spends no turn
    calling it and pays no prompt tokens describing it. The `subagent-task` rule,
    applied to the seam this one needs.
    """
    ctx: Context = await mount({"id": "attachments", "disabled": True})

    assert ctx.get("attachments") is None
    assert "attach" not in ctx.tools.view(ctx).visible


def test_the_default_cap_is_larger_than_either_route_declares() -> None:
    """Stated as a test because the relationship is the argument.

    If this were *tighter* than a route's own limit it would be a second, hidden
    accept policy — refusing a file the provider would have taken. It is looser on
    purpose: the store keeps originals, so a blob over today's route ceiling is
    still there for a route that takes it.
    """
    assert MAX_ATTACH_BYTES > 20 * 1024 * 1024
