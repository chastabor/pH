"""Every method that changes a root is idempotent by one key, by construction.

A reconnecting client cannot know whether its last request landed, so it sends
it again with the same `clientId`/`commandId`. That rule used to be applied per
handler by memory: two handlers spelled it and the third forgot. `MUTATIONS` is
the table that makes forgetting impossible — a method in it gets the guard, one
not in it cannot claim to be idempotent by key — and this file drives every row
of it the same way, so a row added tomorrow is covered the day it is added.

The second rule is the ordering. The key is claimed *between* validating and
acting, so a refusal never consumes a retry: a prompt refused for an unknown
attachment and re-sent under the same key with a known one must act.
"""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from daemon_helpers import running, until

from ph.llm.types import AttachmentRef
from ph.seams.commands import CommandDefinition
from ph.session import now_ms
from ph_app.daemon.server import MUTATIONS, PROJECTIONS
from ph_app.protocol import DaemonError

pytestmark = pytest.mark.anyio

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000100000000f0802000000"
    "9a76829d0000000a49444154789c6360000002000100ff03cf0e0e0000"
    "000049454e44ae426082"
)

Case = tuple[Callable[[Any, Any], Awaitable[dict[str, Any]]], Callable[[Any], int] | None]
"""Per method: build its params against a root, and how many times its effect
happened — or `None` where the effect leaves no countable trace and the reply's
shape is the whole claim."""


async def _prompt(client: Any, root: Any) -> dict[str, Any]:
    return {"prompt": "hello"}


async def _command(client: Any, root: Any) -> dict[str, Any]:
    root.ctx.commands.register(
        CommandDefinition(name="probe", summary="a probe", run=lambda argument, ctx: "ran")
    )
    return {"line": "/probe"}


async def _stage(client: Any, root: Any) -> dict[str, Any]:
    reply = await client.call(
        "attachment/put",
        sessionId=root.id,
        name="p.png",
        mime="image/png",
        contentB64=b64encode(PNG).decode(),
    )
    return {"attachment": reply["attachment"]}


async def _shell(client: Any, root: Any) -> dict[str, Any]:
    return {"command": "echo hi"}


async def _preset(client: Any, root: Any) -> dict[str, Any]:
    return {"preset": "workspace-write"}


async def _credential(client: Any, root: Any) -> dict[str, Any]:
    return {"name": "PROBE_KEY", "value": "shh"}


def _events(kind: str) -> Callable[[Any], int]:
    return lambda root: sum(1 for one in root.session.events if one.type == kind)


CASES: dict[str, Case] = {
    "session/prompt": (_prompt, _events("user/message")),
    "session/command": (_command, _events("command/run")),
    "session/stage": (_stage, lambda root: len(root.staged)),
    "session/shell": (_shell, _events("shell/command")),
    "session/preset": (_preset, _events("permission/preset")),
    "credentials/store": (_credential, None),
}


def test_every_mutation_has_a_case_here_and_no_projection_is_one() -> None:
    """The table and this file agree on what a mutation is.

    A row added to `MUTATIONS` without a case fails here rather than going
    untested; a read that wandered into the mutation table would be given an
    idempotence guard that makes its second call return no data.
    """
    assert set(CASES) == set(MUTATIONS)
    assert not set(MUTATIONS) & set(PROJECTIONS)


@pytest.mark.parametrize("method", sorted(MUTATIONS))
async def test_the_same_key_twice_acts_once_and_says_so(method: str, tmp_path: Any) -> None:
    """One key, one effect, one reply shape — for every row, not two of them.

    Sabotage: drop `once` from the wrapper, and the second call acts again; give
    one handler its own `repeated` reply, and the shape assertion fails for it.
    """
    build, effects = CASES[method]
    async with running(tmp_path) as daemon:
        root = await daemon.root("idempotent")
        client = await daemon.client()
        params = await build(client, root)
        keyed = {"sessionId": root.id, "clientId": "c", "commandId": "1", **params}

        first = await client.call(method, **keyed)
        again = await client.call(method, **keyed)

        assert first.get("repeated") is not True
        assert again["repeated"] is True and again["sessionId"] == root.id, (
            "one repeat shape for every verb, so a client branches on one field"
        )
        if effects is not None:
            await until(lambda: effects(root) >= 1, what=f"{method} to take effect")
            assert effects(root) == 1


async def test_a_refusal_does_not_consume_the_key(tmp_path: Any) -> None:
    """Validate, then claim, then act — in that order, or a refusal eats a retry.

    A prompt naming an attachment this deployment never stored is refused. The
    same client, same key, now with the attachment stored, must *act*: nothing
    was recorded for a request that did nothing.

    Sabotage: claim the key before `prepare`, and the second call comes back
    `repeated` with no message ever sent.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("refused-then-retried")
        client = await daemon.client()
        ghost = AttachmentRef(attachment_id="sha256:" + "0" * 64, mime="image/png", bytes=1)
        keyed = {"sessionId": root.id, "clientId": "c", "commandId": "7", "prompt": "look"}

        with pytest.raises(DaemonError) as refused:
            await client.call("session/prompt", **keyed, attachments=[ghost.to_wire()])
        assert refused.value.reason == "attachment_unknown"

        stored = await _stage(client, root)
        reply = await client.call("session/prompt", **keyed, attachments=[stored["attachment"]])

        assert reply.get("repeated") is not True, (
            "the refused request must not have claimed the key"
        )
        await until(
            lambda: root.session.latest("user/message") is not None, what="the prompt to be logged"
        )


async def test_a_mutation_on_a_passivated_root_brings_it_back(tmp_path: Any) -> None:
    """Every mutation resolves its root through `start`, not a lookup.

    `session/prompt` always did (P5-05); the others reported a passivated root as
    gone. One wrapper, one rule: acting on a session is a reason to have it.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("sleepy")
        supervisor = daemon.server.supervisor
        await supervisor.passivate(root, now=now_ms())
        assert "sleepy" not in supervisor.roots
        client = await daemon.client()

        reply = await client.call(
            "session/preset",
            sessionId="sleepy",
            preset="workspace-write",
            clientId="c",
            commandId="1",
        )

        assert reply["preset"] == "workspace-write"
        assert "sleepy" in supervisor.roots
