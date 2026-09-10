"""P8-07 — the daemon's vocabulary as a value, and its params as models.

Two tables now hold every method the daemon answers, each row naming the model
its params are checked against. The tests here hold the tables against the
documented list — so a method added without a row, or a row added without a
name in the list, fails here rather than at the first client that calls it —
and pin the policy the models bring to the edge: a missing `sessionId` is
`invalid_params` naming the field, a field the method does not take is refused
rather than ignored, a cursor that is not a cursor is refused where a stale one
still resumes from the beginning, and a malformed mutation is refused *before*
a root is mounted for it.
"""

from __future__ import annotations

from typing import Any

import pytest
from daemon_helpers import running

from ph_app.daemon.methods import MutationParams
from ph_app.daemon.server import METHODS, MUTATIONS
from ph_app.protocol import DaemonError, SessionParams

pytestmark = pytest.mark.anyio

VOCABULARY = frozenset(
    {
        "initialize",
        "daemon/hello",
        "daemon/config",
        "daemon/status",
        "shutdown",
        "sessions/list",
        "sessions/browse",
        "session/new",
        "session/attach",
        "session/detach",
        "session/status",
        "session/cancel",
        "session/snapshot",
        "session/prompt",
        "session/command",
        "session/stage",
        "session/shell",
        "session/preset",
        "session/readings",
        "commands/list",
        "screens/list",
        "tools/list",
        "attachment/put",
        "credentials/held",
        "credentials/store",
        "schedule/create",
        "schedule/cancel",
        "schedule/list",
    }
)
"""Every method the daemon answers, by name. A new one is added here *and* as a
row — which is the point: two lists that must agree is one list checked."""


def test_the_two_tables_are_the_whole_vocabulary_and_nothing_else() -> None:
    assert set(METHODS) | set(MUTATIONS) == VOCABULARY
    assert not set(METHODS) & set(MUTATIONS), (
        "a name in both tables would be dispatched as a mutation and its METHODS row never reached"
    )


def test_every_method_about_one_root_requires_its_id() -> None:
    """The `str(params["sessionId"])` the row replaced was a `KeyError` waiting
    at twenty-one sites; the model makes the requirement one declaration."""
    daemon_level = {"initialize", "daemon/hello", "daemon/config", "daemon/status", "shutdown"}
    daemon_level |= {"sessions/list", "sessions/browse"}
    for method, row in {**METHODS, **MUTATIONS}.items():
        if method in daemon_level:
            assert not issubclass(row.params, SessionParams), method
            continue
        assert issubclass(row.params, SessionParams), method
        assert row.params.model_fields["session_id"].is_required(), method
    # And every mutation carries the idempotence pair `_mutate` claims a key
    # from — a mutation whose params lacked it could not be guarded at all.
    assert all(issubclass(one.params, MutationParams) for one in MUTATIONS.values())


async def test_a_missing_session_id_is_refused_by_name(tmp_path: Any) -> None:
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        with pytest.raises(DaemonError) as refused:
            await client.call("session/status")
        assert refused.value.reason == "invalid_params"
        assert "sessionId" in str(refused.value), "the field a client forgot is the whole message"
        assert "session/status" in str(refused.value), "and so is the method it forgot it on"


async def test_a_field_the_method_does_not_take_is_refused_not_ignored(tmp_path: Any) -> None:
    """The dropped field was a client that believed it had said something."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="typed")
        with pytest.raises(DaemonError) as refused:
            await client.call("session/status", sessionId="typed", trust="always")
        assert refused.value.reason == "invalid_params"
        assert "trust" in str(refused.value)


async def test_a_cursor_that_is_not_one_is_refused_where_a_stale_one_is_not(
    tmp_path: Any,
) -> None:
    """Shape is the model's; staleness stays `resume_at`'s. The two answers
    differ on purpose — a wrong shape is a client bug worth a sentence, a wrong
    generation is a client that did nothing wrong reading another log's cursor."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="paged")
        with pytest.raises(DaemonError) as refused:
            await client.call("session/snapshot", sessionId="paged", cursor="yesterday")
        assert refused.value.reason == "invalid_params"
        with pytest.raises(DaemonError) as half:
            await client.call("session/snapshot", sessionId="paged", cursor={"sequence": 3})
        assert half.value.reason == "invalid_params"
        assert "generation" in str(half.value)
        stale = await client.call(
            "session/snapshot", sessionId="paged", cursor={"generation": "1", "sequence": 5}
        )
        assert stale["from"] == 0, "a stale cursor reads as 'seen nothing of this log'"


async def test_attach_refuses_a_cursor_rather_than_ignoring_one(tmp_path: Any) -> None:
    """It accepted one for as long as it existed and never read it — attach
    subscribes to what happens next, and catch-up is `session/snapshot` from the
    point the reply names. A client that believed it had asked for replay was
    silently given a live-only subscription, which is the exact thing this row's
    `extra="forbid"` exists to stop; it is now told."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="live-only")
        with pytest.raises(DaemonError) as refused:
            await client.call(
                "session/attach",
                sessionId="live-only",
                cursor={"generation": "1", "sequence": 3},
            )
        assert refused.value.reason == "invalid_params"
        assert "cursor" in str(refused.value)
        # And the ordinary attach still works.
        assert (await client.call("session/attach", sessionId="live-only"))["sessionId"] == (
            "live-only"
        )


async def test_a_malformed_mutation_is_refused_before_a_root_is_mounted(tmp_path: Any) -> None:
    """`_mutate` parses first. The alternative — resolve the root, then find the
    call malformed — mounts a session a mistyped request named and leaves it
    running, which is a side effect of a request the daemon then refused."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        with pytest.raises(DaemonError) as refused:
            await client.call("session/preset", sessionId="never-mounted", preset="bogus")
        assert refused.value.reason == "invalid_params"
        assert "preset" in str(refused.value), "an unknown preset names the field, not a KeyError"
        assert "never-mounted" not in daemon.server.supervisor.roots


async def test_an_unknown_method_is_still_its_own_refusal(tmp_path: Any) -> None:
    """The table lookup replaced the chain's last `else`; the sentence and the
    code a client branches on must not have moved with it."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        with pytest.raises(DaemonError) as refused:
            await client.call("session/nonsense", sessionId="x")
        assert refused.value.reason == "unknown_method"
