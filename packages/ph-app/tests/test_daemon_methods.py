"""P8-07/P8-11 — the daemon's vocabulary as values, and its params as models.

Two tables hold every method the daemon answers, each row naming the `Verb`
that carries its name, its params model and its reply. The tests here hold the
tables against `verbs.VOCABULARY` — **derived** from those verb declarations,
so what fails is a verb declared and never routed, which is a name a client can
spell and a reply model it can rely on with no handler behind either. The
reverse direction is gone: a row cannot exist without a verb, because the row
builder takes one.

They also pin the policy the models bring to the edge: a missing `sessionId` is
`invalid_params` naming the field, a field the method does not take is refused
rather than ignored, a cursor that is not a cursor is refused where a stale one
still resumes from the beginning, and a malformed mutation is refused *before*
a root is mounted for it.
"""

from __future__ import annotations

from typing import Any

import pytest
from daemon_helpers import running

from ph_app import verbs
from ph_app.daemon import server
from ph_app.daemon.server import METHODS, MUTATIONS, Announced, Method
from ph_app.params import (
    MutationParams,
    NewSessionParams,
    PromptParams,
)
from ph_app.payloads import MutationRepeated, RootStatusReply
from ph_app.protocol import DaemonError, Notify, SessionParams, Verb
from ph_app.verbs import VOCABULARY

pytestmark = pytest.mark.anyio


def test_the_two_tables_are_the_whole_vocabulary_and_nothing_else() -> None:
    """`VOCABULARY` is **derived** from the verbs (issue 74), so what this checks
    now is that every declared verb is actually routed.

    It used to be a hand-written set here beside the two tables — "two lists that
    must agree is one list checked", which was true as far as it went and left
    this set as a third place to edit. A table row cannot exist without a verb
    (the row constructor takes one), so the failure worth catching is the other
    direction: a verb declared in `ph_app.verbs` and given no handler is a name
    a client can spell, a reply model it can rely on, and an `unknown method`
    refusal at the first real call."""
    assert set(METHODS) | set(MUTATIONS) == VOCABULARY, (
        "a verb is declared but not routed to a handler, or routed under a name no verb declares"
    )
    assert not set(METHODS) & set(MUTATIONS), (
        "a name in both tables would be dispatched as a mutation and its METHODS row never reached"
    )


def test_a_row_is_routed_by_the_declared_verb_and_not_an_inline_one() -> None:
    """Identity, not equality — and that is the point of deriving `VOCABULARY`.

    A row takes a `Verb`, so nothing stops one being built inline:
    `Method(Verb("session/status", SessionParams, SomeOtherReply), handler)`
    type-checks, routes, and puts a name in the table that `ph_app.verbs` never
    declared — which would make `VOCABULARY` disagree with the tables in a way
    the test above reports as "a verb declared but not routed", pointing at the
    wrong half. Checking the object is the same one deleted that ambiguity.

    It also pins the property the two ends rely on: a client importing
    `verbs.SESSION_STATUS` and a server routing `verbs.SESSION_STATUS` are
    holding *one* value, so the reply model a caller is typed against is the
    reply model the handler was checked against.
    """
    declared = {verb.name: verb for verb in verbs.VOCABULARY_VERBS}
    for method, row in {**METHODS, **MUTATIONS}.items():
        assert row.verb is declared.get(method), (
            f"{method} is routed by a Verb that is not the one ph_app.verbs declares"
        )


def test_every_handler_answers_with_the_reply_its_verb_declares() -> None:
    """The row's central claim, guarded at runtime because a type cannot guard it.

    `Method(verb, handler)` is only checked against the verb because the tables
    go through the `_unkeyed`/`_mutating` builders, whose return type names
    neither type variable and so cannot receive an expected type. Rewrite the
    tables as the dict literal they look like — `{verb.name: Method(verb, h)}`
    under `dict[str, Method[Any, Any]]` — and mypy solves both variables to
    `Any`, every handler fits every verb, and **nothing fails**: not the type
    check, not the other 2,600 tests. That is issue 83, and this is the check
    that would have caught it.

    Read off the annotation rather than through `typing.get_type_hints`, which
    cannot resolve three of the mutation `act`s: their *third* parameter is a
    `TYPE_CHECKING`-only seam type (`CommandRegistry`,
    `PermissionPresetService`, `CredentialService`), and `get_type_hints`
    resolves every annotation or none. The return alone is what this asserts,
    so the return alone is what it evaluates.

    The four projections are skipped by name and not silently: `_projection` is
    generic, so its handler's return annotation is the type variable `N` and
    there is nothing to compare. They need no guard — `_projection` takes the
    verb and answers through `verb.read`, so the model cannot differ from the
    row's.

    `shutdown` is checked the other way round. It is the one `Announced` row,
    its verb is a `Notify` with no `reply` at all, and its handler must return
    `None` — so what this asserts there is that both halves still say nothing.
    """
    namespace = vars(server)
    projections = {"session/readings", "commands/list", "screens/list", "tools/list"}
    checked = 0
    for method, row in {**METHODS, **MUTATIONS}.items():
        answers = row.handle if isinstance(row, Method | Announced) else row.act
        declared = answers.__annotations__.get("return")
        if method in projections:
            assert declared == "N", f"{method} is no longer a generic projection; guard it here"
            continue
        if isinstance(row, Announced):
            assert declared is None or declared == "None", (
                f"{method} is an Announced row, so its handler must answer nothing, not {declared}"
            )
            assert not hasattr(row.verb, "reply"), f"{method}'s verb must be a Notify"
            checked += 1
            continue
        resolved = eval(declared, namespace) if isinstance(declared, str) else declared
        assert resolved is row.verb.reply, (
            f"{method} answers with {resolved} where its verb declares {row.verb.reply}"
        )
        checked += 1
    assert checked == len(METHODS) + len(MUTATIONS) - len(projections) == 24


def test_the_one_method_with_no_reply_is_a_notify_and_not_a_verb() -> None:
    """`shutdown` cannot be sent through `call`, and the reason is its type.

    A request awaiting a reply would leave the caller waiting on a frame the
    daemon is concurrently losing the ability to write, so "stop" is not a
    question. While `SHUTDOWN` was a `Verb` carrying an unread `ShutdownAck`,
    that rule lived only in prose: `client.call(SHUTDOWN, NoParams())`
    type-checked, and all nineteen call sites using `notify` were convention.

    The guarantee itself is a *type* one and `uv run mypy` is what enforces it —
    `call` takes `Verb`, `notify` takes `Notify`, and neither is a subtype of
    the other, so each door refuses the other's kind. What is asserted here is
    the structural fact underneath: swap `SHUTDOWN` back to a `Verb` and the
    check that matters is one mypy run away, but this fails immediately and by
    name.

    The `notify` half is worth the same care. A `Verb` accepted there would run
    a method on the daemon and silently drop its answer — a different way to be
    wrong about the same distinction.
    """
    assert isinstance(verbs.SHUTDOWN, Notify)
    assert not isinstance(verbs.SHUTDOWN, Verb), "a Notify must not satisfy the `call` door"
    assert not issubclass(Verb, Notify) and not issubclass(Notify, Verb), (
        "neither door may accept the other's kind, so neither type may subclass the other"
    )
    # And it is the only one: a second no-reply method is a decision, not an
    # accident, so it should arrive with its own reason rather than silently.
    assert [one.name for one in verbs.VOCABULARY_VERBS if isinstance(one, Notify)] == ["shutdown"]


def test_the_two_tuples_partition_the_vocabulary_the_way_the_tables_do() -> None:
    """`MUTATING` says which door a verb goes through — `mutate` stamps the
    idempotence key and `call` does not — so it has to agree with the table
    that actually applies the guard. A verb listed as unkeyed but routed as a
    mutation would lose the write-ahead guard that `MUTATIONS`' own docstring
    says cannot be claimed per handler by memory.

    `MUTATING` is the hand-written half and `UNKEYED` is derived as its
    complement, so this is the one direction still worth asserting.
    """
    assert {verb.name for verb in verbs.MUTATING} == set(MUTATIONS)
    assert {verb.name for verb in verbs.UNKEYED} == set(METHODS)


def test_every_method_about_one_root_requires_its_id() -> None:
    """The `str(params["sessionId"])` the row replaced was a `KeyError` waiting
    at twenty-one sites; the model makes the requirement one declaration."""
    daemon_level = {"initialize", "daemon/hello", "daemon/config", "daemon/status", "shutdown"}
    daemon_level |= {"sessions/list", "sessions/browse"}
    for method, row in {**METHODS, **MUTATIONS}.items():
        if method in daemon_level:
            assert not issubclass(row.verb.params, SessionParams), method
            continue
        assert issubclass(row.verb.params, SessionParams), method
        assert row.verb.params.model_fields["session_id"].is_required(), method
    # And every mutation carries the idempotence pair `_mutate` claims a key
    # from — a mutation whose params lacked it could not be guarded at all.
    assert all(issubclass(one.verb.params, MutationParams) for one in MUTATIONS.values())


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
        assert "never-mounted" not in daemon.running.supervisor.roots


async def test_an_unknown_method_is_still_its_own_refusal(tmp_path: Any) -> None:
    """The table lookup replaced the chain's last `else`; the sentence and the
    code a client branches on must not have moved with it."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        with pytest.raises(DaemonError) as refused:
            await client.call("session/nonsense", sessionId="x")
        assert refused.value.reason == "unknown_method"


# ------------------------------------------------------------- typed sends --


async def test_a_model_and_the_keyword_form_put_the_same_frame_on_the_wire(
    tmp_path: Any,
) -> None:
    """`call` has two doors and they must agree (P8-09, issue 74).

    The verb door is the one production uses — a misspelled field is a type
    error at the *sending* end rather than the daemon's `invalid_params`, and
    the params cannot belong to another method — and the keyword door stays for
    callers with no model to build, which is most of this file. A difference
    between them would make every test here evidence about a path nothing ships.

    Compared through `to_wire()` because the doors no longer answer with the
    same *type*: the verb door validates the reply into the verb's model, which
    is the half of issue 74 the keyword door cannot have (it has no verb to read
    the model from). The frame is what has to match, and it does.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call(verbs.SESSION_NEW, NewSessionParams(session_id="twinned"))

        typed = await client.call(verbs.SESSION_STATUS, SessionParams(session_id="twinned"))
        untyped = await client.call("session/status", sessionId="twinned")

        assert isinstance(typed, RootStatusReply), "the verb door names the model, not the caller"
        assert typed.to_wire() == untyped


async def test_a_mutation_is_keyed_by_the_client_without_the_caller_saying_so(
    tmp_path: Any,
) -> None:
    """`mutate` stamps `clientId`/`commandId` onto the model.

    The guard it buys is the point: a reconnecting client re-sends what it
    cannot know landed, and an unkeyed retry runs the effect twice. Before the
    stamp moved onto the model it was two keyword arguments every verb had to
    remember — and `prompt` was the only one that did.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call(verbs.SESSION_NEW, NewSessionParams(session_id="keyed"))

        first = await client.mutate(
            verbs.SESSION_PROMPT, PromptParams(session_id="keyed", prompt="once")
        )
        assert not isinstance(first, MutationRepeated), "the first send is not a repeat"

        # The same counter value cannot recur on one client, so a repeat has to
        # be sent as the frame a reconnect would send.
        again = await client.call(
            "session/prompt",
            sessionId="keyed",
            prompt="once",
            clientId=client.id,
            commandId="1",
        )
        assert again["repeated"] is True, "the stamp `mutate` applied is the one that guards"
