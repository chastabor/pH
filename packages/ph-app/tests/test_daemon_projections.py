"""P5-14 — what a front end reads off a root it cannot reach into.

`PHTuiApp` resolved nine seams out of `ctx` because the harness was in its own
process. Over a socket none of that is reachable, so each one becomes a method
here — and the risk this file exists for is **a projection that quietly says
less than the seam does.** A dropped field is invisible: the UI renders, the
palette fills, and one command is missing from it, or a footer reading that only
appears under load never appears at all.

So every gate below compares the projection against the *seam's own answer* in
the same mount rather than against a literal. A literal passes for a projection
that has been out of date since somebody added a field to `CommandDefinition`.
"""

from __future__ import annotations

from typing import Any

import pytest
from daemon_helpers import running, until

from ph.bundles import BASE, HEADLESS
from ph.cordis import DEPLOYMENT, Profile, load_profile_documents
from ph.seams.commands import CommandDefinition
from ph.seams.tui_status import StatusField, StatusReading
from ph.session import SessionEvent
from ph.testing import RecordedStep, ReplayAdapter, simple_tool, text_chunks, tool_call_chunks
from ph.tools import ToolCallView, ToolResultView
from ph_app.protocol import DaemonError
from ph_app.tui.adapter import TuiEventAdapter

pytestmark = pytest.mark.anyio


async def _furnished(daemon: Any, session_id: str = "projected") -> Any:
    """A root with one status field and one command registered into it.

    Registered here rather than relied on from the profile, and that is the
    point: these gates are about the *projection*, so they must not pass or fail
    on which rows the daemon's profile happens to mount. A field contributed by
    hand also lets the comparison be non-vacuous — an empty list equals an empty
    list, and proves nothing about either.
    """
    root = await daemon.root(session_id)
    root.ctx.tui_status.register(
        StatusField(id="probe", read=lambda session: StatusReading(text="probe", level="warn"))
    )
    root.ctx.commands.register(
        CommandDefinition(
            name="probe",
            summary="a command registered by the test",
            run=lambda argument, ctx: f"ran with {argument!r}",
            argument_hint="<thing>",
        )
    )
    return root


# ------------------------------------------------------------- the footer --


async def test_status_readings_over_the_wire_equal_the_seams_readings(
    tmp_path: Any,
) -> None:
    """The gate this increment is named for.

    Against `ctx.tui_status` in the same mount, so a field a row contributes and
    the projection does not carry is a failure here rather than a footer that is
    silently shorter in the browser than in the terminal.

    Sabotage: drop `level` from the projection — every reading renders `normal`,
    and a warning stops looking like a warning.
    """
    async with running(tmp_path) as daemon:
        root = await _furnished(daemon)
        client = await daemon.client()

        reply = await client.call("session/readings", sessionId=root.id)

        seam = root.ctx.get("tui_status").readings(root.session)
        assert reply["readings"] == [{"text": one.text, "level": one.level} for one in seam]
        assert {"text": "probe", "level": "warn"} in reply["readings"], (
            "the contributed field must survive the projection, level included"
        )


async def test_readings_ride_the_status_notification(tmp_path: Any) -> None:
    """Pushed when the agent moves, because that is when they can have changed.

    A reading is a fold of the log, so the moment worth recomputing is an append
    — not the TUI's 30 Hz tick, which exists for the spinner and would ask this
    thirty times a second to get the same answer.

    Sabotage: send `session.status` without `readings`, and a browser tab shows a
    footer frozen at whatever it held when it attached.
    """
    async with running(tmp_path) as daemon:
        seen: list[dict[str, Any]] = []
        client = await daemon.client(
            on_notify=lambda method, params: (
                seen.append(params) if method == "session.status" else None
            )
        )
        root = await _furnished(daemon, "pushed")
        await client.call("session/attach", sessionId=root.id)

        await daemon.server.supervisor.prompt(root.id, "hello")
        # The turn runs in the root's own task, so the notification arrives on
        # its schedule rather than this one's. Waiting on the fact beats sleeping
        # for a guess.
        await until(lambda: bool(seen), what="a session.status notification")
        with_readings = [one for one in seen if "readings" in one]

        assert with_readings, "no status notification carried readings"
        assert all({"text": "probe", "level": "warn"} in one["readings"] for one in with_readings)


# ------------------------------------------------------- the palette et al --


async def test_the_command_list_is_the_registrys_own(tmp_path: Any) -> None:
    """Every command, and the hint a person needs to type one.

    `run` is deliberately absent — it is a callable, and the client's job is to
    offer the command and send the line back. Asserting the *names* against the
    registry is what catches a projection that filtered or truncated.
    """
    async with running(tmp_path) as daemon:
        root = await _furnished(daemon)
        client = await daemon.client()

        reply = await client.call("commands/list", sessionId=root.id)

        registry = root.ctx.get("commands").list()
        assert [one["name"] for one in reply["commands"]] == [one.name for one in registry]
        assert {
            "name": "probe",
            "summary": "a command registered by the test",
            "argumentHint": "<thing>",
        } in reply["commands"], "every field a palette shows must survive"
        assert all("run" not in one for one in reply["commands"]), "a callable cannot travel"


async def test_the_screen_list_carries_what_a_palette_needs_and_no_body(
    tmp_path: Any,
) -> None:
    """`build` stays in the client; the rest is what orders and labels an entry.

    Saying so is the point: a projection that silently dropped `build` would look
    complete, and P5-15's declarative body is deferred to P7-07 rather than
    absent by oversight.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("projected")
        client = await daemon.client()

        reply = await client.call("screens/list", sessionId=root.id)

        registry = root.ctx.get("tui_screens").list()
        assert [one["id"] for one in reply["screens"]] == [one.id for one in registry]
        for wire, screen in zip(reply["screens"], registry, strict=True):
            assert wire == {
                "id": screen.id,
                "label": screen.label,
                "order": screen.order,
                "key": screen.key,
            }


async def test_the_tool_list_matches_what_the_deployment_offers(tmp_path: Any) -> None:
    """The same answer `--mode rpc` gives, against the same scope.

    `DEPLOYMENT` and not an agent's view: a front end is asking what this
    deployment can do, not what one agent was narrowed to.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("projected")
        client = await daemon.client()

        reply = await client.call("tools/list", sessionId=root.id)

        schemas = root.ctx.tools.schemas(scope=DEPLOYMENT)
        assert [one["name"] for one in reply["tools"]] == [one.name for one in schemas]


async def test_the_config_rows_are_the_composed_profile(tmp_path: Any) -> None:
    """A property of the daemon, not of any root: every root mounts this."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()

        reply = await client.call("daemon/config")

        assert reply["rows"] == list(daemon.server.supervisor.profile.dump())
        assert reply["rows"], "an empty profile would make this vacuous"


# ------------------------------------------------------------ acting on it --


async def test_a_command_runs_in_the_root_and_lands_in_its_log(tmp_path: Any) -> None:
    """Run by the daemon, because that is where the seams a body reaches are.

    And recorded in *this session's* log, so every other attached UI sees that
    somebody ran it — the same rule the composer follows.
    """
    async with running(tmp_path) as daemon:
        root = await _furnished(daemon)
        client = await daemon.client()

        shown = await client.call("session/command", sessionId=root.id, line="/probe here")

        assert shown["shown"] == "ran with 'here'", "the body ran, in the root's own context"

        assert [one.type for one in root.session.events].count("command/run") == 1


async def test_running_a_command_twice_with_one_id_runs_it_once(tmp_path: Any) -> None:
    """Idempotent through `root.remember`, the way `session/prompt` is.

    A client that reconnects and retries cannot tell whether its first call
    landed, and `/compact` is not a verb to run twice on a guess.

    Sabotage: drop the `accepted` check, and the log shows two runs.
    """
    async with running(tmp_path) as daemon:
        root = await _furnished(daemon)
        client = await daemon.client()
        once = {"sessionId": root.id, "line": "/probe", "clientId": "c", "commandId": "1"}

        first = await client.call("session/command", **once)
        again = await client.call("session/command", **once)

        assert first.get("repeated") is not True
        assert again["repeated"] is True
        assert [one.type for one in root.session.events].count("command/run") == 1


async def test_a_credential_is_stored_without_its_value_reaching_the_log_or_the_reply(
    tmp_path: Any,
) -> None:
    """The secret is used and not kept — anywhere a reader could reach it.

    Both halves are asserted because they fail independently: a value echoed in
    the reply leaks to whoever holds the socket, and a value in the log leaks to
    everyone who ever reads the session, forever.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("projected")
        client = await daemon.client()
        secret = "sk-do-not-log-me"

        reply = await client.call(
            "credentials/store", sessionId=root.id, name="ANTHROPIC_API_KEY", value=secret
        )

        assert reply == {"sessionId": root.id, "name": "ANTHROPIC_API_KEY", "stored": True}
        assert secret not in repr([one.to_wire() for one in root.session.events])
        held = await client.call(
            "credentials/held", sessionId=root.id, names=["ANTHROPIC_API_KEY", "OTHER"]
        )
        assert held["held"] == {"ANTHROPIC_API_KEY": True, "OTHER": False}


async def test_a_new_session_records_the_clients_cwd_in_its_header(tmp_path: Any) -> None:
    """Where the person is, not where the daemon is.

    The daemon's own working directory is somewhere neither the person nor their
    files are, so a session that inherited it would be labelled with a lie. The
    header validates that the path is absolute, which is why the client sends a
    resolved one.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()

        await client.call("session/new", sessionId="homed", cwd=str(tmp_path))

        root = daemon.server.supervisor.roots["homed"]
        assert root.session.header.cwd == str(tmp_path)


async def test_a_relative_cwd_is_refused_rather_than_resolved(tmp_path: Any) -> None:
    """Resolving it would resolve it against the *daemon's* directory.

    Which is the one directory that is certainly wrong. `SessionHeader` already
    holds this rule; the gate is that the wire does not route around it.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()

        with pytest.raises(DaemonError):
            await client.call("session/new", sessionId="relative", cwd="./somewhere")


# ----------------------------------------------------------- the card views --


async def _with_a_card(daemon: Any, session_id: str) -> Any:
    """A root that has run one tool call, through the real loop.

    Through the loop and not by hand, because the thing under test is the
    *link*: `tool/result` finds its call through `source_event_seqs`, which only
    the real append path sets. A hand-built pair would pass while the daemon
    could not render a single real result.
    """
    root = await daemon.root(session_id)
    # The shipped replay adapter rather than a hand-rolled streamer: the chunk
    # protocol has one statement in `ph.testing`, and a third copy of it is a
    # third thing to fix when it moves. Registered under `fake`, which is the
    # provider the daemon gives every root, so it shadows it for this one.
    root.ctx.llm.register_adapter(
        ("fake",),
        ReplayAdapter(
            steps=[
                RecordedStep(turn=1, step=1, chunks=tool_call_chunks("c1", "ping", "{}")),
                RecordedStep(turn=1, step=2, chunks=text_chunks("done")),
            ]
        ),
    )
    root.ctx.tools.register(
        simple_tool(
            "ping",
            lambda _args, _run: "pong",
            present_call=lambda args: ToolCallView(card="terminal", title="Ping", input="ping!"),
            present_result=lambda args, result: ToolResultView(
                card="terminal", title="Ping", subtitle="pong"
            ),
        )
    )
    return root


async def test_a_relayed_tool_call_carries_the_view_the_tool_would_have_rendered(
    tmp_path: Any,
) -> None:
    """The one thing the adapter needed `ctx.tools` for, sent instead.

    A front end over a socket has no registry to ask, so without this every card
    renders `generic` with its raw arguments — a visibly worse terminal for being
    remote, which is the split this plan exists to close.

    Sabotage: stop attaching `presentation` in `relay`, and the frames arrive
    with events and nothing to title them by.
    """
    async with running(tmp_path) as daemon:
        frames: list[dict[str, Any]] = []
        client = await daemon.client(
            on_notify=lambda method, params: (
                frames.append(params) if method == "session.event" else None
            )
        )
        root = await _with_a_card(daemon, "carded")
        await client.call("session/attach", sessionId=root.id)

        await root.agent.prompt("go")
        await until(
            lambda: any(one["event"]["type"] == "tool/result" for one in frames),
            what="the tool result to be relayed",
        )

        called = next(one for one in frames if one["event"]["type"] == "tool/call")
        settled = next(one for one in frames if one["event"]["type"] == "tool/result")
        assert called["presentation"] == {"card": "terminal", "title": "Ping", "input": "ping!"}
        # The result view is the harder half: `present_result` takes the *call\'s*
        # arguments, and only the linked `tool/call` event carries them.
        assert settled["presentation"] == {
            "card": "terminal",
            "title": "Ping",
            "subtitle": "pong",
            "isError": False,
        }


async def test_a_snapshot_page_carries_the_same_views_as_the_live_stream(
    tmp_path: Any,
) -> None:
    """A transcript must not look different on replay than it did live.

    One client attached before the turn and one paging in afterwards see one
    session; a card that titled itself differently between them would be the
    multiplex design failing at the only place a person would notice.
    """
    async with running(tmp_path) as daemon:
        root = await _with_a_card(daemon, "paged")
        await root.agent.prompt("go")
        client = await daemon.client()

        page = await client.call("session/snapshot", sessionId=root.id)

        # Sparse and keyed by seq: a page is 2048 events and a turn contributes
        # a handful of cards, so a positional list would be mostly `null`.
        views = {
            event["type"]: page["presentations"][str(event["seq"])]
            for event in page["events"]
            if str(event["seq"]) in page["presentations"]
        }
        assert views["tool/call"]["title"] == "Ping"
        assert views["tool/result"]["subtitle"] == "pong"
        assert set(views) == {"tool/call", "tool/result"}, "nothing else carries a view"


async def test_an_event_that_is_not_a_card_carries_no_view(tmp_path: Any) -> None:
    """Most events are not tool calls, and the relay must not pay for them.

    Also the honest reading of "derived, never appended": a view renders a call,
    so an event that is not one has nothing to render and says so with `None`
    rather than an empty object a client would have to distinguish.
    """
    async with running(tmp_path) as daemon:
        root = await _with_a_card(daemon, "plain")
        await root.agent.prompt("go")
        client = await daemon.client()

        page = await client.call("session/snapshot", sessionId=root.id)

        keyed = {int(seq) for seq in page["presentations"]}
        cards = {
            one["seq"] for one in page["events"] if one["type"] in ("tool/call", "tool/result")
        }
        assert keyed == cards and len(cards) == 2, "only cards are keyed, and the turn made two"


def test_the_adapter_prefers_the_daemons_view_and_falls_back_to_the_tool() -> None:
    """The two sources never compete, and the fallback still works.

    A sidecar arrives exactly when there is no registry to ask, so precedence is
    not really a contest — but stating it as one is what keeps a remote front end
    from silently rendering the generic card if a registry is ever present too.
    Validated rather than trusted: a sidecar that does not parse falls back
    rather than drawing a wrong card.
    """
    event = SessionEvent.from_wire(
        {
            "type": "tool/call",
            "seq": 0,
            "time": 0,
            "data": {"callId": "c1", "name": "ping", "arguments": "{}"},
        }
    )

    rendered = TuiEventAdapter()
    rendered.apply(event, live=False, presentation={"title": "From the daemon", "card": "terminal"})
    junk = TuiEventAdapter()
    junk.apply(event, live=False, presentation={"title": 17, "unexpected": True})

    card = next(item.tool for item in rendered.state.items if item.tool is not None)
    assert card.title == "From the daemon" and card.card == "terminal"
    fallback = next(item.tool for item in junk.state.items if item.tool is not None)
    assert fallback.title == "ping", "an unparseable view renders the plain card, not a wrong one"


async def test_a_preset_switch_is_applied_and_recorded(tmp_path: Any) -> None:
    """The permission posture, changed from a UI that is not in this process.

    Recorded in the log rather than held on the connection, so every other
    attached front end sees the switch and a resume comes back to it — the same
    rule the composer and `!!` follow.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("presets")
        client = await daemon.client()

        reply = await client.call("session/preset", sessionId=root.id, preset="workspace-write")

        assert reply["preset"] == "workspace-write"
        assert [one.type for one in root.session.events].count("permission/preset") == 1


async def test_a_method_whose_seam_is_absent_says_so_and_is_not_unknown(
    tmp_path: Any,
) -> None:
    """ "This deployment does not do that" is not "this daemon is too old".

    Two different sentences with two different client responses: `unknown_method`
    means disable the feature everywhere, `seam_absent` means grey out one button
    for one root. The read-side projections already answer absence with an empty
    list; this is the act side agreeing.

    Sabotage: raise `UnknownMethod` for a missing seam, and a client that met one
    root without credentials stops offering login for every other root too.
    """
    # A deployment that never mounted the seam, which is the only honest way to
    # reach the branch: a mounted root cannot have one taken away, because
    # `ctx.provide` refuses a second claim in the same realm.
    bare = Profile.from_documents(
        [
            *load_profile_documents([BASE, HEADLESS]),
            ("test", [{"id": "credentials", "remove": True}]),
        ]
    )
    async with running(tmp_path, profile=bare) as daemon:
        root = await daemon.root("bare")
        client = await daemon.client()

        with pytest.raises(DaemonError) as refused:
            await client.call(
                "credentials/store", sessionId=root.id, name="ANTHROPIC_API_KEY", value="x"
            )

        assert refused.value.reason == "seam_absent", (
            "the client must be able to branch on this without matching message text"
        )
        assert refused.value.reason != "unknown_method"
