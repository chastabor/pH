"""Profiles per session (session profiles, S5).

A root is mounted from its own log's environment — its base, then its overrides —
and a new one from the profile it was asked for, so a daemon's `--profile` is the
default for a new session and not the profile of every root. The gate: two roots
on two profiles in one daemon, and a root that goes idle and comes back keeps its
overrides because they are in what it is mounted from, not re-applied after —
`test_session_overrides.test_a_session_s_next_start_puts_its_overrides_back`
holds that half, beside the override it keeps.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from daemon_helpers import row_disabled, running
from typer.testing import CliRunner

from ph.json import as_obj, as_seq
from ph.keys import SESSIONS
from ph.paths import resolve_roots
from ph.session import now_ms
from ph.testing import logged_events, not_none, searches_into, write_profile
from ph_app.cli import app
from ph_app.modes import run_rpc
from ph_app.profiles import compose_profile
from ph_app.protocol import DaemonError
from ph_app.sessions import recorded_start, stored_on

pytestmark = pytest.mark.anyio

runner = CliRunner()


async def test_two_roots_on_two_profiles_share_one_daemon(tmp_path: Path) -> None:
    """The first half of the gate: `session/new` names the profile a new session is
    created on, and each root is mounted in its own."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()

        plain = await client.call("session/new", sessionId="plain", profile="headless")
        posed = await client.call("session/new", sessionId="posed", profile="tui")

        assert (plain["profile"], posed["profile"]) == ("headless", "tui")
        assert row_disabled(daemon.held("plain"), "tool-ask-user")
        assert not row_disabled(daemon.held("posed"), "tool-ask-user"), "tui arms it"


async def test_a_root_comes_back_on_the_profile_its_log_records(tmp_path: Path) -> None:
    """Released and started again with no profile asked for — which would be the
    daemon's default — it is mounted as its log says. Sabotage: mount `requested`
    in `session_profile` whatever the log holds, and it comes back on `headless`."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        supervisor = daemon.running.supervisor
        await client.call("session/new", sessionId="kept", profile="tui")
        await supervisor.passivate(daemon.held("kept"), now=now_ms())

        back = await supervisor.start("kept")

        assert back.describe().profile == "tui"
        assert not row_disabled(back, "tool-ask-user")


async def test_a_profile_that_does_not_compose_is_refused_by_name(tmp_path: Path) -> None:
    async with running(tmp_path) as daemon:
        client = await daemon.client()

        with pytest.raises(DaemonError, match='profile "nope" does not compose'):
            await client.call("session/new", sessionId="refused", profile="nope")


async def test_a_session_whose_profile_went_away_still_runs_on_its_log(tmp_path: Path) -> None:
    """The base holds the environment, so a named profile renamed or removed since
    costs the session nothing but the host rows, which the daemon's own serve."""
    write_profile("gone", "extends: headless\nrows:\n  - id: tool-bash\n    disabled: true\n")
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        supervisor = daemon.running.supervisor
        await client.call("session/new", sessionId="orphan", profile="gone")
        await supervisor.passivate(daemon.held("orphan"), now=now_ms())
        resolve_roots().profile_overlay("gone").unlink()

        back = await supervisor.start("orphan")

        assert row_disabled(back, "tool-bash"), "from the log's base"


async def test_a_fork_is_mounted_from_the_base_its_prefix_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork's file continues its root's from `seed_length`, and the base is in that
    prefix, so reading the fork's own file alone found no base and it mounted what
    was asked. Read through the lineage, it is its root's environment — and a
    session on the profile, for `phern profiles adopt`. The lineage is read from the
    directory the fork was found in: given only the id, the walk searched the store
    for the fork's own file a second time.

    Sabotage: drop the lineage read in `_environment_at`, and the fork has no base;
    drop its `family`, and the fork is searched for twice.
    """
    from ph.persistence import jsonl

    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="trunk", profile="tui")
        trunk = daemon.held("trunk")
        fork = trunk.ctx.require(SESSIONS).fork(trunk.session, child_session_id="branch")
        await trunk.ctx.require(SESSIONS).flush(fork)
        searched: list[str] = []
        monkeypatch.setattr(jsonl, "locate_under", searches_into(searched))

        start = recorded_start(resolve_roots().sessions_dir(), "branch")

        assert not_none(start.environment.base).name == "tui"
        assert start.family == fork.header.family
        assert searched == ["branch"]
        assert "branch" in [one for one, _ in stored_on(resolve_roots().sessions_dir(), "tui")]


def test_a_resumed_one_shot_session_runs_on_its_own_profile() -> None:
    """`phern -p --session x` again, with no `--profile`: the default would be
    `headless`, and the session was created on `tui` — so its tools say `tui`."""
    first = runner.invoke(app, ["-p", "hi", "--session", "shot", "--profile", "tui"])
    assert first.exit_code == 0, first.output
    again = runner.invoke(app, ["-p", "again", "--session", "shot"])
    assert again.exit_code == 0, again.output

    headers = [event for event in logged_events("shot") if event.type == "request/header"]
    tools = {
        as_obj(tool).get("name")
        for tool in as_seq(as_obj(as_obj(headers[-1].data)["header"]).get("tools"))
    }
    assert "ask_user" in tools, "the resumed run mounted tui, which arms the question tool"


async def test_rpc_gives_each_session_a_mount_of_its_own() -> None:
    """One session's route is an override on its own mount, so a second session on
    the same rpc endpoint runs on its profile's default — and its log is its own.
    Sabotage: share one mount across sessions again, and the second runs on fake-9."""
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/prompt",
            "params": {"sessionId": "first", "prompt": "hi", "provider": "fake", "model": "fake-9"},
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {"sessionId": "second", "prompt": "hi"},
        },
    ]
    stdin = io.StringIO("".join(f"{json.dumps(one)}\n" for one in requests))

    await run_rpc(compose_profile("headless"), stdin=stdin, out=io.StringIO())

    def model_of(session_id: str) -> object:
        contexts = [event for event in logged_events(session_id) if event.type == "request/context"]
        return as_obj(contexts[-1].data).get("model")

    assert model_of("first") == "fake-9"
    assert model_of("second") == "fake-1"
