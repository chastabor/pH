"""`/profile show | diff | save | use | clear` (session profiles, S7).

The gate: tune a session, save it, start a headless run on the saved profile, and
the two environments match. Around it: `show` says what a session runs with and
why; `diff` lists how its named profile moved; `use` switches the base, keeping the
overrides or clearing them, and starts the root again on it; `clear` stops an
override live; and a profile saved and then used as the base clears the overrides
it now says itself (decision 3's own rule).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from daemon_helpers import Daemon, allowed_hosts, logged_types, row_disabled, run_command, running

from ph.keys import SESSIONS
from ph.paths import resolve_roots
from ph.session_profile import (
    BASE,
    CLEARED,
    SAVED,
    logged_environment,
    resolved_environment,
    saved_base,
)
from ph.testing import not_none, write_profile
from ph_app.daemon.client import DaemonClient
from ph_app.daemon.supervisor import Root
from ph_app.profiles import compose_profile, named_version
from ph_app.profiles_cli import adopt_version
from ph_app.runtime import prompted

pytestmark = pytest.mark.anyio

WORK = "extends: headless\nrows:\n  - id: tool-bash\n    disabled: true\n"


async def _command(client: DaemonClient, root_id: str, line: str) -> str:
    """One `/…` line over the wire — through the mutation, where a restart happens."""
    reply: dict[str, Any] = await client.call(
        "session/command", sessionId=root_id, line=line, clientId="c", commandId=line
    )
    return str(reply["shown"])


async def _sandbox(root: Root, line: str) -> None:
    shown = await run_command(root, line)
    assert "now reachable" in shown, shown


async def _attached(daemon: Daemon, session_id: str) -> tuple[DaemonClient, Root]:
    """A root on the named profile `headless`, and a client attached to it."""
    client = await daemon.client()
    await client.call("session/new", sessionId=session_id, profile="headless")
    await client.call("session/attach", sessionId=session_id)
    return client, daemon.held(session_id)


async def test_a_tuned_session_saved_starts_a_headless_run_in_the_same_environment(
    tmp_path: Path,
) -> None:
    """The gate. A session tuned by an allowance and a model, saved as a named
    profile; a headless run started on that profile records a base that is the
    tuned session's environment, setting for setting. Sabotage: leave the overrides
    out of `save_session`'s documents, and the headless run has neither."""
    async with running(tmp_path) as daemon:
        client, root = await _attached(daemon, "tuned")
        await _sandbox(root, "/sandbox allow host example.com")
        await client.call(
            "session/model",
            sessionId="tuned",
            choice={"provider": "fake", "model": "fake-7"},
            clientId="c",
            commandId="model",
        )

        shown = await _command(client, "tuned", "/profile save mine")

        assert "Saved this session's environment as mine" in shown, shown
        assert SAVED in logged_types("tuned")
        tuned = resolved_environment(logged_environment(root.session))
    async with prompted(compose_profile("mine"), "hi", session_id="headless-run") as (_, run):
        started = not_none(saved_base(run))

    assert list(started.rows) == tuned
    assert started.name == "mine"


async def test_show_names_the_base_the_overrides_and_what_they_change(tmp_path: Path) -> None:
    async with running(tmp_path) as daemon:
        client, root = await _attached(daemon, "shown")
        await _sandbox(root, "/sandbox allow host example.com")

        shown = await _command(client, "shown", "/profile show")

        assert shown.startswith("Base: headless")
        assert "/sandbox allow host example.com  (command)" in shown
        assert "+ example.com" in shown, "what the override changes, by setting"


async def test_diff_lists_how_the_named_profile_moved(tmp_path: Path) -> None:
    write_profile("work", WORK)
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="moved", profile="work")
        write_profile("work", "extends: headless\nrows: []\n")

        shown = await _command(client, "moved", "/profile diff")

        assert "tool-bash" in shown and "true → false" in shown and "(your profile)" in shown
        assert "`/profile use work`" in shown


async def test_use_switches_the_base_keeps_the_overrides_and_starts_the_root_again(
    tmp_path: Path,
) -> None:
    """Decision 5: swapping base profiles keeps the overrides. A base changes rows a
    live mount cannot follow, so the root starts again on it, and whoever was
    watching is watching the new one. Sabotage: drop `restart_wanted` in `_use`, and
    the root runs on as it was, `tool-bash` on."""
    write_profile("work", WORK)
    async with running(tmp_path) as daemon:
        client, root = await _attached(daemon, "switched")
        await _sandbox(root, "/sandbox allow host example.com")

        shown = await _command(client, "switched", "/profile use work")

        back = daemon.held("switched")
        assert back is not root, shown
        assert row_disabled(back, "tool-bash"), "work's"
        assert "example.com" in allowed_hosts(back), "the override, kept"
        assert back.describe().profile == "work"
        assert back.subscribers, "the client attached before is attached now"
        assert logged_types("switched").count(BASE) == 2


async def test_use_with_clear_starts_the_new_base_without_the_overrides(tmp_path: Path) -> None:
    """Decision 11: the person chooses, at the switch, whether the overrides go too.
    Sabotage: ignore `clear_all` in `switch_base`, and the allowance survives."""
    write_profile("work", WORK)
    async with running(tmp_path) as daemon:
        client, root = await _attached(daemon, "cleared")
        await _sandbox(root, "/sandbox allow host example.com")

        await _command(client, "cleared", "/profile use work --clear")

        back = daemon.held("cleared")
        assert "example.com" not in allowed_hosts(back)
        assert logged_environment(back.session).overrides == ()


async def test_a_profile_saved_then_used_clears_the_overrides_it_now_says(
    tmp_path: Path,
) -> None:
    """After a save that is then the base, the allowance is the base's own, so its
    override deviates from nothing and is cleared in the switch — and the host is
    still reachable, from the base."""
    async with running(tmp_path) as daemon:
        client, root = await _attached(daemon, "folded")
        await _sandbox(root, "/sandbox allow host example.com")
        await _command(client, "folded", "/profile save folded")

        await _command(client, "folded", "/profile use folded")

        back = daemon.held("folded")
        assert logged_environment(back.session).overrides == ()
        assert CLEARED in logged_types("folded")
        assert "example.com" in allowed_hosts(back)


async def test_clear_stops_an_override_on_the_live_mount(tmp_path: Path) -> None:
    """A config override is cleared on the running mount, no restart: the record on
    disk, then the row back to its base. Sabotage: skip `_converge` in
    `clear_overrides`, and the host stays reachable until the next start."""
    async with running(tmp_path) as daemon:
        client, root = await _attached(daemon, "live")
        await _sandbox(root, "/sandbox allow host example.com")

        shown = await _command(client, "live", "/profile clear sandbox-allow")

        assert daemon.held("live") is root, "not restarted"
        assert "example.com" not in allowed_hosts(root), shown
        assert CLEARED in logged_types("live")
        assert await _command(client, "live", "/profile clear") == "This session has no overrides."


async def test_save_refuses_a_file_that_is_there_and_a_name_that_is_not_one(
    tmp_path: Path,
) -> None:
    write_profile("work", WORK)
    async with running(tmp_path) as daemon:
        client, _root = await _attached(daemon, "careful")

        taken = await _command(client, "careful", "/profile save work")
        odd = await _command(client, "careful", "/profile save ../elsewhere")
        replaced = await _command(client, "careful", "/profile save work --replace")

        assert "exists; `--replace` writes over it" in taken
        assert "is not a profile name" in odd
        assert "Saved" in replaced
        assert resolve_roots().profile_overlay("work").read_text().count("tool-bash") == 0


async def test_a_save_that_would_not_compose_back_is_taken_back(tmp_path: Path) -> None:
    """The file is composed again before it is kept. An old drop-in under the name —
    read until folded — makes it compose to something else, so nothing is saved.
    Sabotage: skip the round trip in `save_session`, and the file stays."""
    dropins = resolve_roots().profile_dropins("stale")
    dropins.mkdir(parents=True)
    (dropins / "sandbox.yaml").write_text("- id: tool-fs\n  disabled: true\n")
    async with running(tmp_path) as daemon:
        client, _root = await _attached(daemon, "careful")

        shown = await _command(client, "careful", "/profile save stale")

        assert "not saved" in shown and "different environment" in shown, shown
        assert not resolve_roots().profile_overlay("stale").exists()
        assert SAVED not in logged_types("careful")


async def test_a_fork_started_as_a_root_takes_a_version_adopted_for_it(tmp_path: Path) -> None:
    """A fork names its parent as a child does, and was passed over by `opened` as
    one: adopted, it never switched. By `origin`, it is a session of its own.
    Sabotage: test `parent_session` in `SessionHeader.is_subagent` again, and the
    fork's base stays."""
    write_profile("work", WORK)
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
        client = await daemon.client()
        await client.call("session/new", sessionId="trunk", profile="work")
        trunk = daemon.held("trunk")
        sessions = trunk.ctx.require(SESSIONS)
        fork = sessions.fork(trunk.session, child_session_id="branch")
        await sessions.flush(fork)
        write_profile("work", "extends: headless\nrows: []\n")
        lines, missed = await adopt_version("work", ["branch"], named_version("work"))
        assert not missed, lines

        branch = await supervisor.start("branch")

        assert not row_disabled(branch, "tool-bash"), "the adopted version"
        assert logged_environment(branch.session).adopted is None, "and made the base"
