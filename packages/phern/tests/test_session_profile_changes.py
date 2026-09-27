"""A changed named profile on restart, and adopting on purpose (session profiles, S6).

A session runs on the version of its named profile it started on. When the file
has moved since, a start compares the two and lists the difference, attributed. A
person's own edit, with a person there to ask, holds the root until they decide;
a start nobody attends — a schedule's — keeps the saved version, and so does a
change that is pH's alone. "No" starts it unchanged; "yes" makes the new version
the base, keeps the session's overrides over it, then this start's options. And
`phern profiles adopt` takes the new version ahead of time, for every session on
the profile, each keeping its own overrides.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from anyio import wait_all_tasks_blocked
from daemon_helpers import (
    Daemon,
    allowed_hosts,
    logged_types,
    row_disabled,
    run_command,
    running,
    until,
)
from typer.testing import CliRunner

from ph.json import as_obj, as_seq
from ph.llm import retry
from ph.paths import resolve_roots
from ph.session import now_ms
from ph.session_profile import ADOPTED, BASE, DECLINED, LoggedEnvironment, logged_environment
from ph.testing import logged_events, not_none, write_profile
from ph_app.cli import app
from ph_app.daemon.client import DaemonClient
from ph_app.daemon.supervisor import Root
from ph_app.payloads import ProfileAskReply, ProfileDecision
from ph_app.profiles import named_version, profile_or_exit
from ph_app.profiles_cli import adopt_version
from ph_app.sessions import recorded_environment

pytestmark = pytest.mark.anyio

runner = CliRunner()

WORK = "extends: headless\nrows:\n  - id: tool-bash\n    disabled: true\n"
EDITED = (
    "extends: headless\nrows:\n  - id: tool-bash\n    disabled: false\n"
    "  - id: tool-result-offload\n    disabled: true\n"
)


def _environment(session_id: str) -> LoggedEnvironment:
    """What the session's log on disk says about its environment."""
    return recorded_environment(resolve_roots().sessions_dir(), session_id)


async def _created_then_released(daemon: Daemon, session_id: str) -> Root:
    """A root on `work`, its base on disk, and released — as a daemon that stopped."""
    client = await daemon.client()
    await client.call("session/new", sessionId=session_id, profile="work")
    supervisor = daemon.running.supervisor
    root = daemon.held(session_id)
    await supervisor.passivate(root, now=now_ms())
    return root


async def _person(
    daemon: Daemon, decision: ProfileDecision, seen: list[dict[str, Any]]
) -> DaemonClient:
    """A front end that declares `asks` and answers every `profile/ask` with `decision`."""

    async def answer(params: dict[str, Any]) -> ProfileAskReply:
        seen.append(params)
        return ProfileAskReply(decision=decision)

    client = await daemon.client("asks")
    client.handlers["profile/ask"] = answer
    return client


async def test_an_edited_profile_holds_a_root_started_for_a_person_and_not_for_a_schedule(
    tmp_path: Path,
) -> None:
    """The first half of the gate. The person's file moved while nothing ran: a start
    for a front end that can ask is held, with the list naming the edited setting;
    a start with nobody to ask — `rehydrate`'s, for a schedule — keeps the version
    it has, and says how far behind it is. Sabotage: hold whenever the profile
    moved, `asks` or not, and the scheduled root is held too."""
    write_profile("work", WORK)
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
        await _created_then_released(daemon, "attended")
        await _created_then_released(daemon, "scheduled")
        write_profile("work", EDITED)

        held = await supervisor.start("attended", asks=True)
        kept = await supervisor.start("scheduled")

        assert held.status == "needs-profile-decision"
        assert kept.status == "idle"
        assert kept.describe().profile_changes == 2
        assert row_disabled(kept, "tool-bash"), "the version it started on"
        seen: list[dict[str, Any]] = []
        client = await _person(daemon, "later", seen)
        await client.call("session/attach", sessionId="attended")
        await until(lambda: bool(seen), what="the profile ask")
        listing = seen[0]["listing"]
        assert "tool-bash" in listing and "true → false" in listing and "(your profile)" in listing


async def test_a_moved_ph_default_alone_asks_nothing_and_is_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision 14: pH moved a default beneath a file nobody edited. No question —
    the session keeps its saved settings — and the list says whose change it is.
    Sabotage: hold on any change rather than a person's, and this root is held."""
    write_profile("work", WORK)
    async with running(tmp_path) as daemon:
        await _created_then_released(daemon, "upgraded")

        class Moved(retry.Config):
            max_attempts: int = 5

        spec = retry.apply.__ph_plugin__  # type: ignore[attr-defined]
        monkeypatch.setattr(retry.apply, "__ph_plugin__", replace(spec, config_model=Moved))
        root = await daemon.running.supervisor.start("upgraded", asks=True)

        assert root.status == "idle" and root.describe().profile_changes == 1
        shown = runner.invoke(app, ["profiles", "diff", "work", "--session", "upgraded"])
        assert shown.exit_code == 0, shown.output
        assert "llm-retry" in shown.output and "3 → 5" in shown.output
        assert "(pH)" in shown.output or "(pH " in shown.output


async def test_no_starts_it_unchanged_and_is_not_asked_again(tmp_path: Path) -> None:
    """ "Keep" is recorded, so the same version is not asked about at the next start.
    The prompt that arrived while it was held runs then, on the saved version.
    Sabotage: drop `record_declined`, and the next start is held again."""
    write_profile("work", WORK)
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
        await _created_then_released(daemon, "kept")
        write_profile("work", EDITED)
        seen: list[dict[str, Any]] = []
        client = await _person(daemon, "keep", seen)

        await client.call("session/attach", sessionId="kept")
        root = daemon.held("kept")
        await until(lambda: not root.held_on_profile, what="the decision")

        assert row_disabled(root, "tool-bash"), "unchanged"
        assert DECLINED in logged_types("kept")
        await client.call("session/detach", sessionId="kept")
        await supervisor.passivate(root, now=now_ms())
        again = await supervisor.start("kept", asks=True)
        assert again.status == "idle", "that version was declined"
        assert again.describe().profile_changes == 2, "and is still listed"


async def test_a_prompt_held_with_the_root_runs_once_it_is_decided(tmp_path: Path) -> None:
    """Held like a missing credential: a prompt waits in the inbox rather than
    running on either version, and the decision rings it. Sabotage: leave the hold
    out of `_run`'s check, and the turn runs before anybody has answered."""
    write_profile("work", WORK)
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
        await _created_then_released(daemon, "waiting")
        write_profile("work", EDITED)
        root = await supervisor.start("waiting", asks=True)

        await supervisor.prompt("waiting", "hello")
        # Every task given its turn: the doorbell has rung, and a root that did not
        # honor its hold would be mid-turn by now.
        await wait_all_tasks_blocked()
        assert root.status == "needs-profile-decision"
        assert "turn/start" not in logged_types("waiting") and "turn/start" not in [
            event.type for event in root.session.events
        ]
        seen: list[dict[str, Any]] = []
        client = await _person(daemon, "later", seen)
        await client.call("session/attach", sessionId="waiting")

        await until(
            lambda: any(event.type == "turn/end" for event in root.session.events),
            what="the held turn",
        )
        assert logged_types("waiting").count(DECLINED) == 0, "`later` records nothing"


async def test_yes_takes_the_new_version_keeps_the_overrides_then_the_start_options(
    tmp_path: Path,
) -> None:
    """The gate's "yes": the new version is saved with the session and the switch is
    logged; the session's override still applies over it; and this daemon's own
    start option comes last, over both. The front end attached is carried to the
    root that replaces the held one. Sabotage: drop the start documents from
    `session_profile`'s rebuild, and `tool-bash` runs as the new version says."""
    write_profile("work", WORK)
    daemon_profile = profile_or_exit("headless", ["{id: tool-bash, disabled: true}"])
    async with running(tmp_path, profile=daemon_profile) as daemon:
        supervisor = daemon.running.supervisor
        client = await daemon.client()
        await client.call("session/new", sessionId="moved", profile="work")
        first = daemon.held("moved")
        shown = await run_command(first, "/sandbox allow host example.com")
        assert "now reachable" in shown
        await supervisor.passivate(first, now=now_ms())
        write_profile("work", EDITED)
        seen: list[dict[str, Any]] = []
        person = await _person(daemon, "adopt", seen)

        await person.call("session/attach", sessionId="moved")
        held = daemon.held("moved")
        await until(lambda: supervisor.roots.get("moved") not in (None, held), what="the remount")
        back = daemon.held("moved")

        assert row_disabled(back, "tool-result-offload"), "the new version"
        assert "example.com" in allowed_hosts(back), "the override, kept"
        assert row_disabled(back, "tool-bash"), "the daemon's start option, last"
        bases = [event for event in logged_events("moved") if event.type == BASE]
        (adopted,) = [event for event in logged_events("moved") if event.type == ADOPTED]
        assert len(bases) == 2 and adopted.seq < bases[-1].seq, "accepted, then switched to"
        rows = {as_obj(row).get("id"): as_obj(row) for row in as_seq(bases[-1].data.get("rows"))}
        assert rows["tool-bash"]["disabled"] is False, "the base is the version as adopted"
        assert back.subscribers, "the person attached is watching the new root"
        assert back.status == "idle"


def test_adopt_moves_every_session_on_the_profile_and_each_keeps_its_overrides() -> None:
    """`phern profiles adopt`, with nothing running: each session on the profile gets
    the version, applied at its next start, over its own overrides. Sabotage: fold
    the overrides "since the latest base" again, and each loses its own."""
    write_profile("work", WORK)
    for session_id, row in (("first", "tool-fs"), ("second", "tool-attach")):
        started = runner.invoke(
            app,
            [
                "-p",
                "hi",
                "--session",
                session_id,
                "--profile",
                "work",
                "--patch",
                f"{{id: {row}, disabled: true}}",
            ],
        )
        assert started.exit_code == 0, started.output
    write_profile("work", EDITED)

    listed = runner.invoke(app, ["profiles", "diff", "work"])
    adopted = runner.invoke(app, ["profiles", "adopt", "work", "--yes"])

    assert "second, first:" in listed.output or "first, second:" in listed.output, listed.output
    assert adopted.exit_code == 0, adopted.output
    for session_id, row in (("first", "tool-fs"), ("second", "tool-attach")):
        assert logged_types(session_id).count(ADOPTED) == 1
        again = runner.invoke(app, ["-p", "again", "--session", session_id])
        assert again.exit_code == 0, again.output
        env = _environment(session_id)
        rows = {as_obj(one).get("id"): as_obj(one) for one in not_none(env.base).rows}
        assert rows["tool-result-offload"]["disabled"] is True, "the adopted version is the base"
        assert [one.row for one in env.overrides] == [row], "its own override, kept"
        assert logged_types(session_id).count(BASE) == 2


async def test_adopt_goes_through_the_daemon_that_holds_a_session(tmp_path: Path) -> None:
    """A session a daemon holds is held by its lease, so the daemon writes the
    version — and the running root keeps the base it has (decision 18)."""
    write_profile("work", WORK)
    async with running(tmp_path, path=resolve_roots().daemon_socket()) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="live", profile="work")
        root = daemon.held("live")
        write_profile("work", EDITED)

        lines, missed = await adopt_version("work", ["live"], named_version("work"))

        assert not missed, lines
        assert "through the daemon" in lines[0]
        assert logged_environment(root.session).adopted is not None
        assert row_disabled(root, "tool-bash"), "not interrupted"
