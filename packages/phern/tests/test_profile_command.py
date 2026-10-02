"""`/profile show | diff | save | use | clear` (session profiles, S7).

The gate: tune a session, save it, start a headless run on the saved profile, and
the two environments match. Around it: `show` says what a session runs with and
why; `diff` lists how its named profile moved; `use` switches the base, keeping the
overrides or clearing them, and starts the root again on it; `clear` stops an
override live; and a profile saved and then used as the base clears the overrides
it now says itself (decision 3's own rule).
"""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest
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

from ph.cordis import LoaderError, Profile
from ph.keys import SESSIONS
from ph.paths import resolve_roots
from ph.session import now_ms
from ph.session_profile import (
    BASE,
    CLEARED,
    SAVED,
    WITHDRAWN,
    environment_listing,
    logged_environment,
    record_summary,
    resolved_environment,
    saved_base,
)
from ph.testing import not_none, unwritten, write_profile
from ph_app.cli import app
from ph_app.daemon.client import DaemonClient
from ph_app.daemon.supervisor import Root
from ph_app.modes import run_rpc
from ph_app.payloads import ProfileAskReply
from ph_app.profiles import compose_profile, kept_note, named_version
from ph_app.profiles_cli import adopt_version
from ph_app.protocol import DaemonError
from ph_app.runtime import mounted, prompted
from ph_app.sessions import recorded_environment

pytestmark = pytest.mark.anyio

WORK = "extends: headless\nrows:\n  - id: tool-bash\n    disabled: true\n"

runner = CliRunner()


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


# ------------------------------------------------ a version that will not mount --

BROKEN = "extends: headless\nrows:\n  - id: tool-bash\n    disabled: true\n"


def _refusing(
    monkeypatch: pytest.MonkeyPatch,
    refused: Callable[[Profile], bool],
    error: Callable[[Profile], Exception] = lambda profile: LoaderError(
        f"a row of {profile.name} would not apply"
    ),
) -> None:
    """Mounts fail for a profile `refused` picks — by default as one whose row the
    loader refuses, else with the `error` given."""

    @asynccontextmanager
    async def picky(profile: Profile, *, project: Path | None = None) -> AsyncIterator[Any]:
        if refused(profile):
            raise error(profile)
        async with mounted(profile, project=project) as ctx:
            yield ctx

    monkeypatch.setattr("ph_app.runtime.mounted", picky)


def _moved_work(profile: Profile) -> bool:
    """`work` as it composes after the edit — `tool-bash` back on — and not a mount of
    its host rows alone, which has no `tool-bash` row."""
    bash = next((row for row in profile.rows if row.id == "tool-bash"), None)
    return profile.name == "work" and bash is not None and not bash.disabled


async def test_use_of_a_profile_that_will_not_start_leaves_the_session_where_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(2) The switch is an adoption the restart applies, so a version that will not
    mount is withdrawn there — `profile/withdrawn`, with why — and the session starts
    on what it had, overrides and all, rather than failing at every start after. The
    person who asked is told, in the command's own reply, and `/profile show` says so
    after. Sabotage: drop the withdrawal in `runtime.mount_session`, and the root fails
    and stays failed."""
    write_profile("broken", BROKEN)
    _refusing(monkeypatch, lambda profile: profile.name == "broken")
    async with running(tmp_path) as daemon:
        client, root = await _attached(daemon, "steady")
        await _sandbox(root, "/sandbox allow host example.com")

        shown = await _command(client, "steady", "/profile use broken")

        back = daemon.held("steady")
        assert back is not root and back.describe().profile == "headless", shown
        assert "broken did not start (a row of broken would not apply)" in shown
        assert "example.com" in allowed_hosts(back), "its override is still its own"
        env = logged_environment(back.session)
        assert env.adopted is None and not_none(env.base).name == "headless"
        assert not_none(env.declined).reason == "a row of broken would not apply"
        (taken,) = back.session.select(WITHDRAWN)
        assert record_summary(WITHDRAWN, taken.data) == (
            "took back broken's version: it did not start (a row of broken would not apply)"
        ), "as the trajectory reads it"
        assert "Taken back at its last start: broken did not start" in "\n".join(
            environment_listing(env)
        )
        await daemon.running.supervisor.passivate(back, now=now_ms())
        again = await daemon.running.supervisor.start("steady")
        assert again.describe().profile == "headless", "and every start after it"

        # A restart that adopts nothing — a `/profile clear` of an override that
        # turned a row on or off; here the request set by hand — does not say again
        # what an earlier start took back: `Root.withdrew` is that start's alone.
        # Sabotage: read the withdrawal off the log in `_after_command`, and this
        # reply repeats it.
        again.restart_wanted = True
        shown = await _command(client, "steady", "/profile show")
        assert shown.count("runs on the version it had") == 1, "the listing's own line only"


async def test_a_yes_to_a_version_that_will_not_start_keeps_the_session_s_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(2) The same withdrawal behind S6's "yes": the moved profile no longer mounts,
    so the session comes back on the version it had instead of not at all — and the
    version reads as declined, so the next start does not ask about it again, and a
    terminal attaching is told why. Sabotage: leave `declined` alone when the fold
    reads `profile/withdrawn`, and the next start asks the same question again."""
    write_profile("work", WORK)

    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="asked", profile="work")
        await daemon.running.supervisor.passivate(daemon.held("asked"), now=now_ms())
        write_profile("work", "extends: headless\nrows: []\n")
        _refusing(monkeypatch, _moved_work)
        person = await daemon.client("asks")

        asked: list[dict[str, Any]] = []

        async def adopt(params: dict[str, Any]) -> ProfileAskReply:
            asked.append(params)
            return ProfileAskReply(decision="adopt")

        person.handlers["profile/ask"] = adopt
        await person.call("session/attach", sessionId="asked")
        held = daemon.held("asked")
        await until(
            lambda: daemon.running.supervisor.roots.get("asked") not in (None, held),
            what="the start after the yes",
        )

        back = daemon.held("asked")
        assert row_disabled(back, "tool-bash"), "the version it had"
        assert logged_environment(back.session).adopted is None
        # One sentence, from the daemon, for the terminal that attaches; the command
        # line's names its own commands.
        note = back.describe().profile_note
        assert note.startswith("work did not start (a row of work would not apply)")
        assert note.endswith("/profile diff lists what it changes.")
        cli = kept_note(not_none(back.change), "asked")
        assert cli.endswith("`phern profiles diff work --session asked` lists what it changes.")

        await daemon.running.supervisor.passivate(back, now=now_ms())
        await person.call("session/attach", sessionId="asked")
        again = daemon.held("asked")
        assert not again.held_on_profile and len(asked) == 1, "not asked the same again"


def test_a_one_shot_start_takes_back_a_version_that_will_not_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same rule for `phern -p --session`, which is `mount_session`'s for every host:
    a version adopted ahead of time that the loader refuses is taken back, the prompt
    runs on the version the session had, and stderr says why — here another profile's
    version, which no note about the session's own profile would mention. Sabotage:
    mount `session_profile` directly in `prompted` again, and the run fails on the
    refused version; print only the kept note, and stderr says nothing."""
    write_profile("work", WORK)
    write_profile("broken", BROKEN)
    started = runner.invoke(app, ["-p", "hi", "--profile", "work", "--session", "solo"])
    assert started.exit_code == 0, started.output
    lines, missed = anyio.run(adopt_version, "broken", ["solo"], named_version("broken"))
    assert not missed, lines
    _refusing(monkeypatch, lambda profile: profile.name == "broken")

    again = runner.invoke(app, ["-p", "again", "--session", "solo"])

    assert again.exit_code == 0, again.output
    assert "broken did not start (a row of broken would not apply)" in again.stderr
    env = recorded_environment(resolve_roots().sessions_dir(), "solo")
    assert (
        env.adopted is None and not_none(env.declined).reason == "a row of broken would not apply"
    )
    assert not_none(env.base).name == "work"
    types = logged_types("solo")
    assert types.index(WITHDRAWN) < len(types) - 1, "and the prompt ran after it"


def test_a_take_back_the_log_cannot_hold_refuses_the_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`profile/withdrawn` is what stops the refused version being applied at every
    start. Written and not checked, a withdrawal the log could not hold steered
    this start from memory while the disk still said the version was pending, so
    the next start mounted it, failed, and withdrew it again. Now it goes through
    the door every other change here takes, and the start says what was not done.

    Sabotage: put back the unchecked `session_written`, and the command exits 0 with
    the version still pending on disk.
    """
    write_profile("work", WORK)
    write_profile("broken", BROKEN)
    started = runner.invoke(app, ["-p", "hi", "--profile", "work", "--session", "solo"])
    assert started.exit_code == 0, started.output
    lines, missed = anyio.run(adopt_version, "broken", ["solo"], named_version("broken"))
    assert not missed, lines
    _refusing(monkeypatch, lambda profile: profile.name == "broken")
    monkeypatch.setattr("ph.session_profile.session_written", unwritten)

    again = runner.invoke(app, ["-p", "again", "--session", "solo"])

    assert again.exit_code != 0
    assert "broken was not withdrawn" in again.stderr
    env = recorded_environment(resolve_roots().sessions_dir(), "solo")
    assert not_none(env.adopted).name == "broken", "nothing on disk says otherwise"


async def test_rpc_takes_back_a_version_that_will_not_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """And for rpc, which mounts each session it serves from that session's log.
    Sabotage: mount `session_profile` directly in `RpcServer._mount` again, and the
    prompt is refused."""
    write_profile("work", WORK)
    async with prompted(compose_profile("work"), "hi", session_id="served"):
        pass
    write_profile("work", "extends: headless\nrows: []\n")
    lines, missed = await adopt_version("work", ["served"], named_version("work"))
    assert not missed, lines
    _refusing(monkeypatch, _moved_work)
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/prompt",
            "params": {"sessionId": "served", "prompt": "again"},
        },
        {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
    ]
    out = io.StringIO()

    await run_rpc(
        compose_profile("headless"),
        stdin=io.StringIO("".join(f"{json.dumps(one)}\n" for one in requests)),
        out=out,
    )

    replies = {
        frame["id"]: frame
        for frame in map(json.loads, out.getvalue().splitlines())
        if "id" in frame
    }
    assert "error" not in replies[1], replies[1]
    env = recorded_environment(resolve_roots().sessions_dir(), "served")
    assert env.adopted is None and not_none(env.declined).reason == "a row of work would not apply"


async def test_a_start_that_fails_for_another_reason_takes_nothing_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the profile's own refusals withdraw an adoption. A disk that filled, or a
    bug in a row, fails the start as any start fails, and the version stays adopted
    for a start that can mount it. Sabotage: catch every exception in
    `mount_session` again, and the version is taken back over a full disk."""
    write_profile("broken", BROKEN)
    _refusing(
        monkeypatch,
        lambda profile: profile.name == "broken",
        lambda _profile: OSError("no space left on device"),
    )
    async with running(tmp_path) as daemon:
        client, _root = await _attached(daemon, "steady")

        await _command(client, "steady", "/profile use broken")

        env = recorded_environment(resolve_roots().sessions_dir(), "steady")
        assert not_none(env.adopted).name == "broken" and env.declined is None
        assert "steady" not in daemon.running.supervisor.roots


async def test_a_command_that_fails_after_asking_for_a_restart_still_gets_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c) The request is consumed by `session/command`'s `after` step on the error
    path too, so it is not left on the root for the next command to trip over.
    Sabotage: run `after` on success only, and the root runs on as it was."""
    write_profile("work", WORK)

    def failing(_root: Root) -> None:
        raise RuntimeError("the readings would not refresh")

    monkeypatch.setattr("ph_app.daemon.server._refresh_readings", failing)
    async with running(tmp_path) as daemon:
        client, root = await _attached(daemon, "tripped")

        with pytest.raises(DaemonError):
            await _command(client, "tripped", "/profile use work")

        back = daemon.held("tripped")
        assert back is not root and row_disabled(back, "tool-bash")
        assert not back.restart_wanted
