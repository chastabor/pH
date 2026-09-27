"""Overrides, through one door (session profiles, S4).

A session deviates from its base by `profile/override` records: a slash command's
change, a verb's, and each command-line start option that differs. The gate:
every reconfigure leaves a record, on disk before the row is re-applied; a value
the session already runs with leaves none; and the log alone — the base, then the
overrides in order — rebuilds the allowances in force. A change whose record
cannot be written is not made.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from daemon_helpers import running
from typer.testing import CliRunner

from ph.cordis.loader import Mount
from ph.json import JsonObject, as_obj
from ph.keys import COMMANDS, MOUNT, SANDBOX
from ph.paths import resolve_roots
from ph.session import now_ms
from ph.session_profile import OVERRIDE, Override, overrides, rebuilt, saved_base
from ph.testing import logged_events, not_none
from ph_app.cli import app
from ph_app.daemon.supervisor import Root
from ph_app.profiles import compose_profile

pytestmark = pytest.mark.anyio

runner = CliRunner()


def _logged(session_id: str) -> list[Override]:
    """The overrides a session's log holds on disk."""
    return [
        Override.of(event.data) for event in logged_events(session_id) if event.type == OVERRIDE
    ]


async def _sandbox(root: Root, line: str) -> str:
    shown = await root.ctx.require(COMMANDS).dispatch(line, session=root.session)
    assert isinstance(shown, str)
    return shown


def _hosts(root: Root) -> list[str]:
    allowances = not_none(root.ctx.require(SANDBOX).allowances)
    return list(not_none(allowances.network).hosts)


async def test_a_sandbox_change_is_recorded_before_it_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The order the F findings were about: when the row is re-applied, the record
    of the change is already on disk. Sabotage: reconfigure first in
    `session_profile.override`, and the log read at that moment holds nothing."""
    seen_on_disk: list[int] = []
    reconfigure = Mount.reconfigure

    async def watched(self: Mount, row_id: str, config: object) -> object:
        seen_on_disk.append(len(_logged("allowed")))
        return await reconfigure(self, row_id, config)

    monkeypatch.setattr(Mount, "reconfigure", watched)
    async with running(tmp_path) as daemon:
        root = await daemon.root("allowed")

        shown = await _sandbox(root, "/sandbox allow host example.com")

        assert "example.com is now reachable" in shown and "session's log" in shown
        assert seen_on_disk == [1], "the record was on disk when the row changed"
        (change,) = _logged("allowed")
        assert (change.row, change.source, change.command) == (
            "sandbox-allow",
            "command",
            "/sandbox allow host example.com",
        )
        assert "example.com" in _hosts(root)
        assert not resolve_roots().profile_dropins("headless").exists(), "no drop-in any more"


async def test_a_value_already_in_force_leaves_no_record(tmp_path: Path) -> None:
    async with running(tmp_path) as daemon:
        root = await daemon.root("unchanged")
        await _sandbox(root, "/sandbox allow host example.com")

        again = await _sandbox(root, "/sandbox allow host example.com")

        assert "already on the allowlist" in again
        assert len(_logged("unchanged")) == 1


async def test_a_change_whose_record_cannot_be_written_is_not_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sabotage: apply the change before checking the write, and the host is
    reachable with nothing on disk to say so."""

    async def unwritten(*_args: object) -> bool:
        return False

    async with running(tmp_path) as daemon:
        root = await daemon.root("unwritable")
        monkeypatch.setattr("ph.session_profile.session_written", unwritten)

        shown = await _sandbox(root, "/sandbox allow host example.com")

        assert "was not changed" in shown
        assert "example.com" not in _hosts(root)


async def test_the_log_alone_rebuilds_the_allowances_in_force(tmp_path: Path) -> None:
    """The base, then the overrides in log order, over a host that knows nothing of
    this session: the allowance the session runs with is in it."""
    async with running(tmp_path) as daemon:
        root = await daemon.root("rebuilt")
        await _sandbox(root, "/sandbox allow host example.com")
        await _sandbox(root, "/sandbox allow host example.org")

        again = rebuilt(
            not_none(saved_base(root.session)),
            host=compose_profile("headless"),
            overrides=overrides(root.session),
        )

        rows = {as_obj(row).get("id"): as_obj(row) for row in again.resolved({"environment"})}
        config: JsonObject = as_obj(rows["sandbox-allow"]["config"])
        assert as_obj(config["network"])["hosts"] == _hosts(root)


async def test_a_session_s_next_start_puts_its_overrides_back(tmp_path: Path) -> None:
    """A root released and started again is on the daemon's composition, which
    knows nothing of the session — and holds the allowance anyway, from its log.
    Sabotage: drop `reapply_overrides` from `open_session`, and the host is gone."""
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
        root = await daemon.root("returning")
        await _sandbox(root, "/sandbox allow host example.com")
        await supervisor.passivate(root, now=now_ms())

        back = await supervisor.start("returning")

        assert "example.com" in _hosts(back)
        assert len(_logged("returning")) == 1, "put back, not recorded again"
        assert "sandbox-allow" in back.ctx.require(MOUNT).reconfigured


def test_a_start_option_is_logged_where_it_differs_and_once() -> None:
    """`--patch` is a start option: an override when it differs from what the session
    runs in, nothing when it does not — so the same flag at the next start, and a
    flag that says what the base already says, leave no record."""
    patch = "{id: tool-bash, disabled: true}"
    for _ in range(2):
        result = runner.invoke(app, ["-p", "hi", "--session", "flagged", "--patch", patch])
        assert result.exit_code == 0, result.output
    matching = runner.invoke(
        app, ["-p", "hi", "--session", "plain", "--patch", "{id: tool-bash, disabled: false}"]
    )
    assert matching.exit_code == 0, matching.output

    (change,) = _logged("flagged")
    assert change.source == "cli" and change.row == "tool-bash"
    assert change.command == "--patch {id: tool-bash, disabled: true}"
    assert change.entry == {"id": "tool-bash", "disabled": True}
    assert _logged("plain") == []


def test_a_model_chosen_at_start_is_the_session_s_from_then_on() -> None:
    """`--model` is a start option too, and an override of the session's `models`
    row: a later start with no flag runs on it, because the session's list says so.
    An unlisted route joins that list under its own name."""
    first = runner.invoke(
        app, ["-p", "hi", "--session", "routed", "--provider", "fake", "--model", "fake-9"]
    )
    assert first.exit_code == 0, first.output
    later = runner.invoke(app, ["-p", "hi", "--session", "routed"])
    assert later.exit_code == 0, later.output

    (change,) = _logged("routed")
    assert (change.row, change.source, change.command) == (
        "models",
        "cli",
        "--provider fake --model fake-9",
    )
    config = as_obj(change.entry["config"])
    assert config["default"] == "fake-fake-9"
    events = logged_events("routed")
    contexts = [
        as_obj(event.data).get("model") for event in events if event.type == "request/context"
    ]
    assert contexts and set(contexts) == {"fake-9"}, "both starts ran on the chosen route"


async def test_model_moves_the_session_and_its_next_start(tmp_path: Path) -> None:
    """`/model` over the daemon is a verb's override: recorded, then the agent moved —
    and a root released and started again comes back on it."""
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
        client = await daemon.client()
        root = await daemon.root("moved")

        reply = await client.call(
            "session/model",
            sessionId=root.id,
            choice={"provider": "fake", "model": "fake-7"},
            clientId="c",
            commandId="1",
        )
        await supervisor.passivate(root, now=now_ms())
        back = await supervisor.start("moved")

        assert reply["modelKey"] == "fake-fake-7"
        (change,) = _logged("moved")
        assert (change.source, change.command) == ("verb", "/model fake/fake-7")
        assert (
            back.agent.options.model == "fake-7" and back.agent.options.model_key == "fake-fake-7"
        )


async def test_the_door_records_nothing_for_the_setting_in_force(tmp_path: Path) -> None:
    """The door's own comparison, through a caller that does not make one first:
    `/model main` on a root already on `main` asks for the config the `models` row
    already has. Sabotage: drop the comparison in `override`, and a record appears
    for a change that changed nothing."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        root = await daemon.root("steady")

        await client.call(
            "session/model", sessionId=root.id, choice={"key": "main"}, clientId="c", commandId="1"
        )

        assert _logged("steady") == []


def test_a_named_route_runs_on_a_profile_that_lists_no_models() -> None:
    """With no `models` row there is no list to make the choice an override of, so the
    route is the agent's alone — and it still runs. It did not, for a moment: the
    start path looked the default up again after choosing, and a profile listing
    nothing has none. Sabotage: create the agent from `choose(ctx, ModelChoice())`
    again in `start_on`'s callers."""
    result = runner.invoke(
        app,
        [
            "-p",
            "hi",
            "--patch",
            "{id: models, remove: true}",
            "--provider",
            "fake",
            "--model",
            "fake-1",
        ],
    )

    assert result.exit_code == 0, result.output
