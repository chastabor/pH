"""A session's base, recorded (session profiles, S3).

The gate: a new root's log holds its full profile, on disk before its first step;
mounting from that record alone gives the same environment; and a plugin default
that moved under a person's file nobody edited is found by comparing the saved
base with the named profile as it composes now, and is owed to pH — where an
edit to that file is owed to the person.

Audit only: nothing here changes what a session runs with.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from daemon_helpers import running
from typer.testing import CliRunner

from ph.json import JsonObject, JsonValue, as_obj, as_str
from ph.keys import MOUNT
from ph.llm import retry
from ph.session_profile import (
    BASE,
    Difference,
    ProfileBase,
    base_of,
    differences,
    rebuilt,
    saved_base,
)
from ph.testing import logged_events, write_profile
from ph_app.cli import app
from ph_app.profiles import compose_profile, profile_or_exit
from ph_app.runtime import mounted

pytestmark = pytest.mark.anyio

runner = CliRunner()

WORK = "extends: headless\nrows:\n  - id: tool-bash\n    disabled: true\n"


def _work(text: str = WORK) -> None:
    write_profile("work", text)


def _on_disk(session_id: str) -> ProfileBase:
    """The base as the log file on disk holds it — not the in-memory session."""
    (only,) = [event for event in logged_events(session_id) if event.type == BASE]
    return ProfileBase.of(only.data)


def _by_id(rows: Iterable[JsonValue]) -> dict[str, JsonObject]:
    return {as_str(as_obj(row).get("id")): as_obj(row) for row in rows}


def _config(rows: Iterable[JsonValue], row_id: str) -> JsonObject:
    return as_obj(_by_id(rows)[row_id].get("config"))


async def test_a_new_root_s_log_holds_its_full_profile_before_its_first_step(
    tmp_path: Path,
) -> None:
    """Every environment row through its model — defaults it never wrote included —
    on disk before the agent has taken a step, and only one of it."""
    async with running(tmp_path) as daemon:
        root = await daemon.root("based")

        base = _on_disk(root.id)

        assert saved_base(root.session) == base, "what S6 compares against, read back"
        assert not [event for event in root.session.events if event.type == "turn/start"]
        mounted_on = root.ctx.require(MOUNT).profile
        assert base.rows == tuple(mounted_on.resolved({"environment"}))
        assert base.name == mounted_on.name and base.ph_version
        main = as_obj(as_obj(_config(base.rows, "models").get("models")).get("main"))
        assert main["reasoningEffort"] is None, "a default nobody wrote"
        assert "session-persistence" not in _by_id(base.rows), "the host's rows are not the base"


async def test_the_base_alone_rebuilds_the_environment_it_recorded(tmp_path: Path) -> None:
    """The record is enough: mounted over a host whose own profile says nothing of
    the person's file, it runs with the person's settings — because they are in it."""
    _work()
    base = base_of(compose_profile("work"))

    again = rebuilt(base, host=compose_profile("headless"))

    assert _by_id(again.resolved({"environment"})) == _by_id(base.rows)
    assert _by_id(again.resolved({"environment"}))["tool-bash"]["disabled"] is True
    async with mounted(again) as ctx:
        assert ctx.get("sessions") is not None, "and it mounts"


def test_a_moved_default_under_an_unedited_file_is_owed_to_ph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decision 14's case: the person's file is what it was, and pH moved a default
    beneath it. Found by comparing, setting by setting, and not blamed on them."""
    _work()
    saved = base_of(compose_profile("work"))

    class Moved(retry.Config):
        max_attempts: int = 5

    spec = retry.apply.__ph_plugin__  # type: ignore[attr-defined]
    monkeypatch.setattr(retry.apply, "__ph_plugin__", replace(spec, config_model=Moved))
    now = base_of(compose_profile("work"))

    assert differences(saved, now) == [
        Difference(row="llm-retry", setting="config.maxAttempts", before=3, after=5, by="pH")
    ]


def test_an_edit_to_the_person_s_file_is_owed_to_them() -> None:
    _work()
    saved = base_of(compose_profile("work"))

    _work("extends: headless\nrows:\n  - id: tool-bash\n    disabled: false\n")
    now = base_of(compose_profile("work"))

    assert differences(saved, now) == [
        Difference(row="tool-bash", setting="disabled", before=True, after=False, by="person")
    ]


def test_a_start_option_is_not_the_base() -> None:
    """`--patch` is what a session deviates from its base by — S4 logs it as an
    override — so the base is the named profile as it composes without it."""
    composed = profile_or_exit("headless", ["{id: tool-bash, disabled: true}"])

    base = base_of(composed)

    assert {row.id: row for row in composed.rows}["tool-bash"].disabled
    assert _by_id(base.rows)["tool-bash"]["disabled"] is False
    assert all(source.layer != "cli" for source in base.sources)


def test_a_session_from_before_the_record_gets_one_on_its_first_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Logs written before S3 have no base; the first start after it records one,
    and every start after that leaves it alone."""

    async def unrecorded(*_args: object) -> None:
        return None

    monkeypatch.setattr("ph.persistence.opening.opened", unrecorded)
    assert runner.invoke(app, ["-p", "one", "--session", "older"]).exit_code == 0
    monkeypatch.undo()
    assert not [event for event in logged_events("older") if event.type == BASE]

    for prompt in ("two", "three"):
        assert runner.invoke(app, ["-p", prompt, "--session", "older"]).exit_code == 0

    assert len([event for event in logged_events("older") if event.type == BASE]) == 1


def test_a_session_s_base_can_be_read_back_from_its_log() -> None:
    """`phern profiles session <id>`: the audit's reader, from the file on disk."""
    _work()
    assert (
        runner.invoke(app, ["-p", "hi", "--profile", "work", "--session", "audited"]).exit_code == 0
    )

    shown = runner.invoke(app, ["profiles", "session", "audited"])

    assert shown.exit_code == 0, shown.output
    read = yaml.safe_load(shown.stdout)
    assert read["name"] == "work"
    assert _by_id(read["rows"])["tool-bash"]["disabled"] is True
    assert read["sources"][0]["layer"].endswith("work.yaml")
    missing = runner.invoke(app, ["profiles", "session", "nobody"])
    assert missing.exit_code == 2 and "no session" in missing.output
