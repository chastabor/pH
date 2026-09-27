"""Reading the audit (session profiles, S8).

The environment a session ran in at any seq is a fold of its log — the base in
force there and the overrides up to it (item 11) — and it reads the same from the
command line and in the trajectory view. Beside it, which text of each skill was
read, by hash (decision 12).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from daemon_helpers import run_command, running
from typer.testing import CliRunner

from ph.json import as_obj, as_seq
from ph.keys import SESSIONS
from ph.seams.skills import Skill, record_read
from ph.session_profile import OVERRIDE
from ph.testing import logged_events
from ph_app.cli import app
from ph_app.profiles import compose_profile
from ph_app.runtime import prompted
from ph_app.tui.trajectory import build_trajectory

pytestmark = pytest.mark.anyio

runner = CliRunner()


def _started_twice() -> tuple[int, int]:
    """One session, two starts, each with a start option of its own: the seq of each
    override, in order."""
    for patch in ("{id: tool-bash, disabled: true}", "{id: tool-fs, disabled: true}"):
        started = runner.invoke(app, ["-p", "hi", "--session", "audited", "--patch", patch])
        assert started.exit_code == 0, started.output
    first, second = [event.seq for event in logged_events("audited") if event.type == OVERRIDE]
    return first, second


def test_the_environment_at_a_seq_is_its_base_and_the_overrides_up_to_it() -> None:
    """Sabotage: fold the whole log whatever `--at` says, and the first start's
    environment shows the second start's option too."""
    first, _second = _started_twice()

    then = runner.invoke(app, ["profiles", "session", "audited", "--at", str(first)])
    now = runner.invoke(app, ["profiles", "session", "audited"])

    assert then.exit_code == 0, then.output
    assert f"audited at seq {first} of" in then.output
    assert "tool-bash" in then.output and "tool-fs" not in then.output
    assert "tool-bash" in now.output and "tool-fs" in now.output


def test_full_prints_every_row_as_it_ran_at_that_seq() -> None:
    first, _second = _started_twice()

    shown = runner.invoke(app, ["profiles", "session", "audited", "--at", str(first), "--full"])

    assert shown.exit_code == 0, shown.output
    rows = {as_obj(row).get("id"): as_obj(row) for row in as_seq(yaml.safe_load(shown.output))}
    assert rows["tool-bash"]["disabled"] is True
    assert rows["tool-fs"]["disabled"] is False, "the second start's option came later"


def test_a_seq_before_the_base_says_there_was_none_yet() -> None:
    _started_twice()

    shown = runner.invoke(app, ["profiles", "session", "audited", "--at", "-1"])

    assert shown.exit_code == 2
    assert "no recorded base by seq -1" in shown.output


async def test_the_skills_read_by_then_are_listed_with_what_was_read() -> None:
    """Decision 12: the base records where skills are found, and each read records
    which text — so an audit can say two runs followed the same instructions."""
    async with prompted(compose_profile("headless"), "hi", session_id="skilled") as (ctx, session):
        record_read(
            session, Skill(name="review", description="reviews", version="2.0"), "Look.", via="tool"
        )
        await ctx.require(SESSIONS).flush(session)

    shown = runner.invoke(app, ["profiles", "session", "skilled"])

    assert "Skills read by then" in shown.output
    assert "review 2.0 sha256:" in shown.output and "(tool) at seq" in shown.output


async def test_the_trajectory_shows_the_environment_at_each_change(tmp_path: Path) -> None:
    """The auditor's view reads the same fold: each `profile/*` record says what
    changed, and its detail is the environment from there on. Sabotage: render the
    override as a bare harness event again, and its detail is only the payload."""
    async with running(tmp_path) as daemon:
        root = await daemon.root("traced")
        await run_command(root, "/sandbox allow host example.com")

        records = build_trajectory(root.session)

    by_type = {record.type: record for record in records}
    base, change = by_type["profile/base"], by_type[OVERRIDE]
    assert base.summary.startswith("base: ") and "No overrides" in base.detail
    assert change.summary == "sandbox-allow ← /sandbox allow host example.com (command)"
    assert "Overrides, in the order they apply:" in change.detail
    assert "+ example.com" in change.detail
