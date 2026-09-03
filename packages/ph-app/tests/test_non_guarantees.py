"""P5-12 — what the daemon does not promise, asserted (N5, I-2).

Gate: *doctor prints the worker model; docs reviewed.*

This module is P4-16's shape one phase on. Every other daemon test pins a
**mechanism**: `test_daemon.py` drives a real socket, `test_agents_cli.py` drives
the real commands. This one pins the **claims** — the sentences a person reads
before deciding whether to run six agents under one daemon — because a claim is
the one part of a design that rots without any code changing.

So it holds two kinds of assertion and nothing else. First, that the sentences
*arrive*: at `ph doctor`, which is read before a daemon exists, and at
`ph agents doctor`, which is read about one that does. Rule 6 says a caveat only
in the docs is a defect, so a non-guarantee that stopped being printed would be
one nobody is told about, and that is what these hold.

Second, that the two sentences which are **not** obvious from the row they
qualify are still true. Both were found while writing them down, which is the
argument for writing them down: `supervisor/*` crashes are contained per root
(P5-04) rather than not at all.

**One row has been earned back, and a narrower one replaces it.** "The log keeps
the appointment; nothing is watching it" was true and is not: P6-23 gave the
schedule seam an index of what is due, so a daemon wakes exactly the roots that
have an appointment and leaves every other session unleased. Its assertion now
lives in `test_daemon.py` asserting the opposite, which is the correct way for a
non-guarantee to end.

What is left is the honest boundary, and it is a row of its own: **nothing fires
while the daemon is down.** pH keeps an appointment while it runs and catches up
when it starts; a run that has to happen whether or not the daemon is up belongs
to cron, anacron or a systemd timer, which already do this and which pH is not
trying to replace. Stating that is the point — a person who reads "schedule" and
assumes a crontab is the reader this table exists for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from daemon_helpers import running
from typer.testing import CliRunner

from ph_app.cli import app
from ph_app.daemon import recovery
from ph_app.daemon.supervisor import NON_GUARANTEES

pytestmark = pytest.mark.anyio

runner = CliRunner()

CLAIMED = {
    "worker model": "one anyio task per root",
    "per-root memory": "not capped",
    "crash containment": "per root, not per process",
    "CPU": "shared",
    "restart": "not rolling",
    "while the daemon is down": "nothing fires",
    "what `!!` puts in the log": "whatever the command printed",
    "who may run `!!`": "anyone holding the web token",
    "a question a person walked away from": "re-posed only while this daemon runs",
    "per user": "one daemon per $PH_RUNTIME",
}
"""Every N5 row, and the phrase that carries it.

Held here as well as in `NON_GUARANTEES` on purpose, which is the one place in
this suite where restating a constant is the point: a test that read the rows
and asserted they were the rows would pass against an empty tuple, and against a
row someone softened to "memory is managed". The §3 table promises a reviewer
that a non-goal implied as covered is a defect; this is that promise, in a form
that fails.
"""


def test_every_non_guarantee_is_stated_and_none_has_gone_soft() -> None:
    rows = dict(NON_GUARANTEES)
    assert set(rows) == set(CLAIMED), "a non-guarantee was added or dropped without a claim"
    for label, phrase in CLAIMED.items():
        assert phrase in rows[label], f"{label!r} no longer says {phrase!r}"


# ------------------------------------------------------- that they are printed --


def test_ph_doctor_prints_the_worker_model_without_a_daemon(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The gate's first half, and the reader who has not started one yet.

    Before the mount, with the other profile-free section, so a profile that
    refuses to start does not take the answer with it — the same argument P5-11
    made for the socket lifetime, and the reason both live in one list.
    """
    monkeypatch.setenv("PH_RUNTIME", str(tmp_path / "run"))
    result = runner.invoke(
        app, ["doctor"], env={"COLUMNS": "300", "FORCE_COLOR": None, "NO_COLOR": "1"}
    )
    assert result.exit_code == 0, result.output
    assert "daemon isolation" in result.stdout
    assert "one anyio task per root" in result.stdout, "the worker model"
    assert "not capped" in result.stdout
    assert "Isolation *between users* is the operator's layer" in result.stdout


async def test_the_running_daemon_reports_them_beside_the_root_count(tmp_path: Path) -> None:
    """Where the assumption is actually made.

    `daemon/status` says "roots: 3" a few rows above this section, and that line
    is the whole invitation: three roots reads as three things that cannot hurt
    each other. The section is in the same reply because rule 6 puts the caveat
    where the assumption is, not in a document beside it.
    """
    async with running(tmp_path) as daemon:
        await daemon.server.supervisor.start("one")
        await daemon.server.supervisor.start("two")
        status = daemon.server.status()

        assert status["roots"] == 2, "the line that invites the assumption"
        titles = [section["title"] for section in status["sections"]]
        assert "isolation" in titles, f"the daemon reports {titles}"
        section = next(one for one in status["sections"] if one["title"] == "isolation")
        rows = {row["label"]: row["value"] for row in section["rows"]}
        assert rows == dict(NON_GUARANTEES), "the wire carries them verbatim, not a summary"


# ------------------------------------------------- that the two hard ones hold --


async def test_a_roots_own_crash_is_contained_and_the_process_is_the_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Per root, not per process" — the narrower claim, and the true one.

    §3's N5 was written before P5-04 and says pH does not contain crashes
    between roots. It does: a root whose task raises climbs the ladder, gives up
    in its own log, and leaves every other root running. Printing the broad
    version would send a reader provisioning against the wrong failure — so the
    row says where the boundary actually is, and this is that sentence.
    """
    # The ladder's real delays are 0.25 s, 1 s and 5 s, and this test is about
    # where the boundary is rather than about how long it takes to reach it —
    # `test_daemon.py`'s own ladder tests shorten it the same way.
    monkeypatch.setattr(recovery, "RETRY_DELAYS", (0.01, 0.01, 0.01))
    async with running(tmp_path) as daemon:
        supervisor = daemon.server.supervisor
        broken = await supervisor.start("broken")
        healthy = await supervisor.start("healthy")

        async def explode() -> None:
            raise RuntimeError("this root's task is broken")

        broken.agent.run = explode
        await supervisor.prompt("broken", "go")
        with anyio.fail_after(10):
            while not broken.recovery.failed:
                await anyio.sleep(0.01)

        # The ladder is spent, the give-up is in *its* log, and the daemon is
        # still answering — which is the containment the row claims.
        assert broken.status == "failed"
        assert healthy.status == "idle", "one root's crash is not another's"
        assert daemon.server.status()["roots"] == 2

        await supervisor.prompt("healthy", "and you?")
        with anyio.fail_after(10):
            while not any(
                event.type == "assistant/message" for event in healthy.session.events_from(0)
            ):
                await anyio.sleep(0.01)
