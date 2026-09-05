"""P7-01 — `ph attachments gc`, driven the way a person runs it.

The fold's rules are pinned in ph-core against stub stores. What a command adds is
the part that cannot be stubbed: reading logs it did not write, out of a process
that was not running when they were, against a real JSONL store and the real
`$PH_HOME/attachments` directory the mounted profile resolves.

Two claims here, and the second is the one that costs something to get wrong.
Reporting is the default, so a person who runs this because the disk is full sees
the size of the answer before anything acts on it. And a blob referenced by a
session on disk survives `--remove` — which after P7-03 is not a nicety, because
the local copy is the last one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from ph.persistence import session_path
from ph.seams.attachments import digest_of
from ph.session import Session, SessionHeader, SurfaceIntent
from ph_app.cli import app

runner = CliRunner()

PNG = b"\x89PNG\r\n\x1a\n" + b"pixels" * 32
OTHER = PNG + b"other"
OLD = (1_000_000_000.0, 1_000_000_000.0)
"""An mtime well outside `--min-age`, so a test is not waiting out a day."""


def _blob(home: Path, content: bytes) -> Path:
    """One stored attachment, written where the mounted profile will look."""
    root = home / "attachments"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{digest_of(content).partition(':')[2]}.png"
    path.write_bytes(content)
    import os

    os.utime(path, OLD)
    return path


def _blob_now(home: Path, content: bytes) -> Path:
    """A blob written *now*, so `--min-age` holds it back rather than age."""
    root = home / "attachments"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{digest_of(content).partition(':')[2]}.png"
    path.write_bytes(content)
    return path


def _log(sessions: Path, session_id: str, *events: tuple[str, Any]) -> Path:
    """A stored session, written before this process starts.

    Through the real envelope rather than a hand-rolled dict, for
    `test_workspaces_cli`'s reason: the format is ours, and a fixture that spelled
    it out separately pins the test to a shape the backend has moved on from.
    """
    header = SessionHeader(id=session_id, created_at=1, family=session_id)
    path = session_path(sessions, session_id, header.family)
    path.parent.mkdir(parents=True, exist_ok=True)
    session = Session(session_id, header=header)
    for kind, data in events:
        session.append(kind, data, SurfaceIntent("append") if kind == "user/message" else None)
    lines = [{"type": "session/header", "header": session.header.to_wire()}]
    lines += [event.to_wire() for event in session.events]
    path.write_text("".join(f"{json.dumps(line)}\n" for line in lines), encoding="utf-8")
    return path


def _media(content: bytes) -> dict[str, Any]:
    """A `user/message` payload carrying one attachment, as the loop writes it."""
    return {
        "id": "m1",
        "role": "user",
        "content": [
            {
                "type": "media",
                "attachment": {
                    "attachmentId": digest_of(content),
                    "mime": "image/png",
                    "bytes": len(content),
                },
            }
        ],
        "source": {"kind": "user"},
    }


def test_gc_reports_before_it_removes(roots: Path) -> None:
    """The default is the report, and it names the size rather than the files.

    What decides `--remove` is how much is on the other side and whether the
    survey was complete — not a list of digests, which is why the listing is
    capped and the total is not.
    """
    home = roots.parent
    blob = _blob(home, PNG)
    _log(roots, "one", ("turn/start", {"turn": 1}))

    result = runner.invoke(app, ["attachments", "gc", "--profile", "headless"])

    assert result.exit_code == 0, result.output
    assert "1 collectable" in result.stdout
    assert "--remove" in result.stdout
    assert blob.exists(), "the default run deleted content nobody asked it to"


def test_a_referenced_blob_survives_remove(roots: Path) -> None:
    """The rule, through the command: a blob a stored log references is kept.

    The two blobs are the same age and neither session is running; the only thing
    telling them apart is that one is mentioned in a log on disk. After P7-03 the
    local copy is the last one, so collecting it would not degrade that session —
    it would end it.
    """
    home = roots.parent
    kept, dead = _blob(home, PNG), _blob(home, OTHER)
    _log(roots, "one", ("user/message", _media(PNG)))

    result = runner.invoke(app, ["attachments", "gc", "--profile", "headless", "--remove"])

    assert result.exit_code == 0, result.output
    assert "collected 1 blob(s)" in result.stdout
    assert kept.exists(), "a blob a stored session still points at was collected"
    assert not dead.exists()


def test_a_torn_log_refuses_the_whole_collection(roots: Path) -> None:
    """Fail closed, and say which half of the survey was missing.

    A person who reads only "nothing collected" would reasonably run it again with
    the same flag; the reason has to be in the output or the command is a
    dead end.
    """
    home = roots.parent
    blob = _blob(home, PNG)
    (roots / "torn").mkdir(parents=True, exist_ok=True)
    (roots / "torn" / "torn.jsonl").write_text("{not json at all\n", encoding="utf-8")

    result = runner.invoke(app, ["attachments", "gc", "--profile", "headless", "--remove"])

    assert result.exit_code == 0, result.output
    assert "refusing to collect anything" in result.stdout
    assert "would not read" in result.stdout
    assert blob.exists()


def test_a_torn_log_is_reported_even_when_nothing_was_collectable(roots: Path) -> None:
    """The refusal is not conditional on there being something to collect.

    An early "nothing to collect" return read the counts and not the survey's
    completeness, so a store whose log would not read answered with the happier of
    the two sentences and never mentioned the log — while a *fresh* blob is
    exactly the case that produces those counts, since it is held back by
    `--min-age` rather than by anything being wrong.
    """
    _blob_now(roots.parent, PNG)
    (roots / "torn").mkdir(parents=True, exist_ok=True)
    (roots / "torn" / "torn.jsonl").write_text("{not json at all\n", encoding="utf-8")

    result = runner.invoke(app, ["attachments", "gc", "--profile", "headless"])

    assert result.exit_code == 0, result.output
    assert "0 collectable" in result.stdout, "there was indeed nothing to take"
    assert "refusing to collect anything" in result.stdout
    assert "would not read" in result.stdout


def test_a_profile_with_no_store_says_so(roots: Path) -> None:
    """Two rows, one sentence — the shape `ph workspaces gc` already uses.

    Without a store there is nothing to sweep, and a command that answered
    "nothing to collect" would be indistinguishable from a swept store.
    """
    result = runner.invoke(
        app,
        [
            "attachments",
            "gc",
            "--profile",
            "headless",
            "--patch",
            "{id: attachments, disabled: true}",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "mounts no attachment store" in result.stdout
