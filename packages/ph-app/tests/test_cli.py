"""P0-17 — the command line.

Gate: *a one-shot Q&A against the fake adapter writes an inspectable JSONL.*

"Inspectable" means readable by dsh tooling, not just by pH: the log is the
trace (§8), so the first thing the CLI has to earn is a file someone else can
read without a converter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from ph_app.cli import app
from ph_app.profiles import resolve_profile

runner = CliRunner()


def test_dump_config_shows_the_composed_rows() -> None:
    result = runner.invoke(app, ["--dump-config", "--profile", "headless"])
    assert result.exit_code == 0, result.output
    rows = yaml.safe_load(result.stdout)
    ids = [row["id"] for row in rows]
    assert ids[0] == "llm"
    assert "llm-fake" in ids
    # Every row names the layer it came from, so a surprising value is
    # traceable to the file that set it.
    assert all(row["layer"].endswith(".yaml") for row in rows)
    fake = next(row for row in rows if row["id"] == "llm-fake")
    assert fake["config"]["providers"] == ["fake"]
    assert fake["layer"].endswith("headless.yaml")


def test_unknown_profile_is_refused() -> None:
    result = runner.invoke(app, ["--dump-config", "--profile", "nonesuch"])
    assert result.exit_code == 2
    assert "unknown profile" in result.output


def test_doctor_prints_three_roots(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("PH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("PH_RUNTIME", str(tmp_path / "run"))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    for name in ("PH_HOME", "PH_CACHE", "PH_RUNTIME"):
        assert name in result.stdout


def test_events_matrix_is_generated_from_the_registry() -> None:
    result = runner.invoke(app, ["events", "--json"])
    assert result.exit_code == 0, result.output
    matrix = json.loads(result.stdout)
    by_name = {row["name"]: row for row in matrix}
    assert by_name["agent/pre-step"]["mode"] == "waterfall"
    assert by_name["agent/pre-step"]["payload"] == "PreStepRequest"
    assert by_name["session/flush"]["mode"] == "parallel"
    assert by_name["session/event"]["mode"] == "emit"


def test_print_mode_answers_and_writes_a_readable_log(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    result = runner.invoke(app, ["-p", "what is a session log?", "--session", "demo"])
    assert result.exit_code == 0, result.output
    assert "ok" in result.stdout

    path = tmp_path / "sessions" / "demo.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["type"] == "session/header"
    assert records[0]["header"]["version"] == 0

    events = records[1:]
    # dsh's envelope, byte-for-byte: `{type, seq, time, data}` plus the optional
    # camelCase surface fields (D2, Q2).
    assert [event["seq"] for event in events] == list(range(len(events)))
    for event in events:
        assert set(event) <= {
            "type",
            "seq",
            "time",
            "data",
            "ignorable",
            "sourceEventSeqs",
            "surfaceOp",
        }
    user = next(e for e in events if e["type"] == "user/message")
    assert user["surfaceOp"] == "append"
    assert user["data"]["content"][0]["text"] == "what is a session log?"

    # And the harness can read its own log back into the same conversation.
    from ph.persistence.jsonl import read_session
    from ph.session import Session

    header, restored = read_session(path)
    assert header.id == "demo"
    session = Session("demo", seed=restored, header=header)
    assert [m.role for m in session.derive_messages()] == ["user", "assistant"]


def test_each_mode_is_reachable_from_the_command_line(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("PH_HOME", str(tmp_path))

    text = runner.invoke(app, ["-p", "hello"])
    assert text.exit_code == 0, text.output
    assert "ok" in text.stdout

    transcript = runner.invoke(app, ["-p", "hello", "--mode", "transcript"])
    assert transcript.exit_code == 0, transcript.output
    assert "you: hello" in transcript.stdout
    assert "pH: ok" in transcript.stdout

    stream = runner.invoke(app, ["-p", "hello", "--mode", "json"])
    assert stream.exit_code == 0, stream.output
    lines = [json.loads(line) for line in stream.stdout.splitlines() if line.startswith("{")]
    assert lines[0]["type"] == "session/header"
    # The log's own envelopes, not a rendering (I-7).
    assert any(event.get("type") == "turn/end" for event in lines)


def test_an_unknown_mode_is_refused() -> None:
    result = runner.invoke(app, ["-p", "hi", "--mode", "nonsense"])
    assert result.exit_code != 0


def test_bare_invocation_prints_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_profiles_resolve_to_bundle_documents() -> None:
    documents = resolve_profile("headless")
    assert [path.name for path in documents] == ["base.yaml", "headless.yaml"]
    with pytest.raises(ValueError):
        resolve_profile("nope")


def test_the_tui_profile_layers_onto_headless() -> None:
    documents = resolve_profile("tui")
    assert [path.name for path in documents] == ["base.yaml", "headless.yaml", "tui.yaml"]


def test_the_tui_profile_makes_the_workspace_writable() -> None:
    """The only row that differs, and the reason a `tui` profile exists.

    `read-only` is right unattended — nothing can answer an approval prompt. In
    the TUI a person is present, so the workspace is writable and everything
    outside it still asks.
    """
    result = runner.invoke(app, ["--dump-config", "--profile", "tui"])
    assert result.exit_code == 0, result.output
    rows = yaml.safe_load(result.stdout)
    sandbox = next(row for row in rows if row["id"] == "sandbox")
    assert sandbox["config"]["defaultMode"] == "workspace-write"
    assert sandbox["layer"].endswith("tui.yaml")

    headless = runner.invoke(app, ["--dump-config", "--profile", "headless"])
    rows = yaml.safe_load(headless.stdout)
    unattended = next(row for row in rows if row["id"] == "sandbox")
    assert unattended["config"]["defaultMode"] == "read-only"


def test_tui_is_an_accepted_mode() -> None:
    """`--mode tui` is offered and takes no `--print`.

    Only the wiring is asserted here — the TUI itself is covered by the pilot
    and snapshot tests, which drive it without a terminal.
    """
    from ph_app.cli import _MODES

    help_text = runner.invoke(app, ["--help"]).output
    assert "tui" in help_text
    # The one-shot table is for modes that answer a prompt and exit; the TUI
    # has its own entry point because the prompt *is* the interface.
    assert "tui" not in _MODES
