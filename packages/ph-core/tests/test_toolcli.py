"""`ph.toolcli` — a tool reached from a command line is the tool the row registered.

Gates: *a call prints one JSON object and nothing else, whatever went wrong; each
spelling a UTCP client sends for an argument decodes to the value the schema asks
for; the manual a real client reads registers every tool.*

## Why the decoder is tested against the client's own `str()`

`utcp_cli` passes each argument as `str(value)` in an environment variable, so a
list reaches this command as Python's repr and a missing optional as the literal
`MISSING_ARG_<name>`. Those are facts about another project's code, and the table
below writes each down as `str(...)` of the value rather than as a string someone
typed — so the test says what the client does, and breaks if a spelling here was
only ever a guess.

## Why `tool-fs` and not a row made up for the test

The command line is generic over rows, and the claim is that it runs a *real*
tool through the registry. `read` is the smallest shipped tool with an optional
field, an integer, and a failure worth routing.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Any

import anyio
import pytest
from pydantic import BaseModel

from ph import toolcli
from ph.locks import file_lock_held, try_file_lock
from ph.toolcli import (
    CONFIG_ENV,
    INHERITED_ENV,
    STATE_ENV,
    RunProgress,
    ToolCli,
    config_env,
    decode_argument,
    env_names,
    manual_call_template,
    reported,
    reporting_to,
    run,
    run_lock,
    state_dir,
)
from ph.wire import EnvName, WireModel

CLI = ToolCli(
    program="ph-fs",
    manual="ph_fs",
    summary="The filesystem tools, for a test.",
    version="1.2.3",
    rows=({"id": "tool-fs", "name": "tool-fs"},),
)

STRING = {"type": "string"}
NULLABLE = {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None}
STRINGS = {"type": "array", "items": {"type": "string"}}


def _run(tmp_path: Path, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = run(CLI, [*argv, "--root", str(tmp_path)], stdout=out)
    return code, out.getvalue()


def _called(tmp_path: Path, *argv: str) -> tuple[int, dict[str, Any]]:
    code, text = _run(tmp_path, "call", *argv)
    lines = text.splitlines()
    assert len(lines) == 1, f"a call prints exactly one line, not {lines!r}"
    answer: dict[str, Any] = json.loads(lines[0])
    return code, answer


# ----------------------------------------------------------------- decoding ----


@pytest.mark.parametrize(
    ("raw", "schema", "expected"),
    [
        # What the Python client sends: `str(value)`.
        (str(["a", "b"]), STRINGS, ["a", "b"]),
        (str(True), {"type": "boolean"}, True),
        (str(False), {"type": "boolean"}, False),
        (str(7), {"type": "integer"}, 7),
        (str(0.5), {"type": "number"}, 0.5),
        (str(None), NULLABLE, None),
        (str({"k": [1, 2]}), {"type": "object"}, {"k": [1, 2]}),
        # What a client that serializes properly sends.
        (json.dumps(["a", "b"]), STRINGS, ["a", "b"]),
        ("true", {"type": "boolean"}, True),
        ("null", NULLABLE, None),
        # What a client that joins a list sends, and what a person types.
        ("a,b", STRINGS, ["a", "b"]),
        ("docs", STRINGS, ["docs"]),
        ("a\nb", STRINGS, ["a", "b"]),
        # A string is the text as written, even when it looks like something else.
        ("7", STRING, "7"),
        ("[x]", STRING, "[x]"),
        ("true", STRING, "true"),
        ("None", STRING, "None"),
        # Unplaceable text is handed on for the tool's own model to refuse.
        ("seven", {"type": "integer"}, "seven"),
    ],
)
def test_each_spelling_decodes_to_what_the_schema_asks_for(
    raw: str, schema: dict[str, Any], expected: object
) -> None:
    assert decode_argument(raw, schema) == expected


def test_a_literal_with_no_json_spelling_is_not_smuggled_through() -> None:
    """`ast.literal_eval` reads sets and bytes. Neither is JSON, so neither is an
    argument; the text falls through to the comma split a bare list gets."""
    assert decode_argument("{1, 2}", STRINGS) == ["{1", "2}"]


# --------------------------------------------------------------------- call ----


def test_a_call_runs_the_registered_tool_and_prints_one_object(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("first\nsecond\n", encoding="utf-8")

    code, answer = _called(tmp_path, "read", "--path", "note.txt")

    assert code == 0
    assert answer["ok"] is True
    assert "first" in answer["text"] and "second" in answer["text"]
    assert isinstance(answer["value"], dict)


def test_a_missing_optional_argument_is_left_out_and_not_sent_as_a_value(
    tmp_path: Path,
) -> None:
    """`utcp_cli` substitutes `MISSING_ARG_<name>` for a placeholder with no
    argument. Read as text it would be a value; decoded, it would fail `limit`'s
    integer type. Left out, the tool's own default applies."""
    (tmp_path / "note.txt").write_text("one\ntwo\n", encoding="utf-8")

    code, answer = _called(
        tmp_path, "read", "--path=note.txt", "--limit=MISSING_ARG_limit", "--offset=MISSING_ARG_x"
    )

    assert code == 0, answer
    assert "two" in answer["text"]


def test_the_whole_argument_object_is_accepted_and_a_flag_wins_over_it(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("from a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("from b\n", encoding="utf-8")

    _, answer = _called(tmp_path, "read", "--args-json", '{"path": "a.txt"}', "--path=b.txt")

    assert "from b" in answer["text"]


def test_a_tool_failure_is_in_the_object_and_in_the_exit_code(tmp_path: Path) -> None:
    """Both, because a UTCP client reads neither stderr nor the exit code, and a
    person reads both."""
    code, answer = _called(tmp_path, "read", "--path", "absent.txt")

    assert code == 1
    assert answer["ok"] is False
    assert answer["error"]["message"]
    assert answer["error"]["code"]


@pytest.mark.parametrize(
    ("argv", "code"),
    [
        (("nope",), "UNKNOWN_TOOL"),
        (("read", "--bogus", "1"), "BAD_ARGUMENTS"),
        (("read", "--args-json", "{not json"), "BAD_ARGUMENTS"),
        (("read", "--args-json", "[1]"), "BAD_ARGUMENTS"),
    ],
)
def test_a_mistyped_call_is_an_object_too(tmp_path: Path, argv: tuple[str, ...], code: str) -> None:
    """argparse's own answer to a bad flag is a line on stderr and exit 2 — which
    under UTCP is an empty result with nothing saying why."""
    status, answer = _called(tmp_path, *argv)

    assert status == 1
    assert answer == {"ok": False, "error": {"code": code, "message": answer["error"]["message"]}}


def test_a_row_that_will_not_mount_is_an_object_too(tmp_path: Path) -> None:
    broken = ToolCli(
        program="ph-broken",
        manual="ph_broken",
        summary="",
        version="0",
        rows=({"id": "gone", "name": "no-such-plugin-anywhere"},),
    )
    out = io.StringIO()

    code = run(broken, ["call", "anything", "--root", str(tmp_path)], stdout=out)

    answer = json.loads(out.getvalue())
    assert code == 1
    assert answer["ok"] is False
    assert answer["error"]["code"] == "MOUNT_REFUSED"
    assert "no-such-plugin-anywhere" in answer["error"]["message"]


def test_text_format_prints_the_rendering_alone(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("only this\n", encoding="utf-8")

    code, text = _run(tmp_path, "call", "read", "--path", "note.txt", "--format", "text")

    assert code == 0
    assert "only this" in text
    assert not text.lstrip().startswith("{")


def test_a_call_leaves_the_process_logging_as_it_found_it(tmp_path: Path) -> None:
    """The command installs a file handler so nothing reaches stderr. A caller that
    runs it in its own process must not keep that handler afterwards."""
    root = logging.getLogger()
    before = (list(root.handlers), root.level)
    (tmp_path / "note.txt").write_text("x\n", encoding="utf-8")

    _called(tmp_path, "read", "--path", "note.txt")

    assert (list(root.handlers), root.level) == before


def test_a_call_whose_stdout_is_data_writes_nothing_to_stderr_whoever_tries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """A UTCP client merges stderr into the result, so one line from a dependency
    — a model library's progress bar, a native library's warning — is in front of
    the answer and it no longer parses. The descriptor is closed to them for the
    length of the run, which covers what `sys.stderr` does not, and put back
    after. A person asking for prose, or for `--verbose`, keeps the stream."""
    real = toolcli._answered

    def noisy(*args: Any, **kwargs: Any) -> int:  # noqa: ANN401
        os.write(2, b"Loading weights: 100%\n")
        print("a warning", file=sys.stderr)
        return real(*args, **kwargs)

    monkeypatch.setattr(toolcli, "_answered", noisy)
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")

    _called(tmp_path, "read", "--path", "note.txt")
    silent = capfd.readouterr().err
    _run(tmp_path, "call", "read", "--path", "note.txt", "--format", "text")
    prose = capfd.readouterr().err
    _called(tmp_path, "read", "--path", "note.txt", "--verbose")
    verbose = capfd.readouterr().err
    os.write(2, b"after\n")

    assert silent == ""
    assert "Loading weights" in prose and "a warning" in prose
    assert "Loading weights" in verbose
    assert capfd.readouterr().err == "after\n", "stderr was not put back"


# ------------------------------------------------------------------- config ----


def test_row_config_arrives_through_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """How a session's row hands its settings to the command it runs. A row the
    command does not mount is refused by name rather than ignored: a setting that
    silently went nowhere is the failure a person cannot see."""
    monkeypatch.setenv(CONFIG_ENV, json.dumps({"elsewhere": {"x": 1}}))

    _, answer = _called(tmp_path, "read", "--path", "note.txt")

    assert answer["error"]["code"] == "BAD_CONFIG"
    assert "elsewhere" in answer["error"]["message"]
    assert "tool-fs" in answer["error"]["message"]


def test_the_state_directory_is_the_flag_then_the_variable_then_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where a tool keeps what it builds, when its caller says. A session's row
    sets the variable per call; a person types the flag, which wins; and with
    neither a row gets `None` and uses its own default. The flag is the variable
    for the length of the run and no longer."""
    seen: list[Path | None] = []
    real = toolcli._answered

    def spying(*args: Any, **kwargs: Any) -> int:  # noqa: ANN401
        seen.append(state_dir())
        return real(*args, **kwargs)

    monkeypatch.setattr(toolcli, "_answered", spying)
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")

    monkeypatch.delenv(STATE_ENV, raising=False)
    _called(tmp_path, "read", "--path", "note.txt")
    monkeypatch.setenv(STATE_ENV, str(tmp_path / "from-env"))
    _called(tmp_path, "read", "--path", "note.txt")
    _called(tmp_path, "read", "--path", "note.txt", "--state", str(tmp_path / "from-flag"))

    assert seen == [None, tmp_path / "from-env", tmp_path / "from-flag"]
    assert state_dir() == tmp_path / "from-env", "the flag outlived its run"


def test_a_config_file_is_read_and_an_unreadable_one_is_said(tmp_path: Path) -> None:
    _, answer = _called(tmp_path, "read", "--path", "x", "--config", str(tmp_path / "absent.yaml"))

    assert answer["error"]["code"] == "BAD_CONFIG"
    assert "absent.yaml" in answer["error"]["message"]


# ----------------------------------------------------------------- progress ----


def _lines(text: str) -> list[dict[str, Any]]:
    """A progress stream as its records: JSONL, one whole object a line."""
    return [json.loads(line) for line in text.splitlines()]


def test_a_call_with_progress_ends_its_stderr_with_exactly_one_last_line(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """`--progress` keeps stderr for the report, and every call ends it: a tool
    that reports nothing still says `done`, and a failure says `failed` with the
    envelope's own code and sentence — so whoever reads the stream knows a run
    stopped, and why, without reading stdout. Without the flag stderr is silent,
    which is what a UTCP client needs."""
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")

    code, answer = _called(tmp_path, "read", "--path", "note.txt", "--progress")
    done = _lines(capfd.readouterr().err)
    assert code == 0 and answer["ok"] is True
    assert [line["state"] for line in done] == ["done"]

    code, answer = _called(tmp_path, "read", "--path", "absent.txt", "--progress")
    failed = _lines(capfd.readouterr().err)
    assert code == 1
    assert [line["state"] for line in failed] == ["failed"]
    assert failed[0]["code"] == answer["error"]["code"]
    assert failed[0]["failure"] == answer["error"]["message"]

    code, _answer = _called(tmp_path, "nope", "--progress")
    assert [line["code"] for line in _lines(capfd.readouterr().err)] == ["UNKNOWN_TOOL"]

    _called(tmp_path, "read", "--path", "note.txt")
    assert capfd.readouterr().err == ""


def test_progress_is_the_only_thing_on_the_kept_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """The descriptor is still closed to everything else for the length of the
    run: a dependency's progress bar would otherwise be a line a reader has to
    tell from a record."""
    real = toolcli._answered

    def noisy(*args: Any, **kwargs: Any) -> int:  # noqa: ANN401
        os.write(2, b"Loading weights: 100%\n")
        print("a warning", file=sys.stderr)
        return real(*args, **kwargs)

    monkeypatch.setattr(toolcli, "_answered", noisy)
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")

    _called(tmp_path, "read", "--path", "note.txt", "--progress")

    assert [line["state"] for line in _lines(capfd.readouterr().err)] == ["done"]


@pytest.mark.anyio
async def test_a_run_reports_as_it_goes_and_its_end_says_what_it_did() -> None:
    """`reported` is the one door for a long piece of work: the work says how
    much there is and how far it has got, and the door writes the end — with
    what the value says and whether it changed anything, since a run nobody
    waited for is told about from that line alone. Lines in between are held to
    one every `REPORT_EVERY_S`."""
    out = io.StringIO()

    async def work(progress: RunProgress) -> dict[str, int]:
        progress.begin(3)
        progress.tick(1)  # within the interval of `begin`: not written
        progress._written -= toolcli.REPORT_EVERY_S
        progress.tick(2)
        return {"added": 2}

    with reporting_to(out):
        value = await reported(
            work,
            what="documents",
            said=lambda value: f"{value['added']} added",
            changed=lambda value: bool(value["added"]),
            busy=lambda _: False,
        )

    assert value == {"added": 2}
    assert _lines(out.getvalue()) == [
        {"state": "working", "done": 0, "total": 3, "what": "documents"},
        {"state": "working", "done": 2, "total": 3, "what": "documents"},
        {
            "state": "done",
            "done": 3,
            "total": 3,
            "what": "documents",
            "changed": True,
            "note": "2 added",
        },
    ]


@pytest.mark.anyio
async def test_a_run_that_found_its_work_held_by_another_ends_busy_and_not_done() -> None:
    """A run that waited out `wait_s` behind another and did nothing has not
    finished anything: its last line says `busy`, with no note, so whoever is
    told of its end is not told that a run of nothing covered everything."""
    out = io.StringIO()

    async def work(progress: RunProgress) -> dict[str, str]:
        return {"state": "indexing"}

    with reporting_to(out):
        await reported(
            work,
            what="files",
            said=lambda _: "Indexed 0 files.",
            changed=lambda _: False,
            busy=lambda value: value["state"] == "indexing",
        )

    assert _lines(out.getvalue()) == [{"state": "busy", "done": 0, "total": 0, "what": "files"}]


@pytest.mark.anyio
async def test_a_failed_runs_last_line_is_the_commands_and_carries_how_far_it_got() -> None:
    """The work's failure becomes an envelope further out, and that is where its
    last line is written from — the code and sentence a caller reads — with the
    counts, and the spend, the run last reported. Nothing is written after a
    last line."""
    out = io.StringIO()
    reports = toolcli._Reports(out)

    async def work(progress: RunProgress) -> None:
        progress.begin(5)
        progress.spent = lambda: {"requests": 2}
        raise ValueError("the disk is full")

    with toolcli._set(toolcli._REPORTS, reports), pytest.raises(ValueError):
        await reported(
            work, what="files", said=lambda _: "", changed=lambda _: False, busy=lambda _: False
        )
    reports.end(code="INTERNAL_ERROR", failure="ValueError: the disk is full")
    reports.end()

    assert _lines(out.getvalue())[1:] == [
        {
            "state": "failed",
            "done": 0,
            "total": 5,
            "what": "files",
            "code": "INTERNAL_ERROR",
            "failure": "ValueError: the disk is full",
            # What it spent before it failed: a run that cost something and
            # built nothing is the one most worth tracking.
            "spent": {"requests": 2},
        }
    ]


@pytest.mark.anyio
async def test_a_run_waits_its_turn_and_past_the_wait_is_told_another_holds_the_lock(
    tmp_path: Path,
) -> None:
    """One run at a time on what a tool writes. A second waits `wait_s` for the
    first and is then given the lock, or told it is still held — it is a call,
    with a limit of its own, and answers that a run is working rather than
    stand behind it. `call --wait` is the caller's own word for how long."""
    lock = tmp_path / "index.lock"

    async with run_lock(lock, what="the index", wait_s=0.0) as first:
        assert first is True
        assert file_lock_held(lock)
        async with run_lock(lock, what="the index", wait_s=0.05) as second:
            assert second is False
        with toolcli._set(toolcli._WAIT, 0.0):
            async with run_lock(lock, what="the index", wait_s=60.0) as at_once:
                assert at_once is False

    release = try_file_lock(lock, what="the index")
    assert release is not None

    async def later() -> None:
        await anyio.sleep(0.15)
        release()

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(later)
        async with run_lock(lock, what="the index", wait_s=5.0) as waited:
            assert waited is True
    assert not file_lock_held(lock)


# ---------------------------------------------------------------- providers ----

PICKING = ToolCli(
    program="ph-pick",
    manual="ph_pick",
    summary="One of two rows, for a test.",
    version="1.0.0",
    rows=(),
    providers=(
        {"id": "tool-fs", "name": "tool-fs"},
        # Mounts, and registers nothing without a session: all this needs.
        {"id": "tool-ask-user", "name": "tool-ask-user"},
    ),
)


def _picked(tmp_path: Path) -> tuple[int, dict[str, Any]]:
    out = io.StringIO()
    code = run(PICKING, ["manual", "--root", str(tmp_path)], stdout=out)
    answer: dict[str, Any] = json.loads(out.getvalue())
    return code, answer


def test_one_provider_row_is_mounted_the_first_unless_a_config_names_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package whose tools need something one of several rows supplies — an
    embedder that is a local model here and an endpoint there. Which is the
    deployment's choice, made the way every other is: by configuring that row.
    Being named is enough, which is why `config_env` names a row whose config is
    all defaults; and naming two is refused rather than resolved by order."""
    monkeypatch.delenv(CONFIG_ENV, raising=False)
    _, default = _picked(tmp_path)
    monkeypatch.setenv(CONFIG_ENV, json.dumps({"tool-ask-user": {}}))
    _, other = _picked(tmp_path)
    monkeypatch.setenv(CONFIG_ENV, json.dumps({"tool-fs": {}, "tool-ask-user": {}}))
    code, both = _picked(tmp_path)

    assert "read" in {one["name"] for one in default["tools"]}
    assert other["tools"] == []
    assert code == 1
    assert both["error"]["code"] == "BAD_CONFIG"
    assert "tool-fs, tool-ask-user" in both["error"]["message"]


def test_a_config_says_once_which_of_its_fields_name_a_variable() -> None:
    """`EnvName` on the field is the one declaration: the command lists the
    variable in its manual and a session's row hands it across, both by reading
    this — in a nested model too. A name that is not set is not one to pass, and
    one named twice is passed once."""

    class Upstream(WireModel):
        token_env: Annotated[str, EnvName()] = ""

    class Keyed(WireModel):
        api_key_env: Annotated[str, EnvName()] = ""
        proxy_env: Annotated[str, EnvName()] = "PROXY"
        model: str = "KEY_SHAPED_BUT_NOT_MARKED"
        upstream: Upstream = Upstream()

    assert env_names(Keyed()) == ["PROXY"]
    assert env_names(Keyed(api_key_env="FX_KEY", proxy_env="")) == ["FX_KEY"]
    nested = Keyed(upstream=Upstream(token_env="UP_TOKEN"))
    assert env_names(nested, Keyed()) == ["PROXY", "UP_TOKEN"]


def test_a_row_with_default_config_is_still_named_to_the_command() -> None:
    class Settings(BaseModel):
        depth: int = 2

    assert config_env({}) == ({}, [])
    env, passed = config_env({"a": Settings(), "b": Settings(depth=3)})
    assert json.loads(env[CONFIG_ENV]) == {"a": {}, "b": {"depth": 3}}
    assert passed == []


# ------------------------------------------------------------------- manual ----


def _manual(tmp_path: Path) -> dict[str, Any]:
    code, text = _run(tmp_path, "manual")
    assert code == 0
    manual: dict[str, Any] = json.loads(text)
    return manual


def test_the_manual_names_every_tool_with_a_command_that_calls_it(tmp_path: Path) -> None:
    manual = _manual(tmp_path)
    tools = {one["name"]: one for one in manual["tools"]}

    assert manual["manual_version"] == "1.2.3"
    assert "read" in tools
    template = tools["read"]["tool_call_template"]
    command = template["commands"][0]["command"]
    assert template["call_template_type"] == "cli"
    assert command.startswith("ph-fs call read --format json ")
    # One word per field, so a value that starts with `-` is still an argument.
    assert "--path=UTCP_ARG_path_UTCP_END" in command
    # No redirection: the client appends its own `2>&1` after it, so one would
    # do nothing. The command keeps its stderr quiet itself.
    assert ">" not in command


def test_every_template_names_the_roots_or_a_call_builds_a_second_index(tmp_path: Path) -> None:
    """Naming any variable replaces the client's default allowlist, so `PATH` has
    to be said again beside the pH roots."""
    manual = _manual(tmp_path)

    for name in ("PATH", "HOME", "PH_HOME", "PH_CACHE", "PH_RUNTIME", CONFIG_ENV, STATE_ENV):
        assert name in INHERITED_ENV
    for tool in manual["tools"]:
        assert tool["tool_call_template"]["inherit_env_vars"] == list(INHERITED_ENV)
    assert manual_call_template(CLI)["inherit_env_vars"] == list(INHERITED_ENV)


def test_the_output_schema_keeps_its_definitions_where_references_resolve(tmp_path: Path) -> None:
    """pydantic writes `#/$defs/Name`, resolved from the document's root. A tool's
    output is nested under `value` here, so its `$defs` have to be lifted or every
    reference points at nothing."""
    manual = _manual(tmp_path)

    for tool in manual["tools"]:
        outputs = tool["outputs"]
        assert set(outputs["properties"]) == {"ok", "value", "text", "error"}
        assert "$defs" not in outputs["properties"]["value"]
        if "#/$defs/" in json.dumps(outputs["properties"]["value"]):
            assert "$defs" in outputs


def test_a_real_utcp_library_reads_the_manual(tmp_path: Path) -> None:
    """The gate the format exists for: not "it looks like a manual" but "the
    library that reads manuals reads this one", every tool included.

    The import is fenced because `utcp` calls `logging.basicConfig` when it is
    imported, and a handler added to the root logger here would outlive this test.
    """
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        from utcp.data.utcp_manual import UtcpManualSerializer  # type: ignore[import-untyped]
        from utcp.plugins.plugin_loader import (  # type: ignore[import-untyped]
            ensure_plugins_initialized,
        )

        ensure_plugins_initialized()
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
    raw = _manual(tmp_path)

    manual = UtcpManualSerializer().validate_dict(raw)

    assert [one.name for one in manual.tools] == [one["name"] for one in raw["tools"]]
    assert all(one.tool_call_template.call_template_type == "cli" for one in manual.tools)
