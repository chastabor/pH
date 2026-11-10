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
from pathlib import Path
from typing import Any

import pytest

from ph.toolcli import (
    CONFIG_ENV,
    INHERITED_ENV,
    ToolCli,
    decode_argument,
    manual_call_template,
    run,
)

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


def test_a_config_file_is_read_and_an_unreadable_one_is_said(tmp_path: Path) -> None:
    _, answer = _called(tmp_path, "read", "--path", "x", "--config", str(tmp_path / "absent.yaml"))

    assert answer["error"]["code"] == "BAD_CONFIG"
    assert "absent.yaml" in answer["error"]["message"]


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
    # stderr is merged into the result by the client; a dependency's warning
    # must not be.
    assert command.endswith(" 2>/dev/null")


def test_every_template_names_the_roots_or_a_call_builds_a_second_index(tmp_path: Path) -> None:
    """Naming any variable replaces the client's default allowlist, so `PATH` has
    to be said again beside the pH roots."""
    manual = _manual(tmp_path)

    for name in ("PATH", "HOME", "PH_HOME", "PH_CACHE", "PH_RUNTIME", CONFIG_ENV):
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
