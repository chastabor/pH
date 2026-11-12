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
import socket
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import anyio
import pytest
from pydantic import BaseModel

from ph import toolcli
from ph.locks import try_file_lock
from ph.toolcli import (
    CONFIG_ENV,
    HOST_ENV,
    INHERITED_ENV,
    STATE_ENV,
    BackgroundRuns,
    Progress,
    RunProgress,
    ToolCli,
    ToolHost,
    config_env,
    decode_argument,
    env_names,
    manual_call_template,
    read_progress,
    run,
    state_dir,
    write_progress,
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


# --------------------------------------------------------------------- host ----


@contextmanager
def _answering(
    tmp_path: Path, reply: dict[str, Any]
) -> Iterator[tuple[Path, list[dict[str, Any]]]]:
    """A socket that answers every request with `reply`, as a host would, and
    the requests it was sent."""
    path = tmp_path.parent / f"{tmp_path.name}.sock"
    asked: list[dict[str, Any]] = []
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen()
    # Polled, so the thread sees the listener close: closing a socket another
    # thread is blocked in `accept` on does not wake it.
    listener.settimeout(0.05)
    closing = threading.Event()

    def serve() -> None:
        while not closing.is_set():
            try:
                link, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            link.settimeout(None)
            with link, link.makefile("rwb") as stream:
                asked.append(json.loads(stream.readline()))
                stream.write(json.dumps(reply).encode() + b"\n")
                stream.flush()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield path, asked
    finally:
        closing.set()
        thread.join(timeout=5)
        listener.close()
        path.unlink(missing_ok=True)


def test_a_call_is_answered_by_the_host_it_is_told_of_and_alone_when_there_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`$PH_TOOL_HOST` is a hint and never a requirement. With a host there, the
    call is its to answer and nothing is mounted here; with nobody listening, or
    a host that says the call is not its own — another tree, another index — the
    call runs by itself, because it could have been answered without one."""
    (tmp_path / "note.txt").write_text("from the file", encoding="utf-8")
    hosted = {"ok": True, "value": {"text": "from the host"}, "text": "from the host"}

    with _answering(tmp_path, hosted) as (path, asked):
        monkeypatch.setenv(HOST_ENV, str(path))
        _, served = _called(tmp_path, "read", "--path", "note.txt", "--limit", "3")
    with _answering(tmp_path, {"served": False}) as (path, _):
        monkeypatch.setenv(HOST_ENV, str(path))
        _, declined = _called(tmp_path, "read", "--path", "note.txt")
    monkeypatch.setenv(HOST_ENV, str(tmp_path / "nobody.sock"))
    _, alone = _called(tmp_path, "read", "--path", "note.txt")

    assert served == hosted
    assert asked == [
        {
            "root": str(tmp_path),
            "state": "",
            "tool": "read",
            "args_json": None,
            "rest": ["--path", "note.txt", "--limit", "3"],
        }
    ]
    assert "from the file" in declined["text"]
    assert "from the file" in alone["text"]


@pytest.mark.anyio
async def test_a_hosts_run_goes_on_past_the_call_and_the_same_request_joins_it(
    tmp_path: Path,
) -> None:
    """What a host adds to a long piece of work. The call that asked waits a
    moment and is answered "still going"; the same request asked again joins the
    run rather than starting it twice; and the run's last report says its end is
    owed to somebody, which is how whoever started the host knows to pass it on.
    A run that ends while its caller waits hands its value back and owes nothing,
    and one that fails says how far it got. The work says only how much there is
    and how far it is; its start, its end and its failure are written for it.
    """
    runs = BackgroundRuns(grace=0.05)
    release = anyio.Event()
    starts = 0

    async def slow(progress: RunProgress) -> str:
        nonlocal starts
        starts += 1
        progress.begin(3)
        progress.tick(1)
        await release.wait()
        return "counts"

    async def quick(progress: RunProgress) -> str:
        progress.begin(3)
        return "counts"

    async def broken(progress: RunProgress) -> str:
        progress.begin(4)
        progress.tick(2)
        raise ValueError("no")

    def said(value: str) -> str:
        return f"Three files, and one of them was skipped: {value}."

    async with anyio.create_task_group() as tasks:
        host = ToolHost(tasks=tasks)
        first = await runs.run(host, tmp_path, {"paths": ["a"]}, slow, what="files", said=said)
        again = await runs.run(host, tmp_path, {"paths": ["a"]}, slow, what="files", said=said)
        during = runs.working(tmp_path, tmp_path / "lock", what="the index")
        release.set()
        while host.working:
            await anyio.sleep(0.01)
        ended = read_progress(tmp_path)

        answered = await runs.run(host, tmp_path, {"paths": ["a"]}, quick, what="files", said=said)
        unowed = read_progress(tmp_path)
        with pytest.raises(ValueError, match="no"):
            await runs.run(host, tmp_path, {"paths": ["b"]}, broken, what="files", said=said)
        failed = read_progress(tmp_path)
        alone = await runs.run(None, tmp_path, {"paths": ["c"]}, quick, what="files", said=said)

    assert (first, again, starts) == (None, None, 1)
    assert during is not None and during.said() == "1/3 files"
    assert ended is not None and (ended.state, ended.done, ended.owed) == ("done", 3, True)
    assert ended.started > 0
    assert answered == "counts"
    assert unowed is not None and unowed.owed is False
    # What the value says goes with the report of the end, owed or not; a run
    # that failed has no value to say anything of.
    assert unowed.note == ended.note == "Three files, and one of them was skipped: counts."
    assert failed is not None and failed.note == ""
    assert unowed.started > ended.started, "a second run of one request is a new run"
    assert failed is not None and (failed.state, failed.done, failed.failure) == ("failed", 2, "no")
    assert alone == "counts", "with no host the work runs to its end"


def test_a_working_report_with_nobody_holding_the_lock_is_one_a_stopped_run_left(
    tmp_path: Path,
) -> None:
    """A run holds its lock for as long as it works, so "is it still going" is
    asked of the lock and not of a file a killed process cannot correct."""
    runs = BackgroundRuns()
    lock = tmp_path / "writer.lock"
    write_progress(tmp_path, Progress(state="working", done=2, total=9, what="documents"))

    assert BackgroundRuns().working(tmp_path, lock, what="the index") is None
    held = try_file_lock(lock, what="the test")
    assert held is not None
    try:
        found = runs.working(tmp_path, lock, what="the index")
    finally:
        held()

    assert found is not None and found.said() == "2/9 documents"
    assert read_progress(tmp_path / "absent") is None


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
