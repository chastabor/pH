"""`ph.toolcli`: the tools of a package as a command line, and as a UTCP manual.

A package that registers tools through rows gets a console script by naming
those rows: `ToolCli(program=..., rows=...)` and `main`. The script mounts a
small profile: the tool registry, the filesystem, a subprocess service, the
`commands` and `diagnostics` rows, and the rows of the package. It runs a tool
**through the registry**, as `ph.testing.run_tool` does. Each tool therefore
has one body: the body that the row registers. The only difference between a
session and this command is the process that mounted the row.

## Progress, for a run that takes a while

One command is one call. No process stays up. A tool whose work is long, such
as an indexing run, reports its progress on **stderr**, as JSONL, when it is
called with `--progress` (`Progress`, `reported`). It writes a line when it
begins, a line at most every `REPORT_EVERY_S`, and exactly one last line:
`done`, `failed` or `busy`. The process that started the command reads that
stream for a footer. It knows that the run stopped when the stream ends, with a
last line or without one. stdout is unchanged: one envelope. The flag is not
for a UTCP client, which merges the two streams. It is for a caller that reads
the streams separately, as `ph-utcp` does for a run that it starts itself.

One program serves three audiences:

* a **person** who tries a query by hand (`call code_graph --mode define --query x
  --format text`), or who asks whether the package is provisioned (`status`)
* a **UTCP client**, which runs `manual` one time to learn the tools and then
  runs the command that the template of each tool names
* `ph-utcp`, which is a UTCP client inside a pH session

## What the UTCP command-line protocol decides here

These points were read from `utcp_cli` 1.1.5. Every item below is a constraint,
not a preference.

* **Arguments arrive one at a time, as `str(value)`.** A list arrives as the
  repr of Python, a boolean arrives as `True`, and an optional argument that the
  model left out arrives as the literal `MISSING_ARG_<name>`. So there is one
  flag for each top-level field, and `decode_argument` reads each flag by the
  schema of the tool: JSON first, then the Python spellings, then a comma split
  for an array. The manual declares the true types, and this module accepts
  what a client actually sends.
* **stderr is merged into stdout, and the exit code is ignored.** So a call
  prints exactly one JSON object, a failure is reported inside that object, and
  logging goes to a file. For the length of a `manual`, or of a `call` that
  prints JSON, the stderr of the process is also closed to everything in the
  process (`_stderr_discarded`), because this module has no other way to
  silence a dependency that writes a warning or a progress bar. A `2>/dev/null`
  on the template does not work: the client wraps each command as
  `$(<command> 2>&1)`, and the later redirection wins.
* **The environment is cut to an allowlist.** Naming any variable replaces the
  default list of the client, so the template names the defaults again beside
  the pH roots. Without the roots, a UTCP call would build a different index
  from the one that a session uses for the same checkout. A variable that only
  the config of a row names is listed too (`EnvName`).

@module ph.toolcli
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import re
import shlex
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, asynccontextmanager, contextmanager, redirect_stderr
from contextvars import ContextVar
from dataclasses import dataclass
from json import JSONDecodeError
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal, NoReturn, TextIO

import anyio
from pydantic import BaseModel

from .cancel import Canceled
from .cordis import DEPLOYMENT, Context, CordisError, Profile, maybe_await
from .cordis.loader import ProfileDocument
from .documents import DOCUMENT_FAULTS, decode_document
from .json import JsonObject, JsonValue, as_obj, as_str, dumps, loads
from .keys import COMMANDS, DIAGNOSTICS, MOUNT, TOOLS
from .llm.types import text_of
from .locks import wait_for_file_lock
from .paths import resolve_roots
from .seams.commands import CommandContext
from .tools.definition import ToolDefinition, ToolExecutionInput, ToolExecutionResult
from .tools.errors import HarnessError
from .tools.json_schema import schema_of
from .wire import EnvName, WireModel
from .workspace_caches import digest, use

__all__ = [
    "COMMAND_ROWS",
    "CONFIG_ENV",
    "INHERITED_ENV",
    "REPORT_EVERY_S",
    "STATE_ENV",
    "UTCP_VERSION",
    "Progress",
    "RunProgress",
    "ToolCli",
    "claim_home",
    "config_env",
    "decode_argument",
    "env_names",
    "envelope",
    "failure_envelope",
    "main",
    "manual_call_template",
    "read_envelope",
    "reported",
    "reporting_to",
    "run",
    "run_lock",
    "state_dir",
    "state_home",
    "utcp_manual",
    "value_envelope",
    "value_schema",
]

log = logging.getLogger("ph.toolcli")

UTCP_VERSION = "1.2.0"
"""The UTCP version that the generated manual is written against: the one that was read."""

CONFIG_ENV = "PH_TOOL_CONFIG"
"""A JSON object of row configs by row id, for a caller that mounts nothing itself.

This is how the row of a session passes its config across the process boundary.
`ph-utcp` sets the variable on the command that it runs, so the tool indexes
with the settings that the profile names and not with the defaults of the
package. `config_env` writes it and `_row_configs` reads it."""

STATE_ENV = "PH_TOOL_STATE"
"""The directory where a tool keeps its working state, when its caller names one.

Working state is what a tool builds *about* a workspace and does not put in it:
an index, a database, a set of vectors. `ph-utcp` sets this variable on each
call to a directory in the scratch directory of the calling agent, so the state
belongs to that agent. The directory is writable in a read-only session, where
the workspace is not, and the same agent finds it again after a restart.
`--state` is the same setting typed by a person, and it wins. With neither, a
tool keeps its state where it always has: under `$PH_CACHE`, keyed by the
workspace. A row reads the setting through `state_dir`."""

MISSING_ARGUMENT = "MISSING_ARG_"
"""What `utcp_cli` substitutes for a placeholder whose argument was not given."""

COMMAND_ROWS: tuple[JsonObject, ...] = (
    {"id": "tools", "name": "tools"},
    # This command is not a pH start, so it sweeps nothing that a hard-killed
    # run left behind. It also could not judge such a run. When it runs for an
    # agent, it is in the PID namespace of the sandbox, where every process that
    # the journal names looks dead.
    {"id": "subprocess", "name": "subprocess-local", "config": {"sweepOrphans": False}},
    {"id": "fs", "name": "fs-local"},
    {"id": "commands", "name": "commands"},
    {"id": "diagnostics", "name": "diagnostics"},
)
"""What a tool needs around it and no more: no model, no session, no agent.

`subprocess` is here for `ph.seams.changes`, which asks git or jj what changed.
`commands` and `diagnostics` are here so that the provisioning command of a
package and its doctor section exist for `status` and `install` to read."""

INHERITED_ENV: tuple[str, ...] = (
    # The default list of `utcp_cli`. Naming a variable replaces that list and
    # does not extend it, so these are repeated. Without them the command loses
    # `PATH`.
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "USER",
    "LOGNAME",
    "SHELL",
    "TZ",
    "TERM",
    # Where pH keeps things. Without these, a call resolves different roots from
    # the session beside it, and the two build separate indexes of one checkout.
    "PH_HOME",
    "PH_CACHE",
    "PH_RUNTIME",
    "XDG_CACHE_HOME",
    "XDG_RUNTIME_DIR",
    CONFIG_ENV,
    # Another harness sets this to say where the state of a tool goes.
    STATE_ENV,
)

LOG_BYTES = 1024 * 1024
"""How large the log of a command grows before it is rotated."""

_PLACEHOLDER_NAME = re.compile(r"[A-Za-z0-9_]+")
"""The names that the placeholder pattern of `utcp_cli` can carry."""


@dataclass(frozen=True, slots=True)
class ToolCli:
    """The command line of one package: its program, its rows, and what it provisions."""

    program: str
    """The name of the console script, as it is typed: `ph-code-graph`."""
    manual: str
    """The name of the UTCP manual. Letters, digits and underscores: `ph_code_graph`."""
    summary: str
    version: str
    rows: tuple[JsonObject, ...]
    """The rows of the package, as a profile would spell them."""
    providers: tuple[JsonObject, ...] = ()
    """Rows of which exactly one is mounted, after `rows`: the one that the row
    configs name, or the first when they name none.

    This is for a package whose tools need something that only one of several
    rows supplies. An example is an embedder that is a local model in one
    deployment and an endpoint in another. The deployment chooses the row in the
    way that it makes every other choice: it configures that row, in the file
    under `$PH_HOME/tools`, in `$PH_TOOL_CONFIG`, or with `--config`. For this
    purpose, `config_env` names a row even when its config is all defaults.
    Naming two rows is refused."""
    command: str = ""
    """The provisioning command of the package (`code-graph`), or empty for none.

    It is the same `install_or_status` command that a session offers as
    `/code-graph`. It stays out of the manual for the reason that the packages
    give for making it a command: provisioning is something that a person asks
    for."""
    inherit_env: tuple[str, ...] = ()
    """Variables this package reads, beyond `INHERITED_ENV`."""
    module: str = ""
    """The `__main__` of the package. The interpreter that printed the manual runs
    it as `python -P -m <module>`, so the commands of the package run in that
    interpreter, whatever `PATH` the calling harness has. When this is empty, a
    template spells the program by its script name."""

    def invocation(self) -> tuple[str, ...]:
        """How a generated template spells the program.

        `-P` is needed because `-m` otherwise puts the working directory first on
        `sys.path`, and the working directory is the tree that the tool works
        on. A `shlex.py` at the root of a repository would be imported in place
        of the standard library module. The command may hold a credential. For
        discovery, and for a `status` that a person runs, it also runs outside
        any sandbox."""
        return (sys.executable, "-P", "-m", self.module) if self.module else (self.program,)

    def log_file(self) -> Path:
        """Where the command logs. A call prints one object and nothing on stderr,
        so a traceback goes here. A confined command must be granted this
        path."""
        return resolve_roots().cache / "logs" / f"{self.program}.log"


def config_env(configs: Mapping[str, BaseModel]) -> tuple[dict[str, str], list[str]]:
    """What a session gives a command for the configs of these rows.

    The first value is the `$PH_TOOL_CONFIG` that carries the configs, by row
    id. The second value is the variables that the configs mark (`env_names`),
    to pass across by name. They are the `env` and the `pass_env` of a `Manual`.

    Each row is named, with only what differs from the defaults of its model, so
    a default row travels as `{}`. The row is named even then, because a command
    that mounts one of several provider rows learns which one from the name
    (`ToolCli.providers`)."""
    settings = {
        row: config.model_dump(by_alias=True, exclude_defaults=True)
        for row, config in configs.items()
    }
    env = {CONFIG_ENV: dumps(settings)} if settings else {}
    return env, env_names(*configs.values())


def state_dir() -> Path | None:
    """Where the caller said the tools of this process keep their state
    (`STATE_ENV`), or `None` for the default of the tool.

    The value is read from the environment at each use and is not held. A row
    therefore asks for it where it builds a path, and a test that sets the
    variable takes effect. The path is absolute: a relative directory is
    resolved against the working directory, one time, here."""
    raw = os.environ.get(STATE_ENV, "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def state_home(base: Path, workspace: Path) -> Path:
    """Where a tool keeps what it builds about `workspace`: the directory that its
    caller named (`state_dir`), or the directory of `workspace` under `base`.
    That directory is keyed by a digest of the root, so two checkouts do not
    answer the questions of each other."""
    state = state_dir()
    return state if state is not None else base / digest(str(workspace))


def claim_home(base: Path, workspace: Path) -> None:
    """Record that the directory of `workspace` under `base` is in use, and sweep
    the directories whose workspace has been gone for a while
    (`ph.workspace_caches.use`).

    The function does nothing for a directory that the caller named: its owner
    keeps or removes it. The function blocks on file I/O, so it is called
    through a worker thread."""
    if state_dir() is None:
        use(base, workspace)


# ----------------------------------------------------------------- progress ----


class Progress(WireModel):
    """One line of the report of a run: how far a long piece of work has got, or
    how it ended. `call --progress` writes these lines to stderr, one JSON
    object per line.

    `working` lines come while the run goes. The last line is `done`, `failed`,
    or `busy`. `busy` means that another run held what this run writes, and this
    run did nothing. There is exactly one last line. A stream that ends without
    it is a command that was stopped from outside, or that died."""

    state: Literal["working", "done", "failed", "busy"]
    done: int = 0
    total: int = 0
    """Units of work, such as documents or files: the number finished, and the number in all."""
    what: str = ""
    """What the units are, plural: `documents`."""
    code: str | None = None
    """On `failed`: the routable code of the failure, the same code as in the envelope."""
    failure: str | None = None
    """On `failed`: why, in a sentence."""
    changed: bool | None = None
    """On `done`: whether the run changed what it builds. A run that found
    everything current says `false`, and nobody needs to be told that it ended."""
    note: str | None = None
    """On `done`: what the run has to say beyond its count, in sentences. A caller
    that waited for the run read the same text in the result of the tool."""
    spent: dict[str, int] | None = None
    """What the run has spent so far, by unit, as the tool counts it. An example is
    the `requests` and `texts` of an embedder, and the `tokens` that its provider
    reported. It is on every line, the last included, so the reader of the
    stream can track what a run cost, also for a run that failed or was killed.
    Absent for a tool that counts nothing."""

    def said(self) -> str:
        """One compact phrase, for a footer and for a sentence."""
        if self.state == "failed":
            return f"failed: {self.failure}"
        if self.state == "busy":
            return "held by another run"
        counted = f"{self.done}/{self.total}" if self.state == "working" else str(self.total)
        return f"{counted} {self.what}".strip()


@dataclass(slots=True)
class _Reports:
    """Where the progress lines of this command go: the stream that `--progress` kept."""

    out: TextIO
    ended: bool = False
    """A last line has been written, and nothing more is written."""
    current: RunProgress | None = None
    """The run that reports here (`reported`). A last line that is written from
    outside the work carries the counts and the spend of this run as they are
    now. It does not repeat the last line written, which is up to
    `REPORT_EVERY_S` behind."""

    def write(self, progress: Progress) -> None:
        """Write one line, whole and flushed. Nothing is written after the last
        line. The method never raises: a reader that has gone must not stop the
        work that it was reading."""
        if self.ended:
            return
        self.ended = progress.state != "working"
        try:
            self.out.write(dumps(progress.to_wire()) + "\n")
            self.out.flush()
        except (OSError, ValueError):
            log.debug("ph.toolcli: a progress line was not written", exc_info=True)

    def end(self, *, code: str | None = None, failure: str | None = None) -> None:
        """The last line, for a command whose work wrote none: `done`, or `failed`
        with the code and the sentence of the envelope and how far the run had
        got."""
        state: Literal["done", "failed"] = "done" if failure is None else "failed"
        run = self.current
        if run is None:
            self.write(Progress(state=state, code=code, failure=failure))
        else:
            self.write(run.line(state, code=code, failure=failure))


_REPORTS: ContextVar[_Reports | None] = ContextVar("ph_toolcli_reports", default=None)
"""Set by `run` for a `call --progress`. `None` for every other command, whose
work then reports to nobody."""

_WAIT: ContextVar[float | None] = ContextVar("ph_toolcli_wait", default=None)
"""`call --wait`: how long this call waits for another run. It overrides the setting of the row."""

REPORT_EVERY_S = 0.5
"""How often the progress of a run is written, at most."""


def env_names(*configs: BaseModel) -> list[str]:
    """The variables that `configs` name: the value of each set `EnvName` field,
    including the fields of a nested model, one time each, in order.
    `ToolCli.inherit_env` is what a package always reads. These are what the
    settings of one deployment add."""

    def named(config: BaseModel) -> Iterator[str]:
        for name, field_ in type(config).model_fields.items():
            value = getattr(config, name)
            if isinstance(value, BaseModel):
                yield from named(value)
            elif (
                isinstance(value, str)
                and value
                and any(isinstance(mark, EnvName) for mark in field_.metadata)
            ):
                yield value

    return list(dict.fromkeys(name for config in configs for name in named(config)))


@dataclass(slots=True)
class RunProgress:
    """The account that one run gives of itself: how much work there is, and how
    far it has got.

    `reported` passes it to the work and writes the end of the run. The work
    says only what it alone knows. It calls `begin` when it knows how much work
    there is, under whatever lock makes the run its own, and `tick` as it goes."""

    what: str
    """What the units are, plural: `documents`."""
    reports: _Reports | None = None
    """Where the lines go: the stream of `call --progress`, or nobody."""
    spent: Callable[[], Mapping[str, int]] | None = None
    """How to ask what the run has spent so far (`Progress.spent`). Work that
    counts something sets it one time. It is asked as each line is written, so
    every line, including the last line of a failed run, gives the current
    value."""
    total: int = 0
    done: int = 0
    _written: float = 0.0

    def begin(self, total: int) -> None:
        """There are `total` units to go through, and none done yet."""
        self.total, self.done = total, 0
        self._write()

    def tick(self, done: int) -> None:
        """`done` units are finished. A line is written at most every `REPORT_EVERY_S`."""
        self.done = done
        if time.monotonic() - self._written >= REPORT_EVERY_S:
            self._write()

    def line(
        self,
        state: Literal["working", "done", "failed", "busy"],
        *,
        note: str | None = None,
        changed: bool | None = None,
        code: str | None = None,
        failure: str | None = None,
    ) -> Progress:
        """This run's line, as it stands now."""
        spent = None if self.spent is None else dict(self.spent())
        return Progress(
            state=state,
            done=self.done,
            total=self.total,
            what=self.what,
            changed=changed,
            note=note,
            code=code,
            failure=failure,
            spent=spent or None,
        )

    def _write(
        self,
        state: Literal["working", "done", "busy"] = "working",
        *,
        note: str | None = None,
        changed: bool | None = None,
    ) -> None:
        if self.reports is None:
            return
        self._written = time.monotonic()
        self.reports.write(self.line(state, note=note, changed=changed))


async def reported[T](
    work: Callable[[RunProgress], Awaitable[T]],
    *,
    what: str,
    said: Callable[[T], str],
    changed: Callable[[T], bool],
    busy: Callable[[T], bool],
) -> T:
    """The value of `work`, with the run reported as it goes and its end written.
    This function writes the end, so no row can forget the last line.

    `said` gives what the value says beyond its count, in a few lines. It is
    what the result of the tool tells a caller that waited, written for whoever
    is told of a run that nobody waited for. It is required: a run whose note
    says only "finished" can hide a run that skipped everything. `changed` says
    whether the value changed what the run builds, so that a run over a tree
    that had not changed is not news. `busy` says whether the value is the
    answer that another run held what this run writes (`run_lock`). The last
    line then says that, and not that the run finished.

    `run` writes the last line of a failure, from the envelope that the failure
    becomes: the code and the sentence that a caller reads. This function
    handles the traceback. A traceback that the tool did not raise as its own
    failure goes to the log of the command. Otherwise a run that nobody waited
    for would leave no traceback.
    """
    reports = _REPORTS.get()
    progress = RunProgress(what, reports)
    if reports is not None:
        reports.current = progress
    try:
        value = await work(progress)
    except (HarnessError, Canceled):
        raise
    except Exception:
        log.exception(
            "ph.toolcli: a %s run failed at %d of %d", what, progress.done, progress.total
        )
        raise
    if busy(value):
        progress._write("busy")
        return value
    progress.done = progress.total
    progress._write("done", note=said(value), changed=changed(value))
    return value


@asynccontextmanager
async def run_lock(lock: Path, *, what: str, wait_s: float) -> AsyncIterator[bool]:
    """Hold `lock` for one run against every other run. The wait for the lock is
    at most `wait_s`, or `call --wait` where the caller gave it. Yields whether
    the lock is held.

    A second run is expected to queue behind the first and then find most of its
    work done. But a caller is a call with a limit of its own, so the wait is
    bounded. After the bound, the body runs with `False`, and answers that
    another run is working instead of waiting behind it. This function does not
    need to make the directory, because taking the lock makes it."""
    patience = _WAIT.get()
    release: Callable[[], None] | None = None
    # A free lock is taken before the first await, so a bound of 0 still takes it.
    with anyio.move_on_after(wait_s if patience is None else patience):
        release = await wait_for_file_lock(lock, what=what)
    try:
        yield release is not None
    finally:
        if release is not None:
            release()


# ----------------------------------------------------------------- decoding ----


def _kinds(schema: Mapping[str, object]) -> set[str]:
    """The JSON types a field's schema admits, through one level of `anyOf`."""
    found: set[str] = set()
    declared = schema.get("type")
    if isinstance(declared, str):
        found.add(declared)
    elif isinstance(declared, list):
        found.update(one for one in declared if isinstance(one, str))
    for key in ("anyOf", "oneOf"):
        members = schema.get(key)
        if isinstance(members, list):
            for member in members:
                if isinstance(member, Mapping):
                    found |= _kinds(member)
    return found


def _plain(value: object) -> JsonValue:
    """A Python literal as JSON. Raises `ValueError` for a literal with no JSON
    spelling: a set, bytes, a key that is not a string."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(one) for one in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _plain(one) for key, one in value.items()}
    raise ValueError(f"{type(value).__name__} has no JSON spelling")


def _structured(raw: str) -> list[JsonValue] | dict[str, JsonValue] | None:
    """`raw` as a list or an object, read as JSON or as a Python literal. These are
    the two spellings that the `str()` of a client produces for one. `None` when
    it is neither."""
    try:
        found: JsonValue = loads(raw)
    except JSONDecodeError:
        try:
            found = _plain(ast.literal_eval(raw))
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    return found if isinstance(found, (list, dict)) else None


_TRUE = frozenset({"true", "1", "yes", "on"})
_FALSE = frozenset({"false", "0", "no", "off"})


def decode_argument(raw: str, schema: Mapping[str, object]) -> JsonValue:
    """The text of one flag as the value that its schema asks for.

    **Decoding is lenient, because the registry validates afterwards.** A text
    that this function cannot place is passed on as the string that it is. The
    model of the tool then refuses it with a message that names the field. So
    nothing here raises on the phrasing of a client.

    A string field takes its text as written, with one exception: where the
    schema admits `null`, `None` and `null` mean null. The cost is that a search
    for a symbol literally named `None` is not possible under a nullable field.
    The benefit is an explicit null from the Python client, whose `str(None)`
    would otherwise be a query.

    This function never sees an argument that the client left out. The caller
    drops a `MISSING_ARG_` placeholder before decoding, because no value, `None`
    included, is the same as no argument.
    """
    kinds = _kinds(schema)
    text = raw.strip()
    if "null" in kinds and text in ("None", "null"):
        return None
    if "boolean" in kinds:
        folded = text.casefold()
        if folded in _TRUE:
            return True
        if folded in _FALSE:
            return False
    if "integer" in kinds:
        try:
            return int(text)
        except ValueError:
            pass
    if "number" in kinds:
        try:
            return float(text)
        except ValueError:
            pass
    if kinds & {"array", "object"}:
        found = _structured(text)
        if isinstance(found, list) and "array" in kinds:
            return found
        if isinstance(found, dict) and "object" in kinds:
            return found
        if "array" in kinds and "string" not in kinds:
            # What is left is a bare list: `a,b` from a client that joins, or one
            # path that a person typed. Split on commas and newlines.
            return [one.strip() for one in re.split(r"[,\n]", text) if one.strip()]
    return raw


# ------------------------------------------------------------------- manual ----


def _properties(definition: ToolDefinition) -> dict[str, Mapping[str, object]]:
    properties = definition.parameters.get("properties")
    if not isinstance(properties, Mapping):
        return {}
    return {str(name): schema for name, schema in properties.items() if isinstance(schema, Mapping)}


def _result_schema(definition: ToolDefinition) -> dict[str, object]:
    """The envelope that `call` prints, with the output of the tool as its `value`.

    `$defs` is lifted to the root of the envelope. Pydantic writes its
    references as `#/$defs/Name`, which resolve from the root of the document
    that they are in. A definition that was left inside `value` would therefore
    point to nothing. `value_schema` is the inverse."""
    value = dict(schema_of(definition.output.schema))
    definitions = value.pop("$defs", None)
    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean", "description": "Whether the call succeeded."},
            "value": value,
            "text": {
                "type": "string",
                "description": "The result as prose, written for a model to read.",
            },
            "error": {
                "type": "object",
                "properties": {"code": {"type": "string"}, "message": {"type": "string"}},
            },
        },
        "required": ["ok"],
    }
    if definitions is not None:
        schema["$defs"] = definitions
    return schema


def _template(cli: ToolCli, definition: ToolDefinition, inherit: list[str]) -> dict[str, object]:
    words = [shlex.join((*cli.invocation(), "call", definition.name, "--format", "json"))]
    for name in _properties(definition):
        if _PLACEHOLDER_NAME.fullmatch(name) is None:
            raise ValueError(
                f'tool "{definition.name}" has a field "{name}" that a UTCP placeholder '
                "cannot name; placeholders take letters, digits and underscores"
            )
        # `--name=value` is one word. A value that starts with `-` (a query for
        # `-x`, a negative number) is then an argument and not a flag.
        words.append(f"--{name}=UTCP_ARG_{name}_UTCP_END")
    # `shlex.join` is not used for the whole line: a placeholder must stay a
    # bare word for the client to substitute it.
    return {
        "call_template_type": "cli",
        "commands": [{"command": " ".join(words)}],
        "inherit_env_vars": inherit,
    }


def utcp_manual(
    cli: ToolCli, definitions: Sequence[ToolDefinition], named: Sequence[str]
) -> dict[str, object]:
    """The UTCP manual for `definitions`: what `manual` prints.

    The names are the names of the tools. A client prefixes each with the name
    of the manual when it registers them (`ph_code_graph.code_index`). `named`
    is the variables that the configs of the mounted rows name (`EnvName`)."""
    inherit = [*_inherited(cli), *named]
    return {
        "utcp_version": UTCP_VERSION,
        "manual_version": cli.version,
        "tools": [
            {
                "name": one.name,
                "description": one.description,
                "inputs": one.parameters,
                "outputs": _result_schema(one),
                "tags": [cli.manual],
                "tool_call_template": _template(cli, one, inherit),
            }
            for one in definitions
        ],
    }


def manual_call_template(cli: ToolCli) -> dict[str, object]:
    """What a harness registers to discover the tools of this package."""
    return {
        "name": cli.manual,
        "call_template_type": "cli",
        "commands": [{"command": shlex.join((*cli.invocation(), "manual"))}],
        "inherit_env_vars": _inherited(cli),
    }


def _inherited(cli: ToolCli) -> list[str]:
    """What every command of the package inherits: the defaults and its own."""
    return [*INHERITED_ENV, *cli.inherit_env]


# ----------------------------------------------------------------- envelope ----


def failure_envelope(code: str, message: str) -> dict[str, object]:
    """A failure as `call` prints it. `read_envelope` raises it as a
    `HarnessError` with `code`."""
    return {"ok": False, "error": {"code": code, "message": message}}


def value_envelope(value: object, text: str | None = None) -> dict[str, object]:
    """A value as `call` prints it: with its rendering, where there is one."""
    return (
        {"ok": True, "value": value} if text is None else {"ok": True, "value": value, "text": text}
    )


def envelope(result: ToolExecutionResult) -> dict[str, object]:
    """The outcome of one call as the object that `call` prints.

    `value` is the declared output of the tool. `text` is the `render` of the
    tool, which is what a model in a session reads. A failure carries its
    routable code when the error had one, and its kind otherwise.
    `read_envelope` is the inverse."""
    text = text_of(result.content)
    if not result.is_error:
        return value_envelope(result.value, text)
    failure = result.error
    if failure is None:
        return failure_envelope("TOOL_FAILED", text)
    code = (failure.info or {}).get("code") or f"TOOL_{failure.kind.upper()}"
    return failure_envelope(code, failure.message)


def read_envelope(answer: object) -> tuple[JsonValue, str | None]:
    """The value and the rendering in an answer that `call` printed.

    A failure is raised as the `HarnessError` that the tool raised, with its own
    code. It therefore reaches a model in another harness as it would have
    reached one in this harness. An answer that is not an envelope is a
    `ValueError`: the caller knows which tool broke its promise, and says so."""
    if not isinstance(answer, Mapping) or not isinstance(answer.get("ok"), bool):
        raise ValueError("not an envelope")
    if not answer["ok"]:
        error = as_obj(answer.get("error"))
        raise HarnessError(
            as_str(error.get("message"), "the tool failed"),
            as_str(error.get("code"), "TOOL_FAILED"),
        )
    text = answer.get("text")
    return answer.get("value"), text if isinstance(text, str) else None


def value_schema(outputs: Mapping[str, object]) -> dict[str, object]:
    """The schema of the `value` of an envelope, from the `outputs` of a manual
    for it.

    This is the inverse of `_result_schema`. There, the definitions of the tool
    are lifted to the root of the envelope so that its references resolve. Here,
    they come back down to the root of the value for the same reason."""
    properties = outputs.get("properties")
    value = properties.get("value") if isinstance(properties, Mapping) else None
    if not isinstance(value, Mapping):
        return {}
    definitions = outputs.get("$defs")
    return {**value, "$defs": definitions} if isinstance(definitions, Mapping) else dict(value)


# -------------------------------------------------------------------- mount ----


class _Refused(Exception):
    """This command line cannot do what was asked. One sentence and a code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _document(path: Path, origin: str) -> JsonValue:
    try:
        return decode_document(path)
    except DOCUMENT_FAULTS as error:
        raise _Refused("BAD_CONFIG", f"{origin} will not read: {error}") from error


def _row_configs(cli: ToolCli, config_file: Path | None) -> dict[str, JsonObject]:
    """Row configs by row id: the package file, then the environment, then `--config`.

    A later layer wins, row by row. The file under `$PH_HOME/tools` is where a
    person keeps settings for the command that they type. The environment is
    how the row of a session passes its own settings across (`config_env`). The
    flag is for a one-time trial.
    """
    layers: list[tuple[str, JsonValue]] = []
    home = resolve_roots().home / "tools" / f"{cli.program}.yaml"
    if home.is_file():
        layers.append((str(home), _document(home, str(home))))
    raw = os.environ.get(CONFIG_ENV, "").strip()
    if raw:
        try:
            layers.append((f"${CONFIG_ENV}", loads(raw)))
        except JSONDecodeError as error:
            raise _Refused("BAD_CONFIG", f"${CONFIG_ENV} is not JSON: {error}") from error
    if config_file is not None:
        layers.append((str(config_file), _document(config_file, str(config_file))))

    known = {as_str(row["id"]) for row in (*cli.rows, *cli.providers)}
    merged: dict[str, JsonObject] = {}
    for origin, document in layers:
        if document is None:
            continue
        if not isinstance(document, Mapping):
            raise _Refused("BAD_CONFIG", f"{origin} must be a mapping of row id to config")
        for row_id, config in document.items():
            if row_id not in known:
                raise _Refused(
                    "BAD_CONFIG",
                    f'{origin} names row "{row_id}", which {cli.program} does not mount '
                    f"(it mounts {', '.join(sorted(known))})",
                )
            merged[row_id] = {**merged.get(row_id, {}), **as_obj(config)}
    return merged


@asynccontextmanager
async def _mounted(cli: ToolCli, root: Path, config_file: Path | None) -> AsyncIterator[Context]:
    """The rows of the package over `COMMAND_ROWS`, mounted at `root` and unwound on exit."""
    configs = _row_configs(cli, config_file)
    rows: list[JsonValue] = [dict(row) for row in COMMAND_ROWS]
    for row in (*cli.rows, *_provider(cli, configs)):
        config = configs.get(as_str(row["id"]))
        rows.append({**row, "config": {**as_obj(row.get("config")), **config}} if config else row)
    ctx = Context()
    try:
        await Profile.from_documents([ProfileDocument(cli.program, rows)]).mount(ctx, project=root)
        yield ctx
    finally:
        await ctx.drain()
        await ctx.dispose()


def _provider(cli: ToolCli, configs: Mapping[str, JsonObject]) -> tuple[JsonObject, ...]:
    """The one row of `cli.providers` that this run mounts: the row that a config
    names, or the first. Empty for a package that has none."""
    if not cli.providers:
        return ()
    named = [row for row in cli.providers if as_str(row["id"]) in configs]
    if len(named) > 1:
        raise _Refused(
            "BAD_CONFIG",
            f"{cli.program} mounts one of {', '.join(as_str(row['id']) for row in cli.providers)}"
            f", and the configs name {', '.join(as_str(row['id']) for row in named)}; "
            "configure one",
        )
    return (named[0] if named else cli.providers[0],)


def _named(ctx: Context) -> list[str]:
    """The variables that the configs of the mounted rows name (`env_names`). Each
    config is read as it runs, from its fork, so a private copy (`isolate`) is
    read too."""
    forks = ctx.require(MOUNT).forks.values()
    configs = (fork.spec.resolve_config(fork.config) for fork in forks)
    return env_names(*(config for config in configs if config is not None))


def _definitions(ctx: Context) -> list[ToolDefinition]:
    registry = ctx.require(TOOLS)
    found = (registry.get(name, scope=DEPLOYMENT) for name in registry.names(scope=DEPLOYMENT))
    return [one for one in found if one is not None]


# ----------------------------------------------------------------- commands ----


def _tool_arguments(
    cli: ToolCli, definition: ToolDefinition, args_json: str | None, rest: Sequence[str]
) -> JsonObject:
    """The arguments of the call: `--args-json` first, then one flag per field
    over it.

    A flag whose value is the `MISSING_ARG_` placeholder of `utcp_cli` was not
    given."""
    arguments: dict[str, JsonValue] = {}
    if args_json is not None:
        try:
            given = loads(args_json)
        except JSONDecodeError as error:
            raise _Refused("BAD_ARGUMENTS", f"--args-json is not JSON: {error}") from error
        if not isinstance(given, Mapping):
            raise _Refused("BAD_ARGUMENTS", "--args-json must be a JSON object")
        arguments.update(given)

    properties = _properties(definition)
    parser = _Parser(
        prog=f"{cli.program} call {definition.name}", add_help=False, allow_abbrev=False
    )
    for name, schema in properties.items():
        flags = [f"--{name}"]
        if "_" in name:
            flags.append(f"--{name.replace('_', '-')}")
        if _kinds(schema) - {"null"} == {"boolean"}:
            # A person types a bare `--forget`. A client sends `--forget True`.
            parser.add_argument(*flags, dest=name, nargs="?", const="true", default=None)
        else:
            parser.add_argument(*flags, dest=name, default=None)
    parsed = vars(parser.parse_args(list(rest)))
    for name, schema in properties.items():
        raw = parsed.get(name)
        if raw is not None and not raw.startswith(MISSING_ARGUMENT):
            arguments[name] = decode_argument(raw, schema)
    return arguments


async def _call(
    cli: ToolCli, ctx: Context, tool: str, args_json: str | None, rest: Sequence[str]
) -> dict[str, object]:
    registry = ctx.require(TOOLS)
    definition = registry.get(tool, scope=DEPLOYMENT)
    if definition is None:
        names = ", ".join(registry.names(scope=DEPLOYMENT)) or "none"
        return failure_envelope(
            "UNKNOWN_TOOL", f'{cli.program} has no tool "{tool}" (it has {names})'
        )
    result = await registry.execute(
        ToolExecutionInput(
            call_id=f"{cli.program}-{os.getpid()}",
            name=tool,
            arguments=_tool_arguments(cli, definition, args_json, rest),
            # No agent exists to narrow anything, so this is the view of the
            # deployment itself. The scope is stated on purpose, as the
            # `tools/list` of `--mode rpc` states it.
            scope=DEPLOYMENT,
        )
    )
    return envelope(result)


def _tools(cli: ToolCli, ctx: Context) -> str:
    lines: list[str] = []
    for one in _definitions(ctx):
        summary = one.description.strip().splitlines()[0] if one.description.strip() else ""
        lines.append(f"{one.name}  {summary}")
        lines.extend(
            f"    --{name}  {schema.get('description', '')}".rstrip()
            for name, schema in _properties(one).items()
        )
    return "\n".join(lines) if lines else f"{cli.program} registered no tools."


async def _provision(cli: ToolCli, ctx: Context, verb: str) -> str:
    """The `install_or_status` command of the package, and its doctor section.

    The command is called on the definition and not dispatched. `dispatch`
    records both halves of a command in the log of a session, and there is no
    session here to record in. This function is reached only for a
    `cli.command`, which a row of the package must register. When no row
    registers it, a declaration is missing, and the refusal says so."""
    definition = ctx.require(COMMANDS).get(cli.command)
    if definition is None:
        raise _Refused(
            "NO_COMMAND", f"{cli.program} names /{cli.command}, which none of its rows registers"
        )
    invocation = CommandContext(ctx=ctx, session=None, scope=ctx, line=f"/{cli.command}")
    answer = await maybe_await(definition.run(verb, invocation))
    lines = [answer] if answer else []
    if verb == "status":
        for title, rows in ctx.require(DIAGNOSTICS).report():
            lines.append(f"{title}:")
            lines.extend(f"  {label}: {value}" for label, value in rows)
    return "\n".join(lines) if lines else f"{cli.program} has nothing to {verb}."


# ---------------------------------------------------------------------- cli ----


class _Parser(argparse.ArgumentParser):
    """An `ArgumentParser` whose usage errors are refusals, not `sys.exit(2)`.

    argparse writes its complaint to stderr and exits. Under UTCP, stderr is
    discarded and the exit code is ignored, so a mistyped flag would come back
    as an empty result. The error is raised instead, so that `call` can report
    it in the envelope."""

    def error(self, message: str) -> NoReturn:
        raise _Refused("BAD_ARGUMENTS", f"{self.prog}: {message}")


def _parser(cli: ToolCli) -> argparse.ArgumentParser:
    # No abbreviations anywhere: a `--for…` field of a tool must not be read as
    # a prefix of one of the flags of this parser.
    parser = argparse.ArgumentParser(prog=cli.program, description=cli.summary, allow_abbrev=False)
    parser.add_argument("--version", action="version", version=f"{cli.program} {cli.version}")

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--root",
            type=Path,
            default=None,
            help="The workspace. Defaults to the working directory.",
        )
        sub.add_argument(
            "--config",
            type=Path,
            default=None,
            help="A YAML or JSON file of row configs by row id, over $PH_HOME/tools "
            f"and ${CONFIG_ENV}.",
        )
        sub.add_argument(
            "--state",
            type=Path,
            default=None,
            help="The directory the tools keep their working state in — an index, a "
            f"database. Over ${STATE_ENV}; without either, under $PH_CACHE, per workspace.",
        )
        sub.add_argument(
            "--verbose", action="store_true", help="Log at INFO, and to stderr as well."
        )

    commands = parser.add_subparsers(dest="verb", required=True, metavar="command")

    call = commands.add_parser(
        "call",
        allow_abbrev=False,
        help="Run one tool and print its result.",
        description="Run one tool. Its fields are flags (`tools` lists them), or pass "
        "the whole argument object with --args-json.",
    )
    call.add_argument("tool")
    call.add_argument("--args-json", default=None, help="The arguments, as one JSON object.")
    call.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="json: one object with ok/value/text (the default, and what a UTCP "
        "client reads). text: the result as prose.",
    )
    call.add_argument(
        "--progress",
        action="store_true",
        help="Report how far the run has got on stderr, as JSONL: `working` lines as "
        "it goes and one last line, `done`, `failed` or `busy`. For a caller that "
        "reads stderr apart from stdout; a UTCP client merges them.",
    )
    call.add_argument(
        "--wait",
        type=float,
        default=None,
        metavar="SECONDS",
        help="How long to wait for another run that holds what this tool writes, "
        "over the row's own setting. 0 answers at once that one is working.",
    )
    common(call)

    common(commands.add_parser("tools", help="List the tools and their flags."))

    manual = commands.add_parser("manual", help="Print this package's UTCP manual.")
    manual.add_argument(
        "--registration",
        action="store_true",
        help="Print the call template a harness registers to discover the manual, "
        "rather than the manual.",
    )
    common(manual)

    if cli.command:
        common(commands.add_parser("status", help="Report whether the package is ready."))
        common(commands.add_parser("install", help="Fetch what the package needs, now."))
    return parser


@contextmanager
def _logging_to_file(cli: ToolCli, *, verbose: bool) -> Iterator[None]:
    """Send logging to `cli.log_file()` for the body, and to nowhere else.

    A handler is always installed, and it is a `NullHandler` when the file
    cannot be made. With no handler, `logging` falls back to writing warnings on
    stderr, which a UTCP client reads as part of the result. The level is
    WARNING unless `--verbose`, because one process per call appends to one
    file. The file is rotated at `LOG_BYTES`, with one file kept back, because a
    deployment fact that deserves a warning is then a line per call. Processes
    that rotate at the same moment can each lose a line. That is acceptable for
    a diagnostic.

    The change is undone on exit. The process is about to exit, so nothing
    there depends on the undo. A caller that runs this in its own process, such
    as a test, does depend on it.
    """
    root = logging.getLogger()
    level = root.level
    handler: logging.Handler
    path = cli.log_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=LOG_BYTES, backupCount=1, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(process)d %(levelname)s %(name)s: %(message)s")
        )
    except OSError:
        handler = logging.NullHandler()
    added = [handler, *([logging.StreamHandler(sys.stderr)] if verbose else [])]
    for one in added:
        root.addHandler(one)
    root.setLevel(logging.INFO if verbose else logging.WARNING)
    logging.captureWarnings(True)
    try:
        yield
    finally:
        logging.captureWarnings(False)
        root.setLevel(level)
        for one in added:
            root.removeHandler(one)
            one.close()


@contextmanager
def _stderr_discarded() -> Iterator[None]:
    """Point the stderr of the process at nothing for the body. This covers
    `sys.stderr` and the descriptor under it, so a native library and a child
    are covered too.

    It is for a run whose stdout is data. A UTCP client merges the two streams,
    so one progress bar from a model library ("Loading weights: 100%") comes
    before the answer, and the answer no longer parses. What would have been
    written is lost on purpose. A failure is in the object that the run prints,
    and its traceback is in the log file. `--verbose` keeps the stream, for a
    person.

    The stream is put back on exit, for the same reason as in `_logging_to_file`.
    """
    sys.stderr.flush()
    with open(os.devnull, "w", encoding="utf-8") as nothing, redirect_stderr(nothing):
        # `sys.stderr` is redirected as well as the descriptor: a host that
        # replaced the object (the capture of a test, an embedding application)
        # writes somewhere else.
        try:
            saved = os.dup(2)
        except OSError:
            # There is no descriptor to redirect: a caller closed it, and nothing
            # can be written to it.
            yield
            return
        try:
            os.dup2(nothing.fileno(), 2)
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)


@contextmanager
def _state_at(directory: Path | None) -> Iterator[None]:
    """Set `--state` as `$PH_TOOL_STATE` for the body, so that a row reads the
    setting from one place (`state_dir`). The variable is put back on exit, for
    the same reason as in `_logging_to_file`: a caller that runs this in its own
    process."""
    if directory is None:
        yield
        return
    before = os.environ.get(STATE_ENV)
    os.environ[STATE_ENV] = str(directory.expanduser().resolve())
    try:
        yield
    finally:
        if before is None:
            del os.environ[STATE_ENV]
        else:
            os.environ[STATE_ENV] = before


@contextmanager
def _set[T](variable: ContextVar[T], value: T) -> Iterator[None]:
    """`variable` as `value` for the body."""
    token = variable.set(value)
    try:
        yield
    finally:
        variable.reset(token)


@contextmanager
def reporting_to(out: TextIO) -> Iterator[None]:
    """Send the progress of the runs in the body to `out`, as JSONL. `call
    --progress` does this with its stderr. This function is for a caller that
    runs a tool in its own process and wants the report that `reported` writes."""
    with _set(_REPORTS, _Reports(out)):
        yield


@contextmanager
def _reporting() -> Iterator[None]:
    """Keep the stderr of the process for progress lines, and for nothing else.

    The function takes a copy of the descriptor before `_stderr_discarded`
    points the original at nothing. What a dependency writes is still lost, and
    what `reported` writes reaches whoever started the command. A process with
    no stderr to copy reports to nobody."""
    try:
        kept = os.dup(2)
    except OSError:
        yield
        return
    with os.fdopen(kept, "w", encoding="utf-8") as out, reporting_to(out):
        yield


def run(cli: ToolCli, argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """Run the command line one time. Returns the exit code.

    The exit code is for a person and for a script: 0, or 1 when the answer is a
    failure. A UTCP client never sees it, so `call` says the same thing in what
    it prints.
    """
    out = stdout if stdout is not None else sys.stdout
    parser = _parser(cli)
    known, rest = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    if rest and known.verb != "call":
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    as_json = known.verb == "manual" or (known.verb == "call" and known.format == "json")
    with ExitStack() as stack:
        stack.enter_context(_logging_to_file(cli, verbose=known.verbose))
        stack.enter_context(_state_at(known.state))
        if known.verb == "call":
            stack.enter_context(_set(_WAIT, known.wait))
            if known.progress:
                # This runs before stderr is pointed at nothing: this is the
                # stream that is kept.
                stack.enter_context(_reporting())
        if as_json and not known.verbose:
            stack.enter_context(_stderr_discarded())
        return _answered(cli, known, rest, out, as_json=as_json)


def _answered(
    cli: ToolCli, known: argparse.Namespace, rest: Sequence[str], out: TextIO, *, as_json: bool
) -> int:
    verb: str = known.verb
    if verb == "manual" and known.registration:
        print(dumps(manual_call_template(cli)), file=out)
        return 0
    root = (known.root or Path.cwd()).resolve()

    async def answered() -> dict[str, object] | str:
        """One mount for whichever verb was asked."""
        async with _mounted(cli, root, known.config) as ctx:
            if verb == "call":
                return await _call(cli, ctx, known.tool, known.args_json, rest)
            if verb == "manual":
                return utcp_manual(cli, _definitions(ctx), _named(ctx))
            if verb == "tools":
                return _tools(cli, ctx)
            return await _provision(cli, ctx, verb)

    answer: dict[str, object] | str
    try:
        answer = anyio.run(answered)
    except (_Refused, HarnessError) as error:
        # A refusal by this command line, or a provisioning command that could
        # not do what was asked. The answer is a sentence and a code, and the
        # exit code is a failure.
        answer = failure_envelope(error.code, str(error))
    except CordisError as error:
        # A row that declined, or a profile that would not compose. This is a
        # deployment fact with a sentence, the same sentence that `phern` prints
        # for it.
        log.warning("%s could not mount", cli.program, exc_info=True)
        answer = failure_envelope("MOUNT_REFUSED", str(error))
    except Exception as error:
        # A bug, but one that a UTCP client would otherwise read as an empty
        # result. The traceback is in the log file, and the sentence is in the
        # answer.
        log.exception("%s failed", cli.program)
        answer = failure_envelope("INTERNAL_ERROR", f"{type(error).__name__}: {error}")

    if isinstance(answer, str):
        print(answer, file=out)
        return 0
    reports = _REPORTS.get()
    if reports is not None:
        # The last line, where the work wrote none: a tool that reports nothing,
        # or a failure. A failure is reported as the envelope reports it.
        failed = None if answer["ok"] else as_obj(answer["error"])
        reports.end(
            code=None if failed is None else as_str(failed.get("code")),
            failure=None if failed is None else as_str(failed.get("message")),
        )
    if as_json:
        print(dumps(answer), file=out)
    elif answer["ok"]:
        print(answer["text"], file=out)
    else:
        print(f"{cli.program}: {as_obj(answer['error'])['message']}", file=sys.stderr)
    return 1 if answer.get("ok") is False else 0


def main(cli: ToolCli) -> NoReturn:
    """The console script's body."""
    sys.exit(run(cli))
