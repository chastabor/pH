"""`ph.toolcli` — a package's tools as a command line, and as a UTCP manual.

A package that registers tools through rows gets a console script by naming those
rows: `ToolCli(program=..., rows=...)` and `main`. The script mounts a small
profile — the registry, the filesystem, a subprocess door, and the package's own
rows — and runs a tool **through the registry**, the way `ph.testing.run_tool`
does. So there is one body per tool: the one the row registers. What differs
between a session and this command is which process mounted it.

## Progress, for a run that takes a while

One command is one call: there is no process that stays up. A tool whose work is
long — an indexing run — says how far it has got on **stderr**, as JSONL, when
it is called with `--progress` (`Progress`, `reported`): a line as it begins, a
line at most every `REPORT_EVERY_S`, and exactly one last line, `done` or
`failed`. Whoever started the command reads that stream for a footer, and knows
the run stopped when the stream ends: with a last line, or without one. stdout
is unchanged — one envelope. Not for a UTCP client, which merges the two
streams: the flag is for a caller that reads them apart, as `ph-utcp` does for a
run it starts itself.

Three audiences, one program:

* a **person** trying a query by hand (`call code_graph --mode define --query x
  --format text`), or asking whether the package is provisioned (`status`);
* a **UTCP client**, which runs `manual` once to learn the tools and then runs the
  command each tool's template names;
* `ph-utcp`, which is the second of those inside a pH session.

## What the UTCP command-line protocol decides here

Read from `utcp_cli` 1.1.5; every item below is a constraint, not a taste.

* **Arguments arrive one at a time, as `str(value)`.** A list arrives as Python's
  repr, a boolean as `True`, and an optional argument the model left out arrives
  as the literal `MISSING_ARG_<name>`. So there is one flag per top-level field,
  and `decode_argument` reads each by the tool's own schema: JSON first, then the
  Python spellings, then a comma split for an array. The manual declares the true
  types and this accepts what a client actually sends.
* **stderr is merged into stdout, and the exit code is ignored.** So a call
  prints exactly one JSON object, a failure is reported inside it, logging goes
  to a file — and for the length of a `call` or a `manual` the process's own
  stderr is closed to everything in it (`_stderr_discarded`), because a
  dependency that writes a warning or a progress bar is not ours to silence any
  other way. Not with a `2>/dev/null` on the template: the client wraps each
  command as `$(<command> 2>&1)`, and the later redirection wins.
* **The environment is cut to an allowlist.** Naming any variable replaces the
  client's default list, so the template names the defaults again beside the pH
  roots — without them a UTCP call would build a different index from the one a
  session uses for the same checkout. A variable only a row's config names is
  listed too (`EnvName`).

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
"""The UTCP version the generated manual is written against — the one read."""

CONFIG_ENV = "PH_TOOL_CONFIG"
"""A JSON object of row configs by row id, for a caller that mounts nothing itself.

How a session's row hands its config across the process boundary: `ph-utcp` sets
this on the command it runs, so the tool indexes with the settings the profile
names rather than with the package's defaults. `config_env` writes it and
`_row_configs` reads it."""

STATE_ENV = "PH_TOOL_STATE"
"""The directory a tool keeps its working state in, when its caller says where.

An index, a database, a set of vectors: what a tool builds *about* a workspace
and does not put in it. `ph-utcp` sets this on each call to a directory of the
calling agent's scratch, so the state is the agent's own — writable in a
read-only session, where the workspace is not, and found again by the same agent
after a restart. `--state` is the same thing typed by a person, and wins. With
neither, a tool keeps its state where it always has: under `$PH_CACHE`, keyed by
the workspace. `state_dir` is how a row reads it."""

MISSING_ARGUMENT = "MISSING_ARG_"
"""What `utcp_cli` substitutes for a placeholder whose argument was not given."""

COMMAND_ROWS: tuple[JsonObject, ...] = (
    {"id": "tools", "name": "tools"},
    # Not a pH start, so it sweeps nothing a hard-killed run left behind. And it
    # could not judge one: run for an agent it is in the sandbox's own PID
    # namespace, where every process the journal names looks dead.
    {"id": "subprocess", "name": "subprocess-local", "config": {"sweepOrphans": False}},
    {"id": "fs", "name": "fs-local"},
    {"id": "commands", "name": "commands"},
    {"id": "diagnostics", "name": "diagnostics"},
)
"""What a tool needs around it and no more: no model, no session, no agent.

`subprocess` is here for `ph.seams.changes`, which asks git or jj what changed;
`commands` and `diagnostics` so a package's provisioning command and its doctor
section exist for `status` and `install` to read."""

INHERITED_ENV: tuple[str, ...] = (
    # `utcp_cli`'s own default list. Naming a variable replaces that list rather
    # than extending it, so these are repeated or the command loses `PATH`.
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
    # Where pH keeps things. Dropped, a call resolves different roots from the
    # session beside it and the two build separate indexes of one checkout.
    "PH_HOME",
    "PH_CACHE",
    "PH_RUNTIME",
    "XDG_CACHE_HOME",
    "XDG_RUNTIME_DIR",
    CONFIG_ENV,
    # So another harness can say where a tool's state goes, by setting it.
    STATE_ENV,
)

LOG_BYTES = 1024 * 1024
"""How large a command's log grows before it is rotated."""

_PLACEHOLDER_NAME = re.compile(r"[A-Za-z0-9_]+")
"""The names `utcp_cli`'s placeholder pattern can carry."""


@dataclass(frozen=True, slots=True)
class ToolCli:
    """One package's command line: its program, its rows, and what it provisions."""

    program: str
    """The console script's name, as it is typed: `ph-code-graph`."""
    manual: str
    """The UTCP manual's name. Letters, digits and underscores: `ph_code_graph`."""
    summary: str
    version: str
    rows: tuple[JsonObject, ...]
    """The package's rows, as a profile would spell them."""
    providers: tuple[JsonObject, ...] = ()
    """Rows of which exactly one is mounted, after `rows`: the one the row configs
    name, or the first when they name none.

    For a package whose tools need something only one of several rows supplies —
    an embedder that is a local model in one deployment and an endpoint in
    another. Which one is a deployment's choice, and it makes it the way it makes
    every other: by configuring that row, in the file under `$PH_HOME/tools`, in
    `$PH_TOOL_CONFIG` (`config_env` names a row even when its config is all
    defaults, for this) or with `--config`. Naming two is refused."""
    command: str = ""
    """The package's provisioning command (`code-graph`), or empty for none.

    The same `install_or_status` command a session offers as `/code-graph`. It
    stays out of the manual for the reason the packages give for making it a
    command: provisioning is something a person asks for."""
    inherit_env: tuple[str, ...] = ()
    """Variables this package reads, beyond `INHERITED_ENV`."""
    module: str = ""
    """The package's `__main__`, run as `python -P -m <module>` by the interpreter
    that printed the manual — so its commands run there, whatever `PATH` the
    calling harness has. Empty spells the program by its script name."""

    def invocation(self) -> tuple[str, ...]:
        """How a generated template spells the program.

        `-P`, because `-m` otherwise puts the working directory first on
        `sys.path`, and the working directory is the tree the tool is about: a
        `shlex.py` at a repository's root would be imported in place of the
        standard library's, in a command that may hold a credential and — for
        discovery and a person's `status` — runs outside any sandbox."""
        return (sys.executable, "-P", "-m", self.module) if self.module else (self.program,)

    def log_file(self) -> Path:
        """Where the command logs. A call prints one object and nothing on stderr,
        so this is where a traceback goes — and a path a confined command has to
        be granted."""
        return resolve_roots().cache / "logs" / f"{self.program}.log"


def config_env(configs: Mapping[str, BaseModel]) -> tuple[dict[str, str], list[str]]:
    """What a session hands a command for these rows' configs: the
    `$PH_TOOL_CONFIG` that carries them, by row id, and the variables they mark
    (`env_names`) to pass across by name — a `Manual`'s `env` and `pass_env`.

    Each row is named, with only what differs from its model's defaults — so a
    default row travels as `{}`. Named even then, because being named is how a
    command that mounts one of several provider rows learns which
    (`ToolCli.providers`)."""
    settings = {
        row: config.model_dump(by_alias=True, exclude_defaults=True)
        for row, config in configs.items()
    }
    env = {CONFIG_ENV: dumps(settings)} if settings else {}
    return env, env_names(*configs.values())


def state_dir() -> Path | None:
    """Where the caller said this process's tools keep their state (`STATE_ENV`),
    or `None` for the tool's own default.

    Read from the environment at each use rather than held, so a row asks it
    where it builds a path and a test that sets the variable is read. Absolute:
    a relative directory is taken from the working directory, once, here."""
    raw = os.environ.get(STATE_ENV, "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def state_home(base: Path, workspace: Path) -> Path:
    """Where a tool keeps what it builds about `workspace`: the directory its
    caller named (`state_dir`), or `workspace`'s own under `base` — keyed by a
    digest of the root, so two checkouts do not answer each other's questions."""
    state = state_dir()
    return state if state is not None else base / digest(str(workspace))


def claim_home(base: Path, workspace: Path) -> None:
    """Record that `workspace`'s directory under `base` is in use, and sweep the
    ones whose workspace has been gone a while (`ph.workspace_caches.use`).

    Nothing for a directory the caller named: that is its owner's to keep or
    remove. Blocking — file I/O — so called through a worker thread."""
    if state_dir() is None:
        use(base, workspace)


# ----------------------------------------------------------------- progress ----


class Progress(WireModel):
    """One line of a run's report: how far a long piece of work has got, or how
    it ended. What `call --progress` writes to stderr, a JSON object a line.

    `working` lines come as the run goes. The last line is `done`, `failed`, or
    `busy` — another run held what this one writes, and it did nothing — and
    there is exactly one: a stream that ends without it is a command that was
    stopped from outside, or died."""

    state: Literal["working", "done", "failed", "busy"]
    done: int = 0
    total: int = 0
    """Units of work — documents, files — finished, and in all."""
    what: str = ""
    """What the units are, plural: `documents`."""
    code: str | None = None
    """On `failed`: the failure's routable code, the envelope's own."""
    failure: str | None = None
    """On `failed`: why, in a sentence."""
    changed: bool | None = None
    """On `done`: whether the run changed what it builds. A run that found
    everything current says `false`, and nobody needs telling it ended."""
    note: str | None = None
    """On `done`: what the run has to say beyond its count, in sentences — what a
    caller that waited for it read in the tool's result."""
    spent: dict[str, int] | None = None
    """What the run has spent so far, by unit, as the tool counts it — an
    embedder's `requests`, `texts`, and the `tokens` its provider reported. On
    every line, the last included, so whoever reads the stream can track what a
    run cost, a run that failed or was killed as well. Absent for a tool that
    counts nothing."""

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
    """Where this command's progress lines go: the stream `--progress` kept."""

    out: TextIO
    ended: bool = False
    """A last line has been written, and nothing more is."""
    current: RunProgress | None = None
    """The run reporting here (`reported`), whose counts and spend a last line
    written from outside the work carries — as they are, and not as the last
    line written, which is at most every `REPORT_EVERY_S` behind."""

    def write(self, progress: Progress) -> None:
        """One line, whole and flushed. Nothing after the last line; and never
        raises — a reader that has gone must not stop the work it was reading."""
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
        with the envelope's code and sentence and how far the run had got."""
        state: Literal["done", "failed"] = "done" if failure is None else "failed"
        run = self.current
        if run is None:
            self.write(Progress(state=state, code=code, failure=failure))
        else:
            self.write(run.line(state, code=code, failure=failure))


_REPORTS: ContextVar[_Reports | None] = ContextVar("ph_toolcli_reports", default=None)
"""Set by `run` for a `call --progress`; `None` for every other command, whose
work then reports to nobody."""

_WAIT: ContextVar[float | None] = ContextVar("ph_toolcli_wait", default=None)
"""`call --wait`: how long this call waits for another run, over the row's own."""

REPORT_EVERY_S = 0.5
"""How often a run's progress is written, at most."""


def env_names(*configs: BaseModel) -> list[str]:
    """The variables `configs` name: each set `EnvName` field's value, a nested
    model's included, once each, in order. `ToolCli.inherit_env` is what a
    package always reads; these are what one deployment's settings add."""

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
    """One run's account of itself: how much there is, and how far it has got.

    Handed to the work by `reported`, which writes the run's end; the work says
    only what it alone knows — `begin` once it knows how much there is, under
    whatever lock makes the run its own, and `tick` as it goes."""

    what: str
    """What the units are, plural: `documents`."""
    reports: _Reports | None = None
    """Where the lines go: `call --progress`'s stream, or nobody."""
    spent: Callable[[], Mapping[str, int]] | None = None
    """How to ask what the run has spent so far (`Progress.spent`), set once by
    work that counts something: asked as each line is written, so a line —
    a failed run's last included — says it as it is."""
    total: int = 0
    done: int = 0
    _written: float = 0.0

    def begin(self, total: int) -> None:
        """There are `total` units to go through, and none done yet."""
        self.total, self.done = total, 0
        self._write()

    def tick(self, done: int) -> None:
        """`done` units are through: written at most every `REPORT_EVERY_S`."""
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
    """`work`'s value, with the run reported as it goes and its end written —
    by this door, so no row can forget the last line.

    `said` is what the value says beyond its count, in a few lines: what the
    tool's result tells a caller that waited, for whoever is told of a run
    nobody waited for. Required: a run whose note says only "finished" can hide
    one that skipped everything. `changed` is whether the value changed what the
    run builds, so a run over a tree that had not moved is not news. `busy` is
    whether the value is the answer that another run held what this one writes
    (`run_lock`): its last line says that, and not that it finished.

    A failure's last line is `run`'s to write, from the envelope the failure
    becomes — the code and the sentence a caller reads. What is done here is the
    traceback: one the tool did not raise as its own failure goes to the
    command's log, where a run nobody waited for would otherwise leave none.
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
    """Hold `lock` for one run against every other, waited for at most `wait_s` —
    or `call --wait`, where the caller said. Yields whether it is held.

    A second run is expected to queue behind the first and then find most of
    its work done. But a caller is a call with a limit of its own, so the wait
    is bounded: past it the body runs with `False`, and answers that another
    run is working instead of standing behind it. The directory needs no making
    here — taking the lock makes it."""
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
    """A Python literal as JSON. `ValueError` for one with no JSON spelling — a
    set, bytes, a key that is not a string."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(one) for one in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _plain(one) for key, one in value.items()}
    raise ValueError(f"{type(value).__name__} has no JSON spelling")


def _structured(raw: str) -> list[JsonValue] | dict[str, JsonValue] | None:
    """`raw` as a list or an object, read as JSON or as a Python literal — the two
    spellings a client's `str()` produces for one. `None` when it is neither."""
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
    """One flag's text as the value its schema asks for.

    **Lenient on the way in, because the registry validates on the way out.** A
    text this cannot place is handed on as the string it is, and the tool's own
    model refuses it with a message naming the field — so nothing here raises on
    a client's phrasing.

    A string field takes its text as written, with one exception: where the
    schema admits `null`, `None` and `null` mean it. That costs a search for a
    symbol literally named `None` under a nullable field, and buys an explicit
    null from the Python client, whose `str(None)` is otherwise a query.

    An argument the client left out is not this function's to see: the caller
    drops a `MISSING_ARG_` placeholder before decoding, because no value — `None`
    included — is the same as no argument.
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
            # path a person typed. Split on commas and newlines.
            return [one.strip() for one in re.split(r"[,\n]", text) if one.strip()]
    return raw


# ------------------------------------------------------------------- manual ----


def _properties(definition: ToolDefinition) -> dict[str, Mapping[str, object]]:
    properties = definition.parameters.get("properties")
    if not isinstance(properties, Mapping):
        return {}
    return {str(name): schema for name, schema in properties.items() if isinstance(schema, Mapping)}


def _result_schema(definition: ToolDefinition) -> dict[str, object]:
    """The envelope `call` prints, with the tool's own output as its `value`.

    `$defs` is lifted to the envelope's root: pydantic writes its references as
    `#/$defs/Name`, which resolve from the root of the document they are in, so a
    definition left inside `value` would be a pointer to nothing. `value_schema`
    is the inverse."""
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
        # `--name=value`, one word: a value that starts with `-` — a query for
        # `-x`, a negative number — is then an argument and not a flag.
        words.append(f"--{name}=UTCP_ARG_{name}_UTCP_END")
    # Not `shlex.join` for the whole line: a placeholder has to stay a bare word
    # for the client to substitute it.
    return {
        "call_template_type": "cli",
        "commands": [{"command": " ".join(words)}],
        "inherit_env_vars": inherit,
    }


def utcp_manual(
    cli: ToolCli, definitions: Sequence[ToolDefinition], named: Sequence[str]
) -> dict[str, object]:
    """The UTCP manual for `definitions`: what `manual` prints.

    Names are the tools' own. A client prefixes each with the manual's name when
    it registers them (`ph_code_graph.code_index`). `named` is the variables the
    mounted rows' configs name (`EnvName`)."""
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
    """What a harness registers to discover this package's tools."""
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
    """One call's outcome as the object `call` prints.

    `value` is the tool's declared output and `text` is its own `render` — what a
    model in a session reads. A failure carries its routable code when the error
    had one, and its kind otherwise. `read_envelope` is the inverse."""
    text = text_of(result.content)
    if not result.is_error:
        return value_envelope(result.value, text)
    failure = result.error
    if failure is None:
        return failure_envelope("TOOL_FAILED", text)
    code = (failure.info or {}).get("code") or f"TOOL_{failure.kind.upper()}"
    return failure_envelope(code, failure.message)


def read_envelope(answer: object) -> tuple[JsonValue, str | None]:
    """The value and the rendering in an answer `call` printed.

    A failure is raised as the `HarnessError` the tool raised, with its own code,
    so it reaches a model in another harness as it would have in this one. An
    answer that is not an envelope is a `ValueError`: the caller knows which
    tool broke its promise and says so."""
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
    """The schema of an envelope's `value`, from a manual's `outputs` for it.

    `_result_schema`'s inverse: there the tool's definitions are lifted to the
    envelope's root so its references resolve, and here they come back down to
    the value's root for the same reason."""
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

    Later wins, row by row. The file under `$PH_HOME/tools` is where a person
    keeps settings for the command they type; the environment is how a session's
    row hands its own across (`config_env`); the flag is for trying something once.
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
    """The package's rows over `COMMAND_ROWS`, mounted at `root` and unwound on exit."""
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
    """The one of `cli.providers` this run mounts: the one a config names, or the
    first. None for a package that has none."""
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
    """The variables the mounted rows' configs name (`env_names`): each as it
    runs, read off its fork, so a private copy (`isolate`) is read too."""
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
    """The call's arguments: `--args-json` first, then one flag per field over it.

    A flag whose value is `utcp_cli`'s `MISSING_ARG_` placeholder was not given."""
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
            # A bare `--forget` for a person; `--forget True` from a client.
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
            # No agent exists to narrow anything, so this is the deployment's own
            # view — said on purpose, as `--mode rpc`'s `tools/list` says it.
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
    """The package's own `install_or_status` command, and its doctor section.

    Called on the definition rather than dispatched: `dispatch` records both halves
    of a command in a session's log, and there is no session here to record in.
    Reached only for a `cli.command`, which a row of the package must register —
    one that does not is a declaration missing, and said."""
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

    argparse writes its complaint to stderr and exits, and under UTCP stderr is
    discarded and the exit code ignored — so a mistyped flag would come back as an
    empty result. Raised instead, so `call` can say it in the envelope."""

    def error(self, message: str) -> NoReturn:
        raise _Refused("BAD_ARGUMENTS", f"{self.prog}: {message}")


def _parser(cli: ToolCli) -> argparse.ArgumentParser:
    # No abbreviations anywhere: a tool's own `--for…` field must not be read as
    # a prefix of one of this parser's flags.
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
        "it goes and one last line, `done` or `failed`. For a caller that reads "
        "stderr apart from stdout; a UTCP client merges them.",
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

    A handler is always installed, a `NullHandler` when the file cannot be made:
    with none, `logging` falls back to writing warnings on stderr, which a UTCP
    client reads as part of the result. At WARNING unless `--verbose`, because one
    process per call appends to one file; and rotated at `LOG_BYTES`, keeping one
    file back, because a deployment fact worth a warning is then a line per call.
    Processes that rotate at the same moment can lose a line each; it is a
    diagnostic.

    Undone on the way out. The process is about to exit, so nothing depends on
    it there; a caller that runs this in its own process — a test — does.
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
    """Point the process's stderr at nothing for the body: `sys.stderr`, and the
    descriptor under it, so a native library and a child are covered too.

    For a run whose stdout is data. A UTCP client merges the two streams, so one
    progress bar from a model library — "Loading weights: 100%" — is in front of
    the answer and the answer no longer parses. What would have been written is
    lost on purpose; a failure is in the object the run prints, and its traceback
    is in the log file. `--verbose` keeps the stream, for a person.

    Put back on the way out, for `_logging_to_file`'s reason.
    """
    sys.stderr.flush()
    with open(os.devnull, "w", encoding="utf-8") as nothing, redirect_stderr(nothing):
        # `sys.stderr` as well as the descriptor: a host that replaced the object
        # — a test's capture, an embedding application — writes somewhere else.
        try:
            saved = os.dup(2)
        except OSError:
            # No descriptor to redirect: a caller closed it, and nothing can be
            # written to it.
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
    """`--state` as `$PH_TOOL_STATE` for the body, so there is one place a row
    reads it from (`state_dir`). Put back on the way out, for `_logging_to_file`'s
    reason: a caller that runs this in its own process."""
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
    """Send the progress of the body's runs to `out`, as JSONL: what
    `call --progress` does with its stderr, for a caller that runs a tool in its
    own process and wants the report `reported` writes."""
    with _set(_REPORTS, _Reports(out)):
        yield


@contextmanager
def _reporting() -> Iterator[None]:
    """Keep the process's stderr for progress lines, and for nothing else.

    A copy of the descriptor, taken before `_stderr_discarded` points the
    original at nothing: what a dependency writes is still lost, and what
    `reported` writes reaches whoever started the command. A process with no
    stderr to copy reports to nobody."""
    try:
        kept = os.dup(2)
    except OSError:
        yield
        return
    with os.fdopen(kept, "w", encoding="utf-8") as out, reporting_to(out):
        yield


def run(cli: ToolCli, argv: Sequence[str] | None = None, *, stdout: TextIO | None = None) -> int:
    """Run the command line once. Returns the exit code.

    The exit code is for a person and a script: 0, or 1 when the answer is a
    failure. A UTCP client never sees it, which is why `call` says the same thing
    in what it prints.
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
                # Before stderr is pointed at nothing: this is the stream kept.
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
        # A refusal of this command line's, or a provisioning command that could
        # not do what was asked: a sentence and a code, and a failing exit.
        answer = failure_envelope(error.code, str(error))
    except CordisError as error:
        # A row that declined, or a profile that would not compose: a deployment
        # fact with a sentence, the same one `phern` prints for it.
        log.warning("%s could not mount", cli.program, exc_info=True)
        answer = failure_envelope("MOUNT_REFUSED", str(error))
    except Exception as error:
        # A bug, but one a UTCP client would otherwise read as an empty result.
        # The traceback is in the log file; the sentence is in the answer.
        log.exception("%s failed", cli.program)
        answer = failure_envelope("INTERNAL_ERROR", f"{type(error).__name__}: {error}")

    if isinstance(answer, str):
        print(answer, file=out)
        return 0
    reports = _REPORTS.get()
    if reports is not None:
        # The last line, where the work wrote none: a tool that reports nothing,
        # or a failure — said as the envelope says it.
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
