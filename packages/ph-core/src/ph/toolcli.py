"""`ph.toolcli` — a package's tools as a command line, and as a UTCP manual.

A package that registers tools through rows gets a console script by naming those
rows: `ToolCli(program=..., rows=...)` and `main`. The script mounts a small
profile — the registry, the filesystem, a subprocess door, and the package's own
rows — and runs a tool **through the registry**, the way `ph.testing.run_tool`
does. So there is one body per tool: the one the row registers. What differs
between a session and this command is which process mounted it.

## A host, for a tool that is slow to start

`serve` mounts the same rows once and answers calls on a unix socket, so what a
row loaded — an embedding model — is loaded once. `call` uses the host named in
`$PH_TOOL_HOST` when one is there and runs by itself when not; it never starts
one. A row can tell which it is in (`TOOL_HOST`): in a host a long piece of work
may outlive the call that asked for it, so an index tool there answers at once,
works in the background and reports to a `progress.json` in its state directory
(`Progress`), which whoever started the host watches.

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
  listed too (`ToolEnv`).

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
from dataclasses import dataclass, field
from functools import partial
from json import JSONDecodeError
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal, NoReturn, TextIO

import anyio
from anyio.abc import SocketStream, TaskGroup
from pydantic import BaseModel

from .cordis import (
    DEPLOYMENT,
    Context,
    CordisError,
    Disposer,
    Profile,
    ServiceKey,
    maybe_await,
)
from .cordis.loader import ProfileDocument
from .documents import DOCUMENT_FAULTS, decode_document
from .framing import FramingError, answer_one, ask, listening
from .json import JsonObject, JsonValue, as_obj, as_seq, as_str, dumps, loads
from .keys import COMMANDS, DIAGNOSTICS, TOOLS
from .llm.types import text_of
from .locks import try_file_lock
from .paths import resolve_roots, write_atomic
from .resources import run_until_signaled
from .seams._registry import claim_entry
from .seams.commands import CommandContext
from .tools.definition import ToolDefinition, ToolExecutionInput, ToolExecutionResult
from .tools.errors import HarnessError
from .tools.json_schema import schema_of
from .wire import WireModel
from .workspace_caches import digest, use

__all__ = [
    "AGENT_ENV",
    "CONFIG_ENV",
    "HOST_ENV",
    "HOST_ROWS",
    "INHERITED_ENV",
    "PROGRESS_FILE",
    "REPORT_EVERY_S",
    "STATE_ENV",
    "TOOL_ENV",
    "TOOL_HOST",
    "UTCP_VERSION",
    "BackgroundRuns",
    "Progress",
    "RunProgress",
    "ToolCli",
    "ToolEnv",
    "ToolHost",
    "claim_home",
    "config_env",
    "decode_argument",
    "envelope",
    "failure_envelope",
    "main",
    "manual_call_template",
    "read_envelope",
    "read_progress",
    "run",
    "state_dir",
    "state_home",
    "utcp_manual",
    "value_envelope",
    "value_schema",
    "write_progress",
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

HOST_ENV = "PH_TOOL_HOST"
"""The unix socket of a running `serve`, for `call` to use instead of mounting.

Set by whoever started the host — `ph-utcp`, per call, for the agent the host
serves. A `call` that finds nothing listening there runs by itself, so the
variable is a hint and never a requirement."""

AGENT_ENV = "PH_TOOL_AGENT"
"""The id of the agent whose tree a command works on, for a tool that asks the
session it runs for to do something on that agent's account.

Set by `ph-utcp` beside `STATE_ENV`, and naming the same agent: the one the state
belongs to, which for a child that shares its parent's index is the parent. A
word from inside a sandbox, so what a session does with it is bookkeeping and
never a grant."""

PROGRESS_FILE = "progress.json"
"""In a tool's state directory: how far a background piece of work has got."""

MISSING_ARGUMENT = "MISSING_ARG_"
"""What `utcp_cli` substitutes for a placeholder whose argument was not given."""

HOST_ROWS: tuple[JsonObject, ...] = (
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
    # So another harness can say where a tool's state goes, and which host
    # answers for it, by setting them.
    STATE_ENV,
    HOST_ENV,
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
    """The package's `__main__`, run as `python -m <module>` by the interpreter that
    printed the manual — so its commands run there, whatever `PATH` the calling
    harness has. Empty spells the program by its script name."""

    def invocation(self) -> tuple[str, ...]:
        """How a generated template spells the program."""
        return (sys.executable, "-m", self.module) if self.module else (self.program,)

    def log_file(self) -> Path:
        """Where the command logs. A call prints one object and nothing on stderr,
        so this is where a traceback goes — and a path a confined command has to
        be granted."""
        return resolve_roots().cache / "logs" / f"{self.program}.log"


def config_env(configs: Mapping[str, BaseModel]) -> dict[str, str]:
    """The `$PH_TOOL_CONFIG` that hands these rows' configs, by row id, to a command.

    Each row is named, with only what differs from its model's defaults — so a
    default row travels as `{}`. Named even then, because being named is how a
    command that mounts one of several provider rows learns which
    (`ToolCli.providers`)."""
    settings = {
        row: config.model_dump(by_alias=True, exclude_defaults=True)
        for row, config in configs.items()
    }
    return {CONFIG_ENV: dumps(settings)} if settings else {}


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


# ----------------------------------------------------------- host, progress ----


class Progress(WireModel):
    """How far a background piece of work has got: what a host's row writes to
    `PROGRESS_FILE` and whoever started the host reads."""

    state: Literal["working", "done", "failed"]
    done: int = 0
    total: int = 0
    """Units of work — documents, files — finished, and in all."""
    what: str = ""
    """What the units are, plural: `documents`."""
    failure: str = ""
    """Why it failed, when it did."""
    started: float = 0.0
    """When the run began, in seconds since the epoch: which run this is."""
    owed: bool = False
    """Whether the run's end is news to somebody: a call was answered "still
    going", or nobody asked for the run at all — a host started it unasked. A run
    whose own caller got its counts back owes nothing, and whoever watches this
    file says nothing more about it."""
    note: str = ""
    """What a finished run has to say beyond its count, in sentences: what a
    caller that waited for it read in the tool's result. Whoever is owed word of
    the end is told this with it."""

    def said(self) -> str:
        """One compact phrase, for a footer and for a sentence."""
        if self.state == "failed":
            return f"failed: {self.failure}"
        counted = f"{self.done}/{self.total}" if self.state == "working" else str(self.total)
        return f"{counted} {self.what}".strip()


def write_progress(directory: Path, progress: Progress) -> None:
    """Record `progress` in `directory`, whole or not at all. Never raises: a
    report that cannot be written must not stop the work it reports."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        write_atomic(directory / PROGRESS_FILE, progress.model_dump_json(), durable=False)
    except OSError:
        log.warning("ph.toolcli: progress was not written to %s", directory, exc_info=True)


def read_progress(directory: Path) -> Progress | None:
    """What is recorded in `directory`, or `None` for nothing readable."""
    try:
        return Progress.model_validate_json((directory / PROGRESS_FILE).read_bytes())
    except (OSError, ValueError):
        return None


@dataclass(slots=True)
class ToolHost:
    """What `serve` gives the rows it mounts: a process that outlives one call.

    A row asks for it with `ctx.get(TOOL_HOST)`. `None` is a command run for one
    call, where work has to finish before the answer; a host is where it may go
    on after (`BackgroundRuns.run`)."""

    tasks: TaskGroup | None = None
    """Where background work runs. Set once the host is serving; a row reads the
    host at mount and uses it at a call, which is after."""
    working: int = 0
    """Background pieces of work not yet over. A host with any does not go idle."""


TOOL_HOST: ServiceKey[ToolHost] = ServiceKey("tool_host")
"""The host a row is mounted in, when it is mounted in one (`ToolHost`)."""


@dataclass(slots=True)
class ToolEnv:
    """The variables this command's rows read that only their configs name.

    `ToolCli.inherit_env` is what a package always reads. This is what one
    deployment's settings add: the variable an endpoint's key is in, named in a
    row's config and so known only once that row is mounted. A row registers the
    name (`contribute_item(ctx, TOOL_ENV, name, …)`), and `manual` lists it in
    each tool's template, because a UTCP client hands a command only the
    variables its template names. A name, and never what the variable holds.

    Provided by this command to the rows it mounts, and absent where a row is
    mounted any other way, so a contribution there waits for nothing."""

    ctx: Context
    names: list[str] = field(default_factory=list)

    def register(self, name: str, /, *, scope: Context | None = None) -> Disposer:
        return claim_entry(self.ctx.owner_for(scope), self.names, name, label=f"tool-env({name})")


TOOL_ENV: ServiceKey[ToolEnv] = ServiceKey("tool_env")

REPORT_EVERY_S = 0.5
"""How often a run's progress is written, at most."""

_UNASKED: ContextVar[bool] = ContextVar("ph_toolcli_unasked", default=False)
"""Set by a host for the calls it makes of itself — the tools it was told to
warm — so a run they start knows nobody is waiting for its counts. Task-local,
so a person's call that starts a run beside a warm one is not mistaken for it."""


@dataclass(slots=True)
class RunProgress:
    """One run's account of itself: how much there is, and how far it has got.

    Handed to the work by `BackgroundRuns.run`, which writes the run's start of
    life, its end and its failure; the work says only what it alone knows —
    `begin` once it knows how much there is, under whatever lock makes the run
    its own, and `tick` as it goes."""

    runs: BackgroundRuns
    home: Path
    what: str
    """What the units are, plural: `documents`."""
    owed: bool = False
    """`Progress.owed`: whether anybody is owed word of the end."""
    raise_if_canceled: Callable[[], None] | None = None
    """The caller's own cancellation, asked at every `tick` while the run is the
    caller's alone. A run in a host is not: it goes on past a caller that gives
    up, and other calls join it, so nothing of one caller's stops it."""
    started: float = field(default_factory=time.time)
    total: int = 0
    done: int = 0
    _written: float = 0.0

    def begin(self, total: int) -> None:
        """There are `total` units to go through, and none done yet."""
        self.total, self.done = total, 0
        self._write("working")

    def tick(self, done: int) -> None:
        """`done` units are through: known at once in this process, and written
        at most every `REPORT_EVERY_S`. Raises when the caller that owns the run
        was canceled."""
        if self.raise_if_canceled is not None:
            self.raise_if_canceled()
        self.done = done
        self._write("working", to_file=time.monotonic() - self._written >= REPORT_EVERY_S)

    def _write(
        self,
        state: Literal["working", "done", "failed"],
        failure: str = "",
        *,
        note: str = "",
        to_file: bool = True,
    ) -> None:
        progress = Progress(
            state=state,
            done=self.done,
            total=self.total,
            what=self.what,
            failure=failure,
            started=self.started,
            owed=self.owed,
            note=note,
        )
        self.runs._progress[self.home] = progress
        if to_file:
            self._written = time.monotonic()
            write_progress(self.home, progress)


@dataclass(slots=True)
class _Run:
    """One background run in a host, and what became of it."""

    request: object
    progress: RunProgress
    finished: anyio.Event = field(default_factory=anyio.Event)
    value: object = None
    error: Exception | None = None


@dataclass(slots=True)
class BackgroundRuns:
    """The long pieces of work a row does about a directory — indexing runs — and
    how far each has got: for one call, and for a host.

    `run` is the one door. With no host the work runs to its end, as any tool
    body does. In a host it goes on past the call that asked for it: the call
    waits `grace` and is answered with the work's value, or with `None` — "still
    going", which the row turns into an answer of its own. Either way the run's
    life is written to its directory's progress file (`RunProgress`), which is
    what a session watches for the footer and the note when it ends.
    """

    grace: float = 5.0
    """How long a host's call waits for its run before answering "still going".
    The row's to set: long enough that an ordinary run answers with its counts,
    like a call that ran to the end."""
    _runs: dict[Path, list[_Run]] = field(default_factory=dict)
    """Per directory, the runs a host has not finished."""
    _progress: dict[Path, Progress] = field(default_factory=dict)
    """Per directory, what the run working on it last reported."""

    def working(self, home: Path, lock: Path, *, what: str) -> Progress | None:
        """The run building what is in `home` right now, or `None`.

        A host knows its own. Any other process asks the file and then `lock`,
        the file lock a run holds for as long as it works: a `working` report
        with the lock free is one a stopped run left behind."""
        if self._runs.get(home):
            return self._progress.get(home)
        found = read_progress(home)
        if found is None or found.state != "working":
            return None
        release = try_file_lock(lock, what=what)
        if release is None:
            return found
        release()
        return None

    async def run[T](
        self,
        host: ToolHost | None,
        home: Path,
        request: object,
        work: Callable[[RunProgress], Awaitable[T]],
        *,
        what: str,
        said: Callable[[T], str],
        raise_if_canceled: Callable[[], None] | None = None,
    ) -> T | None:
        """`work`'s value, or `None` while it goes on in `host` past this call.
        What it raised is raised here.

        `said` is what the value says beyond its count, in a few lines: what the
        tool's result tells a caller that waited. Written with the end
        (`Progress.note`), so whoever is told of a run nobody waited for is told
        that too. Required: a run whose note says only "finished" can hide one
        that skipped everything.

        One run a request in a host, and the same request asked again while its
        run is going joins it: a model that calls an index tool to see how far it
        has got must not queue the work a second time. Different requests each
        get a run, and take their turns on whatever lock the work holds.

        `raise_if_canceled` is the caller's (`ToolRunContext`'s), and stops the
        work at its next `tick` only where the run is that caller's alone: with
        no host."""
        if host is None:
            progress = RunProgress(self, home, what, raise_if_canceled=raise_if_canceled)
            return await self._through(progress, work, said)
        runs = self._runs.setdefault(home, [])
        mine = next((one for one in runs if one.request == request), None)
        if mine is None:
            # Unasked work — the host warming a tool — has no caller to give its
            # counts to, so its end is owed from the start.
            mine = started = _Run(request, RunProgress(self, home, what, owed=_UNASKED.get()))
            runs.append(started)

            async def drive() -> None:
                try:
                    value: T = await self._through(started.progress, work, said)
                    started.value = value
                except Exception as error:
                    started.error = error
                finally:
                    runs.remove(started)
                    started.finished.set()
                    host.working -= 1

            if host.tasks is None:
                raise RuntimeError("the tool host is not serving yet")
            # Counted here and not in the task: a host asked whether it is idle
            # between this line and the task's first step must not say yes.
            host.working += 1
            host.tasks.start_soon(drive)
        with anyio.move_on_after(self.grace):
            await mine.finished.wait()
        if mine.error is not None:
            raise mine.error
        if not mine.finished.is_set():
            # Answered "still going": whoever asked is owed word of its end,
            # which the run's last report will carry.
            mine.progress.owed = True
            return None
        value: T = mine.value  # type: ignore[assignment]
        return value

    async def _through[T](
        self,
        progress: RunProgress,
        work: Callable[[RunProgress], Awaitable[T]],
        said: Callable[[T], str],
    ) -> T:
        """`work` with its end written: `done` with what `said` makes of its
        value, or `failed` with the reason and how far it got — by this door, so
        no row can forget either."""
        try:
            value = await work(progress)
        except BaseException as error:
            progress._write("failed", str(error) or type(error).__name__)
            raise
        progress.done = progress.total
        progress._write("done", note=said(value))
        return value


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
    it registers them (`ph_code_graph.code_index`). `named` is what the mounted
    rows registered (`ToolEnv`)."""
    inherit = [*_inherited(cli), *dict.fromkeys(named)]
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
    """A failure as `call` prints it, and as anything else that answers in this
    envelope does — a session embedding for a command. `read_envelope` raises it
    as a `HarnessError` with `code`."""
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
async def _mounted(
    cli: ToolCli, root: Path, config_file: Path | None, host: ToolHost | None = None
) -> AsyncIterator[Context]:
    """The package's rows over `HOST_ROWS`, mounted at `root` and unwound on exit.
    In `host`, when this is a `serve`: provided before the rows mount, so a row
    reads it where it registers its tools."""
    configs = _row_configs(cli, config_file)
    rows: list[JsonValue] = [dict(row) for row in HOST_ROWS]
    for row in (*cli.rows, *_provider(cli, configs)):
        config = configs.get(as_str(row["id"]))
        rows.append({**row, "config": {**as_obj(row.get("config")), **config}} if config else row)
    ctx = Context()
    try:
        ctx.provide(TOOL_ENV, ToolEnv(ctx))
        if host is not None:
            ctx.provide(TOOL_HOST, host)
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


# --------------------------------------------------------------------- host ----

_HOST_CONNECT_S = 2.0
"""How long `call` waits to reach a host before running by itself."""

_IDLE_S = 15 * 60.0
"""How long a host with nothing to do stays up, unless `--idle` says."""


def _asked_of_host(
    root: Path, tool: str, args_json: str | None, rest: Sequence[str]
) -> dict[str, object] | None:
    """One call's answer from the host in `$PH_TOOL_HOST`, or `None` to run it here.

    `None` for every way a host can be absent — no variable, nothing listening,
    a host for another tree, one that died part-way — because a call that could
    have been answered here must not fail for want of a process nobody promised.
    No limit on the wait for the answer: how long a call may take is its
    caller's to say, and the caller ends this process when it has waited enough.
    """
    path = os.environ.get(HOST_ENV, "").strip()
    if not path:
        return None
    request = {
        "root": str(root),
        "state": str(state_dir() or ""),
        "tool": tool,
        "args_json": args_json,
        "rest": list(rest),
    }
    try:
        answer = ask(path, request, connect_timeout=_HOST_CONNECT_S)
    except (OSError, FramingError):
        return None
    return answer if isinstance(answer.get("ok"), bool) else None


async def _answered_call(
    cli: ToolCli, ctx: Context, tool: str, args_json: str | None, rest: Sequence[str]
) -> dict[str, object]:
    """One call's envelope, whatever happened — a host answers with it either way."""
    try:
        return await _call(cli, ctx, tool, args_json, rest)
    except _Refused as error:
        return failure_envelope(error.code, str(error))
    except Exception as error:
        log.exception("%s failed a served call", cli.program)
        return failure_envelope("INTERNAL_ERROR", f"{type(error).__name__}: {error}")


async def _serve(cli: ToolCli, root: Path, known: argparse.Namespace) -> int:
    """Mount once and answer calls on `--socket` until idle or told to stop.

    One request a connection, a frame each way (`ph.framing`). Calls run side by
    side in the one mount, as a session's do. The host leaves when nothing has
    asked for `--idle` seconds and no background work is running, and whoever
    wanted it starts another; so it never needs to be stopped to be tidy, and a
    killed one costs only what its rows had not yet saved.
    """
    path = Path(known.socket)
    if await listening(path, timeout=_HOST_CONNECT_S):
        # Two starts raced, or a host outlived whoever started it. One is enough.
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    host = ToolHost()
    serving = 0
    last = anyio.current_time()

    async with _mounted(cli, root, known.config, host) as ctx:

        async def respond(request: dict[str, Any]) -> dict[str, object]:
            if request.get("root") != str(root) or request.get("state") != str(state_dir() or ""):
                # Another tree, or another index of this one. The caller runs it.
                return {"served": False}
            rest = request.get("rest")
            args_json = request.get("args_json")
            return await _answered_call(
                cli,
                ctx,
                as_str(request.get("tool")),
                args_json if isinstance(args_json, str) else None,
                [str(one) for one in as_seq(rest)],
            )

        async def answer(stream: SocketStream) -> None:
            nonlocal serving, last
            serving += 1
            try:
                await answer_one(stream, respond)
            finally:
                serving -= 1
                last = anyio.current_time()

        async def warm(tool: str) -> None:
            # A call the host makes of itself: nobody waits for its counts.
            _UNASKED.set(True)
            answered = await _answered_call(cli, ctx, tool, None, ())
            if not answered.get("ok"):
                log.warning("%s could not warm %s: %s", cli.program, tool, answered.get("error"))

        try:
            async with (
                await anyio.create_unix_listener(path) as listener,
                anyio.create_task_group() as tasks,
            ):
                host.tasks = tasks
                tasks.start_soon(listener.serve, answer)
                for tool in known.warm or ():
                    tasks.start_soon(warm, tool)
                while serving or host.working or anyio.current_time() - last < known.idle:
                    await anyio.sleep(min(1.0, max(known.idle / 4, 0.05)))
                tasks.cancel_scope.cancel()
        finally:
            host.tasks = None
            path.unlink(missing_ok=True)
    return 0


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

    serve = commands.add_parser(
        "serve",
        help="Stay up and answer calls on a socket.",
        description="Mount once and answer `call`s on a unix socket, so what a tool "
        f"loads is loaded once. A `call` with ${HOST_ENV} set to the socket uses it.",
    )
    serve.add_argument("--socket", required=True, help="The unix socket to listen on.")
    serve.add_argument(
        "--idle",
        type=float,
        default=_IDLE_S,
        help="Leave after this many seconds with no call and no background work.",
    )
    serve.add_argument(
        "--warm",
        action="append",
        metavar="TOOL",
        help="A tool to call once, with no arguments, when the host starts — an index "
        "tool, which in a host returns at once and works in the background. Repeatable.",
    )
    common(serve)

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
    as_json = known.verb in ("manual", "serve") or (known.verb == "call" and known.format == "json")
    with ExitStack() as stack:
        stack.enter_context(_logging_to_file(cli, verbose=known.verbose))
        stack.enter_context(_state_at(known.state))
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
                return utcp_manual(cli, _definitions(ctx), ctx.require(TOOL_ENV).names)
            if verb == "tools":
                return _tools(cli, ctx)
            return await _provision(cli, ctx, verb)

    def refused(code: str, message: str) -> int:
        if as_json:
            print(dumps(failure_envelope(code, message)), file=out)
        else:
            print(f"{cli.program}: {message}", file=sys.stderr)
        return 1

    try:
        if verb == "serve":
            # Stopped by a signal as every host is (`ph.resources`): the mount
            # unwinds, and a second signal or a hung unwind ends it at once.
            return run_until_signaled(partial(_serve, cli, root, known))
        answer = (
            _asked_of_host(root, known.tool, known.args_json, rest) if verb == "call" else None
        ) or anyio.run(answered)
    except (_Refused, HarnessError) as error:
        # A refusal of this command line's, or a provisioning command that could
        # not do what was asked: a sentence and a code, and a failing exit.
        return refused(error.code, str(error))
    except CordisError as error:
        # A row that declined, or a profile that would not compose: a deployment
        # fact with a sentence, the same one `phern` prints for it.
        log.warning("%s could not mount", cli.program, exc_info=True)
        return refused("MOUNT_REFUSED", str(error))
    except Exception as error:
        # A bug, but one a UTCP client would otherwise read as an empty result.
        # The traceback is in the log file; the sentence is in the answer.
        log.exception("%s failed", cli.program)
        return refused("INTERNAL_ERROR", f"{type(error).__name__}: {error}")

    if isinstance(answer, str):
        print(answer, file=out)
        return 0
    if as_json:
        print(dumps(answer), file=out)
    elif answer["ok"]:
        print(answer["text"], file=out)
    else:
        print(f"{cli.program}: {as_obj(answer['error'])['message']}", file=sys.stderr)
    return 0 if verb == "manual" or answer["ok"] else 1


def main(cli: ToolCli) -> NoReturn:
    """The console script's body."""
    sys.exit(run(cli))
