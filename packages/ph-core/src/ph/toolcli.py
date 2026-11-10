"""`ph.toolcli` — a package's tools as a command line, and as a UTCP manual.

A package that registers tools through rows gets a console script by naming those
rows: `ToolCli(program=..., rows=...)` and `main`. The script mounts a small
profile — the registry, the filesystem, a subprocess door, and the package's own
rows — and runs a tool **through the registry**, the way `ph.testing.run_tool`
does. So there is one body per tool: the one the row registers. What differs
between a session and this command is which process mounted it.

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
  prints exactly one JSON object, a failure is reported inside it, and logging
  goes to a file. Each generated template ends in `2>/dev/null` as well, because
  a dependency that writes a warning is not ours to silence.
* **The environment is cut to an allowlist.** Naming any variable replaces the
  client's default list, so the template names the defaults again beside the pH
  roots — without them a UTCP call would build a different index from the one a
  session uses for the same checkout.

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
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import NoReturn, TextIO

import anyio
from pydantic import BaseModel

from .cordis import DEPLOYMENT, Context, CordisError, Profile, maybe_await
from .cordis.loader import ProfileDocument
from .documents import DOCUMENT_FAULTS, decode_document
from .json import JsonObject, JsonValue, as_obj, as_str, dumps, loads
from .keys import COMMANDS, DIAGNOSTICS, TOOLS
from .llm.types import text_of
from .paths import resolve_roots
from .seams.commands import CommandContext
from .tools.definition import ToolDefinition, ToolExecutionInput, ToolExecutionResult
from .tools.errors import HarnessError
from .tools.json_schema import schema_of

__all__ = [
    "CONFIG_ENV",
    "HOST_ROWS",
    "INHERITED_ENV",
    "UTCP_VERSION",
    "ToolCli",
    "config_env",
    "decode_argument",
    "envelope",
    "main",
    "manual_call_template",
    "read_envelope",
    "run",
    "utcp_manual",
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

MISSING_ARGUMENT = "MISSING_ARG_"
"""What `utcp_cli` substitutes for a placeholder whose argument was not given."""

HOST_ROWS: tuple[JsonObject, ...] = (
    {"id": "tools", "name": "tools"},
    {"id": "subprocess", "name": "subprocess-local"},
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
)

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

    Only what differs from each model's defaults, and no variable at all when
    nothing does — so a default row's command runs with nothing set."""
    settings = {
        row: dumped
        for row, config in configs.items()
        if (dumped := config.model_dump(by_alias=True, exclude_defaults=True))
    }
    return {CONFIG_ENV: dumps(settings)} if settings else {}


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


def _template(cli: ToolCli, definition: ToolDefinition) -> dict[str, object]:
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
    # for the client to substitute it, and the redirect has to stay a redirect.
    return {
        "call_template_type": "cli",
        "commands": [{"command": " ".join(words) + " 2>/dev/null"}],
        "inherit_env_vars": [*INHERITED_ENV, *cli.inherit_env],
    }


def utcp_manual(cli: ToolCli, definitions: Sequence[ToolDefinition]) -> dict[str, object]:
    """The UTCP manual for `definitions`: what `manual` prints.

    Names are the tools' own. A client prefixes each with the manual's name when
    it registers them (`ph_code_graph.code_index`)."""
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
                "tool_call_template": _template(cli, one),
            }
            for one in definitions
        ],
    }


def manual_call_template(cli: ToolCli) -> dict[str, object]:
    """What a harness registers to discover this package's tools."""
    return {
        "name": cli.manual,
        "call_template_type": "cli",
        "commands": [{"command": shlex.join((*cli.invocation(), "manual")) + " 2>/dev/null"}],
        "inherit_env_vars": [*INHERITED_ENV, *cli.inherit_env],
    }


# ----------------------------------------------------------------- envelope ----


def _failure(code: str, message: str) -> dict[str, object]:
    return {"ok": False, "error": {"code": code, "message": message}}


def envelope(result: ToolExecutionResult) -> dict[str, object]:
    """One call's outcome as the object `call` prints.

    `value` is the tool's declared output and `text` is its own `render` — what a
    model in a session reads. A failure carries its routable code when the error
    had one, and its kind otherwise. `read_envelope` is the inverse."""
    text = text_of(result.content)
    if not result.is_error:
        return {"ok": True, "value": result.value, "text": text}
    failure = result.error
    if failure is None:
        return _failure("TOOL_FAILED", text)
    code = (failure.info or {}).get("code") or f"TOOL_{failure.kind.upper()}"
    return _failure(code, failure.message)


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

    known = {as_str(row["id"]) for row in cli.rows}
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
    """The package's rows over `HOST_ROWS`, mounted at `root` and unwound on exit."""
    configs = _row_configs(cli, config_file)
    rows: list[JsonValue] = [dict(row) for row in HOST_ROWS]
    for row in cli.rows:
        config = configs.get(as_str(row["id"]))
        rows.append({**row, "config": {**as_obj(row.get("config")), **config}} if config else row)
    ctx = Context()
    try:
        await Profile.from_documents([ProfileDocument(cli.program, rows)]).mount(ctx, project=root)
        yield ctx
    finally:
        await ctx.drain()
        await ctx.dispose()


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
        return _failure("UNKNOWN_TOOL", f'{cli.program} has no tool "{tool}" (it has {names})')
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
    process per call appends to one file and nothing rotates it.

    Undone on the way out. The process is about to exit, so nothing depends on
    it there; a caller that runs this in its own process — a test — does.
    """
    root = logging.getLogger()
    level = root.level
    handler: logging.Handler
    path = cli.log_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
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
    with _logging_to_file(cli, verbose=known.verbose):
        return _answered(cli, known, rest, out)


def _answered(cli: ToolCli, known: argparse.Namespace, rest: Sequence[str], out: TextIO) -> int:
    verb: str = known.verb
    as_json = verb == "manual" or (verb == "call" and known.format == "json")
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
                return utcp_manual(cli, _definitions(ctx))
            if verb == "tools":
                return _tools(cli, ctx)
            return await _provision(cli, ctx, verb)

    def refused(code: str, message: str) -> int:
        if as_json:
            print(dumps(_failure(code, message)), file=out)
        else:
            print(f"{cli.program}: {message}", file=sys.stderr)
        return 1

    try:
        answer = anyio.run(answered)
    except _Refused as error:
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
    if verb == "manual" or known.format == "json":
        print(dumps(answer), file=out)
    elif answer["ok"]:
        print(answer["text"], file=out)
    else:
        print(f"{cli.program}: {as_obj(answer['error'])['message']}", file=sys.stderr)
    return 0 if verb == "manual" or answer["ok"] else 1


def main(cli: ToolCli) -> NoReturn:
    """The console script's body."""
    sys.exit(run(cli))
