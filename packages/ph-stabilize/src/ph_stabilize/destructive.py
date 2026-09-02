"""What a call would do that cannot be taken back — parsed, not pattern-matched.

**Why this is not a regex.** The shipped classifier was twelve regexes run over
the call's arguments rendered as JSON, and it had the defect that shape invites:
`dumps` escapes a real newline to the two characters `\\` and `n`, so every
pattern anchored with `\\b` — twelve of the thirteen — stopped matching the
moment its payload was on a second line. `rm -rf`, `DROP TABLE`, `shutil.rmtree`
and `curl | sh` were all ungated inside a multi-line cell, which is *every*
`run_code` cell, the surface the row exists for. Nothing failed; the gate simply
did not fire.

That is a symptom, and the cause is that a regex cannot decide this question. A
command line is a grammar — quoting, operators, redirects, nesting — and a
pattern language that cannot count parentheses cannot be made to see it. Every
fix is another escape hatch: `rm -rf` and `rm -fr` and `rm  -r`, then `"rm"` in
quotes, then `$(echo rm) -rf`. So the text is **parsed** into commands,
arguments, pipes and redirects, and the judgement is made against that structure.

Three dialects, each with a parser and a table:

* **shell** — `shlex` splits it (POSIX quoting is the part worth not writing),
  then pipelines and redirects are read off the operator tokens. Case-sensitive,
  because the shell is: `RM` is not `rm`.
* **sql** — comments and string literals are removed, statements split on `;`,
  and the leading keyword decides. Case-insensitive, because SQL is.
* **python** — `ast`, so `shutil.rmtree` is a call to that function rather than
  a substring, and `subprocess.run("rm -rf /")` hands its argument back to the
  shell reader.

**The tables are a starting point and are meant to grow.** Neither list claims to
be complete — no list of this kind ever is — and the honest posture is that this
gates what it knows and says what it found. A dialect nobody has written a parser
for contributes nothing rather than guessing, and adding one is a table plus a
reader.

@module ph_stabilize.destructive
"""

from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

__all__ = [
    "PYTHON_CALLS",
    "PYTHON_METHODS",
    "SHELL_RULES",
    "SQL_STATEMENTS",
    "Dialect",
    "Finding",
    "ShellRule",
    "decode",
    "dialect_of",
    "findings",
    "strings_in",
]

Dialect: TypeAlias = Literal["shell", "sql", "python"]


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing a parser found, in the words a person is shown.

    `text` is the fragment reconstructed from the parse rather than the raw
    input, so what an operator reads back is what the classifier actually
    understood — a prompt quoting the whole 4 KB cell teaches nothing, and one
    quoting the regex that matched teaches less.
    """

    dialect: Dialect
    text: str
    reason: str

    def __str__(self) -> str:
        return f"{self.text} ({self.reason})"


# --------------------------------------------------------------- decoding --

_ESCAPES = (("\\r\\n", "\n"), ("\\n", "\n"), ("\\r", "\n"), ("\\t", " "))


def decode(text: str) -> str:
    """Turn escaped whitespace back into whitespace before anything reads it.

    Two different producers put `\\n` in a string as two characters: a model
    writing a shell command inside a JSON blob it composed by hand, and any
    caller that re-encoded a payload on the way here. Both make a line boundary
    invisible to a reader that expects one, which is precisely how the old
    classifier lost `\\b`.

    Deliberately unconditional and slightly greedy: decoding a `\\n` that was
    *meant* literally can only add a boundary where the parsers look for one, so
    the error direction is more findings rather than fewer. For an approval gate
    that is the side to be wrong on.
    """
    for escaped, real in _ESCAPES:
        text = text.replace(escaped, real)
    return text


def strings_in(value: Any, *, depth: int = 0) -> Iterator[str]:
    """Every string leaf of a call's arguments, decoded.

    The arguments are walked rather than serialized. Rendering them to JSON and
    scanning the result is what escaped the newlines out of existence, and it
    also meant every reader had to see through the envelope's own punctuation.
    A `MappingProxyType` is a `Mapping`, so the frozen tree walks like any other.
    """
    if depth > 8:
        return
    if isinstance(value, str):
        yield decode(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from strings_in(item, depth=depth + 1)
    elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        for item in value:
            yield from strings_in(item, depth=depth + 1)


# ------------------------------------------------------------------ shell --


@dataclass(frozen=True, slots=True)
class ShellRule:
    """When one command name is destructive, and how to say so.

    `always` is the command whose whole job is removal; the rest need a flag or a
    subcommand, because `git push` is ordinary and `git push --force` is not.
    """

    reason: str
    always: bool = False
    subcommand: str = ""
    flags: frozenset[str] = field(default_factory=frozenset)


SHELL_RULES: dict[str, tuple[ShellRule, ...]] = {
    "rm": (
        ShellRule("removes a directory tree", flags=frozenset({"-r", "-R", "--recursive", "-rf"})),
    ),
    "rmdir": (ShellRule("removes directories", always=True),),
    "shred": (ShellRule("overwrites a file so it cannot be recovered", always=True),),
    "dd": (ShellRule("writes a raw block device or file", always=True),),
    "mkfs": (ShellRule("makes a filesystem, destroying what is there", always=True),),
    "mkswap": (ShellRule("reformats a device as swap", always=True),),
    "wipefs": (ShellRule("erases filesystem signatures", always=True),),
    "fdisk": (ShellRule("rewrites a partition table", always=True),),
    "parted": (ShellRule("rewrites a partition table", always=True),),
    "sed": (ShellRule("edits files in place", flags=frozenset({"-i", "--in-place"})),),
    "find": (
        ShellRule("deletes what it finds", flags=frozenset({"-delete"})),
        ShellRule("runs a command on what it finds", flags=frozenset({"-exec", "-execdir", "-ok"})),
    ),
    "chmod": (
        ShellRule("changes permissions recursively", flags=frozenset({"-R", "--recursive"})),
    ),
    "chown": (ShellRule("changes ownership recursively", flags=frozenset({"-R", "--recursive"})),),
    "truncate": (ShellRule("truncates a file", flags=frozenset({"-s", "--size"})),),
    "git": (
        ShellRule(
            "rewrites published history",
            subcommand="push",
            flags=frozenset({"-f", "--force", "--force-with-lease"}),
        ),
        ShellRule(
            "discards commits and working-tree changes",
            subcommand="reset",
            flags=frozenset({"--hard"}),
        ),
        ShellRule(
            "deletes untracked files", subcommand="clean", flags=frozenset({"-f", "--force"})
        ),
        ShellRule("deletes a branch", subcommand="branch", flags=frozenset({"-D"})),
    ),
    "docker": (
        ShellRule("removes containers", subcommand="rm", flags=frozenset({"-f", "--force"})),
        ShellRule("removes images", subcommand="rmi"),
        ShellRule("removes unused objects", subcommand="prune"),
    ),
    "kubectl": (ShellRule("deletes cluster objects", subcommand="delete"),),
    "terraform": (ShellRule("destroys managed infrastructure", subcommand="destroy"),),
}
"""Command name → when it is destructive. Case-sensitive, because the shell is.

A starting point rather than a claim of completeness: what is here is what the
port plan's own list named plus the neighbours that share its shape. A command
missing from this table is not gated by the preset, which is what a deployment's
own `when:` patterns and its `interrupt_on` entries are for.

**`rm` gates on `-r` only, which is narrower than what it replaces.** The regex
this supersedes wanted `-r` *or* `-f`, and a first draft here gated `rm` with no
flag at all. Both are too wide for the same reason: an agent removing a build
artifact or a temp file is doing routine work, and a gate that prompts on routine
work is one a person learns to approve without reading — which the `hitl` module
calls worse than no gate. What is left is the case a person actually wants to see
before it happens, a **tree** going away.

The cost is stated rather than hidden: `rm -f one-file` and a bare `rm one-file`
are now ungated, so a single irreversible deletion passes without a prompt. That
is the deliberate trade — recursion is the multiplier, and this list is tuned for
what is worth interrupting a person over rather than for everything that cannot
be undone. A deployment that wants the wider net writes `rm` into `interrupt_on`
with its own rule, or adds a pattern to `when:`.
"""

_FETCHERS = frozenset({"curl", "wget", "fetch"})
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish"})
_RAW_DEVICES = re.compile(r"^/dev/(sd[a-z]|nvme\d|hd[a-z]|disk\d|vd[a-z])")
_OPERATORS = frozenset({"|", "||", "&&", ";", "&", "\n"})
_REDIRECTS = frozenset({">", ">>", ">|"})


def _tokenize(line: str) -> list[str]:
    """One line of shell as tokens, with operators kept as tokens of their own."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        # Unbalanced quoting. A shell would refuse it too, but refusing to look
        # is how a gate is evaded by appending a quote, so fall back to a plain
        # split: the words are still there, only the grouping is lost.
        return line.split()


def _has_flag(argv: Sequence[str], flag: str) -> bool:
    """Whether `argv` carries `flag`, allowing for the spellings a shell allows.

    `--force` matches `--force` and `--force=x`. A single-character `-i` matches
    `-i`, `-i.bak` (an argument stuck to the flag) and `-ri` (a bundle), because
    all three are the same flag to the command reading them — and a classifier
    that only saw the first is one `sed -ri` defeats.
    """
    for token in argv:
        if token == flag or token.startswith(f"{flag}="):
            return True
        if (
            len(flag) == 2
            and flag[0] == "-"
            and token.startswith("-")
            and not token.startswith("--")
        ):
            body = token[1:]
            if body.startswith(flag[1]) or (body.isalpha() and flag[1] in body):
                return True
    return False


def _simple_commands(text: str) -> Iterator[list[str]]:
    """Each simple command in the text, split on newlines and shell operators."""
    for line in text.splitlines():
        if not line.strip():
            continue
        current: list[str] = []
        for token in _tokenize(line):
            if token in _OPERATORS:
                if current:
                    yield current
                current = []
            else:
                current.append(token)
        if current:
            yield current


def _shell_findings(text: str) -> list[Finding]:
    commands = list(_simple_commands(text))
    found: list[Finding] = []
    for argv in commands:
        found.extend(_command_findings(argv))
    found.extend(_pipeline_findings(text, commands))
    return found


def _command_findings(argv: list[str]) -> Iterator[Finding]:
    name = argv[0].rsplit("/", 1)[-1]
    rules = SHELL_RULES.get(name)
    if rules is None:
        rules = SHELL_RULES.get(name.split(".", 1)[0]) if name.startswith("mkfs.") else None
    if rules is not None:
        positional = [token for token in argv[1:] if not token.startswith("-")]
        subcommand = positional[0] if positional else ""
        for rule in rules:
            if rule.subcommand and rule.subcommand != subcommand:
                continue
            if rule.flags and not any(_has_flag(argv, flag) for flag in rule.flags):
                continue
            if not rule.always and not rule.flags and not rule.subcommand:
                continue
            yield Finding("shell", " ".join(argv[:3]), rule.reason)
            break
    for index, token in enumerate(argv):
        if token in _REDIRECTS and index + 1 < len(argv):
            target = argv[index + 1]
            if _RAW_DEVICES.match(target):
                yield Finding("shell", f"{token} {target}", "writes directly to a block device")


def _pipeline_findings(text: str, commands: list[list[str]]) -> Iterator[Finding]:
    """A network fetch piped into a shell — the shape no single command shows."""
    if "|" not in text:
        return
    names = [argv[0].rsplit("/", 1)[-1] for argv in commands if argv]
    if any(name in _FETCHERS for name in names) and any(name in _SHELLS for name in names):
        yield Finding("shell", " | ".join(names), "runs code fetched from the network")


# -------------------------------------------------------------------- sql --

SQL_STATEMENTS: dict[str, str] = {
    "DROP": "drops a table, database or schema",
    "TRUNCATE": "empties a table",
    "DELETE": "deletes rows",
    "UPDATE": "rewrites rows",
    "ALTER": "changes a schema",
    "GRANT": "changes access control",
    "REVOKE": "changes access control",
}
"""Leading statement keyword → what it does. Matched case-insensitively, because
the parser has already decided this is SQL and SQL keywords are not case-bound."""

_SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_SQL_LITERAL = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_SQL_LEAD = re.compile(r"^\s*([A-Za-z]+)")
_SQL_KEYWORDS = frozenset(SQL_STATEMENTS) | {"SELECT", "INSERT", "CREATE", "WITH", "REPLACE"}


def _sql_statements(text: str) -> Iterator[tuple[str, str]]:
    """`(keyword, statement)` for each statement, comments and literals removed.

    Literals go before the split so a `;` or a `WHERE` inside a quoted string
    cannot end a statement or make an unqualified `DELETE` look qualified.
    """
    stripped = _SQL_LITERAL.sub("''", _SQL_COMMENT.sub(" ", text))
    for statement in stripped.split(";"):
        lead = _SQL_LEAD.match(statement)
        if lead is not None:
            yield lead.group(1).upper(), statement


def _sql_findings(text: str) -> list[Finding]:
    found: list[Finding] = []
    for keyword, statement in _sql_statements(text):
        reason = SQL_STATEMENTS.get(keyword)
        if reason is None:
            continue
        if keyword in {"DELETE", "UPDATE"} and not re.search(r"\bWHERE\b", statement, re.I):
            reason = f"{reason}, and names no WHERE clause — every row"
        found.append(Finding("sql", " ".join(statement.split()[:4]), reason))
    return found


# ----------------------------------------------------------------- python --

PYTHON_CALLS: dict[str, str] = {
    "shutil.rmtree": "removes a directory tree",
    "os.remove": "removes a file",
    "os.unlink": "removes a file",
    "os.rmdir": "removes a directory",
    "os.removedirs": "removes directories",
    "os.truncate": "truncates a file",
}
"""Dotted module function → what it does. Matched on the whole path, so a local
named `remove` is not mistaken for `os.remove`."""

PYTHON_METHODS: dict[str, str] = {
    "unlink": "removes a file",
    "rmtree": "removes a directory tree",
    "rmdir": "removes a directory",
}
"""Method name → what it does, matched on *any* receiver.

Separate from the dotted table because the receiver is usually unknowable:
`Path("/etc/hosts").unlink()` builds its path in the call, and a table that had
to name the receiver would have to enumerate every way of spelling one. The cost
is a false positive on some other object's `unlink`, which for an approval gate
is the affordable direction."""

_SHELL_ESCAPES = frozenset(
    {
        "os.system",
        "os.popen",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
        "run",
        "call",
        "Popen",
    }
)


def _dotted(node: ast.AST) -> str:
    """The dotted name of a call target: `shutil.rmtree`, or the bare method."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _python_findings(text: str) -> list[Finding]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # A cell that will not parse will not run either, so there is nothing to
        # gate — but say nothing rather than falling through to another reader,
        # which would tokenize Python as shell and invent findings.
        return []
    found: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func)
        reason = PYTHON_CALLS.get(name) or PYTHON_METHODS.get(name.rsplit(".", 1)[-1])
        if reason is not None:
            found.append(Finding("python", f"{name}(...)", reason))
        if name in _SHELL_ESCAPES or name.rsplit(".", 1)[-1] in {"system", "popen"}:
            found.extend(_escaped_shell(node))
    return found


def _escaped_shell(node: ast.Call) -> Iterator[Finding]:
    """A Python call that hands a command to a shell — read the command as shell."""
    if not node.args:
        return
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        yield from _shell_findings(first.value)
    elif isinstance(first, ast.List | ast.Tuple):
        argv = [
            element.value
            for element in first.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
        if argv:
            yield from _command_findings(argv)


# ------------------------------------------------------------- dispatching --

_PYTHON_MARKERS = (
    ast.Import,
    ast.ImportFrom,
    ast.Call,
    ast.Attribute,
    ast.Assign,
    ast.AugAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.For,
    ast.While,
    ast.With,
    ast.Try,
)


def dialect_of(text: str) -> Dialect:
    """Which parser reads this string.

    Ordered by how *specific* the evidence is, because the readers disagree about
    ambiguous input: `rm -rf /tmp` is, unhelpfully, a valid Python expression
    (`rm - rf / tmp`). So Python is claimed only when the tree contains something
    a shell command could not produce — an import, a call, an assignment — and
    SQL only when a statement keyword leads. Everything else is shell, which is
    the dialect a bare command line is in.
    """
    for keyword, _statement in _sql_statements(text):
        if keyword in _SQL_KEYWORDS:
            return "sql"
        break
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "shell"
    if any(isinstance(node, _PYTHON_MARKERS) for node in ast.walk(tree)):
        return "python"
    return "shell"


_READERS = {"shell": _shell_findings, "sql": _sql_findings, "python": _python_findings}


def findings(value: Any) -> list[Finding]:
    """Everything destructive in one call's arguments, deduplicated in order.

    Each string leaf is dispatched to the reader for its own dialect, so a cell
    that shells out is read as Python *and* its command as shell, and a tool
    carrying both a query and a path is read as each.
    """
    found: list[Finding] = []
    seen: set[tuple[str, str]] = set()
    for text in strings_in(value):
        if not text.strip():
            continue
        for one in _READERS[dialect_of(text)](text):
            key = (one.text, one.reason)
            if key not in seen:
                seen.add(key)
                found.append(one)
    return found
