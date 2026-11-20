"""What a call would do that cannot be undone: parsed, not pattern-matched.

**Why this is not a regex.** The shipped classifier was twelve regexes that ran
over the arguments of the call rendered as JSON. It had the defect that this
shape invites. `dumps` escapes a real newline to the two characters `\\` and
`n`, so every pattern anchored with `\\b` (eleven of the twelve) stopped
matching when its payload was on a second line. `rm -rf`, `DROP TABLE`,
`shutil.rmtree` and `curl | sh` were all ungated inside a multi-line cell. That
is *every* `run_code` cell, the surface that the row exists for. Nothing failed.
The gate did not fire.

That is a symptom. The cause is that a regex cannot decide this question. A
command line is a grammar, with quoting, operators, redirects and nesting. A
pattern language that cannot count parentheses cannot be made to see a grammar.
Every fix leaves another way past the pattern: `rm -rf` and `rm -fr` and
`rm  -r`, then `"rm"` in quotes, then `$(echo rm) -rf`. The text is therefore
**parsed** into commands, arguments, pipes and redirects, and the judgment is
made against that structure.

There are three dialects, each with a parser and a table:

* **shell**: `shlex` splits the text, because POSIX quoting is the part that is
  not worth writing again. Pipelines and redirects are then read from the
  operator tokens. The match is case-sensitive, because the shell is: `RM` is
  not `rm`.
* **sql**: comments and string literals are removed, statements are split on
  `;`, and the leading keyword decides. The match is case-insensitive, because
  SQL is.
* **python**: `ast` parses the text, so `shutil.rmtree` is a call to that
  function and not a substring. `subprocess.run("rm -rf /")` passes its argument
  back to the shell reader.

**The tables are a starting point and are meant to grow.** None of the lists
claims to be complete, because no list of this kind is ever complete. The
correct position is that this module gates what it knows and reports what it
found. A dialect for which nobody has written a parser contributes nothing, and
the module does not guess. To add a dialect, add a table and a reader.

@module ph_stabilize.destructive
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ph.json import JsonValue

import ast
import re
import shlex
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

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
    """One thing that a parser found, in the words that a person is shown.

    `text` is the fragment rebuilt from the parse and not the raw input. An
    operator therefore reads back what the classifier really understood. A
    prompt that quotes the whole 4 KB cell teaches nothing, and a prompt that
    quotes the regex that matched teaches less.
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

    Two different producers put `\\n` in a string as two characters. One is a
    model that writes a shell command inside a JSON blob that it composed by
    hand. The other is any caller that encoded a payload again on the way here.
    Both hide a line boundary from a reader that expects one, and that is
    exactly how the old classifier lost `\\b`.

    The decoding is unconditional and slightly greedy, on purpose. To decode a
    `\\n` that was *meant* literally can only add a boundary where the parsers
    look for one, so the error direction is more findings and not fewer. For an
    approval gate, that is the correct side to be wrong on.
    """
    for escaped, real in _ESCAPES:
        text = text.replace(escaped, real)
    return text


def strings_in(value: JsonValue, *, depth: int = 0) -> Iterator[str]:
    """Every string leaf of the arguments of a call, decoded.

    The arguments are walked and not serialized. The old classifier rendered
    them to JSON and scanned the result. That escaped the newlines so that no
    reader saw them, and every reader also had to see through the punctuation of
    the JSON envelope. A `MappingProxyType` is a `Mapping`, so the frozen tree
    walks like any other.
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

    `always` marks a command whose whole job is removal. The other rules need a
    flag or a subcommand, because `git push` is ordinary and `git push --force`
    is not.
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
            "deletes a published branch or tag",
            subcommand="push",
            flags=frozenset({"--delete", "-d"}),
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

The table is a starting point and does not claim to be complete. It holds what
the list in the port plan named, plus the neighbors that have the same shape.
The preset does not gate a command that is missing from this table. The `when:`
patterns and the `interrupt_on` entries of a deployment are for that case.

**`rm` gates on `-r` only, which is narrower than what it replaces.** The regex
that this supersedes wanted `-r` *or* `-f`, and a first draft here gated `rm`
with no flag at all. Both are too wide, for the same reason. An agent that
removes a build artifact or a temp file is doing routine work. A gate that
prompts on routine work is a gate that a person learns to approve without
reading, and the `hitl` module calls that worse than no gate. What remains is
the case that a person really wants to see before it happens: a **tree** is
removed.

The cost is stated and not hidden. `rm -f one-file` and a bare `rm one-file` are
now ungated, so a single irreversible deletion passes without a prompt. That is
the deliberate trade. Recursion is the multiplier, and this list is tuned for
what is worth an interruption of a person, and not for everything that cannot be
undone. A deployment that wants the wider net writes `rm` into `interrupt_on`
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
        # Unbalanced quoting. A shell would refuse it too. But if the gate
        # refused to look, an appended quote would evade it. So fall back to a
        # plain split: the words are still there, and only the grouping is lost.
        return line.split()


def _has_flag(argv: Sequence[str], flag: str) -> bool:
    """Whether `argv` carries `flag`, in any of the forms that a shell allows.

    `--force` matches `--force` and `--force=x`. A single-character `-i` matches
    `-i`, `-i.bak` (an argument attached to the flag) and `-ri` (a bundle). All
    three are the same flag to the command that reads them, and `sed -ri`
    defeats a classifier that sees only the first.
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


_WRAPPERS = frozenset(
    {
        "sudo",
        "doas",
        "env",
        "nohup",
        "nice",
        "command",
        "exec",
        "time",
        "stdbuf",
        "setsid",
        "timeout",
    }
)
"""Commands whose *argument* is the command that matters (D3).

The gate read `argv[0]` and nothing else, so one word in front of any command
was a bypass. `sudo rm -rf /` was not an `rm` rule to it, and neither was a
command behind `env`, `nohup`, `nice`, `exec` or a leading `FOO=1`. This is the
human-in-the-loop gate, and one word in front of a command defeated it. That is
the complete hole.

`timeout` and `nice` take an argument of their own before the command. For that
reason unwrapping scans past the tokens that it does not recognize, after any
wrapper, and does not stop at the first one."""

_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _basename(token: str) -> str:
    """A command as its bare name: `/usr/bin/rm` is `rm`.

    It has a name because six places here ask this question. A bare
    `.rsplit("/", 1)[-1]` reads as string manipulation and not as the question.
    """
    return token.rsplit("/", 1)[-1]


def _rules_for(name: str) -> tuple[ShellRule, ...] | None:
    """The rules that a command name breaks, if any. `mkfs.ext4` is `mkfs`.

    There is one lookup because the two callers had already diverged.
    `_interesting` treated *any* dotted head as a possible command, while
    `_command_findings` unfolded only `mkfs.`. `sudo git.foo` therefore
    unwrapped to a command about which the rules had nothing to say.
    """
    return SHELL_RULES.get("mkfs" if name.startswith("mkfs.") else name)


def _interesting(token: str) -> bool:
    """Whether this token names a command any reader here has something to say about."""
    name = _basename(token)
    return _rules_for(name) is not None or name in _WRAPPERS or name in _SHELLS or name == "xargs"


def _unwrapped(argv: list[str]) -> list[str]:
    """The command that a wrapper wraps: `sudo -u x env FOO=1 rm -rf` → `rm -rf`.

    **It is found by a scan for a command that the tables know**, and not by a
    count of the flags of a wrapper and the values that they take. That count
    needs a table, for each wrapper, of the flags that carry an argument:
    `sudo -u nobody`, `nice -n 5`, `timeout -s KILL`. Such a table is wrong on
    the day that a wrapper gains an option. A scan cannot be wrong in the
    dangerous direction. An argument that happens to spell a command name
    reports something that a person then reads, while a flag table that missed
    one reports nothing at all.

    The result is `[]` when nothing recognizable follows. That is the correct
    answer, because there is no rule here to break.
    """
    while argv:
        if _ASSIGNMENT.match(argv[0]):
            argv = argv[1:]
            continue
        if _basename(argv[0]) not in _WRAPPERS:
            return argv
        rest = argv[1:]
        found = next((index for index, token in enumerate(rest) if _interesting(token)), None)
        if found is None:
            return []
        argv = rest[found:]
    return argv


def _nested(argv: list[str]) -> Iterator[list[str]]:
    """Commands carried *inside* the arguments of this command (D3).

    There are two shapes, and the gate read both as one opaque word. In
    `sh -c "rm -rf x"` the command is a quoted string. In `xargs rm -rf` the
    command is the rest of the argv. Neither is exotic. The first is how a model
    writes a command that needs a shell, and the second is how it writes one
    that needs a list.
    """
    head = _basename(argv[0])
    if head in _SHELLS:
        for index, token in enumerate(argv[1:], start=1):
            if token == "-c" and index + 1 < len(argv):
                yield from _simple_commands(argv[index + 1])
                return
    if head == "xargs":
        rest = argv[1:]
        while rest and rest[0].startswith("-"):
            # `xargs -0 -n1 rm -rf`: the flags of `xargs` itself, then the whole
            # command. To filter every dashed token instead would pass `rm`
            # alone to the rules, and `rm` without `-rf` breaks none of them.
            rest = rest[1:]
        if rest:
            yield rest


def _simple_commands(text: str) -> Iterator[list[str]]:
    """Each simple command in the text, split on newlines and shell operators.

    These are the members of each pipeline, ungrouped. `|` is one of
    `_OPERATORS`, so this is exactly `_pipelines` flattened. It is written that
    way and not as a second scanner. Two loops that must agree about how a line
    tokenizes are two places to edit when `;;` is added or a bare `&` changes
    meaning.
    """
    for group in _pipelines(text):
        yield from group


def _shell_findings(text: str) -> list[Finding]:
    # Tokenized one time. The groups answer both questions: what each command
    # does, and what one command feeds into.
    groups = list(_pipelines(text))
    found: list[Finding] = []
    for group in groups:
        for argv in group:
            found.extend(_command_findings(argv))
    found.extend(_pipeline_findings(groups))
    return found


def _command_findings(argv: list[str]) -> Iterator[Finding]:
    """Every rule that this one command breaks: the command, not its wrapper (D3).

    **Unwrapping decides which *name* to look up, and never whether the argv is
    read.** For a while it decided both, and that opened a hole worse than the
    one it closed. `_unwrapped` answers `[]` when nothing after the wrapper is
    in a table, so `sudo cat x.iso > /dev/sda` returned before the redirect
    reader ran and reported nothing, while the bare `cat x.iso > /dev/sda`
    reported. A gate that catches the plain form and not the `sudo` form has the
    worst shape possible, because `sudo` is what a model writes when it expects
    resistance.
    """
    inner = _unwrapped(argv)
    for nested in _nested(inner or argv):
        # Reported as itself. A person who approves `sh -c "rm -rf /"` is asked
        # about the `rm`, and to name the `sh` would hide it.
        yield from _command_findings(nested)
    if inner:
        # Built one time. The rule loop reads the first of them as a subcommand,
        # and the refspec reader reads the rest of them as refspecs.
        positional = [token for token in inner[1:] if not token.startswith("-")]
        yield from _refspec_findings(inner, positional)
        rules = _rules_for(_basename(inner[0]))
        if rules is not None:
            subcommand = positional[0] if positional else ""
            for rule in rules:
                if rule.subcommand and rule.subcommand != subcommand:
                    continue
                if rule.flags and not any(_has_flag(inner, flag) for flag in rule.flags):
                    continue
                if not rule.always and not rule.flags and not rule.subcommand:
                    continue
                yield Finding("shell", " ".join(argv[:3]), rule.reason)
                break
    # The shape readers run over the *original* argv, whatever the name lookup
    # made of it. A redirect belongs to the command line and not to the command,
    # and `sudo` does not consume it.
    for index, token in enumerate(argv):
        if token in _REDIRECTS and index + 1 < len(argv):
            target = argv[index + 1]
            if _RAW_DEVICES.match(target):
                yield Finding("shell", f"{token} {target}", "writes directly to a block device")


def _refspec_findings(argv: list[str], positional: list[str]) -> Iterator[Finding]:
    """The two destructive pushes that are written as *arguments* (D14).

    `git push origin :branch` deletes the remote branch, because it pushes
    nothing to it. `git push origin +main` forces. Neither carries a flag, so
    the rule table cannot see them. The table matches subcommands and flags,
    which is the correct shape for almost everything and the wrong shape for a
    refspec. They are read here for the same reason as the raw-device redirect
    in `_command_findings`: the danger is in the shape of the argument.
    """
    if _basename(argv[0]) != "git" or not positional or positional[0] != "push":
        return
    for token in positional[1:]:
        if token.startswith(":"):
            yield Finding("shell", f"git push {token}", "deletes a published branch or tag")
        elif token.startswith("+") and len(token) > 1:
            # **The `+` is the force, not the colon** (D14). `+main` is the
            # short form of `+main:main`. The first version asked for a colon,
            # so the form that this docstring names was the one form that
            # passed. A remote name cannot start with `+`, so this catches
            # nothing else that a push takes positionally.
            yield Finding("shell", f"git push {token}", "rewrites published history")


def _pipelines(text: str) -> Iterator[list[list[str]]]:
    """Each run of commands connected by `|`, as a group of its own.

    `_simple_commands` splits on every operator, `|` included. That is correct
    for the question of what each command does, and wrong for the question of
    what one command *feeds into*.
    """
    for line in text.splitlines():
        if not line.strip():
            continue
        group: list[list[str]] = []
        current: list[str] = []
        for token in _tokenize(line):
            if token == "|":
                if current:
                    group.append(current)
                current = []
            elif token in _OPERATORS:
                if current:
                    group.append(current)
                if group:
                    yield group
                group, current = [], []
            else:
                current.append(token)
        if current:
            group.append(current)
        if group:
            yield group


def _pipeline_findings(groups: Sequence[Sequence[list[str]]]) -> Iterator[Finding]:
    """A network fetch piped into a shell: the shape that no single command shows.

    **For each pipeline, not for each text** (D14). The test was "does this text
    contain a `|`, a fetcher anywhere, and a shell anywhere". A script that
    downloaded a file on one line, counted something with `wc` on another line
    and ran an unrelated `bash` on a third was therefore reported as piping the
    network into a shell. A false finding on the human-in-the-loop gate has a
    cost: it teaches a person to approve without reading.
    """
    for group in groups:
        if len(group) < 2:
            continue
        # Unwrapped, for the same reason as in `_unwrapped`: `curl x | sudo bash`
        # is the shape that this exists to catch, and a reader that took `sudo`
        # as the command name missed it.
        names = [_basename(inner[0]) for argv in group if (inner := _unwrapped(argv))]
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
"""Leading statement keyword → what it does. The match is case-insensitive. The
parser has already decided that this is SQL, and SQL keywords have no fixed
case."""

_SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_SQL_LITERAL = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_SQL_LEAD = re.compile(r"^\s*([A-Za-z]+)(?=\s|$)")
r"""The leading keyword: a **whole word**, not an alphabetic prefix (D2).

`^\s*([A-Za-z]+)` took the letters from the front of anything, so `create_app()`
led with `CREATE` and `select_all()` with `SELECT`. Both were read as SQL, and
the Python that followed them (`shutil.rmtree(x)`, `os.remove(x)`) was never
seen by a reader that knows what those calls do. The same prefix read made
`update_config()` produce a confident SQL finding about a function call.

A keyword is a word, so the match ends at whitespace or at the end of the
statement. The letters of `create_app` end at neither."""
_SQL_KEYWORDS = frozenset(SQL_STATEMENTS) | {"SELECT", "INSERT", "CREATE", "WITH", "REPLACE"}


def _sql_statements(text: str) -> Iterator[tuple[str, str]]:
    """`(keyword, statement)` for each statement, with comments and literals removed.

    Literals are removed before the split. A `;` or a `WHERE` inside a quoted
    string then cannot end a statement or make an unqualified `DELETE` look
    qualified.
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

This table is separate from the dotted table because the receiver usually cannot
be known. `Path("/etc/hosts").unlink()` builds its path in the call, and a table
that had to name the receiver would have to list every way to write one. The
cost is a false positive on the `unlink` of some other object. For an approval
gate, that is the affordable direction."""

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


def _imported(nodes: Sequence[ast.AST]) -> tuple[dict[str, str], dict[str, str]]:
    """What the names in this cell really refer to (D4).

    There are two maps: local module name → real module (`import os as o`), and
    local function name → dotted path (`from os import remove`). Both are
    ordinary ways for a cell to write an import. The gate matched the *written*
    name, so `o.remove(x)` and a bare `remove(x)` were unknown to it while
    `os.remove(x)` was caught. Anybody can step around such a rule by accident.

    The function takes the nodes that are already walked, and not the tree. It
    runs inside the `tools/pre-execute` gate, on the event loop, and on every
    string that a call carries. A `write` passes the whole file content to it as
    one leaf. Its own `ast.walk` was a second full traversal of what the caller
    was about to walk anyway. That was a third of the cost of the reader on a
    14 KB cell, for no benefit.
    """
    modules: dict[str, str] = {}
    names: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return modules, names


def _resolved(name: str, modules: dict[str, str], names: dict[str, str]) -> str:
    """A written call target as the dotted path it names."""
    if name in names:
        return names[name]
    head, _, rest = name.partition(".")
    return f"{modules[head]}.{rest}" if rest and head in modules else name


def _python_findings(text: str) -> list[Finding]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # A cell that does not parse does not run either, so there is nothing to
        # gate. Report nothing, and do not fall through to another reader. That
        # reader would tokenize Python as shell and invent findings.
        return []
    nodes = list(ast.walk(tree))
    modules, names = _imported(nodes)
    found: list[Finding] = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        written = _dotted(node.func)
        name = _resolved(written, modules, names)
        reason = PYTHON_CALLS.get(name) or PYTHON_METHODS.get(name.rsplit(".", 1)[-1])
        if reason is not None:
            found.append(Finding("python", f"{name}(...)", reason))
        if name in _SHELL_ESCAPES or name.rsplit(".", 1)[-1] in {"system", "popen"}:
            found.extend(_escaped_shell(node))
    return found


def _escaped_shell(node: ast.Call) -> Iterator[Finding]:
    """A Python call that passes a command to a shell. Read the command as shell."""
    if not node.args:
        return
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        yield from _shell_findings(first.value)
    elif isinstance(first, ast.JoinedStr):
        # An f-string: the literal parts are the command, and the interpolations
        # are its arguments. `subprocess.run(f"rm -rf {d}", shell=True)` is how
        # a cell writes a parameterized command. A reader of `ast.Constant` only
        # left the most ordinary form of a shell escape ungated.
        literal = "".join(
            part.value
            for part in first.values
            if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
        if literal.strip():
            yield from _shell_findings(literal)
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

    The order follows how *specific* the evidence is, because the readers
    disagree about ambiguous input. `rm -rf /tmp` is, unhelpfully, a valid
    Python expression (`rm - rf / tmp`). Python is therefore claimed only when
    the tree contains something that a shell command could not produce: an
    import, a call, an assignment. SQL is claimed only when a statement keyword
    leads. Everything else is shell, which is the dialect of a bare command
    line.
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


def findings(value: JsonValue) -> list[Finding]:
    """Everything destructive in the arguments of one call, deduplicated in order.

    Each string leaf goes to the reader for its own dialect. A cell that shells
    out is therefore read as Python *and* its command is read as shell, and a
    tool that carries both a query and a path is read as each.
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
