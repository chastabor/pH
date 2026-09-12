"""A builtin is not a narrowing, and this is the gate that says so.

`SessionEvent.data` is a `JsonObject`, so every field a reader takes out of it is
a `JsonValue` — maybe a string, maybe a number, maybe absent. `str(...)` accepts
all of those and answers for all of them, which is the problem: `str(None)` is
`"None"` and `str(3)` is `"3"`, so a field that is missing or of the wrong type
comes back looking exactly like a field that was there. Nothing raises, nothing
logs, and the wrong value is a plausible one — a card titled `None`, a dict keyed
`"None"`, a reason line reading `None`.

The narrowing family says the true thing instead: not a string, so nothing.
`as_str` is one of five — `as_bool`, `as_int`, `as_obj`, `as_seq` — whose shared
policy `ph.json` documents: **a mis-shaped field costs a row, not a
raise, and never a fabricated value.** `str()` is the example above because it is
the loudest; `COERCIONS` below has what each of the others gets wrong.

**Two spellings, one defect.** The rule is not "no `str()`" — it is "a JSON read
must not go through a builtin that guesses", and `.get` is only the commoner way
to spell the read. The subscript form was found by this gate's own first version
passing while the tree still fabricated:

* `str(data.get("k"))` — the one everybody thinks of;
* `str(data["k"])` — the same read through a subscript, and `revert.py` had one
  five lines from a converted sibling, same dict, same payload.

**Why a gate rather than a convention.** 151 readers were converted in one pass,
and the reason there were 151 is that nothing stopped the 2nd through the 151st.
`str(x.get(...))` is legal Python on every type, so no linter has a rule for it
and mypy is *satisfied* by it — `str()` is exactly how a reader silences the
checker without answering its question. A ruff rule cannot be configured for
this; an AST walk can, and this is it.

This is the same shape as `test_layering.py` (ph-core may not import Textual) and
`packages/ph-app/tests/test_tui_screens.py` (the terminal may only touch what
`FrontSession` declares): a rule that is cheap to state, expensive to discover by
hand, and silent when broken.

**What this gate does not cover, and why.** One edge, named so the next reader
knows it rather than trusting a line this does not hold.

*f-strings.* `f"{data.get('k')}"` **is** this defect — interpolation calls
`str()` — and it was gated here for about an hour. It came back out because the
gate cannot tell the two apart: a JSON read and a `TypedDict` read are the same
syntax, and without types 154 of the 177 it flagged were correct code formatting
a number it already knew was a number (`ph_code_graph` and `ph_text_index`'s CLI
output, 69 between them). A gate that has to carry 154 exemptions is a list, not
a rule. The honest fix is upstream — a typed payload per event type, parsed once
at the fold boundary — which is the row this gate is standing in for.

@module tests.test_json_narrowing
"""

from __future__ import annotations

import ast
from functools import cache

from workspace_layout import workspace_modules

COERCIONS = {"bool": "as_bool", "int": "as_int", "list": "as_seq", "str": "as_str"}
"""Builtin → the narrowing that should have been called instead.

A table because the family has five members and each one's builtin fails a JSON
read differently:

* `str(None)` is `"None"` — a **fabricated** value that looks like an answer;
* `bool("false")` is `True` — the **opposite** of what the log says;
* `int(None)` **raises**, which on a resume path turns one unreadable row into a
  failed session;
* `list("abc")` is `["a", "b", "c"]` — a string is a `Sequence`, so a field that
  should have been an array silently becomes its own characters.

One rule, four symptoms. `float` has no member and no sites."""


ALLOWED: frozenset[tuple[str, str]] = frozenset(
    {
        # A page's cards are keyed by seq **as a string**, and seq is an int; the
        # next line reads the same field with `as_int`.
        ("ph_app/daemon/follow.py", 'str(one.get("seq"))'),
        # JSON-RPC says an id is "a String or a Number".
        ("ph_app/daemon/duplex.py", 'str(frame.get("id"))'),
        # A row's `id:` is YAML, where `id: 3` is an int and `"3"` is the answer.
        ("ph/cordis/loader.py", 'str(entry.get("id") or entry["name"])'),
        # Tool arguments come from the model and are *rendered*: a numeric
        # argument should show as the number, and `arguments` is often a whole
        # object whose repr is the thing being shown. `simple_views` is the
        # sharpest case — two of its thirteen call sites key on a `list[str]`
        # (`code_index` and `text_index`, both over `paths`), and narrowing
        # blanked the card's line for them.
        ("ph/tools/presentation.py", 'str(args.get(key, ""))'),
        ("ph_rlm/presentation.py", 'str(args.get("program", ""))'),
        ("ph/tools/builtin/ask_user.py", 'str(args.get("question", ""))'),
        ("ph/tools/builtin/bash_tool.py", 'str(args.get("command", ""))'),
        ("ph/tools/builtin/subagent_task.py", 'str(args.get("prompt", ""))'),
        # A row's `name:` is YAML beside the `id:` two lines above it, and the
        # same argument applies: `name: 3` is an int and `"3"` is the answer.
        ("ph/cordis/loader.py", 'str(entry["name"])'),
        # Counters on their way into a CLI table. These are `int`s and the whole
        # point is their text — the gate cannot see that, because a JSON read and
        # a typed read are the same syntax, which is the edge the docstring names.
        ("ph_code_graph/__init__.py", 'str(stats["files"])'),
        ("ph_code_graph/__init__.py", 'str(stats["symbols"])'),
        ("ph_code_graph/__init__.py", 'str(stats["refs"])'),
        ("ph_text_index/__init__.py", 'str(stats["documents"])'),
        ("ph_text_index/__init__.py", 'str(stats["chunks"])'),
        ("ph_text_index/__init__.py", 'str(stats["calibration"])'),
        # A `sqlite3.Row`, not a payload: the column is whatever SQLite stored.
        ("ph_code_graph/_store.py", 'str(row["value"])'),
        ("ph_code_graph/_store.py", 'int(row["value"])'),
        # Truthiness, and that is the question being asked. `as_bool`'s own
        # docstring keeps these three: a Python `dict[str, bool]` with a missing
        # key, "is anything queued" over two lists, and "is this environment
        # variable set and non-empty" — none is reading a JSON boolean.
        ("ph_app/tui/remote.py", "bool(self.held.get(name))"),
        ("ph/agent/inbox.py", 'bool(self._state["next-turn"] or self._state["next-step"])'),
        ("ph/cordis/loader.py", "bool(source.get(target))"),
        # Not JSON either: a typed stats mapping and the file descriptor the guest
        # is handed in its environment.
        ("ph_text_index/__init__.py", 'int(store.stats()["chunks"])'),
        ("ph_runtime/channel.py", "int(os.environ.get(FD_ENV, PROTOCOL_FD))"),
        # **The five boot limits, where a quiet default is the dangerous answer.**
        # `protocol.py` states the rule these keep: "there is exactly one owner of
        # every default: the host. `boot` carries every limit as a required field,
        # so a guest has nothing to guess." `limits.py` reads `0` as *no limit*,
        # so narrowing a malformed frame to `0` boots the guest unsandboxed —
        # where `int()` raises and the guest refuses to start, which is what
        # `_serve` already says it wants: "a guest that misreads one frame at a
        # time is worse than a guest that will not start."
        ("ph_runtime/runner.py", 'int(boot["maxLogBytes"])'),
        ("ph_runtime/runner.py", 'int(boot["maxValueBytes"])'),
        ("ph_runtime/runner.py", 'int(boot["maxSnapshotBytes"])'),
        ("ph_runtime/runner.py", 'int(boot["cpuSeconds"])'),
        ("ph_runtime/runner.py", 'int(boot["addressSpaceBytes"])'),
    }
)
"""The sites where the builtin is the right answer, each with why.

Three kinds, and the reasons differ. **A value that is genuinely not of that type
and whose coercion is wanted anyway** — an integer seq used as a string key, a
JSON-RPC id, a YAML scalar, a model's tool argument on its way to a card.
**A read that is not JSON at all** — a `sqlite3.Row`, a typed stats mapping, an
environment variable. And **a read where the quiet default is the dangerous
answer**, which is the guest's five boot limits: there,
raising is the policy.

`(module path, source text)` rather than a line number, so an entry survives the
code around it moving and dies the moment the expression itself changes. Both
halves are load-bearing: `tui/trajectory.py` and `agents.py` hold the *same*
expression, differing only in the quote style an enclosing f-string forces.

**Fewer entries than sites** — `ask_user`, `bash_tool` and `trajectory` each
hold their expression twice, and that is deliberate: what is being exempted is
the *expression*, not each place somebody wrote it."""


def _reads_a_key(node: ast.AST) -> bool:
    """Whether this expression pulls a value out of a mapping.

    Two spellings, because a payload is read both ways and the defect does not
    care which: `data.get("k")` and `data["k"]`. The `.get` form may be nested
    anywhere inside — `str(d.get("a") or d.get("b") or "?")` is a third of the
    real sites and a shallower test would miss every one.
    """
    for inner in ast.walk(node):
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "get"
        ):
            return True
        if (
            isinstance(inner, ast.Subscript)
            and isinstance(inner.slice, ast.Constant)
            and isinstance(inner.slice.value, str)
        ):
            return True
    return False


def _coercions(tree: ast.AST, source: str) -> list[tuple[int, str]]:
    """Every fabrication of a JSON read in one module: `(line, source text)`."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in COERCIONS
            and len(node.args) == 1
            and _reads_a_key(node.args[0])
        ):
            found.append((node.lineno, ast.get_source_segment(source, node) or ""))
    return found


@cache
def _sites() -> tuple[tuple[str, int, str], ...]:
    """Every coercion in the workspace, found once.

    Cached because both tests below ask the same question of the same 251
    modules, and parsing them twice was a third of this directory's whole run.
    The *triples* are held rather than the parsed trees: tens of kilobytes
    against ~80 MB pinned for the rest of the pytest process.
    """
    found: list[tuple[str, int, str]] = []
    for name, path in workspace_modules():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        found.extend((name, line, text) for line, text in _coercions(tree, source))
    return tuple(found)


def test_no_reader_fabricates_a_value_from_a_json_field() -> None:
    """The narrowing that fits, or an entry in `ALLOWED` saying why it does not."""
    offenders = [
        f"{name}:{line}: {text} — call {COERCIONS[text.split('(', 1)[0]]}"
        for name, line, text in _sites()
        if (name, text) not in ALLOWED
    ]
    assert offenders == [], (
        "these read a JSON field and hand it to a builtin that guesses — see "
        "`COERCIONS` for what each one gets wrong. Call the narrowing named on "
        "each line, or add the site to `ALLOWED` with the reason the value is "
        "legitimately not of that type:\n  " + "\n  ".join(offenders)
    )


def test_every_allowed_site_still_exists() -> None:
    """An exemption that no longer matches anything is an exemption nobody reads.

    The list above is the argument for every deliberate coercion. When one is
    rewritten or deleted, its entry stops describing the tree — and a stale
    exemption is how the next real coercion slips in wearing an old name. This
    caught one the hour it was written: a sweep had converted one of four
    sibling tool-argument sites and left the other three.
    """
    live = {(name, text) for name, _line, text in _sites()}
    assert ALLOWED - live == set(), (
        f"these `ALLOWED` entries match nothing in the tree any more: {sorted(ALLOWED - live)}"
    )
