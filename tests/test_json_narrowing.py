"""`str()` is not a narrowing, and this is the gate that says so.

`SessionEvent.data` is a `JsonObject`, so every field a reader takes out of it is
a `JsonValue` — maybe a string, maybe a number, maybe absent. `str(...)` accepts
all of those and answers for all of them, which is the problem: `str(None)` is
`"None"` and `str(3)` is `"3"`, so a field that is missing or of the wrong type
comes back looking exactly like a field that was there. Nothing raises, nothing
logs, and the wrong value is a plausible one — a card titled `None`, a dict keyed
`"None"`, a reason line reading `None`.

`as_str` is the narrowing that says the true thing instead: not a string, so
nothing. It is the fourth of a family — `as_int`, `as_obj`, `as_seq` — whose
shared policy `ph.session.json` documents: **a mis-shaped field costs a row, not
a raise, and never a fabricated value.**

**Two spellings, one defect.** The rule is not "no `str()`" — it is "a JSON read
must not become a fabricated value", and `.get` is only the commoner way to spell
the read. The subscript form was found by this gate's own first version passing
while the tree still fabricated:

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

**What this gate does not cover, and why.** Two edges, named so the next reader
knows them rather than trusting a line this does not hold.

*Other members of the family.* `int(...)`, `bool(...)` and `list(...)` over a
JSON read are the same defect through the other narrowings — and `bool("false")`
is `True`. `as_int` exists and 26 readers still do not use it; `as_bool` does not
exist yet. Adding a row to `COERCIONS` is where that lands, and the walker is
already general over it.

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

COERCIONS = {"str": "as_str"}
"""Builtin → the narrowing that should have been called instead.

A table because the family has four members and this gate holds one of them. The
walker below is already general over it; `int`/`bool`/`list` join by adding a row
and fixing what it finds, which is a row's work rather than a rewrite."""


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
        # object whose repr is the thing being shown.
        ("ph/tools/presentation.py", 'str(args.get(key, ""))'),
        ("ph_rlm/presentation.py", 'str(args.get("program", ""))'),
        ("ph/tools/builtin/ask_user.py", 'str(args.get("question", ""))'),
        ("ph/tools/builtin/bash_tool.py", 'str(args.get("command", ""))'),
        ("ph/tools/builtin/subagent_task.py", 'str(args.get("prompt", ""))'),
        ("ph_app/tui/adapter.py", 'str(event.data.get("arguments", ""))'),
        ("ph_app/tui/trajectory.py", 'str(data.get("arguments") or "")'),
        ("ph_app/agents.py", "str(data.get('arguments') or '')"),
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
    }
)
"""The sites where a **non-string is stringified on purpose**, each with why.

Every entry is a value that genuinely may not be a string and whose textual form
is wanted anyway — an integer seq used as a key, a JSON-RPC id, a YAML scalar, a
model's tool argument on its way to a card. `as_str` would answer `""` for each
and lose the value.

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
    The *triples* are held rather than the parsed trees: 43 KB against 80 MB
    pinned for the rest of the pytest process.
    """
    found: list[tuple[str, int, str]] = []
    for name, path in workspace_modules():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        found.extend((name, line, text) for line, text in _coercions(tree, source))
    return tuple(found)


def test_no_reader_fabricates_a_value_from_a_json_field() -> None:
    """`as_str`, or an entry in `ALLOWED` saying why the value is not a string."""
    offenders = [
        f"{name}:{line}: {text}" for name, line, text in _sites() if (name, text) not in ALLOWED
    ]
    assert offenders == [], (
        "these read a JSON field and coerce it with `str()`, which turns a "
        "missing or mis-typed field into a plausible wrong value — `str(None)` "
        "is 'None'. Call `as_str` instead, or add the site to `ALLOWED` with the "
        "reason the value is legitimately not a string:\n  " + "\n  ".join(offenders)
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
