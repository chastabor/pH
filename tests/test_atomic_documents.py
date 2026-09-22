"""A whole document another process reads is not written by truncating it (O2).

`write_text_under(path, text)` opens with `"w"`, which empties the file and then
fills it. For a log nobody else reads that is fine and it is why the function
still exists. For a *document* — a JSON object, a YAML mapping, anything a
reader parses as a unit — it is a window in which the file is neither the old one
nor the new one, and a reader that lands in it does not get a stale answer, it
gets an unparseable one: an empty trust file is an untrusted root, half a
settings document is the default settings.

**A gate rather than a convention**, for `write_atomic`'s own reason. L7 selected
its seven sites by "already hand-rolled a temp-and-rename", not by "another
process parses this", so five whole-document writers were left on the truncating
path and nothing said so — the module offered two writers whose only difference a
caller had to know to choose.

The rule is about the *shape of the payload*, which a gate cannot see, so it is
enforced the way it can be: production code does not name `write_text_under`
without an explicit `append=True`. A caller that genuinely wants truncation says
so here.

**Both ways of naming it count**, which is the correction that found the last two
offenders. The first version of this gate matched only an `ast.Call` whose callee
was `write_text_under`, and every remaining truncating writer reached it as a
*value* — `anyio.to_thread.run_sync(write_text_under, path, text)`, which is the
calling convention `write_text_under`'s own docstring prescribes for async code.
The gate was blind to the usage its own module recommends, so it passed while
`$PH_HOME/settings.json` and the sandbox profile drop-in kept tearing. A green
gate read as evidence they did not is worse than no gate, so the reference is
what is matched, and `append=True` is looked for on whichever call carries it —
the direct one, or the `partial`/`run_sync` that will make it.

@module tests.test_atomic_documents
"""

from __future__ import annotations

import ast
from typing import TypeGuard

from workspace_layout import workspace_modules

WRITER = "write_text_under"

ALLOWED: frozenset[tuple[str, str]] = frozenset()
"""Sites that truncate on purpose, keyed by module and the source of the call.

Empty today. An entry here is a claim that nothing outside this process parses
the file as a unit — not that the write is small, and not that it has never been
seen to tear.

Keyed on the source text rather than a line number, following
`test_json_narrowing.py`: a line-number key stops applying the moment anything
above it is edited, which silently re-flags the site it exempted or, worse,
exempts a different one.
"""


def _appends(call: ast.Call) -> bool:
    """Whether `call` passes a literal `append=True`.

    `is True` rather than "an `ast.Constant` is present", because `append=False`
    satisfied the looser test while asking for exactly the truncation this
    forbids.
    """
    return any(
        keyword.arg == "append"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in call.keywords
    )


def _named(node: ast.AST) -> TypeGuard[ast.Name | ast.Attribute]:
    """Whether `node` is a reference to `write_text_under`, imported either way."""
    if isinstance(node, ast.Attribute):
        return node.attr == WRITER
    return isinstance(node, ast.Name) and node.id == WRITER


def _truncating_writes(tree: ast.Module, source: str) -> list[tuple[int, str]]:
    """Every reference to the writer that does not carry `append=True`.

    A reference is cleared by the nearest enclosing call that names it — as the
    callee for a direct call, or as an argument for the deferred forms — so
    `partial(write_text_under, path, text, append=True)` reads as the append it
    is rather than as a bare mention.
    """
    cleared: set[int] = set()
    enclosing: dict[int, ast.Call] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for candidate in (node.func, *node.args):
            if not _named(candidate):
                continue
            enclosing[id(candidate)] = node
            if _appends(node):
                cleared.add(id(candidate))

    # Every reference, not only the ones a call encloses: a bare
    # `handler = write_text_under` followed by `handler(path, text)` is the one
    # spelling neither form above catches, and a gate that has already been
    # wrong once about which spellings count should not guess again.
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not _named(node) or id(node) in cleared:
            continue
        reported = enclosing.get(id(node), node)
        segment = ast.get_source_segment(source, reported) or ast.dump(reported)
        found.append((node.lineno, " ".join(segment.split())))
    return sorted(found)


def test_no_production_module_writes_a_document_by_truncating_it() -> None:
    """The gate. See this module's docstring for what it is protecting.

    Source only: a test may write a scratch file however it likes, and holding
    fixtures to a durability rule about other processes would be noise.

    Sabotage: point `settings.LocalSettings.set` back at `write_text_under` and
    this names the line — which is the case the first version of the gate
    missed, because that site passes the function rather than calling it.
    """
    offenders: list[str] = []
    for module, path in workspace_modules():
        source = path.read_text(encoding="utf-8")
        if WRITER not in source:
            continue
        tree = ast.parse(source, filename=module)
        offenders.extend(
            f"{module}:{line} — {segment}"
            for line, segment in _truncating_writes(tree, source)
            if (module, segment) not in ALLOWED
        )

    assert offenders == [], (
        "these truncate a file another process may be parsing; call `ph.paths.write_atomic`, "
        f"or add the site to `ALLOWED` with the reason it is not a document: {offenders}"
    )
