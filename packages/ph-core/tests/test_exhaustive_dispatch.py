"""Every dispatch over a tagged union names every variant, or fails here.

`mypy --strict` does not flag a `match` with no `case _`, nor an `if/elif` chain
with no `else` — so a terminal `assert_never` protects exactly the functions that
already have one. Placing them by hand is the failure `declarable_fields`'
docstring names: a list is the thing that drifts, and the next renderer written
is caught by nothing.

So the rule is enforced over the *tree* rather than per site: a module that takes
apart one of these unions has to close it. The member sets are read off the
aliases themselves, so this gate cannot drift from the declaration either.

**What it does not reach.** Half the renderers here read the *wire* form, not the
models — `ph_app.wire.text_of_wire` and `ph_app.tui.trajectory` dispatch on a
`block.get("type")` string, where no static mechanism applies. A variant needing
its own rendering falls into their generic bucket silently. That gap is the
argument for keeping the model-side gate tight, not a reason to widen this test
into something that guesses at strings.
"""

from __future__ import annotations

import ast
import pathlib
import typing

import ph
import ph.llm.types
import ph.tools.definition

_MIN_ARMS = 3
"""Naming this many members is a dispatch; fewer is a filter or a `next(...)`.

A `[b for b in blocks if isinstance(b, ToolCallBlock)]` asks one question and is
not obliged to answer for the rest of the union. Two members is still a question
(`isinstance(x, (TextDelta, ReasoningDelta))`); three is a chain.
"""

_EXEMPT = {
    "ph/llm/types.py": "declares the unions; `is_token_delta` asks which three carry text",
}
"""Modules that discriminate without owing an answer for the rest of a union.

One entry, and it is the declaration site. Keep it that way: every addition is a
place the guarantee stops, and the reason has to survive a reader asking why.
"""


def _members(alias: object, module: object) -> set[str]:
    """The variant names of a union alias, read off the alias."""

    def probe(x: int) -> None: ...

    probe.__annotations__["x"] = alias
    resolved = typing.get_type_hints(probe, vars(module), include_extras=False)["x"]
    if typing.get_origin(resolved) is typing.Annotated:
        resolved = typing.get_args(resolved)[0]
    return {one.__name__ for one in typing.get_args(resolved)}


def _gated_unions() -> dict[str, set[str]]:
    types_, definition = ph.llm.types, ph.tools.definition
    return {
        "ContentBlock": _members(types_.ContentBlock, types_),
        "StreamChunk": _members(types_.StreamChunk, types_),
        "PreToolDecision": _members(definition.PreToolDecision, definition),
        "PostToolDecision": _members(definition.PostToolDecision, definition),
    }


def _closes(tree: ast.Module) -> bool:
    """Whether the module closes a dispatch — `assert_never`, or a `Never` helper.

    A local `def _refuse(x: Never) -> NoReturn` carries `assert_never`'s static
    check in its parameter while raising what the runtime case deserves, so a
    call to one counts.
    """
    refusers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for arg in node.args.args
        if isinstance(arg.annotation, ast.Name) and arg.annotation.id == "Never"
    }
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and (node.func.id == "assert_never" or node.func.id in refusers)
        for node in ast.walk(tree)
    )


def _discriminated(tree: ast.Module) -> set[str]:
    """Variant names the module *asks about*, not ones it merely builds.

    Only an `isinstance(x, Variant)` test or a `case Variant()` pattern counts.
    A module that constructs chunks — `ph.llm.fake`, `ph.llm.replay` — names
    every member of the union and dispatches on none of them, and counting bare
    `Name` nodes called both of those a chain.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.MatchClass) and isinstance(node.cls, ast.Name):
            found.add(node.cls.id)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
            and len(node.args) == 2
        ):
            tested = node.args[1]
            for one in tested.elts if isinstance(tested, ast.Tuple) else [tested]:
                if isinstance(one, ast.Name):
                    found.add(one.id)
    return found


def _modules() -> list[tuple[str, pathlib.Path]]:
    """Every shipped module in every package, as `(import-ish path, file)`.

    Rooted off `ph.__path__` the way `test_layering._core_modules` is, so it
    follows an installed layout rather than assuming a checkout shape.
    """
    packages = pathlib.Path(ph.__path__[0]).parents[2]
    found = [
        (str(path.relative_to(source)), path)
        for source in packages.glob("*/src")
        for path in source.rglob("*.py")
    ]
    assert found, f"no modules found under {packages}; the gate would pass vacuously"
    return sorted(found)


def test_a_module_that_takes_a_tagged_union_apart_closes_it() -> None:
    """Sabotage: delete the `case _ as unhandled` arm from any converted
    dispatch, or write a new one without it, and this names the module."""
    unions = _gated_unions()
    open_chains = []
    for rel, path in _modules():
        if rel in _EXEMPT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        named = _discriminated(tree)
        for union, members in unions.items():
            if len(named & members) >= _MIN_ARMS and not _closes(tree):
                open_chains.append(f"{rel}: names {len(named & members)} of {union}")
    assert not open_chains, "dispatch with no terminal arm:\n" + "\n".join(sorted(open_chains))
