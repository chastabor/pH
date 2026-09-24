"""P10-01 — every append in every package names a known type and a writer of record.

`Session.append` refuses a type the read door would refuse (F11), so what is left for
a gate is the question the runtime cannot answer: **who may write each type**. This
walks every shipped module in every package — not ph-core alone, and not literals
alone — and holds each `session.append(<type>, …)` site against
`known_event_types.WRITERS`.

## What counts as an append site

A call `<receiver>.append(first, second, …)` with **at least two** positional
arguments. `list.append` takes one, and `Session.append` always takes a payload, so the
arity is the discriminator — which is also why a receiver's *name* is not asked: a
session is called `session`, `parent`, `revived`, `created` and `source` in this tree.
The type is read from a literal, a module constant, or a constant imported by name
(`ph_rlm.subagents` appends `ph.seams.subagents.STATUS`). A value with no `/` in it (an
inbox target) is not a log type and is skipped.

## Why a site whose type is a variable is a listed decision

The walk cannot see through one. So such sites are named in `VARIABLE_SITES` with what
each may write, and a new one fails `test_no_new_append_takes_its_type_from_a_variable`
— a decision to make, rather than a type the table silently stops covering. Keyed by
the expression's text, so a refactor that changes it is asked about again.

## A pair written through the journal

`ctx.intents` appends `kind.opened` and `kind.settled` for whoever calls it, so those
two sites are listed as writing nothing of their own, and the module that
*declares* a kind — the `IntentKind(opened=…, settled=…)` call, read by the walk —
is the writer of both types. The declaration is the one place the pair is named.

## Why the table is lexical

The module whose code calls `append`, not the row that was running when it did:
`permission-presets` moves the sandbox posture by calling `SandboxSeam.set_mode`, and
the append is `ph.seams.sandbox`'s. That is also why P10-02's runtime check cannot be
the same table read at append time.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass

from workspace_layout import workspace_packages

from ph.session.known_event_types import KNOWN_SESSION_EVENT_TYPES, WRITERS

NOT_A_LOG_APPEND: frozenset[str] = frozenset()
"""A two-argument `append` that is not `Session.append`."""

FOR_THE_DECLARER: frozenset[str] = frozenset()
"""The journal's appends: written for the module that declared the kind."""

SCAFFOLDING: frozenset[str] = KNOWN_SESSION_EVENT_TYPES
"""Test scaffolding that replays whatever types a test hands it."""

VARIABLE_SITES: dict[tuple[str, str], frozenset[str]] = {
    # `Inbox.append(target, message)`: an inbox target, not a log type.
    ("ph.agent_loop.driver", "'next-turn' if waking_after_abort else target"): NOT_A_LOG_APPEND,
    # One recorder for both media records, told which by its caller.
    ("ph.llm.media", "event_type"): frozenset({"attachment/degraded", "attachment/oversized"}),
    # `ctx.intents`, opening and settling a declared kind's pair.
    ("ph.session.journal", "kind.opened"): FOR_THE_DECLARER,
    ("ph.session.journal", "kind.settled"): FOR_THE_DECLARER,
    # `workspace_log` builds a session from hand-written events for a fold test.
    ("ph.testing.builders", "kind"): SCAFFOLDING,
}


@dataclass(frozen=True, slots=True)
class _Walk:
    """What the walk found: literal sites by type, and sites whose type is a variable."""

    static: dict[str, set[str]]
    """Type → the modules that append it with a type the walk could read."""
    variable: set[tuple[str, str]]
    """`(module, expression)` for every site whose type is not a constant."""
    declared: dict[str, str]
    """Types declared with `declare_log_type` in shipped code → their `owner=`."""


def _modules() -> dict[str, ast.Module]:
    found: dict[str, ast.Module] = {}
    for package in workspace_packages():
        for path in sorted(package.rglob("*.py")):
            parts = path.relative_to(package.parent).with_suffix("").parts
            name = ".".join(parts).removesuffix(".__init__")
            found[name] = ast.parse(path.read_text(encoding="utf-8"))
    return found


def _constants(tree: ast.Module) -> dict[str, str]:
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets: list[ast.expr] = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node.value.value
    return found


def _imports(module: str, tree: ast.Module, is_package: bool) -> dict[str, tuple[str, str]]:
    """Each name imported with `from … import`, mapped to `(module, original name)`."""
    names: dict[str, tuple[str, str]] = {}
    package = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        base = node.module or ""
        if node.level:
            parts = package.split(".")[: len(package.split(".")) - (node.level - 1)]
            base = ".".join([*parts, base]) if base else ".".join(parts)
        for alias in node.names:
            names[alias.asname or alias.name] = (base, alias.name)
    return names


def _walk() -> _Walk:
    modules = _modules()
    packages = {name for name in modules for other in modules if other.startswith(name + ".")}
    constants = {name: _constants(tree) for name, tree in modules.items()}
    imports = {name: _imports(name, tree, name in packages) for name, tree in modules.items()}

    def constant(module: str, name: str, depth: int = 0) -> str | None:
        if depth > 8 or module not in modules:
            return None
        if name in constants[module]:
            return constants[module][name]
        imported = imports[module].get(name)
        return None if imported is None else constant(*imported, depth + 1)

    def type_of(module: str, node: ast.expr) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return constant(module, node.id)
        return None

    static: dict[str, set[str]] = defaultdict(set)
    variable: set[tuple[str, str]] = set()
    declared: dict[str, str] = {}
    for module, tree in modules.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
            if name == "IntentKind":
                for keyword in node.keywords:
                    if keyword.arg in ("opened", "settled"):
                        kind = type_of(module, keyword.value)
                        if kind is None:
                            variable.add((module, ast.unparse(keyword.value)))
                        else:
                            static[kind].add(module)
            if name == "declare_log_type" and node.args:
                kind = type_of(module, node.args[0])
                owner = next((k.value for k in node.keywords if k.arg == "owner"), None)
                if kind is not None and isinstance(owner, ast.Constant):
                    declared[kind] = str(owner.value)
            if not (isinstance(callee, ast.Attribute) and name == "append" and len(node.args) >= 2):
                continue
            kind = type_of(module, node.args[0])
            if kind is None:
                variable.add((module, ast.unparse(node.args[0])))
            elif "/" in kind:
                static[kind].add(module)
    return _Walk(static=dict(static), variable=variable, declared=declared)


WALK = _walk()


def _writes() -> set[tuple[str, str]]:
    """Every `(module, type)` the code appends: what the walk read, plus what each
    listed variable site may write."""
    pairs = {(module, kind) for kind, modules in WALK.static.items() for module in modules}
    for (module, _expression), kinds in VARIABLE_SITES.items():
        if kinds is not SCAFFOLDING:
            pairs |= {(module, kind) for kind in kinds}
    return pairs


def test_the_walk_sees_every_package() -> None:
    """A walk that silently shrank would pass every gate below."""
    writers = {module.partition(".")[0] for modules in WALK.static.values() for module in modules}
    assert {"ph", "ph_app", "ph_rlm", "ph_stabilize"} <= writers, writers


def test_the_table_covers_the_vocabulary() -> None:
    assert set(WRITERS) == KNOWN_SESSION_EVENT_TYPES
    assert not [kind for kind, modules in WRITERS.items() if not modules]


def test_every_append_in_every_package_names_a_known_type() -> None:
    """The static half of F11: a type this build writes and would refuse to read."""
    unknown = set(WALK.static) - KNOWN_SESSION_EVENT_TYPES - set(WALK.declared)
    assert not unknown, {kind: sorted(WALK.static[kind]) for kind in unknown}


def test_every_append_site_is_a_writer_of_record() -> None:
    """Sabotage: append `sandbox/mode` from `ph_stabilize.hitl` and this names both."""
    owners = {kind: frozenset({owner}) for kind, owner in WALK.declared.items()}
    strays = sorted(
        (module, kind)
        for module, kind in _writes()
        if module not in WRITERS.get(kind, owners.get(kind, frozenset()))
    )
    assert not strays, strays


def test_every_writer_of_record_still_appends_the_type() -> None:
    """The reverse direction, so a writer that stopped writing is removed from the
    table rather than left to widen it."""
    writes = _writes()
    stale = sorted(
        (module, kind)
        for kind, modules in WRITERS.items()
        for module in modules
        if (module, kind) not in writes
    )
    assert not stale, stale


def test_no_new_append_takes_its_type_from_a_variable() -> None:
    assert WALK.variable == set(VARIABLE_SITES), sorted(WALK.variable ^ set(VARIABLE_SITES))


def test_a_declared_type_is_appended_only_by_its_owner() -> None:
    """None ships today (decision 7); held here so the first one is checked."""
    for kind, owner in WALK.declared.items():
        assert WALK.static.get(kind, set()) <= {owner}, (kind, owner, WALK.static.get(kind))


def test_the_walk_follows_an_imported_constant() -> None:
    """The resolution the table depends on, held on its own: these sites name a
    constant another module defines, never a literal."""
    assert "ph_rlm.subagents" in WALK.static["subagent/status"]
    assert "ph_app.daemon.supervisor" in WALK.static["supervisor/retry"]
