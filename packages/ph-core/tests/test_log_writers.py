"""P10-01, T6 — one door into every log, and a writer of record behind every write.

`Session` has no public `append` (T6): a module that writes a log mints its own writer,
`_LOG = log_writer(__name__)`, and the writer refuses a type its owner is not the
writer of record for (`known_event_types.WRITERS`). That is held twice:

* **statically**, here, over every shipped module in every package: every write is a
  writer's, every writer is minted by its own module at module top and used there
  only, and the table is exactly what the writers write — derived from them, in both
  directions, rather than trusted;
* **at runtime**, by the writer itself (F12): a row that writes a type it does not own
  is refused, a module cannot mint another's writer, and ph.testing's scaffolding
  writer is refused anywhere else.

## What counts as a write

A call `<writer>.append(log, type, data, …)` where `<writer>` is the module's own
binding of `log_writer(__name__)`, and an `IntentKind(opened=…, settled=…,
writer=<writer>)` declaration — whose pair the journal writes through that writer, so
the leaf that declares a kind is the writer of record of both its types. The type is
read from a literal, a module constant, or a constant imported by name
(`ph_rlm.subagents` writes `ph.seams.subagents.STATUS`).

## Why a site whose type is a variable is a listed decision

The walk cannot see through one. So such sites are named in `VARIABLE_SITES` with what
each may write, and a new one fails `test_no_new_write_takes_its_type_from_a_variable`
— a decision to make, rather than a type the table silently stops covering. Keyed by
the expression's text, so a refactor that changes it is asked about again.

## Why the table is lexical

The module whose code writes, not the row that was running when it did:
`permission-presets` moves the sandbox posture by calling `SandboxSeam.set_mode`, and
the write is `ph.seams.sandbox`'s. The runtime check is lexical for the same reason:
the writer is the module's, whoever calls into it.
"""

from __future__ import annotations

import ast
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from workspace_layout import import_base, parsed_modules

from ph.agent_loop.driver import _LOG as DRIVER
from ph.json import JsonObject
from ph.session import Session
from ph.session.known_event_types import (
    KNOWN_SESSION_EVENT_TYPES,
    WRITERS,
    declare_log_type,
)
from ph.session.writers import LogWriteError, LogWriter, log_writer, scaffolding_writer

VARIABLE_SITES: dict[tuple[str, str], frozenset[str]] = {
    # One recorder for both media records, told which by its caller.
    ("ph.llm.media", "event_type"): frozenset({"attachment/degraded", "attachment/oversized"}),
}

NOT_A_LOG_WRITE: frozenset[tuple[str, str]] = frozenset(
    {
        # `Inbox.append(target, message)`: an inbox target, not a log.
        ("ph.agent_loop.driver", "self.inbox"),
        # The journal, writing a declared kind's pair through the kind's own writer —
        # credited to the leaf that declared the kind, which is the writer of record.
        ("ph.session.journal", "kind.writer"),
        # `ph.testing.log_event`: the scaffolding writer tests build logs with.
        ("ph.testing.builders", "SCAFFOLDING"),
    }
)
"""Two-argument `.append` receivers that are not a module's own writer, and why."""


@dataclass(frozen=True, slots=True)
class _Walk:
    """What the walk found in shipped code."""

    static: dict[str, set[str]]
    """Type → the modules that write it, through their writer or a kind they declare."""
    variable: set[tuple[str, str]]
    """`(module, expression)` for every write whose type is not a constant."""
    declared: dict[str, str]
    """Types declared with `declare_log_type` in shipped code → their `owner=`."""
    minted: dict[str, set[str]]
    """Module → the names it binds to its own `log_writer(__name__)` at module top."""
    strays: list[str]
    """Every write, mint or import that breaks the one-door rule, as a sentence."""


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
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        base = import_base(module, is_package, node)
        for alias in node.names:
            names[alias.asname or alias.name] = (base, alias.name)
    return names


def _mints(tree: ast.Module) -> set[str]:
    """The names bound, at module top, to `log_writer(__name__)`."""
    found: set[str] = set()
    for node in tree.body:
        value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if (
            isinstance(value, ast.Call)
            and getattr(value.func, "id", "") == "log_writer"
            and len(value.args) == 1
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id == "__name__"
        ):
            found |= {target.id for target in targets if isinstance(target, ast.Name)}
    return found


def _walk() -> _Walk:
    modules = {name: (parsed.tree, parsed.is_package) for name, parsed in parsed_modules().items()}
    constants = {name: _constants(tree) for name, (tree, _) in modules.items()}
    imports = {name: _imports(name, tree, pkg) for name, (tree, pkg) in modules.items()}
    minted = {name: _mints(tree) for name, (tree, _) in modules.items()}

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
    strays: list[str] = []
    for module, (tree, _) in modules.items():
        own = minted[module]
        in_session = module == "ph.session" or module.startswith("ph.session.")
        for name, (source, original) in imports[module].items():
            if original in minted.get(source, set()):
                strays.append(f"{module} imports {source}'s writer {original} as {name}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            name = callee.attr if isinstance(callee, ast.Attribute) else getattr(callee, "id", "")
            line = f"{module}:{node.lineno}"
            at_top = any(isinstance(top, ast.Assign) and top.value is node for top in tree.body)
            if name == "log_writer" and module != "ph.session.writers" and not (own and at_top):
                strays.append(f"{line} mints a writer other than at module top")
            if name == "scaffolding_writer" and not module.startswith("ph.testing"):
                strays.append(f"{line} reaches for the scaffolding writer")
            if name == "LogWriter" and module != "ph.session.writers":
                strays.append(f"{line} constructs a LogWriter rather than minting one")
            if name == "_append" and len(node.args) >= 2 and not in_session:
                strays.append(f"{line} calls a log's _append past its writer")
            if name == "IntentKind":
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                writer = keywords.get("writer")
                if not (isinstance(writer, ast.Name) and writer.id in own):
                    strays.append(f"{line} declares a kind with a writer that is not its own")
                for key in ("opened", "settled"):
                    value = keywords.get(key)
                    kind = None if value is None else type_of(module, value)
                    if kind is None:
                        variable.add((module, ast.unparse(value) if value else key))
                    else:
                        static[kind].add(module)
            if name == "declare_log_type" and node.args:
                kind = type_of(module, node.args[0])
                owner = next((k.value for k in node.keywords if k.arg == "owner"), None)
                if kind is not None and isinstance(owner, ast.Constant):
                    declared[kind] = str(owner.value)
            if not (isinstance(callee, ast.Attribute) and name == "append" and len(node.args) >= 2):
                continue
            receiver = ast.unparse(callee.value)
            if (module, receiver) in NOT_A_LOG_WRITE:
                continue
            if not (isinstance(callee.value, ast.Name) and callee.value.id in own):
                strays.append(f"{line} appends through {receiver}, which is not its writer")
                continue
            if len(node.args) < 3:
                strays.append(f"{line} calls its writer without a log, a type and a payload")
                continue
            kind = type_of(module, node.args[1])
            if kind is None:
                variable.add((module, ast.unparse(node.args[1])))
            else:
                static[kind].add(module)
    return _Walk(
        static=dict(static),
        variable=variable,
        declared=declared,
        minted={module: names for module, names in minted.items() if names},
        strays=strays,
    )


WALK = _walk()


def _writes() -> set[tuple[str, str]]:
    """Every `(module, type)` the code writes: what the walk read, plus what each listed
    variable site may write."""
    pairs = {(module, kind) for kind, modules in WALK.static.items() for module in modules}
    for (module, _expression), kinds in VARIABLE_SITES.items():
        pairs |= {(module, kind) for kind in kinds}
    return pairs


# ------------------------------------------------------------- statically --


def test_the_walk_sees_every_package() -> None:
    """A walk that silently shrank would pass every gate below."""
    writers = {module.partition(".")[0] for modules in WALK.static.values() for module in modules}
    assert {"ph", "ph_app", "ph_rlm", "ph_stabilize"} <= writers, writers
    assert {module.partition(".")[0] for module in WALK.minted} >= writers


def test_the_table_covers_the_vocabulary() -> None:
    assert set(WRITERS) == KNOWN_SESSION_EVENT_TYPES
    assert not [kind for kind, modules in WRITERS.items() if not modules]


def test_every_write_in_every_package_names_a_known_type() -> None:
    """The static half of F11: a type this build writes and would refuse to read."""
    unknown = set(WALK.static) - KNOWN_SESSION_EVENT_TYPES - set(WALK.declared)
    assert not unknown, {kind: sorted(WALK.static[kind]) for kind in unknown}


def test_nothing_writes_a_log_but_its_own_writer() -> None:
    """The one-door rule, read from the source: every write is the writing module's
    own writer, minted at its module top, used there alone; nothing reaches past a
    writer to the log.

    Sabotage: import `ph.seams.sandbox`'s `_LOG` into `ph_stabilize.hitl` and write
    `sandbox/mode` with it, and this names the import and the write.
    """
    assert not WALK.strays, WALK.strays


def test_the_writers_table_is_what_the_writers_write() -> None:
    """`WRITERS` is derived from the writers, in both directions: a module that writes
    a type the table does not grant it fails here as it would at runtime, and a grant
    nothing writes any more is removed rather than left to widen it.

    Sabotage: add a type to a module's row in `_WRITTEN_BY` that it never writes, or
    write one it has no row for.
    """
    owners = {kind: frozenset({owner}) for kind, owner in WALK.declared.items()}
    granted = {(module, kind) for kind, modules in WRITERS.items() for module in modules}
    written = _writes()
    unowned = sorted(
        (module, kind)
        for module, kind in written
        if module not in WRITERS.get(kind, owners.get(kind, frozenset()))
    )
    unwritten = sorted(granted - written)
    assert not unowned, unowned
    assert not unwritten, unwritten


def test_no_new_write_takes_its_type_from_a_variable() -> None:
    assert WALK.variable == set(VARIABLE_SITES), sorted(WALK.variable ^ set(VARIABLE_SITES))


def test_a_declared_type_is_written_only_by_its_owner() -> None:
    """None ships today (decision 7); held here so the first one is checked."""
    for kind, owner in WALK.declared.items():
        assert WALK.static.get(kind, set()) <= {owner}, (kind, owner, WALK.static.get(kind))


def test_the_walk_follows_an_imported_constant() -> None:
    """The resolution the table depends on, held on its own: these sites name a
    constant another module defines, never a literal."""
    assert "ph_rlm.subagents" in WALK.static["subagent/status"]
    assert "ph_app.daemon.supervisor" in WALK.static["supervisor/retry"]


# --------------------------------------------------------------- at runtime --


def test_a_row_that_writes_a_type_it_does_not_own_is_refused() -> None:
    """F12, closed. This module is a third-party row as far as the table knows: no
    row grants it anything, so its writer refuses every type — a posture record
    first, which is the write that mattered.

    Sabotage: drop the ownership check from `LogWriter.append`, and `sandbox/mode`
    lands.
    """
    mine = log_writer(__name__)
    session = Session("third-party")

    with pytest.raises(LogWriteError, match='not a writer of record for "sandbox/mode"'):
        mine.append(session, "sandbox/mode", {"mode": "danger-full-access"})
    assert session.events == ()


def test_a_session_has_no_public_append() -> None:
    """The door a writer replaces is gone, rather than beside it."""
    assert not hasattr(Session, "append")


def test_a_module_cannot_mint_another_modules_writer() -> None:
    """The owner is read off the calling frame: naming `ph.seams.sandbox` as this
    module's owner would claim its types."""
    with pytest.raises(LogWriteError, match=r"cannot mint the writer of 'ph\.seams\.sandbox'"):
        log_writer("ph.seams.sandbox")


def test_the_scaffolding_writer_is_ph_testings_alone() -> None:
    with pytest.raises(LogWriteError, match="cannot mint the scaffolding writer"):
        scaffolding_writer()


def test_a_declared_type_is_written_by_its_owner_and_nobody_else() -> None:
    """A package's own type: its owner's writer writes it; any other module's does not."""
    declare_log_type("probe/declared", owner=__name__, ignorable=True)
    session = Session("declared")

    log_writer(__name__).append(session, "probe/declared", {"n": 1})
    with pytest.raises(LogWriteError, match='"probe/declared"'):
        DRIVER.append(session, "probe/declared", {"n": 2})
    assert [event.type for event in session.events] == ["probe/declared"]


def test_the_writer_costs_the_hot_path_almost_nothing() -> None:
    """`assistant/chunk` is written once per streamed token, so the door must stay a
    method call and a set lookup in front of the append (T6). Measured against the
    append behind it, the two alternating within each round and the best of each kept,
    so a busy machine slows both halves alike; the door measures at about 1.0-1.05.

    Sabotage: make `LogWriter.append` rebuild its type set from the vocabulary on
    every call (1.2-1.4 here), and this fails.
    """
    payload: JsonObject = {
        "turn": 1,
        "step": 1,
        "attempt": 1,
        "chunk": {"type": "text-delta", "text": "x"},
    }

    def once(write: Callable[[Session], object]) -> float:
        session = Session("bench")
        started = time.perf_counter()
        for _ in range(2000):
            write(session)
        return time.perf_counter() - started

    direct = through = float("inf")
    for _ in range(9):
        direct = min(direct, once(lambda session: session._append("assistant/chunk", payload)))
        through = min(
            through, once(lambda session: DRIVER.append(session, "assistant/chunk", payload))
        )

    assert through <= direct * 1.2, f"{through / direct:.2f} times the bare append"
    assert isinstance(DRIVER, LogWriter) and "assistant/chunk" in DRIVER.types
