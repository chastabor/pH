"""T4 — intent kinds: one leaf per package, found statically, and nothing skipped.

Repair settles the kinds declared in the process doing the resume, and a kind is
declared by importing the module that holds it. So *where* a kind lives, and how
it is reached, decides whether a mass restart settles it. These gates hold the
arrangement `ph.session.kinds` describes, at test time rather than at the first
resume that goes wrong:

* every `IntentKind(...)` in shipped code is in a package's `kinds` leaf, and the
  vocabulary's `INTENT_PAIRS` names exactly those pairs and leaves;
* a leaf imports nothing a cycle could run through;
* each leaf is imported, at module top, by the package that holds it — so any
  process that loaded any part of a package has its kinds;
* no kinds module is imported anywhere except at module top, and the resume path
  holds no function-level import at all;
* a resume whose log holds an open intent of a kind this process never declared is
  refused by name.

Read from source with `ast`, over every package in the workspace, the way
`test_log_writers.py` reads its appends — so a package ph-core never imports is
still held to them.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
from workspace_layout import ParsedModule, import_base, parsed_modules, workspace_tests

from ph.keys import SESSIONS
from ph.persistence import UndeclaredIntentError, resume_session
from ph.session.known_event_types import INTENT_PAIRS, IntentPair
from ph.testing import MountProfile, isolated_intent_kinds, log_event, stored_types

LEAF_MAY_IMPORT = frozenset(
    {"__future__", "ph.session.intents", "ph.session.events", "ph.session.writers", "ph.json"}
)
"""What a leaf may import beyond the standard library: nothing above `ph.session`'s
declarations, so importing a leaf can never cycle back through a seam."""

RESUME_PATH = ("ph.persistence.repair", "ph.persistence.jsonl")
"""The modules a resume runs through, held to no function-level import at all (T4)."""


SHIPPED = parsed_modules()
LEAVES = sorted(name for name in SHIPPED if name.rpartition(".")[2] == "kinds")


def _imported(module: str, is_package: bool, node: ast.Import | ast.ImportFrom) -> set[str]:
    """The absolute module names one import statement may load.

    `from a import b` counts as both `a` and `a.b`, since `b` may be a submodule —
    which is how `from ph.session import kinds` is spelled.
    """
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    base = import_base(module, is_package, node)
    return {base, *(f"{base}.{alias.name}" for alias in node.names)}


def _imports(module: str, found: ParsedModule) -> list[tuple[set[str], bool, int]]:
    """Each import in a module: what it loads, whether it is at module top, its line."""
    top = {id(node) for node in found.tree.body}
    return [
        (_imported(module, found.is_package, node), id(node) in top, node.lineno)
        for node in ast.walk(found.tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]


def _declared_pairs() -> dict[str, tuple[str, str]]:
    """Every `IntentKind(opened=…, settled=…)` in shipped code: opened → (settled, module)."""
    pairs: dict[str, tuple[str, str]] = {}
    for module, found in SHIPPED.items():
        for node in ast.walk(found.tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "IntentKind"):
                continue
            named = {k.arg: k.value for k in node.keywords}
            opened, settled = named.get("opened"), named.get("settled")
            assert isinstance(opened, ast.Constant) and isinstance(settled, ast.Constant), (
                f"{module}:{node.lineno} declares a kind whose types the walk cannot read"
            )
            pairs[str(opened.value)] = (str(settled.value), module)
    return pairs


def test_the_walk_finds_both_leaves() -> None:
    """A walk that silently found nothing would pass every gate below."""
    assert LEAVES == ["ph.session.kinds", "ph_app.kinds"]


def test_every_intent_kind_is_declared_in_a_kinds_leaf() -> None:
    """Sabotage: declare a kind in a seam, and this names the seam."""
    strays = sorted(
        (module, opened)
        for opened, (_settled, module) in _declared_pairs().items()
        if module not in LEAVES
    )
    assert not strays, strays


def test_the_vocabulary_names_exactly_the_pairs_the_leaves_declare() -> None:
    """Both directions: a leaf's pair missing from `INTENT_PAIRS` is one repair would
    skip in a process without that leaf; a row no leaf declares is one it would
    refuse for nothing."""
    declared = {
        opened: IntentPair(settled, module)
        for opened, (settled, module) in _declared_pairs().items()
    }
    assert declared == dict(INTENT_PAIRS)


def test_a_leaf_imports_nothing_a_cycle_could_run_through() -> None:
    """Sabotage: import `ph.seams.shell` from `ph.session.kinds`, and this names it."""
    stdlib = sys.stdlib_module_names
    bad = sorted(
        (leaf, name, line)
        for leaf in LEAVES
        for names, _top, line in _imports(leaf, SHIPPED[leaf])
        for name in names
        if name not in LEAF_MAY_IMPORT
        and name.partition(".")[0] not in stdlib
        # `from ph.session.intents import IntentKind` also yields
        # `ph.session.intents.IntentKind`, which is a name, not a module.
        and not any(name.startswith(f"{allowed}.") for allowed in LEAF_MAY_IMPORT)
    )
    assert not bad, bad


def test_each_package_imports_its_leaf_at_module_top() -> None:
    """The static import that makes loading any part of a package declare its kinds.

    Sabotage: drop `from . import kinds` from `ph_app/__init__.py`, and a daemon that
    resumes a root before anything imports `ph_app.daemon.server` has no
    `CLIENT_COMMAND` to settle its open verbs with.
    """
    missing = [
        leaf
        for leaf in LEAVES
        if not any(
            leaf in names and top
            for names, top, _line in _imports(
                leaf.rpartition(".")[0], SHIPPED[leaf.rpartition(".")[0]]
            )
        )
    ]
    assert not missing, missing


def test_no_kinds_module_is_imported_below_module_top_anywhere() -> None:
    """Shipped code and every suite: a kinds import inside a function, or behind
    `TYPE_CHECKING`, is the runtime discovery T4 removed.

    Sabotage: move `ph.testing`'s `from ..session import kinds` into
    `isolated_intent_kinds`, and this names the line.
    """
    found = [
        (module, line)
        for module, parsed in SHIPPED.items()
        for names, top, line in _imports(module, parsed)
        if not top and any(name in LEAVES or name.startswith(tuple(LEAVES)) for name in names)
    ]
    for name, path in workspace_tests():
        parsed = ParsedModule(ast.parse(path.read_text(encoding="utf-8")), is_package=False)
        found += [
            (name, line)
            for names, top, line in _imports("tests", parsed)
            if not top and any(n in LEAVES or n.startswith(tuple(LEAVES)) for n in names)
        ]
    assert not found, found


def test_the_resume_path_holds_no_function_level_import() -> None:
    """The ten T4 removed: repair's two, `_reconciled`'s four, `resume_session`'s two
    and `isolated_intent_kinds`' two. A resume imports what it needs when its module
    loads, so a missing or cyclic one fails every run, not the first resume."""
    below = [
        (module, line)
        for module in RESUME_PATH
        for _names, top, line in _imports(module, SHIPPED[module])
        if not top
    ]
    (isolated,) = [
        node
        for node in ast.walk(SHIPPED["ph.testing.builders"].tree)
        if isinstance(node, ast.FunctionDef) and node.name == "isolated_intent_kinds"
    ]
    below += [
        ("ph.testing.builders.isolated_intent_kinds", node.lineno)
        for node in ast.walk(isolated)
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    assert not below, below


# ----------------------------------------------------------- the refusal --


async def _stored(mount: MountProfile, tmp_path: Path, *records: tuple[str, dict[str, str]]) -> int:
    """A stored log holding `records`, and how many events the store has for it."""
    ctx = await mount({"id": "session-persistence", "config": {"root": str(tmp_path / "sessions")}})
    sessions = ctx.require(SESSIONS)
    session = sessions.create("verbs")
    for event_type, data in records:
        log_event(session, event_type, data)
    await sessions.flush(session)
    sessions.dispose("verbs")
    return len(stored_types(ctx, "verbs"))


@pytest.mark.anyio
async def test_a_resume_without_a_packages_kinds_is_refused_by_name(
    mount: MountProfile, tmp_path: Path
) -> None:
    """An open `client/command`, resumed by a process that declared only ph-core's
    kinds: refused, naming the type and the leaf, and the log is left as it was.

    `isolated_intent_kinds(core=True)` is that process's table, whatever this
    interpreter has imported. Sabotage: skip `_refuse_undeclared` in repair, and the
    resume succeeds with the verb still open — what every process without the app
    did before T4.
    """
    stored = await _stored(mount, tmp_path, ("client/command", {"command": "c1:k1"}))
    ctx = await mount({"id": "session-persistence", "config": {"root": str(tmp_path / "sessions")}})

    with (
        isolated_intent_kinds(core=True),
        pytest.raises(UndeclaredIntentError, match=r'"client/command".*ph_app\.kinds'),
    ):
        await resume_session(ctx, "verbs")
    assert len(stored_types(ctx, "verbs")) == stored, "nothing written"


@pytest.mark.anyio
async def test_a_settled_intent_of_an_undeclared_kind_resumes(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Refused only when something is open: a verb that settled is nothing repair
    would have to write, so a process without its kind resumes the log."""
    await _stored(
        mount,
        tmp_path,
        ("client/command", {"command": "c1:k1"}),
        ("client/command-settled", {"command": "c1:k1"}),
    )
    ctx = await mount({"id": "session-persistence", "config": {"root": str(tmp_path / "sessions")}})

    with isolated_intent_kinds(core=True):
        revived = await resume_session(ctx, "verbs")
    assert revived.events[-1].data["closed"] == 0
