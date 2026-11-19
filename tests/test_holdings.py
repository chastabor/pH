"""A holding ends once, through one door (I2; DESIGN.md §2.5, "Holdings").

Gate: *every teardown a module registers by hand is either one **only its scope
ends**, and says why, or one **somebody keeps**, and names the door that ends it —
enumerated over every shipped package by reading the source, so a new one cannot
join undecided.*

## The defect this is for

An operation writes an **entry** (a job, a workspace, a kernel, a manual, a child)
and registers the **teardown** that takes it back. The two are one holding, and
the defect is every way they come apart:

* the entry goes another way — deleted, replaced under its key, ended early — and
  the teardown stays on its scope, to run later against whatever holds the key by
  then. `WorkspaceSeam._track` evicted an agent's live workspace this way, and
  `UtcpSeam.add` took a working manual's record from the client;
* a start fails, and what it had already written or registered is left for a
  thing that never existed;
* something ends, and a teardown still on a scope writes about it afterwards.

Each was found one at a time, by reading. What they share is a `Releaser` nobody
kept: `add_disposer` and `effect` hand one back so that the door which ends a
holding early can run the teardown (`releaser()`) or take it off the scope unrun
(`releaser.drop()`), and a call that throws it away has decided that no such door
exists. That decision is the thing this module makes somebody state.

## What it can and cannot see

It reads **direct** calls of `add_disposer` and `effect`. A registration made
through `claim_key`, `claim_entry`, `claim_slot` or `ScopedTable` is not here and
need not be: those release by identity and do nothing else, so a successor under
the same key is out of their reach by construction (§3.2).

A kept releaser is found by its **type**: a class field annotated `Releaser`. One
kept in a local and run before the function returns is seen as kept and asked
nothing more, which is right — it never outlives the call that could forget it.

`ENDED_BY` is checked against the code, and what that check is worth is bounded
the way `test_a_bound_body_names_an_invoker_that_binds` bounds its own: it proves
the named door settles *a* releaser, not that it settles this one on every path.
Deleting a door's only `drop()` turns it red; the per-path guarantee is the
behavioral tests' job, and each holding below has one.

@module tests.test_holdings
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

import pytest
from workspace_layout import parsed_modules

REGISTERS = frozenset({"add_disposer", "effect"})
"""The two calls that put a teardown on a scope and hand back its `Releaser`."""


def _qualified(tree: ast.Module) -> Iterator[tuple[str, ast.AST]]:
    """Every function and class in a module, with its dotted name."""

    def walk(node: ast.AST, path: tuple[str, ...]) -> Iterator[tuple[str, ast.AST]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                inner = (*path, child.name)
                yield ".".join(inner), child
                yield from walk(child, inner)
            else:
                yield from walk(child, path)

    yield from walk(tree, ())


def _label(call: ast.Call) -> str:
    """The `label=` a registration was given, with each interpolation as `{}`."""
    for keyword in call.keywords:
        if keyword.arg != "label":
            continue
        value = keyword.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        if isinstance(value, ast.JoinedStr):
            return "".join(
                str(part.value) if isinstance(part, ast.Constant) else "{}" for part in value.values
            )
    return ""


def _own_statements(scope: ast.AST) -> Iterator[ast.stmt]:
    """The statements of one function or module, not those of what it defines."""
    for child in ast.iter_child_nodes(scope):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
            continue
        if isinstance(child, ast.stmt):
            yield child
        yield from _own_statements(child)


def _discarded() -> set[str]:
    """Every direct registration whose `Releaser` is thrown away, as
    `module:function:label`.

    A call that stands as a statement of its own, awaited or not. One that is
    assigned, returned or passed on has been kept by somebody.
    """
    found: set[str] = set()
    for module, parsed in parsed_modules().items():
        scopes: list[tuple[str, ast.AST]] = [("", parsed.tree), *_qualified(parsed.tree)]
        for name, scope in scopes:
            if isinstance(scope, ast.ClassDef):
                continue
            for statement in _own_statements(scope):
                if not isinstance(statement, ast.Expr):
                    continue
                value = statement.value
                call = value.value if isinstance(value, ast.Await) else value
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr in REGISTERS
                ):
                    found.add(f"{module}:{name}:{_label(call)}")
    return found


def _kept() -> set[str]:
    """Every class field whose annotation names `Releaser`, as `module:Class.field`."""
    found: set[str] = set()
    for module, parsed in parsed_modules().items():
        for name, node in _qualified(parsed.tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for entry in node.body:
                if (
                    isinstance(entry, ast.AnnAssign)
                    and isinstance(entry.target, ast.Name)
                    and "Releaser" in ast.unparse(entry.annotation)
                ):
                    found.add(f"{module}:{name}.{entry.target.id}")
    return found


ROW = "the row's own: what it tears down is reached through nothing but this mount"

ONLY_ITS_SCOPE: dict[str, str] = {
    # `module:function:label` → why nothing but the scope's unwind ends it.
    #
    # A row's `apply`, registering against its own mount. Nothing holds a key to
    # what these take back, so there is no other door to forget them at.
    "ph.agent_loop:apply:agent-loop": ROW,
    "ph.llm.fake:apply:llm-fake": ROW,
    "ph.llm.replay:apply:llm-replay": ROW,
    "ph.persistence.checkpoint_policy:apply:checkpoint.after-the-agents": ROW,
    "ph.seams.telemetry_otel:apply:telemetry.otel": ROW,
    "ph_code_graph.engine:apply:tree-sitter-cache": ROW,
    "ph_rlm.kernel.manager:apply:code-runtime-python": ROW,
    "ph_rlm.subagents:apply:rlm-children.suspend": ROW,
    "ph_utcp:apply:utcp-client": ROW,
    "ph_app.adapters.anthropic:apply:llm({})": ROW,
    "ph_app.adapters.anthropic:apply:uploader({})": ROW,
    "ph_app.adapters.anthropic:apply:http({})": ROW,
    "ph_app.adapters.google:apply:llm({})": ROW,
    "ph_app.adapters.google:apply:uploader({})": ROW,
    "ph_app.adapters.google:apply:http({})": ROW,
    "ph_app.adapters.openai_compatible:apply:llm({})": ROW,
    "ph_app.adapters.openai_compatible:apply:uploader({})": ROW,
    "ph_app.adapters.openai_compatible:apply:http({})": ROW,
    "ph_rlm.snapshot:apply:rlm-kernel-snapshot(runtime)": (
        "the row's policy, on the runtime another row owns: `provide` lets one realm "
        "hold one, and the take-back clears it only while it is still this one"
    ),
    "ph_rlm.skills:give_to_runtime:rlm-skills-python(runtime)": (
        "the row's own entries in the runtime's lists, taken back one occurrence each; "
        "nothing else removes them"
    ),
    # An agent's scope, made by the call that registers on it. The scope is the
    # agent: `dispose(agent_id)` ends it by disposing that scope, and `_forget`
    # pops the roster first so the second of the two doors finds nothing. The
    # agent's last write is registered beside it and is not here: `create` keeps
    # that releaser for the one case with another door, a driver that fails.
    "ph.agent.registry:AgentRegistry.create:agents.entry({})": (
        "the roster entry, on the scope it describes; `dispose(agent_id)` goes "
        "through that scope and `_forget` is guarded by its own pop"
    ),
    "ph.persistence.lease:claim_session:session-lease({})": (
        "a lease is given back only when its scope goes; `_HELD` is discarded on "
        "the failure path before the effect exists"
    ),
    "ph.persistence.protocol:write_on_unwind:session-last-write": (
        "the mount's last write; `_OWED` is what says whether one is still owed, "
        "so it is asked there and not removed here"
    ),
    "ph.resources:temporary_directory:tempdir": (
        "the directory is handed out as a path, and nothing but the scope removes it"
    ),
    "ph_text_index.row:TextIndexRow.register:text-index-embedder({})": (
        "the embedder's claim, on the provider's scope; `unclaim` runs the slot's "
        "own releaser and clears the manual only while it is still this one"
    ),
}
"""Registrations whose `Releaser` is thrown away, and why each may be.

**The question a new entry answers:** can what this tears down be removed,
replaced under its key, or ended early by anything but the scope's unwind? If it
can, the releaser is not this table's — keep it beside the entry and settle it at
that door (`ENDED_BY`). "Its teardown checks first" is not an answer for anything
but a marker the teardown alone reads: `UtcpSeam.add` checked the state by
identity and then deregistered by name."""


ENDED_BY: dict[str, tuple[str, ...]] = {
    # `module:Class.field` → the functions of that module that settle it.
    "ph.seams.jobs:Job.release": ("JobService.forget",),
    "ph.seams.workspace:_Held.dispose": ("WorkspaceSeam._track", "WorkspaceSeam.dispose"),
    "ph_rlm.kernel.manager:PythonCodeRuntime._closers": ("PythonCodeRuntime.close_namespace",),
    "ph_rlm.subagents:_Child.teardown": ("RlmChildProvider._stop",),
    "ph_utcp:UtcpSeam._forgets": ("UtcpSeam.add",),
}
"""Kept releasers, and the door each is settled at: run, or dropped unrun.

One door where there can be one. `_Held.dispose` has two because a workspace is
let go by its agent (`dispose`, which runs it) and replaced by a second acquire
(`_track`, which drops the one it replaces)."""


def test_every_discarded_releaser_says_why_only_its_scope_ends_it() -> None:
    """A new registration that throws its releaser away fails until somebody decides.

    The point of the walk. Before it, the only thing between a hand-written
    teardown and a successor under the same key was that someone would think to
    ask whether the entry could go another way.
    """
    undecided = sorted(_discarded() - set(ONLY_ITS_SCOPE))
    assert undecided == [], (
        "each of these registers a teardown and throws away the releaser. If "
        "anything but the scope's unwind can remove, replace or end what it tears "
        "down, keep the releaser beside the entry and settle it at that door "
        f"(ENDED_BY); if nothing can, say why in ONLY_ITS_SCOPE: {undecided}"
    )


def test_every_kept_releaser_names_its_door() -> None:
    """A releaser somebody stores is one somebody has to settle, and says where."""
    undecided = sorted(_kept() - set(ENDED_BY))
    assert undecided == [], (
        "each of these fields holds a `Releaser` and names no door that runs or "
        f"drops it: add it to ENDED_BY: {undecided}"
    )


def test_no_declaration_outlives_what_it_declares() -> None:
    """The other direction: a table row for code that is gone is a decision about
    nothing, and the next registration to take its name would inherit it."""
    stale = sorted(
        (set(ONLY_ITS_SCOPE) - _discarded()) | (set(ENDED_BY) - _kept()),
    )
    assert stale == [], f"these are declared but no longer exist: {stale}"


@pytest.mark.parametrize("field", sorted(ENDED_BY))
def test_a_named_door_settles_a_releaser(field: str) -> None:
    """`ENDED_BY` is checked against the code, not trusted.

    Each door must exist, must mention the field, and must run or drop a releaser.
    Without this the table is a promise, and the cheapest way past the test above
    is to type a name into it.
    """
    module, _, owner = field.partition(":")
    attribute = owner.rpartition(".")[2]
    functions = dict(_qualified(parsed_modules()[module].tree))
    for door in ENDED_BY[field]:
        assert door in functions, f"{field} names a door that does not exist: {door}"
        source = ast.unparse(functions[door])
        assert attribute in source, f"{door} is named as {field}'s door and never touches it"
        assert ".drop()" in source or f".{attribute}()" in source, (
            f"{door} is named as {field}'s door, and neither drops a releaser nor runs it"
        )
