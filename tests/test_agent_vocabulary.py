"""A fact the agent vocabulary declares is read, never `getattr`-ed.

`AgentHandle` states five read-only facts and `AgentOptions` five settings, and
every one is typed. Reaching them through `getattr(agent, "session", None)`
returns `Any` and erases all of it — silently, because the annotation on the
parameter still says `AgentHandle` and mypy is satisfied either way.

**This is the erasure ANN401 cannot see.** That rule flags a bare `Any` in an
annotation; a `getattr` writes no annotation at all, so the ratchet in
`pyproject.toml` will never reach one. Both sweeps that wrote this gate found the
habit spread across three packages before anyone counted.

**The defaults are the damage, not the `Any`.** Each of these had a second
argument returned when the attribute was missing — and the attribute is never
missing, so the default answered for a case that cannot arise while hiding one
that can. `getattr(agent, "status", "idle")` sat in the one branch that refuses
to compact a working agent, and `StubAgent` had no `status` at all: every stub
read as idle, so the refusal could not fire in any test that used one. Others
collapsed *"there is no agent"* into *"the agent has none of that"*, and one
passed `""` into a parameter declared `str` for an agent that was absent. Three
signatures turned out to be lying about `None` once the probe stopped hiding it.

**Derived, not listed.** The facts are read off the classes themselves, so a
sixth member on `AgentHandle` tomorrow is covered without anybody editing this
file — the same reason `workspace_layout` discovers packages rather than naming
them. Read rather than parsed: a `@property` respelled as a plain annotation, or
the Protocol moved to another module, would quietly empty a syntax match and the
gate would pass for free.

@module tests.test_agent_vocabulary
"""

from __future__ import annotations

import ast
import pathlib

from workspace_layout import workspace_modules

from ph.agent.types import AgentHandle, AgentOptions
from ph.wire import declarable_fields

ALLOWED: frozenset[tuple[str, str]] = frozenset(
    {
        # `parent` arrives on a caller-built payload, so the broken case stays
        # reachable: the next line type-checks the result and raises
        # `SubagentSpawnError` naming the missing `ctx: Context`. Three
        # `type: ignore[arg-type]` in `test_subagent_grant.py` exist to reach it.
        ("ph/seams/subagents.py", "getattr(request.parent, 'ctx', None)"),
        # `workspace_of` is documented fail-soft and `test_workspace.py` hands it
        # a bare `object()` — an absent workspace row must not be fatal.
        ("ph/seams/workspace.py", "getattr(agent, 'id', '')"),
    }
)
"""The sites whose receiver is untrusted at *runtime*, whatever it is declared.

Keyed on the call as written rather than on the attribute, so a second and
genuinely wrong `getattr(_, "ctx", …)` elsewhere in a thousand-line seam is not
pre-blessed by this one — the shape `test_json_narrowing.ALLOWED` settled on.

Not exemptions from the rule so much as the other side of it: neither holds an
`AgentHandle` in practice and both say so on the next line, one by raising and
one by falling soft. A test escaping the type system to reach each branch is the
specification; both were converted by the sweep that wrote this gate and both
were put back when those tests failed."""


def _facts() -> set[str]:
    """Every member `AgentHandle` and `AgentOptions` declare.

    `vars` on the Protocol — the spelling `test_registration_ownership` already
    uses to walk one — and `declarable_fields` on the dataclass, which the wire
    layer ships for exactly this. `typing.get_protocol_members` would say it in
    one call and lands in 3.13; this targets 3.12.
    """
    stated = {name for name in vars(AgentHandle) if not name.startswith("_")}
    return stated | set(declarable_fields(AgentOptions))


def _laundered(path: pathlib.Path, facts: set[str]) -> list[tuple[int, str]]:
    """Every `getattr(_, "<fact>", …)` in this module: `(line, the call as written)`.

    Prefiltered on both the call and a quoted fact, which cuts the parse set to a
    quarter of the modules that mention `getattr` at all. Sound rather than
    merely quick, and for the sibling's reason: an offender's attribute is an
    `ast.Constant` string, so its characters are in the source. Built from
    `facts` so the filter cannot drift from the rule it serves.
    """
    source = path.read_text(encoding="utf-8")
    if "getattr(" not in source or not any(f'"{fact}"' in source for fact in facts):
        return []
    return [
        (node.lineno, ast.unparse(node))
        for node in ast.walk(ast.parse(source, filename=str(path)))
        if isinstance(node, ast.Call)
        if isinstance(node.func, ast.Name) and node.func.id == "getattr"
        if len(node.args) >= 2
        if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)
        if node.args[1].value in facts
    ]


def _sites() -> list[tuple[str, int, str]]:
    """Every laundered read in the shipped packages, exemptions included."""
    facts = _facts()
    return [
        (name, line, call)
        for name, path in workspace_modules()
        for line, call in _laundered(path, facts)
    ]


def test_no_declared_agent_fact_is_reached_through_getattr() -> None:
    """The shipped packages, which are where the type is worth having.

    Sabotage: write `getattr(agent, "id", "")` anywhere in `packages/*/src` and
    this names the line. Add a sixth member to `AgentHandle` and the rule covers
    it with no edit here.
    """
    offenders = [
        f"{name}:{line}: `{call}` — that fact is declared; read it"
        for name, line, call in _sites()
        if (name, call) not in ALLOWED
    ]
    assert offenders == [], (
        "these reach a fact the agent vocabulary declares through `getattr`, which "
        "returns `Any` and takes the type with it — and whose default answers for a "
        "case that cannot arise:\n  " + "\n  ".join(offenders)
    )


def test_every_allowed_site_still_exists() -> None:
    """An exemption that no longer matches anything is an exemption nobody reads.

    The two entries above are the argument for the only two probes left. When one
    is rewritten or deleted its entry stops describing the tree, and a stale
    exemption is how the next real erasure slips in wearing an old name — which
    is what `test_json_narrowing` records happening to its own list.
    """
    live = {(name, call) for name, _, call in _sites()}
    stale = sorted(f"{name}: {call}" for name, call in ALLOWED - live)
    assert stale == [], (
        "these entries no longer match anything in the tree, so they excuse "
        "whatever is written under that name next — delete them:\n  " + "\n  ".join(stale)
    )
