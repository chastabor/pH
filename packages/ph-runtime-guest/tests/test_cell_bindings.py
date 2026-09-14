"""`bound_names` over every binding form, because a miss is silent.

**What this function decides.** A cell's body is wrapped in an `async def` so
that top-level `await` and `return` both work, and a name assigned inside a
function is local to it — so without intervention every variable a cell defined
would vanish at the end of it. `compile_cell` avoids that by declaring each name
the cell binds as `global` in the wrapper, and `bound_names` computes that set
from the AST.

**A form it misses does not raise.** The cell runs, the assignment succeeds, the
variable is local to a wrapper that is about to be discarded, and the *next*
cell gets `NameError` on a name the person watching just saw assigned. Nothing
in the guest can detect this — the compile is valid Python either way — so the
only place the rule can be held is here, against the syntax list.

That list is the module's own docstring, which names `import`, `with ... as`,
`for`, `except ... as`, walrus, `del`, and function and class definitions.
Everything it claims is a case below.

**Why this suite exists at all.** `packages/ph-runtime-guest` had no tests of
its own. Its modules are exercised through `ph-rlm`'s kernel tests — thoroughly,
across a real process boundary, which is the right way to test the half that
runs model-written code — but that path reaches `bound_names` only with whatever
syntax the test cells happen to use, which is assignment and little else. The
forms below were all correct and none of them were covered.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ph_runtime.cell import CELL_FUNCTION, bound_names, compile_cell

pytestmark = pytest.mark.anyio
"""The convention every async suite here follows — one backend, asyncio, set
in the root `conftest.anyio_backend`. Harmless on the synchronous tests
below, and applying it at module scope is how the other suites do it."""


def names_in(source: str) -> set[str]:
    """The top-level bound names of a cell written as source."""
    return bound_names(ast.parse(source).body)


# ---------------------------------------------------------------------------
# Every form the module claims, one case each.

BINDINGS: list[tuple[str, str, set[str]]] = [
    ("assignment", "x = 1", {"x"}),
    ("chained", "a = b = 1", {"a", "b"}),
    ("augmented", "acc = 0\nacc += 1", {"acc"}),
    ("tuple unpack", "a, b = 1, 2", {"a", "b"}),
    ("list unpack", "[a, b] = [1, 2]", {"a", "b"}),
    ("starred", "first, *rest = [1, 2, 3]", {"first", "rest"}),
    ("nested unpack", "(p, [q, r]) = (1, (2, 3))", {"p", "q", "r"}),
    ("annotated", "n: int = 5", {"n"}),
    ("annotated bare", "m: int", {"m"}),
    ("walrus", "if (w := 5):\n    pass", {"w"}),
    ("for", "for i in range(3):\n    pass", {"i"}),
    ("for unpack", "for k, v in []:\n    pass", {"k", "v"}),
    ("async for", "async for a in gen():\n    pass", {"a"}),
    ("with as", "with open('f') as fh:\n    pass", {"fh"}),
    ("with two", "with open('a') as a, open('b') as b:\n    pass", {"a", "b"}),
    ("async with", "async with ctx() as c:\n    pass", {"c"}),
    ("except as", "try:\n    pass\nexcept ValueError as err:\n    pass", {"err"}),
    ("import", "import json", {"json"}),
    ("import as", "import json as j", {"j"}),
    ("from import", "from os import path", {"path"}),
    ("from import as", "from os import path as p", {"p"}),
    ("def", "def fn():\n    pass", {"fn"}),
    ("async def", "async def fn():\n    pass", {"fn"}),
    ("class", "class K:\n    pass", {"K"}),
    ("del", "x = 1\ndel x", {"x"}),
    ("match capture", "match 1:\n    case int() as got:\n        pass", {"got"}),
    ("match star", "match [1]:\n    case [*tail]:\n        pass", {"tail"}),
    ("match mapping rest", "match {}:\n    case {'k': v, **extra}:\n        pass", {"v", "extra"}),
]


@pytest.mark.parametrize(
    ("source", "expected"),
    [pytest.param(source, expected, id=label) for label, source, expected in BINDINGS],
)
def test_a_binding_form_is_seen(source: str, expected: set[str]) -> None:
    found = names_in(source)
    assert expected <= found, (
        f"these names would not survive to the next cell: {sorted(expected - found)}"
    )


# ---------------------------------------------------------------------------
# The other half of the rule, and the more dangerous one to get wrong.


NESTED: list[tuple[str, str, str]] = [
    ("function body", "def fn():\n    inner = 1", "inner"),
    ("lambda parameter", "f = lambda q: q", "q"),
    ("comprehension", "squares = [e * e for e in range(3)]", "e"),
    ("generator expression", "gen = (g for g in range(3))", "g"),
    ("dict comprehension", "d = {dk: 1 for dk in range(3)}", "dk"),
    ("set comprehension", "s = {sv for sv in range(3)}", "sv"),
    ("nested function parameter", "def fn(param):\n    pass", "param"),
    ("class body", "class K:\n    attr = 1", "attr"),
]


@pytest.mark.parametrize(
    ("source", "local"),
    [pytest.param(source, local, id=label) for label, source, local in NESTED],
)
def test_a_name_bound_in_a_nested_scope_is_not_globalized(source: str, local: str) -> None:
    """Declaring one of these `global` would change what the program means.

    The failure is the opposite shape from a miss and worse for it: a cell that
    read a module-level `e` would find the comprehension's last value written
    over it, so the bug is a wrong answer rather than a `NameError`.
    """
    assert local not in names_in(source), (
        f"`{local}` is local to its scope; globalizing it changes the program"
    )


async def run_cell(source: str, namespace: dict[str, object]) -> object:
    """Compile and run one cell the way `runner.Runner` does.

    The wrapper is an `async def`, so executing the compiled module only
    *defines* it; the value comes from awaiting it with the namespace as globals
    — which is also what makes the `global` declarations land there.
    """
    exec(compile_cell(source), namespace)  # running a cell is the unit under test
    cell = namespace.pop(CELL_FUNCTION)
    assert callable(cell)
    return await cell()


async def test_the_names_survive_an_actual_round_trip(tmp_path: Path) -> None:
    """The set is only useful if `compile_cell` acts on it.

    Asserting against `bound_names` alone would pass while the wrapper ignored
    the set entirely, which is the one failure that makes the module pointless.
    Every name here is a form the `BINDINGS` table covers in isolation; this is
    the same claim end to end, through the compile and the await.
    """
    namespace: dict[str, object] = {}
    await run_cell("plain = 1", namespace)
    await run_cell("for loop_value in range(2):\n    pass", namespace)
    await run_cell("import json as parsed_json", namespace)
    # A real file, because a cell's namespace has no `__file__` — it is not a
    # module, which is the point of the wrapper.
    document = tmp_path / "read-me.txt"
    document.write_text("first line\n", encoding="utf-8")
    await run_cell(
        f"with open({str(document)!r}) as handle:\n    first = handle.readline()", namespace
    )

    for name in ("plain", "loop_value", "parsed_json", "handle", "first"):
        assert name in namespace, f"`{name}` did not reach the persistent namespace"


async def test_a_trailing_expression_is_the_cells_value() -> None:
    """The REPL property, which shares the rewrite `bound_names` feeds."""
    namespace: dict[str, object] = {}

    assert await run_cell("total = 2 + 3\ntotal * 2", namespace) == 10
    assert await run_cell("total", namespace) == 5, "the earlier cell's name persisted"


async def test_a_name_from_a_nested_scope_does_not_leak_between_cells() -> None:
    """The negative rule, proved where it would actually bite.

    `test_a_name_bound_in_a_nested_scope_is_not_globalized` asserts the set; this
    asserts the consequence — a comprehension variable must not still be sitting
    in the namespace for the next cell to read.
    """
    namespace: dict[str, object] = {}
    await run_cell("squares = [e * e for e in range(3)]", namespace)

    assert namespace["squares"] == [0, 1, 4]
    assert "e" not in namespace, "the comprehension's variable escaped into the namespace"
