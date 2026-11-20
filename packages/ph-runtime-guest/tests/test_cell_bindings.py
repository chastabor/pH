"""`bound_names` over every binding form, because a missed form fails silently.

**What this function decides.** The body of a cell is wrapped in an `async def`,
so that top-level `await` and `return` both work. A name that is assigned inside
a function is local to it. Without a correction, every variable that a cell
defined would be lost at the end of the cell. `compile_cell` prevents that: it
declares each name that the cell binds as `global` in the wrapper. `bound_names`
computes that set from the AST.

**A form that it misses does not raise.** The cell runs and the assignment
succeeds. But the variable is local to a wrapper that is about to be discarded,
and the *next* cell gets `NameError` on a name that the user just saw assigned.
Nothing in the guest can detect this, because the compiled code is valid Python
in both cases. The rule can therefore be enforced only here, against the list of
syntax forms.

That list is the docstring of the module itself. It names `import`, `with ...
as`, `for`, `except ... as`, walrus, `del`, and function and class definitions.
Each form that it claims is a case below.

**Why this suite exists.** `packages/ph-runtime-guest` had no tests of its own.
The kernel tests of `ph-rlm` exercise its modules thoroughly, across a real
process boundary. That is the correct way to test the half that runs code that a
model wrote. But that path reaches `bound_names` only with the syntax that the
test cells use, which is assignment and little else. The forms below were all
correct, and none of them was covered.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ph_runtime.cell import CELL_FUNCTION, bound_names, compile_cell

pytestmark = pytest.mark.anyio
"""The convention that every async suite here follows: one backend, asyncio, set
in the root `conftest.anyio_backend`. It is harmless on the synchronous tests
below, and the other suites also apply it at module scope."""


def names_in(source: str) -> set[str]:
    """The top-level bound names of a cell written as source."""
    return bound_names(ast.parse(source).body)


# ---------------------------------------------------------------------------
# One case for each form that the module claims.

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
# The other half of the rule. An error here is more dangerous.


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
    """Declaring one of these names `global` would change the meaning of the
    program.

    This failure is the opposite of a miss, and it is worse. A cell that read a
    module-level `e` would find it overwritten with the last value of the
    comprehension. The bug is then a wrong answer and not a `NameError`.
    """
    assert local not in names_in(source), (
        f"`{local}` is local to its scope; globalizing it changes the program"
    )


async def run_cell(source: str, namespace: dict[str, object]) -> object:
    """Compile and run one cell in the way that `runner.Runner` does.

    The wrapper is an `async def`, so executing the compiled module only
    *defines* it. The value comes from awaiting the wrapper with the namespace as
    globals, which is also what makes the `global` declarations apply to that
    namespace.
    """
    exec(compile_cell(source), namespace)  # running a cell is the unit under test
    cell = namespace.pop(CELL_FUNCTION)
    assert callable(cell)
    return await cell()


async def test_the_names_survive_an_actual_round_trip(tmp_path: Path) -> None:
    """The set is useful only if `compile_cell` acts on it.

    An assertion against `bound_names` alone would pass even if the wrapper
    ignored the set, and that failure would make the module pointless. Each name
    here is a form that the `BINDINGS` table covers in isolation. This test makes
    the same claim end to end, through the compile and the await.
    """
    namespace: dict[str, object] = {}
    await run_cell("plain = 1", namespace)
    await run_cell("for loop_value in range(2):\n    pass", namespace)
    await run_cell("import json as parsed_json", namespace)
    # Use a real file, because the namespace of a cell has no `__file__`. A cell
    # is not a module, which is the purpose of the wrapper.
    document = tmp_path / "read-me.txt"
    document.write_text("first line\n", encoding="utf-8")
    await run_cell(
        f"with open({str(document)!r}) as handle:\n    first = handle.readline()", namespace
    )

    for name in ("plain", "loop_value", "parsed_json", "handle", "first"):
        assert name in namespace, f"`{name}` did not reach the persistent namespace"


async def test_a_trailing_expression_is_the_cells_value() -> None:
    """The REPL property. It uses the same rewrite that `bound_names` feeds."""
    namespace: dict[str, object] = {}

    assert await run_cell("total = 2 + 3\ntotal * 2", namespace) == 10
    assert await run_cell("total", namespace) == 5, "the earlier cell's name persisted"


async def test_a_cell_s_annotations_are_its_own_not_the_guest_s() -> None:
    """A cell runs with the annotations of plain Python, not with the `from
    __future__ import annotations` of this package. When the cell inherited that
    import, each annotation was a string. A pydantic model then had to resolve
    the string through a module, and a cell is not a module, so a model that a
    cell defined never built. Sabotage: compile without `dont_inherit`, and the
    annotation is the string `'int'`."""
    namespace: dict[str, object] = {"__name__": "__ph_cell__"}

    annotations = await run_cell("class Point:\n    x: int\nPoint.__annotations__", namespace)

    assert annotations == {"x": int}


async def test_a_name_from_a_nested_scope_does_not_leak_between_cells() -> None:
    """The negative rule, proved where it would cause harm.

    `test_a_name_bound_in_a_nested_scope_is_not_globalized` asserts the set. This
    test asserts the consequence: a comprehension variable must not remain in the
    namespace for the next cell to read.
    """
    namespace: dict[str, object] = {}
    await run_cell("squares = [e * e for e in range(3)]", namespace)

    assert namespace["squares"] == [0, 1, 4]
    assert "e" not in namespace, "the comprehension's variable escaped into the namespace"
