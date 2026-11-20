"""Compile one cell so that `await`, `return`, and persistence all work.

Three requirements conflict. How this module reconciles them is its only
interesting part.

1. **`await` at the top level.** The program is a coroutine on the one event loop
   of the child process. There is no `nest_asyncio` question and no second loop
   to re-enter.
2. **`return` at the top level.** A bare `return` is a syntax error in module
   code, even with `PyCF_ALLOW_TOP_LEVEL_AWAIT`. The body is therefore wrapped in
   an `async def`, which also gives (1) at no cost.
3. **Names persist across cells.** A name that is assigned inside a function is
   *local* to that function. A simple wrapper would lose every variable that the
   cell defined, and the persistent namespace would have no purpose.

The wrapper therefore declares as `global` every name that the cell binds at its
top level. This module computes that set of bound names from the AST. The set
includes the cases that are easy to forget: `import`, `with ... as`, `for`,
`except ... as`, the walrus operator, `del`, and function and class definitions.

A trailing expression becomes the value of the cell, as in a REPL. The result
then reads as `stdout / stderr / result / traceback`, the way it did in
prime-agent.

@module ph_runtime.cell
"""

from __future__ import annotations

import ast
from typing import Any

__all__ = [
    "CELL_FILENAME",
    "CELL_FUNCTION",
    "MAGIC_HINT",
    "MAGIC_PREFIXES",
    "bound_names",
    "compile_cell",
]

CELL_FUNCTION = "__ph_cell__"

CELL_FILENAME = "<cell>"
"""The name that the cell is compiled under. A traceback is also trimmed to this
marker, so the model sees its own frames and not the frames of the runner."""

MAGIC_PREFIXES = ("%%", "%", "!")
"""Every IPython escape prefix. The list is exported because three layers describe
this rule.

The guest refuses the prefixes, the RLM doctrine tells the model about the
refusal, and the conformance suite checks that the two agree. A gap for one
prefix would be a hole in the rule. The list therefore has one home, not three
copies that drift."""

MAGIC_HINT = (
    "IPython magics are not available: pH runs plain Python, so there is no "
    "`%%bash` to bypass the tool pipeline. Use `await tools.bash(command=...)` "
    "for a shell, and ordinary Python for the rest."
)
"""The text that is added to the `SyntaxError` that a magic produces (D2).

The magic was a bypass. It ran one shell command per cell that no
`tools/pre-execute` listener, no approval, and no sandbox `confine()` ever saw.
Removing the mechanism closes the hole. The error therefore explains the governed
route and does not apologize for a missing feature.
"""


def bound_names(body: list[ast.stmt]) -> set[str]:
    """Return every name that the *top level* of the cell binds, so that the
    wrapper can declare it global.

    Only the top level counts. A name that is bound inside a nested function or a
    comprehension is local there in module code too. Declaring it global would
    change the meaning of the program.
    """
    found: set[str] = set()

    def add_target(node: ast.expr) -> None:
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Starred):
            add_target(node.value)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for element in node.elts:
                add_target(element)

    def walk_expression(node: ast.expr) -> None:
        # A walrus binds in the enclosing scope. `if (n := len(x)) > 3:` at the top
        # level therefore defines `n` for later cells.
        for child in ast.walk(node):
            if isinstance(child, ast.NamedExpr):
                add_target(child.target)

    for statement in body:
        if isinstance(statement, (ast.Assign,)):
            for target in statement.targets:
                add_target(target)
            walk_expression(statement.value)
        elif isinstance(statement, (ast.AugAssign, ast.AnnAssign)):
            add_target(statement.target)
            if statement.value is not None:
                walk_expression(statement.value)
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(statement.name)
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            for alias in statement.names:
                if alias.name == "*":
                    continue
                found.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            add_target(statement.target)
            found |= bound_names(statement.body) | bound_names(statement.orelse)
        elif isinstance(statement, (ast.While, ast.If)):
            walk_expression(statement.test)
            found |= bound_names(statement.body) | bound_names(statement.orelse)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            for item in statement.items:
                if item.optional_vars is not None:
                    add_target(item.optional_vars)
            found |= bound_names(statement.body)
        elif isinstance(statement, ast.Try):
            found |= bound_names(statement.body) | bound_names(statement.orelse)
            found |= bound_names(statement.finalbody)
            for handler in statement.handlers:
                if handler.name:
                    found.add(handler.name)
                found |= bound_names(handler.body)
        elif isinstance(statement, ast.Match):
            for case in statement.cases:
                found |= bound_names(case.body)
                for child in ast.walk(case.pattern):
                    if isinstance(child, ast.MatchAs | ast.MatchStar) and child.name:
                        found.add(child.name)
                    elif isinstance(child, ast.MatchMapping) and child.rest:
                        found.add(child.rest)
        elif isinstance(statement, ast.Delete):
            # `del x` also needs `global x`. Without it, `del` raises for a name
            # that is in globals but is not local.
            for target in statement.targets:
                add_target(target)
        elif isinstance(statement, (ast.Expr, ast.Return)) and statement.value is not None:
            walk_expression(statement.value)

    return found


def compile_cell(program: str, filename: str = CELL_FILENAME) -> Any:  # noqa: ANN401
    """Compile `program` into a module that defines `CELL_FUNCTION`.

    :raises SyntaxError: the program does not parse. For a magic, `MAGIC_HINT` is
        appended, because that is the one syntax error with a governed answer.
    """
    try:
        tree = ast.parse(program)
    except SyntaxError as error:
        if _looks_like_magic(program):
            raise SyntaxError(f"{error.msg}. {MAGIC_HINT}") from error
        raise

    body = list(tree.body)
    if body and isinstance(body[-1], ast.Expr):
        last = body[-1]
        body[-1] = ast.copy_location(ast.Return(value=last.value), last)

    declarations: list[ast.stmt] = []
    names = sorted(bound_names(list(tree.body)))
    if names:
        declarations.append(ast.Global(names=names))
    if not body:
        body = [ast.Pass()]

    wrapper = ast.AsyncFunctionDef(
        name=CELL_FUNCTION,
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=declarations + body,
        decorator_list=[],
        returns=None,
        type_comment=None,
        type_params=[],
    )
    module = ast.Module(body=[wrapper], type_ignores=[])
    ast.fix_missing_locations(module)
    # `dont_inherit`: a cell is not this module. It keeps plain annotations and not
    # the `__future__` string annotations of this module, which a pydantic model in
    # the cell cannot resolve.
    return compile(module, filename, "exec", dont_inherit=True)


def _looks_like_magic(program: str) -> bool:
    return program.lstrip().startswith(MAGIC_PREFIXES)
