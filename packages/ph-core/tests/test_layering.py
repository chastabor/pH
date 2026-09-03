"""P0-01 — `ph-core` stays free of front-end dependencies.

The layering rule from §9: `ph-core` may not import Textual, Rich or Typer.
It is enforced by a test rather than a convention because the cost of breaking
it is invisible until someone tries to run the harness headless, in a daemon,
or inside a runtime venv that has no terminal libraries at all.

## What nesting an agent's scope inside its parent's replaced (P6-27)

Every agent used to hang off the *registry*, so a parent and its child were
**siblings** and `parent.ctx.reaches(child.ctx)` was `False`. The relationship
`SessionHeader.parent_session` records, and that B7 is entirely about, had no
representation in the tree that answers questions about it — so `ctx.subagents`
rebuilt the ceiling by hand on every spawn, and `grant_for`'s docstring had to
explain that writing `None` out as an explicit list "is not a nicety".

The subagent provider's `parent.ctx.effect(...)` was likewise doing by hand what
the tree now does by shape.
"""

from __future__ import annotations

import ast
import pathlib

import ph

FORBIDDEN = {"textual", "rich", "typer", "ph_app", "aiohttp", "textual_serve", "jinja2"}
"""Front-end libraries ph-core has no business with, and the app layer above it.

**Not "optional dependencies"** — that rule would be wrong in both directions.
`opentelemetry` is an extra (`ph-core[otel]`) that ph-core imports deliberately
and lazily inside `telemetry_otel.py`, and `rich` is not optional at all. What
these seven share is that they belong to a *presentation* layer: a core module
reaching for one would make the harness's own imports depend on how somebody
chose to look at it. `aiohttp`, `textual_serve` and `jinja2` joined with P7-05.

Whether the *app* layer may import them is a different question with a different
answer, asked by `packages/ph-app/tests/test_app_layering.py`."""


def _core_modules() -> list[pathlib.Path]:
    root = pathlib.Path(ph.__path__[0])
    return sorted(root.rglob("*.py"))


def test_no_front_end_imports_in_ph_core() -> None:
    offenders: list[str] = []
    for path in _core_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in FORBIDDEN:
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")
    assert offenders == [], offenders


def test_every_core_module_has_a_docstring() -> None:
    missing = [
        path.name
        for path in _core_modules()
        if ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) is None
    ]
    assert missing == [], missing
