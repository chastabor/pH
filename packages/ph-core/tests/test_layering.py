"""P0-01 — `ph-core` stays free of front-end dependencies.

The layering rule from §9: `ph-core` may not import Textual, Rich or Typer.
It is enforced by a test rather than a convention because the cost of breaking
it is invisible until someone tries to run the harness headless, in a daemon,
or inside a runtime venv that has no terminal libraries at all.
"""

from __future__ import annotations

import ast
import pathlib

import ph

FORBIDDEN = {"textual", "rich", "typer", "ph_app"}


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
