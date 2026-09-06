"""P6-10 — the seam index cannot fall behind the seams.

The row's gate is *every seam has a page*, and the honest state is that every
seam has an **entry**. This is the enforceable half of that, on P6-02's ratchet
argument: a documentation debt that is a number a test checks is legible and
shrinks; one that is an aspiration in a plan row does not.

**Why a test rather than care.** A doc index is the thing nobody re-reads. The
seam it stops describing is the newest one — exactly the one a reader most needs
listed — and nothing about adding a seam would otherwise make anyone open this
file. Reading the directory instead means the *absence* of an entry is what
breaks, which is the same inversion `test_conformance` makes about protocol
frames and `test_a_listing_row_says_the_same_thing_from_either_backend` makes
about `StoredSession`'s fields.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
"""The checkout root: `<repo>/packages/ph-core/tests/this file`."""

SEAMS = REPO / "packages/ph-core/src/ph/seams"
INDEX = REPO / "docs/seams/README.md"


def _modules() -> list[Path]:
    """Every seam module. `_registry` and friends are helpers, not seams."""
    return sorted(path for path in SEAMS.glob("*.py") if not path.name.startswith("_"))


def test_the_index_names_every_seam_module() -> None:
    """A seam added without an entry fails here rather than going undocumented.

    Matched on the module path (`seams/fs.py`) rather than on the service key,
    because several modules publish no key of their own — a workspace tier, a
    sandbox backend, a runtime invariant — and those are the ones a hand-written
    list forgets first.
    """
    index = INDEX.read_text(encoding="utf-8")
    missing = [path.name for path in _modules() if f"seams/{path.name}" not in index]

    assert not missing, (
        f"docs/seams/README.md does not name: {', '.join(missing)}. "
        "A seam with no entry is one a reader cannot find."
    )


def test_the_index_invents_no_seams() -> None:
    """The other direction: a module renamed or removed leaves a dead entry.

    Worth pinning because the failure is quiet in the useful direction — the
    index goes on describing something that no longer exists, and a reader
    follows it to a file that is not there.
    """
    index = INDEX.read_text(encoding="utf-8")
    named = set(re.findall(r"seams/([a-z_]+\.py)", index))
    actual = {path.name for path in _modules()}

    assert named <= actual, f"named in the index but not in the tree: {sorted(named - actual)}"


def test_every_seam_module_opens_with_a_summary_line() -> None:
    """The index quotes each module's first docstring line, so there has to be one.

    Also the reason the index can stay a quotation rather than a paraphrase: the
    module is the authority, and a summary written twice is one free to drift
    from the code it describes.
    """
    without = [
        path.name
        for path in _modules()
        if not (ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or "").strip()
    ]

    assert not without, f"seam modules with no docstring: {', '.join(without)}"


# ------------------------------------------------------------ the pages --
#
# The index above is a list; a page makes *claims about an API*, which is the
# kind of documentation that rots silently. A renamed method leaves prose that
# still reads correctly and no longer describes anything.


def _pages() -> list[Path]:
    return sorted(path for path in (REPO / "docs/seams").glob("*.md") if path.name != "README.md")


def _module_of(page: Path) -> str:
    """The module a page names in its own header, as an import path.

    Any path under `ph/`, not just `ph/seams/`: `ctx.tools` is core rather than a
    pluggable capability and lives in `ph/tools/registry.py`, and a reference
    that only admitted seam modules would have had no home for the page every
    other page links to.
    """
    found = re.search(r"\*\*Module:\*\* `(ph/[a-z_/]+)\.py`", page.read_text(encoding="utf-8"))
    assert found, f"{page.name} does not name its module in a `**Module:**` line"
    return found.group(1).replace("/", ".")


def test_a_seam_page_only_names_methods_that_exist() -> None:
    """`ctx.fs.read(...)` in prose has to still be a method.

    Checked against the classes the page's own module defines, so the page is
    verified against the code it documents rather than against a copy of it.
    Only calls on the page's *own* key are checked — a page may legitimately
    mention another seam, and that seam's page is where it is verified.
    """
    import importlib
    import inspect

    for page in _pages():
        module = importlib.import_module(_module_of(page))
        available = {
            name
            for _, obj in inspect.getmembers(module, inspect.isclass)
            for name in dir(obj)
            if not name.startswith("__")
        } | {name for name, _ in inspect.getmembers(module, inspect.isfunction)}

        # The page's own key, plus any it declares. A small service is sometimes
        # better documented inside the page for the thing it configures —
        # `ctx.subagent_presets` in `subagents.md` — and an undeclared second key
        # would otherwise be the one part of a page nothing checks.
        text = page.read_text(encoding="utf-8")
        keys = {page.stem} | {
            found.removeprefix("ctx.")
            for found in re.findall(r"\*\*Also documents:\*\* `(ctx\.[a-z_]+)`", text)
        }
        named = {method for key in keys for method in re.findall(rf"ctx\.{key}\.([a-z_]+)\(", text)}
        missing = sorted(named - available)

        assert not missing, f"{page.name} documents methods that do not exist: {missing}"


def test_a_seam_page_declares_which_row_mounts_it() -> None:
    """The row name is how a reader turns the page into a profile line.

    Pinned because it is the one fact a page cannot be useful without and the
    easiest to leave out: the seam's own docstring never mentions it, since a
    module does not know which row id a profile gave it.
    """
    for page in _pages():
        text = page.read_text(encoding="utf-8")
        # `Rows:` too, because a seam legitimately needs more than one — the
        # definition and a backend (`sandbox-policy` plus `sandbox-local`), or a
        # service and its presets. Requiring the singular would have pushed those
        # pages into naming one row and leaving the other to prose.
        assert "**Row:**" in text or "**Rows:**" in text, (
            f"{page.name} does not say which row mounts the seam"
        )


def test_every_python_block_in_the_docs_parses() -> None:
    """A snippet nobody runs is a snippet that stops being valid Python.

    Cheap, and it catches the copy-paste that dropped a bracket — which is the
    failure a reader hits first and reports last.
    """
    broken: list[str] = []
    for page in sorted((REPO / "docs").rglob("*.md")):
        for index, block in enumerate(re.findall(r"```python\n(.*?)```", page.read_text(), re.S)):
            try:
                ast.parse(block)
            except SyntaxError as error:
                broken.append(f"{page.relative_to(REPO)} block {index}: {error}")

    assert not broken, "\n".join(broken)


def test_every_service_key_in_the_index_has_a_page() -> None:
    """**P6-10's gate, enforced rather than merely met.**

    The row asks that every seam have a page, and as of now every one does. A
    state that is true on the day it is reached and unchecked afterwards is the
    same debt again the moment somebody adds a seam — so the index's own service
    rows are read back, and an entry that is not a link fails here.

    Matched on the *table row* rather than on a list of keys: the index is the
    thing a reader uses, so what is pinned is that the reader can get from any
    service to its page.
    """
    text = INDEX.read_text(encoding="utf-8")
    rows = re.findall(r"^\| (\[?`ctx\.[a-z_]+`\]?[^|]*)\|", text, re.M)
    assert rows, "no service rows found — the index's table shape changed"

    unlinked = [row.strip() for row in rows if not row.strip().startswith("[")]

    assert not unlinked, (
        f"service keys with no page: {unlinked}. Write one under docs/seams/ and "
        "link it, or — for a service documented beside another — link that page."
    )
