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
