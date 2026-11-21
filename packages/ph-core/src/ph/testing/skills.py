"""A `SKILL.md` on disk, written the same way in every package.

The format belongs to `ph.seams.skills`. Two packages test against it: ph-core
tests the reader of the seam itself, and ph-rlm tests the package half of
`rlm-skills-python`. The ph-stabilize suite and the repository-level live
suites also write skills with this builder. The builder was first written out
in ph-rlm. It is here so that every package can reach it, before another copy
decides that frontmatter looks slightly different.

@module ph.testing.skills
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..seams.skills import Skill

__all__ = ["KIND_OUTPUT", "skill", "write_skill"]

KIND_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {"kind": {"type": "string", "enum": ["bug", "feature"]}},
    "required": ["kind"],
    "additionalProperties": False,
}
"""The `output` of a sorting skill: which kind of report its child read. The
typed-result tests in ph-core and ph-rlm, and the live suite in
`tests/test_local_typed_results.py`, all return this one shape, so they cannot
drift apart."""


def write_skill(
    root: Path,
    name: str,
    *,
    description: str = "search the web for a query",
    front_name: str | None = None,
    body: str = "Instructions here.",
    extra: str = "",
) -> Path:
    """Write one skill directory and return it.

    `front_name` writes a frontmatter name that disagrees with the directory
    name.

    `extra` is frontmatter that is appended after `description`: the optional
    fields (`version`, `argument-hint`, `allowed-tools`). It is here and not in
    the test that wanted it first, for the reason that the module docstring
    gives: with one more writer, frontmatter starts to look slightly different.
    """
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    front = [
        f"name: {front_name if front_name is not None else name}",
        f"description: {description}",
    ]
    if extra:
        front.append(extra)
    (directory / "SKILL.md").write_text(
        "---\n" + "\n".join(front) + f"\n---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    return directory


def skill(
    name: str,
    *,
    description: str = "does a thing",
    hint: str = "",
    model: str = "",
    effort: str = "",
    output: dict[str, Any] | None = None,
    accepts: Sequence[str] = (),
) -> Skill:
    """One registered skill, with no file behind it.

    Four arguments describe the child that the skill directs:

    - `model`: the key of the model that the skill asks the child to run on.
    - `effort`: the reasoning effort that the skill asks the child to run at.
    - `output`: the shape of the value that the child returns.
    - `accepts`: the media that the skill hands to the child.

    This builder is for the tests that are about the *registry* (visibility,
    narrowing, the catalog) and not about the format on disk. It was written
    out in three modules before it moved here, as `write_skill` also moved
    here from a test.
    """
    return Skill(
        name=name,
        description=description,
        hint=hint,
        model=model,
        effort=effort,
        output=output or {},
        accepts=list(accepts),
    )
