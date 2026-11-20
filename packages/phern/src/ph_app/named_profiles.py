"""A named profile of the user: it `extends` a shipped profile and holds the rows
that differ (S2).

```yaml
# $PH_HOME/profiles/work.yaml
extends: rlm
rows:
  - id: models
    config:
      default: main
      models:
        main: {provider: anthropic, model: claude-sonnet-5}
```

**The file is sparse** (decision 8). It holds only the rows that differ from the
profile that it extends, and that profile supplies every default. It composes as
the shipped layers of that profile and then these rows. These rows set
environment rows only (decision 23). `extends` names a shipped profile. It is
one level deep, and never names another file that a user wrote. `extends` may be
left out of a file that is named after the shipped profile that it layers over:
`tui.yaml` extends `tui`.

**The format before S2** was a bare list of rows under the name of a shipped
profile. Until such a file is folded, it is read as it always was, and
`phern doctor` names it. `phern profiles fold` rewrites it into the format above
and keeps its comments. That is why the edits in this module are edits to text
and not to a parsed tree, which a dump would flatten.

This module holds the format only. `ph_app.profiles` decides which file a name
resolves to, and what composes over what.

@module ph_app.named_profiles
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from ph.cordis import LoaderError
from ph.json import JsonObject, JsonValue, thaw_json

__all__ = [
    "NamedProfile",
    "append_rows",
    "parse_named_profile",
    "render_named_profile",
]


@dataclass(frozen=True, slots=True)
class NamedProfile:
    """One named profile file, read. The name that it is run by is the stem of
    its `path`, for a file under `$PH_HOME/profiles`. A `--profile` path is run
    by its path and has no name."""

    extends: str
    """The shipped profile whose layers it composes over."""
    rows: JsonValue
    """Its entries in the profile grammar, still raw: `compose_rows` reads them."""
    path: Path
    legacy: bool = False
    """A bare list, the overlay format from before S2, that is not folded yet."""


def parse_named_profile(
    raw: JsonValue, path: Path, name: str | None, *, shipped: Collection[str]
) -> NamedProfile:
    """`raw`, the decoded file at `path`, as the named profile `name` that it
    declares. `name` is `None` for a file that is run by its path.

    `shipped` is every profile that `extends` may name. A list is the old
    overlay. It is a profile only under the name of a shipped profile, and it
    layers over that profile.
    """
    if raw is None or isinstance(raw, list):
        if name is None or name not in shipped:
            missing = (
                "a file that is run by its path has no name"
                if name is None
                else f'there is no shipped "{name}"'
            )
            raise LoaderError(
                f"{path}: a list of rows layers over the shipped profile of the same name, "
                f"and {missing}; write `extends: <profile>` and `rows:`"
            )
        return NamedProfile(extends=name, rows=raw or [], path=path, legacy=True)
    if not isinstance(raw, Mapping):
        raise LoaderError(f"{path}: a named profile is `extends: <profile>` and `rows: [...]`")
    unknown = sorted(set(raw) - {"extends", "rows"})
    if unknown:
        raise LoaderError(
            f"{path}: unknown keys {unknown}; a named profile holds `extends:` and `rows:`"
        )
    offered = ", ".join(sorted(shipped))
    extends = raw.get("extends", name if name in shipped else None)
    if not isinstance(extends, str) or not extends:
        raise LoaderError(
            f"{path}: `extends` names the shipped profile this one layers over: {offered}"
        )
    if extends not in shipped:
        raise LoaderError(f'{path}: extends "{extends}", which is not a shipped profile: {offered}')
    if name in shipped and extends != name:
        raise LoaderError(
            f'{path}: a file named after the shipped "{name}" layers over it, so it extends '
            f'"{name}"; give it another name to extend "{extends}"'
        )
    rows = raw.get("rows")
    if rows is not None and not isinstance(rows, list):
        raise LoaderError(f"{path}: `rows` is a list of profile entries")
    return NamedProfile(extends=extends, rows=rows or [], path=path)


def _rows_text(entries: Sequence[JsonObject], *, indent: int) -> str:
    """Entries as a YAML block sequence, each line indented by `indent` spaces."""
    if not entries:
        return ""
    block = yaml.safe_dump(thaw_json(list(entries)), sort_keys=False, default_flow_style=False)
    return _indented(block, indent)


def _comment(text: str) -> str:
    """`text` as YAML comment lines, a blank line as a bare `#`."""
    return "".join(f"# {line}\n" if line else "#\n" for line in text.splitlines())


def render_named_profile(extends: str, entries: Sequence[JsonObject], *, comment: str) -> str:
    """The whole text of a saved named profile: a comment, `extends`, and the rows."""
    body = _rows_text(entries, indent=2)
    return f"{_comment(comment)}extends: {extends}\nrows:{'' if body else ' []'}\n{body}"


_TOP_LEVEL_ROWS = re.compile(r"^rows:[ \t]*(?P<value>[^#\n]*?)[ \t]*(?:#.*)?$", re.MULTILINE)
_ITEM = re.compile(r"^(?P<indent>[ \t]*)- ", re.MULTILINE)


def append_rows(
    text: str, named: NamedProfile, block_comment: str, entries: Sequence[JsonObject]
) -> str:
    """`text`, the file that `named` was read from, with `entries` appended to its
    rows.

    The entries are appended as text, so the comments and the layout of the user
    stay. An appended entry also takes effect when the file already has an entry
    for the same row, because a later entry replaces an earlier one in the same
    document. A legacy list becomes the mapping: it is indented under `rows:`.
    This is a best effort over YAML that a user wrote. The caller composes the
    result and puts the file back when the result differs.

    :raises LoaderError: for rows written as a flow sequence, which text cannot
        append to without rewriting the line a person wrote.
    """
    block = _indented(_comment(block_comment), 2) + _rows_text(entries, indent=2)
    if named.legacy:
        lines = text.splitlines()
        kept = lines if named.rows else [line for line in lines if line.lstrip().startswith("#")]
        indented = "".join(f"  {line}\n" if line.strip() else "\n" for line in kept)
        # A legacy list extends the shipped profile that it is named after.
        return f"extends: {named.extends}\nrows:\n{indented}{block}"
    body = text if text.endswith("\n") else text + "\n"
    found = _TOP_LEVEL_ROWS.search(body)
    if found is None:
        return f"{body}rows:\n{block}"
    value = found.group("value")
    if value == "[]":
        return f"{body[: found.start()]}rows:{body[found.end() :]}{block}"
    if value:
        raise LoaderError(f"{named.path}: `rows:` is a flow sequence, which cannot be appended to")
    item = _ITEM.search(body[found.end() :])
    indent = len(item.group("indent")) if item is not None else 2
    return f"{body}{_indented(_comment(block_comment), indent)}{_rows_text(entries, indent=indent)}"


def _indented(text: str, indent: int) -> str:
    return "".join(f"{' ' * indent}{line}\n" for line in text.splitlines())
