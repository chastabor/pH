"""A person's named profile: `extends` a shipped profile, and the rows that differ (S2).

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

**Sparse** (decision 8): only the rows that differ from the profile it extends,
which supplies every default. Composed as that profile's shipped layers and then
these rows, which set environment rows only (decision 23). `extends` names a
shipped profile — one level, never another person's file — and may be left out of
a file named after the shipped profile it layers over: `tui.yaml` extends `tui`.

**The format before S2** was a bare list of rows under a shipped profile's name.
Until it is folded it is read as it always was, and `phern doctor` names it;
`phern profiles fold` rewrites it into this one and keeps its comments, which is
why the edits here are to text rather than to a parsed tree a dump would flatten.

The format only. Which file a name resolves to, and what composes over what, is
`ph_app.profiles`'.

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
    """One named profile file, read."""

    name: str
    """The name it is run by — its file's stem — or `""` for a `--profile` path."""
    extends: str
    """The shipped profile whose layers it composes over."""
    rows: JsonValue
    """Its entries in the profile grammar, still raw: `compose_rows` reads them."""
    path: Path
    legacy: bool = False
    """A bare list — the overlay format before S2 — not yet folded."""


def parse_named_profile(
    raw: JsonValue, path: Path, name: str, *, shipped: Collection[str]
) -> NamedProfile:
    """`raw`, the decoded file at `path`, as the named profile `name` it declares.

    `shipped` is every profile `extends` may name. A list is the old overlay, so it
    is only a profile under a shipped profile's own name, which it layers over.
    """
    if raw is None or isinstance(raw, list):
        if name not in shipped:
            raise LoaderError(
                f"{path}: a list of rows layers over the shipped profile of the same name, "
                f'and there is no shipped "{name}"; write `extends: <profile>` and `rows:`'
            )
        return NamedProfile(name=name, extends=name, rows=raw or [], path=path, legacy=True)
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
    return NamedProfile(name=name, extends=extends, rows=rows or [], path=path)


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
    """`text` — the file `named` was read from — with `entries` appended to its rows.

    Appended as text, so the person's own comments and layout stay, and every edit
    is one a later entry for the same row survives: a later entry replaces an
    earlier one in the same document. A legacy list becomes the mapping by
    indenting it under `rows:`. This is a best effort over YAML a person wrote —
    the caller composes the result and puts the file back when it differs.

    :raises LoaderError: for rows written as a flow sequence, which text cannot
        append to without rewriting the line a person wrote.
    """
    block = _indented(_comment(block_comment), 2) + _rows_text(entries, indent=2)
    if named.legacy:
        lines = text.splitlines()
        kept = lines if named.rows else [line for line in lines if line.lstrip().startswith("#")]
        indented = "".join(f"  {line}\n" if line.strip() else "\n" for line in kept)
        return f"extends: {named.name}\nrows:\n{indented}{block}"
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
