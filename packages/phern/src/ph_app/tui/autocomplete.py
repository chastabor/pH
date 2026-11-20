"""What the prompt can complete, decided without a running app.

Parsing is separate from candidate lookup, and the reason shows under load. The
parse is pure and runs on every keystroke. To list a directory is I/O, and it
must happen only when the token being typed is a path. So
`parse_completion_token` answers *what is being completed*, and
`build_completion_state` consults the candidate sources only for that kind.

The command source is `PHTuiApp.commands`, the same list that the palette and
the dispatch read. Nothing here keeps a second list of what exists (I7).

@module ph_app.tui.autocomplete
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CompletionItem",
    "CompletionKind",
    "CompletionRequest",
    "CompletionState",
    "PathCompleter",
    "build_completion_state",
    "parse_completion_token",
]

CompletionKind = Literal["command", "file"]

MAX_ITEMS = 12
"""A list longer than the popup is a list nobody reads."""


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """A token the user is part-way through typing."""

    kind: CompletionKind
    prefix: str
    start: int
    """Offset of the sigil (`/`, `@`) in the prompt text."""
    end: int


@dataclass(frozen=True, slots=True)
class CompletionItem:
    """One candidate."""

    label: str
    detail: str = ""
    insert: str = ""
    """What is written into the prompt. Defaults to `label`."""

    @property
    def text(self) -> str:
        return self.insert or self.label


@dataclass(frozen=True, slots=True)
class CompletionState:
    """The candidates for one request, and how to apply one."""

    request: CompletionRequest | None = None
    items: tuple[CompletionItem, ...] = ()
    source: str = ""

    def replace(self, item: CompletionItem) -> str:
        """The prompt text with the in-progress token replaced by `item`."""
        if self.request is None:
            return self.source
        request = self.request
        head = self.source[: request.start]
        tail = self.source[request.end :]
        return f"{head}{item.text}{tail}"


def parse_completion_token(text: str) -> CompletionRequest | None:
    """The token at the end of `text`, if it is one that pH completes.

    A command completes only as the first thing in the prompt. A `/` inside a
    sentence is a path separator or a date, and to offer commands there would
    fire constantly while someone types prose.
    """
    if not text or text[-1].isspace():
        return None
    line_start = text.rfind("\n") + 1
    token_start = max(line_start, text.rfind(" ") + 1, text.rfind("\t") + 1)
    token = text[token_start:]
    if not token:
        return None
    if token.startswith("/"):
        if token_start != 0:
            return None
        return CompletionRequest("command", token[1:], token_start, len(text))
    if token.startswith("@"):
        return CompletionRequest("file", token[1:], token_start, len(text))
    return None


def build_completion_state(
    text: str,
    *,
    commands: Sequence[tuple[str, str]] = (),
    paths: Callable[[str], Sequence[str]] | None = None,
) -> CompletionState:
    """Candidates for whatever `text` ends with. Empty state means no popup."""
    request = parse_completion_token(text)
    if request is None:
        return CompletionState(source=text)
    items: list[CompletionItem] = []
    if request.kind == "command":
        prefix = request.prefix.lower()
        items = [
            CompletionItem(label=f"/{name}", detail=summary, insert=f"/{name} ")
            for name, summary in commands
            if name.lower().startswith(prefix)
        ]
    elif request.kind == "file" and paths is not None:
        items = [
            CompletionItem(label=f"@{path}", detail="", insert=f"@{path}")
            for path in paths(request.prefix)
        ]
    return CompletionState(request=request, items=tuple(items[:MAX_ITEMS]), source=text)


@dataclass(slots=True)
class PathCompleter:
    """Directory listing for `@path` completion, one level at a time.

    To walk a whole repository on each keystroke is the obvious way to write
    this, and it is the wrong one: a large checkout makes the prompt stutter.
    A listing of only the directory that the prefix names is bounded by the
    size of that directory. `os.scandir` answers `is_dir()` from the directory
    entry itself, so a large directory costs one syscall and not one for each
    entry.
    """

    root: str
    limit: int = MAX_ITEMS

    def __call__(self, prefix: str) -> Sequence[str]:
        directory, _, stem = prefix.rpartition("/")
        base = os.path.join(self.root, directory) if directory else self.root
        needle = stem.lower()
        try:
            with os.scandir(base) as entries:
                matches = [
                    entry
                    for entry in entries
                    if not entry.name.startswith(".")
                    if not needle or entry.name.lower().startswith(needle)
                ]
        except OSError:
            return ()
        matches.sort(key=lambda entry: (not entry.is_dir(), entry.name))
        found: list[str] = []
        for entry in matches[: self.limit]:
            relative = f"{directory}/{entry.name}" if directory else entry.name
            found.append(f"{relative}/" if entry.is_dir() else relative)
        return found
