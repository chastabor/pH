"""Every picker's rows, built from pH's own registries.

These are builders, not screens. Each one asks a service what exists and turns
the answer into `Choice` rows — the commands come from `ctx.commands`, the
providers from `ctx.llm`, the postures from `PRESETS`, the sessions from the
store's directory. Nothing here keeps a parallel list of what pH can do, so a
plugin that registers a command or an adapter shows up in the picker without
this module changing (I1, I7).

Sessions are listed as a tree. Forks make them one — the header records
`parent_session` — and a flat list of ids hides which session came from which,
so children are indented under their parent. A session with no fork is a tree
with one node, and the list reads the same either way.

@module ph_app.tui.modals.pickers
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from rich.filesize import decimal

from ph.seams.permission_presets import PRESETS

from ...sessions import SessionSummary
from ..themes import ThemeCatalog
from .base import Choice

__all__ = [
    "command_choices",
    "model_choices",
    "preset_choices",
    "session_choices",
    "theme_choices",
]


def command_choices(definitions: Iterable[Any]) -> list[Choice]:
    """Every registered slash command.

    Definitions rather than the registry: the terminal asks its `FrontSession`
    for them, which is what lets a front end in another process answer with a
    list off the wire instead of a service it cannot hold.
    """
    return [
        Choice(
            value=f"/{definition.name}",
            label=f"/{definition.name} {definition.argument_hint}".rstrip(),
            detail=definition.summary,
        )
        for definition in definitions
    ]


def preset_choices(active: str) -> list[Choice]:
    """The permission postures, with the live one marked."""
    return [
        Choice(
            value=preset.name,
            label=preset.name,
            detail=preset.summary,
            marked=preset.name == active,
        )
        for preset in PRESETS.values()
    ]


def theme_choices(active: str, catalog: ThemeCatalog) -> list[Choice]:
    """Every theme the catalog holds, saying which are the user's own."""
    return [
        Choice(
            value=name,
            label=name,
            detail="user" if name in catalog.user else "built-in",
            marked=name == active,
        )
        for name in catalog.names
    ]


def model_choices(
    providers: Sequence[str], active_provider: str, active_model: str
) -> list[Choice]:
    """The registered providers, the active one carrying its model.

    pH has no model catalogue — a provider knows its own models and Phase 1
    deliberately did not invent a list to go stale. So the rows are providers,
    and the picker's free-text entry is how a model is named: typing
    `anthropic/claude-opus-5` offers itself as the value.
    """
    return [
        Choice(
            value=f"{name}/{active_model}" if name == active_provider else name,
            label=name,
            detail="active" if name == active_provider else "provider",
            marked=name == active_provider,
        )
        for name in providers
    ]


def session_choices(sessions: Sequence[SessionSummary], *, current: str = "") -> list[Choice]:
    """Stored sessions, newest first, **branches** indented under their parent.

    Two kinds of parent link reach this list and they must not render alike. A
    fork is a second conversation and belongs one level in. A segment is the
    *same* conversation in a new file — so a session rolled three times is one
    row, not a staircase of three carrying one inherited title between them.

    So: contract every segment edge, keep every fork edge. Each chain of segments
    collapses to its **tip**, which is both the newest file and the one a resume
    should open; the rows a person picks from are one per conversation.

    A child whose parent is not on disk is shown at the root rather than hidden:
    losing a session from the list is worse than showing it in the wrong place.

    Takes rows rather than anything to read them from: `sessions/browse` folds
    them on the harness — stored logs and live roots in one list, each row
    carrying the `state` the daemon calls it — so this function only arranges.
    """
    summaries = {summary.session_id: summary for summary in sessions}
    under = _contract_segments(summaries)
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for session_id in summaries:
        if session_id not in under:
            continue  # a superseded segment; its tip carries the row
        parent = under[session_id]
        if parent is None:
            roots.append(session_id)
        else:
            children.setdefault(parent, []).append(session_id)

    rows: list[Choice] = []

    def walk(session_id: str, depth: int) -> None:
        summary = summaries[session_id]
        rows.append(_session_row(summary, depth, marked=session_id == current))
        for child in children.get(session_id, ()):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return rows


def _contract_segments(summaries: dict[str, SessionSummary]) -> dict[str, str | None]:
    """Every surviving row, mapped to the row it branches from (`None` = a root).

    A segment chain is one conversation, so it gets one row: the tip. Absence
    from this map is what marks a superseded segment — and mapping a *branch* to
    a surviving row is what lets a fork of an earlier segment land under the
    conversation it came from rather than under a file the list no longer shows.

    One map rather than two, because the two halves were only ever used together
    and the caller had to compose them to ask either question.

    The branch parent is read from the chain's **origin** — the only member whose
    parent link is a fork rather than a continuation — so a rolled session stays
    exactly where it was before it rolled.
    """
    continues = {
        summary.parent: summary.session_id
        for summary in summaries.values()
        if summary.kind == "segment" and summary.parent in summaries
    }
    # Hoisted: this was a linear scan of the values *inside* the loop below, so
    # the contraction was quadratic in the number of listed sessions.
    continued = set(continues.values())
    represents: dict[str, str] = {}
    origin_of: dict[str, str] = {}
    for origin in summaries:
        if origin in continued:
            continue  # not a chain head; it is reached from its own origin
        chain, node = [origin], origin
        while (following := continues.get(node)) is not None and following not in chain:
            chain.append(following)
            node = following
        for member in chain:
            represents[member] = node
        origin_of[node] = origin
    # A segment *cycle* has no head, so nothing above reached it. Each member
    # then stands for itself rather than vanishing from the list — this view's
    # rule is that losing a session is worse than showing it in the wrong place,
    # and dropping the whole map on a corrupt header is the worst version of
    # losing one.
    stranded = [one for one in summaries if one not in represents]
    for one in stranded:
        represents[one] = one
        origin_of[one] = one
    branched: dict[str, str | None] = {}
    for tip, origin in origin_of.items():
        parent = summaries[origin].parent
        # A cycle member is shown at the root. Hanging it under its own partner
        # would put both inside the children graph with nothing above them, and
        # the render walk starts from roots — so the rows would not merely be
        # misplaced, they would all disappear.
        branched[tip] = None if tip in stranded else represents.get(parent) if parent else None
    return branched


def _session_row(summary: SessionSummary, depth: int, *, marked: bool) -> Choice:
    indent = f"{'  ' * depth}{'↳ ' if depth else ''}"
    # A running session's size and mtime are whatever the last flush left, so
    # they would describe the file rather than the conversation. The state is
    # what a person is choosing on.
    detail = (
        "running" if summary.state == "running" else f"{summary.when} · {decimal(summary.size)}"
    )
    if summary.cwd:
        detail = f"{detail} · {summary.cwd}"
    return Choice(
        value=summary.session_id,
        label=f"{indent}{summary.title or summary.session_id}",
        detail=detail,
        marked=marked,
    )
