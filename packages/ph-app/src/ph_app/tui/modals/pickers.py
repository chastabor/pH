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

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.filesize import decimal

from ph.seams.permission_presets import PRESETS

from ..sessions import SessionSummary, session_summaries
from ..themes import ThemeCatalog
from .base import Choice

__all__ = [
    "command_choices",
    "model_choices",
    "preset_choices",
    "session_choices",
    "theme_choices",
]


def command_choices(commands: Any) -> list[Choice]:
    """Every registered slash command."""
    return [
        Choice(
            value=f"/{definition.name}",
            label=f"/{definition.name} {definition.argument_hint}".rstrip(),
            detail=definition.summary,
        )
        for definition in commands.list()
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


def session_choices(sessions_dir: Path, *, current: str = "") -> list[Choice]:
    """Stored sessions, newest first, children indented under their fork parent.

    A child whose parent is not on disk is shown at the root rather than hidden:
    losing a session from the list is worse than showing it in the wrong place.
    """
    summaries = {summary.session_id: summary for summary in session_summaries(sessions_dir)}
    children: dict[str, list[str]] = {}
    roots: list[str] = []
    for summary in summaries.values():
        if summary.parent is not None and summary.parent in summaries:
            children.setdefault(summary.parent, []).append(summary.session_id)
        else:
            roots.append(summary.session_id)

    rows: list[Choice] = []

    def walk(session_id: str, depth: int) -> None:
        summary = summaries[session_id]
        rows.append(_session_row(summary, depth, marked=session_id == current))
        for child in children.get(session_id, ()):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)
    return rows


def _session_row(summary: SessionSummary, depth: int, *, marked: bool) -> Choice:
    indent = f"{'  ' * depth}{'↳ ' if depth else ''}"
    detail = f"{summary.when} · {decimal(summary.size)}"
    if summary.cwd:
        detail = f"{detail} · {summary.cwd}"
    return Choice(
        value=summary.session_id,
        label=f"{indent}{summary.title or summary.session_id}",
        detail=detail,
        marked=marked,
    )
