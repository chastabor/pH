"""Snapshot tests: what the four transcript shapes actually look like.

Streaming, a settled tool card, an error, and a compaction marker — the four
things a person reads a transcript for. Snapshots are the only test that catches
a *rendering* regression: a card that lost its glyph, a theme variable that
stopped resolving, a row that started rendering its markup instead of showing it.

Each app is fed a state directly rather than driven through a model, so the
snapshot is stable: no timing, no token order, no session ids in the frame.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from textual.app import App, ComposeResult
from tui_helpers import until

from ph_app.tui.app import PHTuiApp
from ph_app.tui.state import ChatItem, ToolCard, TuiState
from ph_app.tui.themes import DEFAULT_THEME, fallback_variables, load_catalog
from ph_app.tui.widgets.transcript import TranscriptView


class _Snapshot(App[None]):
    """A transcript and nothing else, so a diff points at the rows."""

    CSS = """
    Screen { background: $ph-background; color: $ph-foreground; }
    """

    def __init__(self, items: list[ChatItem], theme: str = DEFAULT_THEME) -> None:
        super().__init__()
        self.items = items
        self._theme_name = theme

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return fallback_variables()

    def compose(self) -> ComposeResult:
        yield TranscriptView(id="transcript")

    async def on_mount(self) -> None:
        load_catalog().install(self)
        self.theme = self._theme_name
        await self.query_one(TranscriptView).sync(self.items)


def _state(*items: ChatItem) -> list[ChatItem]:
    state = TuiState()
    for item in items:
        state.add(item)
    return state.visible_items()


# ------------------------------------------------------------------------------


def test_streaming_assistant_text(snap_compare: Any) -> None:
    items = _state(
        ChatItem(key="u1", role="user", text="explain the pipeline", seq=1),
        ChatItem(
            key="a1",
            role="assistant",
            text="The pipeline runs **pre-execute**, then guards, then the body.",
            streaming=True,
            seq=2,
        ),
    )
    assert snap_compare(_Snapshot(items), terminal_size=(80, 14))


def test_a_settled_tool_card(snap_compare: Any) -> None:
    card = ToolCard(
        call_id="c1",
        name="read",
        arguments='{"path": "src/ph/tools/registry.py"}',
        title="read",
        subtitle="src/ph/tools/registry.py",
        settled=True,
        body="1  from __future__ import annotations",
    )
    items = _state(
        ChatItem(key="u1", role="user", text="read the registry", seq=1),
        ChatItem(key="t1", role="tool", tool=card, seq=2),
    )
    assert snap_compare(_Snapshot(items), terminal_size=(80, 14))


def test_a_failed_tool_card_and_an_error_row(snap_compare: Any) -> None:
    card = ToolCard(
        call_id="c1",
        name="bash",
        arguments='{"command": "make test"}',
        title="bash",
        subtitle="make test",
        settled=True,
        is_error=True,
        failure_kind="denied",
        body="rejected: writes outside the workspace",
    )
    items = _state(
        ChatItem(key="t1", role="tool", tool=card, seq=1),
        ChatItem(key="e1", role="error", text="the provider returned 429 after 3 retries", seq=2),
    )
    assert snap_compare(_Snapshot(items), terminal_size=(80, 14))


def test_a_code_mode_card_lists_its_dispatches(snap_compare: Any) -> None:
    """Forty writes are forty rows (C2) — here, two."""
    card = ToolCard(
        call_id="c1",
        name="run_code",
        arguments="{}",
        title="run_code",
        subtitle="2 calls",
        settled=True,
        body="ok",
        dispatches=[
            ToolCard(call_id="s1", name="read", arguments='{"path": "a.py"}', settled=True),
            ToolCard(
                call_id="s2",
                name="write",
                arguments='{"path": "b.py"}',
                settled=True,
                is_error=True,
            ),
        ],
    )
    items = _state(ChatItem(key="t1", role="tool", tool=card, seq=1))
    assert snap_compare(_Snapshot(items), terminal_size=(80, 14))


def test_a_code_cell_shows_its_program_and_what_it_cost(snap_compare: Any) -> None:
    """P3-19: the program is the interesting half of a cell.

    Every other card is a header and a body, because its input fits on the header
    line. A cell's input is the code the model wrote. The facts line under it
    carries what the collapsible does not — the dispatch count is the
    collapsible's, from the fold that owns those rows, so one number is never
    rendered twice from two folds inside one widget (A11).
    """
    card = ToolCard(
        call_id="c1",
        name="ipython",
        arguments='{"program": "..."}',
        title="ipython",
        subtitle="3 lines",
        card="terminal",
        input_text="rows = await tools.read(path='data.csv')\ntotal = len(rows)\ntotal",
        settled=True,
        body="[result] 128",
        details={"status": "ok", "dispatches": 2, "attachments": 1, "truncated": True},
        dispatches=[
            ToolCard(call_id="s1", name="read", arguments='{"path": "data.csv"}', settled=True),
            ToolCard(call_id="s2", name="glob", arguments='{"pattern": "*.csv"}', settled=True),
        ],
    )
    items = _state(
        ChatItem(key="u1", role="user", text="how many rows?", seq=1),
        ChatItem(key="t1", role="tool", tool=card, seq=2),
    )
    assert snap_compare(_Snapshot(items), terminal_size=(80, 18))


def test_a_compaction_marker_keeps_what_it_replaced(snap_compare: Any) -> None:
    items = _state(
        ChatItem(key="u1", role="user", text="the original question", seq=1, shadowed=True),
        ChatItem(key="a1", role="assistant", text="the original answer", seq=2, shadowed=True),
        ChatItem(key="c1", role="compaction", text="Earlier: a question and its answer.", seq=3),
        ChatItem(key="u2", role="user", text="and now the next thing", seq=4),
    )
    assert snap_compare(_Snapshot(items), terminal_size=(80, 18))


def test_markup_in_user_text_renders_literally(snap_compare: Any) -> None:
    """The P2-06 gate, as a picture. `[bold]` must appear, not take effect."""
    items = _state(
        ChatItem(key="u1", role="user", text="check foo[0] and [bold]this[/bold]", seq=1),
        ChatItem(key="a1", role="assistant", text="Values like [1, 2] are fine too.", seq=2),
    )
    assert snap_compare(_Snapshot(items), terminal_size=(80, 12))


@pytest.mark.parametrize("theme", ["ph-light", "high-contrast"])
def test_every_theme_renders(snap_compare: Any, theme: str) -> None:
    items = _state(
        ChatItem(key="u1", role="user", text="hello", seq=1),
        ChatItem(key="a1", role="assistant", text="Hello back.", seq=2),
    )
    assert snap_compare(_Snapshot(items, theme=theme), terminal_size=(80, 10))


def test_the_full_app_shell(snap_compare: Any, make_tui_app: Callable[..., PHTuiApp]) -> None:
    """Prompt, status bar and sidebar together — the chrome, once."""
    app = make_tui_app(session_id="snapshot")

    async def mounted(pilot: Any) -> None:
        # The harness mounts in a worker so the shell paints first; the frame
        # under test is the one after it has.
        await until(pilot, lambda: app.front is not None)
        await pilot.pause(0.1)

    assert snap_compare(app, terminal_size=(90, 24), run_before=mounted)
