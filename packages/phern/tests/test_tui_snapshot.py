"""Snapshot tests: what the four transcript shapes look like.

The four shapes are streaming, a settled tool card, an error, and a compaction
marker. They are the four things that a person reads a transcript for.
Snapshots are the only tests that catch a *rendering* regression: a card that
lost its glyph, a theme variable that stopped resolving, a row that started
rendering its markup instead of showing it.

Each app receives a state directly and is not driven through a model. That keeps
the snapshot stable: no timing, no token order, and no session ids in the frame.
"""

from __future__ import annotations

import getpass
from pathlib import Path
from typing import Any

import pytest
from daemon_helpers import daemon_in_thread
from textual.app import App
from tui_helpers import TranscriptApp, revisable, rewritten, rows_of, stand_in, tui_app, until

from ph.session import RevisionSize, Session, substitute
from ph.session.known_event_types import declare_log_type
from ph.testing import isolated_log_types, revision_record
from ph_app.tui.diff_screen import DiffScreen
from ph_app.tui.diffs import build_diffs
from ph_app.tui.state import ChatItem, ToolCard
from ph_app.tui.themes import DEFAULT_THEME, fallback_variables, load_catalog

# ------------------------------------------------------------------------------


def test_streaming_assistant_text(snap_compare: Any) -> None:  # noqa: ANN401
    items = rows_of(
        ChatItem(key="u1", role="user", text="explain the pipeline", seq=1),
        ChatItem(
            key="a1",
            role="assistant",
            text="The pipeline runs **pre-execute**, then guards, then the body.",
            streaming=True,
            seq=2,
        ),
    )
    assert snap_compare(TranscriptApp(items), terminal_size=(80, 14))


def test_a_settled_tool_card(snap_compare: Any) -> None:  # noqa: ANN401
    card = ToolCard(
        call_id="c1",
        name="read",
        arguments='{"path": "src/ph/tools/registry.py"}',
        title="read",
        subtitle="src/ph/tools/registry.py",
        settled=True,
        body="1  from __future__ import annotations",
    )
    items = rows_of(
        ChatItem(key="u1", role="user", text="read the registry", seq=1),
        ChatItem(key="t1", role="tool", tool=card, seq=2),
    )
    assert snap_compare(TranscriptApp(items), terminal_size=(80, 14))


def test_a_failed_tool_card_and_an_error_row(snap_compare: Any) -> None:  # noqa: ANN401
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
    items = rows_of(
        ChatItem(key="t1", role="tool", tool=card, seq=1),
        ChatItem(key="e1", role="error", text="the provider returned 429 after 3 retries", seq=2),
    )
    assert snap_compare(TranscriptApp(items), terminal_size=(80, 14))


def test_a_code_mode_card_lists_its_dispatches(snap_compare: Any) -> None:  # noqa: ANN401
    """Forty writes are forty rows (C2). Here there are two."""
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
    items = rows_of(ChatItem(key="t1", role="tool", tool=card, seq=1))
    assert snap_compare(TranscriptApp(items), terminal_size=(80, 14))


def test_a_code_cell_shows_its_program_and_what_it_cost(snap_compare: Any) -> None:  # noqa: ANN401
    """P3-19: the program is the interesting half of a cell.

    Every other card is a header and a body, because its input fits on the
    header line. The input of a cell is the code that the model wrote. The facts
    line under it carries what the collapsible does not. The dispatch count
    belongs to the collapsible, from the fold that owns those rows. So one
    number is never rendered two times from two folds inside one widget (A11).
    """
    card = ToolCard(
        call_id="c1",
        name="python",
        arguments='{"program": "..."}',
        title="python",
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
    items = rows_of(
        ChatItem(key="u1", role="user", text="how many rows?", seq=1),
        ChatItem(key="t1", role="tool", tool=card, seq=2),
    )
    assert snap_compare(TranscriptApp(items), terminal_size=(80, 18))


def test_a_compaction_marker_keeps_what_it_replaced(snap_compare: Any) -> None:  # noqa: ANN401
    items = rows_of(
        ChatItem(key="u1", role="user", text="the original question", seq=1, shadowed=True),
        ChatItem(key="a1", role="assistant", text="the original answer", seq=2, shadowed=True),
        ChatItem(key="c1", role="compaction", text="Earlier: a question and its answer.", seq=3),
        ChatItem(key="u2", role="user", text="and now the next thing", seq=4),
    )
    assert snap_compare(TranscriptApp(items), terminal_size=(80, 18))


class _DiffSnapshot(App[None]):
    """The diff screen alone, over a log built by hand, opened at one change."""

    def __init__(self, session: Session, at: int) -> None:
        super().__init__()
        self.session = session
        self.at = at

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return fallback_variables()

    async def on_mount(self) -> None:
        load_catalog().install(self)
        self.theme = DEFAULT_THEME
        screen = DiffScreen(build_diffs(self.session), session_id=self.session.id)
        screen.reveal(self.at)
        await self.push_screen(screen)


def _rewritten_then_summarized() -> tuple[Session, int]:
    """A log with a rewrite and the stand-in over it, which are the two edits of
    one node. The second value is the seq of the stand-in."""
    session, _, replied, result = revisable("# a\nfirst line\nsecond line")
    cut = rewritten(session, result, "# a\n(2 lines cut)")
    with isolated_log_types():
        declare_log_type("probe/revised", owner="probe", ignorable=True, audit_only=True)
        summary = stand_in("Read a.md: a heading and two lines.")
        record = revision_record(
            "probe/revised", {"verb": "replace"}, RevisionSize("tokens", 31, 9)
        )
        replacement = substitute(session, (replied.seq, cut.seq), summary, record)
    return session, replacement.seq


def test_the_diff_screen_shows_a_change_its_records_and_its_diff(
    snap_compare: Any,  # noqa: ANN401
) -> None:
    """The group of one node: the table, then each edit with the newest first.
    The record of the stand-in is shown with its recorded size and with its diff
    against what it replaced."""
    session, at = _rewritten_then_summarized()
    assert snap_compare(_DiffSnapshot(session, at), terminal_size=(100, 32))


def test_the_diff_screen_shows_a_node_s_whole_history(
    snap_compare: Any,  # noqa: ANN401
) -> None:
    """After `w`: the node as first written against the node now, as one diff
    over both edits."""
    session, at = _rewritten_then_summarized()
    assert snap_compare(_DiffSnapshot(session, at), press=["w"], terminal_size=(100, 32))


def test_markup_in_user_text_renders_literally(snap_compare: Any) -> None:  # noqa: ANN401
    """The gate of P2-06, as a picture. `[bold]` must appear and must not take effect."""
    items = rows_of(
        ChatItem(key="u1", role="user", text="check foo[0] and [bold]this[/bold]", seq=1),
        ChatItem(key="a1", role="assistant", text="Values like [1, 2] are fine too.", seq=2),
    )
    assert snap_compare(TranscriptApp(items), terminal_size=(80, 12))


@pytest.mark.parametrize("theme", ["ph-light", "high-contrast"])
def test_every_theme_renders(snap_compare: Any, theme: str) -> None:  # noqa: ANN401
    items = rows_of(
        ChatItem(key="u1", role="user", text="hello", seq=1),
        ChatItem(key="a1", role="assistant", text="Hello back.", seq=2),
    )
    assert snap_compare(TranscriptApp(items, theme=theme), terminal_size=(80, 10))


def test_the_full_app_shell(
    snap_compare: Any,  # noqa: ANN401
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prompt, status bar and sidebar together: the chrome, one time.

    The app is built with `tui_app` directly and not from `make_tui_app`. That
    fixture is async now, because the terminal attaches to a daemon (P5-14).
    `snap_compare` drives its own event loop, so it can neither await a fixture
    nor share one. The daemon therefore runs in a thread and the app reaches it
    over the socket. The socket is the only boundary between them, and that is
    closer to production than the fixture with a shared loop.

    `project=` is fixed because the sidebar prints it. When it was taken from
    `Path.cwd()`, this snapshot recorded the checkout of the developer and
    failed on every other machine, CI included.
    """
    with daemon_in_thread(tmp_path) as socket:
        monkeypatch.setenv("PH_RUNTIME", str(socket.parent))
        app = tui_app(home=tmp_path, project=Path("/w/project"), session_id="snapshot")

        async def attached(pilot: Any) -> None:  # noqa: ANN401
            # The attach happens in a worker so that the shell paints first. The
            # frame under test is the frame after the attach.
            await until(pilot, lambda: app.front is not None)
            await pilot.pause(0.1)

        assert snap_compare(app, terminal_size=(90, 24), run_before=attached)


def test_no_snapshot_records_this_machine() -> None:
    """A snapshot that embeds the paths of the developer is one that only the developer can pass.

    `test_the_full_app_shell` did. The sidebar prints `PHTuiApp.project`, which
    is `Path.cwd()`, so the recorded frame said `~/Projects/pH`. The checkout of
    CI is `~/work/pH/pH`, so CI failed on that snapshot and only on that one.
    Nothing caught it here, because here it was true.

    So the check is for the *class*, and not for that path. Any home directory,
    working directory or user name in a recorded frame makes the snapshot
    unportable, whatever produced it. A test that wants to render a path sets a
    fixed one: `make_tui_app(project=Path("/w/project"))`.
    """
    directory = Path(__file__).parent / "__snapshots__"
    machine = {
        str(Path.home()): "$HOME",
        str(Path.cwd()): "the working directory",
        getpass.getuser(): "the user name",
    }
    offenders = [
        f"{recorded.relative_to(directory)} records {what}"
        for recorded in sorted(directory.rglob("*.raw"))
        for value, what in machine.items()
        # A user name of one character, or another trivial value, would match
        # half the file by accident. Only a value of some length can be evidence.
        if len(value) > 3 and value in recorded.read_text(encoding="utf-8")
    ]
    assert offenders == [], "\n".join(offenders)
