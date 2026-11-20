"""`ctx.tui_screens` from the side of the front end (P4-17), driven as a user drives it.

The gates of the row, with one test for each:

- A plugin row registers a screen and gets a verb, a key and a palette entry.
- Unloading the row removes all three.
- The trajectory opens over the chat and returns to it.
- A record jumps to its transcript row and back.

The diff screen is held to the first two as a shipped screen. It also has tests
of its own, over a log that the daemon revised.

Read the lifetime tests first. A screen is easy to make. The slash command and
the key of a screen must disappear with the row that contributed it. The slot
service of dsh has that property, and it is the reason why this is a seam and
not a list in the app. The half that belongs to the seam is in `test_seams.py`
of `ph-core`. This file is the half that needs a terminal.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from daemon_helpers import Daemon
from textual.pilot import Pilot
from tui_helpers import MakeApp, root_of, running, stand_in, turn_done, until

import ph_app.tui
from ph.keys import TUI_SCREENS
from ph.seams.tui_screens import ScreenDefinition
from ph.session import SessionEvent, substitute
from ph.testing import not_none, revision_record
from ph_app.daemon.supervisor import Root
from ph_app.revisions import Revision
from ph_app.screen_rows import DIFF_KEY
from ph_app.tui.app import PHTuiApp
from ph_app.tui.diff_screen import DIFF_LINES, DiffScreen, _edits
from ph_app.tui.diffs import DiffEntry, DiffGroup
from ph_app.tui.frontend import FrontSession
from ph_app.tui.modals.pickers import command_choices
from ph_app.tui.remote import LOCAL_SCREENS
from ph_app.tui.trajectory_screen import SCREEN_ID, TRAJECTORY_KEY, TrajectoryScreen
from ph_app.tui.widgets.prompt import PromptInput
from ph_app.wire import index_at_or_before

pytestmark = pytest.mark.anyio


@pytest.fixture
def tui_profile() -> str:
    """`tui`, the interactive posture in which a user drives these screens.

    The reason is not that `tui` is the profile that has the screens. Every
    named profile mounts the trajectory (`presentation.yaml`). The daemon mounts
    the profile, so the choice is made before any app exists. See the fixture
    that this one overrides, in `app_fixtures.py`.
    """
    return "tui"


def _binding(app: PHTuiApp, binding_id: str) -> Any:  # noqa: ANN401
    """The live binding that carries `binding_id`, or `None`.

    The lookup is by id and not by key. A keymap remaps the id, so the id is the
    only stable name that a test can hold.
    """
    for bindings in app.chat._bindings.key_to_bindings.values():
        for binding in bindings:
            if binding.id == binding_id:
                return binding
    return None


def _routes(app: PHTuiApp, screen_id: str) -> set[str]:
    """Every route that this front end has opened to one screen.

    The set is built by *looking* at the app. A test that compares two
    registrations therefore compares findings, and not two hand-written
    checklists. Two such checklists cannot disagree, and a gate of that shape
    has failed this project two times.
    """
    assert app.front is not None
    definitions = app.front.commands()
    found: set[str] = set()
    if any(one.name == screen_id for one in definitions):
        found.add("command")
    if f"/{screen_id}" in {choice.value for choice in command_choices(definitions)}:
        found.add("palette")
    if _binding(app, screen_id) is not None:
        found.add("key")
    return found


ROUTES = {"command", "palette", "key"}
"""What one `ScreenDefinition` must yield. This is the first gate of the row."""


# ------------------------------------------------- a verb, a key, an entry --


SHIPPED = pytest.mark.parametrize("shipped", LOCAL_SCREENS, ids=lambda one: one.id)
"""Every screen that pH ships. A screen that is added to `LOCAL_SCREENS` is
therefore held to the same gates as the trajectory, with no test of its own."""


@SHIPPED
async def test_a_contributed_screen_becomes_a_verb_a_key_and_a_palette_entry(
    make_tui_app: MakeApp, shipped: ScreenDefinition
) -> None:
    """The gate. One `ScreenDefinition` yields three routes. The row that
    contributed each screen is an ordinary row and not a special case."""
    async with running(make_tui_app()) as (app, _pilot):
        assert _routes(app, shipped.id) == ROUTES

        binding = _binding(app, shipped.id)
        assert binding is not None and binding.key == shipped.key


async def test_a_plugin_screens_key_is_rebindable_like_every_other(
    make_tui_app: MakeApp, tmp_path: Path
) -> None:
    """The binding carries the id of the screen for this reason.

    The test goes through `tui.json`, because that is the claim. A key that
    could not be remapped there would be the one key in the app that a user
    could not change. `config.py` exists to prevent exactly that.
    """
    (tmp_path / "tui.json").write_text(
        json.dumps({"keybindings": {SCREEN_ID: "ctrl+j"}}), encoding="utf-8"
    )
    async with running(make_tui_app()) as (app, pilot):
        await pilot.press("ctrl+j")
        await until(pilot, lambda: isinstance(app.screen, TrajectoryScreen))


@SHIPPED
async def test_a_screen_hidden_in_tui_json_is_not_offered(
    make_tui_app: MakeApp, tmp_path: Path, shipped: ScreenDefinition
) -> None:
    """The terminal hides a screen, and the profile does not (decision 23).

    The row still mounts on the daemon. A session profile that removed it would
    be refused. This terminal opens none of the three routes to the screen.
    Sabotage: drop the `hidden` check in `_screens_of` and every route comes
    back.
    """
    (tmp_path / "tui.json").write_text(
        json.dumps({"hidden_screens": [shipped.id]}), encoding="utf-8"
    )
    async with running(make_tui_app()) as (app, _pilot):
        assert app.front is not None
        assert _routes(app, shipped.id) == set()
        assert app.front.screen(shipped.id) is None


@SHIPPED
async def test_unloading_the_row_takes_the_verb_and_the_key_with_it(
    make_tui_app: MakeApp, tui_daemon: Daemon, shipped: ScreenDefinition
) -> None:
    """I2 across a socket: the routes of a screen do not outlive its row.

    The daemon listens to `screens/change` on the context of the root. The seam
    emits it when a screen is registered and when one is removed. The daemon
    then publishes the whole list as `session.screens`. So a row that is
    unloaded on the daemon takes the verb, the palette entry and the key away
    from a terminal that is one socket away. That is the whole claim that the
    seam makes in process.

    The test runs with each *drawable* screen. The ids that `LOCAL_SCREENS` has
    a builder for are the ids whose routes a client opens. Sabotage: drop the
    `screens/change` listener of the daemon (`drew` in `supervisor.py`) and the
    terminal keeps a verb whose row is gone.
    """
    async with running(make_tui_app()) as (app, pilot):
        assert _routes(app, shipped.id) == ROUTES

        row = root_of(tui_daemon).ctx.get("tui_screens")
        assert row is not None
        entry = row.get(shipped.id)
        assert entry is not None
        await until(pilot, lambda: _routes(app, shipped.id) == ROUTES)

        # Unload it as the removal of the row does: dispose the scope that owns
        # the registration.
        await _unload(root_of(tui_daemon), shipped.id)

        await until(pilot, lambda: _routes(app, shipped.id) == set())
        others = {one.id for one in LOCAL_SCREENS} - {shipped.id}
        assert all(_routes(app, other) == ROUTES for other in others), "it took another's"


async def test_a_screen_this_build_cannot_draw_is_not_offered(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """The limit that P5-14 leaves behind, as a gate and not as a paragraph.

    `ScreenDefinition.build` is the one field that cannot travel. So a socket
    client is told which screens the deployment mounted (`screens/list`), and it
    draws the ones that it has a builder for. Those are `LOCAL_SCREENS`, which
    is every screen that pH ships. A screen that some *other* row contributes
    reaches a remote front end as a name that it cannot open. The client drops
    it, and does not offer it and then fail. A verb that opens nothing is worse
    than a verb that is not there.

    The test has two halves, because a client that dropped everything would
    satisfy the first. The pretend screen gets no routes. The trajectory, which
    is in the same list, gets all three.

    Not enforced (§5 rule 6): closing this gap needs the declarative screen
    bodies of P7-07. After that, the screen of a row can be *described* to a
    client and need not be built in it. The seam has claims of its own: that
    unloading a row takes its screen with it (I2), and that registration order
    does not decide reachability. Those are tested where the seam is, in
    `test_seams.py` of `ph-core`. This test asserts what this transport can and
    cannot carry.
    """
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        pretend = ScreenDefinition(
            id="pretend",
            label="A screen some row contributed",
            key="f9",
            build=lambda session: TrajectoryScreen([], session_id=session.id),
        )
        root.ctx.require(TUI_SCREENS).register(pretend, scope=root.ctx.scope("a-row"))
        await pilot.pause()

        assert _routes(app, "pretend") == set(), "a screen with no local builder was offered"
        assert _routes(app, SCREEN_ID) == ROUTES, "and the one pH ships still is"


# --------------------------------------------------- over the chat, and back --


VIEW_ROWS = 7
"""The height of the transcript in these tests. It is short enough that three
short turns scroll, with room to spare. A jump to the last prompt must be able
to put that prompt at the *top* of the view. So what follows the prompt, which
is its own rows and the rows of the answer, must be taller than the view."""


async def _short_view(app: PHTuiApp, pilot: Any) -> None:  # noqa: ANN401
    """Resize the terminal so that the transcript is `VIEW_ROWS` tall.

    **The view is sized, not the terminal.** The chrome around the view belongs
    to the layout, and it moves. These tests asked for a 12-row terminal when
    the chrome was five rows. The chrome became two rows, the view grew to ten,
    and a jump stopped short with a failure that named the jump.
    """
    view = app._view
    assert view is not None
    chrome = app.size.height - view.size.height
    await pilot.resize_terminal(app.size.width, chrome + VIEW_ROWS)
    await pilot.pause()
    assert view.size.height == VIEW_ROWS, "the transcript did not take the height it was given"


async def _three_turns(app: PHTuiApp, pilot: Any) -> None:  # noqa: ANN401
    """Enough transcript that the view scrolls, so a jump is observable."""
    assert app.front is not None
    await _short_view(app, pilot)
    await pilot.press(*"first")
    await pilot.press(app.keys.submit)
    await until(pilot, turn_done(app))
    for text in ("second", "third"):
        await app.front.submit(text)
    app.state_changed()
    await until(pilot, lambda: app._view is not None and len(app._view._rows) >= 6)


async def test_the_trajectory_opens_over_the_chat_and_escape_returns_to_it(
    make_tui_app: MakeApp,
) -> None:
    """The gate. The trajectory is a pushed screen and not a second app. So
    `escape` leaves it, and it returns to the conversation exactly as it was."""
    async with running(make_tui_app()) as (app, pilot):
        await _three_turns(app, pilot)
        assert app._view is not None
        app._view.scroll_to_seq(0)
        await pilot.pause()
        before = app._view.seq_in_view()

        await pilot.press(TRAJECTORY_KEY)
        await until(pilot, lambda: isinstance(app.screen, TrajectoryScreen))
        # `screen` is bound after the wait. The `isinstance` inside the lambda
        # narrows nothing out here, and `App.screen` is a `Screen[object]`.
        screen = app.screen
        assert isinstance(screen, TrajectoryScreen)
        assert screen.records, "the screen was built from the live session's log"

        await pilot.press("escape")
        await until(pilot, lambda: not isinstance(app.screen, TrajectoryScreen))

        assert app._view.seq_in_view() == before, "the transcript lost its place"
        assert app.focused is app.query_one(PromptInput).area, "the prompt lost focus"


async def test_the_screen_is_a_fold_of_the_log_as_it_stands(make_tui_app: MakeApp) -> None:
    """`build` runs when the screen opens, not at registration. A screen that
    was built one time would show the session as it was when the row mounted."""
    async with running(make_tui_app()) as (app, pilot):
        await pilot.press(TRAJECTORY_KEY)
        await until(pilot, lambda: isinstance(app.screen, TrajectoryScreen))
        opened = app.screen
        assert isinstance(opened, TrajectoryScreen)
        empty = len(opened.records)
        await pilot.press("escape")
        await until(pilot, lambda: not isinstance(app.screen, TrajectoryScreen))

        await pilot.press(*"hello")
        await pilot.press(app.keys.submit)
        await until(pilot, turn_done(app))

        await pilot.press(TRAJECTORY_KEY)
        await until(pilot, lambda: isinstance(app.screen, TrajectoryScreen))
        reopened = app.screen
        assert isinstance(reopened, TrajectoryScreen)
        assert len(reopened.records) > empty


# ------------------------------------------------------- cross-navigation --


def test_the_nearest_row_rule_skips_a_row_with_no_log_position() -> None:
    """`index_at_or_before` is the one rule of the join between the views: the
    last entry at or before the target.

    An entry whose seq is `None` has no position in the log, so it is never the
    answer, and it does not end the search. "No entry is that early" is `None`
    too, and not an index of `-1`, which Python would read as the last row.

    Sabotage: return `-1` for "none", and `rows[index]` in a caller that forgot
    the check silently picks the last row.
    """
    assert index_at_or_before([1, None, 4, 9], 5) == 2
    assert index_at_or_before([1, None, 4], 3) == 0, "the row with no position was chosen"
    assert index_at_or_before([4, 9], 4) == 0, "an exact hit is at or before"
    assert index_at_or_before([None, 4], 3) is None
    assert index_at_or_before([], 3) is None


async def test_a_record_jumps_to_its_transcript_row(make_tui_app: MakeApp) -> None:
    """The gate, over the join that P3-24 stored and nothing read: the
    `source_seq` of a record and the `seq` of a transcript row are the same
    number."""
    async with running(make_tui_app()) as (app, pilot):
        await _three_turns(app, pilot)
        assert app.front is not None and app._view is not None
        first_user = not_none(
            next(item.seq for item in app.front.state.items if item.role == "user")
        )

        await pilot.press(TRAJECTORY_KEY)
        await until(pilot, lambda: isinstance(app.screen, TrajectoryScreen))
        screen = app.screen
        assert isinstance(screen, TrajectoryScreen)
        assert screen.panel.select_seq(first_user)
        await pilot.pause()
        assert screen.panel.selected().source_seq == first_user  # type: ignore[union-attr]

        await pilot.press("enter")
        await until(pilot, lambda: not isinstance(app.screen, TrajectoryScreen))

        assert app._view.seq_in_view() == first_user, "the jump landed elsewhere"


async def test_a_transcript_row_opens_the_record_beside_it(make_tui_app: MakeApp) -> None:
    """The other direction. Rows are widgets in a scroll and not a table with a
    cursor, so "where the reader is" is the topmost visible row."""
    async with running(make_tui_app()) as (app, pilot):
        await _three_turns(app, pilot)
        assert app.front is not None and app._view is not None
        last_user = [item for item in app.front.state.items if item.role == "user"][-1]
        app._view.scroll_to_seq(not_none(last_user.seq))
        await pilot.pause()

        await pilot.press(TRAJECTORY_KEY)
        await until(pilot, lambda: isinstance(app.screen, TrajectoryScreen))
        screen = app.screen
        assert isinstance(screen, TrajectoryScreen)

        selected = screen.panel.selected()
        assert selected is not None and selected.source_seq == last_user.seq


async def test_a_seq_with_no_row_lands_on_the_nearest_one_before_it(
    make_tui_app: MakeApp,
) -> None:
    """A `request/header` is a record for an auditor, and by design it has no
    transcript row. To land a reader next to it is better than to land them
    nowhere."""
    async with running(make_tui_app()) as (app, pilot):
        await _three_turns(app, pilot)
        assert app.front is not None and app._view is not None
        seqs = [item.seq for item in app.front.state.items if item.seq is not None]
        between = seqs[1] + 1
        assert between not in seqs, "pick a seq no row owns"

        assert app._view.scroll_to_seq(between)
        await pilot.pause()
        assert app._view.seq_in_view() == seqs[1]


def test_the_terminal_never_reaches_past_the_front_session() -> None:
    """`app.py` and the widgets may touch only what `FrontSession` declares.

    Every member that the terminal reads from its harness must be one that a
    *remote* harness can answer. `front.ctx` was the obvious offender: nine
    service lookups that exist only while the harness is in this process. But
    the class of thing matters, and not the name. `front.agent`, `front.ctx`
    and `front._stack` all fail in the same way, and only over a socket. The
    terminal ran its harness in process when this gate was written, so running
    the terminal then could not catch one.

    So the check is **membership**, and not a banned word: whatever is read
    from `front` must be in the Protocol. That makes the test update itself,
    because a member that is added to `FrontSession` is allowed here. It is
    also strictly stronger than forbidding `.ctx`. That ban missed every other
    reach-through, and it would have flagged an unrelated `event.ctx`.

    The check reads `app.py` and the widgets. `frontend.py` is deliberately out
    of scope: it declares the two protocols (`FrontSession` and `ModalHost`) and
    holds no implementation. The implementation that ships is `DaemonSession`
    in `remote.py`, a client of the daemon.
    """
    allowed = set(dir(FrontSession)) | set(getattr(FrontSession, "__annotations__", {}))
    root = Path(ph_app.tui.__path__[0])
    watched = [root / "app.py", *sorted((root / "widgets").rglob("*.py"))]
    offenders: list[str] = []
    for path in watched:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr in allowed:
                continue
            # `front.x` and `self.front.x` are the two spellings that the
            # terminal uses to reach its harness.
            base = node.value
            named = isinstance(base, ast.Name) and base.id == "front"
            through_self = isinstance(base, ast.Attribute) and base.attr == "front"
            if named or through_self:
                offenders.append(f"{path.name}:{node.lineno} reads front.{node.attr}")
    assert offenders == [], offenders


# ------------------------------------------------------------- the diff --


async def _turns_with_revisions(
    app: PHTuiApp, pilot: Pilot[str | None], root: Root
) -> list[SessionEvent]:
    """Three turns. The first two user messages are each replaced in the log of
    the daemon after their own turn, so each revision has transcript rows on
    both sides."""
    assert app.front is not None
    front = app.front
    await _short_view(app, pilot)
    await pilot.press(*"first")
    await pilot.press(app.keys.submit)
    await until(pilot, turn_done(app))
    landed: list[SessionEvent] = []
    for text in ("second", "third"):
        asked = [one for one in root.session.events if one.type == "user/message"][-1]
        why = revision_record("compaction/summarized")
        landed.append(substitute(root.session, (asked.seq,), stand_in(f"[{asked.seq}]"), why))
        await front.submit(text)
    await until(pilot, lambda: front.session.seq == root.session.seq)
    app.state_changed()
    await until(pilot, lambda: app._view is not None and len(app._view._rows) >= 8)
    return landed


async def _open_diff(app: PHTuiApp, pilot: Pilot[str | None]) -> DiffScreen:
    await pilot.press(DIFF_KEY)
    await until(pilot, lambda: isinstance(app.screen, DiffScreen))
    screen = app.screen
    assert isinstance(screen, DiffScreen)
    return screen


async def test_the_diff_lists_the_daemons_revisions_and_escape_returns_to_the_chat(
    make_tui_app: MakeApp, tui_daemon: Daemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The screen is built from the mirror that this client keeps of the log.
    The mirror holds every replacement that the daemon wrote, so the screen
    makes no request beyond the ones that the transcript already made. Only the
    row on screen is rendered."""
    rendered: list[int] = []
    render = Revision.texts

    def counting(revision: Revision) -> tuple[str, str]:
        rendered.append(revision.event.seq)
        return render(revision)

    monkeypatch.setattr(Revision, "texts", counting)
    async with running(make_tui_app()) as (app, pilot):
        landed = await _turns_with_revisions(app, pilot, root_of(tui_daemon))

        screen = await _open_diff(app, pilot)
        await pilot.pause()
        assert [group.seq for group in screen.groups] == [one.seq for one in landed]
        assert {group.latest.title for group in screen.groups} == {"stand-in"}
        selected = screen.selected()
        assert selected is not None
        assert set(rendered) == {selected.seq}, "a row nobody looked at was rendered"

        await pilot.press("escape")
        await until(pilot, lambda: not isinstance(app.screen, DiffScreen))
        assert app.focused is app.query_one(PromptInput).area, "the prompt lost focus"


async def test_enter_on_a_change_goes_to_its_transcript_row(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """The join that the trajectory uses, from the other screen: the entry of a
    substitution lands on the row that the replacement drew."""
    async with running(make_tui_app()) as (app, pilot):
        first, _ = await _turns_with_revisions(app, pilot, root_of(tui_daemon))
        assert app._view is not None
        app._view.scroll_to_seq(0)
        await pilot.pause()

        screen = await _open_diff(app, pilot)
        assert screen.select_seq(first.seq)
        await pilot.pause()
        await pilot.press("enter")
        await until(pilot, lambda: not isinstance(app.screen, DiffScreen))

        assert app._view.seq_in_view() == first.seq, "the jump landed elsewhere"


async def test_opened_from_the_chat_it_starts_at_the_nearest_change_before_the_reader(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    async with running(make_tui_app()) as (app, pilot):
        first, second = await _turns_with_revisions(app, pilot, root_of(tui_daemon))
        assert app.front is not None and app._view is not None
        between = next(
            item.seq
            for item in app.front.state.items
            if item.seq is not None and first.seq < item.seq < second.seq
        )

        for reading, expected in ((between, first), (second.seq, second)):
            app._view.scroll_to_seq(reading)
            await pilot.pause()
            screen = await _open_diff(app, pilot)
            # `DiffScreen.on_mount` moves the cursor, so wait for the mount to run.
            await pilot.pause()
            await pilot.pause()
            selected = screen.selected()
            assert selected is not None and selected.seq == expected.seq
            await pilot.press("escape")
            await until(pilot, lambda: not isinstance(app.screen, DiffScreen))


async def _unload(root: Root, screen_id: str) -> None:
    """Dispose the scope that owns the registration of this screen.

    The removal of a row is the disposal of the scope that its `apply` received.
    So this is that unwinding, and not a second way to take a screen away.
    """
    registry = root.ctx.get("tui_screens")
    entry = registry._entries[screen_id]
    await entry.owner.dispose()


def test_edits_past_the_drawn_budget_are_counted_not_rendered() -> None:
    """The group of a summary holds every older summary, and the before of each
    one can be most of a conversation. So when the drawn lines are spent, the
    remaining edits are counted. They are not rendered and diffed on every
    cursor move. Sabotage: render every edit."""
    rendered: list[int] = []

    def entry(seq: int) -> DiffEntry:
        def texts() -> tuple[str, str]:
            rendered.append(seq)
            return "\n".join(f"line {index}" for index in range(DIFF_LINES)), ""

        return DiffEntry(seq=seq, title="rewrite", summary="", producer=None, texts=texts)

    group = DiffGroup(
        kind="revision",
        seq=3,
        reveal=3,
        entries=(entry(1), entry(2), entry(3)),
        whole=lambda: ("", ""),
    )

    shown = _edits(group).plain

    assert rendered == [3], "an edit past the budget was rendered"
    assert "2 older edits; w shows the whole history" in shown
