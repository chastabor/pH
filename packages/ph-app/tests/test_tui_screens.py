"""`ctx.tui_screens` from the front end's side (P4-17), driven as a person does.

The row's gates, one test each: *a plugin row registers a screen and gets a
verb, a key and a palette entry; unloading the row removes all three; the
trajectory opens over the chat and returns to it; a record jumps to its
transcript row and back.*

The lifetime tests are the ones to read. A screen is easy; a screen whose slash
command and whose key disappear with the row that contributed it is the property
dsh's slot service has and the reason this is a seam rather than a list in the
app. The seam's own half is in `ph-core`'s `test_seams.py`; this is the half
that needs a terminal.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tui_helpers import root_of, running, turn_done, until

import ph_app.tui
from ph.seams.tui_screens import ScreenDefinition
from ph_app.tui.app import PHTuiApp
from ph_app.tui.frontend import FrontSession
from ph_app.tui.modals.pickers import command_choices
from ph_app.tui.trajectory_screen import SCREEN_ID, TRAJECTORY_KEY, TrajectoryScreen
from ph_app.tui.widgets.prompt import PromptInput

pytestmark = pytest.mark.anyio

MakeApp = Callable[..., PHTuiApp]


@pytest.fixture
def tui_profile() -> str:
    """`tui`, because this file's subject is a row that profile contributes.

    The daemon mounts the profile now, so the choice is made before any app
    exists — see the fixture this overrides in `conftest.py`.
    """
    return "tui"


def _binding(app: PHTuiApp, binding_id: str) -> Any:
    """The live binding carrying `binding_id`, or `None`.

    Looked up by id rather than by key, because the id is what a keymap remaps
    and therefore the only stable name a test can hold.
    """
    for bindings in app._bindings.key_to_bindings.values():
        for binding in bindings:
            if binding.id == binding_id:
                return binding
    return None


def _routes(app: PHTuiApp, screen_id: str) -> set[str]:
    """Every route this front end has actually opened to one screen.

    Built by *looking*, so a test comparing two registrations compares findings
    rather than two hand-written checklists — a pair of those cannot disagree,
    which is the shape of gate this project has been bitten by twice.
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
"""What one `ScreenDefinition` is supposed to buy — the row's first gate."""


# ------------------------------------------------- a verb, a key, an entry --


async def test_a_contributed_screen_becomes_a_verb_a_key_and_a_palette_entry(
    make_tui_app: MakeApp,
) -> None:
    """The gate. One `ScreenDefinition`, three routes to it — and the row that
    contributed the trajectory is an ordinary row, not a special case."""
    async with running(make_tui_app()) as (app, _pilot):
        assert _routes(app, SCREEN_ID) == ROUTES

        binding = _binding(app, SCREEN_ID)
        assert binding is not None and binding.key == TRAJECTORY_KEY


async def test_a_plugin_screens_key_is_rebindable_like_every_other(
    make_tui_app: MakeApp, tmp_path: Path
) -> None:
    """The reason the binding carries the screen's id.

    Driven through `tui.json`, because that is the claim: a key that could not
    be remapped there would be the one key in the app a person could not change,
    which is precisely what `config.py` exists to prevent.
    """
    (tmp_path / "tui.json").write_text(
        json.dumps({"keybindings": {SCREEN_ID: "ctrl+j"}}), encoding="utf-8"
    )
    async with running(make_tui_app()) as (app, pilot):
        await pilot.press("ctrl+j")
        await until(pilot, lambda: isinstance(app.screen, TrajectoryScreen))


async def test_unloading_the_row_takes_the_verb_and_the_key_with_it(
    make_tui_app: MakeApp, tui_daemon: Any
) -> None:
    """I2 across a socket: a screen's routes do not outlive its row.

    The daemon attaches to `ctx.tui_screens` as the front end it stands in for —
    `present_with` gives it every screen now, every later one, and a disposer per
    registration — and publishes `session.screens` on both edges. So a row
    unloaded on the daemon takes the verb, the palette entry and the key away
    from a terminal one socket removed, which is the whole claim the seam makes
    in process.

    Driven with a *drawable* screen: `trajectory` is the id `LOCAL_SCREENS` has a
    builder for, so it is the one whose routes a client actually opens. Sabotage:
    drop the `present_with` subscription and the terminal keeps a verb whose row
    is gone.
    """
    async with running(make_tui_app()) as (app, pilot):
        assert _routes(app, SCREEN_ID) == ROUTES

        row = root_of(tui_daemon).ctx.get("tui_screens")
        assert row is not None
        entry = row.get(SCREEN_ID)
        assert entry is not None
        await until(pilot, lambda: _routes(app, SCREEN_ID) == ROUTES)

        # Unload it the way removing the row does: dispose the scope that owns
        # the registration.
        await _unload(root_of(tui_daemon), SCREEN_ID)

        await until(pilot, lambda: _routes(app, SCREEN_ID) == set())


async def test_a_screen_this_build_cannot_draw_is_not_offered(
    make_tui_app: MakeApp, tui_daemon: Any
) -> None:
    """The limit P5-14 leaves behind, as a gate rather than a paragraph.

    `ScreenDefinition.build` is the one field that cannot travel, so a socket
    client is told which screens the deployment mounted (`screens/list`) and
    draws the ones it has a builder for — `LOCAL_SCREENS`, which is every screen
    pH ships. A screen some *other* row contributes reaches a remote front end as
    a name it cannot open, so it is dropped rather than offered and then failing:
    a verb that opens nothing is worse than a verb that is not there.

    Two halves, because dropping everything would satisfy the first: the pretend
    screen gets no routes, and the trajectory — same list, same reply — gets all
    three.

    Not enforced (§5 rule 6): closing this needs P7-07's declarative screen
    bodies, after which a row's screen can be *described* to a client rather than
    built in it. The seam's own claims — that unloading a row takes its screen
    with it (I2), and that registration order does not decide reachability — are
    where the seam is, in `ph-core`'s `test_seams.py`; what is asserted here is
    what this transport can and cannot carry.
    """
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        pretend = ScreenDefinition(
            id="pretend",
            label="A screen some row contributed",
            key="f9",
            build=lambda session: TrajectoryScreen([], session_id=session.id),
        )
        root.ctx.tui_screens.register(pretend, scope=root.ctx.scope("a-row"))
        await pilot.pause()

        assert _routes(app, "pretend") == set(), "a screen with no local builder was offered"
        assert _routes(app, SCREEN_ID) == ROUTES, "and the one pH ships still is"


# --------------------------------------------------- over the chat, and back --


async def _three_turns(app: PHTuiApp, pilot: Any) -> None:
    """Enough transcript that the view scrolls, so a jump is observable."""
    assert app.front is not None
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
    """The gate. A pushed screen, not a second app — so leaving it is `escape`
    and what it returns to is the conversation exactly as it was."""
    async with running(make_tui_app(), size=(80, 12)) as (app, pilot):
        await _three_turns(app, pilot)
        assert app._view is not None
        app._view.scroll_to_seq(0)
        await pilot.pause()
        before = app._view.seq_in_view()

        await pilot.press(TRAJECTORY_KEY)
        await until(pilot, lambda: isinstance(app.screen, TrajectoryScreen))
        assert app.screen.records, "the screen was built from the live session's log"

        await pilot.press("escape")
        await until(pilot, lambda: not isinstance(app.screen, TrajectoryScreen))

        assert app._view.seq_in_view() == before, "the transcript lost its place"
        assert app.focused is app.query_one(PromptInput).area, "the prompt lost focus"


async def test_the_screen_is_a_fold_of_the_log_as_it_stands(make_tui_app: MakeApp) -> None:
    """`build` runs at open time, not at registration: a screen that had been
    built once would show the session as it was when the row mounted."""
    async with running(make_tui_app()) as (app, pilot):
        await pilot.press(TRAJECTORY_KEY)
        await until(pilot, lambda: isinstance(app.screen, TrajectoryScreen))
        empty = len(app.screen.records)
        await pilot.press("escape")
        await until(pilot, lambda: not isinstance(app.screen, TrajectoryScreen))

        await pilot.press(*"hello")
        await pilot.press(app.keys.submit)
        await until(pilot, turn_done(app))

        await pilot.press(TRAJECTORY_KEY)
        await until(pilot, lambda: isinstance(app.screen, TrajectoryScreen))
        assert len(app.screen.records) > empty


# ------------------------------------------------------- cross-navigation --


async def test_a_record_jumps_to_its_transcript_row(make_tui_app: MakeApp) -> None:
    """The gate, over the join P3-24 stored and nobody read: a record's
    `source_seq` and a transcript row's `seq` are the same number."""
    async with running(make_tui_app(), size=(80, 12)) as (app, pilot):
        await _three_turns(app, pilot)
        assert app.front is not None and app._view is not None
        first_user = next(item for item in app.front.state.items if item.role == "user")

        await pilot.press(TRAJECTORY_KEY)
        await until(pilot, lambda: isinstance(app.screen, TrajectoryScreen))
        screen = app.screen
        assert isinstance(screen, TrajectoryScreen)
        assert screen.panel.select_seq(first_user.seq)
        await pilot.pause()
        assert screen.panel.selected().source_seq == first_user.seq  # type: ignore[union-attr]

        await pilot.press("enter")
        await until(pilot, lambda: not isinstance(app.screen, TrajectoryScreen))

        assert app._view.seq_in_view() == first_user.seq, "the jump landed elsewhere"


async def test_a_transcript_row_opens_the_record_beside_it(make_tui_app: MakeApp) -> None:
    """The other direction. Rows are widgets in a scroll rather than a table
    with a cursor, so "where the reader is" is the topmost visible row."""
    async with running(make_tui_app(), size=(80, 12)) as (app, pilot):
        await _three_turns(app, pilot)
        assert app.front is not None and app._view is not None
        last_user = [item for item in app.front.state.items if item.role == "user"][-1]
        app._view.scroll_to_seq(last_user.seq)
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
    """A `request/header` is an auditor's record with no transcript row by
    design. Landing a reader next to it beats landing them nowhere."""
    async with running(make_tui_app(), size=(80, 12)) as (app, pilot):
        await _three_turns(app, pilot)
        assert app.front is not None and app._view is not None
        rows = [item for item in app.front.state.items if item.seq >= 0]
        between = rows[1].seq + 1
        assert all(item.seq != between for item in rows), "pick a seq no row owns"

        assert app._view.scroll_to_seq(between)
        await pilot.pause()
        assert app._view.seq_in_view() == rows[1].seq


def test_the_terminal_never_reaches_past_the_front_session() -> None:
    """`app.py` and the widgets may only touch what `FrontSession` declares.

    Every member the terminal reads off its harness has to be one a *remote*
    harness can answer. `front.ctx` was the obvious offender — nine service
    lookups that only exist while the harness is in this process — but it is the
    class of thing that matters, not the name: `front.agent`, `front.ctx`,
    `front._stack` all fail the same way, and only over a socket, which is why no
    amount of running the terminal today would catch one.

    So the check is **membership**, not a banned word: whatever is read off
    `front` must be in the Protocol. That makes the test self-updating — adding a
    member to `FrontSession` licenses it here — and strictly stronger than
    forbidding `.ctx`, which both missed every other reach-through and would have
    flagged an unrelated `event.ctx`.

    `frontend.py` is deliberately out of scope: it *is* the in-process
    implementation, and resolving seams is its whole job.
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
            # `front.x` and `self.front.x` — the two spellings the terminal uses
            # to reach its harness.
            base = node.value
            named = isinstance(base, ast.Name) and base.id == "front"
            through_self = isinstance(base, ast.Attribute) and base.attr == "front"
            if named or through_self:
                offenders.append(f"{path.name}:{node.lineno} reads front.{node.attr}")
    assert offenders == [], offenders


async def _unload(root: Any, screen_id: str) -> None:
    """Dispose whatever scope owns this screen's registration.

    A row's removal is the disposal of the scope its `apply` was handed, so this
    is that unwinding rather than a second way to take a screen away.
    """
    registry = root.ctx.get("tui_screens")
    entry = registry._entries[screen_id]
    await entry.owner.dispose()
