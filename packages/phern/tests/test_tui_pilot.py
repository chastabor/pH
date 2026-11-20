"""Pilot tests: keystrokes drive every modal, as a user would.

The definition of done for Phase 2 is "pilot tests drive every modal", and the
reason is narrower than coverage. A TUI does not fail with wrong values. It
fails with *wrong plumbing*:

- A key reaches nothing, because the binding was compared literally.
- A modal is awaited on the message pump, so the app deadlocks.
- The callback of a picker never fires.

A unit test of the widget shows none of that.

Read the approval test first. It asserts the round trip through the **log**,
`approval/asked` and then `approval/decided`, and not the return value of the
modal. The claim of pH is that the decision is recorded, and not that a dialog
appeared.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, get_args

import pytest
from daemon_helpers import Daemon, private_runtime
from daemon_helpers import until as daemon_until
from textual.app import App
from textual.binding import Binding
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import Button, Input, Static
from tui_helpers import MakeApp, root_of, running, tui_app, turn_done, until

from ph.keys import APPROVAL, COMMANDS, COMPACTION, CREDENTIALS, TOOLS
from ph.persistence.jsonl import HEADER_LINE_TYPE, session_path
from ph.seams.approval import ApprovalRequest, Edited, Responded
from ph.seams.commands import CommandDefinition
from ph.seams.models import ModelChoice
from ph.seams.tui_status import ReadingSlot, StatusReading
from ph.seams.user_questions import UserQuestion
from ph.session import SessionHeader
from ph.testing import StubAgent, not_none, raising, simple_tool
from ph_app.daemon.supervisor import Root
from ph_app.payloads import AskKey
from ph_app.trust import TrustStore
from ph_app.tui.app import VERB_GROUP, ChatScreen, PHTuiApp, run_tui
from ph_app.tui.modals.approval import ApprovalModal
from ph_app.tui.modals.ask_user import AskUserModal
from ph_app.tui.modals.base import Answers, Choice, ChoicePicker, ConfirmModal
from ph_app.tui.modals.pickers import NEW_SESSION
from ph_app.tui.modals.trust import project_trust_modal
from ph_app.tui.state import ChatItem
from ph_app.tui.themes import fallback_variables, load_theme_profile
from ph_app.tui.trajectory_app import run_trajectory
from ph_app.tui.widgets.prompt import PromptInput
from ph_app.tui.widgets.status import Sidebar, StatusBar

pytestmark = pytest.mark.anyio


def _command(name: str = "tidy") -> CommandDefinition:
    """A command that the test registers, under a name that nothing ships.

    The name was `compact` until every shipped profile began to carry the
    stabilize bundle. By then `command-compact` had already claimed that name,
    and `ctx.commands` refused the second registration. The refusal was correct,
    and the tests that read the completion list failed with it. A fixture whose
    name can collide with a real row will collide with one, so this name is a
    word that no row uses.
    """
    return CommandDefinition(name=name, summary="Tidy the workspace", run=lambda *_: None)


# ------------------------------------------------------------------- prompt --


async def test_typing_and_submitting_runs_a_turn(make_tui_app: MakeApp, tui_daemon: Daemon) -> None:
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        root.ctx.require(TOOLS).register(simple_tool("ping"))
        await pilot.press(*"hello")
        await pilot.press(app.keys.submit)
        await until(pilot, turn_done(app))
        roles = [(item.role, item.text) for item in app.front.state.items]
        assert ("user", "hello") in roles
        assert any(role == "assistant" for role, _ in roles)


async def test_a_fresh_tui_session_takes_the_id_the_daemon_gives_it(
    make_tui_app: MakeApp, tui_daemon: Daemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TUI that opens a new session sends no id, and works on the id that
    `session/new` replied with. It used to mint an id and send it. The daemon
    could not tell that id from a stored id, and searched its store for it twice.

    Sabotage: mint the id in `PHTuiApp._open` again, and the store is searched.
    """
    from ph.persistence import families, jsonl

    searched = raising(AssertionError("searched the store for a session the daemon made"))
    monkeypatch.setattr(jsonl, "locate_under", searched)
    monkeypatch.setattr(families, "locate_under", searched)
    async with running(make_tui_app(session_id=None, offer_sessions=False)) as (app, pilot):
        await until(pilot, lambda: app.front is not None)

        assert tui_daemon.held(not_none(app.front).session_id).session is not None


async def test_bracketed_text_reaches_the_log_verbatim(make_tui_app: MakeApp) -> None:
    """The P2-06 gate. `[bold]` is text, not a formatting instruction."""
    async with running(make_tui_app()) as (app, pilot):
        assert app.front is not None
        app.query_one(PromptInput).area.insert("say [bold]hi[/bold] and foo[0]")
        await pilot.press(app.keys.submit)
        await until(pilot, turn_done(app))
        typed = next(item.text for item in app.front.state.items if item.role == "user")
        assert typed == "say [bold]hi[/bold] and foo[0]"


async def test_a_large_paste_becomes_a_placeholder(make_tui_app: MakeApp) -> None:
    from textual import events

    async with running(make_tui_app()) as (app, pilot):
        prompt = app.query_one(PromptInput)
        pasted = "x\n" * 4_000
        prompt.post_message(events.Paste(pasted))
        await pilot.pause()
        assert "#pasted-1" in prompt.area.text
        assert len(prompt.area.text) < 200
        # The text itself is kept, so the harness sees what was pasted.
        assert prompt.text() == pasted


async def test_a_rebound_key_is_honored(make_tui_app: MakeApp, tmp_path: Path) -> None:
    """No widget compares a key literally, and one keymap rebinds the app."""
    (tmp_path / "tui.json").write_text(
        json.dumps({"keybindings": {"submit": "ctrl+s", "command_palette": "ctrl+j"}})
    )
    async with running(make_tui_app()) as (app, pilot):
        assert app.front is not None
        await pilot.press(*"hi")
        await pilot.press("enter")
        await pilot.pause()
        # Enter is no longer submit, so it inserted a newline instead.
        assert app.front.state.status == "idle"
        assert not any(item.role == "user" for item in app.front.state.items)
        await pilot.press("ctrl+j")
        await pilot.pause()
        assert isinstance(app.screen, ChoicePicker), "the remapped palette key opened it"
        await pilot.press(app.keys.cancel)
        await pilot.pause()
        await pilot.press("ctrl+s")
        await until(pilot, turn_done(app))
        assert any(item.role == "user" for item in app.front.state.items)


async def test_a_global_key_does_not_also_edit_the_prompt(make_tui_app: MakeApp) -> None:
    """`ctrl+k` is the palette *and* the delete-to-end-of-line of `TextArea`.

    With priority bindings, the app claims the key first. Before them the key
    did both, silently.
    """
    async with running(make_tui_app()) as (app, pilot):
        prompt = app.query_one(PromptInput)
        prompt.area.insert("hello world")
        prompt.area.move_cursor((0, 5))
        await pilot.press(app.keys.command_palette)
        await pilot.pause()
        assert isinstance(app.screen, ChoicePicker)
        assert prompt.area.text == "hello world"


# ----------------------------------------------------------------- approval --


async def _decide(
    app: Any,  # noqa: ANN401
    pilot: Any,  # noqa: ANN401
    root: Root,
    *,
    arguments: Any = None,  # noqa: ANN401
) -> list[Any]:
    """Put one approval on screen and return the list that the answer lands in.

    This takes the daemon-side `root`, not the front end. The approval is raised
    where the harness is, and the modal appears where the user is. After P5-14
    those are the two ends of a socket. That separation is what is tested.
    """
    answers: list[Any] = []

    async def ask() -> None:
        answers.append(
            await root.ctx.require(APPROVAL).request(
                agent=StubAgent(ctx=root.ctx, session=root.session),
                tool_name="write",
                call_id="c1",
                reason="writes outside the workspace",
                arguments=arguments,
            )
        )

    app.run_worker(ask())
    await until(pilot, lambda: isinstance(app.screen, ApprovalModal))
    return answers


async def test_the_modal_answers_in_the_tools_voice(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """`respond` (P4-05): the body never runs and the model reads an answer.

    A user who knows the answer should not have to reject a call and then
    explain. This decision removes the cost of that round trip.
    """
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        answers = await _decide(app, pilot, root)

        await pilot.click("#approval-respond")
        await pilot.pause()
        # Enter in the box is the gesture that a user makes after typing. The
        # button stays as a second route for the mouse.
        app.screen.query_one("#approval-why", Input).value = "the port is 8080"
        await pilot.press("enter")
        await until(pilot, lambda: bool(answers))

        assert isinstance(answers[0], Responded)
        assert answers[0].message == "the port is 8080"
        decided = next(e for e in root.session.events if e.type == "approval/decided")
        assert decided.data["outcome"] == "responded"


async def test_the_modal_corrects_the_call_rather_than_refusing_it(
    make_tui_app: MakeApp,
    tui_daemon: Daemon,
) -> None:
    """`edit` (P4-05), which opens with the call as it stands.

    The box is prefilled because a user who corrects one wrong path should not
    retype the whole argument object. The user also needs to see what they
    edit, which is the request of the model itself.
    """
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        answers = await _decide(app, pilot, root, arguments={"path": "/etc/hosts"})

        await pilot.click("#approval-edit")
        await pilot.pause()
        box = app.screen.query_one("#approval-why", Input)
        assert json.loads(box.value) == {"path": "/etc/hosts"}, "the edit box did not prefill"
        box.value = json.dumps({"path": "notes.md"})
        await pilot.press("enter")
        await until(pilot, lambda: bool(answers))

        assert isinstance(answers[0], Edited)
        assert answers[0].arguments == {"path": "notes.md"}
        decided = next(e for e in root.session.events if e.type == "approval/decided")
        assert decided.data["arguments"] == {"path": "notes.md"}
        # The ask itself does not carry them, because `tool/call` already did.
        # Two copies of one fact in the log are two copies that can disagree.
        asked = next(e for e in root.session.events if e.type == "approval/asked")
        assert "arguments" not in asked.data


async def test_a_mistyped_edit_keeps_the_box_open(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """This is not a refusal. The user meant to edit and mistyped. To reject the
    call because of a stray comma would be the harness deciding for the user."""
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        answers = await _decide(app, pilot, root, arguments={"path": "a"})

        await pilot.click("#approval-edit")
        await pilot.pause()
        app.screen.query_one("#approval-why", Input).value = "{not json"
        await pilot.press("enter")
        await pilot.pause()

        assert not answers, "a typo was taken as a decision"
        assert isinstance(app.screen, ApprovalModal)


async def test_approval_round_trips_through_the_log(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """The P2-04 gate: asked, decided, and both in the log."""
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        outcomes = await _decide(app, pilot, root)
        assert isinstance(app.screen, ApprovalModal)
        assert app.screen.request.tool_name == "write"
        await pilot.click("#approval-approve")
        await until(pilot, lambda: bool(outcomes))

        assert outcomes == ["allowed-once"]
        types = [event.type for event in root.session.events]
        assert types.count("approval/asked") == 1
        assert types.count("approval/decided") == 1
        decided = next(e for e in root.session.events if e.type == "approval/decided")
        assert decided.data["outcome"] == "allowed-once"


async def test_rejecting_asks_for_a_reason_then_steers_with_it(make_tui_app: MakeApp) -> None:
    async with running(make_tui_app()) as (app, pilot):
        answers: list[Any] = []

        async def ask() -> None:
            answers.append(await app.ask_approval(ApprovalRequest(tool_name="write", call_id="c1")))

        app.run_worker(ask())
        await until(pilot, lambda: isinstance(app.screen, ApprovalModal))
        await pilot.click("#approval-reject")
        await pilot.pause()
        # The first press shows the reason field and does not decide.
        assert app.screen.has_class("-explaining")
        await pilot.press(*"use the helper")
        await pilot.press("enter")
        await until(pilot, lambda: bool(answers))
        assert answers == [("rejected", "use the helper")]


async def test_escape_denies_rather_than_leaving_it_open(make_tui_app: MakeApp) -> None:
    """Absence is not consent: dismissing the modal is a refusal."""
    async with running(make_tui_app()) as (app, pilot):
        answers: list[Any] = []

        async def ask() -> None:
            answers.append(await app.ask_approval(ApprovalRequest(tool_name="rm")))

        app.run_worker(ask())
        await until(pilot, lambda: isinstance(app.screen, ApprovalModal))
        await pilot.press(app.keys.cancel)
        await until(pilot, lambda: bool(answers))
        assert answers[0][0] == "rejected"


async def test_an_ask_is_withdrawn_by_its_root_and_its_name(make_tui_app: MakeApp) -> None:
    """Asks of two roots, both named `ask-1`, are up at the same time, and the
    lower one is withdrawn.

    The desk of each root counts its own asks, so a terminal that is attached to
    two roots can be asked `ask-1` by both. When asks were filed by the name
    alone, the second took the entry of the first, and `ask.settled` for the ask
    of one root took down the modal of the other root. Also, `dismiss` pops
    whichever screen is on top. To withdraw a modal that had another above it
    therefore took the upper one down instead, and the upper ask never got an
    answer.

    Sabotage: file `_open_asks` by `ask.ask_id` alone, or withdraw with a plain
    `dismiss`, and the upper ask is not the one left up.
    """
    async with running(make_tui_app()) as (app, pilot):
        answers: dict[str, Any] = {}

        async def ask(session_id: str, tool_name: str) -> None:
            answers[tool_name] = await app.ask_approval(
                ApprovalRequest(tool_name=tool_name), ask=AskKey(session_id, "ask-1")
            )

        def showing(tool_name: str) -> bool:
            screen = app.screen
            return isinstance(screen, ApprovalModal) and screen.request.tool_name == tool_name

        app.run_worker(ask("root-a", "lower"))
        await until(pilot, lambda: showing("lower"))
        app.run_worker(ask("root-b", "upper"))
        await until(pilot, lambda: showing("upper"))

        app.withdraw_ask(AskKey("root-a", "ask-1"))
        await pilot.pause()
        assert showing("upper"), "the other root's ask is still the one up"
        await pilot.click("#approval-approve")
        await until(pilot, lambda: len(answers) == 2)

        assert answers == {"upper": ("allowed-once", ""), "lower": ("rejected", "")}
        assert not isinstance(app.screen, ApprovalModal), (
            "the withdrawn modal went when it reached the top"
        )


# -------------------------------------------------------------- ask the user --


async def test_ask_user_with_options_returns_the_chosen_one(make_tui_app: MakeApp) -> None:
    async with running(make_tui_app()) as (app, pilot):
        answers: list[Any] = []

        async def ask() -> None:
            answers.append(
                await app.ask_question(
                    UserQuestion(question="which one?", options=["first", "second"])
                )
            )

        app.run_worker(ask())
        await until(pilot, lambda: isinstance(app.screen, AskUserModal))
        await pilot.press("down")
        await pilot.press("enter")
        await until(pilot, lambda: bool(answers))
        assert answers == ["second"]


async def test_ask_user_multi_select_joins_the_ticked_options(make_tui_app: MakeApp) -> None:
    async with running(make_tui_app()) as (app, pilot):
        answers: list[Any] = []

        async def ask() -> None:
            answers.append(
                await app.ask_question(
                    UserQuestion(question="which?", options=["a", "b", "c"], multi_select=True)
                )
            )

        app.run_worker(ask())
        await until(pilot, lambda: isinstance(app.screen, AskUserModal))
        await pilot.click("#ask-opt-0")
        await pilot.click("#ask-opt-2")
        await pilot.click("#ask-answer")
        await until(pilot, lambda: bool(answers))
        assert answers == ["a, c"]


async def test_ask_user_without_options_takes_free_text(make_tui_app: MakeApp) -> None:
    async with running(make_tui_app()) as (app, pilot):
        answers: list[Any] = []

        async def ask() -> None:
            answers.append(await app.ask_question(UserQuestion(question="what port?")))

        app.run_worker(ask())
        await until(pilot, lambda: isinstance(app.screen, AskUserModal))
        await pilot.press(*"8080")
        await pilot.press("enter")
        await until(pilot, lambda: bool(answers))
        assert answers == ["8080"]


# ------------------------------------------------------------------ pickers --


async def test_the_command_palette_inserts_a_command(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        front = app.front
        assert front is not None
        root.ctx.require(COMMANDS).register(_command())
        # A command that is registered on the daemon reaches this palette as a
        # `session.commands` notification. It is therefore there one frame
        # later and not at once. That is the real cost of a list that is one
        # socket away.
        await until(pilot, lambda: any(one.name == "tidy" for one in front.commands()))
        await pilot.press(app.keys.command_palette)
        await pilot.pause()
        assert isinstance(app.screen, ChoicePicker)
        await pilot.press(*"tidy")
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(PromptInput).area.text.startswith("/tidy")


async def test_a_picker_blocks_the_other_global_keys(make_tui_app: MakeApp) -> None:
    """While one picker is open, the key of another must not stack a second one on top."""
    async with running(make_tui_app()) as (app, pilot):
        await pilot.press(app.keys.theme_picker)
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, ChoicePicker)
        await pilot.press(app.keys.model_picker)
        await pilot.pause()
        assert app.screen is picker
        assert len(app.screen_stack) == 2


async def test_the_model_picker_lists_the_profile_s_models_and_moves_the_root(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """`/model` was cosmetic: it rewrote the footer and nothing ran on it. Now the
    picker offers the keys of the profile, and the free text offers a whole
    route. The choice belongs to the daemon: the agent of the root moves, and
    then the footer follows.

    Sabotage: set the footer in `_set_model` again and skip the daemon, and the
    root stays on `fake-1` while the footer says otherwise."""
    async with running(make_tui_app()) as (app, pilot):
        front = not_none(app.front)
        assert [one.key for one in front.models()] == ["main"]
        assert front.state.model_key == "main"

        await pilot.press(app.keys.model_picker)
        await pilot.pause()
        assert isinstance(app.screen, ChoicePicker)
        await pilot.press(*"fake/fake-9")
        await pilot.press("enter")

        root = root_of(tui_daemon)
        await until(pilot, lambda: root.agent.options.model == "fake-9")
        await until(pilot, lambda: front.state.model == "fake-9")
        assert front.state.model_key == "fake-fake-9", "an unlisted route joins the list by name"


async def test_a_typed_model_moves_the_root_without_the_picker(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """`/model fake/fake-9` moves the root as the picker's value would. Textual
    calls an action with only as many arguments as it accepts, and
    `action_open_models` took none. The typed route was dropped without a word,
    and the picker opened instead."""
    async with running(make_tui_app()) as (app, pilot):
        front = not_none(app.front)

        await app._dispatch_command("/model fake/fake-9")

        root = root_of(tui_daemon)
        await until(pilot, lambda: root.agent.options.model == "fake-9")
        await until(pilot, lambda: front.state.model == "fake-9")
        assert not isinstance(app.screen, ChoicePicker), "the picker opened anyway"


async def test_a_typed_argument_to_a_verb_that_takes_none_is_refused(
    make_tui_app: MakeApp,
) -> None:
    """`/theme dark` names no argument that `/theme` takes. The line is refused
    and says so, and the picker does not open as if nothing had been typed."""
    async with running(make_tui_app()) as (app, pilot):
        await app._dispatch_command("/theme dark")
        await pilot.pause()

        said = [str(one.message) for one in app._notifications]
        assert not isinstance(app.screen, ChoicePicker), "the argument was dropped"
        assert "/theme takes no argument" in said, said


async def test_the_terminal_s_own_route_rides_its_session_new(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """`phern --mode tui --model …` moves the session of this terminal, not the
    daemon: `spawn_command` carries no route, and `session/new` carries this
    one. When the route is refused, the start fails with the sentence of the
    daemon, in the way that `phern -p --model` exits. It does not open on a
    route that the user did not ask for."""
    chosen = ModelChoice(provider="fake", model="fake-5")
    async with running(make_tui_app(model_choice=chosen)) as (app, pilot):
        await until(pilot, lambda: root_of(tui_daemon).agent.options.model == "fake-5")
        assert not_none(app.front).state.model == "fake-5"

    # The same session again, now with a request for a key that the profile does
    # not list. This uses `run_test` and not `running`, because `running` waits
    # for an attach that this must not get.
    refused = make_tui_app(model_choice=ModelChoice(key="nope"))
    async with refused.run_test() as pilot:
        await pilot.pause()
        await until(pilot, lambda: isinstance(refused.screen, ConfirmModal))
        assert refused.front is None, "a refused route opened a session anyway"
        assert "not a model this profile lists" in _refused_start(refused)
        await pilot.click("#act-quit")
        await until(pilot, lambda: refused.return_code is not None)
    assert refused.return_code == 1, "a terminal that never had a session did not succeed"
    assert root_of(tui_daemon).agent.options.model == "fake-5", "and it moved nothing"


# ------------------------------------------------------- the start that failed --


def _refused_start(app: App[Any]) -> str:
    """The sentence on the modal that a failed start put up, which is what the user reads."""
    modal = app.screen
    assert isinstance(modal, ConfirmModal), f"no modal is up; the screen is {modal!r}"
    return modal.body_text


def _status_line(app: PHTuiApp, part: str = "line") -> str:
    """What the status bar draws: its line, or (`part="glyph"`) the spinner cell."""
    return str(app.query_one(StatusBar).query_one(f"#status-{part}").render())


async def _submit(app: PHTuiApp, pilot: Pilot[str | None], line: str) -> None:
    """Type `line` and send it.

    A slash line opens the completion popup, which claims enter. Tab accepts,
    and then enter submits. Plain text opens no popup. Tab with no popup belongs
    to the text area, where it moves focus. Tab is therefore pressed only when
    there is a list to accept.
    """
    await pilot.press(*line)
    await pilot.pause()
    if line.startswith("/"):
        await pilot.press(app.keys.accept_completion)
    await pilot.press(app.keys.submit)


async def test_a_start_that_fails_stays_said_and_the_terminal_still_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug that this guards against: a daemon that would not start was
    reported in a toast, and a toast lasts five seconds. It left a terminal with
    its headings drawn and nothing behind them. Every typed line was swallowed,
    `/quit` included, because every slash command went through the front end,
    which was never built. The user who met it read it as a hang.

    No daemon listens on the `$PH_RUNTIME` of this test, and `spawn=False`
    refuses to start one. The open therefore fails in the way that it fails for
    a daemon that exits while it starts. Sabotage: put the bare `return` back
    under `if front is None` in `on_prompt_input_submitted`, and `/theme` opens
    nothing; drop `show_unattached`, and Esc leaves a blank footer over the same
    dead prompt.
    """
    private_runtime(tmp_path, monkeypatch)
    app = tui_app(home=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await until(pilot, lambda: isinstance(app.screen, ConfirmModal))
        assert app.front is None
        assert "no daemon at" in _refused_start(app), "the launcher's own sentence"

        # Esc stays. The sentence moves to the footer, and the prompt answers.
        await pilot.press("escape")
        await until(pilot, lambda: not isinstance(app.screen, ConfirmModal))
        assert "no session" in _status_line(app)
        assert "no daemon at" in _status_line(app)
        await _submit(app, pilot, "hello?")
        await until(pilot, lambda: len(app._notifications) > 0)
        assert any("no daemon at" in str(one.message) for one in app._notifications), (
            "a line typed at no session is answered, not swallowed"
        )

        # This terminal's own verbs still run: `/theme` is one of them.
        await _submit(app, pilot, "/theme")
        await until(pilot, lambda: isinstance(app.screen, ChoicePicker))
        await pilot.press("escape")
        await until(pilot, lambda: not isinstance(app.screen, ChoicePicker))

        # A verb that needs a session says so rather than doing nothing.
        before = len(app._notifications)
        await pilot.press(app.keys.session_picker)
        await until(pilot, lambda: len(app._notifications) > before)

        # And `/quit` leaves, with the code of a start that never happened.
        await _submit(app, pilot, "/quit")
        await until(pilot, lambda: app.return_code is not None)
    assert app.return_code == 1


async def test_try_again_reruns_the_open_after_a_refused_start(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """**Try again** is for the user whose fix is in another terminal: a daemon
    that was started by hand, or an install that was repaired. That user should
    not have to reopen the terminal to find out. The whole open runs again, and
    the refusal is withdrawn.

    A refused route stands in for the daemon that would not start. It is the
    other way in which an open fails after `ensure_daemon`, and it is the one
    that a test can turn on and off. The connection that the first attempt
    opened is dropped before the second attempt, and it is dropped quietly. The
    watcher that reports a daemon *going away* must not fire for a socket that
    this terminal closed on purpose.
    """
    app = make_tui_app(model_choice=ModelChoice(key="nope"))
    async with app.run_test() as pilot:
        await pilot.pause()
        await until(pilot, lambda: isinstance(app.screen, ConfirmModal))
        assert "not a model this profile lists" in _refused_start(app)
        assert app.refusal is not None
        app.model_choice = ModelChoice()
        await pilot.click("#act-retry")
        await until(pilot, lambda: app.front is not None)
        assert app.refusal is None
        await until(pilot, lambda: "no session" not in _status_line(app))
        assert not any("lost the daemon" in str(one.message) for one in app._notifications)


async def test_the_theme_picker_previews_and_applies(make_tui_app: MakeApp, tmp_path: Path) -> None:
    async with running(make_tui_app()) as (app, pilot):
        before = app.theme
        await pilot.press(app.keys.theme_picker)
        await pilot.pause()
        assert isinstance(app.screen, ChoicePicker)
        await pilot.press("down")
        await pilot.pause()
        # Previewed live, before anything was chosen.
        assert app.theme != before
        await pilot.press("enter")
        await pilot.pause()
        assert app.theme != before
        # And remembered: the next launch opens in the chosen theme. The choice
        # is in `$PH_HOME/themes/theme-profile.yaml`, which this pick creates.
        # Before the pick there is no theme file at all (P9-02).
        assert load_theme_profile(tmp_path).chosen == app.theme


async def test_dismissing_the_theme_picker_restores_the_setting(make_tui_app: MakeApp) -> None:
    async with running(make_tui_app()) as (app, pilot):
        before = app.theme
        await pilot.press(app.keys.theme_picker)
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        await pilot.press(app.keys.cancel)
        await pilot.pause()
        assert app.theme == before


async def test_the_permission_picker_records_the_posture(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        front = app.front
        assert front is not None
        await pilot.press(app.keys.permission_picker)
        await pilot.pause()
        assert isinstance(app.screen, ChoicePicker)
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        presets = [e for e in root.session.events if e.type == "permission/preset"]
        assert presets, "the service records the posture, not the front-end"
        chosen = presets[-1].data["preset"]

        # The picker also reads the posture back from the seam, and not from a
        # fold of its own. It used to mark nothing on a session that somebody
        # had already switched, because a client that attached afterwards has no
        # event to fold. When the picker asks again, the mark is on what the log
        # now says.
        marked = [one.name for one in await front.presets() if one.active]
        assert marked == [chosen]


async def test_the_session_picker_lists_and_reopens(make_tui_app: MakeApp) -> None:
    """Choosing a stored session exits with its id, and `run_tui` reopens it."""
    async with running(make_tui_app(session_id="earlier")) as (app, pilot):
        assert app.front is not None
        await pilot.press(*"remember this")
        await pilot.press(app.keys.submit)
        await until(pilot, turn_done(app))

    app = make_tui_app(session_id="later")
    async with running(app) as (app, pilot):
        assert app.front is not None
        await pilot.press(app.keys.session_picker)
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, ChoicePicker)
        assert {choice.value for choice in picker.choices} >= {"earlier", "later"}
        assert any(choice.label == "remember this" for choice in picker.choices)
        current = next(choice for choice in picker.choices if choice.value == "later")
        assert current.marked, "the open session is marked as such"
        await pilot.press(*"earlier")
        await pilot.press("enter")
        await pilot.pause()
    assert app.return_value == "earlier"


# ------------------------------------------------------------------ toggles --


async def test_toggling_tool_results_is_remembered(make_tui_app: MakeApp, tmp_path: Path) -> None:
    async with running(make_tui_app()) as (app, pilot):
        assert app.settings.show_tool_results is True
        await pilot.press(app.keys.toggle_tool_results)
        await pilot.pause()
        assert app.settings.show_tool_results is False
        saved = json.loads((tmp_path / "tui.json").read_text())
        assert saved["show_tool_results"] is False


# -------------------------------------------------------- trust and planning --


async def test_an_untrusted_project_is_not_mounted_until_answered(make_tui_app: MakeApp) -> None:
    """The gate: the project's configuration is read only after the answer."""
    app = make_tui_app(trusted=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.front is None, "nothing may be mounted before the answer"
        assert isinstance(app.screen, ConfirmModal)
        await pilot.click("#act-always")
        await until(pilot, lambda: app.front is not None)
        assert app.trust.trusted(app.project) is True


async def test_trusting_once_does_not_persist(make_tui_app: MakeApp) -> None:
    app = make_tui_app(trusted=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#act-once")
        await until(pilot, lambda: app.front is not None)
        assert app.trust.trusted(app.project) is False


async def test_declining_trust_exits(make_tui_app: MakeApp) -> None:
    app = make_tui_app(trusted=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click("#act-no")
        await pilot.pause()
        assert app.front is None


async def test_project_trust_is_remembered_outside_the_project(tmp_path: Path) -> None:
    project = tmp_path / "repo"
    project.mkdir()
    store = TrustStore(path=tmp_path / "trust.json")
    assert store.trusted(project) is False
    store.trust(project)
    assert TrustStore(path=tmp_path / "trust.json").trusted(project) is True
    # Nothing was written into the project. A repo cannot vouch for itself.
    assert list(project.iterdir()) == []


async def test_the_trust_modal_offers_three_answers(make_tui_app: MakeApp, tmp_path: Path) -> None:
    async with running(make_tui_app()) as (app, pilot):
        chosen: list[Any] = []
        app.push_screen(project_trust_modal(tmp_path), chosen.append)
        await pilot.pause()
        await pilot.click("#act-once")
        await pilot.pause()
        assert chosen == ["once"]


async def test_dismissing_the_trust_modal_does_not_load_the_project(
    make_tui_app: MakeApp, tmp_path: Path
) -> None:
    """The cancel key answers "Don't load it". A dismissal must never be read as
    consent to run the configuration of a project.

    Sabotage: drop `cancel_value="no"` from `project_trust_modal`, and the
    dismissal answers `None`.
    """
    async with running(make_tui_app()) as (app, pilot):
        chosen: list[Any] = []
        app.push_screen(project_trust_modal(tmp_path), chosen.append)
        await pilot.pause()
        await pilot.press(app.keys.cancel)
        await pilot.pause()
        assert chosen == ["no"]


async def test_the_trust_dialog_is_walked_by_tab_and_the_arrows(
    make_tui_app: MakeApp, tmp_path: Path
) -> None:
    """The dialog opens on "this session only", and the keyboard reaches every
    answer.

    Tab and Shift+Tab belong to Textual, which routes them to app actions. The
    app refused those actions behind a dialog, while its own keys were bound on
    the app and kept quiet by a check. Focus therefore could not leave the first
    button, and Enter meant "trust it forever". The arrows belong to the dialog
    and lead to the same actions.

    Sabotage: give `PHTuiApp` a `check_action` that refuses every action but
    quit while a dialog is up, and Tab leaves focus where it was.
    """
    async with running(make_tui_app()) as (app, pilot):
        app.push_screen(project_trust_modal(tmp_path))
        await pilot.pause()

        def focused() -> str | None:
            return app.focused.id if app.focused is not None else None

        assert focused() == "act-once", "Enter on an unread prompt must not trust for good"
        for key, expected in (
            ("tab", "act-no"),
            ("shift+tab", "act-once"),
            ("up", "act-always"),
            ("down", "act-once"),
            ("right", "act-no"),
            ("left", "act-once"),
        ):
            await pilot.press(key)
            await pilot.pause()
            assert focused() == expected, f"{key} left focus on {focused()}"


async def test_the_chat_s_keys_are_quiet_behind_a_dialog_and_quit_is_not(
    make_tui_app: MakeApp, tmp_path: Path
) -> None:
    """A dialog keeps the keys of the chat out of reach by being on top, because
    they are bound on the screen of the chat itself. Quit is bound on the app
    and still works.

    Sabotage: bind the chat's keys on the app (`PHTuiApp.BINDINGS =
    screen_bindings(...)`), and the theme picker opens over the trust dialog.
    """
    async with running(make_tui_app()) as (app, pilot):
        app.push_screen(project_trust_modal(tmp_path))
        await pilot.pause()
        dialog = app.screen

        await pilot.press(app.keys.theme_picker)
        await pilot.pause()
        assert app.screen is dialog, "a chat key opened a picker over the dialog"

        await pilot.press(app.keys.quit)
        await pilot.pause()
        assert app._exit, "quit did not reach the app from behind the dialog"


async def test_the_approval_dialog_is_walked_by_the_arrows_and_its_box_keeps_them(
    make_tui_app: MakeApp,
) -> None:
    """The arrows move among the decisions, and only there. After a decision
    opens the reason box, Left and Right move the cursor of the box, and Up does
    not move focus out of it. The box handles Left and Right itself. Up is the
    key that escapes when the arrows are bound wider than the answers.

    Sabotage: bind the arrows on the dialog (`app.focus_next`, `app.focus_previous`)
    rather than on `Answers`, and Up in the open box moves focus out of it.
    """
    async with running(make_tui_app()) as (app, pilot):
        request = ApprovalRequest(tool_name="bash", arguments={"command": "ls"})
        app.push_screen(ApprovalModal(request))
        await pilot.pause()

        def focused() -> str | None:
            return app.focused.id if app.focused is not None else None

        assert focused() == "approval-approve"
        for key, expected in (
            ("down", "approval-edit"),
            ("down", "approval-respond"),
            ("up", "approval-edit"),
            ("right", "approval-respond"),
            ("left", "approval-edit"),
        ):
            await pilot.press(key)
            await pilot.pause()
            assert focused() == expected, f"{key} left focus on {focused()}"

        await pilot.press("enter")  # "Edit arguments": the box opens on the call
        await pilot.pause()
        box = app.screen.query_one("#approval-why", Input)
        assert app.focused is box
        await pilot.press("end")
        cursor = box.cursor_position
        await pilot.press("left")
        await pilot.pause()
        assert box.cursor_position == cursor - 1, "Left did not move the box's cursor"
        await pilot.press("up")
        await pilot.pause()
        assert app.focused is box, "Up threw focus out of the reason box"


@pytest.mark.parametrize("dialog", ["trust", "approval"])
async def test_a_dialog_s_answers_are_a_centered_column(
    make_tui_app: MakeApp, tmp_path: Path, dialog: str
) -> None:
    """The middle of each answer is the middle of the dialog, whatever the
    length of its label. That holds in every dialog with answers, because they
    share one `Answers` column.

    Sabotage: build `Answers`' children without their `Center`, and the answers
    line up on the left edge with different middles.
    """
    async with running(make_tui_app()) as (app, pilot):
        app.push_screen(
            project_trust_modal(tmp_path)
            if dialog == "trust"
            else ApprovalModal(ApprovalRequest(tool_name="bash"))
        )
        await pilot.pause()
        column = app.screen.query_one(Answers).content_region
        middle = column.x + column.width / 2
        for button in app.screen.query(Button):
            region = button.region
            offset = region.x + region.width / 2 - middle
            assert abs(offset) <= 0.5, f"{button.id} is {offset} columns off center"


async def test_a_picker_filters_on_typed_text(make_tui_app: MakeApp) -> None:
    async with running(make_tui_app()) as (app, pilot):
        chosen: list[Any] = []
        app.push_screen(
            ChoicePicker(title="pick", choices=[Choice("a", "alpha"), Choice("b", "beta")]),
            chosen.append,
        )
        await pilot.pause()
        await pilot.press(*"bet")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert chosen == ["b"]


class _Picking(App[None]):
    """A bare app to open a picker on. `PHTuiApp` would bring a daemon, a socket
    and a mount with it, and a picker needs none of them. It needs only the
    `$ph-*` variables that its CSS names. Textual refuses to parse the CSS
    without them."""

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return fallback_variables()


async def test_a_row_that_cannot_be_chosen_leaves_the_picker_open() -> None:
    """P11-08: a row can be seen and not taken, such as the log of a sub-agent
    among the sessions.

    Choosing the row says why and answers nothing. No caller therefore receives
    an id that it would send only to have it refused. The rows around it answer
    as before. The test runs without a daemon, so it holds in a sandbox that has
    no sockets.

    Sabotage: dismiss with an unavailable row's value, and `chosen` holds it.
    """
    app = _Picking()
    async with app.run_test() as pilot:
        chosen: list[Any] = []
        app.push_screen(
            ChoicePicker(
                title="sessions",
                choices=[
                    Choice("helper", "  ⤷ a child", unavailable="read it elsewhere"),
                    Choice("lead", "the root"),
                ],
            ),
            chosen.append,
        )
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert chosen == [], "an unavailable row answered with its value"
        assert isinstance(app.screen, ChoicePicker), "and the picker stayed open"
        assert [str(one.message) for one in app._notifications] == ["read it elsewhere"]
        await pilot.press("down", "enter")
        await pilot.pause()
        assert chosen == ["lead"]


# -------------------------------------------------------------- completions --


async def test_typing_a_slash_offers_registered_commands(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """The completion sources are the registries of pH, not a second list (I7)."""
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        root.ctx.require(COMMANDS).register(_command())
        prompt = app.query_one(PromptInput)
        await pilot.press(*"/tid")
        await pilot.pause()
        assert prompt.has_class("-completing")
        await pilot.press(app.keys.accept_completion)
        await pilot.pause()
        assert prompt.area.text == "/tidy "
        assert not prompt.has_class("-completing")


async def test_a_disposed_command_leaves_the_completion_list(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        dispose = root.ctx.require(COMMANDS).register(_command())
        prompt = app.query_one(PromptInput)
        # A prefix that matches *only* the fixture command. `/comp` now also
        # matches the shipped `/compact`, so the list would stay open after the
        # disposal, and the gate would pass for the wrong reason.
        await pilot.press(*"/tid")
        await pilot.pause()
        assert prompt.has_class("-completing")
        prompt.clear()
        dispose()
        await pilot.press(*"/tid")
        await pilot.pause()
        # The registry is the source of truth, so unregistering removes the row.
        assert not prompt.has_class("-completing")


async def test_escape_closes_the_completion_list_before_interrupting(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        root.ctx.require(COMMANDS).register(_command())
        prompt = app.query_one(PromptInput)
        await pilot.press(*"/comp")
        await pilot.pause()
        await pilot.press(app.keys.cancel)
        await pilot.pause()
        assert not prompt.has_class("-completing")
        # And the typed text survives: escape closed the popup, not the prompt.
        assert prompt.area.text == "/comp"


async def test_the_idle_prompt_is_one_row_with_its_text_in_the_transcripts_column(
    make_tui_app: MakeApp,
) -> None:
    """No frame and no hint row: an empty prompt costs the terminal one line, and
    the typed text starts where the text of the conversation starts.

    This is asked while the prompt has focus, because focus is what brought a
    border back. The `TextArea:focus` rule of Textual outranks a plain type
    selector. A `border: none` that held for a blurred box therefore failed for
    the only state in which a user ever sees the box. The measurement is against
    a real transcript row and not a column number, because the alignment is the
    purpose and either side can move. The gutter of the transcript is about to
    move.

    Sabotage: drop `compact=True` from the `PromptArea` in `PromptInput.compose`,
    and the box is three rows tall.
    """
    async with running(make_tui_app()) as (app, pilot):
        await pilot.press(*"hi")
        await pilot.press(app.keys.submit)
        await until(pilot, turn_done(app))
        await pilot.pause(0.1)
        prompt = app.query_one(PromptInput)
        said = app.query_one(".-user > .row-body").content_region

        assert app.focused is prompt.area
        assert prompt.area.region.height == 1, "the box has a frame"
        assert prompt.area.content_region.x == said.x, "the prompt's text is out of line"
        assert not prompt.query_one("#hint").display, "an empty hint still takes a row"
        assert prompt.region.height == 1


async def test_the_completion_list_and_its_hint_line_up_with_the_prompt(
    make_tui_app: MakeApp,
) -> None:
    """Every row under the box starts its text where the box starts its text,
    and the hint is there exactly as long as the list is.

    Sabotage: remove `padding-left: 1` from `#completions`, and the list starts a
    column left of the text it completes.
    """
    async with running(make_tui_app()) as (app, pilot):
        prompt = app.query_one(PromptInput)
        listing = prompt.query_one("#completions")
        hint = prompt.query_one("#hint")
        await pilot.press("/")
        await pilot.pause()
        assert prompt.has_class("-completing")
        column = prompt.area.content_region.x

        assert listing.content_region.x == column, "the list is not under the prompt's text"
        assert hint.display and hint.content_region.x == column
        # The shipped commands outnumber the six rows of the list, so the list
        # scrolls. Its scrollbar is in the same gutter that the transcript has:
        # one column, at the left.
        assert listing.max_scroll_y > 0, "the list fit; the scrollbar is untested"
        assert listing.vertical_scrollbar.region.x == listing.region.x
        assert listing.vertical_scrollbar.region.width == 1

        await pilot.press(app.keys.cancel)
        await pilot.pause()
        assert not hint.display, "the hint outlived its list"


# ----------------------------------------------------------------- commands --


async def test_a_slash_line_dispatches_instead_of_prompting(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """A command spends no model turn, and the log says who decided.

    This uses a **harness** verb, deliberately. A client-side verb, such as
    `/theme` or `/model`, changes the display of this terminal and is not an act
    in the session. It is therefore dispatched here and appears in no log. If
    another attached UI rendered "somebody opened a theme picker", it would
    report a fact about a screen that it cannot see. The log records what the
    *session* did, and that is what this test asserts.
    """
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        front = app.front
        assert front is not None
        root.ctx.require(COMMANDS).register(_command())
        await until(pilot, lambda: any(one.name == "tidy" for one in front.commands()))
        await pilot.press(*"/tidy")
        await pilot.pause()
        # The completion popup claims enter first. Tab accepts, and then enter
        # submits.
        await pilot.press(app.keys.accept_completion)
        await pilot.press(app.keys.submit)
        await until(pilot, lambda: "command/run" in [e.type for e in root.session.events])

        types = [event.type for event in root.session.events]
        assert "command/done" in types
        assert not any(kind.startswith("turn/") for kind in types), "a command is not a turn"


async def test_an_unknown_command_is_reported_not_prompted(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        app.query_one(PromptInput).area.insert("/nonsense")
        await pilot.press(app.keys.submit)
        await pilot.pause()
        types = [event.type for event in root.session.events]
        assert not any(kind.startswith("turn/") for kind in types)
        assert "user/message" not in types


def _bound_ids(app: PHTuiApp) -> set[str]:
    """Every key binding id that the chat answers to: those of the app (quit,
    which works behind a dialog) and those of its base screen (the rest)."""
    declared = [*app.BINDINGS, *ChatScreen.BINDINGS]
    return {binding.id for binding in declared if isinstance(binding, Binding) and binding.id}


async def test_every_verb_is_a_command_an_action_and_maybe_a_key(
    make_tui_app: MakeApp,
) -> None:
    """One table, three routes: a verb that lacks any of them is half wired.

    Every row is a command and an action again, because the keyed spellings of
    `/view` moved out to `VIEW_KEYS`. They were never verbs, and while they were
    held here, every consumer of this table had to filter. `VIEW_KEYS` is
    checked below instead: a binding id that it names must be remappable and
    must fire an action that exists.

    A verb's `argument_hint` and its action must also agree. Textual calls an
    action with only as many arguments as it accepts, so an action without a
    parameter loses a typed argument without a word, and a hint is the only
    thing that makes `_RunAction` pass one on.
    """
    from ph_app.tui.commands import TUI_VERBS, VIEW_KEYS, VIEWABLE

    async with running(make_tui_app()) as (app, _pilot):
        front = app.front
        assert front is not None
        # `app.commands()`, not the registry of the daemon. After P5-14 a verb
        # that changes the display of *this* client is built client-side and is
        # never registered anywhere. The list that a user sees is therefore the
        # merge of both ends, which is what "reachable as a command" must mean
        # now.
        registered = {definition.name for definition in app.commands()}
        bound = _bound_ids(app)
        for verb in TUI_VERBS:
            assert verb.name in registered
            action = getattr(app, f"action_{verb.action}")
            takes = any(
                one.kind in (one.POSITIONAL_ONLY, one.POSITIONAL_OR_KEYWORD)
                for one in inspect.signature(action).parameters.values()
            )
            assert takes == bool(verb.argument_hint), f"/{verb.name}: hint and action disagree"
            if verb.key is not None:
                assert verb.key in bound
                assert hasattr(app.keys, verb.key), verb.key
        for key, word, _summary in VIEW_KEYS:
            assert key in bound and hasattr(app.keys, key), key
            assert word in VIEWABLE, f"{key} fires /view {word}, which /view does not take"


async def test_view_toggles_each_panel_and_all_resolves_a_disagreement(
    make_tui_app: MakeApp,
) -> None:
    """`/view`, and the one case where a toggle has no single correct answer.

    Two independent flags cannot be *toggled* together when they disagree, so
    `all` shows both. A user who types it at a half-hidden sidebar asks for
    everything, and not for the inverse of each flag. Both halves are asserted,
    because the rule only needs its comment in the mixed case.

    The setting is persisted, unlike `/view sidebar`. The visibility of a panel
    is a preference, and the visibility of the sidebar belongs to this window.
    `TuiSettings` already draws that line between `show_*` and the sidebar
    position.
    """
    async with running(make_tui_app()) as (app, _pilot):
        assert app.settings.show_tools and app.settings.show_skills, "visible by default"

        await app.run_action("view('tools')")
        assert not app.settings.show_tools and app.settings.show_skills

        # Mixed: one showing, one hidden. `all` makes both visible.
        await app.run_action("view('all')")
        assert app.settings.show_tools and app.settings.show_skills

        # Agreed: `all` is an ordinary toggle again.
        await app.run_action("view('all')")
        assert not app.settings.show_tools and not app.settings.show_skills

        await app.run_action("view('skills')")
        assert app.settings.show_skills and not app.settings.show_tools

        # The two transcript toggles, which `all` deliberately does not touch.
        # `all` is about the panels, and a user who asks to see the sidebar did
        # not ask for a rebuild of the transcript. They are asserted concretely
        # because the table no longer carries a reader that could check them
        # generically.
        for word, field in (("results", "show_tool_results"), ("thinking", "show_thinking")):
            before = getattr(app.settings, field)
            await app.run_action(f"view({word!r})")
            assert getattr(app.settings, field) is not before, f"/view {word} flipped nothing"


async def test_view_keeps_the_keys_the_old_toggles_had(make_tui_app: MakeApp) -> None:
    """`ctrl+o` and `ctrl+t` still do what they did, under a name that moved.

    `tui.json` remaps the binding *id*, so a user who rebound
    `toggle_tool_results` keeps their key across this rename. That is the reason
    why those rows still exist, and why the keys do not hang off `/view`.
    """
    async with running(make_tui_app()) as (app, pilot):
        before = app.settings.show_tool_results

        await pilot.press(app.keys.toggle_tool_results)
        await pilot.pause()

        assert app.settings.show_tool_results is not before
        bound = _bound_ids(app)
        assert {"toggle_tool_results", "toggle_thinking", "toggle_sidebar"} <= bound


async def test_login_stores_a_secret_without_logging_it(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    from ph_app.tui.modals.login import LoginModal

    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        await app.run_action("open_login")
        await pilot.pause()
        # No adapter row declares a key in the headless profile, so the name is
        # typed. That is the free-text path that the picker offers.
        await pilot.press(*"PH_TEST_KEY")
        await pilot.press(app.keys.submit)
        await until(pilot, lambda: isinstance(app.screen, LoginModal))
        await pilot.press(*"s3cret")
        await pilot.press("enter")
        await pilot.pause()

        credentials = root.ctx.require(CREDENTIALS)
        resolved = credentials.resolve(credentials.reference("PH_TEST_KEY"))
        assert resolved is not None
        assert resolved.reveal() == "s3cret"
        # The secret is in the process and nowhere else.
        assert "s3cret" not in repr([event.data for event in root.session.events])
        assert "s3cret" not in repr(resolved)

        # **And the picker knows that it is set now.** `refresh_credentials`
        # asks the daemon, and `action_open_login` awaits it before it builds
        # the picker (on a worker, H7). The front end once answered a
        # synchronous question about one name from a cache that nothing had
        # filled. It then said `False` for every credential, and a deployment
        # with all of them set looked like one with none. No test noticed,
        # because nothing asserted the marker.
        assert app.front is not None
        held = await app.front.refresh_credentials()
        assert held["PH_TEST_KEY"] is True, "the front end cannot see its own store"


# ------------------------------------------------------------------- status --


async def test_the_context_gauge_warns_from_the_share_the_root_compacts_at(
    make_tui_app: MakeApp, tui_daemon: Daemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The number a user needs to see coming is the one where pH will act.

    That number belongs to the root. The compaction seam of the root answers it,
    the daemon states it in the status of the root, and the gauge reads it from
    there. The terminal used to hold a copy of the default (0.85). The gauge
    then turned amber at 85% for a root that compacts at another share, and for
    a root that never compacts.

    Sabotage: compare with a constant 0.85 in `StatusBar.show`, and the gauge of
    a root that compacts at half its window is not amber at 55%.
    """
    async with running(make_tui_app()) as (app, pilot):
        front = not_none(app.front)
        root = root_of(tui_daemon)
        line = app.query_one(StatusBar).query_one("#status-line")
        seam = root.ctx.require(COMPACTION)
        assert front.state.compacts_at == seam.trigger_fraction(), "attach did not state the share"

        async def compacting_at(share: float | None) -> None:
            """The seam answers `share`, and the root tells its watchers."""
            monkeypatch.setattr(type(seam), "trigger_fraction", lambda _seam: share)
            root.publish(root.status_notice())
            await until(pilot, lambda: front.state.compacts_at == share)

        async def gauge_at(tokens: int) -> None:
            front.state.context_window = 1_000
            front.state.tokens = tokens
            app.state_changed()
            await pilot.pause(0.1)

        await compacting_at(0.5)
        await gauge_at(450)
        plain = _ink(line, "fake-1")
        assert _ink(line, "context 45%") == plain, "the gauge warned before the share"
        await gauge_at(550)
        assert _ink(line, "context 55%") != plain, "the gauge did not warn at the share"

        await compacting_at(None)
        await gauge_at(990)
        assert _ink(line, "context 99%") == plain, "a root that never compacts has no warning"


async def test_the_footer_joins_what_qualifies_a_figure_and_separates_the_rest(
    make_tui_app: MakeApp,
) -> None:
    """`● fake-1/high · turn 3 · context 12%/cache 28.8k(98%) · read-only accepted`.

    The readings are built by hand and do not come from the seam, so the test is
    about the shape of the footer itself:

    - One `·` between categories, with a single space on each side.
    - `/` within one category.
    - The `model` and `context` readings on the figure that they qualify,
      wherever the registry ordered them.
    - A `session` reading nowhere on the line at all.
    """
    async with running(make_tui_app()) as (app, _pilot):
        front = app.front
        assert front is not None
        front.state.turn = 3
        front.state.context_window = 1_000
        front.state.tokens = 120
        readings = [
            StatusReading(id="reasoning", slot="model", text="high"),
            StatusReading(id="posture", slot="line", text="read-only accepted"),
            StatusReading(id="sandbox-mode", slot="session", text="workspace-write"),
            StatusReading(id="cache", slot="context", text="cache 28.8k(98%)"),
        ]
        # Drawn and read with no await between the two, so no frame of the app
        # itself can redraw the line from the readings of the seam in the gap.
        app.query_one(StatusBar).show(front.state, readings)

        assert _status_line(app) == (
            "fake-1/high · turn 3 · context 12%/cache 28.8k(98%) · read-only accepted"
        )


async def test_every_reading_slot_is_drawn(make_tui_app: MakeApp) -> None:
    """Each `ReadingSlot` value lands on the status line or in the session panel.

    This is the check that `in_slot` no longer makes at runtime. Without it, a
    slot that ph-core adds and that no widget here asks for would draw nowhere,
    silently. The readings are drawn into the widgets by hand, so the test
    measures placement and not which rows mount.

    Sabotage: drop the `in_slot(readings, "context")` read from `StatusBar.show`,
    and the failure names `context`.
    """
    async with running(make_tui_app()) as (app, _pilot):
        front = app.front
        assert front is not None
        slots = get_args(ReadingSlot)
        readings = [StatusReading(id=f"r-{one}", slot=one, text=f"says-{one}") for one in slots]
        app.query_one(StatusBar).show(front.state, readings)
        app.query_one(Sidebar).show(front.state, readings, session_id=front.session_id, cwd="/w")

        drawn = _status_line(app) + str(app.query_one("#session-facts").render())
        missing = [one for one in slots if f"says-{one}" not in drawn]
        assert not missing, f"no widget draws the {missing} slot"


async def test_a_spinner_frame_moves_the_glyph_and_leaves_the_line_alone(
    make_tui_app: MakeApp,
) -> None:
    """A frame is one cell: the glyph spins, and the line beside it is not rebuilt.

    Sabotage: have `StatusBar.tick` call `show`, and the line is rebuilt (and
    the layout re-run) thirty times a second to move one glyph; or have it leave
    the glyph alone, and the spinner never turns.
    """
    async with running(make_tui_app()) as (app, pilot):
        front = app.front
        assert front is not None
        front.state.status = "running"
        bar = app.query_one(StatusBar)
        bar.show(front.state, [StatusReading(id="posture", text="read-only accepted")])
        await pilot.pause()
        line = bar.query_one("#status-line", Static)
        drawn = line.visual
        was = _status_line(app, "glyph")

        bar.tick()

        now = _status_line(app, "glyph")
        assert now == bar.glyph != was, "the glyph did not move"
        assert line.visual is drawn, "the line was rebuilt for a spinner frame"


async def test_the_status_line_is_the_prompts_top_edge(make_tui_app: MakeApp) -> None:
    """Between the conversation and the box: the glyph at the left, the text at
    the right, and a rule between them.

    The test checks positions and not a picture, so a theme or a font cannot
    pass it by accident:

    - The row ends where the prompt begins.
    - The glyph is the first cell of the row.
    - The text ends one column short of the row. That space keeps the text off
      the border of the side panel.
    - The rule runs from one column past the glyph to one column short of the
      text.

    Sabotage: yield `StatusBar` after `PromptInput` in `PHTuiApp.compose`, and the
    first assertion names the order.
    """
    async with running(make_tui_app(), size=(100, 24)) as (app, pilot):
        await pilot.pause(0.1)
        bar = app.query_one(StatusBar)
        glyph = bar.query_one("#status-glyph").region
        rule = bar.query_one("#status-rule").region
        line = bar.query_one("#status-line").region
        prompt = app.query_one(PromptInput).region

        assert bar.region.bottom == prompt.y, "the status line is not directly above the prompt"
        assert (glyph.x, glyph.width) == (bar.region.x, 1), "the glyph is not the first cell"
        assert line.right == bar.region.right - 1, "the text does not end one space short"
        assert rule.x == glyph.right + 1 and rule.right == line.x - 1, "the rule does not fill"
        assert rule.width > 0


def _ink(widget: Widget, text: str) -> object:
    """The foreground that a widget actually draws `text` in, from its first row.

    It is read from the rendered strip and not from `styles.color`, because most
    of the color of the status line is *inherited*. A span that named a color of
    its own would pass a check of the style of the widget and still draw in
    another color.
    """
    for segment in widget.render_line(0):
        if text in segment.text and segment.style is not None:
            return segment.style.color
    raise AssertionError(f"{text!r} is not on {widget!r}'s first row")


async def test_the_status_line_draws_in_the_headings_color(make_tui_app: MakeApp) -> None:
    """The rule and the text are in the color of the section titles of the side
    panel. That color is a theme role, so this holds for whichever theme is on.
    A warning is still in another color.

    Sabotage: give `_piece`, which `StatusBar.show` calls, back its `$ph-muted`
    span, and the reading is drawn in a different color from the heading.
    """
    async with running(make_tui_app()) as (app, pilot):
        front = app.front
        assert front is not None
        await pilot.pause(0.1)
        heading = _ink(app.query_one(Sidebar).query_one(".section-title"), "session")
        bar = app.query_one(StatusBar)
        bar.show(
            front.state,
            [
                StatusReading(id="posture", text="read-only accepted"),
                StatusReading(id="limits", text="budget 95%", level="warning"),
            ],
        )
        # One layout pass. The text sizes to its content, so until the pass
        # runs, the widget still has the width of the old line and crops the
        # end of the new line.
        await pilot.pause()
        line = bar.query_one("#status-line")
        assert "budget 95%" in _status_line(app), "the app redrew the line from the seam"

        assert _ink(line, "fake-1") == heading, "the model is not in the headings' color"
        assert _ink(line, "read-only accepted") == heading, "a reading is not either"
        assert _ink(bar.query_one("#status-rule"), "─") == heading, "nor is the rule"
        assert _ink(bar.query_one("#status-glyph"), "●") == heading, "nor is the glyph"
        assert _ink(line, "budget 95%") != heading, "a warning looks like everything else"


async def test_a_narrow_terminal_crops_the_status_line_rather_than_wrapping_it(
    make_tui_app: MakeApp,
) -> None:
    """Too narrow for the whole line: it is still one row, and the start is kept.

    The start is the glyph and the model, which are what a user glances for.
    The end is the posture, which the neighbors of the session panel can live
    without. A wrapped line would grow the bar to two rows and push the prompt
    down. That is the space that this layout exists to save.

    **What is visible is the slice of the line that the bar shows**, not
    `render_line`. `render_line` draws the whole of a widget, however much of it
    its container shows. The line keeps its own width and runs past the bar, and
    the bar clips it.
    """
    # The sidebar keeps its 32 columns, so this leaves the line about 27.
    async with running(make_tui_app(), size=(60, 24)) as (app, pilot):
        await pilot.pause(0.1)
        bar = app.query_one(StatusBar)
        line = bar.query_one("#status-line")

        assert bar.region.height == 1
        assert line.region.height == 1
        assert line.region.right > bar.region.right, "the line fit; the test is too wide"
        shown = line.render_line(0).text[: bar.content_region.right - line.region.x]
        assert shown.startswith("fake-1")
        assert "accepted" not in shown, "the line was not cropped"


# ------------------------------------------------------------------- resume --


async def test_resuming_rebuilds_the_transcript_from_the_log(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """The P2-01 gate at the level of the app: a stored session comes back
    readable.

    The adapter test proves that live and replay agree. This test proves that
    the app actually takes that path: `--resume` reaches `session.events` and
    puts the conversation on screen before the user types anything.
    """
    async with running(make_tui_app()) as (app, pilot):
        assert app.front is not None
        await pilot.press(*"remember this")
        await pilot.press(app.keys.submit)
        await until(pilot, turn_done(app))
        before = [(item.role, item.text) for item in app.front.state.visible_items()]

    # **Released before reopening**, which is what makes this a resume at all.
    # The daemon keeps the root, so a second attach would otherwise find it
    # still there. That is the P5-01 promise, and the opposite of what this
    # test is about. After passivation, the next attach mounts from the log and
    # the harness appends `session/resumed`, which is the notice asserted below.
    #
    # The wait is for the *subscription* to go. A watcher is its own claim on
    # the life of a root, and the socket of the closed terminal is torn down a
    # moment after the `async with` above returns.
    root = root_of(tui_daemon)
    await daemon_until(lambda: not root.subscribers, what="the first terminal to detach")
    await tui_daemon.sweep()
    assert not tui_daemon.holds("pilot")

    async with running(make_tui_app(session_id="pilot")) as (resumed, _pilot):
        assert resumed.front is not None
        after = [(item.role, item.text) for item in resumed.front.state.visible_items()]

    # The conversation comes back whole, plus two lines that account for the
    # gap: the release and the resume. A user who did not know that a previous
    # run existed should not have to infer either from the scrollback. The
    # *released* notice is new here, because before P5-14 nothing except this
    # terminal could have let the session go.
    assert [item for item in after if item[0] != "notice"] == before
    assert sum("Resumed" in text for role, text in after if role == "notice") == 1
    assert sum("Released" in text for role, text in after if role == "notice") == 1
    assert ("user", "remember this") in after


async def _verbs_done(app: Any) -> None:  # noqa: ANN401
    """Wait for the verb workers, and only those.

    A verb that carries `@work` schedules and returns (H7), so the work has not
    happened when the call returns. The test waits for the work and does not
    pause past it. A `pause` that happens to be long enough is a test that
    passes on timing, and both attach tests did.

    `wait_for_complete()` with no argument waits for *every* worker. This app
    keeps long-lived workers, such as the daemon watch and the pump, so the bare
    call hangs forever and does not fail. **An empty list does the same.** The
    body is `workers or self`, so a filter that matches nothing falls back to
    all of them. That is why this returns early and does not pass the list
    straight through.

    `startswith`, because the group is per verb (`verb:attach`). One shared
    group would make `exclusive` cancel across verbs.
    """
    waiting = [one for one in app.workers if one.group.startswith(VERB_GROUP)]
    if not waiting:
        return
    await app.workers.wait_for_complete(waiting)


async def test_a_typed_attach_reaches_the_verb_with_its_argument(
    make_tui_app: MakeApp, tmp_path: Path
) -> None:
    """`/attach <path>` goes through the one route that every verb takes, with its
    argument.

    The verb used to be caught by name ahead of the command table, which is the
    string re-dispatch that `commands.py` forbids. The reason was that
    `_RunAction` dropped what followed the name. `_RunAction` forwards it now,
    so the argument arrives at `action_attach` through `run_line`, like the body
    of any other verb. This test checks that it does: the file ends up staged.
    """
    picture = tmp_path / "diagram.png"
    picture.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    async with running(make_tui_app()) as (app, pilot):
        front = app.front
        assert front is not None

        await app._dispatch_command(f"/attach {picture}")
        await _verbs_done(app)
        await pilot.pause()

        staged = front._staged.refs  # type: ignore[attr-defined]
        assert [one.name for one in staged] == ["diagram.png"]


async def test_a_quoted_attach_path_is_one_path(make_tui_app: MakeApp, tmp_path: Path) -> None:
    """H7: `argument.split()` on a path that a user quoted.

    A space in a path is the ordinary case on a Mac, as in `~/Library/Application
    Support/...`. Splitting on whitespace turned one `/attach "my notes.md"`
    into two filenames that do not exist. They were reported as two failures
    that named neither of the things that the user typed. Quoting is what a user
    reaches for, and it is what the shell that they typed it into would have
    honored.
    """
    picture = tmp_path / "my diagram.png"
    picture.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    async with running(make_tui_app()) as (app, pilot):
        front = app.front
        assert front is not None

        await app._dispatch_command(f'/attach "{picture}"')
        await _verbs_done(app)
        await pilot.pause()

        staged = front._staged.refs  # type: ignore[attr-defined]
        assert [one.name for one in staged] == ["my diagram.png"]


async def test_the_trajectory_viewer_keeps_its_own_way_out() -> None:
    """H7: `q` was bound under the id `quit`, which the keymap then remaps.

    `set_keymap` rebinds by binding id, and the ids are a namespace that all
    apps share. The `q` of the trajectory viewer and the quit of the chat app
    are not the same binding, but they had the same id. The `quit: ctrl+d` of
    `tui.json` therefore moved the only documented way out of the viewer onto a
    key that its footer does not mention.

    The assertion is on the binding table and not through a pilot, because what
    went wrong is the *identity*. A pilot that pressed `q` would pass against a
    build where the default keymap happened to leave the key alone.
    """
    from ph_app.tui.config import TuiKeybindings
    from ph_app.tui.trajectory_app import TrajectoryApp

    # `BINDINGS` is declared as a union that admits bare tuples. Every entry
    # here is a `Binding`, and the narrowing says so without an assertion.
    declared = [one for one in TrajectoryApp.BINDINGS if isinstance(one, Binding)]
    keys = {one.id: one.key for one in declared}
    remapped = set(TuiKeybindings().as_map())

    assert keys.get("trajectory_quit") == "q", "`q` is bound under an id the keymap owns"
    # The configured key is still offered, so remapping `quit` is honored too.
    assert "quit" in keys and "quit" in remapped


# ------------------------------------------------- the harness is elsewhere --


async def test_a_turn_started_in_the_tui_finishes_after_the_tui_is_gone(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """**The gate that this whole plan is named for** (P5-01, P5-14).

    A user submits a prompt and closes the terminal, and the work continues,
    because the harness is no longer in the terminal. Then a second terminal
    opens the same session and finds the finished turn in the transcript that it
    rebuilds from the log.

    The test drives `PHTuiApp` and not `DaemonSession`, which is the difference
    from the version in `test_tui_remote.py`. The assertion here is that the
    *app* takes that path: closing the app detaches and ends nothing.

    Sabotage: have the app's `on_unmount` shut the daemon down, or have
    `DaemonSession.close` cancel the turn, and the second terminal finds no
    assistant message.
    """
    async with running(make_tui_app()) as (app, pilot):
        assert app.front is not None
        await pilot.press(*"keep going")
        # Submitted and then left. **Not** waited on. The fake provider answers
        # in microseconds, so to wait for `busy` is a race that a test cannot
        # win, and who wins it is not what is tested. The test is that the root
        # and its work are not this terminal's to end.
        await pilot.press(app.keys.submit)
        await pilot.pause()

    assert tui_daemon.holds("pilot"), "closing a terminal is not a shutdown"
    root = root_of(tui_daemon)
    await daemon_until(
        lambda: root.status == "idle", what="the turn to finish with nobody watching"
    )

    async with running(make_tui_app()) as (second, _pilot):
        assert second.front is not None
        rows = [(item.role, item.text) for item in second.front.state.visible_items()]

    assert ("user", "keep going") in rows
    assert any(role == "assistant" for role, _text in rows), "the turn finished, and it is readable"


async def test_two_terminals_on_one_session_share_the_log_and_not_the_composer(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """The multiplex rule, through two real apps on one root.

    Text that is not submitted never leaves the client, because it is not an
    act in the session. Pressing enter is an act, so it reaches the other
    terminal by the one route that everything else takes. That asymmetry is the
    whole decision that replaced the takeover/demotion design.
    """
    async with (
        running(make_tui_app()) as (first, first_pilot),
        running(make_tui_app()) as (second, second_pilot),
    ):
        assert first.front is not None and second.front is not None

        # Typed and *not* sent: the other terminal must not see it.
        await second_pilot.press(*"a draft")
        await second_pilot.pause()
        assert "a draft" not in first.query_one(PromptInput).area.text

        await first_pilot.press(*"shared")
        await first_pilot.press(first.keys.submit)
        await until(first_pilot, turn_done(first))
        await until(
            second_pilot,
            lambda: any(
                item.text == "shared"
                for item in not_none(not_none(second.front).state).visible_items()
            ),
        )

        assert second.query_one(PromptInput).area.text == "a draft", "the draft is still theirs"


# ------------------------------------------------------------------ history --


def _sent(app: PHTuiApp, *prompts: str) -> None:
    """Put `prompts` in the mirror as sent, oldest first.

    The rows are seeded and not submitted. History is a fold of the `user` rows,
    and a turn for each entry would spend a model call to produce the one row
    that this reads. The keystrokes below then drive the whole real path: the
    source of the app, the walk of the widget, and the box.
    """
    front = not_none(app.front)
    for index, text in enumerate(prompts):
        front.state.add(
            ChatItem(key=f"u{index}", role="user", text=text, seq=index, turn=index + 1)
        )


async def test_up_arrow_recalls_the_previous_prompt(make_tui_app: MakeApp) -> None:
    async with running(make_tui_app()) as (app, pilot):
        _sent(app, "the older one", "the newer one")
        prompt = not_none(app._prompt)
        await pilot.press(app.keys.history_previous)
        await pilot.pause()
        assert prompt.area.text == "the newer one", "newest first"
        await pilot.press(app.keys.history_previous)
        await pilot.pause()
        assert prompt.area.text == "the older one"
        # And it stops at the oldest rather than emptying the box.
        await pilot.press(app.keys.history_previous)
        await pilot.pause()
        assert prompt.area.text == "the older one"


async def test_up_arrow_inside_a_multi_line_draft_still_moves_the_cursor(
    make_tui_app: MakeApp,
) -> None:
    """The behavior that a prompt box may not give up to gain history.

    Sabotage: claim `history_previous` without asking `cursor_at_first_line`, and
    the draft is replaced instead of navigated.
    """
    async with running(make_tui_app()) as (app, pilot):
        # The sent prompt **starts with** the draft below, so that a walk would
        # have something to recall. Without that, the filter alone stops the
        # recall, and this test passes even with the cursor check removed. It
        # did.
        _sent(app, "first line\nsecond line and then more")
        prompt = not_none(app._prompt)
        # Set and not typed: `enter` is submit here, so a newline cannot be
        # keyed into the box at all.
        prompt.area.text = "first line\nsecond line"
        prompt.area.move_cursor(prompt.area.document.end)
        await pilot.pause()

        await pilot.press(app.keys.history_previous)
        await pilot.pause()
        assert prompt.area.text == "first line\nsecond line", "the draft is untouched"
        assert prompt.area.cursor_location[0] == 0, "and the cursor moved up a line"

        # What the *next* press does is deliberately not asserted here. The
        # draft is the filter, so a two-line draft is a prefix of nothing and
        # recalls nothing. Recall from the first line is the subject of
        # `test_up_arrow_recalls_the_previous_prompt`. This test is about the
        # key that the editor keeps.


async def test_the_draft_survives_a_walk_through_history(make_tui_app: MakeApp) -> None:
    async with running(make_tui_app()) as (app, pilot):
        _sent(app, "half a thought, finished")
        prompt = not_none(app._prompt)
        await pilot.press(*"half")
        await pilot.pause()
        assert prompt.area.text == "half"

        await pilot.press(app.keys.history_previous)
        await pilot.pause()
        assert prompt.area.text == "half a thought, finished"

        await pilot.press(app.keys.history_next)
        await pilot.pause()
        assert prompt.area.text == "half", "the half-written prompt was not eaten"


async def test_a_typed_prefix_filters_the_walk(make_tui_app: MakeApp) -> None:
    """What makes it a search rather than a cycle."""
    async with running(make_tui_app()) as (app, pilot):
        _sent(app, "fix the parser", "run the tests")
        prompt = not_none(app._prompt)
        await pilot.press(*"fix")
        await pilot.pause()
        await pilot.press(app.keys.history_previous)
        await pilot.pause()
        # "run the tests" is newer and would win an unfiltered walk.
        assert prompt.area.text == "fix the parser"


async def test_history_search_reveals_the_chosen_prompt_in_the_transcript(
    make_tui_app: MakeApp,
) -> None:
    """The second half of the request: the conversation moves behind the picker.

    Sabotage: drop `on_highlight` from `action_open_history` and nothing is
    revealed as the cursor moves.
    """
    async with running(make_tui_app()) as (app, pilot):
        _sent(app, "the first thing", "the second thing")
        revealed: list[int] = []
        view = not_none(app._view)
        original = view.scroll_to_seq

        def recording(seq: int) -> bool:
            revealed.append(seq)
            return original(seq)

        view.scroll_to_seq = recording  # type: ignore[method-assign]

        await pilot.press(app.keys.history_search)
        await pilot.pause()
        assert isinstance(app.screen, ChoicePicker)
        assert revealed, "the row under the cursor is revealed as the picker opens"
        await pilot.press("down")
        await pilot.pause()
        assert len(revealed) > 1, "and again as the cursor moves"

        await pilot.press("enter")
        await pilot.pause()
        assert not_none(app._prompt).area.text in ("the first thing", "the second thing")


async def test_a_diverged_client_says_so_rather_than_offering_a_short_history(
    make_tui_app: MakeApp,
) -> None:
    """A mirror that missed a frame is a prefix of unknown length.

    The key declines quietly, because a keystroke has nowhere to say why. The
    verb says it in a notification. `action_open_screen` already makes the same
    split.
    """
    async with running(make_tui_app()) as (app, pilot):
        _sent(app, "sent before the gap")
        front = not_none(app.front)
        front._unreadable = 1  # type: ignore[attr-defined]
        assert front.diverged

        await pilot.press(app.keys.history_previous)
        await pilot.pause()
        assert not_none(app._prompt).area.text == "", "the arrows fall back to editing"

        await pilot.press(app.keys.history_search)
        await pilot.pause()
        assert not isinstance(app.screen, ChoicePicker), "and the picker refuses"


async def test_a_rebound_history_key_is_the_one_that_works(
    make_tui_app: MakeApp, tmp_path: Path
) -> None:
    """The rule that `prompt.py` exists to enforce, asserted and not assumed."""
    (tmp_path / "tui.json").write_text(
        json.dumps({"keybindings": {"history_previous": "ctrl+up"}}), encoding="utf-8"
    )
    async with running(make_tui_app()) as (app, pilot):
        assert app.keys.history_previous == "ctrl+up"
        _sent(app, "reachable only by the rebound key")
        prompt = not_none(app._prompt)

        await pilot.press("up")
        await pilot.pause()
        assert prompt.area.text == "", "the default key no longer recalls"

        await pilot.press("ctrl+up")
        await pilot.pause()
        assert prompt.area.text == "reachable only by the rebound key"


# ------------------------------------------------------- the startup picker --


def _stored(home: Path, session_id: str, cwd: str) -> None:
    """A session on the disk of the daemon, in `cwd`.

    **A header and no events.** That is the shortest valid log, and the only one
    worth writing by hand. The envelope invariants are real: an event needs
    `seq` and `time`, a log must start at seq 0, and a surface-eligible event
    needs its `surfaceOp` marker. A hand-written transcript would therefore
    compete with rules that exist to keep logs readable. A header-only session
    lists and resumes, which is all that these tests ask of it. The picker
    labels it by id, because it has no first prompt to quote.
    """
    # Through `SessionHeader` and not hand-written JSON. `createdAt` is
    # camelCase that the model already owns. Since format 1, the model also owns
    # the `<cwd-tag>-<id>` family that decides which *directory* the log lands
    # in. A fixture that built that name itself would derive the layout again.
    header = SessionHeader(id=session_id, created_at=0, cwd=cwd)
    path = session_path(home / "sessions", session_id, header.family)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": HEADER_LINE_TYPE, "header": header.to_wire()}) + "\n",
        encoding="utf-8",
    )


@asynccontextmanager
async def _starting(app: PHTuiApp) -> AsyncIterator[tuple[PHTuiApp, Pilot[str | None]]]:
    """Drive an app through startup **without** waiting for an attach.

    `running` waits for `app.front`, and the startup picker deliberately stands
    in front of that. A test about the picker therefore cannot use `running`.
    """
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        yield app, pilot


async def test_a_directory_with_no_history_opens_straight_into_a_new_session(
    make_tui_app: MakeApp, tmp_path: Path
) -> None:
    """A first run must not meet a modal that lists nothing.

    Sabotage: push the picker without checking that `sessions` is non-empty.
    """
    app = make_tui_app(session_id=None, project=tmp_path / "fresh")
    (tmp_path / "fresh").mkdir()
    async with running(app) as (app, _pilot):
        assert not isinstance(app.screen, ChoicePicker)
        assert not_none(app.front).session_id, "attached without being asked"


async def test_enter_on_the_first_row_starts_a_new_session(
    make_tui_app: MakeApp, tmp_path: Path
) -> None:
    """One keystroke gives the behavior that a user had before this picker existed."""
    project = tmp_path / "work"
    project.mkdir()
    _stored(tmp_path, "older", str(project))
    app = make_tui_app(session_id=None, project=project)
    async with _starting(app) as (app, pilot):
        await until(pilot, lambda: isinstance(app.screen, ChoicePicker))
        await pilot.press("enter")
        await until(pilot, lambda: app.front is not None)
        assert not_none(app.front).session_id != "older", "a fresh session, not the stored one"


async def test_choosing_a_stored_session_attaches_to_it(
    make_tui_app: MakeApp, tmp_path: Path
) -> None:
    project = tmp_path / "work"
    project.mkdir()
    _stored(tmp_path, "older", str(project))
    app = make_tui_app(session_id=None, project=project)
    async with _starting(app) as (app, pilot):
        await until(pilot, lambda: isinstance(app.screen, ChoicePicker))
        await pilot.press("down")
        await pilot.press("enter")
        await until(pilot, lambda: app.front is not None)
        assert not_none(app.front).session_id == "older"


async def test_the_picker_lists_only_this_directory(make_tui_app: MakeApp, tmp_path: Path) -> None:
    """The filter of P9-04, seen from the screen that it was added for."""
    project = tmp_path / "work"
    project.mkdir()
    _stored(tmp_path, "ours", str(project))
    _stored(tmp_path, "theirs", str(tmp_path / "elsewhere"))
    app = make_tui_app(session_id=None, project=project)
    async with _starting(app) as (app, pilot):
        await until(pilot, lambda: isinstance(app.screen, ChoicePicker))
        picker = app.screen
        assert isinstance(picker, ChoicePicker)
        values = [choice.value for choice in picker.choices]
        assert values == [NEW_SESSION, "ours"], "the other directory's session is not offered"


async def test_new_skips_the_picker(make_tui_app: MakeApp, tmp_path: Path) -> None:
    """`--new` is for the user who never wants to be asked."""
    project = tmp_path / "work"
    project.mkdir()
    _stored(tmp_path, "older", str(project))
    app = make_tui_app(session_id=None, project=project, offer_sessions=False)
    async with running(app) as (app, _pilot):
        assert not isinstance(app.screen, ChoicePicker)
        assert not_none(app.front).session_id != "older"


async def test_escape_starts_a_new_session_rather_than_exiting(
    make_tui_app: MakeApp, tmp_path: Path
) -> None:
    """The failure that this test guards against: a picker that quits the app on
    escape is a worse first run than no picker at all."""
    project = tmp_path / "work"
    project.mkdir()
    _stored(tmp_path, "older", str(project))
    app = make_tui_app(session_id=None, project=project)
    async with _starting(app) as (app, pilot):
        await until(pilot, lambda: isinstance(app.screen, ChoicePicker))
        await pilot.press("escape")
        await until(pilot, lambda: app.front is not None)
        assert app.is_running, "still here"
        assert not_none(app.front).session_id != "older"


@pytest.mark.parametrize("code", [1, 0])
async def test_a_terminal_answers_its_exit_code_without_exiting_the_process(
    monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """M4: `run_tui` returns the exit code and does not raise `SystemExit`.

    Raising it made this the only exit-code mechanism of the CLI, because
    nothing else in `cli.py` calls `sys.exit`. The next command that needed an
    exit code would have invented a second mechanism, and a caller that wanted
    to know what happened had to catch `SystemExit` to find out. `cli.py` now
    raises the `typer.Exit` that it owns.

    The property of H3 is the reason why the value matters, and it is asserted
    both ways. A terminal that *crashed* must be distinguishable from one that
    the user quit. Otherwise a supervisor that scripts `phern --mode tui` cannot
    tell whether to restart it.

    Sabotage: `raise SystemExit(app.return_code)` and the call never returns.
    """

    class _App:
        return_code = code

        async def run_async(self) -> None:
            return None

    monkeypatch.setattr("ph_app.tui.app.PHTuiApp", lambda **_kwargs: _App())

    assert await run_tui(daemon_argv=("phern", "daemon")) == code


@pytest.mark.parametrize("code", [1, 0])
async def test_the_trajectory_viewer_answers_its_exit_code_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, code: int
) -> None:
    """The same hole, one branch up in the same function (M4).

    `--mode trajectory` is the other Textual app of the CLI, and its
    `return_code` was discarded. A viewer that crashed therefore exited 0. The
    property that had just been restored for `run_tui` was still broken beside
    it.

    Sabotage: drop `return app.return_code or 0` and a crash reads as success.
    """

    class _App:
        return_code = code

        async def run_async(self) -> None:
            return None

    monkeypatch.setattr("ph_app.tui.trajectory_app.TrajectoryApp", lambda *_a, **_k: _App())
    monkeypatch.setattr("ph_app.tui.trajectory_app.load_records", lambda *_a, **_k: ("s1", []))

    assert await run_trajectory("s1", home=tmp_path) == code
