"""Pilot tests: every modal driven by keystrokes, as a person would.

Phase 2's definition of done is "pilot tests drive every modal", and the reason
is narrower than coverage. A TUI's failure modes are not wrong values, they are
*wrong plumbing*: a key that reaches nothing because the binding was compared
literally, a modal awaited on the message pump so the app deadlocks, a picker
whose callback never fires. None of that shows up in a unit test of the widget.

The approval test is the one to read: it asserts the round trip through the
**log** — `approval/asked` then `approval/decided` — rather than the modal's
return value, because pH's claim is that the decision is recorded, not that a
dialog appeared.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from daemon_helpers import until as settled
from textual.widgets import Input
from tui_helpers import root_of, running, turn_done, until

from ph.seams.approval import ApprovalRequest, Edited, Responded
from ph.seams.commands import CommandDefinition
from ph.seams.user_questions import UserQuestion
from ph.testing import StubAgent, simple_tool
from ph_app.trust import TrustStore
from ph_app.tui.app import PHTuiApp
from ph_app.tui.modals.approval import ApprovalModal
from ph_app.tui.modals.ask_user import AskUserModal
from ph_app.tui.modals.base import Choice, ChoicePicker, ConfirmModal
from ph_app.tui.modals.trust import plan_review_modal, project_trust_modal
from ph_app.tui.widgets.prompt import PromptInput

pytestmark = pytest.mark.anyio

MakeApp = Callable[..., PHTuiApp]


def _command(name: str = "compact") -> CommandDefinition:
    return CommandDefinition(name=name, summary="Compact the context", run=lambda *_: None)


# ------------------------------------------------------------------- prompt --


async def test_typing_and_submitting_runs_a_turn(make_tui_app: MakeApp, tui_daemon: Any) -> None:
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        root.ctx.tools.register(simple_tool("ping"))
        await pilot.press(*"hello")
        await pilot.press(app.keys.submit)
        await until(pilot, turn_done(app))
        roles = [(item.role, item.text) for item in app.front.state.items]
        assert ("user", "hello") in roles
        assert any(role == "assistant" for role, _ in roles)


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


async def test_a_rebound_key_is_honoured(make_tui_app: MakeApp, tmp_path: Path) -> None:
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
    """`ctrl+k` is the palette *and* TextArea's delete-to-end-of-line.

    Priority bindings mean the app claims it first; before them the key did
    both, silently.
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


async def _decide(app: Any, pilot: Any, root: Any, *, arguments: Any = None) -> list[Any]:
    """Put one approval on screen and hand back the list the answer lands in.

    Takes the daemon-side `root`, not the front end: the approval is raised where
    the harness is, and the modal appears where the person is — which after P5-14
    is the two ends of a socket. That separation is the thing under test.
    """
    answers: list[Any] = []

    async def ask() -> None:
        answers.append(
            await root.ctx.approval.request(
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


async def test_the_modal_answers_in_the_tools_voice(make_tui_app: MakeApp, tui_daemon: Any) -> None:
    """`respond` (P4-05): the body never runs and the model reads an answer.

    A person who knows the answer should not have to reject a call and then
    explain — the round trip is the cost this decision removes.
    """
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        answers = await _decide(app, pilot, root)

        await pilot.click("#approval-respond")
        await pilot.pause()
        # Enter in the box, which is the gesture a person actually makes after
        # typing — the button stays a second route for the mouse.
        app.screen.query_one("#approval-why", Input).value = "the port is 8080"
        await pilot.press("enter")
        await until(pilot, lambda: bool(answers))

        assert isinstance(answers[0], Responded)
        assert answers[0].message == "the port is 8080"
        decided = next(e for e in root.session.events if e.type == "approval/decided")
        assert decided.data["outcome"] == "responded"


async def test_the_modal_corrects_the_call_rather_than_refusing_it(
    make_tui_app: MakeApp,
    tui_daemon: Any,
) -> None:
    """`edit` (P4-05), opening on the call as it stands.

    Prefilled because a person correcting one wrong path should not retype the
    whole argument object — and because what they are editing is the model's
    own request, which they need to see.
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
        assert decided.data["arguments"]["path"] == "notes.md"
        # The ask itself does not carry them: `tool/call` already did, and two
        # copies of one fact in the log are two that can disagree.
        asked = next(e for e in root.session.events if e.type == "approval/asked")
        assert "arguments" not in asked.data


async def test_a_mistyped_edit_keeps_the_box_open(make_tui_app: MakeApp, tui_daemon: Any) -> None:
    """Not a refusal: the person meant to edit and mistyped, and rejecting the
    call on a stray comma would be the harness deciding for them."""
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


async def test_approval_round_trips_through_the_log(make_tui_app: MakeApp, tui_daemon: Any) -> None:
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
        # First press reveals the reason field rather than deciding.
        assert app.screen.has_class("-explaining")
        await pilot.press(*"use the helper")
        await pilot.press("enter")
        await until(pilot, lambda: bool(answers))
        assert answers == [("rejected", "use the helper")]


async def test_escape_denies_rather_than_leaving_it_open(make_tui_app: MakeApp) -> None:
    """Absence is not consent — dismissing the modal is a refusal."""
    async with running(make_tui_app()) as (app, pilot):
        answers: list[Any] = []

        async def ask() -> None:
            answers.append(await app.ask_approval(ApprovalRequest(tool_name="rm")))

        app.run_worker(ask())
        await until(pilot, lambda: isinstance(app.screen, ApprovalModal))
        await pilot.press(app.keys.cancel)
        await until(pilot, lambda: bool(answers))
        assert answers[0][0] == "rejected"


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
    make_tui_app: MakeApp, tui_daemon: Any
) -> None:
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        front = app.front
        assert front is not None
        root.ctx.commands.register(_command())
        # A command registered on the daemon reaches this palette as a
        # `session.commands` notification, so it is there a frame later rather
        # than at once — the honest cost of the list living one socket away.
        await until(pilot, lambda: any(one.name == "compact" for one in front.commands()))
        await pilot.press(app.keys.command_palette)
        await pilot.pause()
        assert isinstance(app.screen, ChoicePicker)
        await pilot.press(*"compact")
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(PromptInput).area.text.startswith("/compact")


async def test_a_picker_blocks_the_other_global_keys(make_tui_app: MakeApp) -> None:
    """With one picker open, another picker's key must not stack a second on top."""
    async with running(make_tui_app()) as (app, pilot):
        await pilot.press(app.keys.theme_picker)
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, ChoicePicker)
        await pilot.press(app.keys.model_picker)
        await pilot.pause()
        assert app.screen is picker
        assert len(app.screen_stack) == 2


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
        # And remembered: the next launch opens in the chosen theme.
        assert json.loads((tmp_path / "tui.json").read_text())["theme"] == app.theme


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
    make_tui_app: MakeApp, tui_daemon: Any
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
        assert front.state.preset == presets[-1].data["preset"]


async def test_the_session_picker_lists_and_reopens(make_tui_app: MakeApp) -> None:
    """Choosing a stored session exits with its id; `run_tui` reopens it."""
    async with running(make_tui_app(session_id="earlier")) as (app, pilot):
        assert app.front is not None
        await pilot.press(*"remember this")
        await pilot.press(app.keys.submit)
        await until(pilot, turn_done(app))
        await app.front.flush()

    app = make_tui_app(session_id="later")
    async with running(app) as (app, pilot):
        assert app.front is not None
        # The store buffers until a flush, so the current session reaches the
        # directory the picker lists only after one.
        await app.front.flush()
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
    # Nothing was written into the project — a repo cannot vouch for itself.
    assert list(project.iterdir()) == []


async def test_the_trust_modal_offers_three_answers(make_tui_app: MakeApp, tmp_path: Path) -> None:
    async with running(make_tui_app()) as (app, pilot):
        chosen: list[Any] = []
        app.push_screen(project_trust_modal(tmp_path), chosen.append)
        await pilot.pause()
        await pilot.click("#act-once")
        await pilot.pause()
        assert chosen == ["once"]


async def test_plan_review_defaults_to_rejecting_on_dismissal(make_tui_app: MakeApp) -> None:
    async with running(make_tui_app()) as (app, pilot):
        chosen: list[Any] = []
        app.push_screen(plan_review_modal("1. do the thing"), chosen.append)
        await pilot.pause()
        await pilot.press(app.keys.cancel)
        await pilot.pause()
        assert chosen == ["reject"]


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


# -------------------------------------------------------------- completions --


async def test_typing_a_slash_offers_registered_commands(
    make_tui_app: MakeApp, tui_daemon: Any
) -> None:
    """Completion sources are pH's registries, not a second list (I7)."""
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        root.ctx.commands.register(_command())
        prompt = app.query_one(PromptInput)
        await pilot.press(*"/comp")
        await pilot.pause()
        assert prompt.has_class("-completing")
        await pilot.press(app.keys.accept_completion)
        await pilot.pause()
        assert prompt.area.text == "/compact "
        assert not prompt.has_class("-completing")


async def test_a_disposed_command_leaves_the_completion_list(
    make_tui_app: MakeApp, tui_daemon: Any
) -> None:
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        dispose = root.ctx.commands.register(_command())
        prompt = app.query_one(PromptInput)
        await pilot.press(*"/comp")
        await pilot.pause()
        assert prompt.has_class("-completing")
        prompt.clear()
        dispose()
        await pilot.press(*"/comp")
        await pilot.pause()
        # The registry is the source of truth, so unregistering removes the row.
        assert not prompt.has_class("-completing")


async def test_escape_closes_the_completion_list_before_interrupting(
    make_tui_app: MakeApp, tui_daemon: Any
) -> None:
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        root.ctx.commands.register(_command())
        prompt = app.query_one(PromptInput)
        await pilot.press(*"/comp")
        await pilot.pause()
        await pilot.press(app.keys.cancel)
        await pilot.pause()
        assert not prompt.has_class("-completing")
        # And the typed text survives — escape closed the popup, not the prompt.
        assert prompt.area.text == "/comp"


# ----------------------------------------------------------------- commands --


async def test_a_slash_line_dispatches_instead_of_prompting(
    make_tui_app: MakeApp, tui_daemon: Any
) -> None:
    """A command spends no model turn, and the log says who decided.

    A **harness** verb, deliberately. A client-side one — `/theme`, `/model` —
    changes this terminal's own display and is not an act in the session, so it
    is dispatched here and appears in no log: another attached UI rendering
    "somebody opened a theme picker" would be reporting a fact about a screen it
    cannot see. What the log records is what the *session* did, and that is what
    this asserts.
    """
    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        front = app.front
        assert front is not None
        root.ctx.commands.register(_command())
        await until(pilot, lambda: any(one.name == "compact" for one in front.commands()))
        await pilot.press(*"/compact")
        await pilot.pause()
        # The completion popup claims enter first; tab accepts, then submit.
        await pilot.press(app.keys.accept_completion)
        await pilot.press(app.keys.submit)
        await until(pilot, lambda: "command/run" in [e.type for e in root.session.events])

        types = [event.type for event in root.session.events]
        assert "command/done" in types
        assert not any(kind.startswith("turn/") for kind in types), "a command is not a turn"


async def test_an_unknown_command_is_reported_not_prompted(
    make_tui_app: MakeApp, tui_daemon: Any
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


async def test_every_verb_is_a_command_an_action_and_maybe_a_key(
    make_tui_app: MakeApp,
) -> None:
    """One table, three routes: a verb missing any of them is a half-wired verb."""
    from ph_app.tui.commands import TUI_VERBS

    async with running(make_tui_app()) as (app, _pilot):
        front = app.front
        assert front is not None
        # `front.commands()`, not the daemon's registry: after P5-14 a verb that
        # changes *this* client's display is built client-side and never
        # registered anywhere, so the list a person sees is the merge of both
        # ends — which is the thing "reachable as a command" has to mean now.
        registered = {definition.name for definition in front.commands()}
        bound = {binding.id for binding in app.BINDINGS if hasattr(binding, "id")}
        for verb in TUI_VERBS:
            assert verb.name in registered
            assert hasattr(app, f"action_{verb.action}"), verb.action
            if verb.key is not None:
                assert verb.key in bound
                assert hasattr(app.keys, verb.key), verb.key


async def test_login_stores_a_secret_without_logging_it(
    make_tui_app: MakeApp, tui_daemon: Any
) -> None:
    from ph_app.tui.modals.login import LoginModal

    async with running(make_tui_app()) as (app, pilot):
        root = root_of(tui_daemon)
        assert app.front is not None
        await app.run_action("open_login")
        await pilot.pause()
        # No adapter row declares a key in the headless profile, so the name is
        # typed — which is the free-text path the picker offers.
        await pilot.press(*"PH_TEST_KEY")
        await pilot.press(app.keys.submit)
        await until(pilot, lambda: isinstance(app.screen, LoginModal))
        await pilot.press(*"s3cret")
        await pilot.press("enter")
        await pilot.pause()

        credentials = root.ctx.credentials
        resolved = credentials.resolve(credentials.reference("PH_TEST_KEY"))
        assert resolved is not None
        assert resolved.reveal() == "s3cret"
        # The secret is in the process and nowhere else.
        assert "s3cret" not in repr([event.data for event in root.session.events])
        assert "s3cret" not in repr(resolved)

        # **And the picker knows it is set now.** `credential_held` is a
        # synchronous member answered from the last `credentials/held` reply, so
        # something has to have asked — which is why `action_open_login` is
        # async. Without that the remote front end answered `False` for every
        # credential and a deployment with all of them set looked like one with
        # none: no test noticed, because nothing asserted the marker.
        assert app.front is not None
        await app.front.refresh_credentials(["PH_TEST_KEY"])
        assert app.front.credential_held("PH_TEST_KEY"), "the front end cannot see its own store"


# ------------------------------------------------------------------- status --


async def test_the_context_gauge_warns_from_the_compaction_threshold(
    make_tui_app: MakeApp,
) -> None:
    """The number a user needs to see coming is the one where pH will act."""
    from ph_app.tui.widgets.status import COMPACTION_THRESHOLD, StatusBar

    async with running(make_tui_app()) as (app, pilot):
        front = app.front
        assert front is not None
        front.state.context_window = 1_000
        front.state.tokens = int(1_000 * COMPACTION_THRESHOLD) + 10
        app.state_changed()
        await pilot.pause(0.1)
        rendered = str(app.query_one(StatusBar).query_one("#status-line").render())
        assert "context 86%" in rendered
        assert "fake-1" in rendered


# ------------------------------------------------------------------- resume --


async def test_resuming_rebuilds_the_transcript_from_the_log(
    make_tui_app: MakeApp, tui_daemon: Any
) -> None:
    """The P2-01 gate at the app level: a stored session comes back readable.

    The adapter test proves live and replay agree. This proves the app actually
    takes that path — that `--resume` reaches `session.events` and puts the
    conversation on screen before the person types anything.
    """
    async with running(make_tui_app()) as (app, pilot):
        assert app.front is not None
        await pilot.press(*"remember this")
        await pilot.press(app.keys.submit)
        await until(pilot, turn_done(app))
        await app.front.flush()
        before = [(item.role, item.text) for item in app.front.state.visible_items()]

    # **Released before reopening**, which is what makes this a resume at all.
    # The daemon keeps the root, so attaching again would otherwise just find it
    # still there — the P5-01 promise, and the opposite of what this test is
    # about. Passivated, the next attach mounts from the log and the harness
    # appends `session/resumed`, which is the notice asserted below.
    #
    # The wait is for the *subscription* to go: a watcher is its own claim on a
    # root's life, and the closed terminal's socket tears down a moment after
    # the `async with` above returns.
    root = root_of(tui_daemon)
    await settled(lambda: not root.subscribers, what="the first terminal to detach")
    await tui_daemon.sweep()
    assert not tui_daemon.holds("pilot")

    async with running(make_tui_app(session_id="pilot")) as (resumed, _pilot):
        assert resumed.front is not None
        after = [(item.role, item.text) for item in resumed.front.state.visible_items()]

    # The conversation comes back whole, plus two lines accounting for the gap:
    # the release and the resume. A person who did not know a previous run
    # existed should not have to infer either from the scrollback — and the
    # *released* notice is new here, because before P5-14 nothing but this
    # terminal could have let the session go.
    assert [item for item in after if item[0] != "notice"] == before
    assert sum("Resumed" in text for role, text in after if role == "notice") == 1
    assert sum("Released" in text for role, text in after if role == "notice") == 1
    assert ("user", "remember this") in after


async def test_a_typed_attach_reaches_the_verb_with_its_argument(
    make_tui_app: MakeApp, tmp_path: Path
) -> None:
    """`/attach <path>` goes through the one route every verb takes, argument included.

    The verb used to be caught by name ahead of the command table — the string
    re-dispatch `commands.py` forbids — because `_RunAction` dropped what
    followed the name. It forwards it now, so the argument arrives at
    `action_attach` through `run_command` like any other verb's body, and this
    pins that it does: the file ends up staged.
    """
    picture = tmp_path / "diagram.png"
    picture.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    async with running(make_tui_app()) as (app, pilot):
        front = app.front
        assert front is not None

        await app._dispatch_command(front, f"/attach {picture}")
        await pilot.pause()

        staged = front._staged.refs  # type: ignore[attr-defined]
        assert [one.name for one in staged] == ["diagram.png"]


# ------------------------------------------------- the harness is elsewhere --


async def test_a_turn_started_in_the_tui_finishes_after_the_tui_is_gone(
    make_tui_app: MakeApp, tui_daemon: Any
) -> None:
    """**The gate this whole plan is named for** (P5-01, P5-14).

    A person submits a prompt, closes the terminal, and the work carries on —
    because the harness is not in the terminal any more. Then a second terminal
    opens the same session and finds the finished turn in the transcript it
    rebuilds from the log.

    Driven through `PHTuiApp` rather than through `DaemonSession`, which is the
    difference from `test_tui_remote.py`'s version: what is asserted here is that
    the *app* takes that path — that closing it detaches rather than ending
    anything.

    Sabotage: have the app's `on_unmount` shut the daemon down, or have
    `DaemonSession.close` cancel the turn, and the second terminal finds no
    assistant message.
    """
    async with running(make_tui_app()) as (app, pilot):
        assert app.front is not None
        await pilot.press(*"keep going")
        # Submitted and then left. **Not** waited on: the fake provider answers
        # in microseconds, so waiting for `busy` is a race a test cannot win —
        # and what is being pinned is not who wins it. It is that the root and
        # its work are not this terminal's to end.
        await pilot.press(app.keys.submit)
        await pilot.pause()

    assert tui_daemon.holds("pilot"), "closing a terminal is not a shutdown"
    root = root_of(tui_daemon)
    await settled(lambda: root.status == "idle", what="the turn to finish with nobody watching")

    async with running(make_tui_app()) as (second, _pilot):
        assert second.front is not None
        rows = [(item.role, item.text) for item in second.front.state.visible_items()]

    assert ("user", "keep going") in rows
    assert any(role == "assistant" for role, _text in rows), "the turn finished, and it is readable"


async def test_two_terminals_on_one_session_share_the_log_and_not_the_composer(
    make_tui_app: MakeApp, tui_daemon: Any
) -> None:
    """The multiplex rule, through two real apps on one root.

    Un-submitted text never leaves the client — it is not an act in the session —
    but pressing enter is, so it reaches the other terminal by the one route
    everything else does. That asymmetry is the whole of the decision that
    replaced the takeover/demotion design.
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
            lambda: any(item.text == "shared" for item in second.front.state.visible_items()),
        )

        assert second.query_one(PromptInput).area.text == "a draft", "the draft is still theirs"
