"""`PHTuiApp` — widgets, keys, and the worker that drives a turn.

Everything harness-shaped lives in `frontend.py`; this file is the terminal. It
satisfies `ModalHost`, so the one rule that cannot be forgotten is enforced by
where the code sits: turns run in a Textual worker, and only a worker may await
a modal.

Keys are Textual bindings built from `TUI_VERBS` — plus one per screen a row
contributed through `ctx.tui_screens` (P4-17) — and remapped from `tui.json`
with one `set_keymap`, screens and modals included, since every binding carries
an id. `priority=True` puts them ahead of the prompt's `TextArea`, which
binds `ctrl+k`, `ctrl+y` and others for editing and would otherwise fire as well
as, or instead of, the app. `check_action` keeps them quiet while a modal is up.

Redrawing is **coalesced, not polled**. A streaming turn commits an
`assistant/chunk` every few tokens, and `view.sync` reconciles a widget list
rather than repainting — Textual's own compositor coalescing does not cover it —
so drawing per event spends more time laying out than rendering. Instead the
first change schedules one draw a frame later and every change until then rides
it: same bounded latency, and **nothing runs while nothing happens.**

That last part is the reason it is not a 30 Hz interval any more. Per terminal a
polled flag costs thirty no-op wakeups a second, which is nothing; but
`ph --mode web` runs *one Textual subprocess per browser tab*, and ten idle tabs
polling a flag they will find unchanged is thirty times ten processes waking to
do nothing.

The spinner is the one thing that genuinely wants a clock, because it advances on
wall-clock time rather than on anything arriving — so it gets its own interval,
started when a turn starts and stopped when it ends.

@module ph_app.tui.app
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.timer import Timer

from ph.cordis import Profile
from ph.paths import resolve_roots
from ph.seams.approval import ApprovalAnswer, ApprovalRequest
from ph.seams.permission_presets import PRESETS
from ph.seams.tui_status import StatusReading
from ph.seams.user_questions import UserQuestion

from .autocomplete import PathCompleter
from .commands import app_bindings
from .config import TuiKeybindings, TuiSettings, load_tui_settings, save_tui_settings
from .frontend import FrontSession, open_harness
from .modals.approval import ApprovalModal
from .modals.ask_user import AskUserModal
from .modals.base import Choice, ChoicePicker
from .modals.login import LoginModal, credential_choices
from .modals.pickers import (
    command_choices,
    model_choices,
    preset_choices,
    session_choices,
    theme_choices,
)
from .modals.trust import TrustStore, project_trust_modal
from .screens import Revealing, RevealSeq
from .terminal import TerminalTitle
from .themes import ThemeCatalog, fallback_variables, load_catalog
from .widgets.prompt import PromptInput
from .widgets.status import Sidebar, StatusBar
from .widgets.transcript import TranscriptView

__all__ = ["PHTuiApp", "run_tui"]

log = logging.getLogger("ph_app.tui.app")

FRAME_INTERVAL = 1 / 30
"""Redraws a second. Fast enough that streaming looks continuous, slow enough
that a burst of chunks costs one layout instead of forty."""


class PHTuiApp(App[str | None]):
    """pH in a terminal.

    The exit value is a session id to reopen, or `None`. Resuming is a restart,
    not a mutation: the transcript, the agent scope and every registration
    belong to the session being left, so `run_tui` unwinds this app completely
    and mounts a new one on the chosen session.
    """

    ENABLE_COMMAND_PALETTE = False
    """pH's palette is `ctx.commands`, opened by its own verb."""

    BINDINGS: ClassVar[list[BindingType]] = app_bindings(TuiKeybindings())
    """Defaults; `on_mount` remaps them from the user's settings."""

    CSS = """
    Screen { background: $ph-background; color: $ph-foreground; layers: base; }
    #body { height: 1fr; }
    #main { width: 1fr; height: 1fr; }
    #chrome { height: auto; }
    """

    def __init__(
        self,
        profile: Profile,
        *,
        provider: str = "fake",
        model: str = "fake-1",
        session_id: str | None = None,
        resume: str | None = None,
        home: Path | None = None,
    ) -> None:
        super().__init__()
        self.profile = profile
        self.provider = provider
        self.model = model
        self.session_id = session_id
        self.resume = resume
        self.home = home or resolve_roots().home
        self.settings: TuiSettings = load_tui_settings(self.home)
        self.catalog: ThemeCatalog = load_catalog(self.home)
        self.project = Path.cwd()
        self.trust = TrustStore(path=self.home / "trust.json")
        self.front: FrontSession | None = None
        self.title_writer = TerminalTitle()
        self._paths = PathCompleter(root=str(self.project))
        self._dirty = True
        self._draw_timer: Timer | None = None
        """The one scheduled draw, or `None` when none is pending.

        Its presence *is* the coalescing: a burst of appends schedules one draw
        and then finds it already scheduled."""
        self._spinner: Timer | None = None
        """The frame clock, alive only while a turn is."""
        self._readings: Sequence[StatusReading] = ()
        """The footer as of the last draw, re-rendered by the spinner.

        Cached because a reading is a *fold of the log* and the log only changes
        on an append — which marks the view dirty and draws. Recomputing them per
        frame ran every registered status field over the whole session thirty
        times a second to get the same answer."""
        self._command_disposers: list[Callable[[], Any]] = []
        # Held rather than queried. `App.query_one` searches the *top* screen,
        # so every lookup would fail while a modal is up — and the frame timer
        # runs thirty times a second whether or not one is.
        self._view: TranscriptView | None = None
        self._status: StatusBar | None = None
        self._sidebar: Sidebar | None = None
        self._prompt: PromptInput | None = None

    @property
    def keys(self) -> TuiKeybindings:
        return self.settings.keybindings

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return fallback_variables()

    def add_binding(self, binding: Binding) -> Callable[[], Any]:
        """Bind a key on the live app, and hand back its removal.

        Here rather than in `screens.py` because this is the one place that
        reaches into Textual's internals, and it belongs beside the other
        binding concerns — `BINDINGS`, `set_keymap`, `check_action` — where a
        Textual upgrade will be looked for.

        Written against `BindingsMap.key_to_bindings` rather than the public
        `bind()`, which drops the `id`: the id is what `apply_keymap` matches
        on, so without it a contributed screen's key would be the one key in
        the app a user could not rebind.
        """
        table = self._bindings.key_to_bindings
        table.setdefault(binding.key, []).append(binding)
        self._refresh_bindings()

        def remove() -> None:
            bindings = table.get(binding.key)
            if bindings is None or binding not in bindings:
                return
            bindings.remove(binding)
            if not bindings:
                del table[binding.key]
            self._refresh_bindings()

        return remove

    def _refresh_bindings(self) -> None:
        """Tell the app its bindings moved, if there is still an app to tell.

        Teardown reaches here: unwinding a row's scope removes its key after the
        app has stopped, and `refresh_bindings` walks the screen stack.
        """
        if self.is_running:
            self.refresh_bindings()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Keep global keys quiet while a modal owns the screen — except quit."""
        if action == "quit":
            return True
        if len(self.screen_stack) > 1:
            return False
        return not (action.startswith("open_") and self.front is None)

    # --------------------------------------------------------------- layout --

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            if self.settings.sidebar == "left":
                yield Sidebar(classes="-left")
            with Vertical(id="main"):
                yield TranscriptView(id="transcript")
                with Vertical(id="chrome"):
                    yield PromptInput(self.keys, completion_source=self._completion_source)
                    yield StatusBar()
            if self.settings.sidebar != "left":
                yield Sidebar()

    async def on_mount(self) -> None:
        self.set_keymap(self.keys.as_map())
        self.catalog.install(self)
        self.theme = self.catalog.resolve(self.settings.theme).name
        self._view = self.query_one("#transcript", TranscriptView)
        self._status = self.query_one(StatusBar)
        self._sidebar = self.query_one(Sidebar)
        self._prompt = self.query_one(PromptInput)
        self._sidebar.display = self.settings.sidebar != "off"
        # The first draw is scheduled like every other one. `__init__` cannot:
        # `set_timer` needs a running app.
        self.state_changed()
        self._prompt.area.focus()
        if self.trust.trusted(self.project):
            self.run_worker(self._open(), group="open")
            return
        # Asked before mounting, because mounting is what reads the project's
        # AGENTS.md, its hooks and its configured plugins. `push_screen` with a
        # callback, never awaited: this runs on the message pump.
        self.push_screen(project_trust_modal(self.project), self._answer_trust)

    def _answer_trust(self, answer: str | None) -> None:
        if answer == "trust":
            self.trust.trust(self.project)
        if answer in ("trust", "once"):
            self.run_worker(self._open(), group="open")
            return
        self.exit()

    async def _open(self) -> None:
        """Mount the harness. In a worker, so the shell paints first."""
        try:
            self.front = await open_harness(
                self.profile,
                host=self,
                provider=self.provider,
                model=self.model,
                session_id=self.session_id,
                resume=self.resume,
            )
        except Exception as error:
            log.exception("ph_app.tui: the harness would not mount")
            self.notify(str(error), title="pH could not start", severity="error", markup=False)
            return
        # A screen a row contributed gets its verb, its key and its palette
        # entry from here — but each of those unwinds with the *row*, not with
        # this list, which is what makes unloading one take all three with it.
        self._command_disposers = self.front.attach_surfaces(self)
        self.state_changed()

    async def on_unmount(self) -> None:
        # **Before anything else unwinds.** Both timers call into widgets, and a
        # frame that lands after the status bar has been taken apart queries a
        # node that is no longer there. The old single interval was stopped for
        # free when the app shut its own timers down; two timers this class owns
        # are two this class stops.
        self._spin(False)
        if self._draw_timer is not None:
            self._draw_timer.stop()
            self._draw_timer = None
        for dispose in reversed(self._command_disposers):
            dispose()
        self._command_disposers.clear()
        if self.front is not None:
            await self.front.close()
        self.title_writer.clear()

    # ---------------------------------------------------------------- frames --

    def state_changed(self) -> None:
        """Something the view renders has changed. Draw soon, and once.

        **The single entry point**, and every local mutation goes through it too
        rather than setting the flag: the flag alone was enough while a poll was
        watching it, and is a permanently stale pane now that nothing is. That is
        the one thing this change makes worse if it is got wrong, so there is one
        place to get right.
        """
        self._dirty = True
        if self._draw_timer is None:
            self._draw_timer = self.set_timer(FRAME_INTERVAL, self._draw)

    async def _draw(self) -> None:
        """Render what has arrived since the last draw."""
        self._draw_timer = None
        front, status, view = self.front, self._status, self._view
        if front is None or status is None or view is None:
            return
        self._dirty = False
        self._readings = front.status_readings()
        status.show(front.state, self._readings)
        await view.sync(self._rows(front))
        if self._sidebar is not None and self._sidebar.display:
            self._sidebar.show(front.state, session_id=front.session_id, cwd=str(self.project))
        self._spin(front.state.busy)

    def _spin(self, running: bool) -> None:
        """Start or stop the frame clock, so it exists only while a turn does."""
        if running and self._spinner is None:
            self._spinner = self.set_interval(FRAME_INTERVAL, self._advance)
        elif not running and self._spinner is not None:
            self._spinner.stop()
            self._spinner = None

    def _advance(self) -> None:
        """One spinner frame: the glyph, the title, and the line they sit in.

        Re-renders the status line rather than the transcript, because the glyph
        is a value `show` composes — and with the *cached* readings, so a frame
        costs a string build rather than a fold of the log.

        It also stops itself if the turn ended without a draw noticing. Belt and
        braces: `_draw` is what normally stops it, and a spinner nobody can stop
        is the failure mode of every animation loop ever written.
        """
        front, status = self.front, self._status
        if front is None or status is None or not front.state.busy:
            self._spin(False)
            return
        status.tick()
        self.title_writer.set(f"{status.glyph} working")
        status.show(front.state, self._readings)

    def _rows(self, front: FrontSession) -> list[Any]:
        return front.state.visible_items(
            thinking=self.settings.show_thinking, tool_results=self.settings.show_tool_results
        )

    # ----------------------------------------------------------------- turns --

    async def on_prompt_input_submitted(self, message: PromptInput.Submitted) -> None:
        front = self.front
        if front is None:
            return
        if message.text.startswith("!!"):
            # The person's own shell, not a turn: `!!` spends no tokens and the
            # model never sees it. Logged all the same — pressing enter is an act
            # in the session, so every attached UI renders it from the event.
            self.run_worker(self._shell(front, message.text[2:].strip()), group="shell")
            return
        if message.text.startswith("/"):
            # A command is the human's verb. Sending it as a prompt would spend
            # a turn and make the log say the model chose it.
            await self._dispatch_command(front, message.text)
            return
        if message.queue or front.state.busy:
            # The driver refuses a second concurrent `run()`, and a person
            # typing mid-turn means "also this", not "instead of that".
            front.queue(message.text)
            return
        self._run_turn(message.text)

    async def _shell(self, front: FrontSession, command: str) -> None:
        """Run `!!<command>` and let the log do the rendering.

        Nothing is drawn here: the command and its output arrive as `shell/*`
        events like everything else, which is what keeps `TuiState` entirely
        event-derived and lets the browser and the terminal share one fold.
        """
        if not command:
            self.notify("type `!!<command>` to run a shell command", title="shell", markup=False)
            return
        try:
            await front.shell(command)
        except Exception as error:
            log.exception("ph_app.tui: a shell command failed to start")
            self.notify(str(error), title="shell", severity="error", markup=False)

    async def action_attach(self, argument: str = "") -> None:
        """`/attach <path> …` — stage files for the next prompt.

        **Client-side, and that is the whole point** (I-9): the front end reads
        the bytes with the person's own permissions and hands the harness
        content, never a path. In process that reads this machine; over a socket
        it reads the *client's* machine, which is the only thing a browser tab
        could do and the reason a remote terminal can attach a file the daemon
        has never seen.

        The one verb that takes an argument, so it is the one whose body reads
        one — `_RunAction` forwards what followed the name. Chosen from the
        palette there is nothing to attach, so it says how it is used rather
        than doing nothing.
        """
        front = self.front
        paths = argument.split()
        if front is None:
            return
        if not paths:
            self.notify("type `/attach <path> …` to attach files", title="attach", markup=False)
            return
        try:
            staged = await front.attach(paths)
        except Exception as error:
            # Loud: a person who attached a diagram and got a plain text turn
            # would have no way to tell it was never sent (P7-01).
            log.exception("ph_app.tui: attaching failed")
            self.notify(str(error), title="attach", severity="error", markup=False)
            return
        named = ", ".join(one.name or one.attachment_id[:15] for one in staged)
        self.notify(f"staged for the next prompt: {named}", title="attach", markup=False)
        self.state_changed()

    async def on_prompt_input_cancelled(self, _message: PromptInput.Cancelled) -> None:
        if self.front is not None and self.front.state.busy:
            self.front.cancel()

    async def _dispatch_command(self, front: FrontSession, line: str) -> None:
        name = line.split()[0]
        try:
            shown = await front.run_command(line)
        except KeyError as error:
            self.notify(str(error), title="unknown command", severity="warning", markup=False)
            return
        except Exception as error:
            log.exception("ph_app.tui: a command failed")
            self.notify(str(error), title=name, severity="error", markup=False)
            return
        if shown:
            self.notify(shown, title=name, markup=False)
        self.state_changed()

    @work(exclusive=True, group="turn")
    async def _run_turn(self, text: str) -> None:
        """One turn, in a worker — which is what makes the modals legal."""
        front = self.front
        if front is None:
            return
        try:
            await front.submit(text)
        except Exception:
            log.exception("ph_app.tui: a turn failed")
        finally:
            self.state_changed()
            self.title_writer.set("")
            if self.settings.turn_notification == "bell":
                self.bell()

    # ------------------------------------------------------------ modal host --

    async def ask_approval(self, request: ApprovalRequest) -> tuple[ApprovalAnswer, str]:
        decision = await self.push_screen_wait(ApprovalModal(request))
        return decision.answer, decision.reason

    async def ask_question(self, question: UserQuestion) -> str | None:
        answer = await self.push_screen_wait(AskUserModal(question))
        return answer if isinstance(answer, str) else None

    # -------------------------------------------------------------- actions --
    # One per `TuiVerb`. Reached by key, by `/command`, and by `run_action`.
    # Every picker opens with `push_screen(screen, callback)` and returns: an
    # action runs on the message pump, where awaiting a dismissal deadlocks.

    def _pick(
        self,
        title: str,
        choices: list[Choice],
        then: Callable[[str | None], None],
        **options: Any,
    ) -> None:
        self.push_screen(ChoicePicker(title=title, choices=choices, **options), then)

    def action_open_screen(self, screen_id: str) -> None:
        """Open a screen a row contributed (P4-17).

        One action for every registered screen, reached by its key and by its
        `/<id>` command alike. `build` runs here rather than at registration, so
        what opens is a fold of the log as it stands.
        """
        front = self.front
        if front is None:
            return
        definition = self._screen(screen_id)
        if definition is None:
            self.notify(
                f"no screen is registered as {screen_id!r}",
                title="screen",
                severity="warning",
                markup=False,
            )
            return
        screen = definition.build(front.session)
        if isinstance(screen, Revealing) and self._view is not None:
            # Opened where the reader is, when the screen can take a position.
            seq = self._view.seq_in_view()
            if seq >= 0:
                screen.reveal(seq)
        self.push_screen(screen)

    def on_reveal_seq(self, message: RevealSeq) -> None:
        """A screen asking for the transcript row behind one of its own.

        The other half of the join: the screen names a log seq, the transcript
        owns which widget that is. Popping first, so what the reader lands on is
        the row and not the screen they were leaving.
        """
        message.stop()
        if len(self.screen_stack) > 1:
            self.pop_screen()
        if self._view is not None and not self._view.scroll_to_seq(message.seq):
            definition = self._screen(message.screen_id)
            self.notify(
                "that row has no counterpart in the transcript",
                title=definition.label if definition is not None else "screen",
                severity="warning",
                markup=False,
            )

    def _screen(self, screen_id: str) -> Any:
        """The registered screen with this id, or `None`."""
        front = self.front
        return front.screen(screen_id) if front is not None else None

    def action_open_commands(self) -> None:
        if self.front is not None:
            self._pick("commands", command_choices(self.front.commands()), self._insert_command)

    def _insert_command(self, chosen: str | None) -> None:
        if chosen is not None and self._prompt is not None:
            self._prompt.area.insert(f"{chosen} ")
            self._prompt.area.focus()

    def action_open_models(self) -> None:
        front = self.front
        if front is None:
            return
        providers = front.providers()
        self._pick(
            "model",
            model_choices(providers, front.state.provider, front.state.model),
            self._set_model,
            free_text="provider/model",
        )

    def _set_model(self, chosen: str | None) -> None:
        front = self.front
        if chosen is None or front is None:
            return
        provider, _, model = chosen.partition("/")
        front.state.provider = provider
        front.state.model = model or front.state.model
        self.notify(f"{front.state.provider}/{front.state.model}", title="model", markup=False)
        self.state_changed()

    def action_open_sessions(self) -> None:
        front = self.front
        if front is None:
            return
        # A backend with no per-session file answers the fallback, and the
        # picker lists nothing — honest until P5-14 moves this onto `stored()`.
        # `FrontSession.sessions_directory` carries the rest of the reasoning.
        directory = front.sessions_directory(self.home / "sessions")
        self._pick(
            "sessions",
            session_choices(directory, current=front.session_id),
            self._resume_session,
            free_text="session id",
        )

    def _resume_session(self, chosen: str | None) -> None:
        if chosen is not None and (self.front is None or chosen != self.front.session_id):
            self.exit(chosen)

    def action_open_themes(self) -> None:
        self._pick(
            "theme",
            theme_choices(self.theme, self.catalog),
            self._set_theme,
            on_highlight=self._preview_theme,
        )

    def _preview_theme(self, name: str) -> None:
        """Apply a theme as the cursor passes it — seeing it is how you pick."""
        if name in self.catalog.themes:
            self.theme = name

    def _set_theme(self, chosen: str | None) -> None:
        if chosen is not None and chosen in self.catalog.themes:
            self._save(replace(self.settings, theme=chosen))
        # Chosen or cancelled, the theme in force is the one the settings name.
        self.theme = self.catalog.resolve(self.settings.theme).name

    def action_open_presets(self) -> None:
        if self.front is not None:
            self._pick("permissions", preset_choices(self.front.state.preset), self._set_preset)

    def _set_preset(self, chosen: str | None) -> None:
        front = self.front
        if chosen is None or front is None or chosen not in PRESETS:
            return
        front.set_preset(chosen)
        self.state_changed()

    def action_open_login(self) -> None:
        front = self.front
        if front is None:
            return
        self._pick(
            "credential",
            credential_choices(front.config_rows, front.credential_held),
            self._ask_for_secret,
            free_text="environment variable",
        )

    def _ask_for_secret(self, name: str | None) -> None:
        if name is not None:
            self.push_screen(LoginModal(name), lambda value: self._store(name, value))

    def _store(self, name: str, value: str | None) -> None:
        front = self.front
        # In-process only, and never written down: `notify` names the credential
        # and never the secret.
        if value is None or front is None or not front.store_credential(name, value):
            return
        self.notify(f"{name} set for this session", title="login", markup=False)

    async def action_toggle_thinking(self) -> None:
        self._save(replace(self.settings, show_thinking=not self.settings.show_thinking))
        await self._rebuild()

    async def action_toggle_tool_results(self) -> None:
        self._save(replace(self.settings, show_tool_results=not self.settings.show_tool_results))
        await self._rebuild()

    async def _rebuild(self) -> None:
        """Redraw from scratch — what a display toggle needs."""
        if self._view is not None and self.front is not None:
            await self._view.rebuild(self._rows(self.front))

    def action_toggle_sidebar(self) -> None:
        if self._sidebar is not None:
            self._sidebar.display = not self._sidebar.display
            self.state_changed()

    def _save(self, settings: TuiSettings) -> None:
        """Adopt new settings and persist them. A write failure costs the memory, not the change."""
        self.settings = settings
        try:
            save_tui_settings(self.home, settings)
        except OSError:
            log.warning("ph_app.tui: could not write tui.json", exc_info=True)

    # ---------------------------------------------------------- completions --

    def _completion_source(self) -> dict[str, Any]:
        front = self.front
        commands: list[tuple[str, str]] = []
        if front is not None:
            commands = [(d.name, d.summary) for d in front.commands()]
        return {"commands": commands, "paths": self._paths}


async def run_tui(
    profile: Profile,
    *,
    provider: str,
    model: str,
    session_id: str | None = None,
    resume: str | None = None,
) -> None:
    """Entry point for `--mode tui`.

    Loops so that choosing a session from the picker reopens it: the app exits
    with the id, everything it mounted has unwound, and a fresh app resumes.
    """
    while True:
        app = PHTuiApp(
            profile, provider=provider, model=model, session_id=session_id, resume=resume
        )
        resume = await app.run_async()
        if resume is None:
            return
        session_id = None
