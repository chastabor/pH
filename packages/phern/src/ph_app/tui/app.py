"""`PHTuiApp`: the widgets, the keys, and the worker that drives a turn.

This file is the terminal. `frontend.py` declares what the terminal needs from
a harness, and `remote.py` implements it. `PHTuiApp` satisfies `ModalHost`.
Turns run in a Textual worker, and only a worker may await a modal. The place
of the code enforces that rule, so nobody can forget it.

Keys are Textual bindings. They are built from `TUI_VERBS` and `VIEW_KEYS`,
with one more for each screen that a row contributed through `ctx.tui_screens`
(P4-17). One `set_keymap` call remaps them from `tui.json`, screens and modals
included, since every binding carries an id. `priority=True` puts them ahead of
the `TextArea` of the prompt. The `TextArea` binds `ctrl+k`, `ctrl+y` and
others for editing. Without the priority, its binding would fire as well as the
binding of the app, or in its place. The keys are bound on `ChatScreen`, the
base screen of the app. A dialog or another screen on top therefore takes them
out of the key chain. Only quit is bound on the app, which the key chain of
every screen reaches.

Redrawing is **coalesced, not polled**. A streaming turn commits an
`assistant/chunk` every few tokens. `view.sync` reconciles a widget list and
does not repaint, so the coalescing in the compositor of Textual does not cover
it. A draw for each event therefore spends more time on layout than on
rendering. Instead, the first change schedules one draw a frame later, and that
one draw shows every change until then. The latency has the same bound, and
**nothing runs while nothing happens.**

That last property is the reason why the redraw is not a 30 Hz interval any
more. For one terminal, a polled flag costs thirty wakeups a second that do
nothing, which is negligible. But `phern --mode web` runs *one Textual
subprocess for each browser tab*. Ten idle tabs that poll an unchanged flag are
ten processes, and each one wakes thirty times a second to do nothing.

The spinner is the one thing that needs a clock, because it advances on
wall-clock time and not when something arrives. It therefore has its own
interval, which starts when a turn starts and stops when the turn ends.

@module ph_app.tui.app
"""

from __future__ import annotations

import logging
import shlex
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, ParamSpec

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.timer import Timer
from textual.worker import Worker

from ph.paths import resolve_roots
from ph.seams.approval import ApprovalAnswer, ApprovalRequest
from ph.seams.commands import CommandDefinition
from ph.seams.models import ModelChoice, ModelChoiceError
from ph.seams.permission_presets import PRESET_NAMES
from ph.seams.tui_screens import ScreenDefinition
from ph.seams.tui_status import StatusReading
from ph.seams.user_questions import UserQuestion

from ..daemon.client import DaemonClient
from ..daemon.launch import ensure_daemon
from ..payloads import AskKey, ProfileDecision
from ..protocol import DaemonError
from ..trust import TrustAnswer, TrustStore, trust_path
from .autocomplete import PathCompleter
from .commands import (
    TUI_VERBS,
    VIEW_USAGE,
    VIEWS,
    app_bindings,
    local_commands,
    run_line,
    screen_bindings,
)
from .config import TuiKeybindings, TuiSettings, load_tui_settings, save_tui_settings
from .frontend import FrontSession
from .modals.approval import ApprovalModal
from .modals.ask_user import AskUserModal
from .modals.base import Choice, ChoicePicker, PhModal
from .modals.login import LoginModal, credential_choices
from .modals.pickers import (
    NEW_SESSION,
    command_choices,
    history_choices,
    model_choices,
    preset_choices,
    session_choices,
    theme_choices,
)
from .modals.profile import decision_of, profile_changed_modal
from .modals.start import could_not_start_modal, start_decision
from .modals.trust import project_trust_modal
from .remote import attach_session, browse_on
from .screens import Revealing, RevealSeq
from .state import CatalogEntry, ChatItem, PromptRecord, Surface
from .terminal import TerminalTitle
from .themes import (
    ThemeCatalog,
    ThemeProfile,
    choose_theme,
    fallback_variables,
    load_catalog,
    load_theme_profile,
)
from .widgets.prompt import PromptInput
from .widgets.status import Sidebar, StatusBar, shorten_path
from .widgets.transcript import TranscriptView

__all__ = ["PHTuiApp", "run_tui"]

log = logging.getLogger("ph_app.tui.app")

_P = ParamSpec("_P")

VERB_GROUP = "verb"
"""The prefix of the worker group under which the harness call of a verb runs.

It has a name because three places must agree on it: `verb_work` below, the
gate that asserts that every slow verb carries it, and the pilot tests. The
pilot tests wait for such a worker to finish, and do not pause for a guessed
time."""


def verb_work(
    verb: str,
) -> Callable[[Callable[_P, Coroutine[Any, Any, None]]], Callable[_P, Worker[None]]]:
    """`@work` for a verb that must reach the harness (H7). The policy is read
    from the row of the verb.

    **The policy is `TuiVerb.work`, and not an argument here.** The policy has
    two halves: whether this verb schedules a worker at all, and whether a
    second press supersedes the first. Both are facts about the verb, and
    `TUI_VERBS` is where a verb already says what it is. When they were written
    at the decorator, they were a second copy that a new verb could get wrong,
    and a test had to assert that the two agreed.

    **Each verb has its own group, and that makes `replace` safe.** With one
    shared group and `exclusive=True`, opening the session picker would cancel
    an `/attach` that is in flight. That is the work of a different verb. In
    that case the user asked for an attachment and silently does not get it.
    """
    row = next((one for one in TUI_VERBS if one.name == verb), None)
    if row is None or row.work is None:  # pragma: no cover - the gate refuses it
        raise ValueError(f"{verb!r} is not a verb declaring `work`")
    return work(group=f"{VERB_GROUP}:{row.name}", exclusive=row.work == "replace")


ANSWERED_ELSEWHERE = "Answered in another terminal."
"""Why a modal went away by itself. This is the usual reason, and the default."""

FRAME_INTERVAL = 1 / 30
"""The time between two redraws, in seconds: at most thirty redraws a second.
That is fast enough that streaming looks continuous, and slow enough that a
burst of chunks costs one layout and not forty."""


class ChatScreen(Screen[None]):
    """The base screen of the app: the chat, and the keys that belong to it.

    It is a screen of its own only for its bindings (`commands.screen_bindings`).
    Because they are bound here and not on the app, they are in the key chain
    exactly while this screen is on top. The layout still comes from
    `PHTuiApp.compose`.
    """

    BINDINGS: ClassVar[list[BindingType]] = screen_bindings(TuiKeybindings())
    """The defaults. `PHTuiApp.on_mount` remaps them from the settings of the user."""


class PHTuiApp(App[str | None]):
    """pH in a terminal.

    The exit value is a session id to reopen, or `None`. To resume is to
    restart, and not to mutate. The transcript, the agent scope and every
    registration belong to the session that the user leaves. `run_tui`
    therefore unwinds this app completely and starts a new one on the chosen
    session.
    """

    ENABLE_COMMAND_PALETTE = False
    """pH has its own palette, `PHTuiApp.commands`, which its own verb opens."""

    BINDINGS: ClassVar[list[BindingType]] = app_bindings(TuiKeybindings())
    """The keys that work behind a dialog, which is only quit. `on_mount` remaps
    them with the others. The keys of the chat belong to `ChatScreen`."""

    CSS = """
    Screen { background: $ph-background; color: $ph-foreground; layers: base; }
    #body { height: 1fr; }
    #main { width: 1fr; height: 1fr; }
    #chrome { height: auto; }
    """

    def __init__(
        self,
        *,
        daemon_argv: Sequence[str] = (),
        session_id: str | None = None,
        home: Path | None = None,
        spawn: bool = True,
        offer_sessions: bool = True,
        model_choice: ModelChoice = ModelChoice(),
        profile: str | None = None,
    ) -> None:
        super().__init__()
        self.chat = ChatScreen()
        """The base screen. `get_default_screen` returns it, and the key of a
        contributed screen is bound on it (`add_binding`)."""
        self.profile = profile
        """The profile on which a session that this terminal *creates* runs (S5).
        `None` means the profile of the daemon. A session that this terminal
        resumes runs on the profile that the log of that session records."""
        self.model_choice = model_choice
        """`--provider` and `--model`. `session/new` carries them for the session
        that this terminal opens.

        The route belongs to this session and not to the daemon. `spawn_command`
        carries no route, so a daemon that this terminal starts runs every other
        session on the default of its profile. A daemon that is already serving
        keeps its other sessions where they are. If the daemon refuses the
        route, the start fails with the sentence of the daemon, as
        `phern -p --model` would."""
        self.daemon_argv = tuple(daemon_argv)
        """The command to run when no daemon is listening.

        The caller passes it in, and this app does not build it, because it is
        the command line of the *CLI*: `ph_app.cli.spawn_command` spells the
        options. A terminal that composed it would be a terminal that knows what
        a profile is. After P5-14 it does not: the daemon mounts, and this app
        reads a socket."""
        self.spawn = spawn
        """Whether to start a daemon when none is listening (P7-08).

        `--no-spawn` turns it off. That is for a deployment that runs
        `phern daemon` under an init system. There, a daemon that a UI started
        would be a second supervisor that competes for the same session leases."""
        self.offer_sessions = offer_sessions
        """Whether to offer the sessions of this directory before a new one starts.

        `--new` turns it off. A user who always wants a fresh session should not
        have to press a key to get one. A script that drives the TUI should not
        meet a modal that it did not ask for."""
        self.session_id = session_id
        """The session to open, or `None` for a new one.

        This is one field and not two. `--resume <id>` and `--session <id>` were
        separate because to mount a stored session and to create one were
        different code paths. Over the daemon they are the same calls:
        `session/new` resumes an existing id, `session/attach` subscribes to
        it, and `session/snapshot` pages its log. A resume is therefore an id
        and nothing else."""
        self.home = home or resolve_roots().home
        self.settings: TuiSettings = load_tui_settings(self.home)
        self.catalog: ThemeCatalog = load_catalog(self.home)
        self.theme_profile: ThemeProfile = load_theme_profile(self.home)
        """Which theme to open in, and how `/theme` orders its list.

        It has its own file and is not a field on `self.settings`. `_save`
        writes the whole of `tui.json`. A theme stored there would be rewritten
        by every `/view` toggle, and two files would claim one fact (P9-02)."""
        self.project = Path.cwd()
        self.trust = TrustStore(path=trust_path(self.home))
        self.front: FrontSession | None = None
        self._client: DaemonClient | None = None
        self.refusal: str | None = None
        """Why this terminal has no session: the sentence that its start failed
        with. It is kept until the user retries (H3). It is `None` before that,
        and after the terminal is attached.

        Every place where a user could otherwise be left guessing reads it: the
        status line, a line typed at the prompt, a verb that needs a session.
        The modal that first showed the sentence can be dismissed, and a toast
        is gone in five seconds."""
        self._verbs = local_commands(self)
        """The slash commands of this terminal, built one time, because the
        completion source asks for the list on every keystroke. They belong to
        the app and not to the front end, so `/theme` and `/quit` answer whether
        or not anything is attached (H3)."""
        self.title_writer = TerminalTitle()
        self._paths = PathCompleter(root=str(self.project))
        self._open_asks: dict[AskKey, PhModal[Any]] = {}
        """The ask modals that are up now, keyed by the root and the name under
        which the daemon asked (H1).

        The map lets `withdraw_ask` take a modal down when another terminal
        answers first. An ask with no id is not in it: an in-process caller has
        no second front end to race with."""
        self._dirty = Surface.ALL
        self._draw_timer: Timer | None = None
        """The one scheduled draw, or `None` when no draw is pending.

        Its presence *is* the coalescing: a burst of appends schedules one draw,
        and every later append finds that draw already scheduled."""
        self._spinner: Timer | None = None
        """The frame clock. It exists only while a turn runs."""
        self._readings: Sequence[StatusReading] = ()
        """The footer as of the last draw. The session panel places these
        readings too.

        They are cached because a reading is a *fold of the log*, and the log
        changes only on an append. An append marks the footer dirty, and that
        refreshes this cache. A frame that draws only the sidebar reads the
        cache and does not fold the log again. The spinner reads nothing,
        because its glyph has a cell of its own."""
        self._command_disposers: list[Callable[[], None]] = []
        # The widgets are held and not queried. `App.query_one` searches the
        # *top* screen, so every lookup would fail while a modal is up. The
        # draw timer and the spinner fire whether or not a modal is up.
        self._view: TranscriptView | None = None
        self._status: StatusBar | None = None
        self._sidebar: Sidebar | None = None
        self._prompt: PromptInput | None = None
        self._history_rows: dict[str, str] = {}
        """The prompts of the open history picker, by seq. Each value is the
        *full* text, which the one-line label has truncated.

        The map is held because `on_highlight` receives a `Choice.value` and
        nothing else. It holds text and not whole records: the seq is already
        the key, so a record would carry a second copy of it. The map is cleared
        when the picker closes. A `--resume` replay, which clears
        `TuiState.items` in place, therefore cannot leave the prompts of the
        previous transcript held here."""

    @property
    def keys(self) -> TuiKeybindings:
        return self.settings.keybindings

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return fallback_variables()

    def get_default_screen(self) -> ChatScreen:
        """The screen of the chat, which holds the keys of the chat. See `ChatScreen`."""
        return self.chat

    def add_binding(self, binding: Binding) -> Callable[[], None]:
        """Bind a key on the screen of the chat, and return a function that
        removes it.

        This is here and not in `screens.py` because it is the one place that
        reaches into the internals of Textual. It belongs beside the other
        binding code, `BINDINGS` and `set_keymap`, which is where someone will
        look after a Textual upgrade.

        It writes to `BindingsMap.key_to_bindings` and does not call the public
        `bind()`, because `bind()` drops the `id`. `apply_keymap` matches on the
        id. Without it, the key of a contributed screen would be the one key in
        the app that a user could not rebind.
        """
        table = self.chat._bindings.key_to_bindings
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
        """Tell the app that its bindings changed, if the app is still running.

        Teardown reaches this method. The key of a screen can be removed after
        the app has stopped, and `refresh_bindings` walks the screen stack.
        """
        if self.is_running:
            self.refresh_bindings()

    # --------------------------------------------------------------- layout --

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            if self.settings.sidebar == "left":
                yield Sidebar(classes="-left")
            with Vertical(id="main"):
                yield TranscriptView(id="transcript")
                with Vertical(id="chrome"):
                    # The status line comes first. It is the top edge of the
                    # prompt (see `StatusBar`), so it sits between the
                    # conversation and the box.
                    yield StatusBar()
                    yield PromptInput(
                        self.keys,
                        completion_source=self._completion_source,
                        history_source=self._history_source,
                    )
            if self.settings.sidebar != "left":
                yield Sidebar()

    async def on_mount(self) -> None:
        self.set_keymap(self.keys.as_map())
        self.catalog.install(self)
        self.theme = self.catalog.resolve(self.theme_profile.chosen).name
        self._view = self.query_one("#transcript", TranscriptView)
        self._status = self.query_one(StatusBar)
        self._sidebar = self.query_one(Sidebar)
        self._prompt = self.query_one(PromptInput)
        self._sidebar.display = self.settings.sidebar != "off"
        # The first draw is scheduled like every other draw. `__init__` cannot
        # schedule it, because `set_timer` needs a running app.
        self.state_changed()
        self._prompt.area.focus()
        if self.trust.trusted(self.project):
            # The store already vouches for this directory, so nobody is asked.
            self.run_worker(self._open(None), group="open")
            return
        # The question comes before the mount, because the mount is what reads
        # the `AGENTS.md` of the project, its hooks and its configured plugins.
        # `push_screen` takes a callback and is never awaited, because this
        # runs on the message pump.
        self.push_screen(project_trust_modal(self.project), self._answer_trust)

    def _answer_trust(self, answer: TrustAnswer | str | None) -> None:
        """Carry the answer of the user to the daemon, which enforces it.

        The answer is not written down here. The file has one writer, on the
        side that mounts. `ph_app.trust` says why.
        """
        if answer in ("always", "once"):
            # The callback of `push_screen` is typed by the result of the modal,
            # which is any of its action names. The check above narrows the
            # value to the two that are answers.
            self.run_worker(self._open(answer), group="open")
            return
        self.exit()

    async def _open(self, trust: TrustAnswer | None) -> None:
        """Get a daemon, attach to a session on it, and read it. Runs in a worker.
        `trust` is what the user answered, or `None` when nobody was asked.

        The harness is in the daemon now (P5-14), so this connects and does not
        mount. It starts a daemon first if nothing is listening. If a user had
        to run `phern daemon` before `phern`, the ordinary case would need a
        second command.

        The pump is a worker of its own and outlives this one. `attach_session`
        makes calls that complete only when something reads the replies.
        """
        try:
            started = await ensure_daemon(
                argv=self.daemon_argv,
                spawn=self.spawn,
            )
            client = await DaemonClient.connect(started.path)
            self._client = client
            self.run_worker(client.pump(), group="pump", exclusive=False)
            # The watcher starts before the attach, because the attach is itself
            # a call that a dead daemon never answers.
            self.run_worker(self._watch_daemon(client), group="daemon", exclusive=False)
            chosen = await self._offer_sessions(client)
            self.front = await attach_session(
                client,
                self.session_id or chosen,
                host=self,
                cwd=self.project,
                trust=trust,
                hidden_screens=self.settings.hidden_screens,
                choice=self.model_choice,
                profile=self.profile,
            )
        except Exception as error:
            log.exception("ph_app.tui: could not attach to a daemon")
            await self._could_not_start(str(error), trust)
            return
        # A screen that a row contributed gets its verb, its key and its palette
        # entry from this call. This list releases them only when the app
        # unmounts. When a row unloads, the daemon sends the new screen list,
        # and the front end releases all three and builds them again for the
        # screens that remain (`DaemonSession._wire_screens`).
        self._command_disposers = self.front.attach_surfaces(self)
        self.state_changed()
        if self.front.arrival is not None:
            self.notify(self.front.arrival, title="profile", markup=False)

    async def _offer_sessions(self, client: DaemonClient) -> str | None:
        """The session that the user picked to resume, or `None` for a fresh one.

        **It returns the answer and does not assign `self.session_id`**, which
        it did at first. That field is documented as the input of the *CLI*. To
        write the output of a modal into it made the meaning of the field
        depend on when it was read.

        **It runs between the daemon and the attach**, which is the only place
        where it fits. The daemon folds the list, and the answer decides what
        the attach opens. Trust comes before both. The mount is what reads the
        `AGENTS.md` of the project, its hooks and its configured plugins, so a
        user who has not trusted the directory is not asked which of its
        sessions to open.

        It is legal to await a modal here because `_open` runs in a worker. This
        whole class is built around that rule.

        It is silent in the two cases that matter: a directory that nobody has
        worked in, and a run that already knows its session (`--session`,
        `--resume`, `--new`). A first run must not meet a modal that lists
        nothing.
        """
        if self.session_id is not None or not self.offer_sessions:
            return None
        try:
            sessions = await browse_on(client, cwd=str(self.project))
        except Exception:
            # A picker is a convenience. A failure to build one must not stop
            # the session that it offered to open.
            log.warning("ph_app.tui: could not list this directory's sessions", exc_info=True)
            return None
        if not sessions:
            return None
        chosen = await self.push_screen_wait(
            ChoicePicker(
                title=f"sessions in {shorten_path(str(self.project), 48)}",
                choices=session_choices(sessions, offer_new=True),
                free_text="session id",
            )
        )
        # `None` is escape and `NEW_SESSION` is the first row. They are one
        # answer, because both mean "not one of these".
        return None if chosen == NEW_SESSION else chosen

    async def _could_not_start(self, reason: str, trust: TrustAnswer | None) -> None:
        """Report that there is no session, and why. The reason stays on screen,
        and the user can answer it (H3).

        A toast used to be the whole report, and a toast lasts five seconds. It
        left a terminal with its headings drawn and nothing behind them. The
        prompt swallowed every line. `/quit` did nothing, because every slash
        command was dispatched through the front end, which had not been built.
        The user who met it read it as a hang, and the sentence that would have
        explained it was already gone.

        So the sentence stays, on the status line and under every line typed at
        the prompt, and a modal asks what to do about it:

        - **Retry** runs the whole open again. That is for the case where the
          fix is in another terminal: a daemon that would not start, or an
          install that was repaired.
        - **Quit** leaves, with the exit code that `action_quit` gives a start
          that never happened.
        - Esc stays, and the verbs of this terminal still answer (`run_line`).

        It is legal to await the modal here for the same reason as in
        `_offer_sessions`: this runs in the `_open` worker, never on the message
        pump.
        """
        await self._disconnect()
        self.refusal = reason
        self.state_changed(Surface.FOOTER)
        decision = start_decision(await self.push_screen_wait(could_not_start_modal(reason)))
        if decision == "retry":
            self.refusal = None
            self.state_changed(Surface.FOOTER)
            self.run_worker(self._open(trust), group="open")
        elif decision == "quit":
            await self.action_quit()
        elif self._prompt is not None:
            # To stay means to type next, and the close of a modal leaves nothing
            # focused. The pickers refocus the prompt for the same reason
            # (`_insert_command`).
            self._prompt.area.focus()

    async def _disconnect(self) -> None:
        """Close the connection that this app opened. That happens after a failed
        attach, so that a retry starts clean, and on the way out.

        The watcher is canceled first. It reports a daemon that *went away*, and
        a socket that this terminal closes on purpose is not that.
        """
        client, self._client = self._client, None
        if client is None:
            return
        self.workers.cancel_group(self, "daemon")
        await client.aclose()

    async def on_unmount(self) -> None:
        # **The timers stop before anything else unwinds.** Both timers call
        # into widgets. A frame that lands after the status bar has been taken
        # apart queries a node that is no longer there. The old single interval
        # needed no code here, because the app stopped it when it shut down its
        # own timers. This class owns two timers, so this class stops two.
        self._spin(False)
        if self._draw_timer is not None:
            self._draw_timer.stop()
            self._draw_timer = None
        for dispose in reversed(self._command_disposers):
            dispose()
        self._command_disposers.clear()
        if self.front is not None:
            await self.front.close()
        # **This app opened the connection, so this app closes it.** The front
        # end only detaches. A host that kept one connection across several
        # sessions, for example a browser tab for each session, must not have
        # one of them close the socket that the others are on. Without this
        # close, the daemon keeps the subscription. A subscriber is its own
        # claim on the life of a root, so nothing that the sweep does can ever
        # release it.
        await self._disconnect()
        self.title_writer.clear()

    # ---------------------------------------------------------------- frames --

    def state_changed(self, surfaces: Surface = Surface.ALL) -> None:
        """Something that the view renders has changed. Draw soon, one time, and
        only what changed.

        **This is the single entry point.** Every local mutation goes through it
        too, and does not set the flag directly. The flag alone was enough while
        a poll watched it. Now that nothing polls, a flag that is set with no
        draw scheduled leaves a pane stale permanently. That is the one thing
        that the change to coalesced draws makes worse if it goes wrong, so
        there is one place to get right.

        **The timer is a debounce, and not a frame rate.** It is a `set_timer`,
        and it is armed only when it is not already armed, so no event means no
        draw. It limits how often events *may* redraw. During a streaming turn,
        events arrive faster than that limit, and that is why `surfaces` is
        worth having. The flag used to be one boolean, so a chunk that can only
        change the transcript also redrew the sidebar at the maximum rate.

        The default is `ALL`, so a caller that has not thought about it is
        correct and not stale.
        """
        self._dirty |= surfaces
        if self._draw_timer is None:
            self._draw_timer = self.set_timer(FRAME_INTERVAL, self._draw)

    async def _draw(self) -> None:
        """Render whichever surfaces have moved since the last draw."""
        self._draw_timer = None
        front, status, view = self.front, self._status, self._view
        if status is None or view is None:
            return
        if front is None:
            # There is nothing to fold. The footer says why, and the other
            # surfaces wait for a front end. `_dirty` is left as it is, so the
            # first draw after the attach is a whole one.
            status.show_unattached(self.refusal)
            return
        dirty, self._dirty = self._dirty, Surface.NOTHING
        if Surface.FOOTER in dirty:
            # The readings are read with the footer, because they *are* the
            # right-hand half of the footer. Each one is a fold of the log. To
            # read them again for a frame that draws only the transcript would
            # be the expensive half of a draw that nobody asked for.
            self._readings = front.status_readings()
            status.show(front.state, self._readings)
        if Surface.TRANSCRIPT in dirty:
            await view.sync(self._rows(front))
        if Surface.SIDEBAR in dirty and self._sidebar is not None and self._sidebar.display:
            self._sidebar.show(
                front.state,
                # The same readings the footer got: one read, two placements.
                self._readings,
                session_id=front.session_id,
                cwd=str(self.project),
                show_tools=self.settings.show_tools,
                show_skills=self.settings.show_skills,
            )
        self._spin(front.state.busy)

    def _spin(self, running: bool) -> None:
        """Start or stop the frame clock, so it exists only while a turn does."""
        if running and self._spinner is None:
            self._spinner = self.set_interval(FRAME_INTERVAL, self._advance)
        elif not running and self._spinner is not None:
            self._spinner.stop()
            self._spinner = None

    def _advance(self) -> None:
        """One spinner frame: the glyph and the title, and nothing else.

        The glyph has a cell of its own (`StatusBar`), so a frame repaints one
        cell. `_draw` rebuilds the line beside it when a fact on the line
        changes, and every such change arrives as a `FOOTER`-dirty frame. This
        method used to rebuild the line and run the layout under it again
        thirty times a second, to move one glyph.

        It also stops itself if the turn ended and no draw noticed. That is a
        second safeguard. `_draw` normally stops the spinner, and a spinner
        that nobody can stop is the usual failure of an animation loop.
        """
        front, status = self.front, self._status
        if front is None or status is None or not front.state.busy:
            self._spin(False)
            return
        status.tick()
        self.title_writer.set(f"{status.glyph} working")

    def _rows(self, front: FrontSession) -> list[ChatItem]:
        return front.state.visible_items(
            thinking=self.settings.show_thinking, tool_results=self.settings.show_tool_results
        )

    # ----------------------------------------------------------------- turns --

    async def on_prompt_input_submitted(self, message: PromptInput.Submitted) -> None:
        if message.text.startswith("/"):
            # A command is the verb of the human. To send it as a prompt would
            # spend a turn and make the log say that the model chose it. This
            # check comes first, and before the check for a front end, because
            # the verbs of the terminal need no front end (H3).
            #
            # **It runs on a worker, like the two shell branches below** (H7).
            # This is a handler on the message pump of Textual, and a daemon
            # verb is a round trip: `/compact` spends a whole model call in it.
            # When the command was awaited here, it froze the entire UI for
            # that time, with no keystrokes, no repaint, and not even the
            # spinner. That happened for exactly the command that is meant to
            # take a while. `_dispatch_command` already reports its own
            # failures through `notify`, which is the contract of a worker, so
            # nothing else had to change.
            self.run_worker(self._dispatch_command(message.text), group="command")
            return
        front = self._attached("pH has no session")
        if front is None:
            return
        if message.text.startswith("!!"):
            # This is the shell of the user, and not a turn. `!!` spends no
            # tokens and the model never sees it. It is logged all the same. To
            # press enter is an act in the session, so every attached UI renders
            # it from the event.
            self.run_worker(self._shell(front, message.text[2:].strip()), group="shell")
            return
        if message.text.startswith("!"):
            # `!` is tested after `!!` so that the longer prefix wins: `!!ls`
            # must not read as `!` of `!ls`. `ph_app.shell` says what `surface`
            # then means.
            self.run_worker(
                self._shell(front, message.text[1:].strip(), surface=True), group="shell"
            )
            return
        if message.queue or front.state.busy:
            # The driver refuses a second concurrent `run()`. A user who types
            # during a turn means "also this", and not "instead of that".
            front.queue(message.text)
            return
        self._run_turn(message.text)

    async def _shell(self, front: FrontSession, command: str, *, surface: bool = False) -> None:
        """Run `!<command>` or `!!<command>`. The log does the rendering.

        Nothing is drawn here. The command and its output arrive as `shell/*`
        events like everything else. That keeps `TuiState` entirely derived
        from events, and lets the browser and the terminal share one fold. It
        holds for `!` too: its splice is a further event, never a draw.
        """
        if not command:
            self.notify(
                "type `!!<command>` to run one, or `!<command>` to show the model",
                title="shell",
                markup=False,
            )
            return
        try:
            await front.shell(command, surface=surface)
        except Exception as error:
            log.exception("ph_app.tui: a shell command failed to start")
            self.notify(str(error), title="shell", severity="error", markup=False)

    @verb_work("attach")
    async def action_attach(self, argument: str = "") -> None:
        """`/attach <path> …`: stage files for the next prompt.

        **This runs on the client, and that is the purpose** (I-9). The front
        end reads the bytes with the permissions of the user and hands content
        to the harness, never a path. In process, that reads this machine. Over
        a socket, it reads the machine of the *client*. That is the only thing
        that a browser tab could do, and it is the reason why a remote terminal
        can attach a file that the daemon has never seen.

        This verb takes an argument, so its body reads one: `_RunAction`
        forwards what followed the name. `/view` is the only other verb in
        `TUI_VERBS` that takes one. When the verb is chosen from the palette,
        there is nothing to attach. It then says how it is used, because
        otherwise it would do nothing.
        """
        # `shlex.split`, and not `str.split` (H7). A path with a space in it is
        # the ordinary case on a Mac: `~/Library/Application Support/...`. A
        # split on whitespace turned one `/attach "my notes.md"` into two
        # filenames that do not exist. They were reported as two failures, and
        # neither named what the user typed. A user reaches for quotes, and the
        # shell that they typed the line into would have honored them.
        try:
            paths = shlex.split(argument)
        except ValueError:
            # An unbalanced quote. The user gets their text back, and not a
            # traceback or a silent wrong split of it.
            paths = [argument.strip()] if argument.strip() else []
        front = self._attached("attach")
        if front is None:
            return
        if not paths:
            self.notify("type `/attach <path> …` to attach files", title="attach", markup=False)
            return
        try:
            staged = await front.attach(paths)
        except Exception as error:
            # This is loud. A user who attached a diagram and got a plain text
            # turn would have no way to tell that it was never sent (P7-01).
            log.exception("ph_app.tui: attaching failed")
            self.notify(str(error), title="attach", severity="error", markup=False)
            return
        named = ", ".join(one.name or one.attachment_id[:15] for one in staged)
        self.notify(f"staged for the next prompt: {named}", title="attach", markup=False)
        self.state_changed()

    async def on_prompt_input_canceled(self, _message: PromptInput.Canceled) -> None:
        if self.front is not None and self.front.state.busy:
            self.front.cancel()

    async def _dispatch_command(self, line: str) -> None:
        name = line.split()[0]
        try:
            shown = await run_line(self.commands(), line)
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

    def commands(self) -> list[CommandDefinition]:
        """Every slash command that can run here now. The palette, the completion
        list and `_dispatch_command` read it. The commands of this terminal come
        first. Then come the commands that its front end offers: one for each
        screen, and those of the daemon."""
        front = self.front
        return [*self._verbs, *(front.commands() if front is not None else ())]

    def _attached(self, verb: str) -> FrontSession | None:
        """The front end that a verb needs, or a notification that says why there
        is none.

        A user presses a verb again when it does nothing and says nothing.
        Silent returns here were part of how a failed start came to look like a
        hang (H3). `verb` is the title of the toast, so the toast reads as the
        answer to the key that was pressed.
        """
        front = self.front
        if front is None:
            self.notify(
                self.refusal or "not attached to a session yet",
                title=verb,
                severity="warning",
                markup=False,
            )
        return front

    async def _watch_daemon(self, client: DaemonClient) -> None:
        """Tell the user, one time, when the daemon goes away (H3).

        Nothing used to. The pump ended, `_until_idle` returned, and the
        terminal went on showing a spinner for a turn that no longer existed.
        The UI had no way to name this failure state. The only evidence that a
        user had was that nothing ever happened again.

        This does **not** end the app. The transcript up to the disconnection is
        worth reading, `/sessions` still works once a daemon is back, and
        killing a terminal over a socket is this row's own failure arriving from
        the other side. What the user is owed is the truth on screen.
        """
        await client.closed.wait()
        front = self.front
        if front is not None:
            front.state.stand_down()
        # **Every open ask is withdrawn, for the same reason that one is
        # withdrawn when somebody else answers it: a question that nothing can
        # receive is a control that looks live and is not.** The task that
        # awaited the modal was canceled with the read loop, and Textual leaves
        # a screen on the stack when its waiter goes. Without this loop, the
        # user is left holding a modal while the notification below says that
        # the daemon is gone.
        for ask in list(self._open_asks):
            self.withdraw_ask(ask, reason="")
        self.notify(
            "The daemon is no longer reachable. The agent keeps running; reopen to attach.",
            title="pH lost the daemon",
            severity="error",
            markup=False,
        )
        self.state_changed()

    @work(exclusive=True, group="turn")
    async def _run_turn(self, text: str) -> None:
        """One turn, in a worker. The worker is what makes the modals legal."""
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

    async def ask_approval(
        self, request: ApprovalRequest, *, ask: AskKey | None = None
    ) -> tuple[ApprovalAnswer, str]:
        decision = await self._ask(ask, ApprovalModal(request))
        return decision.answer, decision.reason

    async def ask_question(
        self, question: UserQuestion, *, ask: AskKey | None = None
    ) -> str | None:
        answer = await self._ask(ask, AskUserModal(question))
        return answer if isinstance(answer, str) else None

    async def ask_profile(
        self, name: str, listing: str, *, ask: AskKey | None = None
    ) -> ProfileDecision:
        return decision_of(await self._ask(ask, profile_changed_modal(name, listing)))

    async def _ask[T](self, ask: AskKey | None, modal: PhModal[T]) -> T:
        """Push one ask modal and wait. `withdraw_ask` can find the modal by `ask`
        while it is up."""
        if ask is not None:
            self._open_asks[ask] = modal
        try:
            return await self.push_screen_wait(modal)
        finally:
            if ask is not None:
                self._open_asks.pop(ask, None)

    def withdraw_ask(self, ask: AskKey, *, reason: str = ANSWERED_ELSEWHERE) -> None:
        """`ModalHost`: this ask can no longer be delivered (H1).

        The modal is dismissed with its own cancel value, the same one that Esc
        produces, because the answer is discarded either way: whatever made the
        ask moot did so before this call arrived. What matters is that the modal
        goes and that the user is told *why*, and does not only see it vanish.

        `reason=""` withdraws silently, for a caller that is about to say
        something better. When the daemon goes away, it takes every open ask
        with it, and one notification for each modal would bury the sentence
        that explains them.
        """
        modal = self._open_asks.pop(ask, None)
        if modal is None:
            return
        if reason:
            self.notify(reason, severity="information")
        modal.withdraw()

    # -------------------------------------------------------------- actions --
    # There is one action for each `TuiVerb`. It is reached by key, by
    # `/command`, and by `run_action`. Every picker opens with
    # `push_screen(screen, callback)` and returns. An action runs on the message
    # pump, where to await a dismissal deadlocks.

    def _pick(
        self,
        title: str,
        choices: list[Choice],
        then: Callable[[str | None], None],
        **options: Any,  # noqa: ANN401
    ) -> None:
        self.push_screen(ChoicePicker(title=title, choices=choices, **options), then)

    def action_open_screen(self, screen_id: str) -> None:
        """Open a screen that a row contributed (P4-17).

        This is one action for every registered screen. The key of the screen
        and its `/<id>` command both reach it. `build` runs here and not at
        registration, so the screen that opens is a fold of the log as it is
        now.
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
        if front.diverged:
            # A screen is built from the mirror of the log that this client
            # keeps. A mirror that refused a frame is a *prefix*, and nothing
            # says how short it is. So this says so and draws nothing. The old
            # code rebuilt the session here, and a non-contiguous log refused
            # loudly at exactly this point, which is how the last instance of
            # this fault was found. Now that the mirror is kept incrementally,
            # the refusal happens earlier, so this is the place that must ask
            # about it.
            self.notify(
                "this client missed part of the log, so a screen built from it would "
                "be incomplete — reattach to rebuild it",
                title="screen",
                severity="warning",
                markup=False,
            )
            return
        screen = definition.build(front.session)
        if isinstance(screen, Revealing) and self._view is not None:
            # The screen opens where the reader is, when it can take a position.
            seq = self._view.seq_in_view()
            if seq is not None:
                screen.reveal(seq)
        self.push_screen(screen)

    def on_reveal_seq(self, message: RevealSeq) -> None:
        """A screen asks for the transcript row behind one of its own rows.

        This is the other half of the join. The screen names a log seq, and the
        transcript knows which widget that is. The screen is popped first, so
        that the reader lands on the row and not on the screen that they left.
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

    def _screen(self, screen_id: str) -> ScreenDefinition | None:
        """The registered screen with this id, or `None`."""
        front = self.front
        return front.screen(screen_id) if front is not None else None

    def action_open_commands(self) -> None:
        self._pick("commands", command_choices(self.commands()), self._insert_command)

    def _insert_command(self, chosen: str | None) -> None:
        if chosen is not None and self._prompt is not None:
            self._prompt.area.insert(f"{chosen} ")
            self._prompt.area.focus()

    def action_open_models(self) -> None:
        front = self._attached("model")
        if front is None:
            return
        state = front.state
        self._pick(
            "model",
            model_choices(front.models(), state.model_key, state.route),
            self._set_model,
            free_text="provider/model",
        )

    def _set_model(self, chosen: str | None) -> None:
        if chosen is None:
            return
        try:
            choice = ModelChoice.parse(chosen)
        except ModelChoiceError as error:
            self.notify(str(error), title="model", severity="error", markup=False)
            return
        self.run_worker(self._choose_model(choice), group="model", exclusive=True)

    async def _choose_model(self, choice: ModelChoice) -> None:
        """Ask the daemon for `choice`, and report what it answered.

        The route changes on the daemon and not here. A terminal that set its
        own footer first would show a model that nothing runs on whenever the
        daemon refused. `/model` did exactly that before it was real.
        """
        front = self.front
        if front is None:
            return
        try:
            await front.choose_model(choice)
        except DaemonError as error:
            self.notify(str(error), title="model", severity="error", markup=False)
            return
        state = front.state
        named = "" if state.model_key is None else f"{state.model_key} · "
        route = "" if state.route is None else state.route
        self.notify(f"{named}{route}", title="model", markup=False)
        self.state_changed()

    @verb_work("sessions")
    async def action_open_sessions(self) -> None:
        """Which sessions exist: one question, asked of the harness.

        The harness folds the stored logs and the live roots, because it knows
        both. This terminal therefore reads no session file and needs no
        directory. It used to receive a path and walk it, which worked only
        while the terminal and the harness shared a machine.
        """
        front = self._attached("sessions")
        if front is None:
            return
        self._pick(
            "sessions",
            session_choices(await front.browse_sessions(), current=front.session_id),
            self._resume_session,
            free_text="session id",
        )

    def _resume_session(self, chosen: str | None) -> None:
        if chosen is not None and (self.front is None or chosen != self.front.session_id):
            self.exit(chosen)

    def action_open_history(self) -> None:
        """`/history`: the prompts that the user has sent, and where each one is.

        The cursor reveals as it moves (`on_highlight`), and that is the second
        half of the request. The transcript scrolls to the prompt under the
        cursor through `scroll_to_seq`, the same call that answers a
        `RevealSeq` from a contributed screen. The conversation therefore moves
        behind the picker while the user looks for a prompt.
        """
        front = self._attached("history")
        if front is None:
            return
        if front.diverged:
            # This is the condition that `action_open_screen` refuses on, for
            # the same reason. A mirror that missed a frame is a prefix, and
            # nothing says how short it is. A history folded from it would be
            # incomplete with no sign of it.
            self.notify(
                "this client missed part of the log, so the history folded from it "
                "would be incomplete — reattach to rebuild it",
                title="history",
                severity="warning",
                markup=False,
            )
            return
        history = front.state.prompt_history()
        if not history:
            self.notify("nothing sent yet in this session", title="history", markup=False)
            return
        self._history_rows = {str(record.seq): record.text for record in history}
        self._pick(
            "history",
            history_choices(history),
            self._insert_history,
            on_highlight=self._reveal_history,
        )

    def _reveal_history(self, value: str) -> None:
        """Scroll the transcript to the prompt under the cursor.

        There is no guard on the seq. `scroll_to_seq` already answers `False`
        for a seq that it has no row for, so the rule is written one time.
        """
        if value in self._history_rows and self._view is not None:
            self._view.scroll_to_seq(int(value))

    def _insert_history(self, chosen: str | None) -> None:
        """Put the chosen prompt in the box, to send or to edit first."""
        text = self._history_rows.get(chosen or "")
        self._history_rows = {}  # the picker is gone; so is its data
        if text is None or self._prompt is None:
            return
        self._prompt.replace(text)
        self._prompt.area.focus()

    def action_open_themes(self) -> None:
        self._pick(
            "theme",
            theme_choices(self.theme, self.catalog, self.theme_profile),
            self._set_theme,
            on_highlight=self._preview_theme,
        )

    def _preview_theme(self, name: str) -> None:
        """Apply a theme as the cursor passes it. To see a theme is how a user
        picks one."""
        if name in self.catalog.themes:
            self.theme = name

    def _set_theme(self, chosen: str | None) -> None:
        if chosen is not None and chosen in self.catalog.themes:
            # The choice goes into `$PH_HOME/themes/theme-profile.yaml`, which
            # this call creates on the first pick. A user who has never run
            # `/theme` therefore has no theme file and needs none.
            self.theme_profile = choose_theme(self.home, self.theme_profile, chosen)
        # Chosen or canceled, the theme in force is the one that the theme
        # profile names.
        self.theme = self.catalog.resolve(self.theme_profile.chosen).name

    @verb_work("permissions")
    async def action_open_presets(self) -> None:
        """Ask the daemon which postures there are, then offer them."""
        front = self._attached("permissions")
        if front is not None:
            self._pick("permissions", preset_choices(await front.presets()), self._set_preset)

    def _set_preset(self, chosen: str | None) -> None:
        front = self.front
        # The lookup *is* the membership test, and it narrows the type.
        # `chosen not in PRESETS` checked the same thing and left the value a
        # `str`, which the typed send then refused.
        name = PRESET_NAMES.get(chosen or "")
        if name is None or front is None:
            return
        front.set_preset(name)
        self.state_changed()

    @verb_work("login")
    async def action_open_login(self) -> None:
        """Ask the harness what it holds, then offer the list.

        The question must be asked before the picker is built. The answer is a
        wire call to the daemon, so somebody has to await it. That is an
        argument about *ordering*. It used to settle *who owns the pump* as a
        side effect, because `async def action_*` is the only shape that Textual
        offers.
        """
        front = self._attached("login")
        if front is None:
            return
        held = await front.refresh_credentials()
        self._pick(
            "credential",
            credential_choices(held),
            self._ask_for_secret,
            free_text="environment variable",
        )

    def _ask_for_secret(self, name: str | None) -> None:
        if name is not None:
            self.push_screen(LoginModal(name), lambda value: self._store(name, value))

    def _store(self, name: str, value: str | None) -> None:
        front = self.front
        # The value goes to the daemon, which holds it in the memory of its
        # process and never writes it down. `notify` names the credential and
        # never the secret.
        if value is None or front is None:
            return
        front.store_credential(name, value)
        self.notify(f"{name} set for this session", title="login", markup=False)

    async def action_view(self, what: str = "") -> None:
        """`/view <what>`: the one verb for what is on screen.

        Six things are behind one name, because the user asks the same question
        every time. Three of them were verbs of their own, with names that
        shared no word (`/thinking`, `/tools`, `/sidebar`). The two that the
        sidebar panels needed would have made five. The palette would then list
        five ways to say "show me less", and none of them would say so.

        The setting is decided first and acted on second. `tools`, `skills` and
        `all` are panel state and persist. `sidebar` belongs to this window and
        does not persist. `TuiSettings` already draws that distinction, and this
        keeps it.
        """
        chosen = what.strip().lower()
        if chosen == "sidebar":
            # This one belongs to this window and not to this user. It persists
            # nowhere, and there is no settings field to flip.
            if self._sidebar is not None:
                self._sidebar.display = not self._sidebar.display
            self.state_changed(Surface.SIDEBAR)
            return
        settings, rebuild = self.settings, False
        if chosen == "all":
            # Mixed means show both. A toggle over two independent flags has no
            # single correct answer when they disagree. A user who types `all`
            # at a half-hidden panel means "make it all visible". So this flips
            # when the two flags already agree, and shows both when they do not.
            #
            # The two fields are named here and not routed back through `VIEWS`.
            # To read the flags generically needed a second lambda for each row,
            # whose only job was to report the field that the first one flipped.
            # Only a test could say that the two named the same field. With two
            # panels and two fields, this is one expression that the type
            # checker reads.
            show = not (settings.show_tools and settings.show_skills)
            settings = replace(settings, show_tools=show, show_skills=show)
        elif chosen in VIEWS:
            toggle = VIEWS[chosen]
            settings, rebuild = toggle.flip(settings), toggle.rebuilds
        else:
            self.notify(VIEW_USAGE, title="view", severity="warning", markup=False)
            return
        self._save(settings)
        if rebuild:
            # The transcript is *built* from these two settings, so it is
            # rebuilt. The panels are read at draw time, so they are not. That
            # is all that the second column of `VIEWS` records.
            await self._rebuild()
        # The panels are redrawn either way. The transcript is redrawn only
        # when it was *not* just rebuilt above. The condition is written
        # positively. The first form read `SIDEBAR if not rebuild else ALL`, and
        # a reader had to invert a negation to see that the expensive branch
        # was the one already done.
        self.state_changed(Surface.SIDEBAR if rebuild else Surface.ALL)

    async def _rebuild(self) -> None:
        """Redraw from scratch, which is what a display toggle needs."""
        if self._view is not None and self.front is not None:
            await self._view.rebuild(self._rows(self.front))

    def action_list_tools(self) -> None:
        """`/tools`: the catalog, with what each tool is for.

        The sidebar carries the names because they are charged to every
        request. This carries the descriptions, which are charged to every
        request *too* and are far too long for 32 columns. The list is a
        notification and is not appended to the transcript. A front end may not
        write into a fold of the log (A11), and a notification is where the
        output of every other command already goes.
        """
        front = self._attached("tools")
        if front is not None:
            self._show_catalog("tools", front.state.tools)

    def action_list_skills(self) -> None:
        """`/skills`: what is installed, and what each skill is for."""
        front = self._attached("skills")
        if front is not None:
            self._show_catalog("skills", front.state.skills)

    async def action_quit(self) -> None:
        """`/quit`, and its key. The exit code says whether there was ever a session
        (M4). A terminal that the user leaves after a failed start did not
        succeed at anything that a script running `phern --mode tui` could rely
        on, so it does not answer `0`."""
        self.exit(return_code=1 if self.refusal is not None else 0)

    def _show_catalog(self, what: str, entries: tuple[CatalogEntry, ...]) -> None:
        """Render one catalog as a toast.

        It takes the entries, and not a field name to `getattr` from the state.
        Each of the two callers knows which tuple it means. A name that is
        looked up at run time is a rename that passes the type checker and then
        finds nothing.
        """
        if not entries:
            self.notify(f"no {what} in this deployment", title=what, markup=False)
            return
        lines = [
            f"{one.name} — {one.description}" if one.description else one.name for one in entries
        ]
        self.notify("\n".join(lines), title=f"{what} ({len(entries)})", markup=False)

    def _save(self, settings: TuiSettings) -> None:
        """Adopt new settings and persist them. If the write fails, the change
        still applies, and only the next run does not remember it."""
        self.settings = settings
        try:
            save_tui_settings(self.home, settings)
        except OSError:
            log.warning("ph_app.tui: could not write tui.json", exc_info=True)

    # ---------------------------------------------------------- completions --

    def _history_source(self) -> list[PromptRecord]:
        """Every prompt that this user has sent, for the arrow keys and `/history`.

        It is empty while the mirror has diverged. The arrows then only edit,
        and do not offer a history that is a prefix of unknown length.
        `/history` says so in a notification. A key cannot, so it declines
        silently.
        """
        front = self.front
        if front is None or front.diverged:
            return []
        return front.state.prompt_history()

    def _completion_source(self) -> dict[str, Any]:
        commands = [(d.name, d.summary) for d in self.commands()]
        return {"commands": commands, "paths": self._paths}


async def run_tui(
    *,
    daemon_argv: Sequence[str],
    session_id: str | None = None,
    spawn: bool = True,
    offer_sessions: bool = True,
    model_choice: ModelChoice = ModelChoice(),
    profile: str | None = None,
) -> int:
    """Entry point for `--mode tui`. Returns the exit code, which is `0` when
    clean (M4).

    It takes no composed profile: the daemon composes the one that
    `daemon_argv` names. `profile` is only a name, for the profile on which a
    session that this terminal creates runs (S5). The route is `model_choice`,
    which `session/new` carries for each session that this terminal opens. An
    empty route means the default of the profile, which the attach reports.

    It loops so that a session chosen from the picker is reopened. The app
    exits with the id, everything that it attached has detached, and a fresh
    app attaches to the chosen session. The *root* is untouched by that, which
    is the difference from before: to switch sessions in the picker no longer
    ends the turn that the user was watching.

    **The code is returned and not raised.** This function used to
    `raise SystemExit`. That made it the first and only exit-code mechanism of
    the CLI, because nothing else in `cli.py` calls `sys.exit`, and the next
    command that needed one would have invented a second. A library function
    that exits the process is also one that a caller cannot compose. A test, or
    a future `--mode` that runs the TUI and then does something, would have had
    to catch `SystemExit` to find out what happened. The entry point owns the
    exit, and `cli.py` translates this code into the `typer.Exit` that it
    already uses everywhere else.
    """
    while True:
        app = PHTuiApp(
            daemon_argv=daemon_argv,
            session_id=session_id,
            spawn=spawn,
            offer_sessions=offer_sessions,
            model_choice=model_choice,
            profile=profile,
        )
        chosen = await app.run_async()
        if app.return_code:
            # **The exit code belongs to the app** (H3). Textual sets
            # `return_code` to 1 when the app ended on an unhandled error. To
            # answer `0` here would report that as success. A terminal that
            # crashed and a terminal that the user quit would then look the same
            # to anything that scripts `phern --mode tui`, including a
            # supervisor that decides whether to restart it.
            return app.return_code
        if chosen is None:
            return 0
        session_id = chosen
        # The picker has already done its job one time. To reopen through
        # `/sessions` is already an explicit choice, and to offer the list
        # again on the way in would ask the same question twice.
        offer_sessions = False
