# One harness, many front ends: the TUI as a daemon client, and a browser beside it

## Context

Today `ph --mode tui` **hosts** the harness: `open_harness` (`packages/ph-app/src/ph_app/tui/frontend.py`)
mounts the profile in-process, creates the session and agent, and registers the three
front-end listeners — `session.observe` → `TuiEventAdapter`, `approval.register_answerer`,
`user_questions.register_answerer`. Close the terminal and the harness dies. Separately,
`ph daemon` mounts roots that outlive every client, and `ph agents attach` can *watch* one —
but **nothing that watches can answer.**

That last sentence has two halves and they are not equally true today, which is worth stating
precisely because it sets the priority:

* **Approvals are a live defect.** `ctx.approval.request(...)` suspends a *specific pending tool
  call* inside `agent.run()`. Under the daemon no answerer is registered, so the waterfall's
  `inner` returns `unavailable` (`ph/seams/approval.py:276-283`) and the call is **denied** —
  not because a person said no, but because nobody could be asked. Both doctors report the
  deployment as healthy.
* **Questions are, as of today, theoretical.** `ctx.user_questions.ask(...)` is pH's
  `AskUserQuestion`: free-form or multiple-choice, asked *by the harness*, turn parked until
  answered, degrading to `None`. The row is mounted (`base.yaml:118`), the TUI registers an
  answerer and ships `AskUserModal` — and **`grep '\.ask('` across `packages/*/src` finds no
  caller.** It is a seam with a definition and a consumer and no producer.

Neither is the prompt box. The composer is **push**: the person types whenever they like,
`session/prompt` → `agent.followup`, nothing waiting. Approvals and questions are the harness
**pulling** an answer out of a person with a turn suspended. The socket has push
(`session.event` notifications) and no pull. That is the gap this plan closes.

Plan rows P5-13/14/15 already name the fix and were deferred past Phase 5; this plan lands them,
gives the question seam a producer, and adds the browser.

**Decisions taken with the user (2026-09-02):**

1. **Browser UI = `textual-serve` first.** The same `PHTuiApp` runs in a browser tab as a
   terminal; layout parity is by construction. A native HTML renderer follows later on the same
   view model (recorded as P7-07, not built here).
2. **Multiplexed front ends, not a lease.** Any number of UIs attach to one session; each has a
   *private* composer (un-submitted text never leaves the client); a submitted prompt is a log
   entry everyone sees. This *removes* the takeover/demotion design entirely. The rule it needs:
   **an ask goes to every attached front end, the first answer wins, the others are told
   `ask.settled`.** pH still records exactly one decision.
3. **Web exposure: configurable bind + one-time token.** Default `127.0.0.1`; `--host` may open
   an interface; every request without the token is refused. No TLS, no users — stated.
4. **Two daemon lifetimes, decided by who started it.** `ph daemon` typed explicitly is a
   service and never exits when idle. One auto-spawned by a UI is **ephemeral** and exits when
   quiet — *unless* an appointment is indexed, in which case it stays until it fires. A UI
   silently spawns one when the socket is absent. See **Daemon lifetime**.
5. **Questions get a producer in the same effort: an `ask_user` tool.** Rather than build durable
   `question/*` events for a path nothing exercises — the "knob wired to nothing" this codebase
   declines to ship — land the tool that asks, so the whole path is real and tested end to end.
6. **An unattended run pays nothing for it.** The row ships **disabled in `base.yaml`** and is
   enabled by `tui.yaml` (which `rlm`/`rlm-stable` inherit), so a headless or daemon-only
   posture never offers the tool to the model at all — no schema, no wasted turn, no events. And
   independently, **a question is logged only when it is actually put to a person**: asked with
   nobody attending, it appends nothing and returns "no answer" at once. That covers
   `/autonomous` inside an interactive profile, where the row *is* mounted but the run is
   sandboxed and alone.
7. **`ask_user` is not itself gated by approval.** Asking permission to ask is one more
   interruption for the same person, and the ask has to happen anyway to direct the model.
8. **`!!<command>` runs the person's own shell, and it *is* logged.** Ported from tau, quiet
   variant only. The private-composer rule covers *un-submitted* text; pressing enter is an act
   in the session, so every attached UI renders it — from the event, like everything else. See
   **Increment 1f**.

**Invariants that do not move:** the log vocabulary is fixed except the two `question/*` types
added here; `derive_messages()` stays the model's view and transcripts rebuild from
`session.events`; `TuiState` stays event-derived and Textual-free; ph-core imports neither
Textual nor aiohttp (`test_layering.FORBIDDEN` gains `aiohttp`, `textual_serve`); pH records
approvals, a front end only decides; `session/attach` never replays.

## Architecture

The harness runs in **one place — the daemon.** Every interactive UI is a protocol client over
the existing unix socket (`ph_app/protocol.py`, JSONL). Events already stream as `session.event`
and the client already owns the fold (`TuiEventAdapter`, Textual-free). What is added: (a) a
**server→client request direction** so the daemon can ask a human; (b) a per-root **ask desk**
that registers both answerers and fans an ask out to every attached front end; (c) **wire
projections** for what the TUI reads from the `Context` today; (d) **attachments on the wire**
plus a per-session **staging** queue so a browser upload becomes a chip in every attached
composer; (e) a `DaemonSession` implementing the same surface `PHTuiApp` consumes; (f)
`ph --mode web` composing `textual-serve`'s handlers with an upload route on one aiohttp app.

```
 browser tab ──ws──┐  textual-serve (aiohttp, our Application)
                   │    ├─ /, /ws, /static, /download   textual_serve.Server handlers
                   │    └─ POST /api/attachments  ──► attachment/put + session/stage
                   ▼
   PHTuiApp subprocess ──┐
   terminal PHTuiApp ────┼── DaemonClient ── unix socket ── ph daemon  (service | ephemeral)
   ph agents attach ─────┘                                  ├─ Root(ctx, session, agent)
                                                            ├─ AskDesk → every front end
                                                            └─ staged attachments (per root)
```

---

## Daemon lifetime — who chose it decides

**Nothing here is a new mechanism.** P6-23 already made a restarted daemon catch up:
`Supervisor.rehydrate()` reads `$PH_HOME/schedules.json` (`ScheduleIndex`), mounts **only**
sessions whose `next_at` has passed, and `tick` fires them — with `claim` write-ahead and
coalescing, so "a machine off from Tuesday to Thursday runs Wednesday's work once rather than
twice or never", and `wake_within` declining to resurrect an appointment nobody has confirmed in
months. Session rehydration is equally old: `Supervisor.start` resumes any root whose log exists
(P5-01), which is what `passivate`'s own docstring leans on. **This section is passivation raised
one level** — passivation releases a *root*, this releases the *process*.

`DaemonServer`/`Supervisor` gain `ephemeral: bool`. `ensure_daemon` passes `--ephemeral` when it
spawns; `ph daemon` defaults `False`. `ph daemon --ephemeral` and `ph --mode tui --keep-daemon`
cross over.

**The exit predicate**, on the existing 60 s sweep cadence (`_sweeper`, `SWEEP_EVERY`) — no
fourth timer:

1. `ephemeral`, **and**
2. no open connections — not merely no *attached* ones, so a `ph agents` call in flight is never
   hung up on, **and**
3. `sweep(after=EPHEMERAL_QUIET)` leaves `supervisor.roots` empty, **and**
4. `ScheduleIndex.read()` is empty — **any** appointment keeps the daemon up.

Then `server.stop.set()`. Teardown **already unlinks the socket** (`server.py:673`, inside the
shielded shutdown after `supervisor.aclose()`), so the next client sees an *absent* socket and
starts one rather than hitting the "present and refusing" crash diagnosis.

`EPHEMERAL_QUIET = 60.0`, for both the passivation window and the exit delay — the same question
asked twice. **This interaction is the one that matters:** `PASSIVATE_AFTER` is 90 minutes, so
without an override an ephemeral daemon could not exit until 90 minutes after the last turn. The
ephemeral daemon passivates aggressively *because* it intends to leave; the service daemon keeps
90 minutes because holding a warm root is the point.

**A root parked on a human is releasable, not busy.** It reports `running` today, so
`passivatable` refuses it and an ephemeral daemon would live forever waiting on someone who
closed their laptop. P5-13 settled the direction: *"A root parked on a human is idle, and should
be released: P5-05 sweeps it, the daemon shuts down, and attaching re-asks."* So the `waiting`
status added in 1c counts as releasable. Passivating mid-ask cancels the turn; the log keeps
`approval/asked` (or `question/asked`) with no answer; `repaired()` (1b) leaves that turn **open**
rather than closing it interrupted; attaching re-poses it. Backwards, this becomes one resident
process per abandoned question.

**Spawn race.** Two UIs launching together both find no socket and both spawn. `ensure_daemon`
takes a `filelock` on `$PH_RUNTIME/daemon.lock` (already a dependency; P5-03 uses it for session
leases), re-tries the connect *inside* the lock, spawns only then, waits on `initialize`.

**Visibility.** `ph agents doctor` gains a `lifetime` row — `ephemeral` or `service`, and *why* it
is still up ("3 appointments indexed") — through P5-11's `sections` envelope, so the client needs
no change.

**Cost that becomes user-visible.** Reattaching is mount (~6.5 ms) + resume the log + take the
lease. A long log is the real term; measure before committing, and the UI says "resuming…"
rather than looking hung.

**Gates:** `test_an_auto_started_daemon_exits_once_it_is_quiet_and_nothing_is_scheduled` ·
`test_an_ephemeral_daemon_stays_up_while_an_appointment_is_indexed` (sabotage: drop condition 4)
· `test_an_explicitly_started_daemon_never_exits_when_idle` (sabotage: ignore `ephemeral`) ·
`test_a_root_parked_on_an_approval_is_released_and_re_asks_on_attach` (sabotage: treat `waiting`
as busy) · `test_two_front_ends_starting_at_once_start_one_daemon` (sabotage: drop the lock) ·
`test_a_daemon_that_exited_leaves_no_socket_behind`.

**Non-guarantees (rule 6, into `NON_GUARANTEES`):** an ephemeral daemon fires nothing while it is
down — it stays up to avoid that, but a `kill -9` or a logout reap ends both the process and the
appointment until a UI opens; a schedule created from a TUI keeps a daemon resident indefinitely;
timely scheduling on a machine that reboots still wants a systemd/launchd unit owning
`ph daemon`.

---

## Increment 0 — the seam: `PHTuiApp` stops reaching past the harness

`app.py` touches `front.ctx` in nine places (`_open`, `_screen`, `action_open_{commands,models,
sessions,presets,login}`, `_store`, `_completion_source`; `tui/app.py:238-567`) and 18 pilot-test
sites read `front.ctx`/`front.session`. Nothing over a socket can satisfy that, so the seam lands
first, with no wire change, and ships alone.

- `tui/frontend.py`: a `FrontSession` Protocol — `state`, `adapter`, `session_id`,
  `submit(text, attachments=())`, `queue`, `cancel`, `run_command`, `flush`, `close`,
  `status_readings()`, `config_rows`, `commands()`, `screens()`, `providers()`, `set_preset()`,
  `store_credential()`. `HarnessSession` implements it, behaviour unchanged.
- `tui/app.py`, `tui/commands.py`, `tui/screens.py` consume only `FrontSession`.
- **Gate** `test_the_terminal_never_reaches_past_the_front_session`: AST walk of `tui/app.py` and
  `tui/widgets/` (same shape as `test_layering.py`), asserting every attribute read off `front`
  or `self.front` is a **member of the Protocol**. Membership rather than a banned `.ctx`, because
  the class of defect is "reaches past the seam" and `front.agent` fails identically while a
  string match sees only the one name it was given — and membership makes the gate self-updating:
  adding to `FrontSession` licenses it. **Sabotage:** `front.agent` in `app.py`.

## Increment 1 — protocol prerequisites (real socket, no UI change) — lands P5-13 + the P5-14/15 wire half

**1a. One duplex end, and two framing fixes that gate everything.** *(Landed. The two fixes
became three, and the second end became the same object as the first.)* Both ends assume
"id-bearing frame = reply":
`DaemonClient._pump` (`client.py:86-96`) files every id in `_replies`; `_Connection._handle`
(`server.py:192-195`) runs `respond()` on every inbound frame. Route on `"method" in frame`
(request) vs `"result"/"error"` (reply); server-minted ids are strings `"s<n>"` so they cannot
collide with the client's ints. And `_read` awaits `_handle` inline (`server.py:169-172`) — a
`session/command` whose body asks approval would deadlock — so `_handle` runs via
`start_soon`, bounded by an `IN_FLIGHT` semaphore so "do not await the loop" does not silently
become "accept without limit"; a single writer task keeps frames unmixed.

**And both ends are now one class**, `ph_app/daemon/duplex.py:Peer`, because writing the second
copy is what exposed that the two disagreed on the case that matters: a connection dying
mid-request raised `DaemonGone` on the client and returned `{}` on the server — which reads
downstream as a successful answer with no fields, decodes to `unavailable`, and **denies the
call**, reintroducing from a new direction the exact silent denial P5-13 exists to end. `Peer`
owns the three buffers (a bounded outbox with `tell` refusing and `send` blocking, the in-flight
semaphore, the pending table) and the one thing the ends genuinely differ on is named as such:
an id-less frame is an *event to watch* to a client and a *method whose answer nobody wants* to
the daemon, and `on_notify` is which end this is. `protocol.py` keeps the stateless half and its
zero imports.

**1b. `ask_user`, and durable questions behind it.** *(Landed, except the `repaired()` change —
see the note at the end of this section.)* The producer lands with the durability, so neither is
speculative.

- New tool row `tool-ask-user` (ph-core, `ph/tools/builtin/ask_user.py` — beside the other
  model-facing rows, not in `ph/tools/`), following `tool-todo`'s shape
  (`@plugin("tool-ask-user", inject=["tools", "user_questions"])`, `ToolDefinition(name=
  "ask_user", parameters=AskUserArgs, execute=...)`). Arguments mirror `UserQuestion`:
  `question`, `options?`, `header?`, `multiSelect` (comma-joined into the seam's `str`, which is
  `AskUserModal`'s existing behaviour). Calls `ctx.user_questions.ask(...)` and returns the
  answer.
- **Shipped `disabled: true` in `base.yaml`; `tui.yaml` enables it.** The same convention
  `rlm-stable` uses for the rows its bundles ship disarmed — one place arms it, and it is the
  profile that has a modal. `headless` and any daemon-only posture never see the tool, so the
  model is never offered it, no turn is spent, no prompt tokens are paid, and nothing reaches the
  log. `--patch '{id: tool-ask-user, disabled: false}'` turns it on anywhere.
- **Not gated by approval** (decision 7): asking permission to ask is one more interruption for
  the same person, and the ask is how the model gets directed regardless.
- **A question is logged only when it is put to a person.** `UserQuestionService` gains
  `register_answerer(..., reachable: Callable[[], bool] | None = None)` and an `attended`
  property that is true when some registered answerer says it can reach someone. `ask()` returns
  `None` **without appending** when nothing is attended; only a deliverable question appends. The
  TUI's in-process answerer is always reachable — it *is* the person's screen; the daemon's
  `AskDesk` is reachable iff `front_ends` is non-empty. This is the mechanism that keeps
  `/autonomous` quiet inside an interactive profile, where the row is mounted but the run is
  sandboxed and alone, and it is the one new API surface in this increment.
  *Rule 2 ("log first, act second") is respected:* reachability is decided **before** the ask is
  committed, then the append precedes the waterfall exactly as `_record_asked` does.
- Because the failure mode differs from an approval's — the seam's own docstring: *"an approval
  is a one-shot yes/no about a specific pending call and must fail closed, while a question is
  free-form and its failure mode is 'no answer'"* — a question **never parks unattended**. Asked
  while somebody was attached and then abandoned, it stays pending and re-poses on attach; asked
  with nobody there, it never existed.
- **Nobody attending is a result, not an error** — the tool returns a sentence the model can act
  on ("nobody is attending this run; proceed without an answer"), the way `media_pointer_text`
  does, so `ph -p` degrades instead of hanging.
- `question/asked {askId, question, options, header, multiSelect}` and
  `question/answered {askId, answer}`, appended by `UserQuestionService.ask(question, *,
  session=)` around its waterfall, mirroring `_record_asked`/`_record_decided`
  (`approval.py:310-352`). Add to `KNOWN_SESSION_EVENT_TYPES`, `tui/adapter.py HANDLERS`, and
  `tui/trajectory.py` — the vocabulary tests force both. Both types are **ignorable**, unlike
  `approval/*`: an approval's decision can carry substituted arguments and so changes what ran,
  while a question's answer reaches the model only as the `ask_user` tool result, which is a
  `tool/result` either way. The `askId` is the **tool call id**, so one string joins all four
  records of one exchange — and, more to the point, a minted counter could restart at 1 after a
  resume and answer a question the log was still holding open. It rides on `UserQuestion.ask_id`
  rather than beside the question, so the log's key, the wire frame's and the one a re-posed
  question would be recognised by cannot drift apart. Export `pending_questions(session)`
  beside `pending_approvals`. `repaired()` was to learn to tell a **parked** turn (open turn +
  pending ask) from an **interrupted** one and leave it open — **not done, deliberately**; see
  below.
- **Gates:** `test_the_model_can_ask_the_person_a_question_and_read_the_answer` (fake provider
  emits an `ask_user` call; the answerer replies; the tool result carries it; the log has
  asked+answered) · `test_a_question_nobody_is_there_to_answer_writes_nothing_to_the_log`
  (sabotage: append unconditionally → two events appear) ·
  `test_a_question_asked_of_someone_who_walked_away_re_poses_on_attach` ·
  `test_only_a_profile_with_a_screen_offers_the_model_ask_user` (read off `--dump-config`, both
  profiles and both *layers*; sabotage: enable the row in `base.yaml`, or drop the patch from
  `tui.yaml`) · `test_a_question_cancelled_mid_answer_stays_pending` (a real cancellation, which
  also pins that `ask`'s `except Exception` does not swallow one into a false `declined`) ·
  `test_a_daemon_with_no_front_end_does_not_log_a_question` (sabotage: register the desk's
  answerer without `reachable`) · `test_the_wire_ask_id_is_the_one_the_log_wrote`.

**Deferred out of 1b, with the reason:** `repaired()` still closes a parked turn as interrupted.
Leaving it open is only *safe* once something re-poses the ask on resume, and nothing does yet —
the ask lives in `AskDesk` memory, and the agent loop resumes from a prompt rather than from an
open turn. Worse, the parked case leaves the model's **`tool_use` block unanswered**: it rides the
*assistant message*, and a message carrying one with no matching `tool_result` is a log several
providers reject outright — the exact failure `repair.py` exists to prevent. (An earlier draft
blamed a dangling `tool/call`; that event is not surface-eligible and no provider sees it. P7-15
moved it to after the gate, so a parked turn has none and repairs as not-started.) Closing it is the honest behaviour until the resume half lands;
the question is still in the log, `pending_questions` folds it, and the daemon's non-guarantee
row now says out loud that nothing reads that fold on resume. Re-posing across a *restart* is
what P5-13's repair half closes.

**1c. Server→client asks.** Frames `approval/ask {sessionId, askId, request}` and
`question/ask {…, question}`; the client replies in `respond`'s envelope — **reuse `respond` on
the client verbatim** by giving `DaemonClient` a `handlers: dict[str, Dispatch]`. `askId` is the
key `pending_approvals` already uses (`callId or toolName`, `approval.py:203`), so a re-posed ask
is recognisable.

New `ph_app/daemon/frontend.py` (ph-app, because it knows connections; seams stay
transport-free):

```python
class FrontEnd(Protocol):           # implemented by _Connection
    async def ask(self, method: str, params: dict) -> dict: ...
    def tell(self, method: str, params: dict) -> None: ...

@dataclass(slots=True)
class AskDesk:
    root: Root
    front_ends: set[FrontEnd]
    asks: dict[str, PendingAsk]     # askId → (method, params, answered: anyio.Event, result)
    def attach(self) -> list[Disposer]   # both register_answerer calls on root.ctx
    def join(self, who: FrontEnd) -> None    # re-poses every pending ask to the newcomer
    def leave(self, who: FrontEnd) -> None
    async def answer_approval(self, request, _next=None) -> ApprovalAnswer
    async def answer_question(self, question, _next=None) -> str | None
```

Fan-out: every front end is asked; the first reply settles; later replies are refused
`ask_settled`; the others receive `ask.settled {sessionId, askId, by}`. **No front end → the ask
sits in `asks` and the answerer awaits it** — P5-13's parked state, which survives the daemon
stopping because the log holds the ask with no answer. Turn cancellation propagates through that
await. The `reason` steer mirrors `frontend.py:219-228`. Wired from `Supervisor._start` after
`agents.create` (`supervisor.py:468`), disposers on `exits`. **Answering is a capability the
client declares at `initialize`**, not a flag on each attach: `asks` is the same name the daemon
uses to offer the direction, because it is one feature and each half is useless alone, and
whether a UI can put a modal in front of a person does not vary by which root it is watching —
a per-attach flag let one client answer yes for one session and no for another, two answers to a
question with one. `_Connection.serve`'s `finally` calls `desk.leave`. A `WouldBlock` sending an
ask means that client cannot keep up → `leave`; the ask stays pending.

Also here: `Root.status` gains **`waiting`** (derived: `desk.asks` non-empty), treated as
**releasable** by `passivatable` — see *Daemon lifetime*.

**Gates:** `test_every_attached_front_end_is_asked_and_the_first_answer_wins` (two clients; fire
`root.ctx.approval.request(...)` as `test_tui_frontend.py:55-58` does; both receive the ask; one
answers; the other gets `ask.settled`; the log has one `approval/decided`) ·
`test_an_ask_with_nobody_attached_is_posed_to_whoever_attaches_next` ·
`test_a_late_answer_is_refused_not_recorded` · `test_a_front_end_that_vanishes_mid_ask_is_
dropped_not_answered_for` (sabotage: return `{}` for a dead connection) ·
`test_answering_is_declared_once_for_a_connection_not_per_attach` (sabotage: read the capability
off the attach frame) · the same three for `ask_user`. **Sabotage:** resolve on disconnect.

**1d. Projections** *(landed; its two follow-ups P7-11 — seams describe their own wire form — and P7-12 — a `Frame` through the handler table, validation at the wire edge — have landed too)* (replacing the nine `Context` reach-throughs):

- `session.status` params gain `readings: [{text, level}]`. `StatusField.read(session)` is a fold
  of the log and changes only on append, so push on change (computed in `relay`,
  `supervisor.py:494`, only when subscribers exist, sent only when different); pollable
  `session/readings` for the attach moment. **The TUI's 30 Hz `_tick` is gone**: it was a poll
  watching a flag that was already event-set, so the first change now schedules one draw a frame
  later and every change until then rides it — same bounded latency, nothing running while nothing
  happens. That matters here rather than in the terminal, because increment 3 runs *one Textual
  subprocess per browser tab* and ten idle tabs polling a flag is ten processes waking thirty times
  a second to find it unchanged. The spinner keeps a clock, since it advances on wall-clock time
  rather than on anything arriving — but only while a turn does. The footer is cached between
  draws for the same reason readings are pushed rather than polled: a reading is a fold of the log,
  so a frame could only ever recompute the same answer.
  The hazard is that a poll *forgives* a missed notification and nothing does now, so
  `state_changed` is the single entry point and an AST gate holds it there
  (`test_every_state_mutation_schedules_a_draw`).
- `daemon/config` → `{rows: supervisor.profile.dump()}`; feeds `credential_choices`.
- `commands/list {sessionId}` from `root.ctx.commands.list()`; `session/command {sessionId, line,
  clientId, commandId}` → `{shown}`, running `ctx.commands.dispatch(line, scope=root.agent.ctx,
  session=, agent=)` as `frontend.py:120` does, idempotent via `root.remember`. The TUI's own
  verbs (`TUI_VERBS`) stay client-side; the client merges.
- `screens/list {sessionId}` → `[{id, label, order, key}]`. `build()` still runs **in the client**
  against the session it rebuilt from its snapshot — enough for textual-serve, since it *is* the
  TUI. P5-15's declarative body is deferred to P7-07 and said so.
- `tools/list {sessionId}` — copy `rpc_mode.py:69-73` against `root.ctx`.
- `session/new` gains `cwd` → `SessionHeader.cwd` (validator demands absolute; verify
  `store.create(meta=)` reaches the header). `session/preset`, `session/credential` — the value
  never logged; extend `test_login_stores_a_secret_without_logging_it`.
- **Presentation sidecar:** the adapter needs `ctx.tools` only for `present_call/present_result`.
  The daemon renders and attaches the view when relaying and when paging snapshots (`cards.py`);
  `tools: Any` becomes a `CardPresenter` Protocol with `RuntimePresenter` (in-process) and
  `FramePresenter` (reads the sidecar). Derived — never appended, and **beside** the event rather
  than inside it, which `_EventWire`'s `extra="forbid"` enforces rather than merely permitting.
  The wire shape is `ToolCallView`/`ToolResultView` `to_wire()` verbatim rather than a fourth
  hand-written `{title, subtitle, card, inputText}`: the models already are the statement of what
  a card may set, and `FramePresenter` validates against them, so a field added to a view reaches
  a browser tab without anyone editing a projection.
  **Where `present_result`'s arguments come from is the subtle part**: a `tool/result` does not
  carry them (`tool/call` recorded them before the body ran, B4), and the link is already in the
  log — `batch.py` appends the result with `source_event_seqs = (call_seq,)`. So it is one lookup,
  not a scan, which is what makes it affordable on a path that runs per streamed chunk. That
  needed `Session.at(seq)` — O(1) by A1, since `seq` *is* the index — because both obvious
  spellings are wrong for a per-event caller: `events[seq]` rebuilds the whole-log cache and
  `events_from(seq)` copies the entire tail.

**Gates** (all in `test_daemon_projections.py`, which registers its own status field and command
into the root so the assertions are about the *projection* rather than about which rows the
daemon's profile happens to mount — an empty list equals an empty list and proves nothing):
`test_status_readings_over_the_wire_equal_the_seams_readings` (against `ctx.tui_status` in the
same mount; sabotage: drop `level`) · `test_readings_ride_the_status_notification` (sabotage: send
`session.status` without them) · `test_the_command_list_is_the_registrys_own` ·
`test_the_screen_list_carries_what_a_palette_needs_and_no_body` ·
`test_the_tool_list_matches_what_the_deployment_offers` · `test_the_config_rows_are_the_composed_profile` ·
`test_a_command_runs_in_the_root_and_lands_in_its_log` ·
`test_running_a_command_twice_with_one_id_runs_it_once` (sabotage: drop the `accepted` check) ·
`test_a_credential_is_stored_without_its_value_reaching_the_log_or_the_reply` ·
`test_a_new_session_records_the_clients_cwd_in_its_header` ·
`test_a_relative_cwd_is_refused_rather_than_resolved` ·
`test_a_relayed_tool_call_carries_the_view_the_tool_would_have_rendered` (a real turn through the
loop, because the thing under test is the `source_event_seqs` link only the real append path sets;
sabotage: stop attaching `presentation`) · `test_a_snapshot_page_carries_the_same_views_as_the_live_stream` ·
`test_an_event_that_is_not_a_card_carries_no_view` · `test_the_frame_presenter_reads_what_the_daemon_rendered`.

**Mutations are a table too** (`MUTATIONS`, beside `PROJECTIONS`): every method that changes a root — `session/prompt`, `session/command`, `session/stage`, `session/preset`, `credentials/store` — is a `Mutation(prepare, act)` and one wrapper in `_dispatch` resolves the root through `start` (acting on a passivated session brings it back), runs `prepare` (which may refuse), claims the client's idempotence key with `Root.once`, then runs `act`. The order is the point: a refusal never consumes a retry, and a crash between the claim and the effect re-runs rather than loses. One repeat reply for every verb, `{**describe(), repeated: true}`. `attachment/put` is deliberately not a row — content-addressed, so a retry is already a no-op, and its reply *is* the reference a repeat must still return. Gates in `test_daemon_mutations.py` are parametrized over the table, so a row added tomorrow is covered the day it is added; a structural test pins `CASES == MUTATIONS` and `MUTATIONS ∩ PROJECTIONS = ∅`.

`CAPABILITIES` gains `"projections"`. One name differs from the sketch above: held-ness is
`session/credentials` (plural, a batch) because the picker asks about every name at once and the
per-name spelling walked the scope chain per name. Its reply is a `{name: bool}` map with no field
a secret could travel in, which is what makes "never the value" structural rather than remembered.

**1e. Attachments on the wire + staging.** *(landed)*

- `attachment/put {name, mime, contentB64}` → `AttachmentRef.to_wire()` via
  `store.save_bytes(...)`. **The client reads the bytes** — the human door (I-9), with the
  person's permissions, and the only path a browser can take. `MAX_LINE` is 8 MiB
  (`framing.py:30`); refuse over 5 MiB with `attachment_too_large` naming the limit. Chunking
  deferred and stated.
- `session/stage {sessionId, attachment}` → per-root `staged: list[AttachmentRef]`; notification
  `session.staged` to every watcher, so an upload becomes a chip in *every* attached composer.
  **Not in the log**, deliberately: composer state — un-submitted intent — shared but transient,
  lost on daemon restart with the blob still in the store. Rule 6 note.
- `session/prompt` gains `attachments: [AttachmentRef]` and drains `staged`; checks
  `store.exists(ref)` → `attachment_unknown`; builds via `prompt_message(text, refs)`
  (`attach.py:54`), replacing the bare `create_user_message` at `supervisor.py:990`.
- TUI verb `/attach <path> …` → the same two methods, so both UIs attach one way. Intercepted in
  the app rather than registered as a `ctx.commands` entry, and the reason is the same one that
  puts the read on the client: a command body runs where the profile is *mounted*, so `/attach`
  there would read the daemon's filesystem and quietly attach the wrong file — or nothing. It is
  the one verb that needs an argument, so its `action_attach` says how to use it rather than doing
  nothing from the palette.
- **What is deferred, and stated**: the composer *chip*. `session.staged` is broadcast and every
  client can read it; drawing it is a widget change that belongs with increment 4, whose gate is
  the browser upload appearing in the terminal's composer. Until then a `/attach` is confirmed by
  a notification and the file rides the next prompt.
- **Gates** (`test_daemon_attachments.py`): `test_a_prompt_over_the_socket_carries_its_attachment`
  (asserted on the `user/message` the log kept, because a reply saying "accepted" would pass for a
  daemon that dropped the reference) · `test_a_staged_attachment_rides_the_next_prompt_from_any_client`
  · `test_an_attachment_this_deployment_never_stored_is_refused` (sabotage: skip the `exists`
  check) · `test_staging_reaches_every_attached_front_end` (sabotage: return the list without
  publishing) · `test_a_staged_attachment_rides_one_prompt_and_not_the_next` (sabotage: read the
  tray without draining it) · `test_the_tray_is_not_in_the_log` ·
  `test_a_file_too_large_for_a_frame_is_refused_by_name` · `test_the_same_file_twice_is_one_blob` ·
  `test_the_client_sends_content_and_gets_back_a_reference`.

`CAPABILITIES` gains `"attachments"` and `"staging"` (`"asks"` and `"projections"` landed with 1c
and 1d); `PROTOCOL_VERSION` stays 1 — clients read the block. The size limit is checked on the
**encoded** length before decoding: base64 is 4/3 of the bytes, so decoding first to measure would
be the allocation the refusal exists to avoid.

## Increment 1f — `!!`: the person's own shell — new row P7-10 *(landed)*

`!!<command>` in the composer runs a shell command in the session's workspace and appends what
happened. Ported from tau (`session.py:2712-2751`) with one deliberate narrowing: tau ships a
*pair* — `!` splices the output into the model's context, `!!` does not — and pH takes only the
quiet one. Someone who wants the model to see it pastes it.

**Logged, and not model-visible — and pH's filter is type-level, which decides the shape.**
`derive_messages()` walks the *surface*, and `_surface_op_of` (`session/surface.py:126`) makes
surface membership a property of the **event type**: a surface-eligible type (`user/message`,
`assistant/message`, `tool/result`) **must** carry a `surfaceOp` and both ops — `append` and
`replace` — put it on the surface; a non-eligible type **may not** carry one and is therefore
invisible to the model by construction. So "log it as a `tool/result` and filter it out of the
context" is not expressible: `SurfaceError` refuses it. Dedicated `shell/*` types get the same
outcome through the mechanism pH already has, and **rendering is unaffected** — the adapter
renders any type as a `ToolCard`, and Code Mode's `terminal` card kind already exists, so it
looks like a tool call in both UIs.

There is a second, harder reason not to borrow the tool types even if the surface allowed it: a
`tool/result` derives a `Message` that providers expect to pair with a `tool_use` block from an
assistant message. A human's shell command has none, so an on-surface tool result is an
**orphan** — an error at Anthropic. That is why tau's `!` splices a `UserMessage` carrying the
command and output as *text* (`session.py:2734-2743`) rather than a tool pair, and it is what
`!` would have to do here if it is ever added.

I3 is untouched throughout: it says model-visible implies logged, and this is the converse.

*Recorded, not built:* a third `SurfaceOp` — "on the log, off the surface" — would be the general
form of dsh's context filter and would let an eligible type be hidden per event. Nothing needs it
(type eligibility already gives `!!` its outcome) and it would make a two-valued field
three-valued for every consumer, so it waits for a second caller.

**Everyone sees it, and that is the point.** The private-composer rule is about text you have not
sent. Pressing enter on `!!ls` is an act in the session, so there is **one** rendering path and
not two: the person who typed it reads it back off the same event as everybody else, and
`TuiState` stays 100 % event-derived — which is what lets the browser and the terminal keep
sharing one fold.

**Two events, because rule 2 is "log first, act second."** `shell/command {command, cwd}` is
appended *before* the command runs; `shell/result {exitCode, ok, output, truncated}` after. A
`!!` that hangs, or that takes the daemon down with it, still shows in the log what was started —
and that is exactly the command worth knowing about. One event appended on completion would lose
it. `tool/call`/`tool/result` is the shape this mirrors.

**Where it runs:** `ctx.shell.run(command, agent=root.agent)` in the root's context, so it
inherits the workspace cwd and the containment tier the session already has. Not the client's
process — a browser tab has no shell, and "the session's shell" is the honest meaning.

**Method:** `session/shell {sessionId, command, clientId, commandId}` → `{exitCode, ok}`, with
the *output* arriving as events rather than in the reply, so every attached UI gets it the same
way and by the same route. Idempotent through `root.remember`, like `session/prompt`.

**Vocabulary:** both types into `KNOWN_SESSION_EVENT_TYPES` — **ignorable**, since a reader
skipping them loses the account of what a person poked at and not the conversation — plus
`tui/adapter.py HANDLERS` (a `ToolCard` row reusing the `terminal` card kind Code Mode already
renders) and `tui/trajectory.py`. The two vocabulary tests force both.

**Access:** anyone holding the web token, deliberately — the same authority the terminal already
has, where a person can approve any tool call the model makes. Into `NON_GUARANTEES`.

**Secrets in the output: already handled, and worth knowing why.** Output is persisted, so the
obvious worry is `!!env` writing credentials into the log for good. It does not: `ctx.shell.run`
passes `env=scrub_env(...)`, which drops every name matching
`KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL` from the inherited environment
(`seams/subprocess.py:53`), and `env=None` means "scrubs and inherits" rather than "inherits". The
harness reads a provider key through `ctx.credentials` and never puts it in the child's
environment, so it is not there to be printed. **No scrubbing of shell output is needed**, and
adding it would be a second mechanism for a hazard the first one already closed.

What stays true and belongs in `NON_GUARANTEES`: a command that reads a secret from somewhere pH
does not know about — a file, a keychain, a fetched token — puts it in the log like any other
output. `!!` is the person's own shell with the person's own reach, and the log keeps what it is
told.

**Gates** (`test_daemon_shell.py`): `test_a_shell_command_is_logged_and_never_reaches_the_model`
(`derive_messages` byte-identical across it; sabotage: add `shell/result` to the surface-eligible
set — it fails) · `test_the_command_is_logged_before_it_runs` — **observed mid-flight**, because
the obvious form does not hold: asserting the two events' relative order passes for a daemon that
appends both on completion, so the child blocks on a file the test controls and the log is read
while it is still running (sabotage: append both after `run` — the wait times out) ·
`test_a_shell_command_from_one_ui_appears_in_every_attached_one` ·
`test_a_shell_child_never_inherits_a_credential` (over `ctx.shell`, pinning `scrub_env`'s reach
from the path `!!` actually takes) · `test_the_output_and_the_exit_code_reach_the_log` ·
`test_an_empty_command_is_refused_rather_than_run`.

The append/run/append body lives in `ph_app/shell.py`, the sibling of `attach.py`, called by both
front ends — the twin the first draft had was already diverging inside one increment. The event
carries **facts, not a rendering**: `stdout`, `stderr`, `exitCode`, `cwd` and `confinedBy` stay apart
on the log the way `BashValue` keeps them apart for the model, and `shell_body` renders one column
at the edge, shared, so `!!make` and a model's `bash("make")` read alike. `shell/result` cites its
command by `commandSeq` — the shape every other pair in the fold uses — because "the one in flight"
is a single-composer assumption on a deliberately multiplexed session. `session/shell` is a `MUTATIONS` row, so it inherited the write-ahead idempotence guard and the
`start`-resolution without spelling either — which is what that table is for, and the parametrized
mutation gates covered it the moment it was added. `CAPABILITIES` gains `"shell"`; two rows join
`NON_GUARANTEES` (what `!!` puts in the log, and who may run it).

## Increment 2 — the TUI switch-over — lands P5-14

*(Landed. 2a: the daemon lifetime, `ensure_daemon`, `DaemonSession`. 2b: the app
attaches instead of mounting, and every pilot test drives it over a real socket.
2c: `open_harness` and `HarnessSession` are gone. See the notes at the end of
this section for what the switch-over turned up.)*

- New `tui/remote.py`: `DaemonSession(FrontSession)` over `DaemonClient` — subscribe →
  `session/snapshot` pages → live, through `daemon/follow.py::Followed` — the buffer-then-live,
  dedupe-by-`seq` core that `ph agents attach` had already written, lifted so there is one copy
  and one invariant tested from both callers; `TuiEventAdapter(tools=None)` fed `Frame(view=view_of(...))` from the sidecar
  the daemon already sends (`presentations`, sparse, keyed by seq);
  `handlers["approval/ask"]` and `["question/ask"]` → `host.ask_approval`/`ask_question` **in a
  Textual worker** (`push_screen_wait` is illegal on the pump task); `session.status` → status +
  readings; `session.staged` → composer chips.

  Four things it settled that the plan had left open:

  * **One status field, and a bool the widgets actually read.** `TuiState.status` now carries
    the root's own word — `idle`, `running`, `waiting`, `retrying`, `passivated` — and
    `TuiState.busy` drives the spinner. The first cut kept a parallel `root_status` on the
    remote session, mapped down to a two-valued display type; that was three writers of one
    fact, and an in-process screen that could never show `retrying` because its type had no
    room for it. Every widget read `status == "running"`, so `busy` was the bool they wanted.
  * **Verbs are merged, and `run_command` routes on which side owns the name.** `/model` and
    `/theme` change *this* client's display and mean nothing to a daemon serving three of them;
    the harness's own commands come from `commands/list` and go back as `session/command`. The
    local half is built from `TUI_VERBS` on demand — there is no local registry to register into.
  * **A screen's `build` cannot travel**, which `ScreenDefinition` already said. `screens/list`
    names what the deployment mounted and `LOCAL_SCREENS` names what this build can draw; the
    intersection is what a person is offered, and `trajectory` (every screen pH ships) is in it,
    built client-side with `sessions=None` — a state the screen already supports, costing the
    fork action exactly as `ph trajectory <file>` already does. *Not enforced:* a screen a
    third-party row contributes reaches a remote front end as nothing at all. P7-07.
  * **`flush()` is a no-op and `close()` only detaches.** In process, flush is what stops a
    session being lost on exit, because the terminal *is* the harness; here the daemon owns the
    log and flushes on its checkpoint policy, on passivation and in teardown. A client asking
    for a flush would be asking for a guarantee it neither provides nor can verify.

  Wire additions it needed: `session/cancel` (deliberately *not* a `MUTATIONS` row — cancel is
  idempotent by construction, and a key would make an honest retry answer `repeated` and leave
  the turn running) and `sessionsDirectory` on `daemon/config` (the daemon knows where sessions
  live, because it mounted the profile; a daemon-level fact, so not a per-root projection — and
  the persistence seam grew `directory()` so neither front end walks `locate().parent.parent`). Plus `DaemonClient.mutate`, which stamps the
  `clientId`/`commandId` every `MUTATIONS` verb needs — `prompt` was the only caller that had
  it, and five other verbs are in that table.
- New `daemon/launch.py`: `ensure_daemon()` per *Daemon lifetime* — lock, re-check *inside* it,
  spawn, wait on the **connect** rather than on the socket file (the file exists before `serve`
  is listening, which is exactly the window a `path.exists()` poll lands in); `spawn=False`
  refuses with "run `ph daemon`"; a present-but-refusing socket keeps today's crash diagnosis
  (`agents.py:96-139`) because this function stops at "something is listening". The argv is
  `sys.executable -m ph_app` and not a bare `ph` — a `PATH` lookup can find a different install,
  an older version, or nothing at all in a checkout with no console script, and a daemon composed
  by a different pH than its client is one whose profile, event vocabulary and wire version
  nobody chose. That needed a new `ph_app/__main__.py`. Measured: a spawned daemon answers in
  ~0.24 s, so `SPAWN_TIMEOUT = 30` is slack rather than a wait.
- **Trust:** the daemon mounts, so the daemon *enforces* — `session/new {cwd, trust}` consults
  `TrustStore` and refuses `untrusted_project` unless `trust: "once"|"always"`; the UI still
  *asks*, with the same modal. **Gate** `test_the_daemon_refuses_to_mount_an_untrusted_project`;
  **sabotage:** enforce only client-side.
- Picker: `tui/sessions.py merge(stored, live)` → **running | passivated | stored** (live from
  `sessions/list`; passivated = a stored log whose tail is `supervisor/passivated` with no live
  root). `cwd` now populated. **Gate**
  `test_the_picker_tells_running_from_passivated_from_stored`.
- Sequencing inside the increment: **2a** `DaemonSession` behind `PH_TUI_DAEMON=1`, in-process
  default; **2b** flip the default and point `make_tui_app` (`tests/conftest.py:17`) at an
  in-process daemon via `daemon_helpers.running(...)`, exposing `root =
  daemon.server.supervisor.roots[id]` so the 18 pilot sites become `root.ctx`/`root.session`
  (mechanical); all 10 `.raw` snapshots and `test_tui_frontend.py` (now `DaemonSession` +
  `StubHost`) pass unchanged; **2c** delete in-process `open_harness`.
- **The gate this plan is named for:**
  `test_a_turn_started_in_the_tui_finishes_after_the_tui_is_gone` — slow fake provider,
  `pilot.press` a prompt, `app.exit()` mid-turn, assert `root.status == "running"`, open a second
  `PHTuiApp(resume=id)`, `until(turn_done)`. P5-01's gate, driven through the thing it names.
  **Sabotage:** `DaemonSession.close()` cancels the turn.
- **Multiplex gate:** `test_two_tuis_on_one_session_share_the_log_and_not_the_composer`.

**Landed in 2b/2c.** `PHTuiApp` takes `daemon_argv` and no `Profile` — the
terminal no longer knows what a profile is, which is the honest consequence of
the daemon doing the mounting. `run_tui` threads the argv from `cli.py`, and
`--no-spawn` reaches it. The test fixture starts an in-process daemon on the
resolved socket, so `ensure_daemon` finds it listening and no test ever spawns a
process; what was `app.front.ctx` is `root_of(tui_daemon).ctx`, one side of a
socket from the screen asserting on it. All 10 `.raw` snapshots pass unchanged —
the layout is identical over the wire — and the one sync snapshot test drives a
daemon in a *thread*, because `snap_compare` owns its own event loop and the
socket is the only boundary between them.

Six things the switch-over turned up that the plan had not:

* **`Root.describe` now carries `provider` and `model`.** A client could not know
  its own route until the first turn, because the only other statement of it is
  `request/context` — appended when a request is built. "no model" in a footer
  where the in-process one said `fake-1` was the whole of a snapshot diff.
* **The `seen` sentinel was `0`, and seq 0 is a real event.** So the opening
  event of every session with no history was silently dropped — invisible until
  `Session(seed=…)` refused a log that did not start at 0, which only happens
  when somebody opens a screen. `ph agents attach` had the same latent bug.
* **Catch-up was folded as *live*.** A page of history holds a turn's
  `assistant/chunk` records *and* the `assistant/message` that superseded them,
  so a fold told they were live builds a streaming row and replaces it in one
  pass — a widget mount inside a widget mount. `Followed`'s sink now takes the
  phase; `Frame(live=False)` has existed for this since P3.
* **`Edited` and `Responded` could not cross the socket at all.** They are frozen
  dataclasses, so a front end putting one in a frame raised `TypeError` *inside*
  the task group answering the ask — which the desk reads as "this front end
  cannot answer" and drops it for. P4-05's whole point was unreachable under the
  daemon. The seam grew `answer_to_wire`, the inverse of the decoder it already
  had.
* **A projection has to be re-sent when it changes.** The command list was a
  snapshot taken at attach, so a row mounted mid-session added a verb no remote
  palette would ever show. `CommandRegistry.observe` + a `session.commands`
  notification, the same shape `session.status` already has.
* **Whoever opens the connection closes it.** `close()` only detaches — a host
  keeping one connection across several sessions must not have one session close
  the socket the others are on — so the *app* closes the client it made.
  Without that the daemon kept the subscription, and a subscriber is its own
  claim on a root's life, so nothing could ever release it.

Also landed: `TrustStore` moved out of `tui/modals/` and `session/new` enforces
it (`untrusted_project`), because the daemon is what reads a project's
`AGENTS.md`; the picker merges `sessions/list` with the stored logs so a session
started moments ago — whose log is still in a write buffer — is in the list;
screens get their verb, key and palette entry on the remote path through a
`screen_routes` lifted out of the in-process presenter.

**One thing deliberately not done.** A screen contributed by a third-party row
reaches a remote front end as nothing at all —
`test_a_screen_this_build_cannot_draw_is_not_offered` is that limit as a gate —
because `ScreenDefinition.build` cannot travel; P7-07's declarative bodies are
what close it. The picker's third state, `passivated`, was dropped for a reason
that survived review: telling it from `stored` means reading the *tail* of every
log, and nothing a person then does differs.

**What the cleanup pass changed, because the review found the switch-over had
left work half-done.** Deleting `HarnessSession` orphaned `present_screens`,
`_Presenter` and `register_tui_commands` — all three had exactly one caller and
were dead, while `screen_routes` was extracted "because there are two callers
now" when there was one. Deleted; the remote front end builds a screen's verb,
palette entry and key itself.

* **Two registries, one mechanism.** `CommandRegistry` had grown a hand-rolled
  observer (a callback list, a per-listener `try`, two entries in the P6-12
  ownership tables). `ph.tools`'s `tools/change` is the precedent: a declared
  cordis `emit` and `ctx.on`. Both registries now do that — `commands/change`
  and `screens/change` — and `claim_key` gained a `then` hook, because the
  announcement has to run **after** the removal and **inside** what the owner
  holds. That ordering is the whole finding: `present_with` looks like the right
  hook for screens and is not, since its per-screen undo is a disposer on the
  registering row's scope and unwinds *before* `claim_key`'s removal — so a
  presenter told "this screen is going" reads a table that still contains it.
  With the event, `test_unloading_the_row_takes_the_verb_and_the_key_with_it` is
  back, over a socket.
* **The login screen was broken.** Moving the credentials fetch out of
  `attach_session` left `refresh_credentials` with no caller, so
  `credential_held` answered `False` for everything and a deployment with every
  key set looked like one with none. No test noticed. `action_open_login` is
  async now and asks first — the shape the session picker already used — and the
  login test asserts the marker.
* **Dead threads pulled.** `PHTuiApp.provider`/`.model` were write-only across
  five files; `resume` and `session_id` became one field, since `session/new`
  resumes an existing id through the same call that creates one; `"from"` on the
  attach reply had no client left. `readings` now ride that reply beside the
  status they belong to, which removed a fourth parallel fetch and a
  hand-assembled frame shape no wire message has.
* **`--mode tui` composed a profile it threw away** — a full plugin import per
  start — and silently dropped `--patch`. The branch moved above the composition
  (where `--mode trajectory` already sits) and `spawn_command` carries the patch
  to the daemon, which is what composes.
* **The picker shows the daemon's own word.** `SessionSummary.state` was a
  two-value string invented client-side; `sessions/list` already carries
  `Root.status`, so a root that is `waiting` — parked on somebody else's approval
  modal, which is exactly what you want to know before joining — no longer reads
  as `running`.
* **`$PH_HOME/trust.json` had three spellings**, one keyed off an injectable
  `home` and one off `$PH_HOME`, so they disagreed whenever those differed and
  the gate passed for a directory nobody had vouched for. One `trust_path()`, one
  `TrustAnswer` literal shared by both ends, and `always` recorded only after the
  mount succeeds.
* **A latent Textual bug, surfaced by the new timing.** `TranscriptView.sync`
  did not serialise its own mounts, so two overlapping draws called `mount()`
  while one was pending — `MountError`, reproducible only under coverage. Fixed
  in the widget, where the ordering constraint lives, with the companion guard
  that a draw landing after the view is detached renders nothing.

**One gate that had to be rewritten to mean anything.**
`test_the_catch_up_and_the_live_stream_never_draw_one_event_twice` was written as an integration
test — run a turn, attach a second front end, compare transcripts — and it **passed with the
`seq` check deleted**, because a turn against the fake provider finishes long before the overlap
it would have to be caught in. So it is now two tests: one that the two routes agree
(`..._rebuilds_it_exactly`), and one driven against `Followed` directly
(`test_an_event_arriving_on_both_routes_is_folded_once`) where the overlap is constructed rather
than hoped for. That one fails under its sabotage; the first never could.

Also landed, and not in the plan: `Root.idle_for` falls back to the header's `created_at`. It
measured quiet from the log's last event and returned `0` for a log with *no* events — so a
session created and never used was permanently unpassivatable, holding a mounted profile for
the life of the daemon. P5-05 has always had that; P7-08 is what turns it into a resident
*process*, since `spent()` cannot be true while such a root exists. The first cut added a
non-durable `mounted_at` clock; `created_at` is the same instant in the only case it is read, is
already in the log, and makes an empty *stored* session releasable at once rather than a
minute after each rehydrate.

`PH_TUI_DAEMON` was not needed: there was no period in which both front ends had
to coexist, because the pilot suite converted in one pass and the in-process one
had no other caller. So 2a's flag and 2b's flip collapsed into the switch-over,
and 2c followed immediately — which is also why `FrontSession` now has one
implementation. It stays a protocol: P7-07's renderer is the second consumer, the
AST gate holds the terminal to reading nothing else, and every member being
shaped around what the screen needs rather than which service supplies it is
what made the move possible at all.

## Increment 2d — one daemon, many repositories — new row P7-14

*(Landed.)* Two things the switch-over exposed once the picker could show a repo
path, both asked as questions and both real.

**`sessions/browse`.** The picker read the daemon's filesystem: `daemon/config`
handed over a `sessionsDirectory` and the client walked it, peeking forty log
headers. It worked only while client and daemon shared a machine — true today,
since the socket is a unix socket, and true for `--mode web`, since
textual-serve runs the TUI as a *local* subprocess — but it is the mechanism
P7-07's renderer cannot use, and it let the two ends disagree about which
`$PH_HOME` they meant. Now the daemon folds one list: stored logs *and* its own
live roots, each row carrying the `state` the supervisor calls it (`running`,
`waiting`, `retrying` — a root parked on somebody else's approval modal is not
one that is working) and the `cwd` from its header. `merge_live`,
`live_sessions` and `sessions_directory` are gone; `SessionSummary` is a
`WireModel` and `ph_app/sessions.py` left `tui/` because the reader did.

**It does not need to be per-repository, and the premise is worth correcting.**
There is one daemon per *user* — `$PH_RUNTIME/daemon.sock`, per boot, per user —
holding many roots, each with its own mounted `Context` and its own `cwd`. So any
daemon can list every session and no restart is needed to reach one in another
checkout. A daemon per repository would collide on that socket and would
multiply session leases and schedulers for nothing.

**But a root did not work in its own repository, and that is what made a
per-repo daemon look necessary.** `fs-local` rooted every mount at
`config.root or Path.cwd()` — the *daemon's* directory — so two sessions in two
checkouts read and wrote the same tree while each header recorded where it
belonged. It was right by accident while the terminal was the harness, because
the process really was in the repo. Fixed at the mount, not after it: `mounted`
takes a `project=` and provides `project_root` **before the first row**, the way
`Profile.mount` already provides `ctx.mount` and for the same reason — the fs
seam fixes its root while applying, and `workspace-lifecycle` reads that root
there to discover the project's provisioning, so a root mounted first and rebased
afterwards has already branched its worktree from the wrong tree. A per-mount
value rather than a profile setting, because one composition is mounted once per
session and those sessions are in different repositories; composing per root
would re-import every plugin to change one path. `fs.rebase` is untouched: it is
a claimed slot the workspace row owns, and a session's `cwd` is what that row
branches *from*.

Where the directory comes from: the client's `cwd` for a new session, and the
session's own header for a resumed one — read off disk by `recorded_cwd`, because
on a resume there is no client to ask and no store to ask either until the mount
exists.

**And no path the model reads names the machine any more.** `glob` and `grep`
already answered workspace-relative by slicing the walk's prefix; `read`, `write`
and `edit` echoed absolutes, and so did the `fs/observed` record. All four go
through `FsService.named`, `resolve`'s inverse. The reason is the prompt cache
(A11/A12): an absolute path inside the workspace names the same file as its
relative form and additionally puts the run's own directory into the transcript,
so replaying the session against a fresh workspace — a retried job, a
re-provisioned worktree, the same repo checked out elsewhere — changes every one
of those strings and moves the cached prefix for a difference the conversation
cannot see. A path *outside* the workspace keeps its absolute form: it is not a
name the workspace can express, and the approval that let it through is what made
it legible.

**Gates:** `test_a_root_works_in_the_directory_its_session_names` (sabotage: drop
`project=` and both roots land in the daemon's cwd) ·
`test_a_resumed_root_returns_to_its_own_repo` (sabotage: resolve the directory
only from the client's parameter) · `test_no_path_the_model_reads_names_the_machine`
(sabotage: `str(target)` from any of the three) ·
`test_a_read_records_the_workspace_relative_name` ·
`test_the_picker_reads_no_session_file` (sabotage: fold the live roots out of
`browse_of` and a person cannot find the session they are sitting in).

**Not enforced (§5 rule 6):** nothing *stops* a model passing an absolute path —
`fs.resolve` still accepts one, and it must, because a path outside the workspace
has no other spelling. What is guaranteed is that no path pH *writes down* is
absolute-inside-the-workspace. A model that insists on absolutes still moves its
own prefix, and the tool descriptions already say relative paths resolve against
the workspace. And `recorded_cwd` is filesystem-shaped, like the listing fold it
sits beside: a backend that keeps sessions elsewhere answers `""`, and the root
mounts where the profile says.

## Increment 3 — `ph --mode web` on `textual-serve` — new row P7-05

*(Landed.)* `ph_app/web/serve.py` composes **our own** `web.Application` from
upstream's three handlers, two lifecycle hooks and two asset paths, because
`Server._make_app` builds its app inside `serve()` with no hook for a middleware
— and a token gate has to be a middleware, since the handlers are upstream's and
cannot be edited. Three details of that re-statement turned out to be upstream's
requirements rather than choices, and each is now a gate:

* `aiohttp_jinja2.setup` with upstream's own loader, because `handle_index` is
  `@aiohttp_jinja2.template`-decorated and renders nothing without it;
* the route **names** `websocket` and `static`, which `handle_index` resolves out
  of the router to build the page's socket URL and asset prefix;
* upstream's `statics_path`/`templates_path`, resolved relative to its own
  `server.py` — read off the instance rather than guessed.

`on_startup` is only a banner today and `on_shutdown` is empty; both are wired
anyway, because a release that moved subprocess cleanup into either would leak a
process per tab. Our own `_announce` runs after the banner, because upstream's
names the *command* (a long `python -m ph_app --mode tui …`) and the public URL
**without** the token, which is the half that does not work.

**The token is exchanged for a cookie**, and that was not optional: the page's
websocket and asset URLs are built by upstream's template, so they cannot carry a
query parameter this module chose — a query-only gate would have served the shell
and refused the socket it opens. `compare_digest`, because `==` on a secret is a
prefix oracle.

**Gates:** `test_textual_serve_still_exposes_what_we_compose` (sabotage: rename
any handler) · `test_the_page_is_rendered_by_upstreams_own_template` ·
`test_the_shell_is_refused_without_the_token` ·
`test_the_websocket_is_refused_without_the_token` (asserted separately, because
it is the door a cookie-only scheme leaves open) ·
`test_the_token_is_exchanged_for_a_cookie` — which renders the real page, so it
is also what catches a dropped `jinja2` setup or a renamed route ·
`test_a_wrong_token_is_refused_like_none_at_all` ·
`test_every_launch_mints_its_own_token` ·
`test_a_tab_runs_this_python_not_whatever_is_on_path` ·
`test_the_cli_does_not_import_the_web_extra` (sabotage: import `.web.serve` at
the top of `cli.py`, and every `ph -p` pays for an HTTP stack — and fails outright
wherever the extra is not installed). Smoke-tested through the CLI itself: 403,
200 with the token, 403 on `/ws`.

**Not enforced (§5 rule 6):** no TLS, no users, no revocation. One secret per
launch is the whole scheme — anyone holding the URL has whatever authority the
terminal has, which includes approving tool calls and running `!!`. `--host` on a
shared machine is a decision this module cannot make for a person, so it prints
the consequence and proceeds. Deliberately **not** `--session`: each tab is its
own subprocess and so its own front end, and a shared id would put every tab on
one session by default.

**Also true and worth knowing:** `fail_under = 91` measures `source = ["ph"]` —
ph-core only — so none of this module, and none of Increment 2's ph-app work, is
in that figure. Widening the source is P6-02's debt, not this row's.

**What the cleanup pass changed, including two things I had got wrong.**

* **One `reinvoke` in `cli.py`.** "How pH starts pH" was spelled three times: the
  daemon's argv, the web branch's, and `tui_command`'s copy of the
  `sys.executable -m ph_app` argument. `spawn_command` is now one line over
  `reinvoke`, the web branch is another, and `serve.py` no longer knows pH's
  module name. Only the daemon's copy had been pinned, so a renamed option would
  have surfaced in a browser tab; `test_pH_reinvokes_itself_one_way` pins both.
* **The web server runs under anyio**, like every other mode. `web.run_app`
  installs its own signal handlers and owns the loop, which made `--mode web` the
  one mode whose shutdown was not pH's — and P7-06's upload route needs a
  `DaemonClient` with this server's lifetime, which is a task group rather than
  an addition. `shutdown_timeout=GRACE_SECONDS` too: aiohttp's default is 60,
  and two budgets for one shutdown is how the inner one becomes dead code.
* **`webbrowser.open` moved after the bind and onto a thread.** It scans `PATH`
  for every known browser first (~90 ms), and a `$BROWSER` naming a plain command
  makes it *wait for that process to exit* — so done first it either hands out a
  URL the socket is not answering on yet or never starts the server at all.
* **One owner for the exposure notice.** It was printed twice, in two files, in
  two styles, on two sides of the bind, with the CLI comparing against a
  `"127.0.0.1"` literal it did not own. `WebServer.notices()` holds the sentences
  and knows its own host; `cli.py` prints them — the shape `ph daemon` already
  uses for its socket path and linger warning.
* **The import probe generalised.** It asserted "the CLI does not import the web
  extra" inside `test_web.py`; the real rule is that importing `ph_app.cli` drags
  in nothing heavy or extra-only, and `cli.py` had made that promise about
  **textual** since P5-14 with nothing testing it. Now
  `packages/ph-app/tests/test_app_layering.py` — the app layer's half of
  ph-core's `FORBIDDEN` — covers textual, textual_serve, aiohttp, jinja2 and
  opentelemetry in one subprocess. Sabotage confirms it catches the textual case.
* **ph-core's `FORBIDDEN` rationale was wrong.** I wrote that those libraries are
  forbidden because they "arrive through an optional extra" — but `opentelemetry`
  is an extra ph-core imports deliberately, and `rich` is not optional at all.
  The rule is *presentation* libraries, and it now says so.
* **`shlex.join`'s reason was wrong too, in a way that invited a regression.** I
  wrote that upstream "takes a string and splits it itself"; it actually runs the
  command through `asyncio.create_subprocess_shell`, i.e. `sh -c`. So the quoting
  is a shell-injection boundary rather than a convenience — `--patch '{id: x,
  disabled: false}'` reaches it — and a future "simplification" to `" ".join`
  would have been a hole. The join now lives at the one call site, in `cli.py`.
* **The multiplex claim was overstated.** `serve.py` said each tab was "exactly
  the multiplex the design already permits — several UIs watch one session". It
  is not, by default: `app.py` does `session_id or new_session_id()`, so two tabs
  are two *new* sessions. The multiplex is *reachable* through the picker, which
  is what the docstring says now. The per-tab default is forced anyway —
  `Server.command` is fixed at construction and varies nothing per request.
* Deleted: the `SERVER` AppKey (written, never read — P7-06 adds it with a
  reader), `run_web`'s dead `title` parameter, the `_gate()` factory (the
  middleware is a plain module function reading the token off the app, so it
  holds no server), the `OPEN_PATHS` frozenset-of-one, and
  `test_the_page_is_rendered_by_upstreams_own_template` — whose
  `hasattr(..., "__wrapped__")` was a tautology about `functools.wraps`, and
  whose claim the end-to-end render test already carries (proven by sabotage).



- Dependency `textual-serve>=1.1.3` (brings `aiohttp`, `jinja2`) as extra `ph-app[web]`;
  `--mode web` without it fails with the install hint. It needs only `textual>=0.66`, so the
  existing `<9` pin is untouched.
- `textual_serve.Server` builds its `web.Application` inside `serve()` with no route hook, but
  `handle_index`, `handle_websocket`, `handle_download` and the static path are public. New
  `ph_app/web/serve.py` composes **our own** `web.Application`: those four routes + `POST
  /api/attachments` + a token middleware, reusing `Server.on_startup/on_shutdown`. **Gate**
  `test_textual_serve_still_exposes_the_four_handlers_we_compose`, so an upstream rename fails at
  test time rather than at `ph --mode web`.
- One subprocess per browser tab (textual-serve's model) — each is one more `DaemonSession`,
  which the multiplex design already permits.
- Token minted per launch, in the URL (`--open` launches the browser with it); middleware refuses
  `/`, `/ws`, `/api/*` without it. `--host` default `127.0.0.1`, `--port` default free.
- **Gate:** `aiohttp.test_utils.TestClient` — `/` without a token is 403; with it serves the
  shell; the websocket handshake reaches textual-serve's handler (a stub `command` keeps CI
  browser-free).

## Increment 4 — upload from the browser — P7-06

*(Landed.)* Drop a file on the page → `POST /api/attachments` → `attachment/put`
+ `session/stage` → `session.staged` puts the chip in **every** attached UI,
including the terminal on the server, and the next prompt from any of them
carries it as a `MediaBlock`.

**The routing question decided the session design.** Upstream gives a tab no
identity: `Server.command` is fixed at construction, `_build_environment` copies
the server's own `os.environ`, and the websocket query carries only width,
height and font size. So nothing distinguishes one tab from another, and an
upload cannot be routed to "this tab's session" — the id has to be one the
*server* knows. Increment 3 had left it unset, which does not avoid the problem:
it makes each tab a different new session with nothing able to say which one a
file belongs to. So a launch mints one (or takes `--session`) and every tab is on
it, which is the plan's own decision 2 realised in a browser — several UIs, one
conversation, private composers. A tab therefore differs from a second terminal,
which does start a new session; the difference is honest, and the picker still
moves a tab elsewhere.

**The drop zone is inserted, not templated.** `app_index.html` is upstream's and
has no jinja blocks, so the choice was to fork a template whose other half ships
as JS in the wheel, or to add to what it rendered. Insertion keeps upstream's
page authoritative; `CLOSING_BODY` is the one thing that can break, it is a named
constant, and a page that no longer contains it is served unmodified with a
warning — the terminal still works and `/attach` still does.

**The upload's daemon client is per request.** An upload is rare and already
costs a file read; a unix connect is sub-millisecond; and a client held for the
process's life is one that goes stale when the daemon passivates or an ephemeral
one exits — staleness this route would then have to detect and repair for no
gain.

Two defects the real path found that the tests did not:

* **A daemon refusal arrived as a 500.** Uploading before opening a tab is the
  ordinary state between launching and using it, and the daemon's honest `no
  session "x"` became `500 Internal Server Error` with an `ExceptionGroup`
  traceback in the log. Worse, the obvious fix would not have worked: the calls
  run inside a task group — the pump must be reading for a reply to arrive — so
  anyio wraps whatever leaves it and an `except DaemonError` sees nothing. That
  is the `_alone` hazard; `except*` is the fix, and
  `test_an_upload_before_any_tab_exists_says_what_to_do` is the gate.
* **`attachmentId`, not `id`.** The route echoed the wrong key, so a browser and
  the log would have called one blob by two names. Caught by asserting on the
  reference the *log* kept rather than on the reply.

**Gates:** `test_a_browser_uploaded_file_reaches_the_model_as_a_media_block` —
the plan's own, asserted on the `user/message` because a 201 would pass for a
server that dropped the file, and asserting the same bytes twice are one blob
(sabotage: stop draining `staged` in `Supervisor.prompt`) ·
`test_the_page_offers_somewhere_to_drop_a_file` ·
`test_an_upload_needs_the_token_like_everything_else` (this route *writes*, so it
is the one a gate must not miss) ·
`test_an_oversized_upload_is_refused_before_it_is_buffered` (chunked against the
daemon's own 5 MiB ceiling, so the refusal costs no memory) ·
`test_an_upload_with_no_daemon_says_so` ·
`test_an_upload_before_any_tab_exists_says_what_to_do`. Verified by hand with
curl against a real daemon: a sentence before a tab exists, a content-addressed
`sha256:` reference after, and 403 with no cookie.

**Not enforced (§5 rule 6):** no chunked upload — 5 MiB is one frame's worth, and
a 40 MB video gets a sentence naming the limit rather than a truncated blob.
Nothing here reads the person's filesystem: the *page* takes the bytes, which is
the browser's half of the human door (I-9), and `/attach` still reads paths on
whichever machine the front end is on — the server's, for a tab.

**What the cleanup pass changed, and one more defect the real path found.**

* **A malformed upload was a 500.** `request.multipart()` *asserts* on a body
  that is not multipart, so an authorised POST with no `-F` reached a person as
  an `AssertionError` traceback — the same defect as the daemon refusal one
  paragraph up, found the same way (by hand, not by a test). Both guards now
  answer: 415 for a body that is not multipart, 400 for a part named something
  else, one sentence between them. The scan for a part named `file` went with it:
  the only producer is the drop zone, and a loop looking for a shape nothing
  sends was inventing a caller in order to be lenient with it.
* **The put/stage pair has one author.** `attach.py::stage_bytes` — beside
  `Tray` and `prompt_message`, in the module that already owns the human door —
  is what both front ends call. It was written twice, so the base64 encoding, the
  `attachment` reply key and the `attachment=` parameter each had two spellings,
  and P7-07 would have made three. Not a *daemon* method: `attachment/put` is
  content-addressed and deliberately outside `MUTATIONS` while `session/stage` is
  keyed, and collapsing them would move a file's name and type — the human half
  of the decision — onto the daemon.
* **`connected` came out of `ph agents`.** The connect-pump-close task group,
  including the "assign inside the group, return outside" subtlety anyio's typing
  forces, was re-derived here from scratch. It is `daemon/client.py::connected`
  now, and its docstring says which of the two failures crosses the group — which
  is the whole reason the `except OSError` and the `except*` are nested rather
  than side by side.
* **A front end imports the daemon's client, never its server** —
  `test_a_front_end_imports_the_daemons_client_and_not_its_server`. It was true
  by intention and false in fact: `ph_app/daemon/__init__.py` re-exported
  `DaemonServer` and `Supervisor`, so importing the *client* executed the
  supervisor, every seam and a `Profile`. That package exports nothing now. The
  claim is the boundary rather than the 75 modules: a process that can mount a
  session is a second supervisor competing for the leases the first one holds.
* **The mime ladder has one author too.** `ph.seams.attachments.mime_of`, whose
  own docstring already argued for it — the classification decides the *stored
  file's extension* (`path_for`) and whether a block reaches the model as an
  image or a document, so a second copy is how one PNG becomes two things. The
  browser's declared `Content-Type` still wins over the guess, which is a real
  difference and not a third ladder.
* **`MAX_ATTACHMENT_BYTES` moved to `daemon/framing.py`.** Its docstring was
  always a derivation from `MAX_LINE`; it lived in `server.py`, so a front end
  had to name the supervisor's module to read an integer.
* **`client_max_size` was inert and its comment claimed the opposite.** aiohttp
  applies it in `read()` and `post()` and *never* in `read_chunk`, which is the
  only reader on this path — so the chunked check is load-bearing and the `+ 64
  KiB` headroom was guarding nothing. One number now, and the sentence says which
  of the two actually refuses. The read is a `bytearray` at 64 KiB a chunk rather
  than a list joined at the end: 80 awaits for a 5 MiB drop instead of 640, and
  ~5 MB less peak.
* **`notices()` names the session.** It is the one place the item above's
  surprise — a second *tab* is not a second session, unlike a second *terminal* —
  becomes visible rather than discovered, and it is the id for `ph agents attach`
  afterwards.
* Smaller: `WebServer` is a `@dataclass(slots=True)` like the other 54 in the
  package, so adding a field is one edit rather than three kept in step by eye;
  `_index` is a method behind `partial` rather than a closure factory, which is
  what `TOKEN`'s own docstring argues for eight lines above it; `--resume` works
  for `--mode web` and its help no longer says "tui only"; the drop zone's dead
  `.busy` rule is gone and an error message restores the prompt instead of
  sitting there until the page reloads; `run_web`'s docstring stopped predicting
  a held `DaemonClient` that `_stage` argues against.

**One thing left alone, and measured.** `ph.session.dumps` uses
`ensure_ascii=False`, which costs 54 ms of the ~65 ms a 5 MiB upload spends
building its frame — base64 is pure ASCII, so `ensure_ascii=True` would produce
identical bytes in 9 ms. It is not this row's to change: every other caller is a
log line, where readable UTF-8 is the deliberate choice, and switching it would
change the bytes of every log written afterwards.

- The shell template adds a drop zone; `POST /api/attachments` (multipart) → `attachment/put` →
  `session/stage` → the chip appears in the terminal composer via `session.staged`; the next
  submit from *any* client carries it.
- **Gate:** `test_a_browser_uploaded_file_reaches_the_model_as_a_media_block` — POST bytes → `201
  {attachmentId: "sha256:…"}` → prompt → `user/message` carries the `MediaBlock`, blob
  content-addressed, twice → one blob. **Sabotage:** stop draining `staged` in `Supervisor.prompt`.

## Increment 5 — bookkeeping and docs

- Plan rows: P5-13 landed (1a–1c), P5-14 landed (0, 1d, 2), P5-15 protocol half landed with the
  declarative body deferred and why; new **P7-05** (web on textual-serve), **P7-06** (upload),
  **P7-07** (native HTML renderer — `ph_app/ui/frame.py` projection layer, `ScreenData`
  declarative screens, layout-parity golden), **P7-08** (ephemeral daemon lifetime), **P7-10** (`!!` shell), **P7-09**
  (`ask_user` tool + durable questions, disabled outside interactive profiles, logged only when
  attended), each with its non-guarantees.
- DESIGN §4.1/4.2: `--mode web`, `--keep-daemon`, `ph daemon --ephemeral`; §3 the ask seams now
  have a transport; §8 known gaps: staged attachments not durable, chunked upload absent,
  `build()` still in-process.
- `test_layering.FORBIDDEN += {"aiohttp", "textual_serve"}`.

*(Landed.)* `Implementation_Plan.md` now carries P5-13 and P5-14 as landed, P5-15
as **half** — a screen's schema travels, its `build()` cannot, and the gap is a
gate rather than a note — plus seven new rows: **P7-05** (web on textual-serve),
**P7-06** (upload), **P7-07** (the native renderer, scoped and not built),
**P7-08** (ephemeral daemon lifetime), **P7-09** (`ask_user`), **P7-10** (`!!`)
and **P7-14** (one daemon, many repositories). Each carries its own non-guarantees,
because §5 rule 6 says a caveat that lives only in a doc is a defect.

**The row numbering.** Increment 2d called its row `P7-11a`, and that suffix would
have claimed kinship with P7-11 (seams describe their own wire form) that it does
not have — the two share nothing. `P7-14`, cited nowhere but here, so renaming it
cost one line. And the phase intro now says out loud that Phase 7 holds the front
ends as well as the media: the rows arrived through it (a browser is where a
person's *file* comes from, so P7-06 **is** P7-01's human door reached from a
tab), and renumbering them afterwards would have broken every citation in the
source to save a heading.

**DESIGN.** §3.1 gains the paragraph the ask seams had been missing: they now have
a *transport*, and it changed the seam rather than sitting beside it —
`register_answerer(reachable=)`, because whether anyone is there to ask is no
longer knowable from the fact that a listener registered, and `answer_to_wire`,
because an answer has to be serialisable. §4.1/4.2 gain `--keep-daemon`. §8 gains
four gaps: a staged attachment is not durable, there is no chunked upload, a
third-party row's screen is invisible to a remote front end, and `repaired()`
still closes a turn parked on a human as interrupted. `docs/dev-notes/phase-5.md`
moves from "three deferred" to "fourteen landed, one half" and says where the
phase went next.

**A second cleanup pass, once the first one's result could be read.**

* **The unwrap moved into the thing that creates it.** `connected` runs the
  exchange inside a task group, so anyio wraps whatever leaves it — a hazard
  every caller was solving for itself, one with `_alone` and one with `except*`,
  and `connected`'s own docstring conceded the two spellings out loud. The group
  is *its*, so unwrapping it is too; both callers now write the plain
  `except DaemonError` they meant, and `_ask` lost a branch that existed only to
  re-raise the member instead of the wrapper.
* **The mime ladder had a hole exactly where it mattered.** `mime_of` was
  extracted last pass and its docstring named three callers; it had two — the
  daemon still defaulted to octet-stream, throwing away the name it had on the
  next line. Worse, "the declared type wins" is wrong at the one door that has a
  declaration: **browsers send `application/octet-stream` for any extension they
  do not recognise**, so a dropped `.png` from such a browser stored as a
  document, got no suffix out of `EXTENSIONS`, missed `IMAGE_MIMES`, and reached
  the model as a file — the same picture being an image through `--attach` and a
  document through a tab, which is the split the extraction was written to
  prevent. `mime_for(declared, name)` is the whole ladder, three callers, one
  author, gated at both ends.
* **One refusal was not one status.** `_stage` answered every daemon refusal with
  503 and "open a tab first". A file of exactly 5 MiB reaches
  `attachment_too_large` — this route's ceiling is on the raw bytes and the
  daemon's is on their base64 expansion, which is a hair larger — and being told
  to open a tab is advice about the wrong problem. `_refused` keeps the daemon's
  own sentence, picks the status off `DaemonError.reason` (which *is*
  `Refusal.code` after a round trip, so it is the wire's vocabulary rather than a
  second one invented at the edge), and attaches advice to the one reason that
  has a next step.
* **An oversized drop no longer crosses the wire to be thrown away.**
  `Content-Length` is refused before the body is read; the chunk loop stays as
  the backstop for a chunked body, which has no length to check.
* **`tab_command` beside `spawn_command`.** The tab's argv was assembled inline
  in the `--mode web` branch, and the argv gate "tested" it by writing its *own*
  composition — one that omitted `--session` and `--keep-daemon`, because no
  caller was there to disagree. A gate marking its own homework. Two named
  functions now, both one-liners over `reinvoke`, and the gate calls the thing
  the CLI calls.
* **The reference, not one field of it.** `stage_bytes` returned the attachment
  id as a string, so `_upload` rebuilt a two-field subset by hand; it returns the
  `AttachmentRef` now and the route replies with `to_wire()`, so a browser and
  the log describe one blob with one set of field names.
* **A third layering claim, and one probe.** `test_app_layering` had two
  near-identical subprocess probes; they share `_dragged_in` now and the third
  row is new: **`ph_app.attach` imports no daemon at all.** Its `DaemonClient`
  annotation is under `TYPE_CHECKING`, nothing tested that, and a guard nobody
  tests is a guard somebody deletes while tidying imports.
* **DESIGN §8's `repaired()` row has a gate.** It was the one of the four that
  is a statement about what a pure function returns for one log shape — the row
  most likely to go quietly false the moment P5-13's resume half lands.
  `test_a_turn_parked_on_a_human_is_closed_as_interrupted` builds the parked log,
  asserts `pending_approvals` folds it, and asserts repair closes the turn
  anyway; it is a gate on a *non-guarantee*, and it will fail when the gap closes.
* Smaller: `serving(tmp_path, monkeypatch)` in `daemon_helpers` — the pin and the
  socket in one step, because the *order* is the rule and three files wrote it as
  two bare lines; `DEFAULT_HOST`/`DEFAULT_PORT` in `web/__init__.py`, since the
  CLI and the dataclass each spelled them and a test asserting the server's would
  have passed for a CLI that had moved; `MAX_ATTACHMENT_SIZE` beside the
  constant, so two doors cannot round one limit down two ways; `title` off the
  dataclass, since its own docstring said it was not a knob; `browser_on` takes
  the server rather than making one; and four arguments that were being made
  twice each are made once.

**One thing the bookkeeping found, and built rather than documented.**
`--keep-daemon` was decided in this plan's decision 4 — "`ph daemon --ephemeral`
and `ph --mode tui --keep-daemon` cross over" — and only the first half was ever
built. Writing the DESIGN line would have documented a flag that did not exist,
which is the failure rule 6 is about pointing the other way. It is five lines
(`spawn_command(keep=)` drops `--ephemeral`, and the web branch forwards the flag
to its tabs) and one gate, `test_keep_daemon_starts_a_service_instead`, whose
sabotage is the one a flag has when nobody tests it: ignore `keep`, and
`--keep-daemon` reads as accepted and does nothing.

## Reuse (do not rewrite)

`_Follow`/`_catch_up` (`agents.py:315-426`) · `respond`/`notification`/`capabilities`
(`protocol.py`) · `pending_approvals` (`approval.py:194`) and `_record_asked`/`_record_decided`
(`:310-352`) · `ScheduleIndex` + `Supervisor.rehydrate`/`wake_and_tick` · `sweep(after=)` and
`passivatable` · `AskUserModal` (`tui/modals/ask_user.py`, already built) · `tool-todo`'s row
shape · `prompt_message`/`ingest` (`attach.py`) · `AttachmentStore.save_bytes/exists` ·
`filelock` (as P5-03 uses it) · the socket unlink already in `serve()`'s teardown
(`server.py:673`) · `daemon_helpers.running`, `tests/tui_helpers.running`, `StubHost` ·
`ph_app/wire.py` readers · `session_summaries` · `TrustStore` · `lifetime()`.

## Verification

Per increment: `uv run pytest -q` green, `uv run mypy`, `uv run ruff check`, coverage ≥ 91.
Every gate above must **fail under its named sabotage** before it counts.

End to end, by hand: `ph --mode tui` (silently spawns an ephemeral daemon) → start a long prompt
→ close the terminal → `ph agents` shows the root still running → `ph --mode web --open` → the
same session resumes in the browser mid-turn → drop a PNG on the page → the chip appears in
*both* the browser and a second terminal TUI → submit → `ph agents attach` shows a `user/message`
carrying a `MediaBlock` → ask the model to use `ask_user`, and the question appears in both UIs;
answering either settles both and the answer reaches the model → an approval does the same →
close everything, wait a minute → the daemon is gone and its socket with it → `ph --mode tui`
brings it back with the session rehydrated → `/schedule` an appointment, close every UI → the
daemon **stays**, and `ph agents doctor` says why.

Then the unattended side: `ph -p "…" --profile headless` → `ph --dump-config` shows
`tool-ask-user` disabled and `tools/list` does not offer it → run the same prompt under `tui`
with every UI detached → the model's `ask_user` call returns "nobody is attending" and
`ph agents attach --all` shows **no** `question/*` events for it.
