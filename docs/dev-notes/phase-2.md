# Phase 2 — The terminal front-end

**Status:** complete · **Gate:** `ruff` + `ruff format` + `mypy --strict` on `ph-core` and `ph-app` + 446 tests (34 pilot, 9 snapshot, 2 harness-stub; 445 pass, 1 provider smoke test skips without a key), green.

Phase 1 proved every model action goes through one governed pipeline. Phase 2 had
to prove something the pipeline cannot: that a **person** can sit in front of it
— see what is happening, be asked before it happens, and come back tomorrow to
the same conversation they left.

The hard part was not the widgets. It was keeping one claim true: *the transcript
is a fold over the log*. Everything else here follows from that.

---

## What landed

| Item | Delivered | Where |
|---|---|---|
| P2-01 | `PHTuiApp`, the harness bridge, the transcript state model, streaming rows | `tui/{app,frontend,state,adapter}.py`, `tui/widgets/transcript.py` |
| P2-02 | Prompt with paste placeholders, `@path`/`/command` completion, TUI verbs registered into `ctx.commands`, sidebar | `tui/widgets/{prompt,status}.py`, `tui/{autocomplete,commands}.py` |
| P2-03 | Pickers: model, session, session tree, theme, login | `tui/modals/{base,pickers,login}.py` |
| P2-04 | Approval modal (`allowed-once`/`rejected` + reason), ask-user modal, permission-preset switcher, plan review | `tui/modals/{approval,ask_user,trust}.py` |
| P2-05 | Themes and keybindings from `$PH_HOME`, terminal title, turn notification, project trust | `tui/themes/`, `tui/{config,terminal}.py`, `tui/modals/trust.py` |
| P2-06 | Textual discipline: no f-string markup, `notify(markup=False)`, no modal awaited on the pump | enforced in `tui/widgets/`, `tui/modals/` |
| P2-07 | `--mode tui`, the `tui` profile, this note | `ph_app/cli.py`, `ph_app/profiles/tui.yaml` |

**Definition of done, met:** pilot tests drive every modal (approval, ask-user,
trust, plan review, and all five pickers); a resumed session rebuilds its
transcript from `session.events` alone, tool cards included.

---

## Decisions taken inside Phase 2

### 1. The transcript is rebuilt from the log, never from `derive_messages()`

`derive_messages()` is the **model's** view, and compaction deliberately shadows
what it replaced. Rebuilding a human's transcript from it would silently delete
conversation the person already read — the worst possible resume, because it
looks like it worked.

So `TuiEventAdapter` folds `session.events`, and a compaction *marks* the range
it replaced rather than dropping it: the rows stay, dimmed, and
`ChatItem.is_visible_to_model` answers `False` for them. `shadowed` is a separate
flag from `role == "compaction"` on purpose — the summary is what the model sees
now, the shadowed rows are what it no longer sees, and conflating them makes a
compaction indistinguishable from the history it stands in for.

The gate is a test that runs one prompt with a tool call in it, collects the
transcript **live** through `session/event`, then replays the same session from
its events, and asserts the two agree — row for row and card for card. A
live-only shortcut in the adapter would show up there and nowhere else.

### 2. The log's in-memory payloads are `Mapping`, not `dict`

This one cost a debugging round and is worth writing down. A `SessionEvent` is
frozen, so `event.data` is a `MappingProxyType` over **tuples**; only the
persisted-then-reloaded form is a plain `dict` over `list`s. The adapter reads
both.

The first version tested `isinstance(value, dict)`. Result: replay worked
perfectly and the live path silently saw nothing — the tool card never settled
and a second, duplicate row appeared for every result. `_obj`/`_seq` now test
`Mapping`/`Sequence`, and the docstring says why, because this is the same trap
Phase 1 hit with `isinstance(args, dict)` on a `run_code` argument.

### 3. A modal is awaited in a worker; a picker is opened with a callback

Textual's rule is that `push_screen_wait` may only be called from a worker.
pH's two seam answerers — approval and user-questions — are invoked from inside
`agent.run()`, which the app drives in a worker, so they are legal. Everything
opened from a key handler or a command body runs on the message pump, where
awaiting a dismissal deadlocks, so those use `push_screen(screen, callback)`.

`ModalHost` exists to keep that a property of the *caller* rather than a rule
someone has to remember: `frontend.py` needs "something that can put a question
on screen", and `app.py` is the thing that knows it may only do so from a
worker. The claim that the bridge is testable without a terminal is made true by
`test_tui_frontend.py`, where a plain object answers both seams and the log
records the decisions exactly as under the real app.

### 4. pH records the decision; the front-end only decides

`ApprovalService.request` appends `approval/asked`, waterfalls to the answerer,
and appends `approval/decided`. The modal returns an outcome and writes nothing.
The pilot test asserts the round trip through the **log** rather than the modal's
return value, because pH's claim is that the decision is recorded, not that a
dialog appeared.

A rejection may carry a reason, and a reason is worth more than a refusal — "no,
use the existing helper" redirects a turn where a bare denial only stops it. It
is delivered as `agent.steer(...)`, i.e. as what it is: user input at the next
step boundary. The first draft invented an `approval/reason` event for it, which
would have written a log this build refuses to read — `KNOWN_SESSION_EVENT_TYPES`
is closed, and a front-end is not entitled to extend it.

### 5. A `/line` is dispatched, not prompted

Typing `/compact` used to reach the model as a prompt. It now goes to
`ctx.commands.dispatch`, which spends no turn and records `command/run` /
`command/done`. The test asserts no `turn/*` event appears — routing a human's
verb through a model turn is both slower and dishonest, since the log would show
the model deciding something the person decided.

The TUI's own verbs (`/model`, `/theme`, `/sessions`, `/permissions`, `/login`,
`/thinking`, `/tools`, `/sidebar`) are registered **into `ctx.commands`** rather
than written as key handlers. One registry, so the palette lists them, the prompt
completes them, and the log records them, all without a second list (I7). Keys
are shortcuts for the frequent ones, never the only route.

### 6. Never compare a key literal

Every key is read from `TuiKeybindings`, which loads from `$PH_HOME/tui.json`,
and the field names double as Textual binding ids — so one
`App.set_keymap(keybindings.as_map())` rebinds the app, its screens and every
modal at once. The verbs live in one table (`TUI_VERBS`): each is a slash
command in `ctx.commands`, an `action_*` on the app, and, when keyed, a binding.
Adding one is a row plus a method; nothing re-dispatches on a name string.

The app bindings are `priority=True`, and the probe that decided it found a bug
the first cut had shipped: the prompt is a `TextArea`, which itself binds
`ctrl+k`, `ctrl+y`, `ctrl+x`, `ctrl+u`, `ctrl+w` and `ctrl+z` for editing. The
old `on_key` chain ran *after* the key had bubbled, so `ctrl+k` opened the
palette **and** deleted to the end of the line; `ctrl+y` opened the theme picker
**and** redid an edit. Priority bindings are checked before the focused widget
sees the key, so the app now claims it cleanly, and `check_action` keeps every
binding but `quit` quiet while a modal is up — which also ends the other wart,
where a picker's key stacked a second picker on top of the first.

The prompt's own keys (`submit`, `cancel`, completion) are decided in
`PromptArea.on_key`, Textual's public hook: handlers run subclass-first and stop
once `prevent_default()` is called, so a key the prompt claims never reaches the
editor and one it does not claim is edited as usual. (Its helper is named
`intercept_key`, not `handle_key` — that is Textual's own dispatch hook on
`Widget`, and shadowing it breaks every unclaimed key. Found by a test.)

### 7. `$ph-*` variables always resolve

Textual parses `App.CSS` at startup, before a theme is chosen, and a `$ph-*`
that resolves nowhere is a hard parse failure — not a default colour. So
`get_theme_variable_defaults()` returns the default theme's palette as the
fallback palette. Switching to one of Textual's own themes now degrades the
colours instead of crashing the app.

Themes are data: 19 named roles in a JSON file, and the parser refuses a theme
that is *missing* a role **or** that names an unknown one. A typo'd role would
otherwise leave the real one at its default and read as a rendering bug.

### 8. Markdown for the model's voice, literal text for the person's

Assistant and reasoning rows render Markdown; user rows do not. Two reasons,
and neither is taste:

* what someone typed is shown as they typed it; and
* a *streamed* row that rendered Markdown while a *replayed* one showed raw
  asterisks would make a resumed session look different from the session the
  person left — the same failure as §1, one layer up. `TranscriptView` therefore
  builds the same widget for both and finalizes it immediately when the row is
  already settled.

Streaming appends through `MarkdownStream.write`; re-`update()`ing the
accumulated text per delta re-parses the whole message per token and is visibly
janky by the second paragraph. The view tracks how much of each row it has
written and sends only the tail, which also keeps the adapter free of any notion
of what has been drawn.

### 9. Redraw on a frame timer, not per event

A streaming turn commits an `assistant/chunk` every few tokens. Syncing the view
on each one spends more time in layout than in rendering, so the adapter marks
the state dirty and a 30 Hz tick draws whatever arrived. The spinner and the
terminal title share that clock — and one counter, owned by the status bar.

Only the spinner is per-frame. The first cut forced a full sync on every frame
while a turn ran, and `Static.update` always re-lays-out, so a 200-row
transcript re-rendered 203 widgets thirty times a second while waiting on the
model — measured at 9 ms a frame, a quarter of a core, and linear in transcript
length. Now `state_changed()` alone marks a frame dirty, and every settled row
remembers what it last drew and skips `update()` when nothing changed. The
scroll-to-bottom is Textual's `anchor()`, engaged once the transcript first
overflows: anchoring an underfull pane scrolls it to a *negative* offset and
every row renders pushed to the bottom, which is how that quirk was found.

Widget references are held rather than queried, and that is a correctness point
as much as an efficiency one: `App.query_one` searches the **top** screen, so
every lookup fails while a modal is up. The frame timer found this the first
time a test opened one.

### 10. Trust is asked before mounting, and remembered outside the project

Mounting is what reads the project's `AGENTS.md`, its hooks and its configured
plugins — all of which run with the user's permissions. So the prompt comes
first, and `_open()` happens in the callback. The answer is stored under
`$PH_HOME`, never in the project: a file inside the repository could declare the
repository trustworthy, which is the one thing the prompt exists to prevent.

### 11. The `tui` profile differs from `headless` by exactly one row

`read-only` is right for an unattended run: nobody is there to answer an
approval prompt, so nothing should be writable without one. In the TUI a person
is present, the status bar names the posture, and one key changes it — so the
workspace is writable and everything outside it still asks. That is the whole
difference, it is a row addressed by id, and a deployment that disagrees
overrides the same id in `$PH_HOME/profiles/tui.yaml`.

### 12. No model catalogue was invented

pH has no list of models, because a provider knows its own and Phase 1
deliberately did not ship a list to go stale. The model picker lists registered
*providers* and its filter doubles as free-text entry, so
`anthropic/claude-opus-5` offers itself as a value. The login picker works the
same way: candidate credentials come from scanning the **composed configuration**
for `apiKeyEnv`, so an adapter added by a plugin appears without the picker
changing.

### 13. `tau-ai` was declined a second time, on new evidence

Phase 1 declined it because `ProviderToolCallEvent` would have falsified
`assistant/chunk`'s replay-fidelity promise. Phase 2 re-examined it for the ~1.3k
lines of TUI helpers, and declined again: taking them pulls in ~14k lines of
`tau_coding` plus a second agent framework, typed against tau's own protocols
rather than pH's. The port plan names vendoring or reimplementation as the
fallback, and this is the case it had in mind.

---

## Fixed along the way

* **`ph-app` could not build a wheel.** A `force-include` named
  `src/ph_app/profiles`, which `packages = ["src/ph_app"]` already ships, so
  hatchling refused: *"a second file is being added to the wheel archive at the
  same path"*. Pre-existing and latent — nothing had built a wheel. Removed; the
  wheel now ships the profile YAML and the theme JSON.
* **`--resume` came up blank.** `TuiEventAdapter.replay` assigned a fresh
  `TuiState`, but the app and the frontend both hold a reference to the one they
  were given — so the replay filled an object nobody was reading. It now resets
  in place. Caught by the app-level resume test, not by the adapter test, which
  is why that test exists in addition.
* **Terminal control sequences leaked into captured output** as literal
  `]2;pH`. Textual redirects `sys.stdout` while it drives the screen, so the
  title has to be written to `sys.__stdout__`, and only when that is a tty.
  These bytes are instructions, not output.
* **The sidebar never appeared.** `App.set_class` marks the app, and the
  selector said `Screen.-sidebar`. It is now `Widget.display` on the sidebar
  itself — the mechanism for exactly this, and no CSS to get wrong.
* **The spinner, the queue guard and cancel were all dead.** Nothing ever set
  `TuiState.status` to `"running"`, so every branch keyed on it — the footer
  spinner, "queue rather than start a second turn", `escape` interrupting — was
  unreachable, and the pilot tests' `while running` waits exited at once and
  passed by accident. `HarnessSession.submit` now owns the transition, and the
  tests wait on the log's `turn/end` instead.
* **The context gauge never moved.** `state.tokens` had no writer. The adapter
  now folds it from `assistant/message.usage` with the token meter's own formula
  (`input + output + cache read + cache write`), so the amber threshold is
  reachable; the estimate branch stays the meter's job.
* **Code Mode dispatches were compose-time content.** Live, the card widget
  mounts before any `tool/code-dispatch-start` exists, so the "governed calls"
  section appeared on replay and never live — a live/replay divergence at the
  widget level, the property the P2-01 gate protects but only asserted on state.
  Dispatch rows are now mounted as they arrive, keyed by call id.
* **`Widget._render` and `DOMNode.name` were both shadowed** — by a picker's
  list refresh and by the login modal's credential name. Both renamed.

### 14. What the `/simplify` pass moved to the right layer

Four reviewers (reuse, simplification, efficiency, altitude) read the Phase 2
diff. Besides the fixes above, what changed and why:

* **`open_harness` uses `ph_app.runtime.mounted`** rather than re-deriving the
  mount/drain/dispose sequence — the helper's own docstring names the TUI as a
  caller, and `conftest.py` names that re-derivation as the thing it exists to
  stop. The front-end listens on `session.observe()` — the per-session feed —
  instead of the store-wide bus with an identity filter, so a subagent's events
  are never received rather than received and discarded. `close()` flushes,
  because disposal does not and the app had to remember to pair them.
* **The adapter is a table.** `HANDLERS` maps event type → method and
  `RECORDLESS` names the six known types that deliberately produce no row; a
  test holds `HANDLERS ∪ RECORDLESS = KNOWN_SESSION_EVENT_TYPES` (plus the one
  admitted forward reference, `todo/write`). A new event type can no longer go
  silently unrendered. It also reuses `thaw_json`, `parse_arguments` and
  `parse_request_context` instead of three local equivalents, and its
  log-reading helpers moved to `tui/wire.py` so the session lister shares them.
* **Themes are read once.** `ThemeCatalog` replaces per-name `load_user_themes`
  calls — startup scanned `$PH_HOME/themes` once per theme — and owns the
  unknown-theme fallback and the `$ph-*` fallback variables, so the snapshot
  test app and the real app bootstrap identically.
* **Display toggles live in `TuiSettings`** and persist. Before, the toggle
  state had three homes (settings, the view, a filter argument) and
  `save_tui_settings` was defined and never called; now a toggle or a theme pick
  is remembered for the next launch.
* **Session pickers are one function.** The tree version was dead (nothing
  supplied fork parents); the flat one is now the tree, reading
  `SessionHeader.parent_session`, and a session without forks is a tree with
  one node. The picker also lists the *store's* root, not `$PH_HOME/sessions`,
  since a profile may point persistence elsewhere.
* **Removed:** the unreachable `_await_modal` fallback and the `Answered`
  message that existed for it; the `agents=` completion source nothing passed;
  `TerminalTitle.enabled`/`prefix`; the `"desktop"` notification nobody
  implemented; `notify_turn_end` (Textual's `App.bell()` writes through the
  driver and is headless-safe); the three package `__init__` re-export lists
  nothing imported; per-modal `keybindings` parameters, made redundant by the
  keymap.
* **Kept, deliberately:** `ToolCard.card` — the tool's declared `CardKind`,
  which P3-19's code cell is the first widget to branch on; `is_visible_to_model`
  — the domain word for `not shadowed`, which the trajectory view will use;
  `plan_review_modal`, unopened until Phase 4's `plan/mode` but a P2-04
  deliverable; and `TuiSettings` as its own file (see sharp edges).

---

## Deliberately deferred

* **The trajectory view.** dsh ships two front-end projections —
  `ui-conversation` (the chat) and `ui-trajectory` (a table/timeline of every
  record, inspectable by producer, with search and cross-navigation). Phase 2
  delivered the first. The second was missing from both planning documents and
  has been added as **P3-24/P3-25**, with the two candidate shapes and the
  decision written up in port plan §5.5.

  Six known event types therefore render nowhere today: `request/header`,
  `step/start`, `step/end`, `approval/policy`, `fs/observed`,
  `session/end-seed`. That is deliberate — none of them is chat content — but it
  does mean the TUI currently shows no system-prompt snapshot and no step
  timings. The transcript also reads `source.kind` only, to pick user-vs-context,
  and discards the producer name, so "inspect by source" is not possible yet.
  Nothing is lost by waiting: the log records all of it, so the view is a
  projection we can add without a migration (I4, I6).

  The session *picker* is not this. It chooses which session to open; it says
  nothing about what happened inside one.
* **`/compact`, `/revert`, `/refine` bodies** — Phase 4 owns compaction and the
  history rewrites. The commands the TUI registers are the front-end's own
  verbs; the harness verbs arrive with the machinery they drive.
* **Resuming in place.** Choosing a session from the picker *works* — as a
  restart. The app exits with the chosen id, everything it mounted unwinds, and
  `run_tui` mounts a fresh app on that session. Swapping the harness underneath
  a live agent scope would need the scope, its registrations and its artifacts
  to unwind first, which is Phase 3's territory; the restart needs none of it.
* **The context gauge's threshold** is pinned to `COMPACTION_THRESHOLD = 0.85`
  so the number a user sees coming is the one at which the harness will act.
  Phase 4 owns the acting.
* **Todo rows** are rendered from `state.todos`, and the adapter has a
  `todo/write` handler — but `todo/write` is not in
  `KNOWN_SESSION_EVENT_TYPES`, so nothing in this build can append one. The
  handler is unreachable until Phase 4 adds the type *and* the tool; the sidebar
  shows `—`. Listed here rather than deleted because the projection is the part
  Phase 2 owns.

---

## Known sharp edges

* `TranscriptView` keys rows by `ChatItem.key`. Two events producing the same
  key would collide silently rather than raise; the keys are seq-derived, so
  this needs a bug in the adapter to happen, but nothing enforces it.
* The frame timer redraws the whole visible row list. It is idempotent and
  cheap for a screenful, but there is no windowing: a transcript of thousands of
  rows will mount thousands of widgets. Textual handles that better than a naive
  implementation would, and it has not been measured.
* `TranscriptView` anchors only once `max_scroll_y > 0`, because Textual's
  `anchor()` on an underfull pane produces a negative scroll offset. If a future
  Textual clamps that, the guard becomes redundant but not wrong.
* `TuiSettings` is a second preferences store beside `ctx.settings`
  (`$PH_HOME/settings.json`). It reads before any context is mounted, which is
  why it exists; folding it into `settings.json` under `tui.*` keys is a
  reasonable later move and was left alone here.
