# Themes, prompt history, session pickup, and a daemon that leaves

*Four front-end rows and one process-lifetime row. Phase 9 — the front end a person
lives in.*

## Context

Four asks, each of which lands on machinery that already exists. This section says
what is there today, verified against the source, so that each increment below is a
change to a named thing rather than a new subsystem.

**1. Themes.** `TuiTheme` (`packages/phern/src/ph_app/tui/themes/__init__.py:57`) is a
closed set of seventeen color roles; a theme is JSON, parsed by `parse_theme`
(`:115`), and the built-ins load through the *same* parser as a user's own — the
module's opening rule, and the reason a user theme cannot hit a validation path the
shipped ones never exercise. `load_user_themes` (`:158`) globs `$PH_HOME/themes/*.json`
(`:169`); `ThemeCatalog` (`:180`) merges built-ins and user themes and is read once at
start. `/theme` is `action_open_themes` (`app.py:690`) → `theme_choices`
(`modals/pickers.py:77`) → `ChoicePicker` with live preview as the cursor moves →
`_set_theme` (`app.py:703`), which persists the pick into `$PH_HOME/tui.json`.

Three themes ship: `ph-dark`, `ph-light`, `high-contrast`. **tau ships no catppuccin
either** — what tau has, and pH does not, is a theme *format* rich enough to express
one legibly: a `vars` palette block whose names are substituted into every color
field (`sources/tau/src/tau_coding/tui/themes/__init__.py:195`), and a parser that
reports every problem in one message rather than dying on the first. A catppuccin
theme is twenty-six colors drawn from a fourteen-color palette; written without the
indirection it is twenty-six hex literals with no way to see that `base` was used in
four places.

**2. Prompt history.** There is none. `PromptInput._decide` (`widgets/prompt.py:153`)
claims `up`/`down` **only while a completion list is open** (`:172` returns early when
`self._completions is None`), so with no completions the `TextArea` moves the cursor,
which is what a multi-line box must keep doing. Every prompt the person has sent is
already in the client's mirror of the log as a `ChatItem` with `role == "user"` and a
`seq` (`state.py:150`), and the transcript can already scroll to a seq —
`TranscriptView.scroll_to_seq` (`widgets/transcript.py:523`), reached by the
`RevealSeq` message a registered screen posts (`app.py:613`).

**3. Startup session list.** `PHTuiApp._open` (`app.py:279`) asks for trust, starts or
finds a daemon, connects, and attaches to `self.session_id or new_session_id()`
(`:300`) — so a bare `phern` always opens a new session and the previous work in this
directory is reachable only by pressing `ctrl+r` afterwards. The listing verb exists
(`sessions/browse`, `verbs.py:143`), the fold behind it already merges stored logs and
live roots (`daemon/projections.py:243`), and every row already carries the `cwd` its
header recorded (`sessions.py:58`). What is missing is a filter — `session_summaries`
(`sessions.py:97`) takes a limit of 50 and no directory — and a moment to ask.

**4. Daemon lifetime.** `DaemonServer.spent()` (`daemon/server.py:1257`) is
`ephemeral` **and** no open connections **and** `supervisor.unwanted(after=EPHEMERAL_QUIET)`
(`supervisor.py:1061`), where `unwanted` requires every root to be `passivatable`
(`supervisor.py:1406`) — which includes *sixty seconds of log quiet* — and the books
to be empty. It is evaluated on the sixty-second sweep (`server.py:1276`). So today,
closing the TUI leaves the daemon resident for **sixty to a hundred and twenty
seconds**, and the person has no way to see that, because nothing on the wire says
anything about the process's own life: the sidebar draws session id, sandbox and cwd
(`widgets/status.py:244`) and nothing else.

---

## Decisions taken with the user (2026-09-16)

1. **The theme profile is the user's, in `$PH_HOME`, and is written lazily.** Not a pH
   profile row and not a daemon-side setting: a front end may be on another machine,
   and the colors belong to the person at the terminal. **There is nothing to set up
   before the first run** — `DEFAULT_THEME` applies, and the file comes into existence
   the first time `/theme` picks something, or when the person writes one by hand.
   **Precedence, not migration** (2026-09-16, second pass): the YAML decides, else the
   default. `tui.json`'s `theme` key is not read as a fallback — pH is unreleased and
   the only file in existence holds the default, so the clause could never change an
   outcome. `TuiSettings` loses the field so that `tui.json` cannot go on recording a
   theme it no longer decides; it keeps the keybindings, sidebar, bell and view
   toggles, which are its own.

   `profiles/tui.yaml` is a different file and is not involved: it is the pH *plugin*
   profile — the sandbox row, the trajectory screen, arming `tool-ask-user` — and holds
   no color. Said here because the two names are one word apart.
2. **YAML, alongside JSON.** `$PH_HOME/themes/*.yaml` and `*.json` both load, through
   one parser. The profile itself is `$PH_HOME/themes/theme-profile.yaml`. It does not
   go in `$PH_HOME/profiles/`, which the profile loader lists for `--profile` names —
   a theme document there would offer itself as a plugin profile and be refused at
   compose time.
3. **Prompt history is this session only.** Every row is in the client's own mirror,
   so there is no wire verb and no second store, and every row can be revealed in the
   transcript — which is the half of the ask that a shell-style history file could not
   satisfy.
4. **The startup picker leads with "start a new session".** Shown only when this
   directory has prior sessions; `enter` is today's behavior, the arrow keys resume.
5. **Only an ephemeral daemon leaves; the keep-alive defaults to zero.** `phern daemon`
   typed at a prompt is still a service and still stays. One a UI spawned exits as
   soon as the last front end detaches, unless a task is running, a schedule is on the
   books, or a keep-alive was asked for.

---

## Phase 9 rows

| Row | What it lands | Depends on |
|---|---|---|
| P9-01 | **Landed.** Themes in YAML or JSON, a `vars` palette, all problems reported at once, catppuccin ×4 | — |
| P9-02 | **Landed.** `$PH_HOME/themes/theme-profile.yaml` — sole owner of the theme, written by `/theme`, absent by default | P9-01 |
| P9-03 | **Landed.** Prompt history on the arrow keys, searched, and revealed in the transcript | — |
| P9-04 | **Landed.** `sessions/browse` filtered by working directory | — |
| P9-05 | **Landed.** The startup session picker | P9-04 |
| P9-06 | **Landed.** An ephemeral daemon exits when the last front end detaches | — |
| P9-07 | **Landed.** The lifetime on the wire and in the side panel | P9-06 |
| P9-08 | Docs, non-guarantees, `Implementation_Plan.md` §4 | all |

`plans/Implementation_Plan.md` §4 gains a **Phase 9 — the front end a person lives in**
section carrying these rows; P9-08 is where that edit lands.

---

## P9-01 — themes as data, in either notation, with a palette

> **Landed** (2026-09-16). `themes/__init__.py` gained `_decode`, `_palette`,
> `_color_problem`, `theme_document` and a derived `COLOR_ROLES`; four
> catppuccin flavors ship as YAML beside the three JSON built-ins.
> `mypy` clean on 442 files, 183 TUI tests green including the ten SVG snapshots.
>
> **One deviation, and it removes code rather than adding it: no Rich-keyword
> reserve list.** The plan said to port tau's refusal of a var named `on`, `bold`
> or `dim`. That rule guards *token-wise* substitution into Rich style strings,
> which is a hazard tau has because its role values may be `bold #061a1a on
> #a7f3f0`. Every pH role is a single color, so substitution is whole-value and a
> var named `on` cannot corrupt anything — porting the rule would have been
> carrying a guard for a hazard this code does not have. The reasoning is in the
> comment at the substitution, so the next person to compare the two files finds
> it there rather than wondering.
>
> Two smaller notes. `LoaderError` is imported from `ph.cordis`, not
> `ph.cordis.loader` — the latter is not an explicit export and `mypy --strict`
> says so. And `pyyaml` is now declared in `packages/phern/pyproject.toml`: P9-02
> writes YAML, `cli.py` already imported it directly, and arriving here through
> `ph-core` understated the coupling (`anyio`'s comment, one dependency up, makes
> the same argument).

**Files:** `packages/phern/src/ph_app/tui/themes/__init__.py`, four new theme files in
that directory, `packages/phern/tests/test_tui_themes.py` (new).

**The parser keeps its shape and gains three things.**

*One document, two notations.* A private `_load_document(path_or_resource)` dispatches
on the suffix: `.yaml`/`.yml` through `ph.cordis.loader.safe_yaml_load`, `.json`
through `json.loads`. `safe_yaml_load` already answers a `JsonValue` tree with no
custom tags and no implicit date coercion, so `parse_theme` is unchanged below the
decode — which is the point. `load_user_themes` globs all three suffixes; `_builtins`
iterates the package directory as it already does. A directory holding both
`mocha.yaml` and `mocha.json` is a person's own ambiguity: the YAML wins and the JSON
is skipped with a log line, rather than the two racing on dictionary order.

*A `vars` palette.* `parse_theme` accepts an optional top-level `vars` mapping of
name → single color, substituted token-wise into every role value before validation —
tau's `_parse_vars`/`_substitute_vars`, ported for the reason the module header already
gives for having ported the rest. Two rules come with it: a var value must be a single
token that parses as a color, and a var *name* may not be one of Rich's style
keywords (`on`, `bold`, `dim`, …), because substitution happens by token and a var
named `on` would corrupt every value it touched.

*Every problem at once.* `parse_theme` today raises on the first fault
(`themes/__init__.py:123`). Against a hand-written catppuccin file — twenty-six roles
over a fourteen-color palette — that is twenty-six edit-and-retry cycles. It collects
into a `problems` list and raises one `ThemeError` naming all of them, as tau's does.
The refusals themselves do not change: a missing role, an unknown role and a
non-boolean `dark` are still refusals, because a typo'd role name that was *ignored*
would leave the real one at its default and read as a rendering bug.

*Color validation at load.* Each resolved role is parsed with
`textual.color.Color.parse`; a value Textual cannot read is a problem on the list
rather than an exception at the first paint. pH's roles reach Textual only — they
become `$ph-*` CSS variables and `Theme` slots (`:85`) — so unlike tau, which also
feeds Rich style strings and therefore validates under both libraries, one check is
the honest one here. Stated because the next person to port a tau field will ask.

**The four new themes ship as YAML with a palette; the three existing ones stay
JSON.** Deliberately mixed: the module's founding rule is that built-ins take no
private path, and a shipped set that is entirely JSON would leave the YAML branch
exercised only by files nobody has written yet.

```yaml
# packages/phern/src/ph_app/tui/themes/catppuccin-mocha.yaml
dark: true
vars:
  base:     "#1e1e2e"
  mantle:   "#181825"
  surface0: "#313244"
  overlay0: "#6c7086"
  text:     "#cdd6f4"
  mauve:    "#cba6f7"
  green:    "#a6e3a1"
  red:      "#f38ba8"
  peach:    "#fab387"
background: base
foreground: text
surface: mantle
panel: mantle
muted: overlay0
border: surface0
accent: mauve
success: green
error: red
warning: peach
# … the remaining seven roles
```

`catppuccin-mocha`, `catppuccin-macchiato`, `catppuccin-frappe` (dark) and
`catppuccin-latte` (light). `BUILTIN_THEME_NAMES` is already derived from the shipped
files (`:154`), so a fifth is a fifth file.

**Guarantee.** A theme is refused at load, with every fault named in one message, or it
renders. A palette name used in four places is edited in one.

**Gates** (`test_tui_themes.py`): `test_a_yaml_theme_and_its_json_twin_parse_identically`
(sabotage: drop the suffix dispatch and the YAML never loads) ·
`test_a_palette_name_substitutes_into_every_role` ·
`test_a_var_named_after_a_rich_keyword_is_refused` (sabotage: drop the keyword set) ·
`test_a_theme_with_three_faults_names_all_three` (sabotage: restore the first-fault
raise) · `test_every_shipped_theme_parses_and_every_role_is_a_color_textual_reads` —
which is the one that holds the four new files honest, and is written over
`_builtins()` so a fifth theme is covered by existing.

---

## P9-02 — the theme profile, created on the first pick

> **Landed** (2026-09-16). `ThemeProfile`, `load_theme_profile`,
> `save_theme_profile`, `theme_profile_path` and `choose_theme` live in
> `themes/__init__.py` rather than a sibling `profile.py` — the profile needs
> `DEFAULT_THEME` and `load_user_themes` needs `PROFILE_FILENAME`, which is an
> import cycle for a file that is data either way. `TuiSettings.theme` is gone;
> `app.py`, `trajectory_app.py` and `theme_choices` read the profile.
>
> **The theme tests moved rather than multiplied.** The plan added
> `test_tui_themes.py` beside the five theme tests already in `test_tui_state.py`,
> which would have left theme coverage in two files. The section moved wholesale
> and the new gates joined it there; `test_tui_state.py`'s docstring says where it
> went.
>
> **Two ph-core changes came out of the cleanup pass and are not P9's own.**
> `ph.documents` (`read_document`/`decode_document`/`DOCUMENT_FAULTS`) replaces the
> read-text-decode-fall-back ladder that `tui.json`, `settings.json` and the theme
> profile had each written separately, with drift: two of the three would have
> crashed on a non-UTF-8 byte, and none of them caught `LoaderError`, which is what
> `safe_yaml_load` actually raises.
>
> **The libyaml swap was tried and reverted, and the attempt paid for itself.**
> Basing `SafeRowLoader` on `CSafeLoader` measured 8x on a real theme document with
> identical output (`_builtins()`: 5.88 ms → ~1.1 ms), but subclassed with this
> module's two customizations it makes `Resolver.resolve` answer `None` for every
> node kind — every tag undefined, the first profile load dead with "could not
> determine a constructor for the tag None". It reproduces **only while `coverage`
> is tracing this module's import**, which is to say under CI's own `--cov` gate,
> at collection, and never in an ordinary run. Each ingredient is fine alone: a
> bare `CSafeLoader` subclass built under coverage parses, and so does one carrying
> the resolver rebuild. Not understood, so not shipped — the reasoning sits in
> `SafeRowLoader`'s docstring so the next person measuring YAML cost finds it
> before repeating the experiment.
>
> What the attempt *did* find is a real bug, kept: the timestamp stripping was
> item-assigning into `yaml_implicit_resolvers`, a dict PyYAML shares across every
> loader it defines, so pH had been removing date parsing from `yaml.safe_load`
> **process-wide** — for itself and for any library in the same interpreter. It now
> rebuilds the table on the subclass, as PyYAML's own `add_implicit_resolver` does,
> and `test_the_row_loader_does_not_take_timestamps_from_everyone_else` holds it,
> with a sabotage note recording that the pre-existing timestamp test kept passing
> the whole time it was broken.

**Files:** `packages/phern/src/ph_app/tui/themes/__init__.py` (or a sibling
`profile.py`), `tui/app.py`, `tui/trajectory_app.py`, `tui/modals/pickers.py`,
`tui/config.py`.

**The file does not exist until something writes it, and its absence is a complete
answer.** `DEFAULT_THEME` applies, `/theme` lists the built-ins and whatever is in
`$PH_HOME/themes/`, and nothing asks the person to configure anything to start.

```yaml
# $PH_HOME/themes/theme-profile.yaml — written by /theme, or by hand
default: catppuccin-mocha
dark: catppuccin-mocha
light: catppuccin-latte
order:                      # listed first in /theme, in this order
  - catppuccin-mocha
  - catppuccin-latte
  - ph-dark
```

`ThemeProfile` is a frozen dataclass with `default`, `dark`, `light` and `order`, all
optional. `load_theme_profile(home)` returns an empty one for a missing file and for an
unreadable one — the rule `tui.json` already lives by (`config.py:11`): a broken
preference file costs the customization, not the session. A name in `order` that no
longer resolves is dropped from the ordering rather than refusing the file.

**Precedence, and nothing else: the YAML, else `DEFAULT_THEME`.** There is no migration
step and no legacy read, because there is nothing a legacy read could change. The
`theme` key has been in `tui.json` since 2026-09-11, pH is unreleased (no tags, version
`0.1.0`), and the one file in existence holds `ph-dark` — which *is* `DEFAULT_THEME`. A
fallback clause reading it could not produce a different answer in any reachable state,
and a knob wired to nothing is what §5 rule 6 exists to keep out. Changing to `ph-light`
is exactly what the user described: write `ph-light` into `theme-profile.yaml`, and the
YAML is the file that answers.

**`TuiSettings` loses the field, which is what makes the YAML the only writer.**
`save_tui_settings` persists the whole dataclass through `asdict` (`config.py:103`), so
a `theme` left on it would be rewritten into `tui.json` by every *unrelated* preference
change — a `/view thinking` toggle updating the copy that no longer decides anything.
Two files claiming one fact, the losing one still being maintained, is the drift this
codebase spends its comments on. So the field goes: `tui.json` keeps the keybindings,
the sidebar side, the bell and the four view toggles — which it genuinely owns — and the
theme lives in the YAML. The three readers of `settings.theme` (`app.py:248`,
`app.py:707`, `trajectory_app.py:158`) read the profile instead; `_set_theme`
(`app.py:703`) writes it.

**Nothing has to be cleaned up.** `tui_settings_from_json` is already tolerant of keys
it has no field for — *"an older pH wrote fewer keys, a newer one writes more, and
neither should make the other refuse to start"* (`config.py:118`) — so the leftover
`"theme"` in an existing file is ignored on read and gone on the next write, with no
upgrade step and no code to delete later.

`theme_choices` takes the profile and emits the ordered names first, then the rest
alphabetically. Ordering says nothing but position — there is no second word for
it, and `default` is already taken by what the profile records. Exactly one row
carries `default`: the theme this profile opens in, which is not the same row as
the `marked` dot, since the dot follows the cursor while the picker previews. The
`dark`/`light` pair is carried and read by nothing in this row — it is what a later
`/theme toggle` or a terminal-appearance follow would use, and shipping the field
without the verb is called out here rather than implied.

> **The `dark`/`light` pair was removed after review** (2026-09-16). Two reviewers
> flagged it independently as a knob wired to nothing, and the feature it was
> storage for does not exist: Textual 8.2.8 has no OS-appearance detection —
> `action_toggle_dark` flips between two hardcoded theme names for backwards
> compatibility, and `ansi_theme_dark`/`light` are ANSI color mapping. Following
> the terminal's appearance means querying it (OSC 11 plus luminance, which tau's
> `_is_dark_background` already computes) rather than reading a platform setting,
> because this front end can be a browser tab or a terminal on another machine —
> "the OS" has to mean the *client's*. The fields come back with the row that
> reads them, which costs nothing: `load_theme_profile` reads past a key it has no
> field for, so the two directions stay compatible.

**Guarantee.** A first run has no theme file and needs none. One `/theme` pick creates
`$PH_HOME/themes/theme-profile.yaml`, and from then on that file is the only thing that
decides the theme and the only thing that records a pick.

**Gates:** `test_a_first_run_with_no_profile_uses_the_default_theme` ·
`test_the_first_theme_pick_writes_the_profile` (sabotage: write `tui.json` instead, and
the file the docs name never appears) ·
`test_an_unrelated_preference_change_does_not_write_a_theme` — `/view thinking`, then
assert no `theme` key in `tui.json`; the one that pins the single writer, and it fails
the moment the field is put back ·
`test_a_leftover_theme_key_in_tui_json_is_ignored` ·
`test_an_unreadable_profile_costs_the_ordering_and_not_the_session` ·
`test_the_profile_order_leads_the_picker` — the last driven through `theme_choices`,
not through the screen, for the reason `test_tui_screens` exists.

`test_tui_pilot.py:414` asserts today that a pick lands in `tui.json`; it is rewritten
against the YAML rather than deleted, since it is the end-to-end half of the same claim.

---

## P9-03 — prompt history on the arrow keys, and where it sits in the log

> **Landed** (2026-09-16). `TuiState.prompt_history()` folds the `user` rows;
> `PromptRecord` carries text, seq and turn; `PromptInput` gained a `_Walk` and
> `history_source`; `/history` and `history_search` open a picker whose
> `on_highlight` reveals. Seven gates, two sabotage-checked.
>
> **One design bug, caught by its own test.** The first version guarded "this
> widget is writing the box" with a flag set and cleared around the write — but
> `on_text_area_changed` arrives on the **message pump**, so the flag was already
> back to `False` when the handler ran, and every recall looked like the person
> typing and ended the walk it had just started. The second `up` did nothing. It
> is now a remembered *value* — the last text the walk put in the box, compared
> against what the box holds — which does not depend on when the handler runs.
>
> **Two gates were wrong before they were right, both because the draft is the
> filter.** `test_the_draft_survives_a_walk_through_history` typed a draft that
> was a prefix of nothing, so no walk began and nothing was preserved — correct
> behavior, wrong expectation. And
> `test_up_arrow_inside_a_multi_line_draft_still_moves_the_cursor` *passed under
> its own sabotage*: with a two-line draft matching no history, the filter stopped
> the recall whatever the cursor check did. It now seeds a prompt the draft is a
> prefix of, which isolates the cursor check and fails when it is removed.

**Files:** `tui/widgets/prompt.py`, `tui/app.py`, `tui/config.py`, `tui/commands.py`,
`tui/modals/pickers.py`, `packages/phern/tests/test_tui_pilot.py`.

**Three keys, all configurable, none compared as a literal.** `TuiKeybindings` gains
`history_previous` (`up`), `history_next` (`down`) and `history_search` (`alt+r` —
`ctrl+r` is the session picker). The widget reads `self.keys`, as everything in this
file already does; the rule that a rebound key must work is the reason `prompt.py`
exists in the shape it does.

**Up-arrow means history only where it cannot mean editing.** `_decide` claims
`history_previous` when the completion list is closed **and** the cursor is on the
first line of the box; `history_next` when it is on the last line. Anywhere else the
`TextArea` keeps the key and moves the cursor — a multi-line prompt whose up-arrow
stopped navigating would be a worse box than one with no history.

**The provider is a callable, like completions.** `PromptInput` already takes
`completion_source` (`prompt.py:96`); it gains `history_source`, and `PHTuiApp` supplies
one that folds `front.state.items` to the `role == "user"` rows, newest first, dropping
consecutive duplicates, each carrying its `text` and its `seq`. Nothing is stored: the
mirror *is* the history, which is what keeps it correct across a resume and free at
every other moment.

**Walking it.** Index −1 is the draft — whatever was typed before the first up-arrow —
kept and restored when the walk comes back down past the newest entry, so a half-written
prompt is never eaten. **Typing filters**: with a non-empty draft, the walk visits only
prompts that start with it, which is the behavior every shell has and the reason the
ask says "search" rather than "cycle".

**Searching, and seeing where it was.** `history_search` (and `/history`, a new
`TuiVerb`) opens a `ChoicePicker` over the same rows — one filter semantics, because
`Choice.matches` is already the one everything else uses. Each row's `detail` is its
turn and time; choosing one inserts it into the prompt. The second half of the ask —
*view these as they are recorded in the log* — is the picker's `on_highlight`: moving
the cursor scrolls the transcript to that row's `seq` through the existing
`RevealSeq` → `on_reveal_seq` → `scroll_to_seq` join (`app.py:613`,
`widgets/transcript.py:523`), so the conversation moves behind the picker as you walk
your own prompts. No new mechanism; the same one a registered screen uses.

**Where it honestly cannot work.** A client whose mirror diverged
(`FrontSession.diverged`, `frontend.py:161`) holds a prefix of the log with no way to
know how short. History from it would silently be partial, so `/history` says so —
the wording `action_open_screen` already uses for the same condition — and the arrow
keys fall back to editing.

**Guarantee.** Every prompt a person sent this session is reachable from the prompt box
without touching the mouse, in the order they sent them, and choosing one shows where
it sits in the conversation.

**Gates** (pilot tests, driven by keystrokes, as `test_tui_pilot.py` requires):
`test_up_arrow_recalls_the_previous_prompt` ·
`test_up_arrow_inside_a_multi_line_draft_still_moves_the_cursor` (sabotage: claim the
key unconditionally) · `test_the_draft_survives_a_walk_through_history` ·
`test_a_typed_prefix_filters_the_walk` ·
`test_history_search_reveals_the_chosen_prompt_in_the_transcript` (sabotage: drop the
`on_highlight` wiring and the transcript never moves) ·
`test_a_diverged_client_says_so_rather_than_offering_a_short_history` ·
`test_a_rebound_history_key_is_the_one_that_works` — the rule this file is built to
enforce, asserted rather than assumed.

---

## P9-04 — `sessions/browse`, filtered by working directory

> **Landed** (2026-09-16). `BrowseParams(cwd, limit)` — every field defaulted, so
> the bare `{}` every older client sends is unchanged and `PROTOCOL_VERSION` did
> not move. `session_summaries` filters *during* the scan, so `limit` bounds the
> matching rows rather than the files looked at; `browse_of` applies the same test
> to the live roots it merges; `CAPABILITIES` gained `browse-cwd`.
>
> **The scan's cost was measured afterwards, in review, and needed a fix.**
> Stopping at `limit` *matches* means a directory with fewer than fifty sessions —
> nearly every directory — walks the whole store, and that now runs before the
> first prompt. `_summarize` therefore takes the `cwd` and returns as soon as the
> header fails it, before the title scan that is the expensive half of a row it
> would discard: measured 28% off the filtered scan at 500 logs. The remaining
> bound is the whole store, which has no retention, so if that ever stops being a
> few hundred the answer is a per-directory index rather than a faster scan.
>
> Four gates. The one the row was written for —
> `test_browse_filtered_by_cwd_finds_a_session_older_than_the_limit` — is
> sabotage-checked against filtering after the limit, which is the mistake that
> makes a month-old repo list as empty on a busy machine.

**Files:** `ph_app/params.py` (or wherever `NoParams`' siblings live), `ph_app/verbs.py`,
`ph_app/daemon/server.py`, `ph_app/daemon/projections.py`, `ph_app/sessions.py`,
`ph_app/tui/remote.py`, `ph_app/tui/frontend.py`.

`SESSIONS_BROWSE` moves from `NoParams` to `BrowseParams(cwd: str = "", limit: int = 50)`.
Every field is defaulted, so a client that sends `{}` — every client before this row —
behaves exactly as it does now, and `PROTOCOL_VERSION` does not move.

**The filter goes where the summaries are folded, not after them.** `session_summaries`
(`sessions.py:97`) sorts by mtime and takes the newest `limit`; filtering the *result*
by `cwd` would let a directory's older sessions fall off the fifty-row window before
the filter ever saw them, so a repo worked in last month would list as empty. It takes
a `cwd: str = ""` and skips non-matching headers during the scan, before the limit
applies. The comparison is on the recorded string as the header wrote it — no
`resolve()`, because the daemon may not have that path and a symlinked checkout is the
person's own business; a mismatch costs a row in the picker and `/sessions` still lists
everything.

`browse_of` (`projections.py:243`) threads the parameters through and applies the same
filter to the live roots it merges in. `CAPABILITIES` (`server.py:281`) gains
`browse-cwd`, so a client can tell "this daemon filtered" from "there is nothing here"
rather than inferring it from an empty list. `FrontSession.browse_sessions` grows
`cwd: str = ""`.

**Guarantee.** Asking for one directory's sessions returns that directory's sessions,
however far back they go, and asking for none returns what it returns today.

**Gates:** `test_browse_filtered_by_cwd_finds_a_session_older_than_the_limit`
(sabotage: filter after the limit, and it fails) ·
`test_an_unfiltered_browse_is_unchanged` · `test_a_live_root_in_this_directory_is_listed_with_its_status`
· `test_a_client_that_sends_no_params_still_gets_every_session` — the compatibility
claim, asserted over a real socket in `test_daemon_methods`' style.

---

## P9-05 — the startup session picker

> **Landed** (2026-09-16). `_offer_sessions` sits between the daemon connect and
> the attach; `session_choices(offer_new=True)` leads with the new-session row;
> `browse_before_attach` keeps the wire call in `remote.py`, since
> `FrontSession.browse_sessions` is a method on the session this picker is
> deciding. `--new` skips it. Six gates, the empty-directory one sabotage-checked.
>
> **The row hit a gap the plan did not see, and it was fatal to the feature as
> written.** `Supervisor.sessions_directory()` answered by asking a *mounted
> root's* store — and a daemon that has just started holds none, so
> `sessions/browse` before the first attach returned an empty list. The picker
> runs before the first attach by definition, so its main case — open pH in a
> directory you have worked in, on a daemon that just came up — could never have
> shown anything. Verified rather than reasoned: `roots: 0 →
> sessions_directory(): None → browse_of: []`, resolving only after a root mounts.
>
> The fix is a fallback to the daemon's **own** `$PH_HOME/sessions`, and it is
> narrower than it looks: that is the literal default
> `session-persistence-jsonl` computes (`resolve_roots().sessions_dir()`), so in
> any deployment that has not overridden the row it is the same path the store
> would have named — and `supervisor.py:664` already derived it that way. A
> deployment that *did* set `session-persistence-jsonl.root` gets a cold browse of
> an absent directory, listing nothing, until its first root mounts. That
> non-guarantee is recorded on the method.
>
> **The gates' fixture is a header-only log, on purpose.** Hand-writing a
> transcript is a race against invariants that are right to exist — an event needs
> `seq` and `time`, a log must start at seq 0, a surface-eligible event needs its
> `surfaceOp` marker — each of which refused a draft of `_stored` in turn. A
> header and no events is the shortest log that both lists and resumes.

**Files:** `tui/app.py`, `ph_app/cli.py`, `tui/modals/pickers.py`,
`packages/phern/tests/test_tui_pilot.py`.

In `_open` (`app.py:279`), after the daemon answers and before `attach_session`: when
`self.session_id is None`, browse this working directory; with no rows, proceed exactly
as today; with rows, `push_screen_wait` a picker whose first row is `start a new
session`, pre-selected. `enter` is today's behavior; `escape` is the same; the arrow
keys resume. Awaiting a modal is legal here and only here — `_open` runs in a worker,
which is the rule `frontend.py` states and `PHTuiApp` is shaped around.

**After trust, before the attach.** Trust is asked first because mounting is what reads
the project's `AGENTS.md`, its hooks and its configured plugins (`app.py:264`); the
picker needs a daemon to ask, and the attach is what it decides.

`session_choices` (`pickers.py:111`) already shapes the rows — segments contracted,
forks indented — and gains a leading `Choice` for the new session. `free_text="session id"`
stays, so a session id from elsewhere can still be typed.

**Two ways past it**, both for people who never want it: a new `phern --new` flag skips it
outright, and `phern --session <id>` / `--resume <id>` already bypass it by giving
`session_id` a value.

**Guarantee.** Opening pH in a directory you have worked in offers that directory's
work, and one keystroke gets the behavior you have today. Opening it anywhere else is
unchanged.

**Gates:** `test_a_directory_with_no_history_opens_straight_into_a_new_session`
(sabotage: show the picker unconditionally) ·
`test_enter_on_the_first_row_starts_a_new_session` ·
`test_choosing_a_stored_session_attaches_to_it` ·
`test_the_picker_lists_only_this_directory` ·
`test_new_skips_the_picker` · `test_escape_starts_a_new_session_rather_than_exiting` —
the failure mode worth pinning, because a picker that quits the app on escape is a
worse first run than no picker.

---

## P9-04b — the lineage directory carries the working directory (format 1)

> **Landed** (2026-09-17), taken with the 0.2.0 bump and no backwards
> compatibility, by decision: existing sessions are discarded rather than
> migrated.
>
> **What changed.** A root's `family` — which *is* the directory its log lives in
> — is now `<cwd-tag>-<id>`, six hex of the sha256 of the working directory. It is
> derived in `SessionHeader`'s own `default_factory` from the `cwd` already beside
> it, and inherited unchanged by every fork, segment and subagent through the one
> construction gate in `SessionStore.create`. `family_dirs(root, tag=…)` then skips
> a whole repo's lineages **without a `scandir` of their files, without a `stat`,
> and without opening one**. `SESSION_FORMAT_VERSION` moves 0 → 1, so a 0.1.x log
> is refused loudly rather than going quietly unfindable.
>
> **Measured, 500 sessions, one of them this repo's: 9.66 ms → 0.21 ms (46x).**
> That is far better than the 3.46 ms predicted for the shape originally proposed
> — a tag in the *session id*, filtering on filenames. The prediction was right for
> that shape and wrong for this one: a filename filter still has to `stat` every
> log to sort by mtime, so it lands on the listing floor, while a directory filter
> never reaches the files at all. The complexity genuinely changed —
> O(all sessions) → O(families) + O(this repo's sessions) — which the id version
> could not have done.
>
> **Why the directory and not the id.** Session ids stay opaque: they cross the
> wire, appear in `--session`, and people type them, so putting derived,
> unverifiable, unrepairable content in one buys a constant factor at the cost of
> a second source of truth for where a session belongs. The family was already a
> directory and already inherited — the tag rides a mechanism that existed.
>
> **The tag is a filter, not an identity.** 24 bits collide; the header's own `cwd`
> still confirms every match, so a collision costs one directory scan and never a
> wrong row. A session created with no cwd stays untagged and is found by no
> directory search, which is the honest answer rather than a placeholder.
>
> **It surfaced a latent bug in the test builders.** `reference_fork` defaulted a
> child's family to the parent's *id*, justified by "forking a root, whose family
> is its own id" — true until now, and invisible because the two strings were
> equal. A fork of a tagged root would have been filed in a directory of its own.
> Fixed at the builder and its call site, and
> `test_a_lineage_keeps_its_working_directory_tag` pins the inheritance.
>
> **Gates:** `test_a_filtered_listing_opens_no_other_directorys_logs` — counts
> `_summarize` calls, because the whole value is the reads that do not happen
> (sabotage: drop the `tag=`, and every log is opened) ·
> `test_a_session_with_no_cwd_belongs_to_no_directory` ·
> `test_a_lineage_keeps_its_working_directory_tag` ·
> `test_a_session_with_no_working_directory_is_untagged`.

---

## P9-06 — an ephemeral daemon leaves when the last front end does

> **Landed** (2026-09-17). `holds()` replaces `spent()`'s yes/no with the list of
> reasons — `client`, `task`, `schedule`, `keep-alive` — and `spent()` is
> `ephemeral and not holds()`. `Supervisor.busy()` and `booked()` answer the two
> supervisor-side terms. `check_lifetime()` runs on the connection transition and
> on the sweep. `--keep-alive` on `phern daemon` and `phern`, `daemon_keep_alive` in
> `tui.json`, forwarded through `spawn_command`. Eight gates, every one
> sabotage-checked.
>
> **`EPHEMERAL_QUIET` and `Supervisor.unwanted` are deleted, not retuned.** The
> plan kept the constant "on the root half"; it had no root-half job. The sweep
> has always released roots on `passivate_after`, and the exit was the only
> reader — so what survived the change was a named sixty seconds nothing
> consulted, which is the shape of a knob wired to nothing. `unwanted` went with
> it for the same reason: its one caller was the old `spent`.
>
> **The row opened a hole the plan did not see, and the launch suite caught it.**
> `launch.listening()` — a connect and an immediate close — is how every spawn,
> every stale-socket check and every `_await_socket` poll asks whether a daemon is
> there. Reading that close as "the last client left" stopped the daemon the poll
> had just declared ready, so the UI that started it connected to nothing:
> `test_daemon_launch.py` failed three ways, none of which named this line.
>
> The fix is in the predicate, not in one of its callers. `_Connection.spoke` —
> the first frame is what makes a connection a client rather than a knock — keeps
> one job, arming the keep-alive window, because that window is about a *client*
> leaving. The exit reads `DaemonServer.served`, the same question about the
> process: an auto-started daemon may not leave until somebody has spoken to it
> or `launch.SPAWN_TIMEOUT` has passed, which is the launcher's own number, so
> after it there is nobody left to protect.
>
> Guarding the teardown alone would have left the other two callers — the sweep,
> and a scheduled root finishing a turn inside the spawn window — to find the
> same hole by another door. In the predicate it also closes the case a
> per-connection guard could not: a knock still open when the last real client
> leaves now ends the daemon when it closes, rather than waiting for a sweep.
>
> `served` is in `spent()` and deliberately not in `holds()`: a daemon waiting
> for the client that spawned it is not *held* by anything — nobody could name a
> reason — it simply may not go yet. Among the reasons it would have put a
> `starting` on the wire and `held · starting` in a sidebar for the first thirty
> seconds of every session.

**Files:** `daemon/server.py`, `daemon/recovery.py`, `ph_app/cli.py`,
`tui/config.py`, `packages/phern/tests/test_daemon_lifetime.py`.

**The exit predicate stops asking about quiet and starts asking the three questions the
person named.** `spent()` (`server.py:1257`) becomes:

1. `ephemeral` — unchanged, and still the condition that cannot be derived from the
   others: an idle service daemon is indistinguishable from a spent ephemeral one from
   the inside.
2. **no open connection** — unchanged, and still *connected* rather than *attached*, so
   a `phern agents doctor` between its request and its reply is never hung up on.
3. **no root is working** — any root whose `status` is not `idle` or `waiting` holds the
   process. This is `passivatable`'s first clause (`supervisor.py:1431`) and nothing
   else from it: the sixty-second quiet window is about releasing a *root*, and reusing
   it here is what put a minute between the last detach and the exit.
4. **nothing is on the books** — `supervisor.appointments()` non-empty, or any mounted
   root with a live schedule, keeps it up. Unchanged in meaning.
5. **the keep-alive has expired** — `keep_alive_until`, set when the last connection closes,
   `None` when the keep-alive is zero.

`EPHEMERAL_QUIET` keeps its job on the root half — an ephemeral daemon still releases
its roots aggressively, because it intends to leave — and its docstring gains the
sentence saying it is no longer the exit's window.

**Evaluated on the event, with the sweep as backstop.** `_handle`'s `finally`
(`server.py:1299`) already runs when a connection ends; it calls a new
`check_lifetime()` which either sets `stop` or arms the keep-alive deadline. The same call
happens when a turn ends and when a schedule is canceled — the two other transitions
that can make a held daemon unheld. The sixty-second sweep keeps calling it, for the
case nothing else can cover: a keep-alive that expires with no event to notice it. This is
the plan's own argument for one cadence per question, honored — the cadence is not a
new timer, and the prompt path is not a poll.

`DaemonServer.open_connections: int` (`server.py:1096`) becomes
`connections: set[_Connection]`. The count answered one question; the set answers two,
the second being *who to tell* in P9-07, and `spent()` reads `not self.connections`.

**The keep-alive.** `phern daemon --keep-alive <duration>` and `phern --keep-alive <duration>`
(forwarded through `spawn_command`, `cli.py:656`, beside the `--ephemeral` it already
spells), plus `daemon_keep_alive` in `$PH_HOME/tui.json` so a person sets it once.
Precedence is CLI, then `tui.json`, then zero. The client's preference reaching the
daemon through the argv the client composes is the only route that works — a daemon
may not read a front end's preference file, and after P5-14 it might not be on the
same machine.

Service daemons are untouched: `phern daemon` without either flag stays, and
`--keep-daemon` still means "the one you spawn is a service".

**Guarantee.** Closing the last front end ends an auto-started daemon at once, unless a
task is running, a schedule is on the books, or a keep-alive was asked for — and in each of
those cases, *which* one is a fact the process can state.

**Gates** (`test_daemon_lifetime.py`, extending the existing module):
`test_an_auto_started_daemon_exits_when_the_last_connection_closes` (sabotage: leave the
check on the sweep alone, and it takes a minute) ·
`test_a_running_turn_holds_an_ephemeral_daemon_past_the_last_detach` ·
`test_an_indexed_appointment_holds_it` (already covered; re-pinned against the new
predicate) · `test_a_keep_alive_holds_it_for_exactly_as_long_as_it_says` ·
`test_the_keep_alive_expires_without_a_client_to_notice` — the backstop, driven by the
sweep with no connection at all · `test_an_explicitly_started_daemon_still_never_exits`
· `test_the_exit_no_longer_waits_on_the_root_quiet_window` (sabotage: put
`EPHEMERAL_QUIET` back into `spent`).

---

## P9-07 — the lifetime on the wire, and in the side panel

> **Landed** (2026-09-17). `DaemonNotice` is a family of its own beside
> `SessionScoped`, with `DaemonLifetime` its one member. `daemon/lifetime` is
> read once in `attach_session`'s startup group; `daemon.lifetime` is broadcast
> from `_announce_lifetime` to `server.connections` when the answer moves, and
> `DaemonSession.dispatch` handles it above the `sessionId` filter.
> `DaemonServer.lifetime()` is the one builder behind the verb, the notice and
> the doctor's `daemon lifetime` section. Seven gates, every one
> sabotage-checked.
>
> **The sibling table the plan called for is a sibling *type* instead.**
> `DAEMON_NOTICES` and `daemon_notice_of` were written and then deleted: one
> method and one reader do not earn a second mapping, and the reader narrowed
> straight back to the concrete type it had just looked up. What the base class
> earns on its own is the exclusion — `Mapping[str, type[SessionNotice]]`
> structurally cannot hold a process-level frame, so one can never be dispatched
> behind the `sessionId` filter by an author's oversight. A second member is the
> moment to write the table.
>
> **`--keep-alive` implies `--ephemeral`, and the contradiction is refused.** The
> two flags were one concept — when does this daemon leave — split across two
> spellings, and the split is what made `--keep-alive` alone mean *nothing* on
> `phern daemon` and need a warning line to say so. "Stay up five minutes after
> the last client leaves" is a statement about leaving, so it carries the
> lifetime with it.
>
> Merging them into one flag was considered and is not available: typer has no
> optional-value option (`--ephemeral` with no argument is a usage error, and
> with `is_flag=False` it swallows the following flag as its value), and the two
> commands express the lifetime in opposite directions on purpose — `phern
> daemon` opts into leaving, `phern` opts into staying — so a single merged flag
> could not read the same on both. The *duration* is the part that means the same
> thing on both, which is the argument for it staying its own flag.
>
> The implication lives at each call site rather than in the parser, for that
> same reason: a parser that decided the lifetime would have to know which
> command it was serving. `--keep-daemon` with a *typed* `--keep-alive` is now
> refused rather than warned — two opposite answers, of which the person meant
> one. A *configured* keep-alive is not a contradiction and is simply overridden.
>
> **`--linger` is `--keep-alive`.** The name collided with `ph.lingering`, an
> unrelated pre-existing concept — `loginctl enable-linger`, whether
> `$XDG_RUNTIME_DIR` survives logout — and `phern agents doctor` was printing a
> row called `linger` in each of two adjacent sections, meaning different things.
> The collision was not theoretical: it made a doctor assertion stop
> distinguishing what it was written for, and forced the new gate to select rows
> by section title rather than by label. Renamed while both halves are
> unreleased — the flag, `daemon_keep_alive` in `tui.json`, the `keep-alive`
> hold, `keep_alive_ms` on the wire and the `keep alive` row in the doctor.
>
> **`exits_at` and the countdown are cut, and the reason is the mechanism.** The
> keep-alive is armed when the last connection closes and cleared by the first frame
> of the next one — so a client is by construction never connected while a
> deadline is running, and `phern agents doctor` cancels the very window it would
> have printed. The field could only ever have carried `null`. What a connected
> client can say truthfully is not "it leaves at 9:41" but "it leaves five
> minutes after you do", so `keep_alive_ms` stays and `exits_at` never existed.
>
> **The sidebar line answers "if I close this window", not "what is the daemon
> doing".** `client` is filtered out of the reasons before rendering: it is true
> of the front end drawing the line, so printing it unconditionally would spend a
> row of a 32-column panel saying the person has a window open. What is left —
> `held · task`, `held · schedule` — is the set of reasons that survive their
> leaving. A front end with no daemon draws no row at all.
>
> **The filter needed a fact the plan's frame did not carry**, so `clients` is on
> it: `holds` says `client` whenever *anybody* is connected, which reads
> identically with one terminal and with two while the sentence flips. Dropping
> it unconditionally promised `exits on detach` to both of two open terminals and
> was wrong for whichever closed first. With the count, a second terminal reads
> `held · client` — and a client *arriving* is announced as well as one leaving,
> which was the transition nothing had been sent for.
>
> **Two pieces of plumbing the plan did not name.** `Supervisor.recheck_lifetime`
> is a callback `serve` points at `check_lifetime` — named for the question
> because more than one thing raises it, and registered as its own `ctx.on`
> listener beside the others rather than as a line inside `announce`, whose body
> is guarded on there being watchers to announce to. The case that matters most
> has none: a detached `phern -p`, whose daemon should leave when the work it was
> started for is finished. `schedule/create` and `schedule/cancel` call it from
> their handlers, which is the other way `holds()` moves without an event. What
> is *not* wired — a root parking on a person, and a keep-alive expiring — is written
> down in `holds()` itself, with the sweep named as its backstop (§5 rule 6).
>
> And `_duration` moved from `ph_app.agents` to `ph.text.duration`: the sidebar
> renders the same kind of thing, and two copies of a duration format is two
> sentences for one fact.
>
> **The vocabulary is typed.** `DaemonMode` and `Hold` are `Literal` aliases in
> the `TrustAnswer` mould, so `holds()` returning a fifth reason is an error
> where it is written rather than a blank row in somebody's terminal. The cost is
> the ordinary one `PROTOCOL_VERSION` documents: a front end older than a new
> value refuses the frame rather than ignoring it, and the fix is to restart the
> daemon.

**Files:** `ph_app/payloads.py`, `ph_app/verbs.py`, `daemon/server.py`,
`tui/remote.py`, `tui/state.py`, `tui/widgets/status.py`, `ph_app/agents.py`.

**The first frame that is not about a session.** Every notice today is a
`SessionScoped` (`payloads.py:168`) and the client discards anything whose `sessionId`
is not its own (`remote.py:311`) — correct, and exactly wrong for a fact about the
process. So:

* `DaemonLifetime(WireModel)`: `mode` (`"service" | "ephemeral"`), `holds: list[str]`
  drawn from `client`, `task`, `schedule`, `keep-alive`, `exits_at: int | None`,
  `keep_alive_ms: int`.
* `daemon/lifetime` — a `Verb` read at attach, so a client that connects mid-window
  draws the truth rather than waiting for a change.
* `daemon.lifetime` — a notification, broadcast to `server.connections` whenever
  `check_lifetime()` finds a different answer. `NOTICES`/`notice_of` (`payloads.py:618`)
  gain a sibling `DAEMON_NOTICES` table, and `DaemonSession.dispatch` handles the
  method **before** the `sessionId` filter, the way it already handles the fed methods
  (`remote.py:297`). `test_payloads` holds the new table against its own family the way
  it holds the existing one.

**Why not a `StatusReading`.** The readings already flow to the sidebar by slot
(`widgets/status.py:239`) and it would be one line to put this among them. It would
also be wrong: a reading is *a fold of one session's log, contributed by a row*
(`ph/seams/tui_status.py`), and the daemon's own lifetime is neither. Putting it there
would make a per-root seam answer a per-process question, and the first deployment that
mounted no `tui-status` registrant would lose it.

**The sidebar.** One line in the session facts block (`widgets/status.py:244`), beneath
`cwd`:

```
id      01J9…
sandbox workspace-write
cwd     ~/Projects/pH
daemon  held · task          # or: exits in 9:41 · service · exits on detach
```

Rendered from `TuiState.lifetime`; the countdown re-renders on the spinner's clock when
one is running and on ordinary draws otherwise, so a waiting daemon does not start a
timer in an idle terminal. The word is the *reason*, not the state — `held · task`
answers "why is this still here", which is the question a person actually has.

**And in the doctor.** `DaemonServer.report()` (`server.py:1170`) gains a `lifetime`
section through the existing `sections` envelope, so `phern agents doctor` prints the
same four facts with no client change. The plan that introduced ephemerality promised
this row and it was never built; this is where it lands.

**Guarantee.** Why the daemon is still running is visible in the terminal that started
it and in the doctor, and both read it from the process rather than re-deriving it.

**Gates:** `test_the_sidebar_names_the_reason_a_daemon_is_held` ·
`test_a_lifetime_notice_reaches_a_client_watching_another_session` (sabotage: route it
through the `sessionId` filter and it is dropped) ·
`test_a_client_that_attaches_mid_window_draws_the_countdown` ·
`test_every_daemon_notice_has_a_reader` — `test_payloads`' existing shape, applied to
the new family · `test_the_doctor_prints_the_same_holds_the_sidebar_does`.

---

## P9-08 — docs and bookkeeping

* `docs/dev-notes/phase-9.md` — what was traded, in the series' shape. Three things
  belong in it: why the theme profile is not a pH profile row, why history is the
  mirror rather than a file, and why the keep-alive is the client's argv rather than a
  daemon-side setting.
* `plans/Implementation_Plan.md` §4 — the Phase 9 table above, and a "done when" row in
  §6.
* `DESIGN.md` — the front-end section gains the theme notation and the lifetime line;
  it is written against the source, so it is edited last.
* No seam page: nothing here adds a seam. Said explicitly, because four of these five
  rows touch the TUI and the reflex is to reach for `ctx.settings`.

**Non-guarantees (rule 6 — stated where they would be assumed, and into
`NON_GUARANTEES` where the daemon reports them):**

* A theme file is read **at start**. Editing one while pH is running changes nothing
  until the next launch; there is no watcher, and `ThemeCatalog`'s whole point is that
  the directory is scanned once.
* The `dark`/`light` pair in the theme profile is carried and **read by nothing** in
  this phase. No verb toggles on terminal appearance.
* Prompt history is **this session's**. Resuming a session gets that session's prompts;
  a fork gets its own, because a reference-forked child holds only its own events —
  the same fact `sessions.py:110` already works around for titles.
* The startup picker lists what `sessions/browse` can see: sessions whose header
  recorded this exact `cwd` string. A repo reached through a different symlink lists
  separately, and `/sessions` is the way to find it.
* An ephemeral daemon that exits **fires nothing while it is down**. A schedule keeps
  it resident, so the ordinary case holds — but `kill -9`, a logout reap, or a reboot
  ends the process and the appointment until a UI opens again. Timely scheduling on a
  machine that reboots still wants a systemd or launchd unit owning `phern daemon`.
* The keep-alive is a **floor on how long the process stays, not a ceiling**: a task that
  outruns it keeps the daemon up, which is the point.

---

## Reuse (do not rewrite)

* `ChoicePicker` and `Choice.matches` — history search, the startup picker and
  `/theme` are all one filter semantics.
* `RevealSeq` → `on_reveal_seq` → `TranscriptView.scroll_to_seq` — the history picker's
  "where was this" is the existing join, not a second one.
* `session_choices`' segment contraction and fork indentation — the startup picker adds
  a row and shapes nothing.
* `parse_theme`'s closed role set and its refusal of unknown roles — the palette is an
  input to it, not a replacement for it.
* `ph.cordis.loader.safe_yaml_load` — the one YAML entry point; themes do not get a
  second.
* `passivatable`'s clauses — the exit predicate names the ones it wants and derives
  nothing in parallel.
* `spawn_command` — the keep-alive is one more option beside `--ephemeral`, in the module
  that already spells that command line.
* `TuiKeybindings.as_map()` / `set_keymap` — three new bindings, no new mechanism, and
  no key compared as a literal anywhere.

## Verification

`./test.sh` — lint, format, types, test — plus, per increment, the named gates above;
each sabotage is applied and the named test confirmed to fail before the row is called
done. `uv run mypy` stays clean: the wire additions are models and
verbs, which is where `Verb[P, R]` already makes a mismatched pair unexpressible.

Two things are worth measuring rather than assuming:

* **The startup picker's cost.** `sessions/browse` on a store with many families is a
  header peek per file. It is on the path to the first prompt now, so it is timed on a
  store of a few hundred sessions before this is called done; if it is not comfortably
  under a frame, the limit comes down and the picker says it is showing the newest.
* **The history fold.** `state.items` is walked per open of the picker, not per
  keystroke; a session with thousands of rows should still open it instantly. If not,
  the fold is cached on `seq` the way the sidebar caches its catalogs
  (`widgets/status.py:260`).
