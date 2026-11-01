# Tighter chrome: a status rule over a borderless prompt, and a left gutter

*Five increments on the terminal front end. Decorative rows and columns go; the
status line becomes the prompt's top border; the scrollbar moves into the one
column the content never uses. Planned 2026-10-31 against 0.10.0.*

## Context

Everything below is verified against the source as of commit `fb8b3c9`, so each
increment is a change to a named thing.

**Layout.** `PHTuiApp.compose` (`packages/phern/src/ph_app/tui/app.py:341`) builds
`#body` (Horizontal) → optional left `Sidebar` → `#main` (Vertical) → `TranscriptView
#transcript` over `#chrome` (Vertical) → `PromptInput#prompt` then `StatusBar#status`
→ optional right `Sidebar`. The status line is the last row of `#main`, under the
prompt; the sidebar is outside `#main`, so nothing in this plan touches it.

**The prompt** (`tui/widgets/prompt.py:88`, `PromptInput.DEFAULT_CSS`). The
`TextArea` carries `border: round $ph-border` and `:focus { border: round
$ph-accent }` — two rows and two columns of chrome, and the focus cue. Textual's
`TextArea` default `padding: 0 1` puts the text at column 2 inside the border, one
column right of the transcript's text. `#completions` is a `ListView` with its own
`border: round $ph-border`, `max-height: 6`, and Textual's default vertical scrollbar:
two columns wide, on the right. `#hint` (“tab to accept”) sits at column 0.

**The transcript** (`tui/widgets/transcript.py`). `TranscriptView` is a `VerticalScroll`
with `padding: 1 0 0 0`. Every row kind carries `margin: 0 1 1 1`, so plain text is at
column 1; a tool card, a thinking row and a compaction marker add `border-left: outer
…; padding-left: 1`, so their highlight bar is at column 1 and their text at column 3.
The view's scrollbar is Textual's default: right side, two columns.

**The status line** (`tui/widgets/status.py`, `StatusBar`). A `Vertical` of height 1
holding one `Static#status-line`. `show()` builds the parts `glyph`, `model`, `turn N`,
`N queued`, `context N%`, then every `line`-slot reading, each part after the first
spelled `[$ph-muted]·[/] value`, and joins them with **two** spaces — which is the
doubled gap the screenshot shows (`●  flash  · read-only accepted`). `show_unattached`
draws the same row before a session exists. `tick()`/`glyph` feed the spinner and the
terminal title; `app._advance` redraws the line each frame while a turn runs.

**Readings** (`packages/ph-core/src/ph/seams/tui_status.py`). A row contributes a
`StatusField(id, read, order, slot)`; the registry stamps `id` and `slot` onto each
`StatusReading{text, level}`. `ReadingSlot = Literal["line", "session"]`;
`status.in_slot` renders any slot it does not know on the line, which is what keeps a
new slot additive. Contributors today, by `(order, id)`:

| id | row | order | slot | text |
|---|---|---|---|---|
| `reasoning` | `token_meter` | 5 | line | `high` |
| `limits` | `ph_stabilize.limits` | 10 | line | budget |
| `posture` | `permission_presets` | 10 | line | `read-only accepted` |
| `sandbox-mode` | `sandbox` | 11 | session | (sidebar) |
| `sandbox` | `commands.sandbox` | 12 | line | denials |
| `cache` | `token_meter` | 14 | line | `cache 1.8k hit (98%)`, `cache 2.0k stored`, or both joined by ` · ` |

So today's full line reads `●  flash  · turn 3  · context 12%  · high  · read-only
accepted  · cache 28.8k hit (98%)`. The effort is already on the line; it is not
*beside the model*, and the cache figure is not beside the gauge it qualifies.

**Scrollbar side in Textual 8.2.8** (pinned `>=8.2.5,<9`). The right side is not a
style; it is arithmetic in three members of `Widget`, and the compositor consumes
exactly those three plus `Widget.gutter`:

* `scrollbar_gutter` (`widget.py:2179`) → `Spacing(0, size_v, size_h, 0)`;
* `_get_scrollable_region` (`:3913`) → `region.split_vertical(-size_v)`, forced on when
  `scrollbar-gutter: stable`;
* `_arrange_scrollbars` (`:3951`) → the bar in the right strip;
* `_compositor.py:584-593` shrinks by `styles.gutter` then calls
  `_get_scrollable_region`; `:708` calls `_arrange_scrollbars`; `:922` translates mouse
  coordinates by `widget.gutter.top_left`, so a gutter on the left is honored by
  hit-testing with no further change.

`scrollbar-gutter: stable` and `scrollbar-size-vertical` are ordinary CSS.

**Tests that pin this.** `packages/phern/tests/test_tui_snapshot.py` records ten
frames (`__snapshots__/test_tui_snapshot/*.raw`), including `test_the_full_app_shell`
at 90×24 through a real daemon in a thread. `test_tui_pilot.py::_status_line` reads
`#status-line` and every assertion on it is a substring (`"no session"`, `"context
86%"`, `"fake-1"`). `packages/ph-core/tests/test_loop_policies.py:426,446` pin the
cache wording exactly. `test_daemon_projections.py:80` and `test_tui_remote.py:449`
compare readings across the wire as whole models.

## Target

At 100 columns, with a tool card scrolled into view and the completion list open:

```
  ·
  This session is waiting for LLAMA_API_KEY, which is not set here — supply it to continue.
 ▐✓ read  packages/phern/README.md
 ▐  232 lines
  pH
  Reading the README first.
─────────────────────────────── ● flash/high · turn 3 · context 12%/cache 28.8k(98%) · read-only accepted
  /
 ▐/history Search the prompts you have sent.
  /permissions Change what pH may do without asking.
  /login Provide a provider credential for this process.
  tab to accept
```

(The leading column shown as a space above is column 0 of `#main`: the scrollbar
track in the transcript and in the completion list, a blank gutter beside the
prompt and the hint. Column 1 is content everywhere. The `▐` glyphs are a
one-column scrollbar thumb; the card's own highlight bar sits at column 1, where it
is today relative to the text.)

Rules, in the order the ask stated them:

1. **No wrapping border on the prompt.** Its top edge is the status line.
2. **The status line is a rule.** Right-aligned text; the rest of the row filled with
   `─` in `$ph-border`; the row keeps its `$ph-panel` background.
3. **Order and spelling:** `glyph model/effort · turn N · [N queued ·] context
   N%/cache … · <remaining line readings in order>`. Single spaces around every `·`.
4. **A single left gutter column** on the prompt, so its text is at column 1 with the
   transcript's text.
5. **Scrollbars on the left, one column wide**, in the transcript and in the completion
   list, which loses its border.
6. **Colors unchanged** except the status bar, which draws in the headings' role
   (`$ph-accent`). No theme role is added, removed or re-mapped.

## Decisions taken for the plan — confirm or overturn

* **The glyph leads the cluster.** The mockup `------------- * model/effort * turn 3 …`
  has one more mark than separators; the first is read as the spinner/idle glyph
  (`⠋`/`●`), which also keeps the busy signal on the row. Dropping it instead is one
  deleted part.
* **“Single left border space” is a blank column, not a line.** Padding, not
  `border-left`. The accent-on-focus border goes away with the border; the cursor is
  the focus cue, and the prompt is focused whenever no modal is. If a cue is wanted
  later, column 0 of the prompt can show `▏` in `$ph-accent` under `:focus-within` —
  one CSS rule, nothing structural.
* **“Slides over the card bars” means column 0.** The scrollbar takes the column
  immediately left of the card highlight bars, and the gutter is reserved
  (`scrollbar-gutter: stable`) so content never shifts when a bar appears. A
  scrollbar *overlaid on* column 0 content would need a custom bar widget on its own
  layer and a view that pretends it has none; not proposed.
* **The cache wording is tightened in ph-core**, since the mockup drops ` hit` and the
  space before `(`: `cache 28.8k(98%)`; a write-only request `cache 2.0k stored`; both
  `cache 28.8k(98%)+2.0k stored`. The inner ` · ` has to go regardless, because `·` is
  now the category separator.
* **The slash joins are declared by the contributing row, drawn by the footer.** Two
  new `ReadingSlot` values, `model` and `context`, say “beside the model name” and
  “beside the context gauge”; the front end joins those with `/`. The alternative —
  the footer matching ids `reasoning` and `cache` — is exactly what `in_slot`'s
  docstring was written to remove, and a renamed id would silently move the reading.
* **Out of scope:** the trajectory and diff screens, the modals and the sidebar keep
  their scrollbars and borders. Only `#main`'s chrome changes.

## T1 — Status line content: single spaces, order, slash joins

**ph-core, `seams/tui_status.py`.** `ReadingSlot = Literal["line", "session", "model",
"context"]`, with the docstring saying what each means for a front end: `model` and
`context` qualify a figure the footer already draws, so a front end that draws that
figure attaches them to it; one that does not renders them on the line. Update the
`ReadingSlot`/`StatusField.slot` docstrings, which currently argue for exactly two
values.

**ph-core, `seams/token_meter.py:423`.** `("reasoning", meter.reasoning_reading, 5,
"model")` and `("cache", meter.cache_reading, 14, "context")`, passing `slot=` into the
`StatusField`. `cache_reading`: the hit part becomes `f"{thousands(read)}({share}%)"`
with no space; the stored part is unchanged; parts join with `+`. Update
`test_loop_policies.py:426,446` to the new strings, and add one assertion that the two
fields register with their slots (the registry is what stamps them, so a test through
`TuiStatusRegistry.readings` is the honest one).

**phern, `tui/widgets/status.py`.** `_SLOTS` gains the two names. `StatusBar.show`
builds *segments*, each a list of `(markup, value)` pieces joined by `/`, then joins the
segments with `" · "` (the separator carries its own single spaces; the two-space
`"  ".join` goes):

1. `[$ph-accent]$glyph[/]` — stays a prefix, not a segment: `● ` then the first segment.
2. model: `[$ph-muted]$model[/]` + every `model`-slot reading, each keeping its own
   level style (`$ph-warning` when `warning`, else `$ph-muted`).
3. `turn $turn` when set; `$queued queued` when set.
4. context: the gauge when `pressure` is not `None` (its amber rule unchanged), + every
   `context`-slot reading. When the gauge is absent and readings exist, the readings
   alone form the segment; absent both, no segment.
5. every `line`-slot reading, in registry order, one segment each.

Keep every value a `$variable` substitution — a reading's text is a row's string, and
the module's rule is that no row string reaches the markup parser. `show_unattached`
is untouched here (T2 moves both onto the new widget).

**Verify:** `uv run pytest packages/ph-core/tests/test_loop_policies.py
packages/phern/tests/test_tui_pilot.py packages/phern/tests/test_daemon_projections.py
packages/phern/tests/test_tui_remote.py -q`. The wire tests pass unchanged: they compare
the models the seam produced with the ones the client parsed, and both sides are this
build.

**What breaks, by design:** a remote front end built before this change validates
`StatusReading` through pydantic and rejects `slot="model"`. `PROTOCOL_VERSION`
(`ph_app/protocol.py:84`) is 10; whether a widened literal counts as a protocol change
is that file's rule to state — read its docstring before deciding, and bump if it says
any wire-model change does.

## T2 — The status line becomes the prompt's top rule (done)

**`app.py` `compose`.** Inside `#chrome`, `StatusBar()` is yielded *before*
`PromptInput(...)`. Nothing else in the app reads the order: `_status`, `_prompt` and
`_view` are found by query.

**`status.py`.** `StatusBar` is a `Horizontal` holding a Textual `Rule#status-rule`
(its own `width: 1fr`, one column of right margin) and the unchanged
`Static#status-line` (`width: auto`). The layout gives the text its own width and the
rule what is left, so nothing measures a string against the terminal, and
`#status-line` still renders exactly the line `show()` built — which is what the
pilot's exact-line test reads. This replaced the planned `render()` override, which
would have put the rule inside the text widget.

The bar has one column of right padding, so the text stops a cell short of the side
panel's border instead of touching it. The padding cell stays blank when a narrow
terminal crops the text, because the row clips its children at its content edge.

Too narrow for the line, the rule keeps one cell (a fraction never sizes below one)
and the text runs past the row, which clips its right end: the posture goes first and
the glyph and model stay. No `text-wrap`/`text-overflow` rule is needed, because an
auto-width `Static` is never narrower than its content and so never wraps.
`show_unattached()` lands in the same widget, so "attaching…" and "no session" also
sit at the right.

**Color (changed after T3, at the user's request).** Text and rule draw in
`$ph-accent`, set once on the bar and inherited: the role the side panel's section
titles use, and the one Markdown headings one to three resolve to, since the theme
maps it onto Textual's `primary`. A role, so the bar follows the theme. Only a warning
reading and the gauge past the compaction threshold name a color of their own. The
unattached lines lost their doubled spaces at the same time (`● attaching…`,
`● no session · why`). `test_the_status_line_draws_in_the_headings_color` compares
the drawn colors against a side panel heading.

**Tests.** `test_the_status_line_is_the_prompts_top_edge` (the row ends where the
prompt begins, the text ends where the row does, the rule runs from the row's start
to one column short of the text, and the text ends one column short of the row;
sabotaged by swapping the compose order) and
`test_a_narrow_terminal_crops_the_status_line_rather_than_wrapping_it` (one row at
60×24, the visible slice keeps `● fake-1` and loses `accepted`).
`test_the_full_app_shell` was regenerated here rather than in T5, so the suite is
green at this increment.

## T3 — Borderless prompt and completion list, left gutter (done)

**`prompt.py`.** The box is `PromptArea(..., compact=True)`: Textual's own borderless
`TextArea`, applied with `!important`. A plain `border: none` here held only while
the box was blurred, because the library's `TextArea:focus { border: tall … }`
outranks a type selector, and the prompt is focused whenever no modal is up.

```
PromptInput { height: auto; }
PromptInput > TextArea { height: auto; max-height: 8; background: $ph-surface; }
PromptInput > #completions {
    height: auto; max-height: 6; display: none; background: $ph-panel; padding-left: 1;
}
PromptInput > #hint { display: none; color: $ph-muted; padding-left: 1; }
PromptInput.-completing > #completions, PromptInput.-completing > #hint { display: block; }
```

* The box keeps Textual's default `padding: 0 1`: the left pad is the gutter column,
  so text lands at column 1 with the transcript's, and the right pad keeps it off the
  side panel's border. The idle-prompt test compares the box's text column against a
  real transcript row, so a change to either side's padding fails there.
* The hint's text is set once in `compose`; the bindings are fixed for the widget's
  life, and the hint is hidden whenever there is nothing to accept.
* The `:focus` rule is gone with the border.
* `max-height: 8` keeps the box's content height where it was; the old 10 included two
  border rows. `PromptInput`'s own cap was dropped: its children's caps already bound
  it, and 16 was less than their sum.
* The list keeps `max-height: 6`, which now shows six items where the border left four.
  Completion is transient, so the space it already took shows more choices.
* The list's `padding-left: 1` holds column 0 until T4 puts the scrollbar there; the
  hint shows only while the list does, for `/` commands and `@` paths alike.

**Tests.** `test_the_idle_prompt_is_one_row_with_its_text_in_the_transcripts_column`
(focused, one row, text at column 1, no hint row; sabotaged by dropping `compact=True`)
and `test_the_completion_list_and_its_hint_line_up_with_the_prompt` (both start at the
prompt's text column, the hint leaves with the list; sabotaged by dropping the list's
padding). The full-shell snapshot was regenerated: idle chrome is two rows, down from
five.

**A test premise moved.** The screens tests that jump around a scrolled transcript
asked for a 12-row terminal, which left seven rows of transcript under five rows of
chrome. With two rows of chrome the view grew to ten and one jump stopped short. The
turn helpers in `test_tui_screens.py` now resize the terminal so the *transcript* is
`VIEW_ROWS` (seven) tall, whatever the chrome costs, and assert that it is.

## T4 — The scrollbar on the left

**New module `tui/widgets/gutter.py`** — a bare mixin, deliberately not a `DOMNode`
subclass, for the reason `selection._SourceSelection` records: Textual builds a
widget's default CSS from a single `__bases__` chain and a non-`DOMNode` mixin is
stepped over, while a `Widget`-derived one would truncate the chain and drop the real
base's CSS.

```python
class LeftScrollbar:
    """Put the vertical scrollbar in the left gutter. See the module docstring."""

    @property
    def scrollbar_gutter(self) -> Spacing:
        return Spacing(0, 0, self.scrollbar_size_horizontal, self.scrollbar_size_vertical)

    def _get_scrollable_region(self, region: Region) -> Region: ...
    def _arrange_scrollbars(self, region: Region) -> Iterable[tuple[Widget, Region]]: ...
```

Each override mirrors the Textual member it replaces line for line with the vertical
strip taken from the left: `region.split_vertical(size_v)` keeping the *second* piece
for content and the first for the bar; the corner gap, when both bars show, at the
bottom-left. The `scrollbar-gutter: stable` branch of `_get_scrollable_region` is
reproduced, since that is what reserves the column while nothing scrolls. The module
docstring names the three members and why they and not CSS decide the side, so the
next reader does not look for a `scrollbar-side` property that does not exist.

**Apply it:**

* `class TranscriptView(LeftScrollbar, VerticalScroll)`, with
  `TranscriptView { … scrollbar-size-vertical: 1; scrollbar-gutter: stable; }`.
* `class CompletionList(LeftScrollbar, ListView)` in `prompt.py`, yielded as
  `#completions`; `query_one("#completions", ListView)` still matches. Its
  `padding-left: 1` from T3 goes: the reserved gutter is column 0 now.
* Row margins `0 1 1 1` → `0 1 1 0` on `TranscriptRow`, `StreamingMessage` and
  `ToolCardWidget` (the compaction and thinking variants inherit). With the reserved
  gutter at column 0, plain text is at column 1 and a card bar at column 1 — the
  picture today, shifted left by one.

**Guard the private API.** `tests/test_tui_gutter.py` (phern):

1. Mount a `LeftScrollbar` + `VerticalScroll` with more rows than fit at a fixed size;
   assert `view.vertical_scrollbar.region.x == view.region.x`, the first child's
   `region.x == view.region.x + 1`, and `view.vertical_scrollbar.region.width == 1`.
2. Assert that `Widget` still defines all three overridden names
   (`hasattr` on each, plus that `_arrange_scrollbars`' signature is unchanged). A
   Textual bump that renames one then fails here with the name, rather than drawing
   the bar on the right in production with every test green.
3. A pilot step: scroll the transcript with the wheel over the content and drag the
   bar with the mouse at column 0; both move `scroll_y`. Hit-testing is the one path
   the mixin does not override, and `_compositor.py:922` is the line that makes it
   work — this test is what notices if that changes.

## T5 — Snapshots, docs, bookkeeping

* Regenerate the frames T3 and T4 move (the full shell was already regenerated in T2): `uv run pytest packages/phern/tests/test_tui_snapshot.py
  --snapshot-update`, **outside the sandbox** (the full-shell test attaches over a unix
  socket), then read `snapshot_report.html` before committing the `.raw` files.
  Check in the full-shell frame: rule at the right over the prompt, prompt text at
  column 1, no rounded corners anywhere in `#main`, sidebar unchanged.
* `docs/seams/tui_status.md`: the slot table gains `model` and `context` with the
  one-sentence meaning each. `docs/seams/token_meter.md`: the cache wording.
  `packages/phern/README.md:232` mentions footer readings only in passing; no change.
* `DESIGN.md` has no section on the footer; nothing to update there.
* Theme files: untouched. If the one-column thumb reads too faint against
  `$ph-background`, that is a `scrollbar-color` rule on `TranscriptView`, not a theme
  role — and it is the user's call, not this plan's.

## Order

T1 → T2 → T3 + T4 together → T5. T1 is pure logic and lands with tests and no
snapshot churn; T2 moves the row and is the one visual change that is reviewable on its
own; T3 and T4 are one picture (the gutter column is shared by the prompt, the list and
the transcript) and are regenerated once in T5.

## Reuse (do not rewrite)

* `in_slot` — extend `_SLOTS`; the fallback-to-line rule is the mechanism T1 relies on.
* `Content.from_markup(..., **values)` — every reading stays a substituted variable.
* `_SourceSelection`'s mixin rule — `LeftScrollbar` follows it exactly.
* The pilot's `_status_line` helper and `make_tui_app`/`daemon_in_thread` fixtures.

## Verification

1. `uv run pytest packages/ph-core/tests/test_loop_policies.py packages/phern/tests
   -q` outside the sandbox; then `./test.sh` for the four gates.
2. Manual at 90×24 and at 50×24: status rule, crop, completion popup over `/`,
   wheel and drag on the left bar, a resumed session with a tool card (bar at column 1,
   thumb at column 0 beside it).
3. `test_every_theme_renders[ph-light]` and `[high-contrast]` frames eyeballed for the
   rule's contrast on each background.
