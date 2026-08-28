# `ctx.tui_screens` — a registration seam for the front end (P4-17)

**Status:** built. Written first as a design, after P3-25 shipped the trajectory
as a separate `App` which worked and left two of §5.5's shape-(b) features
unbuilt; then implemented against it. The design is kept below as written, and
[what the build changed](#what-the-build-changed-and-why) is recorded at the end
rather than edited into it — a design silently rewritten to match its
implementation records nothing.

## What this is for

Three asks, one mechanism:

1. **Plugin screens.** A row should be able to contribute a screen to the TUI
   the way it already contributes a tool, a command or a prompt section.
2. **The trajectory, reachable from the chat.** P3-25's view opens only via
   `ph --mode trajectory`; there is no route from a running session.
3. **Cross-navigation by `sourceSeq`.** The join is stored on both sides and
   read by neither.

(1) is the general form of (2), and (3) falls out once both views can coexist.

## The model: dsh's slot service, ported narrowly

`packages/client/ui-slots` is a named-slot registry that ~35 `ui-*` packages
contribute into — `conversation.view`, `conversation.composer`,
`conversation.chat.node`, `sidebar.workspaces`, `settings.section`,
`tool.call.toolview`. The registration call is:

```ts
ctx.slots.register({ name, id, order, label, inject })
```

and dsh's own comment states the property that makes it pH-shaped: *"the
registration rides the slot service's effect wrapper, so plugin unload removes
the tab."*

**Direction is the load-bearing detail.** `ui-trajectory` registers *into*
`conversation.view` at `order: 10`. The conversation is the host. pH keeps that:
`PHTuiApp` remains the shell.

### What pH ports, and what it does not

| dsh | pH (P4-17) | why |
|---|---|---|
| A slot *hierarchy* (`settings.section`, `conversation.chat.node`, …) | **One slot: screens.** | pH's TUI has one extension point worth opening today. A hierarchy with one member is a hierarchy nobody can check. |
| `inject` returning a per-registrant props bag | `build(session) -> Screen` | Textual screens take constructor args; there is no props/store layer to mirror. |
| Locale-bound `label: () => t(...)` | `label: str` | pH has no locale service. A thunk would be a hook for a feature that does not exist. |
| Registration returns void, disposal via effect wrapper | Returns a `Disposer`, via `claim_key` | Same guarantee, existing helper — `claim_key` already checks identity before removing, so a disposer cannot tear down a successor. |

## The contract

```python
@dataclass(frozen=True, slots=True)
class ScreenDefinition:
    """One screen a plugin contributes to the TUI."""

    id: str
    """Addressable name. Becomes `/‹id›` and the binding id `set_keymap` remaps."""
    label: str
    """What the palette and the picker call it."""
    build: Callable[[Any], Screen[None]]
    """`build(session) -> Screen`. Given the *front end's* session, so a screen
    is a projection of a log rather than a holder of harness state."""
    order: int = 100
    """Sort order in the palette, as dsh orders slot entries."""
    key: str | None = None
    """A `TuiKeybindings` field, when the screen wants a default key."""
```

```python
class TuiScreenRegistry:
    """The service published as `ctx.tui.screens`."""  # built as ctx.tui_screens

    def register(self, screen: ScreenDefinition, *, scope: Context | None = None) -> Disposer: ...
    def list(self) -> list[ScreenDefinition]: ...  # order, then id
    def get(self, screen_id: str) -> ScreenDefinition | None: ...
```

Mounted by a `tui-screens` row in `ph-base` so the seam exists whether or not a
front end is running — the same rule every other seam follows, and what lets a
headless run register a screen nothing draws without special-casing.

### What registering buys, automatically

One `ScreenDefinition` yields, with no further wiring:

- a `TuiVerb` generated from it, so `/‹id›` appears in `ctx.commands` beside the
  built-ins (the existing `register_tui_commands` loop already does this for
  `TUI_VERBS`; it gains a second source);
- a remappable binding when `key` is set — `id` doubles as the binding id, which
  is what `App.set_keymap` rebinds, so a plugin's key obeys `tui.json` like
  every other;
- a command-palette entry;
- `escape` popping back to the chat, because it is a pushed screen and not a
  second app.

Disposing the registration removes all of them. That is the dsh property, and
`claim_key` already provides it.

## `TrajectoryScreen` as the first registrant

`TrajectoryApp` currently owns the panel, the bindings and the fork action.
Split it:

- **`TrajectoryScreen(Screen[None])`** — the panel, the header, the fork action.
  Takes a session (live) *or* a record list (stored). Owns nothing else.
- **`TrajectoryApp(App)`** — keeps `--mode trajectory`: loads records with
  nothing mounted and pushes one `TrajectoryScreen`. Its whole body becomes the
  loading and the push.
- **`ph-app`'s TUI row** registers `ScreenDefinition(id="trajectory", label="Trajectory",
  order=10, build=…, key="trajectory_view")`.

Both compose the same screen, so **(a) is a strict subset of (b) again** — the
claim §5.5 makes and P3-25 broke.

The fork action's session-store dependency stays where it is: the screen asks
for `ctx.sessions` when it has one and reports honestly when it does not,
exactly as today.

## Cross-navigation

Both directions, over the join that already exists:

- **Record → transcript.** `TrajectoryRecord.source_seq` is a session event seq;
  `ChatItem.seq` is the same number. Pop the screen and scroll the transcript to
  the row whose `seq` matches, falling back to the nearest preceding row when a
  record has no visible counterpart (a `request/header` has none by design).
- **Transcript → record.** From a focused row, push the screen with its `seq`
  pre-selected.

Neither direction needs new state. What it needs is for `TranscriptView` to
expose "scroll to seq", which it does not today — that is the one real widget
addition in this row.

## Why not the inversion (trajectory as the shell)

Recorded because it was asked and the answer is not obvious:

1. **It makes "nothing mounted" conditional.** The property is what lets the
   view read a crashed run, a child's log or a replay fixture. As the shell it
   either mounts the harness up front — losing it — or defers the mount, which
   relocates the trust prompt and profile load into a screen transition.
2. **It taxes the common path.** Most launches are a conversation. A shell you
   traverse to reach one is friction on the majority case to serve the minority.
3. **It inverts the dependency.** The auditor's view would own the app and so
   would have to know the chat's mounting and trust flow — the coupling P3-25's
   cleanup pass removed.

dsh does not do it either: its trajectory is a tab in `conversation.view`.

## What this design deliberately leaves out

- **A slot hierarchy.** One extension point now; a second registrant is what
  should motivate a second slot.
- **Screens for non-session data.** `build` takes a session because both known
  registrants are session projections. A screen that wants something else is the
  argument for widening the signature, and should come with one.
- **Live-follow in the trajectory.** The screen reads the session's events when
  pushed. Following the stream on the frame tick is shape (a)'s feature and is
  additive afterwards.

## Gates

- a test row registers a screen and gets a verb, a key and a palette entry;
- disposing the registration removes all three;
- the trajectory opens over a live chat and `escape` returns to it with the
  transcript's scroll and focus intact;
- a record jumps to its transcript row, and a transcript row to its record;
- `ph --mode trajectory` still mounts nothing — asserted as today, by reading a
  file with no context in the test.

---

## What the build changed, and why

Nine departures. Six are the design meeting a constraint it had not checked;
three are things it had simply not thought through.

### The seam had to live in `ph-core`, so `build` cannot be typed

A *plugin* is what registers a screen, and a plugin depends on `ph-core` alone —
`ph-rlm` does not depend on `ph-app` and must not. So `ScreenDefinition` is in
`ph.seams.tui_screens`, where `test_layering.py` forbids importing Textual. The
signature is `Callable[[Any], Any]`, and the front end states the real type where
it draws one. `ph.tools.registry` already owns `ToolCallView`/`CardKind` on the
same terms: core owns the presentation *vocabulary*, `ph-app` owns the pixels.

### `ctx.tui.screens` → `ctx.tui_screens`

Cordis services are flat keys, and multi-word ones are the idiom already —
`session_persistence`, `permission_presets`, `user_questions`. A `ctx.tui`
namespace object holding exactly one member is the slot hierarchy this design
rejects, one level up and with the same argument against it.

### `key` is a default key, not a `TuiKeybindings` field name

The design said both "a `TuiKeybindings` field" and "a plugin's key obeys
`tui.json` like every other", and those are incompatible: a plugin cannot add a
field to that dataclass, so a field name would have made plugin screens the one
class of key a person could not rebind. `key` is now the literal default
(`"f2"` for the trajectory), the binding id is the screen's id, and
`TuiKeybindings.extra` carries binding ids this build has no field for straight
from `tui.json` into `set_keymap`. A test drives it through the file.

### The seam grew a presenter, because the gate was otherwise untrue

"Unloading the row removes all three" does not follow from `claim_key` alone.
The command and the key are made by the *front end*, and the app's own disposer
list would have kept them alive after the row that contributed the screen was
gone. So `present_with(presenter)` attaches a front end, and every presentation
it makes is registered on the **registration's** scope. `Context.add_disposer`
hands back an idempotent release, which is exactly what lets one teardown belong
to two lifetimes — the row unloading and the front end detaching — with whichever
runs second doing nothing.

### `scope=ctx` is the registrant's job, and is documented as such

A service cannot discover its caller's activation scope, so a row that wants its
screen to unwind with it has to say so. The default is the service's own context,
which is right for something the harness contributes and wrong for a row's; the
docstring says which, in those words.

### The registry row is in `ph-base`; the trajectory row is not

`tui-screens` is in `base.yaml` as designed. The **registrant** could not be:
`base.yaml` may not name a `ph-app` plugin, or a `ph-core`-only install could not
compose the base profile. `tui-screen-trajectory` is inserted by `ph-app`'s
`profiles/tui.yaml` — the interactive posture, which is where a registrant earns
its place anyway. The TUI test fixture grew a `profile=` argument for it.

### `TrajectoryApp` overrides `get_default_screen` instead of pushing

The design said "loads records with nothing mounted and pushes one
`TrajectoryScreen`". `get_default_screen` is the same thing without a second
layout pass, and it keeps `escape` meaning "back to the chat" only where there
*is* a chat: the screen checks the stack depth rather than assuming it was
pushed.

### Transcript → record is the topmost visible row, not a focused one

"From a focused row" assumed the transcript has per-row focus. It does not — it
is widgets in a `VerticalScroll` — and inventing one is a larger change than this
row. `TranscriptView.seq_in_view()` answers the honest version of the question:
the first row still on screen. `RevealSeq` (screen → shell) and the `Revealing`
protocol (shell → screen) are the two halves, both optional, so a screen with no
position to take is opened as it is. A third, `RevealHost`, is how a screen asks
whether anyone can answer before offering the action — structurally, so
`--mode trajectory` declines and any future shell that can reveal is served
without a screen ever counting the screen stack to guess where it is.

### The fold was quadratic, and this row put it on a keypress

`build_trajectory` asked `is_fork_boundary(log, seq)` per record, and that
rescans the prefix: 467 ms to fold an 8 000-event session. It only ever ran at
`--mode trajectory` startup, so it had never been felt — and this row made it the
body of a key handler on the message pump. `ph.session.fork_boundaries(log)`
answers the same rule for the whole log in one pass, which keeps A6 in `store.py`
where `open_turn_at`'s docstring argues it belongs; a test pins the two
statements to each other over a log with open turns, closed turns and events
between them. 7 000 events: 30 ms, and linear.

### Two fixes outside the design, which the gates depended on

The second is the anchor. `TranscriptView.sync` re-armed it on **every** dirty
frame:
`if not self.is_anchored: self.anchor()`. Textual releases that anchor when the
reader scrolls away, so the next event snapped them straight back — which would
also have made a jump to a record last exactly until the next chunk arrived. It
re-arms from the bottom now, and on the first frame that has anything to scroll.
Scrolling up in a live conversation stays put for the first time as well.
