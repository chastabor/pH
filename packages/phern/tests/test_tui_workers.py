"""H7 — no handler on Textual's message pump may quietly await the harness.

`PHTuiApp`'s `action_*` and `on_*` methods run *on the pump*. Anything awaited
inside one is time the UI is frozen for: no keystrokes, no repaint, not even the
spinner. `/compact` spends a whole model call in `run_command`, and awaiting it
inline froze the terminal for the one command whose job is to take a while.

The fix for that call site was a worker. The rule it implies is not enforced by
the fix, and `app.py` itself names three entry points to the same verbs —
"Reached by key, by `/command`, and by `run_action`" — of which only one was
wrapped. Today's slow verbs happen to be keyless, so the rest is latent rather
than broken; latent is exactly what a gate is for.

**A table rather than an analysis.** Deciding "does this await the harness"
from the syntax tree would mean following `front.*` through `FrontSession` into
two transports, and would answer wrongly the first time somebody introduced an
indirection. So each handler is classified here, once, by a person — and a new
one fails this file until it is classified too. That is
`test_registration_ownership`'s BOUND table, for the same reason: a rule
everybody has to remember is a rule that gets forgotten.

**Per verb, not at the funnel.** `run_action` is the one place every entry point
goes through, and wrapping *it* was considered and rejected: it would change
every action's timing at once, and an action that returns as soon as it is
scheduled races every pilot test that presses a key and then asserts. Doing it
in the four verbs that actually make a wire call touches only those four, and
nothing presses a key for them.
"""

from __future__ import annotations

import ast
import pathlib

import ph_app.tui.app
from ph_app.tui.app import VERB_GROUP
from ph_app.tui.commands import TUI_VERBS

APP = pathlib.Path(ph_app.tui.app.__file__)
"""Asked of the module, not spelled from the checkout layout — the spelling
`test_tui_frames` already uses for this same file, and the one that still
resolves when the suite runs against an installed distribution."""

TREE = ast.parse(APP.read_text(encoding="utf-8"), filename=str(APP))
"""Parsed once. Three assertions walk it, and the module is 55 KB."""

UI_LOCAL: dict[str, str] = {
    "on_mount": "builds the screen and starts the pump's own tasks",
    "on_unmount": "tears down what `on_mount` started",
    "on_prompt_input_canceled": "clears local input state",
    "on_prompt_input_submitted": "routes; every slow branch is handed to a worker",
    "action_view": "toggles panel state and redraws from rows it already holds",
}
"""Every handler that may stay `async` on the pump: each awaits only this
process's own state.

The table is exhaustive by assertion below, so a new `async def action_*` or
`on_*` fails until somebody decides which side it is on."""

SCHEDULES_WORK = {f"action_{verb.action}" for verb in TUI_VERBS if verb.work is not None}
"""Verbs that need the harness, derived from where a verb is already declared.

Hand-listed here at first, which copied the verb→action mapping into a test file
and asked a new slow verb to be remembered in two places. `TuiVerb.work` is the
declaration; this is the gate reading it. `UI_LOCAL` stays a hand table
because `on_mount` and its kind have no declaration site anywhere."""


_SCHEDULING = {"work", "verb_work"}
"""Decorators that turn an `async def` into a sync callable returning a `Worker`.

Both spellings, because `verb_work` is `work` with this app's group and
exclusivity already chosen — a handler carrying either never holds the pump."""


def _decorators(node: ast.AsyncFunctionDef) -> set[str]:
    return {
        getattr(one.func if isinstance(one, ast.Call) else one, "id", "")
        for one in node.decorator_list
    }


def _pump_handlers() -> set[str]:
    """Async handlers that run *on* the pump — so `@work` ones are not among them.

    A decorated verb is a sync callable returning a `Worker`, which
    `_dispatch_action` does not await, so it never holds the pump however long
    its body runs.
    """
    return {
        node.name
        for node in ast.walk(TREE)
        if isinstance(node, ast.AsyncFunctionDef)
        and (node.name.startswith("action_") or node.name.startswith("on_"))
        and not (_decorators(node) & _SCHEDULING)
    }


def test_no_pump_handler_awaits_the_harness() -> None:
    """The whole rule, in one assertion each way.

    Sabotage: add an `async def action_anything` to `PHTuiApp` and the first
    assertion names it; make one of the four `async def` again and the second
    does.
    """
    found = _pump_handlers()

    assert found - set(UI_LOCAL) == set(), (
        "these run async on the message pump — classify them in UI_LOCAL, or make "
        f"them sync and hand the work to a worker: {sorted(found - set(UI_LOCAL))}"
    )
    assert set(UI_LOCAL) - found == set(), (
        f"these no longer exist and the table is stale: {sorted(set(UI_LOCAL) - found)}"
    )
    assert set(SCHEDULES_WORK) & found == set(), (
        "a verb that awaits the harness went back to awaiting it on the pump: "
        f"{sorted(set(SCHEDULES_WORK) & found)}"
    )


def test_every_harness_verb_hands_off_a_worker() -> None:
    """And they schedule rather than merely being absent from the pump table.

    A verb that dropped its wire call entirely would satisfy the test above;
    this is what says the work still happens.

    The *policy* needs no assertion any more: `verb_work` reads `TuiVerb.work`
    for both the group and the exclusivity, so there is no second copy to
    disagree with the row. It used to be spelled at the decorator, and this file
    hand-listed the mapping to check the two matched.

    Sabotage: remove `@verb_work` from any verb declaring `work`.
    """
    decorated = {
        node.name
        for node in ast.walk(TREE)
        if isinstance(node, ast.AsyncFunctionDef)
        for one in node.decorator_list
        if isinstance(one, ast.Call) and getattr(one.func, "id", None) == "verb_work"
    }

    assert decorated == SCHEDULES_WORK, (
        "every verb declaring `work` carries `@verb_work`, and only those: "
        f"{sorted(decorated ^ SCHEDULES_WORK)}"
    )
    assert VERB_GROUP == "verb", "the pilot helper filters workers by this prefix"


def test_only_the_side_effecting_verb_declines_to_be_replaced() -> None:
    """`work` is a decision per verb, and the wrong value loses a file.

    A second press of a *picker* means "I meant this one", so cancelling the
    first round trip is right. A second `/attach` is a second file, and
    cancelling the first to honour it drops an attachment the person asked for —
    silently, because the worker simply stops.

    Read off the rows rather than a literal, so the four names are not spelled a
    third time: `TUI_VERBS` has them, `SCHEDULES_WORK` derives from it, and this
    asks the same table which of them queues.
    """
    queues = {verb.name for verb in TUI_VERBS if verb.work == "queue"}

    assert queues == {"attach"}, "a verb with an effect was made replaceable"
