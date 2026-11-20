"""H7: no handler on the message pump of Textual may quietly await the harness.

The `action_*` and `on_*` methods of `PHTuiApp` run *on the pump*. Whatever one
of them awaits is time for which the UI is frozen: no keystrokes, no repaint,
not even the spinner. `/compact` spends a whole model call in `run_line`. When
that call was awaited inline, the terminal froze for the one command whose job
is to take a while.

The fix for that call site was a worker. The fix does not enforce the rule that
it implies. `app.py` itself names three entry points to the same verbs: "It is
reached by key, by `/command`, and by `run_action`". Only one of the three was
wrapped. The slow verbs of that time had no key, so the rest was latent and not
broken. A gate exists for exactly what is latent.

**A table and not an analysis.** To decide "does this await the harness" from
the syntax tree would mean following `front.*` through `FrontSession` into the
transport behind it. It would also answer wrongly the first time that somebody
introduced an indirection. So a person classifies each handler here, one time,
and a new handler fails this file until it is classified too. The BOUND table
of `test_registration_ownership` exists for the same reason: a rule that
everybody must remember is a rule that is forgotten.

**For each verb, not at the funnel.** `run_action` is the one place that every
entry point goes through. Wrapping *it* was considered and rejected. It would
change the timing of every action at the same time, and an action that returns
as soon as it is scheduled races every pilot test that presses a key and then
asserts. The change is made in the four verbs that make a wire call, and it
touches only those four. When this was decided, nothing pressed a key for them.
Two of the four now have a key (`session_picker` and `permission_picker`), and
`test_tui_pilot.py` presses both.
"""

from __future__ import annotations

import ast
import pathlib

import ph_app.tui.app
from ph_app.tui.app import VERB_GROUP
from ph_app.tui.commands import TUI_VERBS

APP = pathlib.Path(ph_app.tui.app.__file__)
"""The path is asked of the module, and not spelled from the layout of the
checkout. `test_tui_frames` already uses this spelling for the same file, and
it still resolves when the suite runs against an installed distribution."""

TREE = ast.parse(APP.read_text(encoding="utf-8"), filename=str(APP))
"""Parsed one time. Two tests walk it, and the module is about 67 KB."""

UI_LOCAL: dict[str, str] = {
    "on_mount": "builds the screen and starts the pump's own tasks",
    "on_unmount": "tears down what `on_mount` started",
    "on_prompt_input_canceled": "clears local input state",
    "on_prompt_input_submitted": "routes; every slow branch is handed to a worker",
    "action_view": "toggles panel state and redraws from rows it already holds",
    "action_quit": "exits with this app's own code; async only to match Textual's base",
}
"""Every handler that may stay `async` on the pump. Each one awaits only the
state of this process.

The assertion below makes the table exhaustive. A new `async def action_*` or
`on_*` therefore fails until somebody decides which side it is on."""

SCHEDULES_WORK = {f"action_{verb.action}" for verb in TUI_VERBS if verb.work is not None}
"""The verbs that need the harness, derived from where a verb is already declared.

The set was listed by hand here at first. That copied the mapping from verb to
action into a test file, and a new slow verb had to be remembered in two
places. `TuiVerb.work` is the declaration, and this gate reads it. `UI_LOCAL`
stays a hand-written table, because `on_mount` and the handlers like it are
declared nowhere."""


_SCHEDULING = {"work", "verb_work"}
"""Decorators that turn an `async def` into a sync callable that returns a `Worker`.

Both spellings are here, because `verb_work` is `work` with the group and the
exclusivity of this app already chosen. A handler that carries either one never
holds the pump."""


def _decorators(node: ast.AsyncFunctionDef) -> set[str]:
    return {
        getattr(one.func if isinstance(one, ast.Call) else one, "id", "")
        for one in node.decorator_list
    }


def _pump_handlers() -> set[str]:
    """Async handlers that run *on* the pump. The `@work` handlers are not among them.

    A decorated verb is a sync callable that returns a `Worker`.
    `_dispatch_action` does not await it, so it never holds the pump, however
    long its body runs.
    """
    return {
        node.name
        for node in ast.walk(TREE)
        if isinstance(node, ast.AsyncFunctionDef)
        and (node.name.startswith("action_") or node.name.startswith("on_"))
        and not (_decorators(node) & _SCHEDULING)
    }


def test_no_pump_handler_awaits_the_harness() -> None:
    """The whole rule, in three assertions.

    1. Every async handler on the pump is classified in `UI_LOCAL`.
    2. `UI_LOCAL` names no handler that is gone.
    3. No verb that needs the harness is on the pump. The first assertion
       already fails for such a verb, unless somebody also lists the verb in
       `UI_LOCAL` to make the first assertion pass. The third assertion is for
       that case.

    Sabotage: add an `async def action_anything` to `PHTuiApp`, and the first
    assertion names it. Remove `@verb_work` from one of the four verbs, and the
    first assertion names that verb. Then list the verb in `UI_LOCAL` as well,
    and the third assertion names it.
    """
    found = _pump_handlers()

    assert found - set(UI_LOCAL) == set(), (
        "these run async on the message pump — classify them in UI_LOCAL, or make "
        f"them sync and hand the work to a worker: {sorted(found - set(UI_LOCAL))}"
    )
    assert set(UI_LOCAL) - found == set(), (
        f"these no longer exist and the table is stale: {sorted(set(UI_LOCAL) - found)}"
    )
    assert SCHEDULES_WORK & found == set(), (
        "a verb that awaits the harness went back to awaiting it on the pump: "
        f"{sorted(SCHEDULES_WORK & found)}"
    )


def test_every_harness_verb_hands_off_a_worker() -> None:
    """The verbs schedule work, and are not only absent from the pump table.

    A verb that dropped its wire call completely would satisfy the test above.
    This test says that the work still happens.

    The *policy* needs no assertion any more. `verb_work` reads `TuiVerb.work`
    for both the group and the exclusivity, so there is no second copy that
    could disagree with the row. The policy used to be written at the decorator,
    and this file listed the mapping by hand to check that the two matched.

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
    """`work` is a decision for each verb, and the wrong value loses a file.

    A second press of a *picker* means "I meant this one", so it is right to
    cancel the first round trip. A second `/attach` is a second file. To cancel
    the first in order to honor the second drops an attachment that the user
    asked for. It is dropped silently, because the worker simply stops.

    The names are read from the rows and not from a literal, so the four names
    are not spelled a third time. `TUI_VERBS` has them, `SCHEDULES_WORK` derives
    from it, and this test asks the same table which of them queues.
    """
    queues = {verb.name for verb in TUI_VERBS if verb.work == "queue"}

    assert queues == {"attach"}, "a verb with an effect was made replaceable"
