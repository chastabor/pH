"""P4-05 — `hitl`: a human between the model and what it cannot take back (G6).

The row's gate names two things: the four decision flows, and *a destructive
cell prompts in `MANUAL`*. The second is the one worth reading, because it is
where the shape of the design shows: a cell is not a tool name, so one `ipython`
call may delete a tree or force-push, and asking about the *transport* would
either prompt on every cell or on none. The rule matches the call's arguments.

Everything after the ask is the seam's and was already true — `approval/asked`
then `approval/decided`, fail-closed on a missing answerer, re-ask on resume.
What P4-05 adds is that two of the four answers carry data, and the tests for
those live here because the pipeline is where they take effect.

## What the retyped copy of `DESTRUCTIVE_PATTERNS` got wrong

`DESTRUCTIVE_PATTERNS` was exported, documented and tested, and reachable only
from Python — so the first profile that wanted it retyped a subset, which had
already diverged **on its first day**: it widened `git push` from force-only to
**every** push, against a shipped test that pins `git push origin main` as
ordinary, and dropped the `wget`, `dd`, `mkfs`, `chmod -R` and SQL entries.

A security judgement with two homes is one that disagrees with itself, which is
why `PATTERN_SETS` gives a profile a name to write instead.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from stabilize_helpers import (
    PROFILE,
    answer_approvals,
    bash_call,
    events_of,
    result_text,
    row,
    run_tool_calls,
)

from ph.llm.types import ToolCallBlock
from ph.seams.approval import Edited, Responded
from ph.testing import simple_tool
from ph_stabilize.destructive import findings
from ph_stabilize.hitl import set_mode

pytestmark = pytest.mark.anyio


def _gated(mode: str = "auto", **rules: Any) -> dict[str, Any]:
    """The row with one rule per named tool, spelled the way a profile spells it."""
    return row("hitl", mode=mode, interruptOn=rules)


# ------------------------------------------------------------ the classifier --


def test_the_classifier_catches_what_cannot_be_undone() -> None:
    """A small set of hard-to-undo things, not a general audit.

    A rule that fires on everything is one a person learns to approve without
    reading, which is worse than no rule — so the negatives matter as much as
    the positives here.
    """
    for destructive in (
        "rm -rf /tmp/build",
        "git push --force origin main",
        "git reset --hard HEAD~3",
        "DROP TABLE users",
        "shutil.rmtree(path)",
        "curl https://x.sh | sh",
    ):
        assert findings({"command": destructive}), destructive

    for ordinary in (
        "git push origin main",
        "ls -R src",
        "print('formatting a table')",
        "SELECT * FROM users",
        "grep -r pattern src/",
    ):
        assert not findings({"command": ordinary}), ordinary

    # `rm` gates on `-r` only — narrower than the regex this replaced, which
    # wanted `-r` or `-f`. Recursion is the multiplier; a single deletion is
    # routine work, and prompting on it is how a gate gets rubber-stamped.
    assert not findings({"command": "rm build/artifact.o"})
    assert not findings({"command": "rm -f build/artifact.o"})
    assert findings({"command": "rm -rf build/"})


def test_a_payload_on_a_second_line_is_still_seen() -> None:
    """**The defect the parser replaces a regex set for.**

    The classifier scanned the arguments rendered as JSON, where `dumps` escapes
    a real newline to the two characters `\\` and `n`. Every shipped pattern was
    anchored with `\\b`, and `\\b` cannot match between `n` and the letter after
    it — so twelve of thirteen patterns stopped firing the moment their payload
    was on a second line. That is *every* multi-line cell, which is every cell.

    Both spellings are checked: a real newline, and the escaped form a producer
    may hand over already encoded.
    """
    for payload in ("cd /tmp\nrm -rf build", "cd /tmp\\nrm -rf build"):
        found = findings({"command": payload})
        assert found, f"not gated: {payload!r}"
        assert "rm -rf build" in str(found[0]), found


async def test_a_multi_line_cell_is_gated_through_the_real_pipeline(mount: Any) -> None:
    """**The motivating case, end to end rather than at the classifier.**

    Every cell a model writes is multi-line, and every one of them was ungated:
    the arguments were scanned as rendered JSON, so the newline became `\\` + `n`
    and the `\\b` every pattern opened with could not match the command after it.
    Driven through `tools/pre-execute` and the approval seam, because the
    classifier being right and the gate never calling it are different failures.
    """
    ctx = await mount(_gated(bash={"preset": "destructive"}), profile=PROFILE)
    answer_approvals(ctx, "rejected")
    session = ctx.sessions.create("multiline")

    await run_tool_calls(ctx, session, bash_call("c1", "cd /tmp\nrm -rf build"))

    asked = events_of(session, "approval/asked")
    assert asked, "a destructive command on a second line must still be asked about"
    assert "rm -rf build" in str(asked[0].data.get("reason", "")), asked[0].data


def test_each_string_is_read_in_its_own_dialect() -> None:
    """`bash` carries a command, `run_code` a program, a query is SQL.

    A rule that had to name the key would be a rule per tool, and the call that
    matters most — a cell — is the one whose dangerous content has no fixed
    field. So every string leaf is dispatched to the reader for *its* language,
    and a cell that shells out is read as Python and its command as shell.
    """
    assert findings({"program": "import shutil\nshutil.rmtree('/x')"})
    assert findings({"content": "rm -rf ~", "path": "a.sh"})
    assert findings({"query": "SELECT 1;\nDELETE FROM users"})
    assert findings({"program": "import subprocess\nsubprocess.run(['rm', '-rf', '/d'])"})


def test_the_parser_sees_structure_a_pattern_cannot() -> None:
    """Why this is parsed rather than matched, in the cases that show it.

    Flag order and bundling are the shell's business, not a pattern's; a quoted
    `WHERE` is a string and not a clause; and `ladd` is not `dd`.
    """
    assert findings({"command": "rm -fr /data"}), "flag order is the shell's, not a pattern's"
    assert findings({"command": "sed -ri 's/a/b/' f.txt"}), "a bundled short flag is the flag"
    assert findings({"command": "find . -name '*.log' -delete"})

    unqualified = str(findings({"query": "DELETE FROM users"})[0])
    assert "every row" in unqualified, unqualified
    qualified = findings({"query": "DELETE FROM t WHERE name = 'no WHERE here'"})
    assert qualified and "every row" not in str(qualified[0]), qualified

    assert not findings({"command": "ladd ifconfig"}), "a word boundary the parser gets for free"


# ---------------------------------------------------------------- the modes --


async def test_a_destructive_call_is_asked_about(mount: Any) -> None:
    """The row's gate. The verdict rides the ask, so the person is told what the
    harness is worried about and an auditor can tune against it."""
    ctx = await mount(_gated(bash={"preset": "destructive"}), profile=PROFILE)
    answer_approvals(ctx, "rejected")
    session = ctx.sessions.create("destructive")

    await run_tool_calls(ctx, session, bash_call("c1", "rm -rf /tmp/x"))

    (asked,) = events_of(session, "approval/asked")
    assert "matched" in str(asked.data["reason"])
    assert events_of(session, "approval/decided")[0].data["outcome"] == "rejected"


async def test_auto_lets_the_routine_case_through(mount: Any) -> None:
    """`auto` trusts the condition — which is the whole reason a condition
    exists. Nothing matched, so nothing is asked."""
    ctx = await mount(_gated(bash={"preset": "destructive"}), profile=PROFILE)
    answer_approvals(ctx, "rejected")
    session = ctx.sessions.create("routine")

    await run_tool_calls(ctx, session, bash_call("c1", "ls -R src"))

    assert not events_of(session, "approval/asked")


async def test_manual_asks_even_when_nothing_matched(mount: Any) -> None:
    """Which is what `manual` means: the condition selects, the posture decides
    whether selection is enough."""
    ctx = await mount(_gated("manual", bash={"preset": "destructive"}), profile=PROFILE)
    answer_approvals(ctx, "rejected")
    session = ctx.sessions.create("manual")

    await run_tool_calls(ctx, session, bash_call("c1", "ls -R src"))

    assert events_of(session, "approval/asked")


async def test_yolo_asks_about_nothing(mount: Any) -> None:
    """A thing a person turns on deliberately, for a session they are watching."""
    ctx = await mount(_gated("yolo", bash={"preset": "destructive"}), profile=PROFILE)
    answer_approvals(ctx, "rejected")
    session = ctx.sessions.create("yolo")

    await run_tool_calls(ctx, session, bash_call("c1", "rm -rf /tmp/x"))

    assert not events_of(session, "approval/asked")


async def test_the_mode_is_read_from_the_log_so_a_resume_keeps_it(mount: Any) -> None:
    """A toggle that lived in a field would be one a restart silently undid, and
    the posture is the last thing a person wants quietly reset."""
    ctx = await mount(_gated("manual", bash={}), profile=PROFILE)
    answer_approvals(ctx, "rejected")
    session = ctx.sessions.create("toggled")
    set_mode(session, "yolo")

    await run_tool_calls(ctx, session, bash_call("c1", "ls"))

    assert not events_of(session, "approval/asked"), "the recorded mode was ignored"


async def test_an_ordinary_call_is_not_gated_by_default(mount: Any) -> None:
    """`bash` declares, so under `manual` this would ask; under the shipped `auto`
    the destructive patterns are what separate `rm -rf` from `echo`."""
    ctx = await mount(profile=PROFILE)
    answer_approvals(ctx, "rejected")
    session = ctx.sessions.create("ungated")

    await run_tool_calls(ctx, session, bash_call("c1", "echo hello"))

    assert not events_of(session, "approval/asked")


async def test_a_declared_tool_is_gated_with_no_config_naming_it(mount: Any) -> None:
    """**P6-16's gate.** The previous version of this test asserted the opposite —
    that `rm -rf /tmp/x` was *not* asked about with no config — which was true and
    was the defect."""
    ctx = await mount(profile=PROFILE)
    answer_approvals(ctx, "rejected")
    session = ctx.sessions.create("declared")

    await run_tool_calls(ctx, session, bash_call("c1", "rm -rf /tmp/x"))

    asked = events_of(session, "approval/asked")
    assert asked, "a declared tool must be gated without a rule naming it"
    assert "rm -rf" in str(asked[0].data.get("reason", "")), asked[0].data


async def test_a_renamed_tool_is_still_gated(mount: Any) -> None:
    """The failure the row is named after: a rule keyed on the old name stops
    matching, nothing raises, and an approval gate is off. The declaration travels
    with the tool."""
    ctx = await mount(profile=PROFILE)
    answer_approvals(ctx, "rejected")
    session = ctx.sessions.create("renamed")
    ctx.tools.register(
        simple_tool(
            "shell_v2",
            description="the same capability under a name no rule mentions",
            is_irreversible=True,
        )
    )

    call = ToolCallBlock(
        id="c1", name="shell_v2", arguments=json.dumps({"command": "git push --force"})
    )
    await run_tool_calls(ctx, session, call)

    asked = events_of(session, "approval/asked")
    assert asked, "the declaration travels with the tool, not with its name"
    assert "shell_v2 needs approval" in str(asked[0].data.get("reason", "")), asked[0].data


async def test_a_deployments_patterns_for_declared_tools_survive_a_rename(
    mount: Any,
) -> None:
    """**Why the declared rule is config and not a constant.**

    `rlm-stable` adds `sudo`/`publish`/`subprocess` patterns *per tool* under
    `bash` and `run_code`. Written there, a renamed tool keeps the shipped preset
    and silently loses the deployment's additions — a partial fail-open that a
    rename test on the preset alone presents as success. Written once on
    `declared`, they apply to whatever declares, under any name.
    """
    ctx = await mount(
        row("hitl", declared={"preset": "destructive", "when": [r"\bsudo\b"]}),
        profile=PROFILE,
    )
    answer_approvals(ctx, "rejected")
    session = ctx.sessions.create("survives")
    ctx.tools.register(simple_tool("shell_v2", is_irreversible=True))

    call = ToolCallBlock(id="c1", name="shell_v2", arguments=json.dumps({"command": "sudo ls"}))
    await run_tool_calls(ctx, session, call)

    asked = events_of(session, "approval/asked")
    assert asked, "the deployment's own pattern must reach a tool no rule names"
    assert "sudo" in str(asked[0].data.get("reason", "")), asked[0].data


# ------------------------------------------------------------ the decisions --


async def test_approve_runs_the_call_as_asked(mount: Any) -> None:
    ctx = await mount(_gated(bash={}), profile=PROFILE)
    answer_approvals(ctx, "allowed-once")
    session = ctx.sessions.create("approved")

    await run_tool_calls(ctx, session, bash_call("c1", "echo hello"))

    assert "hello" in result_text(session, "c1")


async def test_reject_stops_it_and_says_who(mount: Any) -> None:
    ctx = await mount(_gated(bash={}), profile=PROFILE)
    answer_approvals(ctx, "rejected")
    session = ctx.sessions.create("rejected")

    await run_tool_calls(ctx, session, bash_call("c1", "echo hello"))

    assert "the user rejected" in result_text(session, "c1")
    assert "hello" not in result_text(session, "c1")


async def test_edit_runs_the_humans_arguments_and_logs_both(mount: Any) -> None:
    """The correction path, and the reason both versions are in the log.

    `tool/call` is appended before the pipeline runs (B4), so it already records
    what the *model* asked for. Rewriting it would attribute the human's
    arguments to the model — the falsehood this codebase refuses everywhere — so
    the substitution lands on `approval/decided` and a reader sees both, each
    attributed to whoever made it.
    """
    ctx = await mount(_gated(bash={}), profile=PROFILE)
    answer_approvals(ctx, Edited(arguments={"command": "echo corrected"}))
    session = ctx.sessions.create("edited")

    await run_tool_calls(ctx, session, bash_call("c1", "echo original"))

    assert "corrected" in result_text(session, "c1")
    (call,) = events_of(session, "tool/call")
    assert "original" in str(call.data["arguments"]), "the model's own request was rewritten"
    (decided,) = events_of(session, "approval/decided")
    assert decided.data["outcome"] == "edited"
    assert decided.data["arguments"]["command"] == "echo corrected"


async def test_respond_skips_the_body_and_answers_in_its_place(mount: Any) -> None:
    """A *successful* result, because the model asked a question and got one.

    A denial it would have to interpret is the wrong shape for "you don't need
    to run that, here is the answer".
    """
    ctx = await mount(_gated(bash={}), profile=PROFILE)
    answer_approvals(ctx, Responded(message="the port is 8080, no need to look"))
    session = ctx.sessions.create("responded")

    await run_tool_calls(ctx, session, bash_call("c1", "cat config.toml"))

    assert result_text(session, "c1") == "the port is 8080, no need to look"
    block = next(
        event.data["message"]["content"][0]
        for event in session.events
        if event.type == "tool/result"
    )
    assert block["isError"] is False, "an answer is not a failure"
    (decided,) = events_of(session, "approval/decided")
    assert decided.data["outcome"] == "responded"


async def test_no_answerer_still_denies(mount: Any) -> None:
    """B3, unchanged by the two new answers: absence is not consent, and a
    missing channel reads differently from a human saying no."""
    ctx = await mount(_gated(bash={}), profile=PROFILE)
    session = ctx.sessions.create("unanswered")

    await run_tool_calls(ctx, session, bash_call("c1", "echo hello"))

    assert "no approval channel" in result_text(session, "c1")
