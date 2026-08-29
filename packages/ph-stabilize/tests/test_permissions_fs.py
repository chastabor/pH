"""P4-06 — `permissions-fs`: path rules over filesystem access (G7, E9).

The row's two gates are first-match-wins and recursive-delete-fails-closed, and
they are tested for opposite reasons. First-match is tested because *order* is
the whole configuration language: a narrow `allow` above a broad `deny` is how
an exception is written, and a rule set evaluated by specificity — or by
"strictest wins" — would silently mean something else. Recursive delete is
tested because the honest answer is a refusal: the rules describe paths that do
not exist yet, so a subtree cannot be checked one path at a time.

Everything reaches the disk through `ctx.fs`, never through a tool name, which
is what makes the same rules cover a Code Mode binding and an MCP server's
writer. The tests therefore drive `ctx.fs` directly wherever the tool layer
would only be re-testing the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from stabilize_helpers import (
    PROFILE,
    answer_approvals,
    events_of,
    result_block,
    result_text,
    row,
    run_tool_calls,
    scoped_agent,
)

from ph.llm.types import ToolCallBlock
from ph.seams.fs import FsDenied
from ph.testing import FAKE_OPTIONS, StubAgent, StubWorkspaceProvider, run_tool
from ph_stabilize.permissions_fs import (
    BOUNDED_REACH,
    UNBOUNDED_REACH,
    FsPermissions,
    Rule,
)

pytestmark = pytest.mark.anyio


def _policy(*rules: Rule, root: str = "/w") -> FsPermissions:
    """The decision half on its own, for the questions no tool can ask yet."""
    return FsPermissions(
        rules=rules, roots=lambda agent=None: getattr(agent, "root", None) or Path(root)
    )


DEFAULT_WRITE_SCOPE: dict[str, Any] = {
    "operations": ["write"],
    "paths": ["**"],
    "scope": "outside-workspace",
    "mode": "interrupt",
}
"""E6's rule, spelled here as the bundle spells it.

Restated rather than read from `bundle.yaml` because a row patch replaces a
row's config *wholesale* — so a test that adds a rule of its own drops the
bundle's, and composing the two by hand is what makes the ordering the tests are
about visible in the test.
"""


async def _scoped(mount: Any, tmp_path: Path, *rules: dict[str, Any]) -> Any:
    """The bundle's default write scope, with any explicit rules *above* it.

    Above, because first-match-wins is the whole precedence mechanism: an
    operator's rule decides before the default is reached, and a test that
    appended instead would be testing the opposite arrangement.
    """
    (tmp_path / "project").mkdir(exist_ok=True)
    return await _mounted(mount, tmp_path, *rules, DEFAULT_WRITE_SCOPE)


async def _mounted(mount: Any, tmp_path: Path, *rules: dict[str, Any]) -> Any:
    """The row with its ACL spelled the way a profile spells it, over a temp root."""
    return await mount(
        row("permissions-fs", rules=list(rules)), row("fs", root=str(tmp_path)), profile=PROFILE
    )


# --------------------------------------------------------------- first match --


def test_the_first_matching_rule_decides_and_nothing_after_it_is_read() -> None:
    """Order is the configuration language.

    A narrow `allow` above a broad `deny` is how every path ACL anyone has used
    expresses an exception. Evaluated by specificity, or by "strictest wins",
    the same two lines would mean the opposite — so the test pins the direction
    rather than the outcome of one pair.
    """
    policy = _policy(
        Rule(paths=("secrets/public.txt",), mode="allow"),
        Rule(paths=("secrets/**",), mode="deny"),
    )
    allowed = policy.decide("read", Path("/w/secrets/public.txt"))
    denied = policy.decide("read", Path("/w/secrets/private.txt"))
    assert allowed is not None and allowed.mode == "allow"
    assert denied is not None and denied.mode == "deny"

    # The same two rules the other way up: the broad deny now wins both, which
    # is the point — reordering is how an operator changes the answer.
    reversed_policy = _policy(
        Rule(paths=("secrets/**",), mode="deny"),
        Rule(paths=("secrets/public.txt",), mode="allow"),
    )
    shadowed = reversed_policy.decide("read", Path("/w/secrets/public.txt"))
    assert shadowed is not None and shadowed.mode == "deny"


def test_no_rule_matches_means_allow() -> None:
    """The default, and it has to be this one: a permission row whose empty
    config refused everything would be a row nobody could layer."""
    assert _policy(Rule(paths=("secrets/**",))).decide("read", Path("/w/src/main.py")) is None
    assert _policy().decide("write", Path("/w/anything")) is None


def test_a_rule_only_answers_for_the_operations_it_names() -> None:
    """`read` and `write` are separate questions, and a read-only tree is the
    case that motivates saying so."""
    policy = _policy(Rule(operations=("write",), paths=("vendor/**",), mode="deny"))
    assert policy.decide("read", Path("/w/vendor/lib.py")) is None
    denied = policy.decide("write", Path("/w/vendor/lib.py"))
    assert denied is not None and denied.mode == "deny"


def test_a_path_matches_by_either_spelling() -> None:
    """Relative to the workspace, or absolute.

    A rule written `.env` should catch the file in the workspace, and one
    written `/etc/**` should catch a file that has no relative spelling at all —
    an operator should not have to know which form the seam resolved to.
    """
    policy = _policy(Rule(paths=(".env",), mode="deny"), Rule(paths=("/etc/**",), mode="deny"))
    assert policy.decide("read", Path("/w/.env")) is not None
    assert policy.decide("read", Path("/etc/shadow")) is not None
    assert policy.decide("read", Path("/w/src/.env.example")) is None


# --------------------------------------------------------- recursive delete --


def test_a_recursive_delete_is_refused_when_a_rule_could_match_inside_it() -> None:
    """G7's gate, and the reason it is a refusal rather than a walk.

    The rules describe paths, including paths that do not exist yet; the tree is
    enumerated by the operating system at delete time. So "is anything in here
    protected?" cannot be answered by checking the directory's own path, and
    checking each child would still race the ones created after the check.
    """
    policy = _policy(Rule(operations=("write",), paths=("build/keep/**",), mode="deny"))
    assert policy.deletion_reason(Path("/w/build"), recursive=True) is not None
    # Not recursive: one path, one question, and this one is not protected.
    assert policy.deletion_reason(Path("/w/build"), recursive=False) is None
    # A sibling tree the rule cannot reach into stays deletable, or the fence
    # would refuse every delete anyone ever attempted.
    assert policy.deletion_reason(Path("/w/dist"), recursive=True) is None


def test_a_leading_wildcard_refuses_every_recursive_delete() -> None:
    """`**/*.env` could match anything, so it does: a pattern with no literal
    head tells you nothing about where it will hit, and the only safe reading of
    "I cannot tell" on a delete is no."""
    policy = _policy(Rule(operations=("write",), paths=("**/*.env",), mode="deny"))
    assert policy.deletion_reason(Path("/w/anywhere"), recursive=True) is not None


def test_containment_is_on_separators_not_on_string_prefixes() -> None:
    """`/w/xyz` is not inside `/w/x`. A prefix test would have refused deletes of
    every directory whose name merely started the same way."""
    policy = _policy(Rule(operations=("write",), paths=("/w/x/**",), mode="deny"))
    assert policy.deletion_reason(Path("/w/xyz"), recursive=True) is None
    assert policy.deletion_reason(Path("/w/x"), recursive=True) is not None


def test_an_allow_rule_never_causes_a_recursive_refusal() -> None:
    """The fail-closed check looks for rules that would *object*. An `allow`
    inside the tree is agreement, and treating it as a possible objection would
    make writing an exception the way to forbid a delete."""
    policy = _policy(Rule(operations=("write",), paths=("build/**",), mode="allow"))
    assert policy.deletion_reason(Path("/w/build"), recursive=True) is None


# ------------------------------------------------------------ through the fs --


async def test_a_denied_read_never_opens_the_file(mount: Any, tmp_path: Path) -> None:
    """Gated on `fs/read-intent`, so the refusal happens before the open — a
    check that ran after would be a report, which is the failure this seam
    exists to avoid."""
    (tmp_path / "secret.env").write_text("TOKEN=hunter2", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path, {"paths": ["*.env"], "mode": "deny"})

    with pytest.raises(FsDenied) as refused:
        await ctx.fs.read("secret.env")
    assert "denied by permissions-fs" in str(refused.value)


async def test_a_denied_write_and_edit_are_both_refused(mount: Any, tmp_path: Path) -> None:
    """One `write` rule covers create, overwrite and in-place edit.

    Splitting them would mean "you may not write here" left `edit` open, which
    is not what anyone means when they write that line.
    """
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.py").write_text("x = 1\n", encoding="utf-8")
    ctx = await _mounted(
        mount, tmp_path, {"operations": ["write"], "paths": ["vendor/**"], "mode": "deny"}
    )

    with pytest.raises(FsDenied):
        await ctx.fs.write("vendor/lib.py", "x = 2\n")
    with pytest.raises(FsDenied):
        await ctx.fs.edit("vendor/lib.py", "x = 1", "x = 2")
    # And nothing changed on disk, which is the only assertion that proves the
    # gate ran before the write rather than after it.
    assert (tmp_path / "vendor" / "lib.py").read_text(encoding="utf-8") == "x = 1\n"
    # The read the rule did not mention is still allowed.
    assert "x = 1" in (await ctx.fs.read("vendor/lib.py")).text


async def test_a_concealed_path_is_absent_from_glob_and_grep(mount: Any, tmp_path: Path) -> None:
    """Enumeration is filtered, not asked about — and filtered during the walk.

    `grep` reads every file it visits, so post-filtering its matches would
    return nothing while having read the whole protected file. The assertion
    that matters is that the *contents* never appear.
    """
    (tmp_path / "app.py").write_text("TOKEN = 1\n", encoding="utf-8")
    (tmp_path / "secret.env").write_text("TOKEN=hunter2\n", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path, {"paths": ["*.env"], "mode": "deny"})

    paths = await ctx.fs.glob("**/*")
    assert any(path.endswith("app.py") for path in paths)
    assert not any(path.endswith("secret.env") for path in paths)

    matches = await ctx.fs.grep("TOKEN")
    assert [match.path for match in matches] == [str(tmp_path / "app.py")]
    assert not any("hunter2" in match.text for match in matches)


async def test_interrupt_asks_and_the_answer_decides(mount: Any, tmp_path: Path) -> None:
    """`interrupt` is an approval, through the same seam `hitl` uses.

    One approval channel, so a deployment configures its answerer once — and a
    refusal reads as a refusal rather than as a path that quietly did nothing.
    """
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path, {"paths": ["notes.md"], "mode": "interrupt"})
    agent = StubAgent(ctx, ctx.sessions.create("asked"))
    answers: list[str] = ["allowed-once"]
    answer_approvals(ctx, lambda: answers[0])

    assert "hello" in (await ctx.fs.read("notes.md", agent=agent)).text
    answers[0] = "rejected"
    with pytest.raises(FsDenied) as refused:
        await ctx.fs.read("notes.md", agent=agent)
    # The seam's own sentence, so a model can tell a human's "no" from a missing
    # channel — only one of the two is worth re-planning around.
    assert "the user rejected" in str(refused.value)
    assert events_of(agent.session, "approval/decided")


async def test_interrupt_without_an_answerer_denies(mount: Any, tmp_path: Path) -> None:
    """Fail closed, the same way every other path through `ctx.approval` does:
    a missing channel is a misconfigured deployment, not consent."""
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path, {"paths": ["notes.md"], "mode": "interrupt"})
    agent = StubAgent(ctx, ctx.sessions.create("unanswered"))

    with pytest.raises(FsDenied) as refused:
        await ctx.fs.read("notes.md", agent=agent)
    assert "no approval channel" in str(refused.value)


async def test_an_agentless_interrupt_denies_rather_than_prompting(
    mount: Any, tmp_path: Path
) -> None:
    """No agent means no session, and a prompt raised outside a session is worse
    than useless.

    `ApprovalService.request` derives the session from the agent, so an agentless
    ask would put a question in front of a human with **nothing written to the
    log** — and would walk straight past the `approval_policy: "never"`
    short-circuit a deployment set on purpose. Both existing consumers of the
    seam refuse for exactly this reason; the first draft of this row did not.
    """
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path, {"paths": ["notes.md"], "mode": "interrupt"})
    asked: list[Any] = []
    answer_approvals(ctx, lambda: asked.append("prompted") or "allowed-once")

    with pytest.raises(FsDenied):
        await ctx.fs.read("notes.md")
    assert asked == [], "a human was prompted with nowhere to record the answer"


async def test_a_rule_may_carry_its_own_words_for_the_prompt(mount: Any, tmp_path: Path) -> None:
    """`description`: the path is not always enough to decide by, and an operator
    who wrote the rule knows why it is there."""
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8")
    ctx = await _mounted(
        mount,
        tmp_path,
        {"paths": ["notes.md"], "mode": "interrupt", "description": "shared team notes"},
    )
    agent = StubAgent(ctx, ctx.sessions.create("described"))
    answer_approvals(ctx, lambda: "rejected")

    with pytest.raises(FsDenied):
        await ctx.fs.read("notes.md", agent=agent)
    (asked,) = events_of(agent.session, "approval/asked")
    assert asked.data["reason"] == "shared team notes"


async def test_nothing_is_restricted_by_default(mount: Any, tmp_path: Path) -> None:
    """Layering the bundle must not start refusing file access — `hitl`'s rule,
    for the same reason: a harness that refuses on first run is one whose rules
    nobody chose."""
    (tmp_path / "secret.env").write_text("TOKEN=1\n", encoding="utf-8")
    ctx = await mount(row("fs", root=str(tmp_path)), profile=PROFILE)

    assert "TOKEN" in (await ctx.fs.read("secret.env")).text
    assert await ctx.fs.glob("**/*.env")


async def test_a_tool_call_is_covered_without_the_row_naming_the_tool(
    mount: Any, tmp_path: Path
) -> None:
    """The reason the rules hang off the seam rather than off `tools/pre-execute`.

    `read` is never mentioned in this module. It is covered because it goes
    through `ctx.fs`, which is also what covers a Code Mode binding, a second
    editing tool, and an MCP server's file writer — where a name-keyed rule
    would have silently covered none of them, the failure `compaction`'s
    truncation pass and `offload` both argue against in their own docstrings.

    And it arrives as a **denial**, not a failure: `FsDenied` carries
    `failure_kind`, so a Code Mode program cannot read a policy veto as a tool
    that merely broke and try something else.
    """
    (tmp_path / "secret.env").write_text("TOKEN=hunter2\n", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path, {"paths": ["*.env"], "mode": "deny"})
    session = ctx.sessions.create("gated")

    await run_tool_calls(
        ctx,
        session,
        ToolCallBlock(id="c1", name="read", arguments=json.dumps({"path": "secret.env"})),
    )

    assert "denied by permissions-fs" in result_text(session, "c1")
    assert "hunter2" not in result_text(session, "c1")
    block = result_block(session, "c1")
    assert block is not None and block.is_error is True


async def test_an_empty_rule_set_attaches_nothing(mount: Any, tmp_path: Path) -> None:
    """The row mounted with no rules is the row doing nothing at all.

    Not "allowing everything through a predicate that says yes": `hide` is
    consulted once per file a walk visits, so a `grep` over a repository would
    pay a Python call per candidate to answer a question nobody asked. The
    service is still published, because "no rules" is an answer `ph doctor`
    wants to be able to give.
    """
    (tmp_path / "secret.env").write_text("TOKEN=1\n", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path)

    assert ctx.fs_permissions.rules == ()
    assert "TOKEN" in (await ctx.fs.read("secret.env")).text
    assert await ctx.fs.glob("**/*.env")


# ------------------------------------------------------------------- reach --


async def test_the_reach_message_toggles_with_a_sandbox(mount: Any, tmp_path: Path) -> None:
    """E9, and the reason `reach` reads the seam rather than a mount-time bool.

    These rules bound access *through* `ctx.fs`; a code cell calling `open()`
    directly never fires an intent. Saying so is only worth anything if the
    sentence changes when the situation does — a disclaimer that is always there
    is one people stop reading, and one computed at mount would be the wrong
    sentence for a profile that layers its backend afterwards.
    """
    ctx = await _mounted(mount, tmp_path, {"paths": ["*.env"], "mode": "deny"})
    assert ctx.fs_permissions.reach == UNBOUNDED_REACH
    assert "not covered" in ctx.fs_permissions.reach

    # P6-04's backend does not exist yet, so the stand-in is what the seam
    # actually reads: a provider in the slot. The claim under test is the
    # toggle, not what any particular backend enforces.
    ctx.sandbox.register_provider(object())
    assert ctx.fs_permissions.reach == BOUNDED_REACH
    assert "not covered" not in ctx.fs_permissions.reach


def test_a_policy_with_no_sandbox_seam_reports_unconfined() -> None:
    """Fail closed, and also simply true: nothing is holding the line for the
    paths these rules cannot see."""
    assert _policy().reach == UNBOUNDED_REACH
    assert not _policy().confined


async def test_an_anchored_rule_still_applies_inside_a_worktree(mount: Any, tmp_path: Path) -> None:
    """The hole D21 opened, and the reason the root is asked per agent.

    A rule written `secrets/**` is anchored to *the workspace*. With the root
    captured at mount, an agent at the `worktree` tier writes in a directory
    outside that prefix, so `_spellings` offered only the absolute path and the
    rule silently stopped applying — in the one place a rule is most needed,
    with nothing failing to say so.
    """
    ctx = await _mounted(mount, tmp_path, {"operations": ["write"], "paths": ["secrets/**"]})
    session = ctx.sessions.create("s1")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    ctx.workspace.register_provider(StubWorkspaceProvider(root=tmp_path / "trees"))
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id=agent.id, base=tmp_path, session=session
    )
    assert workspace.root != ctx.fs.root, "this test is only meaningful for a fresh root"

    result = await run_tool(
        ctx,
        "write",
        {"path": str(workspace.root / "secrets" / "key.pem"), "content": "x"},
        agent=agent,
    )

    assert result.is_error
    assert not (workspace.root / "secrets" / "key.pem").exists()


def test_two_agents_are_judged_against_their_own_roots(tmp_path: Path) -> None:
    """The same rule, two agents, two answers — which is the point.

    `secrets/**` names a different directory in each agent's tree, and a policy
    that resolved one root for everybody would answer for the wrong one.
    """

    class _Agent:
        def __init__(self, root: Path) -> None:
            self.root = root

    policy = _policy(Rule(operations=("write",), paths=("secrets/**",)), root="/w")
    one, two = _Agent(Path("/w/a")), _Agent(Path("/w/b"))

    assert policy.objection("write", Path("/w/a/secrets/k"), one) is not None
    assert policy.objection("write", Path("/w/a/secrets/k"), two) is None
    assert policy.objection("write", Path("/w/b/secrets/k"), two) is not None


# ----------------------------------------------- the default write scope --


async def test_writing_inside_the_workspace_never_asks(mount: Any, tmp_path: Path) -> None:
    """E6's first half, and the whole reason the row exists.

    Not one prompt, for any number of writes: the agent owns this checkout, so
    asking about its own tree is asking about nothing.
    """
    ctx = await _scoped(mount, tmp_path)
    agent, workspace = await scoped_agent(ctx, tmp_path)
    asked = answer_approvals(ctx, "rejected")

    for name in ("one.txt", "two.txt", "nested/three.txt"):
        await run_tool(
            ctx, "write", {"path": str(workspace.root / name), "content": "x"}, agent=agent
        )

    assert asked == []
    assert (workspace.root / "nested" / "three.txt").read_text(encoding="utf-8") == "x"


async def test_writing_to_scratch_never_asks(mount: Any, tmp_path: Path) -> None:
    """Scratch is outside the worktree *by design* (E5) and is the one place a
    read-only or ephemeral agent is told it may write — so a scope covering only
    `root` would prompt on exactly the writes the design invites."""
    ctx = await _scoped(mount, tmp_path)
    agent, workspace = await scoped_agent(ctx, tmp_path)
    asked = answer_approvals(ctx, "rejected")

    await run_tool(
        ctx, "write", {"path": str(workspace.scratch / "notes.md"), "content": "kept"}, agent=agent
    )

    assert asked == []
    assert (workspace.scratch / "notes.md").read_text(encoding="utf-8") == "kept"


async def test_one_write_outside_asks_once(mount: Any, tmp_path: Path) -> None:
    """E6's second half. The rare prompt is the meaningful one, and it carries
    the boundary being left rather than only the path."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    ctx = await _scoped(mount, tmp_path)
    agent, workspace = await scoped_agent(ctx, tmp_path)
    asked = answer_approvals(ctx, "allowed-once")

    await run_tool(ctx, "write", {"path": str(outside / "escape.txt"), "content": "x"}, agent=agent)

    assert len(asked) == 1
    assert str(outside / "escape.txt") in str(asked[0].reason)
    assert str(workspace.root) in str(asked[0].reason), "the prompt does not say what is being left"
    assert (outside / "escape.txt").exists()


async def test_a_refused_write_outside_does_not_happen(mount: Any, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    ctx = await _scoped(mount, tmp_path)
    agent, _workspace = await scoped_agent(ctx, tmp_path)
    answer_approvals(ctx, "rejected")

    result = await run_tool(
        ctx, "write", {"path": str(outside / "escape.txt"), "content": "x"}, agent=agent
    )

    assert result.is_error
    assert not (outside / "escape.txt").exists()


async def test_no_answerer_denies_rather_than_allowing(mount: Any, tmp_path: Path) -> None:
    """Fail-closed, the reading every asking row in this package shares: an
    approval nobody can answer is not an approval."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    ctx = await _scoped(mount, tmp_path)
    agent, _workspace = await scoped_agent(ctx, tmp_path)

    result = await run_tool(
        ctx, "write", {"path": str(outside / "escape.txt"), "content": "x"}, agent=agent
    )

    assert result.is_error
    assert not (outside / "escape.txt").exists()


async def test_an_agent_with_no_workspace_is_unaffected(mount: Any, tmp_path: Path) -> None:
    """No workspace, no scope to be outside of.

    A profile that layers this row without the lifecycle gets today's behaviour
    rather than a boundary drawn around a directory nobody chose.
    """
    ctx = await _scoped(mount, tmp_path)
    session = ctx.sessions.create("s1")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    answer_approvals(ctx, "rejected")
    target = tmp_path / "anywhere.txt"

    await run_tool(ctx, "write", {"path": str(target), "content": "x"}, agent=agent)

    assert target.read_text(encoding="utf-8") == "x"


async def test_an_explicit_deny_beats_the_default(mount: Any, tmp_path: Path) -> None:
    """The layering *is* the precedence.

    Waterfall listeners run outermost-first in registration order, so
    `permissions-fs` layered before this row refuses first and this row is never
    reached. That is what makes "default write scope" a default rather than an
    override of the operator's own rules.
    """
    ctx = await _scoped(mount, tmp_path, {"operations": ["write"], "paths": ["secrets/**"]})
    agent, workspace = await scoped_agent(ctx, tmp_path)
    answer_approvals(ctx, "allowed-once")

    result = await run_tool(
        ctx,
        "write",
        {"path": str(workspace.root / "secrets" / "key.pem"), "content": "x"},
        agent=agent,
    )

    assert result.is_error
    assert not (workspace.root / "secrets" / "key.pem").exists()
