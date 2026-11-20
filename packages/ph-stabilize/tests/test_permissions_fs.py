"""P4-06: `permissions-fs`, path rules over filesystem access (G7, E9).

The row has two gates: first-match-wins and recursive-delete-fails-closed. They
are tested for opposite reasons. First-match is tested because *order* is the
whole configuration language. An exception is written as a narrow `allow` above
a broad `deny`. A rule set evaluated by specificity, or by "strictest wins",
would silently mean something else. Recursive delete is tested because the
correct answer is a refusal. The rules describe paths that do not exist yet, so
a subtree cannot be checked one path at a time.

Everything reaches the disk through `ctx.fs`, never through a tool name. The
tests therefore drive `ctx.fs` directly wherever the tool layer would only test
the pipeline again.

**How much that placement gives today.** `tool-fs` is the *only* caller of
`FsService.write` and `FsService.edit` in the tree. `FsService.read` has two
more callers: the engines of `ph-code-graph` and `ph-text-index`. So most of the
coverage that this placement earns is in the future. A Code Mode binding is the
namespaced face of a governed tool and does route through here.
`memory-agents-md` reads with `Path.open`, and the harness service of `ph-rlm`
reads with `Path.read_text`. They are **not** covered. An MCP server is a remote
client and can never reach `ctx.fs` at all. An earlier version of this docstring
claimed that the rules covered its writer, and that was never true. The
placement is still right, because the next writer is covered there at no cost.
But the argument for it is "one gate, and not one gate per tool". The argument
is not a list of things that are already behind the gate.

**Why a walk does not ask `_refuses_under` the question of a delete.**
`_could_match_under` answers `True` for a pattern that starts with `**`, because
for a *delete* "could match anywhere" must round to refusal. For a *walk* the
same rounding refuses everywhere. `deny read **/.env` is the idiomatic
spelling, and with it every directory "could" hold a match. Measured on this
repository before the guard: **11 489 results down to 6.** The guard was first a
`require_head` flag on `_refuses_under`. The walk now asks `covers_whole`
instead (D11).

## Why `_prefix_of` is cached

`_spellings` runs per gated read *and per file that a walk visits*. P4-06
measured `Path.relative_to` at **26 µs there, which was 92% of the check**.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from stabilize_helpers import (
    PROFILE,
    answer_approvals,
    events_of,
    result_block,
    row,
    run_tool_calls,
    scoped_agent,
)

from ph.agent.types import AgentDriver
from ph.cordis import DEPLOYMENT
from ph.keys import AGENTS, FS, SANDBOX, SESSIONS, SPILL_STORE, WORKSPACE
from ph.llm.types import ToolCallBlock
from ph.seams.fs import FsDenied, matches_glob
from ph.testing import (
    FAKE_OPTIONS,
    MountProfile,
    StubAgent,
    StubSandboxProvider,
    StubWorkspaceProvider,
    noted,
    report_section,
    result_text,
    run_tool,
)
from ph_stabilize.permissions_fs import (
    BOUNDED_REACH,
    FS_PERMISSIONS,
    UNBOUNDED_REACH,
    FsPermissions,
    Rule,
)

pytestmark = pytest.mark.anyio


def _policy(*rules: Rule, root: str = "/w") -> FsPermissions:
    """The decision half alone, for the questions that no tool can ask yet."""

    def roots(agent: AgentDriver | None = None) -> Path:
        return getattr(agent, "root", None) or Path(root)

    return FsPermissions(rules=rules, roots=roots)


DEFAULT_WRITE_SCOPE: dict[str, Any] = {
    "operations": ["write"],
    "paths": ["**"],
    "scope": "outside-workspace",
    "mode": "interrupt",
}
"""The rule of E6, written here as the bundle writes it.

The rule is restated and not read from `bundle.yaml`, because a row patch
replaces the config of a row *wholesale*. A test that adds a rule of its own
therefore drops the rule of the bundle. Composing the two by hand makes the
ordering that the tests are about visible in the test.
"""


async def _scoped(
    mount: MountProfile,
    tmp_path: Path,
    *rules: dict[str, Any],
) -> Any:  # noqa: ANN401
    """The default write scope of the bundle, with any explicit rules *above* it.

    Above, because first-match-wins is the whole precedence mechanism. A rule of
    the operator decides before the default is reached. A test that appended
    the explicit rules would test the opposite arrangement.
    """
    (tmp_path / "project").mkdir(exist_ok=True)
    return await _mounted(mount, tmp_path, *rules, DEFAULT_WRITE_SCOPE)


async def _mounted(
    mount: MountProfile,
    tmp_path: Path,
    *rules: dict[str, Any],
) -> Any:  # noqa: ANN401
    """The row with its ACL written as a profile writes it, over a temporary root."""
    return await mount(
        row("permissions-fs", rules=list(rules)), row("fs", root=str(tmp_path)), profile=PROFILE
    )


# --------------------------------------------------------------- first match --


def test_the_first_matching_rule_decides_and_nothing_after_it_is_read() -> None:
    """Order is the configuration language.

    Every path ACL that anyone has used expresses an exception as a narrow
    `allow` above a broad `deny`. Evaluated by specificity, or by "strictest
    wins", the same two lines would mean the opposite. So the test checks the
    direction, and not the outcome of one pair.
    """
    policy = _policy(
        Rule(paths=("secrets/public.txt",), mode="allow"),
        Rule(paths=("secrets/**",), mode="deny"),
    )
    allowed = policy.decide("read", Path("/w/secrets/public.txt"))
    denied = policy.decide("read", Path("/w/secrets/private.txt"))
    assert allowed is not None and allowed.mode == "allow"
    assert denied is not None and denied.mode == "deny"

    # The same two rules in the opposite order. The broad deny now wins both.
    # That is the point: an operator changes the answer by reordering.
    reversed_policy = _policy(
        Rule(paths=("secrets/**",), mode="deny"),
        Rule(paths=("secrets/public.txt",), mode="allow"),
    )
    shadowed = reversed_policy.decide("read", Path("/w/secrets/public.txt"))
    assert shadowed is not None and shadowed.mode == "deny"


def test_no_rule_matches_means_allow() -> None:
    """The default, and it must be this one. A permission row whose empty config
    refused everything would be a row that nobody could layer."""
    assert _policy(Rule(paths=("secrets/**",))).decide("read", Path("/w/src/main.py")) is None
    assert _policy().decide("write", Path("/w/anything")) is None


def test_a_rule_only_answers_for_the_operations_it_names() -> None:
    """`read` and `write` are separate questions. A read-only tree is the case
    that makes this worth stating."""
    policy = _policy(Rule(operations=("write",), paths=("vendor/**",), mode="deny"))
    assert policy.decide("read", Path("/w/vendor/lib.py")) is None
    denied = policy.decide("write", Path("/w/vendor/lib.py"))
    assert denied is not None and denied.mode == "deny"


def test_a_path_matches_by_either_spelling() -> None:
    """Relative to the workspace, or absolute.

    A rule written `.env` must catch the file in the workspace. A rule written
    `/etc/**` must catch a file that has no relative spelling at all. An
    operator should not have to know which form the seam resolved to.
    """
    policy = _policy(Rule(paths=(".env",), mode="deny"), Rule(paths=("/etc/**",), mode="deny"))
    assert policy.decide("read", Path("/w/.env")) is not None
    assert policy.decide("read", Path("/etc/shadow")) is not None
    assert policy.decide("read", Path("/w/src/.env.example")) is None


def test_a_dot_dot_spelling_cannot_slip_past_a_deny() -> None:
    """A rule matches a path string, so the string must be the real one (D6).

    Nothing collapsed `..`, so `public/../secrets/key` bypassed
    `deny read secrets/**`. The relative spelling offered to the matcher still
    had the `..` in it, and `secrets/**` does not match `public/../secrets/key`.
    The file that is read is the same file in both cases. Only the name that was
    asked for differed, and that is the whole bypass.

    Both spellings are checked, because both are offered: the absolute spelling
    that a rule like `/etc/**` uses, and the workspace-relative spelling that a
    rule like `secrets/**` uses.
    """
    policy = _policy(Rule(operations=("read",), paths=("secrets/**",), mode="deny"))

    assert policy.decide("read", Path("/w/secrets/key")) is not None, "the plain spelling"
    assert policy.decide("read", Path("/w/public/../secrets/key")) is not None
    assert policy.decide("read", Path("/w/./secrets/key")) is not None
    # A path that really is elsewhere is still elsewhere.
    assert policy.decide("read", Path("/w/public/key")) is None


# --------------------------------------------------------- recursive delete --


def test_a_recursive_delete_is_refused_when_a_rule_could_match_inside_it() -> None:
    """The gate of G7, and the reason that it is a refusal and not a walk.

    The rules describe paths, including paths that do not exist yet. The
    operating system enumerates the tree at delete time. So a check of the path
    of the directory itself cannot answer "is anything in here protected?". A
    check of each child would still race the children created after the check.
    """
    policy = _policy(Rule(operations=("write",), paths=("build/keep/**",), mode="deny"))
    assert policy.deletion_reason(Path("/w/build"), recursive=True) is not None
    # Not recursive: one path, one question, and this one is not protected.
    assert policy.deletion_reason(Path("/w/build"), recursive=False) is None
    # A sibling tree that the rule cannot reach into stays deletable. Otherwise
    # the fence would refuse every delete that anyone attempted.
    assert policy.deletion_reason(Path("/w/dist"), recursive=True) is None


def test_a_dot_dot_spelling_cannot_slip_past_a_recursive_refusal() -> None:
    """The recursive check normalizes the path, as `decide` and `objection` do (D6).

    A `Path` keeps a `..` segment. The recursive branch of `deletion_reason`
    built its spellings from the path as given. `/w/x/../build` then matched no
    rule that names `build`, and the delete that `/w/build` refuses was allowed.

    Sabotage: pass `path.as_posix()` to `_spellings` in `deletion_reason` again,
    and the first assertion fails.
    """
    policy = _policy(Rule(operations=("write",), paths=("build/keep/**",), mode="deny"))

    assert policy.deletion_reason(Path("/w/x/../build"), recursive=True) is not None
    assert policy.deletion_reason(Path("/w/dist/../build"), recursive=True) is not None
    # A path that really is elsewhere is still deletable.
    assert policy.deletion_reason(Path("/w/build/../dist"), recursive=True) is None


def test_a_leading_wildcard_refuses_every_recursive_delete() -> None:
    """`**/*.env` could match anything, so every recursive delete is refused. A
    pattern that starts with `**` says nothing about where it will match, and the
    only safe reading of "I cannot tell" on a delete is no."""
    policy = _policy(Rule(operations=("write",), paths=("**/*.env",), mode="deny"))
    assert policy.deletion_reason(Path("/w/anywhere"), recursive=True) is not None


def test_containment_is_on_separators_not_on_string_prefixes() -> None:
    """`/w/xyz` is not inside `/w/x`. A prefix test would have refused the delete
    of every directory whose name merely started the same way."""
    policy = _policy(Rule(operations=("write",), paths=("/w/x/**",), mode="deny"))
    assert policy.deletion_reason(Path("/w/xyz"), recursive=True) is None
    assert policy.deletion_reason(Path("/w/x"), recursive=True) is not None


def test_an_allow_rule_never_causes_a_recursive_refusal() -> None:
    """The fail-closed check looks for rules that would *object*. An `allow`
    inside the tree is agreement. If it counted as a possible objection, writing
    an exception would be the way to forbid a delete."""
    policy = _policy(Rule(operations=("write",), paths=("build/**",), mode="allow"))
    assert policy.deletion_reason(Path("/w/build"), recursive=True) is None


# ------------------------------------------------------------ through the fs --


async def test_a_denied_read_never_opens_the_file(mount: MountProfile, tmp_path: Path) -> None:
    """The gate is on `fs/read-intent`, so the refusal happens before the open. A
    check that ran after the open would be a report, which is the failure that
    this seam exists to avoid."""
    (tmp_path / "secret.env").write_text("TOKEN=hunter2", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path, {"paths": ["*.env"], "mode": "deny"})

    with pytest.raises(FsDenied) as refused:
        await ctx.require(FS).read("secret.env", scope=DEPLOYMENT)
    assert "denied by permissions-fs" in str(refused.value)


async def test_a_denied_write_and_edit_are_both_refused(
    mount: MountProfile, tmp_path: Path
) -> None:
    """One `write` rule covers create, overwrite and in-place edit.

    If they were split, "you may not write here" would leave `edit` open. Nobody
    means that when they write that line.
    """
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "lib.py").write_text("x = 1\n", encoding="utf-8")
    ctx = await _mounted(
        mount, tmp_path, {"operations": ["write"], "paths": ["vendor/**"], "mode": "deny"}
    )

    with pytest.raises(FsDenied):
        await ctx.require(FS).write("vendor/lib.py", "x = 2\n", scope=DEPLOYMENT)
    with pytest.raises(FsDenied):
        await ctx.require(FS).edit("vendor/lib.py", "x = 1", "x = 2", scope=DEPLOYMENT)
    # Nothing changed on disk. This is the only assertion that proves that the
    # gate ran before the write and not after it.
    assert (tmp_path / "vendor" / "lib.py").read_text(encoding="utf-8") == "x = 1\n"
    # The read the rule did not mention is still allowed.
    assert "x = 1" in (await ctx.require(FS).read("vendor/lib.py", scope=DEPLOYMENT)).text


async def test_a_concealed_path_is_absent_from_glob_and_grep(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Enumeration is filtered, not asked about, and it is filtered during the walk.

    `grep` reads every file that it visits. A filter over its matches would
    return nothing but would have read the whole protected file. The assertion
    that matters is that the *contents* never appear.
    """
    (tmp_path / "app.py").write_text("TOKEN = 1\n", encoding="utf-8")
    (tmp_path / "secret.env").write_text("TOKEN=hunter2\n", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path, {"paths": ["*.env"], "mode": "deny"})

    paths = await ctx.require(FS).glob("**/*", scope=DEPLOYMENT)
    assert any(path.endswith("app.py") for path in paths)
    assert not any(path.endswith("secret.env") for path in paths)

    matches = await ctx.require(FS).grep("TOKEN", scope=DEPLOYMENT)
    assert [match.path for match in matches] == [str(tmp_path / "app.py")]
    assert not any("hunter2" in match.text for match in matches)


async def test_a_denied_tree_is_never_entered(mount: MountProfile, tmp_path: Path) -> None:
    """P6-19. `deny secrets/**` refuses the directory, not each file in it.

    The glob is the reason that this needed a second question and could not
    reuse the check for a file. `**` names what is *under* `secrets`, so `secrets` itself
    does not match its own rule. A concealment check on the directory says
    "nothing to hide" and the walk descends. The rule is then enforced one time
    per file, in a tree that the walk should never have entered.
    `_refuses_under` is the judgment that `deletion_reason` already made for
    exactly this shape, and the walk screen is its second consumer.
    """
    (tmp_path / "secrets").mkdir()
    for index in range(4):
        (tmp_path / "secrets" / f"key{index}.env").write_text(f"TOKEN=hunter{index}\n")
    (tmp_path / "app.py").write_text("TOKEN = 1\n")
    ctx = await _mounted(mount, tmp_path, {"paths": ["secrets/**"], "mode": "deny"})

    paths = await ctx.require(FS).glob("**/*", scope=DEPLOYMENT)
    assert any(path.endswith("app.py") for path in paths)
    assert not any("secrets" in path for path in paths), "neither the tree nor its contents"

    matches = await ctx.require(FS).grep("TOKEN", scope=DEPLOYMENT)
    assert [match.path for match in matches] == [str(tmp_path / "app.py")]
    assert not any("hunter" in match.text for match in matches)


async def test_a_tree_an_allow_rule_names_is_still_entered(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A walk prunes for a rule that refuses, and never for an `allow`.

    `covers_whole` judges from the pattern alone and cannot see which files
    exist. `docs/**` covers the whole of `docs`, so the pruning check must skip
    an `allow` rule. Otherwise a rule that was written to *permit* something
    would hide it.
    """
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "readme.md").write_text("hello\n")
    ctx = await _mounted(mount, tmp_path, {"paths": ["docs/**"], "mode": "allow"})

    walked = await ctx.require(FS).glob("**/*", scope=DEPLOYMENT)
    assert any(path.endswith("readme.md") for path in walked)


async def test_a_workspace_scoped_read_rule_does_not_prune_the_workspace(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The over-refusal that pruning could have introduced, and the guard against it.

    A rule written `paths: ["**"]` covers every directory in the tree. Scoped
    `outside-workspace`, it is precisely the rule that is written to exempt the
    workspace. Without the scope guard that `decide` also has, one such rule
    prunes the whole walk and the agent can enumerate nothing at all.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("TOKEN = 1\n")
    ctx = await _mounted(
        mount,
        tmp_path,
        {
            "operations": ["read"],
            "paths": ["**"],
            "scope": "outside-workspace",
            "mode": "deny",
        },
    )

    walked = await ctx.require(FS).glob("**/*", scope=DEPLOYMENT)
    assert any(path.endswith("app.py") for path in walked)


async def test_interrupt_asks_and_the_answer_decides(mount: MountProfile, tmp_path: Path) -> None:
    """`interrupt` is an approval, through the same seam that `hitl` uses.

    There is one approval channel, so a deployment configures its answerer one
    time. A refusal reads as a refusal, and not as a path that quietly did
    nothing.
    """
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path, {"paths": ["notes.md"], "mode": "interrupt"})
    session = ctx.require(SESSIONS).create("asked")
    agent = StubAgent(ctx, session)
    answers: list[str] = ["allowed-once"]
    answer_approvals(ctx, lambda: answers[0])

    assert "hello" in (await ctx.require(FS).read("notes.md", agent=agent, scope=DEPLOYMENT)).text
    answers[0] = "rejected"
    with pytest.raises(FsDenied) as refused:
        await ctx.require(FS).read("notes.md", agent=agent, scope=DEPLOYMENT)
    # The sentence of the seam itself, so that a model can tell the "no" of a
    # human from a missing channel. Only one of the two is worth re-planning
    # around.
    assert "the user rejected" in str(refused.value)
    assert events_of(session, "approval/decided")


async def test_a_canceled_call_is_not_asked_about(mount: MountProfile, tmp_path: Path) -> None:
    """A prompt is a question to a person, so an abandoned turn does not ask one.

    The test passes no `cancel=` argument. The seam defaults to the token of the
    agent itself, so this checks the claim and not the plumbing. All three
    operations are tested, because a canceled read costs a wasted question and a
    canceled write costs a file.
    """
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path, {"paths": ["notes.md"], "mode": "interrupt"})
    session = ctx.require(SESSIONS).create("canceled")
    agent = StubAgent(ctx, session)
    asked = answer_approvals(ctx, "allowed-once")
    agent.signal.cancel("the user stopped the turn")
    fs = ctx.require(FS)

    with pytest.raises(FsDenied):
        await fs.read("notes.md", agent=agent, scope=DEPLOYMENT)
    with pytest.raises(FsDenied):
        await fs.write("notes.md", "new", agent=agent, scope=DEPLOYMENT)
    with pytest.raises(FsDenied):
        await fs.edit("notes.md", "hello", "bye", agent=agent, scope=DEPLOYMENT)
    assert asked == [], "a canceled turn must not reach the answerer"
    assert (tmp_path / "notes.md").read_text(encoding="utf-8") == "hello\n", "nothing was written"


async def test_interrupt_without_an_answerer_denies(mount: MountProfile, tmp_path: Path) -> None:
    """Fail closed, in the same way as every other path through `ctx.approval`. A
    missing channel is a misconfigured deployment, not consent."""
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path, {"paths": ["notes.md"], "mode": "interrupt"})
    agent = StubAgent(ctx, ctx.require(SESSIONS).create("unanswered"))

    with pytest.raises(FsDenied) as refused:
        await ctx.require(FS).read("notes.md", agent=agent, scope=DEPLOYMENT)
    assert "no approval channel" in str(refused.value)


async def test_an_agentless_interrupt_denies_rather_than_prompting(
    mount: MountProfile, tmp_path: Path
) -> None:
    """No agent means no session, and a prompt raised outside a session is worse
    than useless.

    `ApprovalService.request` derives the session from the agent. An ask with no
    agent would put a question in front of a human with **nothing written to the
    log**. It would also skip the `approval_policy: "never"` short-circuit that
    a deployment set on purpose. Both existing consumers of the seam refuse for
    exactly this reason. The first draft of this row did not.
    """
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path, {"paths": ["notes.md"], "mode": "interrupt"})
    asked: list[Any] = []
    answer_approvals(ctx, lambda: noted(asked, "prompted", "allowed-once"))

    with pytest.raises(FsDenied):
        await ctx.require(FS).read("notes.md", scope=DEPLOYMENT)
    assert asked == [], "a human was prompted with nowhere to record the answer"


async def test_a_rule_may_carry_its_own_words_for_the_prompt(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`description`: the path is not always enough to decide by, and an operator
    who wrote the rule knows why it is there."""
    (tmp_path / "notes.md").write_text("hello\n", encoding="utf-8")
    ctx = await _mounted(
        mount,
        tmp_path,
        {"paths": ["notes.md"], "mode": "interrupt", "description": "shared team notes"},
    )
    session = ctx.require(SESSIONS).create("described")
    agent = StubAgent(ctx, session)
    answer_approvals(ctx, lambda: "rejected")

    with pytest.raises(FsDenied):
        await ctx.require(FS).read("notes.md", agent=agent, scope=DEPLOYMENT)
    (asked,) = events_of(session, "approval/asked")
    assert asked.data["reason"] == "shared team notes"


async def test_nothing_is_restricted_by_default(mount: MountProfile, tmp_path: Path) -> None:
    """Layering the bundle must not start to refuse file access. This is the rule
    of `hitl`, for the same reason: a harness that refuses on first run has rules
    that nobody chose."""
    (tmp_path / "secret.env").write_text("TOKEN=1\n", encoding="utf-8")
    ctx = await mount(row("fs", root=str(tmp_path)), profile=PROFILE)

    assert "TOKEN" in (await ctx.require(FS).read("secret.env", scope=DEPLOYMENT)).text
    assert await ctx.require(FS).glob("**/*.env", scope=DEPLOYMENT)


async def test_a_tool_call_is_covered_without_the_row_naming_the_tool(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The reason that the rules are attached to the seam and not to
    `tools/pre-execute`.

    `read` is never mentioned in this module. The tool is covered because it
    goes through `ctx.fs`. A Code Mode binding and a second editing tool are
    covered in the same way. A rule keyed by tool name would have silently
    covered none of them. That is the failure that the truncation pass of
    `compaction` and `offload` both argue against in their own docstrings.

    The result also arrives as a **denial**, not a failure. `FsDenied` carries
    `failure_kind`, so a Code Mode program cannot read a policy veto as a tool
    that merely broke and then try something else.
    """
    (tmp_path / "secret.env").write_text("TOKEN=hunter2\n", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path, {"paths": ["*.env"], "mode": "deny"})
    session = ctx.require(SESSIONS).create("gated")

    await run_tool_calls(
        ctx,
        session,
        ToolCallBlock(id="c1", name="read", arguments=json.dumps({"path": "secret.env"})),
    )

    assert "denied by permissions-fs" in result_text(session, "c1")
    assert "hunter2" not in result_text(session, "c1")
    block = result_block(session, "c1")
    assert block is not None and block.is_error is True


async def test_an_empty_rule_set_attaches_nothing(mount: MountProfile, tmp_path: Path) -> None:
    """The row mounted with no rules does nothing at all.

    It does not "allow everything through a predicate that says yes". A screen
    is consulted for each file that a walk visits, so a `grep` over a repository
    would pay a Python call per candidate to answer a question that nobody
    asked. The service is still published, because "no rules" is an answer that
    `phern doctor` must be able to give.
    """
    (tmp_path / "secret.env").write_text("TOKEN=1\n", encoding="utf-8")
    ctx = await _mounted(mount, tmp_path)

    assert ctx.require(FS_PERMISSIONS).rules == ()
    assert "TOKEN" in (await ctx.require(FS).read("secret.env", scope=DEPLOYMENT)).text
    assert await ctx.require(FS).glob("**/*.env", scope=DEPLOYMENT)


# ------------------------------------------------------------------- reach --


async def test_the_reach_message_toggles_with_a_sandbox(
    mount: MountProfile, tmp_path: Path
) -> None:
    """E9, and the reason that `reach` reads the seam and not a mount-time bool.

    These rules bound access *through* `ctx.fs`. A code cell that calls `open()`
    directly never fires an intent. Saying so is only useful if the sentence
    changes when the situation changes. People stop reading a disclaimer that is
    always there. A sentence computed at mount would be the wrong sentence for a
    profile that layers its backend afterwards.
    """
    ctx = await _mounted(mount, tmp_path, {"paths": ["*.env"], "mode": "deny"})
    assert ctx.require(FS_PERMISSIONS).reach == UNBOUNDED_REACH
    assert "not covered" in ctx.require(FS_PERMISSIONS).reach

    # The test does not mount the P6-04 backend (`sandbox-local`). The stand-in is
    # what the seam reads: a provider in the slot. The claim under test is the
    # toggle, and not what any particular backend enforces.
    ctx.require(SANDBOX).register_provider(StubSandboxProvider())
    assert ctx.require(FS_PERMISSIONS).reach == BOUNDED_REACH
    assert "not covered" not in ctx.require(FS_PERMISSIONS).reach


async def test_the_reading_reaches_doctor_even_with_no_rules(
    mount: MountProfile, tmp_path: Path
) -> None:
    """P4-12. The empty case is the one worth printing. A deployment that wrote no
    rules has a *wider* reach than one that wrote a deny list. A report that
    appeared only after someone configured something would tell people what they
    cannot do, and never that nothing stopped them.
    """
    ctx = await _mounted(mount, tmp_path)

    rows = report_section(ctx, "File permissions")

    assert rows["rules"] == "none — nothing is refused here"
    assert rows["reach"] == UNBOUNDED_REACH


async def test_the_reading_names_each_rule(mount: MountProfile, tmp_path: Path) -> None:
    """A person who reads `phern doctor` asks "what is refused here". A count
    answers a different question."""
    ctx = await _mounted(mount, tmp_path, {"paths": ["*.env"], "mode": "deny"})

    rows = report_section(ctx, "File permissions")

    assert rows["rules"] == "1"
    assert "*.env" in " ".join(f"{label} {value}" for label, value in rows.items())


def test_a_policy_with_no_sandbox_seam_reports_unconfined() -> None:
    """Fail closed, and also simply true: nothing bounds the paths that these
    rules cannot see."""
    assert _policy().reach == UNBOUNDED_REACH
    assert not _policy().confined


async def test_an_anchored_rule_still_applies_inside_a_worktree(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The hole that D21 opened, and the reason that the root is asked per agent.

    A rule written `secrets/**` is anchored to *the workspace*. When the root was
    captured at mount, an agent at the `worktree` tier wrote in a directory
    outside that prefix. `_spellings` then offered only the absolute path, and
    the rule silently stopped applying. That happened in the one place where a
    rule is most needed, and nothing failed to show it.
    """
    ctx = await _mounted(mount, tmp_path, {"operations": ["write"], "paths": ["secrets/**"]})
    session = ctx.require(SESSIONS).create("s1")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    ctx.require(WORKSPACE).register_provider(StubWorkspaceProvider(root=tmp_path / "trees"))
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s1", agent_id=agent.id, base=tmp_path, session=session
    )
    assert workspace.root != ctx.require(FS).root, "this test is only meaningful for a fresh root"

    result = await run_tool(
        ctx,
        "write",
        {"path": str(workspace.root / "secrets" / "key.pem"), "content": "x"},
        agent=agent,
    )

    assert result.is_error
    assert not (workspace.root / "secrets" / "key.pem").exists()


def test_two_agents_are_judged_against_their_own_roots(tmp_path: Path) -> None:
    """The same rule, two agents, two answers. That is the point.

    `secrets/**` names a different directory in the tree of each agent. A policy
    that resolved one root for every agent would answer for the wrong one.
    """

    policy = _policy(Rule(operations=("write",), paths=("secrets/**",)), root="/w")
    one, two = _Agent(Path("/w/a")), _Agent(Path("/w/b"))

    assert policy.objection("write", Path("/w/a/secrets/k"), one) is not None
    assert policy.objection("write", Path("/w/a/secrets/k"), two) is None
    assert policy.objection("write", Path("/w/b/secrets/k"), two) is not None


# ----------------------------------------------- the default write scope --


async def test_writing_inside_the_workspace_never_asks(mount: MountProfile, tmp_path: Path) -> None:
    """The first half of E6, and the purpose of the default write scope.

    There is not one prompt, for any number of writes. The agent owns this
    checkout, so a question about its own tree is a question about nothing.
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


async def test_a_write_named_through_a_symlink_into_the_workspace_never_asks(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**Both sides of the boundary must be in one spelling, and only one of them
    is canonical by construction.**

    Workspace roots are canonicalized where they are minted
    (`ph.paths.canonical`), because the kernel matches the path that it
    resolves. The *candidate* is not canonical: `FsService.resolve` passes an
    absolute path through as the model wrote it, deliberately. A path can name
    the tree of this agent through a link. `$TMPDIR` produces such a path on
    macOS, where `/var` is `/private/var`. That path reaches the gate with a
    different spelling from the root that it is inside, and a plain prefix
    compare would interrupt a person about a write that the sandbox permits.
    That is the disagreement of E6, in the fail-safe direction and not the
    fail-open one.

    The resolution happens only *after* the cheap compare has said "outside".
    The ordinary write therefore pays no syscall for it, and this write is
    answered correctly.
    """
    ctx = await _scoped(mount, tmp_path)
    agent, workspace = await scoped_agent(ctx, tmp_path)
    asked = answer_approvals(ctx, "rejected")
    link = tmp_path / "as-linked"
    link.symlink_to(workspace.root)

    await run_tool(ctx, "write", {"path": str(link / "inside.txt"), "content": "x"}, agent=agent)

    assert asked == [], "the agent's own tree, named the other way"
    assert (workspace.root / "inside.txt").read_text(encoding="utf-8") == "x"


async def test_writing_to_scratch_never_asks(mount: MountProfile, tmp_path: Path) -> None:
    """Scratch is outside the worktree *by design* (E5). It is also the one place
    where a read-only or ephemeral agent is told that it may write. A scope that
    covered only `root` would prompt on exactly the writes that the design
    invites."""
    ctx = await _scoped(mount, tmp_path)
    agent, workspace = await scoped_agent(ctx, tmp_path)
    asked = answer_approvals(ctx, "rejected")

    await run_tool(
        ctx, "write", {"path": str(workspace.scratch / "notes.md"), "content": "kept"}, agent=agent
    )

    assert asked == []
    assert (workspace.scratch / "notes.md").read_text(encoding="utf-8") == "kept"


async def test_one_write_outside_asks_once(mount: MountProfile, tmp_path: Path) -> None:
    """The second half of E6. The rare prompt is the meaningful one, and it names
    the boundary that the write leaves and not only the path."""
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


async def test_a_refused_write_outside_does_not_happen(mount: MountProfile, tmp_path: Path) -> None:
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


async def test_no_answerer_denies_rather_than_allowing(mount: MountProfile, tmp_path: Path) -> None:
    """Fail closed, as every asking row in this package does: an approval that
    nobody can answer is not an approval."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    ctx = await _scoped(mount, tmp_path)
    agent, _workspace = await scoped_agent(ctx, tmp_path)

    result = await run_tool(
        ctx, "write", {"path": str(outside / "escape.txt"), "content": "x"}, agent=agent
    )

    assert result.is_error
    assert not (outside / "escape.txt").exists()


async def test_an_agent_with_no_workspace_is_unaffected(
    mount: MountProfile, tmp_path: Path
) -> None:
    """No workspace, so no scope to be outside of.

    A profile that layers the rule without a containment tier keeps the
    behavior that it has today. It does not get a boundary drawn around a
    directory that nobody chose.
    """
    ctx = await _scoped(mount, tmp_path)
    session = ctx.require(SESSIONS).create("s1")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    answer_approvals(ctx, "rejected")
    target = tmp_path / "anywhere.txt"

    await run_tool(ctx, "write", {"path": str(target), "content": "x"}, agent=agent)

    assert target.read_text(encoding="utf-8") == "x"


async def test_an_explicit_deny_beats_the_default(mount: MountProfile, tmp_path: Path) -> None:
    """The order of the rules *is* the precedence.

    The explicit rule is above the default write scope in the one list, and the
    first match wins. So the explicit `deny` refuses first and the default is
    never reached. That is what makes "default write scope" a default and not an
    override of the rules of the operator.
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


async def test_a_leading_wildcard_hides_the_file_without_refusing_the_tree(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The idiomatic deny, and the walk that it must not shut down.

    `_could_match_under` answers `True` for a pattern that starts with `**`,
    because for a recursive *delete* "could match anywhere" must round to
    refusal. Applied to a walk, the same rounding refuses everywhere.
    `deny **/.env` is the way that anyone would write the rule. It made every
    directory a candidate for pruning, and `glob("**/*")` over this repository
    went from 11 489 results to 6. The file that the rule names is still
    concealed as the walk passes it. A leading wildcard cannot justify a refusal
    to look.
    """
    for relative in ("src/a.py", "docs/b.md", "src/.env", "top.txt"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x\n")
    ctx = await _mounted(mount, tmp_path, {"paths": ["**/.env"], "mode": "deny"})

    # Files only, because `_walk` never yields a directory. So this is every
    # file in the tree except the one that the rule names.
    found = sorted(
        path[len(str(tmp_path)) + 1 :]
        for path in await ctx.require(FS).glob("**/*", scope=DEPLOYMENT)
    )
    assert found == ["docs/b.md", "src/a.py", "top.txt"]

    # The delete side keeps the harder rounding. That is the reason that the two
    # callers of `_refuses_under` differ. It is not an oversight.
    policy = _policy(Rule(paths=("**/.env",), mode="deny"), root=str(tmp_path))
    assert policy.deletion_reason(tmp_path / "src", recursive=True) is not None


async def test_a_directory_the_sandbox_allows_is_not_outside_the_workspace(
    mount: MountProfile, tmp_path: Path
) -> None:
    """P6-38, E6: the prompt boundary is the enforced boundary. A write into a
    directory that `sandbox-allow` binds writable must not be asked about as if
    it left the workspace. The backend would permit it, and a prompt that
    describes a boundary that is not there is the drift that the shared set
    exists to prevent."""
    cache = tmp_path / "shared-cache"
    cache.mkdir()
    ctx = await mount(
        {"id": "sandbox-allow", "config": {"paths": [str(cache)]}},
        {
            "insert": [
                {
                    "id": "permissions-fs",
                    "name": "permissions-fs",
                    "config": {
                        "rules": [
                            {
                                "operations": ["write"],
                                "paths": ["**"],
                                "mode": "deny",
                                "scope": "outside-workspace",
                            }
                        ]
                    },
                }
            ]
        },
    )
    (tmp_path / "project").mkdir()
    agent, _ = await scoped_agent(ctx, tmp_path)
    permissions = ctx.require(FS_PERMISSIONS)

    assert permissions.objection("write", cache / "wheel.whl", agent=agent) is None
    assert (
        permissions.objection("write", tmp_path.parent / "elsewhere.txt", agent=agent) is not None
    )

    # And per agent (S7b, item 3): an agent narrowed out of the directory is asked
    # about it, because its sandbox no longer binds it. Sabotage: ask
    # `allowed_paths_of` without the agent, and the narrowed agent writes there
    # unasked.
    ctx.require(SANDBOX).restrict_paths((), scope=agent.ctx)
    assert permissions.objection("write", cache / "wheel.whl", agent=agent) is not None


async def test_a_spilled_result_is_readable_but_not_writable(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The reach that the retrieval hint depends on, stated and not incidental.

    `tool-result-offload` replaces an oversized result with a preview and the
    path that it was written to under `$PH_HOME/spill`. That path is outside the
    workspace of the agent. With the shipped rule set, a read of it works only
    because that rule set restricts *writes* and nothing else. Without the
    exemption, a deployment that adds `deny read outside-workspace` would hand
    the model a path and then refuse the path when the model follows it,
    silently, in every session. The harness telling the model that something is
    on disk and then denying it is the failure that the docstring of the spill
    seam rules out in its first lines.

    Reads only, and that half is worth a second check. Nothing invites an agent
    to write into the store of the harness. An agent that could write there
    could forge a blob under a locator that the log already names.

    And only the session of this agent. The locator is `root/<owner>/…` with the
    session id as owner. An exemption for the store *root* would have let any
    agent read the spilled results of every other session. That is wider than
    the claim, which is only that a path handed to this session can be followed.
    The test asks the store for the locator and does not write it by hand,
    because a test that writes the layout out still passes when the layout
    moves.

    **The producer must be mounted.** That is the purpose of folding the claims
    and not assuming the layout. A deployment that spills nothing hands the
    model no paths, and an exemption for one would grant a reach that nothing
    asked for. `tool-result-offload` registers the claim, so it is in the
    profile here exactly as it is in a real profile.
    """
    ctx = await mount(
        {
            "insert": [
                {"id": "tool-result-offload", "name": "tool-result-offload"},
                {
                    "id": "permissions-fs",
                    "name": "permissions-fs",
                    "config": {
                        "rules": [
                            {
                                "operations": ["read", "write"],
                                "paths": ["**"],
                                "mode": "deny",
                                "scope": "outside-workspace",
                            }
                        ]
                    },
                },
            ]
        },
    )
    (tmp_path / "project").mkdir()
    agent, _ = await scoped_agent(ctx, tmp_path)
    permissions = ctx.require(FS_PERMISSIONS)
    store = ctx.require(SPILL_STORE)
    session = agent.session
    assert session is not None
    spilled = store.plan(
        owner=session.id, suggested_name="large_tool_results/call-7", content=b"x"
    ).locator

    assert permissions.objection("read", spilled, agent=agent) is None
    assert permissions.objection("write", spilled, agent=agent) is not None
    # The blob of another session is not for this agent to follow. The exemption
    # is the paths that the agent was handed, and not the store.
    other = store.plan(
        owner="s-someone-else", suggested_name="large_tool_results/call-7", content=b"x"
    ).locator
    assert permissions.objection("read", other, agent=agent) is not None
    # The rule still applies everywhere else, so this is an exemption. The scope
    # is not switched off.
    assert permissions.objection("read", tmp_path.parent / "elsewhere.txt", agent=agent) is not None


async def test_every_row_that_hands_the_model_a_path_is_exempt(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Not only the two offloads: the set is what the rows declare.

    `compaction-summarize` spills the pre-compaction history, and its
    replacement tells the model it "has been saved to {file_path} should you
    need to refer back to it". Under the first cut, an unconditional
    `root/<session id>`, that read was exempt by an accident of the layout.
    Folding the claims made the set correct and, until this, one row short.
    Compaction shipped alongside the offloads, so nothing noticed. A profile
    that kept compaction without the offloads would have pointed the model at a
    file and then refused the read.

    Sabotage: drop `hands_paths=True` from the compaction claim.
    """
    ctx = await mount(
        {
            "insert": [
                {"id": "compaction-summarize", "name": "compaction-summarize"},
                {
                    "id": "permissions-fs",
                    "name": "permissions-fs",
                    "config": {
                        "rules": [
                            {
                                "operations": ["read"],
                                "paths": ["**"],
                                "mode": "deny",
                                "scope": "outside-workspace",
                            }
                        ]
                    },
                },
            ]
        },
    )
    (tmp_path / "project").mkdir(exist_ok=True)
    agent, _ = await scoped_agent(ctx, tmp_path)
    session = agent.session
    assert session is not None
    store = ctx.require(SPILL_STORE)
    history = store.plan(
        owner=session.id, suggested_name="conversation_history/1.md", content=b"x"
    ).locator

    assert ctx.require(FS_PERMISSIONS).objection("read", history, agent=agent) is None


def test_a_wildcard_that_is_not_a_star_still_makes_a_head() -> None:
    """D11: `fs.literal_head` and `_WILDCARD` say why the alphabet is shared.

    `sec?ets/**` and `[sf]ecrets/**` are ordinary globs that `matches_glob`
    honors. A head that was split on `*` alone read them as literal directory
    names that no tree contains. So the rule concealed each file that it named,
    but it refused nothing and pruned nothing.
    """
    for pattern in ("sec?ets/**", "[sf]ecrets/**"):
        policy = _policy(Rule(operations=("write",), paths=(pattern,), mode="deny"))
        assert policy.deletion_reason(Path("/w"), recursive=True) is not None, pattern

    # The refusal is still bounded. This is not "refuse everything", which is
    # the other way to pass. Both spellings now show it. The check matches
    # segment by segment, so a leading `[sf]` constrains the first segment
    # exactly as `sec?` does, and does not read as "could match anywhere".
    for pattern in ("sec?ets/**", "[sf]ecrets/**"):
        bounded = _policy(Rule(operations=("write",), paths=(pattern,), mode="deny"))
        assert bounded.deletion_reason(Path("/w/elsewhere"), recursive=True) is None, pattern


class _Agent(StubAgent):
    """A real `AgentHandle` that carries a workspace root.

    `roots` is the only thing here that looks at the agent, and it reads
    `.root`. `objection` and `screen` still declare `AgentHandle`, which a bare
    three-field class never satisfied. So this class subclasses the stub that
    the type checker already accepts, and does not cast something that is not
    an `AgentHandle`.
    """

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root


def _walk(policy: FsPermissions, path: str) -> str:
    """What the walk screen does with a directory: `screen` for `is_dir=True`."""
    return policy.screen(path, path.rsplit("/", 1)[-1], _Agent(Path("/w")), True)


@pytest.mark.parametrize(
    "pattern", ["secrets/**", "sec?ets/**", "sec*ets/**", "[sf]ecrets/**", "?ecrets/**"]
)
def test_every_spelling_of_a_directory_rule_guards_that_directory(pattern: str) -> None:
    """D11: a glob was read as a path, and the answer came out inverted.

    `_could_match_under` compared the *literal head* of the pattern (everything
    before the first wildcard) as a path. That is exact only when the head stops
    at a separator. `sec?ets/**` has the head `sec`, which is an ancestor of
    nothing that the rule covers and exactly equal to a directory that the rule
    can never touch. So the rule **allowed** `rm -rf secrets` and walked into
    it, while it **refused** `rm -rf sec` and pruned that. That is the one
    direction in which a permission gate must never fail, and its mirror, at
    the same time.

    Every spelling below means "the secrets directory", so every one must
    behave as the plain `secrets/**` does: guard `secrets`, and leave alone the
    directories that it cannot name. They agree because the match is segment by
    segment, in the dialect of the glob itself.

    Sabotage: compare `literal_head(pattern)` as a path again and the `?` and
    `*` rows allow deleting `secrets` while refusing `sec`.
    """
    policy = _policy(Rule(operations=("read", "write"), paths=(pattern,), mode="deny"))

    assert policy.deletion_reason(Path("/w/secrets"), recursive=True) is not None, (
        f"{pattern} let a recursive delete destroy the directory it guards"
    )
    assert _walk(policy, "/w/secrets") == "prune", f"{pattern} walked into what it conceals"

    for elsewhere in ("sec", "elsewhere"):
        assert policy.deletion_reason(Path(f"/w/{elsewhere}"), recursive=True) is None, (
            f"{pattern} refused deleting {elsewhere}/, which it can never name"
        )
        assert _walk(policy, f"/w/{elsewhere}") == "yield", (pattern, elsewhere)


def test_a_walk_prunes_what_a_rule_covers_wholesale_and_nothing_else() -> None:
    """D11: the walk and the delete ask different questions.

    A delete asks whether the rule could name *anything* that it removes, and
    must refuse on "could". A walk asks whether the rule names *everything* in a
    directory, because only then is skipping the directory lossless. A rule that
    covers part of a directory is enforced file by file as the walk passes.

    One function used to answer both, with a flag that adjusted it for the walk.
    No setting of the flag was right for every rule below. It hid too much:
    `deny read src/*.pem` pruned all of `src`, and every allowed file in it
    vanished from glob and grep. It also missed: `deny read **/node_modules/**`
    pruned no `node_modules` at all. And `**/.env` must never prune the tree, or
    the idiomatic spelling of the most common rule hides the whole workspace.

    Sabotage: have `screen` ask the delete's question and `src` is pruned.
    """
    cases = [
        ("**/.env", "/w/src", "yield"),
        ("src/*.pem", "/w/src", "yield"),
        ("secrets/*.key", "/w/secrets", "yield"),
        ("a/secret/**", "/w/a", "yield"),
        ("*/secret/**", "/w/a", "yield"),
        ("*/secret/**", "/w/a/secret", "prune"),
        ("**/node_modules/**", "/w/a/node_modules", "prune"),
        ("secrets/**", "/w/secrets/a/b", "prune"),
    ]
    for pattern, directory, expected in cases:
        policy = _policy(Rule(operations=("read",), paths=(pattern,), mode="deny"))
        assert _walk(policy, directory) == expected, (pattern, directory)

    # The delete still refuses every one of them, because each could name
    # something that the delete would remove.
    for pattern, directory, _ in cases:
        policy = _policy(Rule(operations=("write",), paths=(pattern,), mode="deny"))
        assert policy.deletion_reason(Path(directory), recursive=True) is not None, pattern


def test_a_double_star_inside_a_segment_still_refuses_a_delete() -> None:
    """D11 review: the permissive regression that the first version introduced.

    `matches_glob` compiles `**` to cross separators *wherever it appears*:
    `data**/key` matches `data/v1/x/key`. The first segment walk recognized `**`
    only as a whole segment. It matched `data**` against `data` as a one-segment
    glob, ran out of pattern, and **allowed** a delete of `data/v1`. The
    literal-head code that it replaced had refused that delete. It was a second
    copy of the dialect, wrong in the one direction in which a delete gate must
    never be wrong.

    Sabotage: treat `**` as special only when it is a whole segment, and the
    first two cases are allowed.
    """
    for pattern, directory, inside in (
        ("data**/key", "/w/data/v1", "data/v1/x/key"),
        ("**.env", "/w/a", "a/b/.env"),
        ("a/b**", "/w/a/bc/d", "a/bc/d/e"),
    ):
        assert matches_glob(inside, pattern), "the fixture no longer shows the dialect crossing"
        policy = _policy(Rule(operations=("write",), paths=(pattern,), mode="deny"))
        assert policy.deletion_reason(Path(directory), recursive=True) is not None, (
            f"deleting {directory} would remove {inside}, which {pattern} protects"
        )


def test_a_rule_does_not_refuse_what_it_provably_cannot_name() -> None:
    """The two places where D11 *loosened* a refusal, stated so that nobody finds
    them later.

    Over-refusing is the safe failure for a delete, so a change in that
    direction is the one worth writing down. Each case below was refused by the
    literal-head approximation and is not refused now, because the pattern
    cannot match anything that the delete would remove:

    * `secrets`, with no `**`, names the directory entry itself. A delete of
      `secrets/old` does not delete `secrets`, so there is nothing to refuse.
      The old code refused it because `secrets/old` is a path descendant of the
      head. That is true and not relevant.
    * `*/secret/**` needs its *second* segment to be `secret`. Nothing under
      `a/other` has one. The old code refused it only because a headless
      pattern rounded to "could match anywhere".

    Sabotage: treat a pattern that runs out above the directory as a match and
    the first case is refused again.
    """
    exact = _policy(Rule(operations=("write",), paths=("secrets",), mode="deny"))
    assert exact.deletion_reason(Path("/w/secrets"), recursive=True) is not None
    assert exact.deletion_reason(Path("/w/secrets/old"), recursive=True) is None

    nested = _policy(Rule(operations=("write",), paths=("*/secret/**",), mode="deny"))
    assert nested.deletion_reason(Path("/w/a/other"), recursive=True) is None


def test_a_recursive_delete_of_the_workspace_root_sees_relative_rules() -> None:
    """D11: the root is the one path with no relative spelling.

    `_spellings` strips a prefix that has a trailing slash. The root itself
    never matched that prefix and came back with its absolute name alone. A
    relative rule then could not see that a delete of the root deletes
    everything that the rule names. `rm -rf <workspace>` was refused by
    `deny write /w/secrets/**` and allowed by `deny write secrets/**`, which is
    the same rule written the way that people write rules.
    """
    relative = _policy(Rule(operations=("write",), paths=("secrets/**",), mode="deny"))

    assert relative.deletion_reason(Path("/w"), recursive=True) is not None
    # The absolute spelling of the same rule always worked. Both now agree.
    absolute = _policy(Rule(operations=("write",), paths=("/w/secrets/**",), mode="deny"))
    assert absolute.deletion_reason(Path("/w"), recursive=True) is not None
    # And a tree the rule cannot reach is still deletable.
    assert relative.deletion_reason(Path("/w/build"), recursive=True) is None


def test_a_rule_whose_glob_cannot_compile_is_refused_rather_than_inert() -> None:
    """N6: the literal fallback is right for the `glob` of a model and wrong for
    an ACL.

    `matches_glob` answers a pattern that it cannot compile by matching it
    literally. A `glob` argument is a string that the model typed, and a policy
    check must not raise out from under the call that it checks. For a `deny`
    rule the same answer is the one failure direction that an ACL must never
    have. `[z-a]/**` is a bad character range, so the rule stopped covering
    `secrets/**` and started covering the eight literal characters, which nobody
    will use as a file name.

    Both populations are asserted, because the fix is that they diverge: config
    is refused, and the matcher still falls back for everyone else.

    Sabotage: drop `Rule._usable_globs` and the rule is accepted, guarding nothing.
    """
    with pytest.raises(ValidationError) as caught:
        Rule(operations=("read",), paths=("secrets/**", "[z-a]/**"), mode="deny")
    assert "not a usable glob" in str(caught.value)

    # The matcher itself is unchanged: inert, and it says so in the log.
    assert not matches_glob("secrets/key.pem", "[z-a]/**")
    assert matches_glob("[z-a]/**", "[z-a]/**")
