"""`hitl` — a human between the model and the actions it cannot take back (P4-05, G6).

Deep Agents' `interrupt_on` config and its `manual | auto | yolo` posture, on
`tools/pre-execute`: when a rule matches, the listener returns `Ask` and the
registry routes it through `ctx.approval` (B3). Everything after that already
exists — the seam records `approval/asked` and `approval/decided`, fails closed
on a missing answerer, and re-asks on resume because an `asked` with no `decided`
*is* the pending state.

**Four decisions, not two.** `approve` and `reject` were already reachable;
P4-05 adds `edit` (run it with these arguments instead) and `respond` (do not
run it — tell the model this). Both exist because stopping a turn to say "wrong
path" or "you don't need that, the answer is X" costs a round trip that answering
in place does not.

**The classifier is a predicate, and its verdict is logged with the ask.** A rule
may be `true` (always ask) or carry a `when` — a pattern set matched against the
call's arguments. What it matched goes into the ask's `reason`, so a person is
told *why* they are being asked and an auditor can see what the classifier
thought. A gate whose reasoning is invisible is one nobody can tune.

**`run_code` is the case that motivates the shape.** A cell is not a tool name;
one `ipython` call may delete a tree, force-push, or open a socket. So the
default patterns read the program text, and the row asks about *that cell*
rather than about the transport — which is why matching is on arguments and not
on the tool.

@module ph_stabilize.hitl
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field

from ph.cordis import Context, plugin
from ph.seams.approval import ApprovalDecisionName
from ph.session import Session
from ph.session.json import dumps
from ph.tools.definition import Ask, ToolExecution
from ph.tools.registry import RUN_CODE
from ph.wire import WireModel

__all__ = [
    "DESTRUCTIVE_PATTERNS",
    "PATTERN_SETS",
    "ApprovalMode",
    "Config",
    "Rule",
    "apply",
    "matches",
    "set_mode",
]

ApprovalMode = Literal["manual", "auto", "yolo"]
"""deepagents-code's posture, and the one knob a person reaches for most.

`manual` asks about every configured call. `auto` asks only about the ones a
rule matched — the shipped middle, where a `when` predicate is what separates a
routine write from `rm -rf`. `yolo` asks about nothing, and is a thing a person
turns on deliberately for a session they are watching.
"""

DESTRUCTIVE_PATTERNS: tuple[str, ...] = (
    r"\brm\s+-[a-zA-Z]*[rf]",
    r"\bgit\s+push\b.*--force|\bgit\s+push\s+-f\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b",
    r"\bTRUNCATE\s+TABLE\b",
    r"\bDELETE\s+FROM\b(?!.*\bWHERE\b)",
    r"\bshutil\.rmtree\b",
    r"\bos\.remove\b|\bos\.unlink\b|\bPath\([^)]*\)\.unlink\b",
    r"\bmkfs\b|\bdd\s+if=",
    r"\bchmod\s+-R\b|\bchown\s+-R\b",
    r">\s*/dev/sd[a-z]",
    r"\bcurl\b.*\|\s*(ba)?sh\b|\bwget\b.*\|\s*(ba)?sh\b",
)
"""What the shipped classifier looks for, from the port plan's own list.

Case-insensitive, matched against the *text* of a call's arguments. Deliberately
a small set of things that are hard to undo rather than a general audit: a rule
that fires on everything is one a person learns to approve without reading, and
that is worse than no rule. `DELETE FROM` without a `WHERE` is the one lookahead,
because the version with a clause is ordinary work.
"""


PATTERN_SETS: dict[str, tuple[str, ...]] = {"destructive": DESTRUCTIVE_PATTERNS}
"""Pattern sets a profile can name instead of retyping.

Reachable from a profile, not only from Python: **a security judgement with two
homes is one that disagrees with itself**, and the first profile that wanted this
list retyped a subset of it.

A named set rather than `${...}` interpolation, because the loader's vocabulary
is closed on purpose (I-8) and this is a *typed field on a row's config*, which
is the mechanism a row already has for saying what it accepts.
"""


class Rule(WireModel):
    """When to ask about one tool."""

    preset: str = ""
    """A shipped pattern set to use, by name — see `PATTERN_SETS`.

    Unioned with `when`, so a profile names the curated set and adds the
    deployment-specific patterns beside it rather than choosing between them.
    """
    when: tuple[str, ...] = ()
    """Patterns matched against the call's arguments. Empty means *always ask* —
    the `true` form in upstream's config, spelled as the absence of a condition
    rather than as a second type."""
    allowed_decisions: tuple[ApprovalDecisionName, ...] = ()
    """What the front end should offer; empty means all four.

    Spelled as the seam's own `Literal` rather than as bare strings, so a typo in
    a profile fails at config load instead of rendering a modal with no buttons
    and nothing to say why. Carried on the ask so a deployment can withhold
    `edit` for a tool whose arguments must not be hand-written."""
    description: str = ""
    """Extra words for the prompt, when the tool's name is not enough."""

    def patterns(self) -> tuple[str, ...]:
        """Everything this rule matches on. Empty still means *always ask*."""
        named = PATTERN_SETS.get(self.preset, ()) if self.preset else ()
        return (*named, *self.when)


class Config(WireModel):
    """Row config."""

    mode: ApprovalMode = "auto"
    interrupt_on: dict[str, Rule] = Field(default_factory=dict)
    """Tool name → when to ask. Empty by default: layering this bundle must not
    start prompting, because a harness that asks about everything on first run
    teaches its user to stop reading the prompts."""


@lru_cache(maxsize=64)
def _compiled(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """One rule's patterns, compiled once.

    Bounded by the number of configured rules, which is fixed for a deployment.
    `re`'s own cache would serve, but it is a shared 512-entry LRU and this runs
    on every gated call.
    """
    return tuple(re.compile(one, re.IGNORECASE | re.DOTALL) for one in patterns)


def matches(arguments: Any, patterns: tuple[str, ...]) -> list[str]:
    """What in this call's arguments a pattern matched — the *text*, not the rule.

    The arguments are rendered to text and scanned, rather than any field being
    named: `run_code` carries a program, `bash` a command, `write` a path and a
    body, and a rule that had to know which key to read would be a rule per tool.

    The canonical encoder, because `execution.arguments` is the *frozen* tree:
    `json.dumps` cannot walk a `MappingProxyType`, so it would hand the whole
    tree to `default=` and scan a Python repr — which contains the same
    substrings, and would therefore have gone on matching while being wrong.
    """
    try:
        text = dumps(arguments)
    except (TypeError, ValueError):
        text = str(arguments)
    found: list[str] = []
    for pattern in _compiled(patterns):
        hit = pattern.search(text)
        if hit is not None and hit.group(0) not in found:
            found.append(hit.group(0))
    return found


def set_mode(session: Session, mode: ApprovalMode) -> None:
    """Record a posture change. The last one recorded is the one in force."""
    session.append("approval/mode", {"mode": mode})


def _mode(session: Session | None, config: Config) -> ApprovalMode:
    """The posture in force: the last one recorded, else the row's default.

    A fold, so the toggle survives a resume and the log says when it moved. Its
    *own* event rather than a field on the seam's `approval/policy`: that one is
    also written by `permission-presets`, which knows nothing about a posture, so
    riding it would have made switching preset silently reset the posture — the
    generic hazard of a last-write-wins fold with two writers, one of which does
    not know the field exists.
    """
    event = None if session is None else session.latest("approval/mode")
    recorded = None if event is None else event.data.get("mode")
    return recorded if recorded in ("manual", "auto", "yolo") else config.mode


@plugin("hitl", inject=["approval"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Ask a human before a configured call runs."""

    def rule_for(execution: ToolExecution) -> Rule | None:
        """This call's rule, resolving the reserved transport name.

        A profile writes `run_code` because that is the *reserved* transport
        name — unregisterable, unshadowable, unrestrictable — but the registry
        renames the transport in place to whatever a presentation row calls it,
        so under `rlm` the name reaching here is `ipython`. Keyed literally, the
        Code Mode half of a deployment's human gate is **silently inert**: the
        config parses, the tests pass, and nothing ever asks.

        Resolved rather than requiring profiles to write the presentation name,
        because that name is a *rendering* choice and this is a policy about the
        transport itself — a deployment that renamed its presentation would
        otherwise turn its own approval gate off without touching it.
        """
        rule = config.interrupt_on.get(execution.name)
        if rule is not None:
            return rule
        scope = getattr(execution, "scope", None)
        if execution.name == ctx.tools.view(scope).transport_name:
            return config.interrupt_on.get(RUN_CODE)
        return None

    async def gate(execution: ToolExecution, next_: Any) -> Any:
        rule = rule_for(execution)
        if rule is None:
            return await next_(execution)
        mode = _mode(execution.session, config)
        if mode == "yolo":
            return await next_(execution)
        patterns = rule.patterns()
        matched = matches(execution.arguments, patterns) if patterns else []
        if mode == "auto" and patterns and not matched:
            # A rule with a condition, in the posture that trusts it: nothing
            # matched, so this is the routine case the condition exists to let
            # through. `manual` asks anyway, which is what `manual` means.
            return await next_(execution)
        return Ask(
            reason=_reason(execution, rule, matched),
            allowed_decisions=rule.allowed_decisions,
        )

    ctx.on("tools/pre-execute", gate)


def _reason(execution: ToolExecution, rule: Rule, matched: list[str]) -> str:
    """What the person is told, and what the log records with the ask.

    The classifier's verdict is part of it: a prompt that says only "approve
    this?" makes a person guess what the harness is worried about, and an
    auditor reading `approval/asked` later has nothing to tune against. What it
    quotes is the *matched text* rather than the pattern that matched: showing
    someone the classifier's own regex source presents an implementation where
    `rm -rf /tmp/x` presents their call back to them.
    """
    parts = [rule.description or f"{execution.name} needs approval"]
    if matched:
        parts.append(f"matched {', '.join(repr(one) for one in matched)}")
    return " — ".join(parts)
