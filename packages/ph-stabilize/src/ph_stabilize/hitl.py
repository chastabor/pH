"""`hitl`: a human between the model and the actions that it cannot take back (P4-05, G6).

This row ports the `interrupt_on` config of Deep Agents and its
`manual | auto | yolo` posture. It listens on `tools/pre-execute`. When a rule
matches, the listener returns `Ask`, and the registry routes the ask through
`ctx.approval` (B3).

**The declaration of the tool is the default. A configured rule is the
override** (P6-16). A tool declares `is_irreversible` and is then gated under
`Config.declared`. A key in `interrupt_on` replaces that for one tool, by name.
Upstream has only the name, and a name is a rendering choice. For example, pH
renames its own Code Mode transport in place when a presentation row gives it
another name. A gate that is keyed on the name alone then stops working, with no
error, for the surface that most needs a gate.

Everything after the ask already exists (`ph.seams.approval`). The seam records
`approval/asked` and `approval/decided`. It fails closed when no answerer is
present. It settles an ask that a crash left open and does not ask it again: on
resume, repair closes an `asked` with no `decided` as `interrupted`, because the
turn that waited for it is gone.

**Four decisions, not two.** `approve` and `reject` were already reachable.
P4-05 adds `edit` (run the call with these arguments instead) and `respond` (do
not run the call, and tell the model this). Both exist because stopping a turn
to say "wrong path" or "you don't need that, the answer is X" costs a round
trip, and an answer in place does not.

**The classifier parses. It does not match patterns.** `preset: destructive`
runs `ph_stabilize.destructive`. That module reads each string argument in its
own dialect (shell through `shlex`, SQL as statements, Python through `ast`) and
judges the *structure*. The regexes that it replaces could not read a command
line as a grammar, and they had the failure that this invites. The arguments
were scanned as rendered JSON, so a real newline became the two characters `\\`
and `n`, and every pattern anchored with `\\b` stopped matching anything on a
second line. A `when:` list is still regex, as the escape hatch of the
deployment. It is matched against the decoded strings and not against the
envelope.

The findings go into the `reason` of the ask, in the words of the parser, for
example *`rm -rf build` (removes a directory tree)*. The user is then told what
the harness saw, and an auditor can tune it. Nobody can tune a gate whose
reasoning is invisible.

**`run_code` is the case that motivates the shape.** A cell is not a tool name:
one `python` call may delete a tree, force-push, or open a socket. The
classifier therefore reads the program, follows a `subprocess.run(...)` back
into the shell reader, and asks about *that cell* and not about the transport.

@module ph_stabilize.hitl
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Final, Literal

from pydantic import Field

from ph.cordis import Context, Next, plugin
from ph.json import JsonValue
from ph.keys import APPROVAL, TOOLS
from ph.seams.approval import ApprovalDecisionName
from ph.session import Session
from ph.session.writers import log_writer
from ph.tools.definition import Ask, PreToolDecision, ToolExecution
from ph.tools.registry import RUN_CODE
from ph.wire import WireModel
from ph_stabilize.destructive import findings, strings_in

_LOG = log_writer(__name__)

__all__ = [
    "DESTRUCTIVE",
    "ApprovalMode",
    "Config",
    "Rule",
    "apply",
    "set_mode",
]

ApprovalMode = Literal["manual", "auto", "yolo"]
"""The posture of deepagents-code, and the setting that a user changes most often.

`manual` asks about every configured call. `auto` asks only about the calls that
a rule matched. `auto` is the middle posture and the shipped one. In it, a
`when` predicate separates a routine write from `rm -rf`. `yolo` asks about
nothing. A user turns it on deliberately, for a session that they are watching.
"""

DESTRUCTIVE: Final = "destructive"
"""The one shipped preset: the parsed classifier in `ph_stabilize.destructive`.

It is a name and not a pattern list, because what it selects is no longer a
list. The tables that the classifier dispatches to (`SHELL_RULES`,
`SQL_STATEMENTS`, `PYTHON_CALLS`) are public in that module. A deployment can
read there what it gates, and this module does not export a second copy that
could drift from them.
"""


class Rule(WireModel):
    """When to ask about one tool."""

    preset: str = ""
    """A shipped classifier to run, by name. `destructive` is the only one.

    The result is the union with `when`. A profile can therefore take the parsed
    classifier *and* add its own site-specific patterns, and does not have to
    choose between them.
    """
    when: tuple[str, ...] = ()
    """Patterns that are matched against the arguments of the call. When
    `preset` is also empty, an empty value means *always ask*. This is the
    `true` form in the config of upstream, written as the absence of a condition
    and not as a second type."""
    allowed_decisions: tuple[ApprovalDecisionName, ...] = ()
    """The decisions that the front end should offer. Empty means all four.

    The type is the `Literal` of the seam itself and not bare strings. A typo in
    a profile then fails at config load. Without that, the typo would render a
    modal with no buttons and no explanation. The value is carried on the ask,
    so a deployment can withhold `edit` for a tool whose arguments must not be
    written by hand."""
    description: str = ""
    """Extra words for the prompt, when the tool's name is not enough."""

    def conditional(self) -> bool:
        """Whether this rule has a condition at all.

        A rule with neither a preset nor a pattern is the `true` form, which
        upstream writes as a bare rule: *always ask*. It is kept as the absence
        of a condition and not as a second type. The question is asked here so
        that `gate` does not have to derive "did anything have a chance to
        match" again from two fields.
        """
        return bool(self.preset or self.when)

    def found_in(self, arguments: JsonValue) -> list[str]:
        """What this rule found in the arguments of one call, in words that a
        person can read.

        The parser of the preset runs first, then the patterns of the
        deployment. Both read the *decoded string leaves* (`strings_in`) and not
        the arguments rendered to JSON. The rendered form put a literal `\\n`
        between a command and the line before it, and silently removed the
        anchor of every shipped pattern.
        """
        found = [str(one) for one in findings(arguments)] if self.preset == DESTRUCTIVE else []
        if self.when:
            found.extend(match for match in _matches(arguments, self.when) if match not in found)
        return found


class Config(WireModel):
    """Row config."""

    mode: ApprovalMode = "auto"
    declared: Rule = Field(default_factory=lambda: Rule(preset="destructive"))
    """The rule that a tool gets when it declares `is_irreversible` and no rule
    names it.

    **This is the curated destructive set, not "ask every time".** `bash` and a
    code cell declare because *any* command may reach outside the tree. A gate
    that asks on `ls` gets approved without being read. Under `manual`, a
    declaration does mean every call. The rule is configurable here and is not a
    constant. The extra patterns and the description of a deployment are then
    written one time, at the capability, and they survive a rename of the tool.
    A per-tool entry would silently stop applying."""
    interrupt_on: dict[str, Rule] = Field(default_factory=dict)
    """Per-tool overrides, by name. Empty means every tool gets what it declares."""


@lru_cache(maxsize=64)
def _compiled(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """The patterns of one rule, compiled one time.

    The cache is bounded by the number of configured rules, which is fixed for a
    deployment. The cache of `re` itself would work, but it is a shared
    512-entry LRU, and this function runs on every gated call.
    """
    return tuple(re.compile(one, re.IGNORECASE | re.DOTALL) for one in patterns)


def _matches(arguments: JsonValue, patterns: tuple[str, ...]) -> list[str]:
    """What the patterns of a deployment matched: the *text*, not the rule.

    Each decoded string leaf is scanned separately, and the arguments are not
    rendered to JSON. The escaping of the envelope turned a newline into two
    characters and stopped `\\b` from matching. Scanning the leaves also means
    that a pattern cannot match across a field boundary that does not exist.
    """
    found: list[str] = []
    for pattern in _compiled(patterns):
        for text in strings_in(arguments):
            hit = pattern.search(text)
            if hit is not None and hit.group(0) not in found:
                found.append(hit.group(0))
                break
    return found


def set_mode(session: Session, mode: ApprovalMode) -> None:
    """Record a posture change. The last one recorded is the one in force."""
    _LOG.append(session, "approval/mode", {"mode": mode})


def _mode(session: Session | None, config: Config) -> ApprovalMode:
    """The posture in force: the last one recorded, or else the default of the row.

    This is a fold, so the toggle survives a resume and the log says when the
    posture changed. The posture has its *own* event and is not a field on the
    `approval/policy` event of the seam. `permission-presets` also writes that
    event and knows nothing about a posture. If the posture were a field there,
    a change of preset would silently reset the posture. That is the general
    hazard of a last-write-wins fold with two writers, when one writer does not
    know that the field exists.
    """
    event = None if session is None else session.latest("approval/mode")
    recorded = None if event is None else event.data.get("mode")
    return recorded if recorded in ("manual", "auto", "yolo") else config.mode


@plugin("hitl", affects="environment", inject=[APPROVAL], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Ask a human before a configured call runs."""

    def rule_for(execution: ToolExecution) -> Rule | None:
        """The rule for this call: a configured override, or else what the tool
        declares.

        A profile writes `run_code` because that is the *reserved* transport
        name. The registry renames the transport in place to the name that a
        presentation row gives it, so a `run_code` entry is also honored for the
        presented name. The lookup is scope-aware, because an agent-shadowed
        registration is a different tool with the same name.
        """
        view = ctx.require(TOOLS).view(execution.scope)
        rule = config.interrupt_on.get(execution.name)
        if rule is None and execution.name == view.transport_name:
            rule = config.interrupt_on.get(RUN_CODE)
        if rule is not None:
            return rule
        definition = view.visible.get(execution.name)
        if definition is not None and definition.irreversible(execution.arguments):
            return config.declared
        return None

    async def gate(
        execution: ToolExecution,
        next_: Next[PreToolDecision],
    ) -> PreToolDecision:
        rule = rule_for(execution)
        if rule is None:
            return await next_(execution)
        mode = _mode(execution.session, config)
        if mode == "yolo":
            return await next_(execution)
        conditional = rule.conditional()
        matched = rule.found_in(execution.arguments) if conditional else []
        if mode == "auto" and conditional and not matched:
            # The rule has a condition, and the posture trusts the condition.
            # Nothing matched, so this is the routine case that the condition
            # exists to let through. `manual` asks anyway, which is what
            # `manual` means.
            return await next_(execution)
        return Ask(
            reason=_reason(execution, rule, matched),
            allowed_decisions=rule.allowed_decisions,
        )

    ctx.on("tools/pre-execute", gate)


def _reason(execution: ToolExecution, rule: Rule, matched: list[str]) -> str:
    """What the user is told, and what the log records with the ask.

    The verdict of the classifier is part of the text. A prompt that says only
    "approve this?" makes the user guess what concerns the harness, and an
    auditor who reads `approval/asked` later has nothing to tune against. The
    text quotes the *matched text* and not the pattern that matched. The regex
    source of the classifier would show the user an implementation, where
    `rm -rf /tmp/x` shows the user their own call.
    """
    parts = [rule.description or f"{execution.name} needs approval"]
    if matched:
        parts.append(f"matched {', '.join(repr(one) for one in matched)}")
    return " — ".join(parts)
