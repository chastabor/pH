"""`hitl` — a human between the model and the actions it cannot take back (P4-05, G6).

Deep Agents' `interrupt_on` config and its `manual | auto | yolo` posture, on
`tools/pre-execute`: when a rule matches, the listener returns `Ask` and the
registry routes it through `ctx.approval` (B3).

**The tool's declaration is the default; a configured rule is the override**
(P6-16). A tool declares `is_irreversible` and is gated under `Config.declared`;
a key in `interrupt_on` replaces that for one tool by name. Upstream has only the
name, and a name is a rendering choice — this bundle's own Code Mode transport is
renamed in place by a presentation row — so a gate keyed on it alone goes quietly
inert for the surface that most needs one. Everything after that already
exists — the seam records `approval/asked` and `approval/decided`, fails closed
on a missing answerer, and re-asks on resume because an `asked` with no `decided`
*is* the pending state.

**Four decisions, not two.** `approve` and `reject` were already reachable;
P4-05 adds `edit` (run it with these arguments instead) and `respond` (do not
run it — tell the model this). Both exist because stopping a turn to say "wrong
path" or "you don't need that, the answer is X" costs a round trip that answering
in place does not.

**The classifier parses; it does not pattern-match.** `preset: destructive` runs
`ph_stabilize.destructive`, which reads each string argument in its own dialect —
shell through `shlex`, SQL as statements, Python through `ast` — and judges the
*structure*. The regexes it replaces could not see a command line as a grammar,
and had the failure that invites: the arguments were scanned as rendered JSON, so
a real newline became the two characters `\\` and `n`, and every `\\b`-anchored
pattern stopped matching anything on a second line. A `when:` list is still
regex, as the deployment's own escape hatch, but it is matched against the
decoded strings rather than the envelope.

What was found goes into the ask's `reason` in the parser's words — *`rm -rf
build` (removes files)* — so a person is told what the harness saw and an auditor
can tune it. A gate whose reasoning is invisible is one nobody can tune.

**`run_code` is the case that motivates the shape.** A cell is not a tool name;
one `ipython` call may delete a tree, force-push, or open a socket. So the
classifier reads the program, follows a `subprocess.run(...)` back into the shell
reader, and asks about *that cell* rather than about the transport.

@module ph_stabilize.hitl
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Final, Literal

from pydantic import Field

from ph.cordis import Context, plugin
from ph.seams.approval import ApprovalDecisionName
from ph.session import Session
from ph.tools.definition import Ask, ToolExecution
from ph.tools.registry import RUN_CODE
from ph.wire import WireModel
from ph_stabilize.destructive import findings, strings_in

__all__ = [
    "DESTRUCTIVE",
    "ApprovalMode",
    "Config",
    "Rule",
    "apply",
    "set_mode",
]

ApprovalMode = Literal["manual", "auto", "yolo"]
"""deepagents-code's posture, and the one knob a person reaches for most.

`manual` asks about every configured call. `auto` asks only about the ones a
rule matched — the shipped middle, where a `when` predicate is what separates a
routine write from `rm -rf`. `yolo` asks about nothing, and is a thing a person
turns on deliberately for a session they are watching.
"""

DESTRUCTIVE: Final = "destructive"
"""The one shipped preset: the parsed classifier in `ph_stabilize.destructive`.

A name rather than a pattern list, because what it selects is no longer a list.
The tables it dispatches to — `SHELL_RULES`, `SQL_STATEMENTS`, `PYTHON_CALLS` —
are public there, so a deployment reads what it gates without this module
re-exporting a copy that could drift from it.
"""


class Rule(WireModel):
    """When to ask about one tool."""

    preset: str = ""
    """A shipped classifier to run, by name — `destructive` is the only one.

    Unioned with `when`, so a profile takes the parsed classifier *and* adds its
    own site-specific patterns rather than choosing between them.
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

    def conditional(self) -> bool:
        """Whether this rule has a condition at all.

        A rule with neither a preset nor a pattern is the `true` form upstream
        spells as a bare rule: *always ask*. Kept as the absence of a condition
        rather than a second type, and asked here so `gate` does not have to
        re-derive "did anything have a chance to match" from two fields.
        """
        return bool(self.preset or self.when)

    def found_in(self, arguments: Any) -> list[str]:
        """What this rule found in one call's arguments, in a person's words.

        The preset's parser first, then the deployment's own patterns. Both read
        the *decoded string leaves* — `strings_in` — rather than the arguments
        rendered to JSON, which is what put a literal `\\n` between a command and
        the line before it and silently unanchored every shipped pattern.
        """
        found = [str(one) for one in findings(arguments)] if self.preset == DESTRUCTIVE else []
        if self.when:
            found.extend(match for match in _matches(arguments, self.when) if match not in found)
        return found


class Config(WireModel):
    """Row config."""

    mode: ApprovalMode = "auto"
    declared: Rule = Field(default_factory=lambda: Rule(preset="destructive"))
    """What a tool earns by declaring `is_irreversible` with no rule naming it.

    **The curated destructive set, not "ask every time."** `bash` and a code cell
    declare because *any* command may reach past the tree, and asking on `ls` is
    how a gate gets rubber-stamped; under `manual` a declaration does mean every
    call. Configurable here — not a constant — so a deployment's extra patterns
    and description are written once at the capability and survive a tool being
    renamed, where a per-tool entry would silently stop applying."""
    interrupt_on: dict[str, Rule] = Field(default_factory=dict)
    """Per-tool overrides, by name. Empty means every tool gets what it declares."""


@lru_cache(maxsize=64)
def _compiled(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """One rule's patterns, compiled once.

    Bounded by the number of configured rules, which is fixed for a deployment.
    `re`'s own cache would serve, but it is a shared 512-entry LRU and this runs
    on every gated call.
    """
    return tuple(re.compile(one, re.IGNORECASE | re.DOTALL) for one in patterns)


def _matches(arguments: Any, patterns: tuple[str, ...]) -> list[str]:
    """What a deployment's own patterns matched — the *text*, not the rule.

    Each decoded string leaf is scanned separately rather than the arguments
    rendered to JSON: the envelope's escaping is what turned a newline into two
    characters and stopped `\\b` matching, and scanning leaves also means a
    pattern cannot match across a field boundary that does not exist.
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
        """This call's rule: a configured override, else what the tool declares.

        A profile writes `run_code` because that is the *reserved* transport name,
        but the registry renames the transport in place to whatever a presentation
        row calls it — so a `run_code` entry is also honoured for the presented
        name, and the lookup is scope-aware because an agent-shadowed registration
        is a different tool with the same name.
        """
        view = ctx.tools.view(execution.scope)
        rule = config.interrupt_on.get(execution.name)
        if rule is None and execution.name == view.transport_name:
            rule = config.interrupt_on.get(RUN_CODE)
        if rule is not None:
            return rule
        definition = view.visible.get(execution.name)
        if definition is not None and definition.irreversible(execution.arguments):
            return config.declared
        return None

    async def gate(execution: ToolExecution, next_: Any) -> Any:
        rule = rule_for(execution)
        if rule is None:
            return await next_(execution)
        mode = _mode(execution.session, config)
        if mode == "yolo":
            return await next_(execution)
        conditional = rule.conditional()
        matched = rule.found_in(execution.arguments) if conditional else []
        if mode == "auto" and conditional and not matched:
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
