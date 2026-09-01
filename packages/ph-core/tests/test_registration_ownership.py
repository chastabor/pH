"""P6-12 — a registration made by a row is an effect of that row (I2).

Gate: *a row that registers through a seam loses that registration when the row
unmounts, **enumerated over every seam exposing a `scope=` parameter** rather
than a hand-written list, so a new seam cannot join unchecked.*

The defect this replaces was one line repeated twenty times. Every seam wrote
`scope or self.ctx`, which reads as a lifetime default and is not one: `self.ctx`
is the *seam's* context, so a registration made by a row became an effect of the
seam and outlived the row that made it. I2 — "registrations and acquired
resources are effects that unwind" — therefore held only where a caller
remembered `scope=`, which was 1 of 38 call sites in the tree.

**P0-02's gate passed anyway**, because it exercises `ctx.effect` and
`add_disposer` directly; the seam path, which is how nearly every real
registration is made, was never its subject. That is the same shape of
unfalsifiable gate this plan hit at P3-23 and P3-24, so the test here is built to
be falsifiable in the way those were not: the surface is **discovered by
introspection**, and a method that is neither exercised nor classified fails.

The behavioural assertion is deliberately generic — *the seam's own context does
not grow an effect* — rather than per-registry. It needs no knowledge of what
each registry stores, so it holds for a seam this module has never heard of, and
it is exactly the property that was wrong.

## What the binding costs

Recorded here rather than in `context.py`, because these are the numbers that
justify two shapes a reader would otherwise be tempted to tidy into one.

**`running` is a class, not a `@contextmanager`.** The generator form costs a
frame, a `StopIteration` and two `send`s: **709 ns** per entry against **316 ns**
for byte-identical branch logic. This stopped being a once-per-activation helper
at P6-26/P6-29 — it is entered once per tool call, once per slash command, once
per prompt row per `assemble`, and once per telemetry sink per record, at
twenty-one sites. In situ against the pre-row build on base+headless, a **tool
call is at parity: 26.67 µs against 26.74** — the cheaper form pays for the whole
pair mechanism.

**What it does not pay for is `assemble`**, and §5 rule 6 wants that said rather
than discovered in a profiler. Prompt assembly enters the binding once per
*contributing row*, so the cost scales in the thing plugins add: **55.06 µs
against 49.85** with the nine rows base+headless contributes, ~1.2 µs per row
beyond that. Once per turn against a model round-trip, so it is affordable — but
a deployment with forty prompt rows pays forty bindings.

**`_invoke` spells the binding inline rather than calling `running`**, one scale
down from the same argument: `emit` fires once per streamed chunk. Cross-process
A/B on a 2 000-chunk turn, headless with two listeners: **8.2 ms** before P6-25,
**14.3 ms** through a `@contextmanager`, **10.5 ms** inline.

An earlier version of that paragraph priced the inline form at 120 ns per
listener and reported the row as free. Both were wrong, and the correction is why
`_invoke` has a `None` branch at all: the 120 ns counted only the `set`/`reset`
pair (**93 ns** measured) and missed the `inspect.isawaitable` the same function
introduced — **172 ns**, run twice per listener, which made the awaitability
question cost more than the ownership it was serving.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import io
import pkgutil
import re
import textwrap
import token
import tokenize
import typing
from collections.abc import Callable, Iterator
from functools import cache
from itertools import accumulate
from typing import Any, get_args

import pytest

import ph
from ph.cordis import DEPLOYMENT, Context, events, plugin, running
from ph.cordis.context import maybe_await
from ph.cordis.events import DispatchMode

pytestmark = pytest.mark.anyio


def _modules() -> list[Any]:
    """Every public module in the `ph` tree, imported once.

    **A module that fails to import is recorded, not just skipped.** Every walk
    in this file is built on this list, so a module that drops out drops out of
    all of them *silently* — and each of those walks is a gate whose whole claim
    is that it cannot be passed by omission. Caught while trying to falsify the
    P6-32 field check: two attempts to plant a defect produced an import error
    instead, the module vanished, and the gate went green both times. A check
    that a broken module can defeat is one an author can defeat by accident.

    The skip itself stays, because an optional extra genuinely may not be
    installed — that is what `SKIPPED_IMPORTS` is for, and
    `test_no_module_is_skipped_for_a_bad_reason` decides which reasons are
    allowed.
    """
    if not _IMPORTED:
        for found in pkgutil.walk_packages(ph.__path__, prefix="ph."):
            if any(part.startswith("_") for part in found.name.split(".")[1:]):
                continue
            try:
                _IMPORTED.append(importlib.import_module(found.name))
            except Exception as error:  # pragma: no cover - an extra is not installed
                SKIPPED_IMPORTS.append((found.name, error))
    return _IMPORTED


SKIPPED_IMPORTS: list[tuple[str, BaseException]] = []
"""Modules `_modules` could not import, and why. See `_modules`."""


def test_no_module_is_skipped_for_a_bad_reason() -> None:
    """A walk that silently loses a module is a gate that can be passed by omission.

    Every enumeration here — `_scoped_methods`, `_row_bodies`, `_provider_fields`,
    `_declared_classes` — reads `_modules()`. A module that raises on import is
    absent from all of them and nothing says so, which is exactly how a planted
    defect can look like a pass.

    The one legitimate reason is an optional extra: `ph-core[otel]` is not always
    installed, and a `ModuleNotFoundError` naming something *outside* the `ph`
    tree is that case. Anything else — a `SyntaxError`, a `NameError` from an
    identifier someone forgot to import, a circular import — is a broken module,
    and it must fail here rather than quietly shrink the surface every other test
    in this file measures.
    """
    _modules()
    wrong = [
        f"{name}: {type(error).__name__}: {error}"
        for name, error in SKIPPED_IMPORTS
        if not (
            isinstance(error, ModuleNotFoundError) and not str(error.name or "").startswith("ph")
        )
    ]
    assert wrong == [], (
        "these modules were dropped from every walk in this file for a reason that "
        f"is not a missing optional extra: {wrong}"
    )


_IMPORTED: list[Any] = []


def _declared_classes() -> Iterator[Any]:
    """Every class *defined* in the `ph` tree, once each.

    The `cls.__module__ != module.__name__` test is what keeps every walk in this
    file sound — without it an imported name is attributed to whichever module
    re-exported it, and the same class is enumerated once per importer. It was
    written out at each of the three walks below, in three slightly different
    spellings, which is three chances for one to drift; it is written here.
    """
    for module in _modules():
        for cls in vars(module).values():
            if inspect.isclass(cls) and cls.__module__ == module.__name__:
                yield cls


def _declared_methods(cls: Any) -> Iterator[tuple[str, Any]]:
    """The callables a class defines itself, dunders excluded."""
    for name, member in vars(cls).items():
        if not name.startswith("__") and callable(member):
            yield name, member


def _scoped_methods() -> set[str]:
    """Every `Class.method` taking a keyword-only `scope=`, found by walking the code.

    Introspection rather than a list, which is the whole point: a registry with a
    `scope=` parameter joins this set the moment it is written, and fails
    `test_every_scoped_method_is_accounted_for` until somebody decides which kind
    it is. A hand-written list would have to be remembered, and the thing this
    row fixes is precisely a rule twenty modules had to remember.

    **The whole `ph` tree, not `ph.seams` plus two names.** The first version
    walked `ph.seams` and then appended `ph.tools.registry` and
    `ph.system_prompt.assembly` by hand — a hand-written list of modules inside
    a helper whose argument is that hand-written lists go stale, and one that a
    `scope=`-taking registry under `ph/commands/` or `ph/agent/` would have
    joined without joining anything.
    """
    names: set[str] = set()
    for name, parameter in _scope_parameters():
        if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue
        if parameter.default is inspect.Parameter.empty:
            # `scope` with no default: the caller must say, so there is no
            # default to get wrong. `FsService.screen` was the first, required
            # for P6-18's reason — its owner decides *reach*, not just teardown,
            # so a forgotten scope would widen policy rather than merely delay
            # cleanup.
            #
            # P6-32 turns that one seam's judgement into the rule: a policy
            # reader takes `scope: Boundary` with no default, so `ToolRuntime`'s
            # `get`/`names`/`schemas`, `SkillService`'s readers and the five on
            # `ctx.fs` leave this walk as they convert.
            #
            # **They do not leave unwatched, and an earlier version of this
            # comment said they lost nothing — which was true of the default
            # question and false of the other one.** Leaving drops a method out
            # of `NOT_A_LIFETIME` too, and that table is held by
            # `test_the_classification_is_a_check_and_not_a_promise` — the check
            # that caught a real misfiling on its first run. So that test asserts
            # the same property directly over `Boundary` parameters, and
            # `test_a_boundary_parameter_never_has_a_default` covers the default
            # half. Partitioned by introspection rather than named, so no list
            # needs an edit when a seam moves.
            continue
        names.add(name)
    return names


def _scope_parameters() -> Iterator[tuple[str, inspect.Parameter]]:
    """Every `("Class.method", parameter)` whose signature takes `scope`.

    The iteration half of `_scoped_methods`, split out when the P6-32 gate
    became its second consumer — this file's own `_declared_classes` docstring
    says what a walk written out twice costs, and this was becoming the fourth
    copy of the signature loop. Each caller keeps only its filter.
    """
    for cls in _declared_classes():
        for method_name, method in _declared_methods(cls):
            try:
                signature = inspect.signature(method)
            except (TypeError, ValueError):  # pragma: no cover - builtins
                continue
            parameter = signature.parameters.get("scope")
            if parameter is not None:
                yield f"{cls.__name__}.{method_name}", parameter


def test_a_boundary_parameter_never_has_a_default() -> None:
    """P6-32's durable half: "everything" cannot become the convenient answer again.

    `scope: Context | None = None` conflated *"I did not state a boundary"* with
    *"I mean the deployment"*, so every reader had to pick a default for the
    ambiguous case and the convenient one was `scope or self.ctx` — the mount,
    the widest boundary there is. That single default is one root cause wearing
    four faces (P6-12, P6-24, and twice in P6-31), each found and fixed as though
    it were local.

    `Boundary` is `Context | Deployment` and has no `None` member, so the type
    says which question is being asked, and this checks **both** shapes one
    arrives in: a reader's parameter, and a caller-built payload's field.

    What stops the regression is the missing *default*: a caller that states
    nothing fails mypy where it holds a typed reference, and raises
    `TypeError` where it reaches the seam through `ctx.<seam>` — which is
    `Any`, so mypy cannot see it. This asserts the property the two layers
    rest on, over whatever has been converted so far, so a seam joining the
    migration is covered the moment it does.
    """
    offenders = [
        name
        for name, parameter in _scope_parameters()
        if "Boundary" in str(parameter.annotation)
        and parameter.default is not inspect.Parameter.empty
    ]
    # Payload fields too. A caller-built request carries the boundary for two
    # seams — `ToolExecutionInput.scope` and `AssembleContext.scope` — and this
    # walk reads *parameters named `scope`*, so it could see neither. That is the
    # half of the surface where the regression is cheapest to make: a field
    # regains a default with one keystroke and no call site changes.
    for cls in _declared_classes():
        if not dataclasses.is_dataclass(cls):
            continue
        try:
            hints = typing.get_type_hints(cls)
        except Exception:  # pragma: no cover - an unresolvable forward ref
            hints = {}
        offenders += [
            f"{cls.__name__}.{entry.name}"
            for entry in dataclasses.fields(cls)
            if "Boundary" in str(hints.get(entry.name, entry.type))
            and (
                entry.default is not dataclasses.MISSING
                or entry.default_factory is not dataclasses.MISSING
            )
        ]
    assert offenders == [], (
        "a `Boundary` parameter with a default reintroduces the thing P6-32 deleted — "
        f"the caller that says nothing gets the widest answer: {sorted(offenders)}"
    )


NOT_A_LIFETIME: dict[str, str] = {
    # `scope=` here names *what to answer for* or *where to dispatch* — never
    # whose lifetime a thing belongs to. Redirecting these to the activating row
    # would change what a caller sees or who hears an event, which is B7's
    # subject and not this row's. Asserted, not just declared: the static check
    # below requires each of these to be free of `owner_for`.
    "CommandRegistry.dispatch": "the boundary the command body runs for",
    # Work run *in* a scope, forwarding to whatever owns the artifact.
    "ShellService.run": "the scope the command runs for",
    "SubprocessService.run": "the scope the child runs for",
    # cordis dispatch. Found only once the walker covered the whole `ph` tree
    # rather than `ph.seams` plus two hand-written names — six methods that a
    # narrower walk had made invisible.
    "Context.bail": "which scope's listeners to dispatch to",
    "Context._hooks": "which scope's listeners to dispatch to",
    "Context.emit": "which scope's listeners to dispatch to",
    "Context.parallel": "which scope's listeners to dispatch to",
    "Context.serial": "which scope's listeners to dispatch to",
    "Context.waterfall": "which scope's listeners to dispatch to",
}
"""Methods whose `scope=` is not a lifetime, and why each one is not."""


# ------------------------------------------------------- how to exercise each --


def _recipe(name: str, key: str) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    def keep(call: Callable[[Any], Any]) -> Callable[[Any], Any]:
        RECIPES[name] = (key, call)
        return call

    return keep


def _command(service: Any) -> Any:
    return service.register(_definition("p612"))


def _diagnostic(service: Any) -> Any:
    from ph.seams.diagnostics import Diagnostic

    return service.register(Diagnostic(id="p612", read=lambda: [("a", "b")]))


def _status(service: Any) -> Any:
    from ph.seams.tui_status import StatusField

    return service.register(StatusField(id="p612", read=lambda _s: "x"))


def _skill(service: Any) -> Any:
    from ph.seams.skills import Skill

    return service.register(Skill(name="p612", description="a skill"))


def _section(service: Any) -> Any:
    from ph.system_prompt.assembly import PromptSection

    return service.section(PromptSection(name="p612", text="text"))


def _variable(service: Any) -> Any:
    return service.variable("p612", lambda: "value")


def _prompt_tools(service: Any) -> Any:
    return service.tools(lambda _scope: [])


def _guard(service: Any) -> Any:
    return service.guard(lambda _execution: None)


def _tool(service: Any) -> Any:
    from ph.testing import simple_tool

    return service.register(simple_tool("p612_tool"))


def _restrict(service: Any) -> Any:
    from ph.seams._restriction import NameFilter

    return service.restrict(NameFilter(deny=("nothing",)))


def _approval(service: Any) -> Any:
    return service.register_answerer(lambda _request: None)


def _question(service: Any) -> Any:
    return service.register_answerer(lambda _question: None)


def _note(service: Any) -> Any:
    from ph.seams.compaction import CompactionNote

    return service.note(CompactionNote(name="p612", text=lambda _session: "note"))


RECIPES: dict[str, tuple[str, Callable[[Any], Any]]] = {
    "CommandRegistry.register": ("commands", _command),
    "DiagnosticsRegistry.register": ("diagnostics", _diagnostic),
    "TuiStatusRegistry.register": ("tui_status", _status),
    "SkillService.register": ("skills", _skill),
    "SystemPromptService.section": ("system_prompt", _section),
    "SystemPromptService.variable": ("system_prompt", _variable),
    "SystemPromptService.tools": ("system_prompt", _prompt_tools),
    "ToolRuntime.guard": ("tools", _guard),
    "ToolRuntime.register": ("tools", _tool),
    "ToolRuntime.restrict": ("tools", _restrict),
    "ApprovalService.register_answerer": ("approval", _approval),
    "UserQuestionService.register_answerer": ("user_questions", _question),
    "CompactionSeam.note": ("compaction", _note),
}
"""`Class.method` → (service key, a call that registers something).

One entry each, and deliberately the *least* a registration needs: this
module is about who owns the result, not about what any registry stores. The
service key is what makes the assertion generic — `service.ctx` is the seam's
own context, and the whole defect was registrations landing there.

A literal, like every other table-driven test in this suite and like the two
sets below it. An earlier version built this through a `@_recipe` decorator,
which is a dict insert wearing a function: twelve module-level names nothing
referenced, and an import-order dependency between the decorators running and
`parametrize` reading the result.
"""


# ------------------------------------------------------------- the property --


@pytest.mark.parametrize("name", sorted(RECIPES))
async def test_a_registration_is_an_effect_of_the_row_that_made_it(mount: Any, name: str) -> None:
    """The gate. A row registers, the row unmounts, the registration goes.

    Asserted **against the seam's own context**, not against the registry's
    contents: the defect was `scope or self.ctx` handing the seam ownership, so
    "the seam grew an effect" is the failure itself rather than a proxy for it.
    That also makes the check indifferent to what any registry stores, which is
    what lets one assertion cover a dict-keyed table, a list of contributions, a
    single slot, an event listener and an acquired artifact alike.
    """
    key, register = RECIPES[name]
    root = await mount()
    service = root.get(key)
    assert service is not None, f"{key} is not mounted by base + headless"
    seam_effects = len(service.ctx._effects)

    @plugin(f"p612-{key}", inject=[key])
    async def row(ctx: Context, _config: Any) -> None:
        register(ctx.get(key))

    fork = root.plugin(row)
    await root.reconcile()

    activation = fork.ctx
    assert activation is not None, "the row did not activate"
    assert len(activation._effects) >= 1, "the registration did not land on the row's scope"
    assert len(service.ctx._effects) == seam_effects, (
        f"{name} registered on the seam's context — it will outlive the row (I2)"
    )

    await fork.dispose()
    assert len(service.ctx._effects) == seam_effects


async def test_a_registration_outside_any_activation_still_lands_on_the_service(
    mount: Any,
) -> None:
    """Today's behaviour, kept — which is what makes the change strictly additive.

    A test standing a service up by hand, or a mode wiring one directly, is not
    a row and has no activation scope. `owner_for` falls through to the service,
    exactly as `scope or self.ctx` did, so nothing outside the plugin tree
    changes.
    """
    root = await mount()
    commands = root.commands
    before = len(commands.ctx._effects)

    commands.register(_definition("bare"))
    assert len(commands.ctx._effects) == before + 1
    assert Context.current_owner() is None, "no activation is in flight here"


async def test_an_explicit_scope_still_wins(mount: Any) -> None:
    """`scope=` keeps meaning "register on someone else's lifetime".

    The agent-shadowing case, which is the reason the parameter exists: a
    registration made for one agent must unwind with that agent and not with the
    row that made it. It is now the *only* thing that overrides the default,
    rather than the thing a caller had to remember to get the ordinary case
    right.
    """
    root = await mount()
    commands = root.commands
    agent = root.scope("agent")

    @plugin("p612-scoped", inject=["commands"])
    async def row(ctx: Context, _config: Any) -> None:
        ctx.commands.register(_definition("scoped"), scope=agent)

    fork = root.plugin(row)
    await root.reconcile()
    assert commands.get("scoped") is not None

    # The row goes; the registration stays, because it was never the row's.
    await fork.dispose()
    assert commands.get("scoped") is not None, "an explicit scope was overridden"

    await agent.dispose()
    assert commands.get("scoped") is None, "it did not unwind with the scope it was given"


# ------------------------------------------------------------ the enumeration --

NOT_EXERCISED: frozenset[str] = frozenset(
    {
        # Registrations whose owner goes through `owner_for` like the rest, but
        # which this module does not *drive* — each needs a provider, a running
        # kernel or a real subprocess to call at all, and the property under
        # test is about who owns the disposer rather than about what the call
        # does. The shapes are all covered above: a dict-keyed table
        # (`CommandRegistry.register`), a list of contributions
        # (`SystemPromptService.section`), a single slot (`CompactionSeam.note`),
        # an event listener (`ApprovalService.register_answerer`) and a layered
        # registry (`ToolRuntime.guard`).
        #
        # Listed rather than skipped silently: a reader sees the whole surface,
        # and a new seam has to be put on one of these lists to get past
        # `test_every_scoped_method_is_accounted_for`.
        "CodeRuntimeSeam.register",
        "CodeRuntimeSeam.register_sdk_renderer",
        "CompactionSeam.register",
        "FsService.rebase",
        "JobService.start",
        "SandboxSeam.register_provider",
        "SessionTelemetry.add_sink",
        "SkillService.restrict",
        "SubagentService.register_provider",
        "SubprocessService.spawn",
        "SystemPromptService.context",
        "ToolRuntime.present_as",
        "ToolRuntime.present_transport",
        "ToolRuntime.register_code_namespace",
        "ToolRuntime.register_transport",
        "TuiScreenRegistry.present_with",
        "TuiScreenRegistry.register",
        "WorkspaceSeam.acquire",
        "WorkspaceSeam.provision",
        "WorkspaceSeam.register_provider",
    }
)
"""Registrations covered by `owner_for` but not driven here. See the comment."""


def test_every_scoped_method_is_accounted_for() -> None:
    """The "cannot join unchecked" half, and the reason this is introspection.

    A new seam method taking `scope=` joins `_scoped_methods()` the moment it is
    written. It then fails here until somebody decides which kind it is: a
    registration whose owner is the activating row, a read whose `scope=` names
    what to answer *for*, or one that requires an explicit scope and so has no
    default to get wrong.

    That decision is the thing P6-12 found nobody making — twenty seams wrote
    `scope or self.ctx` because it was what the seam next door wrote, and the
    question of whose lifetime a registration belongs to was never actually put.
    """
    discovered = _scoped_methods()
    classified = set(RECIPES) | set(NOT_A_LIFETIME) | NOT_EXERCISED

    unclassified = discovered - classified
    assert unclassified == set(), (
        "these take scope= and nobody has said what it means — add each to "
        f"RECIPES, NOT_A_LIFETIME or NOT_EXERCISED: {sorted(unclassified)}"
    )
    stale = classified - discovered
    assert stale == set(), f"classified but no longer present: {sorted(stale)}"


#: The two spellings that resolve a lifetime owner. `running_for` is
#: `owner_for` and `layer_for` in one call (P6-29), for a registry that has to
#: *record* both because it will invoke the registered body later — so a method
#: using it resolves an owner just as much as one calling `owner_for` directly.
#: Listed rather than matched loosely: `owner_for` is a substring of nothing
#: else here, and a third spelling should have to be added on purpose.
OWNER_RESOLVERS = ("owner_for", "running_for")


def _resolves_an_owner(name: str) -> bool:
    return any(spelling in _source_of(name) for spelling in OWNER_RESOLVERS)


def _code_of(source: str) -> str:
    """One method's source with its prose removed — comments and strings.

    Every check in this module asks whether a method *does* something by looking
    for a token in its source, and this file's own subject is a codebase whose
    house style is long argumentative docstrings that name mechanisms constantly.
    Those two collide: `CommandRegistry.dispatch` explains in a comment why it no
    longer restates `owner_for`'s fallback, and the classification check read
    that as a call and accused the method of resolving a lifetime. A prose
    mention is the *opposite* of a call — it is usually there to say the method
    deliberately does not do the thing.

    Tokenising rather than regexing, because "is this token inside a string" is
    exactly what a tokeniser knows and a regex has to guess. Docstrings come out
    as `STRING` tokens and go with the comments; an unparseable fragment is
    returned as-is rather than dropped, since a false *positive* here is a
    misfiled name and a false negative is a hole.
    """
    dedented = textwrap.dedent(source)
    try:
        pieces = list(tokenize.generate_tokens(io.StringIO(dedented).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):  # pragma: no cover
        return source
    # Blanked **in place and to the same shape**, never filtered out. Joining the
    # surviving tokens instead turned `running(` into `running (` and silently
    # broke every check in this module that matches a call by its bracket. Every
    # character becomes a space except a newline, which stays one — so offsets,
    # line numbers and every adjacency outside the blanked span are preserved,
    # which is also what lets the spans be applied in any order.
    starts = list(accumulate((len(one) for one in dedented.splitlines(True)), initial=0))
    out = list(dedented)
    for piece in pieces:
        if piece.type not in (token.COMMENT, token.STRING):
            continue
        begin = starts[piece.start[0] - 1] + piece.start[1]
        finish = starts[piece.end[0] - 1] + piece.end[1]
        out[begin:finish] = (one if one == "\n" else " " for one in out[begin:finish])
    return "".join(out)


@cache
def _source_of(name: str) -> str:
    """The code of a method plus the same-class helpers it delegates to.

    Memoised: 115 calls over 65 distinct names in one run of this module, and the
    repeats predate the prose-blanking — `SystemPromptService.assemble` is asked
    for four times, `CommandRegistry.dispatch` three. Deterministic within a
    session, keyed by a plain string, over modules imported once. Worth ~20 ms of
    the module's 0.65 s, most of which is `_code_of` tokenising the same bodies
    again.

    Was `_owner_resolution`, which named one *use* of it — grep this for
    `owner_for` and you have answered "does this resolve an owner". P6-30 added
    two more uses (does this bind, does this claim a slot), so the name was
    describing the first caller rather than the function.

    Followed transitively, because that is how far the tree actually goes:
    `ToolRuntime.register` → `_register` → `_claim`, and only the last of the
    three answers the ownership question. A one-hop version reported the first
    two as never resolving an owner, which would have been a false accusation in
    the direction that matters — it is the *failures* of this test that get
    acted on.
    """
    class_name, method_name = name.split(".")
    for module in _modules():
        cls = getattr(module, class_name, None)
        if cls is None or not inspect.isclass(cls) or cls.__module__ != module.__name__:
            continue
        seen: set[str] = set()
        pending = [method_name]
        source = ""
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            member = vars(cls).get(current)
            if member is None or not callable(member):
                continue
            code = _code_of(inspect.getsource(member))
            source += code
            # The *code*, not the raw source: a docstring writing `self._helper(`
            # would otherwise pull an unrelated method in, which is the same
            # thing `_code_of` was written to stop one line up. Latent today —
            # nothing in the tree spells a `self._x(` call in prose — and there
            # is no reason to derive one thing two ways while it stays that way.
            pending.extend(re.findall(r"self\.(_\w+)\(", code))
        return source
    raise AssertionError(f"{name} was discovered but cannot be located")


def test_the_classification_is_a_check_and_not_a_promise() -> None:
    """Every classified method is held against what its code actually does.

    Without this, `NOT_EXERCISED` is a trust list — 20 names under the claim
    "these go through `owner_for` like the rest" that nothing verifies, and the
    cheapest way past the gate is to type a name into a set. That is the same
    "remembered rule" this row exists to delete, and the same unfalsifiable-gate
    shape the module docstring criticises at P3-23 and P3-24.

    So the claim is checked instead: a registration must *resolve an owner*, and
    a method whose `scope=` is a visibility or dispatch target must not. It found
    a real misfiling on its first run — `SkillService.restrict` sat in the
    not-a-lifetime set while the code routed it through `owner_for`, so the gate
    was asserting a prose claim the diff falsified one file over.
    """
    missing = [
        name for name in sorted(set(RECIPES) | NOT_EXERCISED) if not _resolves_an_owner(name)
    ]
    assert missing == [], (
        "these are classified as registrations but never resolve an owner — "
        f"they still default to the seam, which is the P6-12 defect: {missing}"
    )

    confused = [name for name in sorted(NOT_A_LIFETIME) if _resolves_an_owner(name)]
    assert confused == [], (
        "these are classified as visibility or dispatch targets but resolve a "
        f"lifetime owner — one of the two is wrong: {confused}"
    )

    # The same check for the readers that *left* this table (P6-32). Converting
    # `scope` to a `Boundary` with no default drops a method out of
    # `_scoped_methods` — that walk skips no-default parameters — and the five fs
    # readers, `SkillService`'s and `ToolRuntime`'s went with it. The comment
    # there says they "lose nothing by leaving", which is true of the *default*
    # question and false of this one: leaving took them out of the falsifiability
    # check too, and this check is the one that caught a real misfiling on its
    # first run (`SkillService.restrict` routing through `owner_for` while filed
    # not-a-lifetime). A `Boundary` names a boundary to answer *for*, never a
    # lifetime to own, so the same assertion holds and needs no table.
    widened = sorted(
        name
        for name, parameter in _scope_parameters()
        if "Boundary" in str(parameter.annotation) and _resolves_an_owner(name)
    )
    assert widened == [], (
        "these take a `Boundary` — a scope to answer for — but resolve a lifetime "
        f"owner, which is the other question entirely: {widened}"
    )


def test_the_three_classifications_do_not_overlap() -> None:
    """A name in two tables would be a decision made twice, differently.

    The accounting test unions them, so an overlap would hide behind the union
    and each table would claim a different thing about the same method.
    """
    for left, right in (
        ("RECIPES", "NOT_A_LIFETIME"),
        ("RECIPES", "NOT_EXERCISED"),
        ("NOT_A_LIFETIME", "NOT_EXERCISED"),
    ):
        tables = {
            "RECIPES": set(RECIPES),
            "NOT_A_LIFETIME": set(NOT_A_LIFETIME),
            "NOT_EXERCISED": set(NOT_EXERCISED),
        }
        overlap = tables[left] & tables[right]
        assert overlap == set(), f"{left} and {right} both claim {sorted(overlap)}"


# ------------------------- P6-25: a listener is an effect of who wrote it --
#
# P6-12 set `_ACTIVATING` around `apply`, which made the rule true for the
# synchronous extent of a mount and left two gaps either side of it. Both were
# reproduced before this was written: a listener dispatched from *another* row's
# `apply` had its registration stolen when that row unmounted, and a
# registration made *after* `apply` returned still landed on the seam and
# outlived its row entirely. Dispatch now binds the scope that registered the
# listener, so "who is running" has one answer across both.


PROBE = {mode: f"probe/{mode}" for mode in get_args(DispatchMode)}
for _mode, _event in PROBE.items():
    events.declare(_event, _mode, owner="probe", doc="a P6-25 probe")
"""One throwaway event per dispatch mode, declared once at import.

`EventRegistry.declare` is idempotent for the same name *and* mode and raises
only on a mode conflict, so the constraint is one event per mode rather than one
per test — the shape `test_cordis_dispatch.py` already uses. Keyed off
`DispatchMode` so a sixth member gets an event without anyone remembering.
"""


def _definition(name: str, run: Any = None) -> Any:
    """The least a `CommandDefinition` needs, for tests that only watch it come and go.

    `run` for the one test that needs the body to *do* something — a second
    inline `CommandDefinition` beside this would be the same four fields with
    one changed, which is the difference a reader then has to go and find.
    """
    from ph.seams.commands import CommandDefinition

    return CommandDefinition(name=name, summary="s", run=run or (lambda *a, **k: None))


async def test_a_listener_registers_on_its_own_scope_not_the_emitters(mount: Any) -> None:
    """Gap one: the dispatch that *steals*.

    Row B writes a listener; row A emits during its own `apply`; B's listener
    registers a command. Under P6-12 alone the command belonged to **A**,
    because `_ACTIVATING` said "A is being applied" for the whole dynamic extent
    of A's activation — so disposing A took B's registration with it. A narrower
    failure than the leak it replaced, and a stranger one.
    """
    event = PROBE["emit"]
    root = await mount()
    commands = root.commands

    @plugin("p625-b", inject=["commands"])
    async def row_b(ctx: Context, _config: Any) -> None:
        ctx.on(event, lambda: ctx.commands.register(_definition("from-b")))

    @plugin("p625-a", inject=["commands"])
    async def row_a(ctx: Context, _config: Any) -> None:
        ctx.emit(event)

    b = root.plugin(row_b)
    await root.reconcile()
    a = root.plugin(row_a)
    await root.reconcile()
    assert commands.get("from-b") is not None

    await a.dispose()
    assert commands.get("from-b") is not None, "the emitting row took the listener's registration"

    await b.dispose()
    assert commands.get("from-b") is None, "it did not unwind with the row that wrote it"


async def test_a_registration_made_after_apply_returns_still_belongs_to_its_row(
    mount: Any,
) -> None:
    """Gap two: the dispatch that *leaks*, and the bigger half.

    A `profile/mounted` listener, a turn hook — anything *dispatched* after
    `apply` has returned saw `current_owner() is None` and fell back to the seam,
    so the registration outlived its row exactly as it did before P6-12. That is
    why `register_when_composed` carried a hand-written `scope=ctx`: not because
    the call was unusual, but because it was the one place somebody had noticed.

    **A tool body and a command body joined in P6-26**, which binds each to the
    agent it runs for. What is still not covered is a provider claimed through
    `claim_slot` and called later: it has no agent, only the row that registered
    it, so binding it is `claim_*` retaining its owner rather than the invoker
    knowing a scope. Named here rather than only in the plan, because this module
    is where a reader comes to learn what the rule covers, and a gate that
    overstates its reach is worse than one that admits a gap.
    """
    event = PROBE["emit"]
    root = await mount()
    commands = root.commands

    @plugin("p625-late", inject=["commands"])
    async def late(ctx: Context, _config: Any) -> None:
        ctx.on(event, lambda: ctx.commands.register(_definition("deferred")))

    fork = root.plugin(late)
    await root.reconcile()
    root.emit(event)
    assert commands.get("deferred") is not None

    await fork.dispose()
    assert commands.get("deferred") is None, "a deferred registration outlived its row"


async def test_register_when_composed_needs_no_explicit_scope(mount: Any) -> None:
    """The gate's third clause, and the reason it is worth asserting.

    That helper's `scope=ctx` was load-bearing and is now redundant. Removing it
    is only safe if the ordinary call is correct — so this drives the real
    helper on the real `profile/mounted` event rather than trusting the removal.
    """
    from ph.tools.registry import register_when_composed

    root = await mount()
    tools = root.tools

    @plugin("p625-composed", inject=["tools"])
    async def row(ctx: Context, _config: Any) -> None:
        from ph.testing import simple_tool

        register_when_composed(ctx, lambda: simple_tool("p625_tool"))

    fork = root.plugin(row)
    await root.reconcile()
    await root.serial("profile/mounted")
    assert tools.get("p625_tool", scope=DEPLOYMENT) is not None

    await fork.dispose()
    assert tools.get("p625_tool", scope=DEPLOYMENT) is None, (
        "the tool outlived the row that built it"
    )


@pytest.mark.parametrize("mode", sorted(get_args(DispatchMode)))
@pytest.mark.parametrize("shape", ["sync", "async"])
async def test_every_dispatch_mode_runs_a_listener_as_its_own_scope(
    mount: Any, mode: str, shape: str
) -> None:
    """Every mode, and **both listener shapes** — which is the axis that mattered.

    Parametrised over `DispatchMode` itself rather than a copied list, so a sixth
    member becomes a sixth case with no edit here, and dispatched through
    `getattr(root, mode)` so a member with no dispatch loop raises rather than
    falling into an `else`.

    The `shape` axis is here because its absence hid a real defect for a whole
    review cycle. Every listener in the first version was **synchronous**, and
    `emit` bound around the *call* — which for an `async def` listener only
    builds the coroutine. The body ran later on a task that had copied the
    context after the binding was released, so it saw `None`, and both gaps this
    row exists to close survived for the most ordinary listener shape in the
    harness. A gate that only tests the easy shape is the gate that let it
    through.
    """
    event = PROBE[mode]
    root = await mount()
    seen: list[Context | None] = []

    @plugin(f"p625-{mode}-{shape}", inject=["commands"])
    async def row(ctx: Context, _config: Any) -> None:
        if shape == "sync":

            def listener(*args: Any) -> None:
                seen.append(Context.current_owner())

            ctx.on(event, listener)
        else:

            async def listener(*args: Any) -> None:
                seen.append(Context.current_owner())

            ctx.on(event, listener)

    fork = root.plugin(row)
    await root.reconcile()
    activation = fork.ctx

    extra = {"inner": lambda *a: None} if mode == "waterfall" else {}
    await maybe_await(getattr(root, mode)(event, **extra))
    # `emit` schedules an async listener rather than awaiting it.
    await root.drain()

    assert seen == [activation], f"{mode}/{shape} did not run the listener as its own scope"
    assert Context.current_owner() is None, f"{mode}/{shape} left the binding set"


def test_the_declared_modes_and_the_registry_agree() -> None:
    """`DispatchMode` and `events._MODES` are two spellings of one closed set.

    Nothing held them against each other, so a mode added to one and not the
    other would be accepted by `declare` and rejected by `check`, or the reverse.
    The parametrization above reads the `Literal`, which makes this the one
    remaining place the two could drift.
    """
    from ph.cordis.events import _MODES

    assert set(get_args(DispatchMode)) == set(_MODES)


# --- P6-26: a registry-invoked body runs as the agent it was invoked for -----
#
# "Who is running" covered a plugin's `apply` (P6-12) and a listener's dispatch
# (P6-25) — everything cordis invokes *as a listener*. It did not cover code a
# **registry** invokes: a tool's `execute`, a command's `run`. Those fell through
# to the seam for lifetime and, after P6-27 nested agents, to the *global* layer
# for visibility — so a body inside a contained child installed a tool the whole
# deployment could see, which was the last way to escape a ceiling P6-27 had just
# made structural.


async def test_a_tool_body_registers_inside_the_agent_it_runs_for(mount: Any) -> None:
    """The containment half, and why this row carries B7 rather than only I2.

    A tool body is ordinary Python that a row wrote and an agent invoked. Before
    this it ran as nobody: `layer_for(None)` resolved to the registry's own
    context, whose isolation is `None` — the global layer — so a registration
    made here was visible to every agent in the deployment, including the ones
    the caller had been narrowed away from.
    """
    from ph.testing import FAKE_OPTIONS, run_tool, simple_tool

    ctx = await mount()
    parent = ctx.agents.create(ctx.sessions.create("p626-parent"), FAKE_OPTIONS)
    child = ctx.agents.create(ctx.sessions.create("p626-child"), FAKE_OPTIONS, parent=parent)

    def smuggle(_args: Any, run: Any) -> str:
        run.scope.tools.register(simple_tool("p626_smuggled"))
        return "done"

    ctx.tools.register(simple_tool("p626_smuggler", execute=smuggle))
    await run_tool(ctx, "p626_smuggler", {}, agent=child)

    assert "p626_smuggled" in ctx.tools.view(child.ctx).visible, "the child kept what it made"
    assert "p626_smuggled" not in ctx.tools.view(parent.ctx).visible, "it escaped upward"
    assert "p626_smuggled" not in ctx.tools.view(ctx).visible, "it reached the deployment"


async def test_a_row_registering_at_mount_still_lands_globally(mount: Any) -> None:
    """The half that must not have moved, and the reason it did not have to.

    `layer_for` following "who is running" was held back on the grounds that
    moving visibility would change what an agent can see. The two cases are told
    apart by `isolation` for free: a row's activation scope is *not* isolated, so
    it inherits the mount's — `None` at the root — and a row's registration
    still lands on the global layer exactly as before. Only an *agent's* scope is
    its own isolation, and only then does the answer change.
    """
    from ph.testing import FAKE_OPTIONS, simple_tool

    ctx = await mount()
    agent = ctx.agents.create(ctx.sessions.create("p626-plain"), FAKE_OPTIONS)

    @plugin("p626-row", inject=["tools"])
    async def row(scope: Context, _config: Any) -> None:
        scope.tools.register(simple_tool("p626_global"))

    ctx.plugin(row)
    await ctx.reconcile()

    assert "p626_global" in ctx.tools.view(ctx).visible
    assert "p626_global" in ctx.tools.view(agent.ctx).visible, "a row's tool stopped being global"


async def test_a_command_body_runs_as_its_row_for_the_agent_that_typed_it(mount: Any) -> None:
    """The same gap one registry over, and both halves of the answer (P6-29).

    `CommandRegistry.dispatch` handed the body `CommandContext(ctx=self.ctx, …)` —
    the *registry's* context, which is the `scope or self.ctx` shape P6-12 named
    — so a command that registered anything did it globally and permanently.

    P6-26 bound the typing agent for both questions. That is right for
    visibility and wrong for lifetime: whose code a command body is does not
    depend on who typed the slash, exactly as `_invoke` binds a listener's own
    scope rather than the emitter's. So the owner is the row that registered the
    command and the layer is the agent — asserted separately here, because a
    single assertion cannot tell the two apart and that is how P6-26 shipped
    with one of them wrong.
    """
    from ph.testing import FAKE_OPTIONS

    ctx = await mount()
    agent = ctx.agents.create(ctx.sessions.create("p629-cmd"), FAKE_OPTIONS)
    seen: list[tuple[Context | None, Context | None]] = []

    @plugin("p629-row", inject=["commands"])
    async def row(scope: Context, _config: Any) -> None:
        scope.commands.register(
            _definition(
                "p629",
                run=lambda _arg, _run: seen.append(
                    (Context.current_owner(), Context.current_layer())
                ),
            )
        )

    fork = ctx.plugin(row)
    await ctx.reconcile()
    await ctx.commands.dispatch("/p629", agent=agent)

    assert seen == [(fork.ctx, agent.ctx)], (
        "the command body must run as the row that registered it, for the agent that typed it"
    )

    # And with no agent there is still an answer, where P6-26 bound nothing.
    seen.clear()
    await ctx.commands.dispatch("/p629")
    assert seen == [(fork.ctx, fork.ctx)], "a command dispatched without an agent ran as nobody"


async def test_a_tool_body_registers_as_its_row_and_for_its_agent(mount: Any) -> None:
    """P6-29's property, and the two ways one `Context` got it wrong.

    A tool body is the registering row's code, run for one agent. Those are two
    scopes on unrelated branches of the tree, and a single-valued binding has to
    pick one:

    * **the agent** is what P6-26 bound, and it makes a registration outlive the
      row whose code made it — I2 verbatim, and asserted here by unmounting the
      row while the agent lives;
    * **the row** alone would strand the agent's `_Layer` under a disposed key,
      because the disposer would hang off a scope the agent's teardown never
      reaches. Measured before the fix: the layer count went 2 → 2 across the
      agent's disposal where it goes 2 → 1 today.

    So neither is the lifetime. The registration is meaningful only while
    **both** are alive, and `ToolRuntime._claim` releases on whichever ends
    first. Both orders are driven, because an intersection that only works one
    way round is the failure this docstring exists to rule out.
    """
    from ph.testing import FAKE_OPTIONS, run_tool, simple_tool

    for ends_first in ("the row", "the agent"):
        ctx = await mount()
        agent = ctx.agents.create(ctx.sessions.create("p629-tool"), FAKE_OPTIONS)
        other = ctx.agents.create(ctx.sessions.create("p629-other"), FAKE_OPTIONS)

        def smuggle(_args: Any, run: Any) -> str:
            run.scope.tools.register(simple_tool("p629_made"))
            return "done"

        @plugin("p629-tool-row", inject=["tools"])
        async def row(scope: Context, _config: Any) -> None:
            scope.tools.register(simple_tool("p629_carrier", execute=smuggle))

        fork = ctx.plugin(row)
        await ctx.reconcile()
        await run_tool(ctx, "p629_carrier", {}, agent=agent)

        # B7: the layer is the agent it ran for, and nobody else.
        assert "p629_made" in ctx.tools.view(agent.ctx).visible, "the agent lost what it made"
        assert "p629_made" not in ctx.tools.view(other.ctx).visible, "it reached another agent"
        assert "p629_made" not in ctx.tools.view(ctx).visible, "it reached the deployment"

        layers = len(ctx.tools._layers)
        if ends_first == "the row":
            await fork.dispose()
            assert "p629_made" not in ctx.tools.view(agent.ctx).visible, (
                "a tool made by a tool body outlived the row that registered the tool (I2)"
            )
        else:
            await agent.ctx.dispose()
        assert len(ctx.tools._layers) == layers - 1, (
            f"{ends_first} ended and the agent's layer was stranded under a dead key"
        )


async def test_a_prompt_provider_runs_as_its_row_for_the_scope_being_assembled(
    mount: Any,
) -> None:
    """Two of the four bindings `assemble` enters, proved one at a time.

    This existed because `_row_bodies()` could not see either provider — they sat
    in an `Any`-typed bucket — so behaviour was the only thing holding them.
    Parameterising `_Registration[T]` closed that, and both are in `BOUND` now.
    The test does not become redundant, because what the table proves is weaker
    than it reads: `test_a_bound_body_names_an_invoker_that_binds` says so in its
    own docstring — `SystemPromptService.assemble` enters four bindings, so
    deleting one leaves the static check green. This is the per-body guarantee
    for two of those four, and it is the reason the disclaimer over there can be
    honest rather than an excuse.

    Both halves, as everywhere else: the owner is the row that contributed the
    provider, the layer is the scope being assembled.
    """
    from ph.testing import FAKE_OPTIONS

    ctx = await mount()
    agent = ctx.agents.create(ctx.sessions.create("p629-prompt"), FAKE_OPTIONS)
    seen: dict[str, tuple[Context | None, Context | None]] = {}

    @plugin("p629-prompt-row", inject=["system_prompt"])
    async def row(scope: Context, _config: Any) -> None:
        def variable() -> str:
            seen["variable"] = (Context.current_owner(), Context.current_layer())
            return "v"

        def tools(_target: Context) -> list[Any]:
            seen["tools"] = (Context.current_owner(), Context.current_layer())
            return []

        scope.system_prompt.variable("p629", variable)
        scope.system_prompt.tools(tools)

    fork = ctx.plugin(row)
    await ctx.reconcile()

    await ctx.system_prompt.assemble(agent.ctx)

    assert seen["variable"] == (fork.ctx, agent.ctx), "the variable provider ran as the wrong scope"
    assert seen["tools"] == (fork.ctx, agent.ctx), "the tools provider ran as the wrong scope"


async def test_a_refused_registration_does_not_reassign_the_survivor(mount: Any) -> None:
    """The pair is written only once the mutation it describes is accepted.

    `_Layer.by` is a dict parallel to `_Layer.tools`, which is the shape the five
    sibling registries rejected in favour of a `_Registered(value, by)` record.
    It survives in the tools registry for a structural reason — `_claim`'s
    `mutate`/`undo` closures are built before the pair is known — and this is the
    hazard that buys: written *before* `mutate`, a registration that `add` then
    **refused** for a duplicate name still overwrote the surviving tool's pair,
    so the tool that stayed ran as the row whose registration had just been
    rejected. Every body of it — `execute`, `render`, `presentation_meta`,
    `finalize_content` — and once that row unmounted, `owner_for` warned and
    anything they registered landed on the seam.

    Invisible until the next `_changed()` rebuilt the view from the corrupted
    cell, which is why it is asserted through a rebuild rather than through the
    read straight after the refusal — that one is served by a stale cache and
    passes either way.
    """
    from ph.testing import simple_tool

    ctx = await mount()
    first, second = ctx.scope("row-a"), ctx.scope("row-b")

    with running(first, ctx):
        ctx.tools.register(simple_tool("p629_dup"))
    with pytest.raises(ValueError), running(second, ctx):
        ctx.tools.register(simple_tool("p629_dup"))

    # Force the rebuild the stale view was hiding behind.
    with running(first, ctx):
        ctx.tools.register(simple_tool("p629_other"))

    assert ctx.tools.view(ctx).by["p629_dup"].owner is first, (
        "a refused registration reassigned the surviving tool to the rejected row"
    )


# --- P6-30: the other surface — a body a registry *invokes* -------------------
#
# `_scoped_methods()` above walks methods that take a keyword-only `scope=`,
# which is the right walk for P6-12's defect and blind to P6-26's. A body a
# registry invokes is not a method with a `scope=` parameter — `ToolRuntime`'s
# `dispatch` and `CommandRegistry.dispatch` are invisible to it, as is every
# registry P6-29 reached. The proof that the gap mattered: P6-26 shipped with a
# tool's `render` running bound to an unrelated row's activation scope, and every
# test in this module was green.
#
# So there is a second walk, over the *values* rather than the methods: every
# callable a row hands to a registry, and every single-slot provider. Each must
# be classified, and a `BOUND` entry must name an invoker whose source actually
# binds — the same source-text falsifiability the `owner_for` check above uses,
# which is what keeps these tables from becoming a second thing to remember.


def _row_bodies() -> set[str]:
    """Every `Class.field` on a dataclass whose *resolved* type is a callable.

    Resolved through `get_type_hints` rather than read off `field.type`, because
    the annotation a reader sees is often an alias: `PromptSection.text` is
    `PromptText`, which is `str | Callable[...]`, and a string match on the raw
    annotation missed it along with `PromptContext.text` — the two bodies in the
    registry whose conflated owner and layer are the reason P6-29 exists.

    **What it can see is anything reachable by the word `Callable`**, at any
    depth of an annotation: a field that *is* one (`ToolDefinition.execute`), a
    field whose type is a dataclass that has one (`_Registered.definition`, since
    `CommandDefinition.run` is enumerated in its own right), and a container
    parameterised by one (`list[_Registration[ToolsProvider]]`). That last case
    is why `SystemPromptService._tools` and `._variables` are in the tables:
    those two contributions have no class of their own — a bare callable and a
    tuple — so the bucket was the only place their type could be named, and
    `_Registration` had been `Any` until it took a parameter. `_sections` and
    `_contexts` stay invisible *as buckets* and need not be visible: their bodies
    are found directly, as `PromptSection.text` and `PromptContext.text`.

    **What it cannot see is a body behind a nominal type**, and that is not a
    gap in the annotation but in this walk's premise. A provider is an object
    satisfying a Protocol, so `AdapterHandle.adapter: LlmAdapter` names its type
    exactly and mentions no callable anywhere. An earlier version of this
    paragraph claimed the opposite — that a container hides only what it is
    parameterised *by*, and that `Any` is the one parameterisation naming nothing
    — which was a tidy rule and false: two live registries were unbound and
    invisible to every walk here at the moment it was written. `_provider_fields`
    below is the answer, and it discriminates on the Protocol rather than on how
    the registration happened to be spelled.
    """
    names: set[str] = set()
    for cls in _declared_classes():
        if not dataclasses.is_dataclass(cls):
            continue
        try:
            hints = typing.get_type_hints(cls)
        except Exception:  # pragma: no cover - an unresolvable forward ref
            hints = {}
        for entry in dataclasses.fields(cls):
            if "Callable" in str(hints.get(entry.name, entry.type)):
                names.add(f"{cls.__name__}.{entry.name}")
    return names


def _provider_fields() -> set[str]:
    """Every `Class.field` whose type is a **Protocol** the tree declares.

    The other half of the surface, and the half `_row_bodies` is structurally
    blind to: a provider is an *object satisfying a Protocol* — `LlmAdapter`,
    `SubagentProvider`, `CompactionEngine` — so the annotation names a class and
    never the word `Callable`, however precisely it is written.

    **Was a source match on `claim_slot(`, and that was one token too narrow.**
    It described how five providers happened to be *registered* rather than what
    they are, so it saw exactly the seams using the at-most-one helper and missed
    the two that do not: `SubagentService` claims named providers through
    `claim_key` because "run a child" has genuinely different answers in one
    deployment, and `LlmRuntime.register_adapter` builds its handle by hand and
    takes no `scope=` at all. Both invoked a row's body unbound, and both were
    invisible to every walk in this module — which is how they survived P6-29.
    The type is the honest discriminator because it is the thing that is true of
    a provider no matter how it got there.
    """
    names: set[str] = set()
    for cls in _declared_classes():
        if not dataclasses.is_dataclass(cls):
            continue
        try:
            hints = typing.get_type_hints(cls)
        except Exception:  # pragma: no cover - an unresolvable forward ref
            hints = {}
        for entry in dataclasses.fields(cls):
            hint = hints.get(entry.name)
            for part in typing.get_args(hint) or (hint,):
                if inspect.isclass(part) and getattr(part, "_is_protocol", False):
                    names.add(f"{cls.__name__}.{entry.name}")
    return names


BOUND: dict[str, str] = {
    # `Class.field` → the method that invokes it, whose source must bind.
    #
    # cordis itself, where the owner and the layer coincide and one scope says
    # both — the case P6-12 and P6-25 were built for.
    "Hook.callback": "_invoke",
    "PluginSpec.apply": "Context.reconcile",
    "_Dependent.activate": "Context.reconcile",
    # The tools registry. All four of the definition's pipeline bodies, not one:
    # P6-26 bound `execute` alone, and `render` and `presentation_meta` — the
    # next two statements in the same closure — ran bound to whichever
    # `tools/execute` wrapper had called the inner, measured on the headless
    # profile as `plugin(session-checkpoint-policy)`.
    "ToolDefinition.execute": "ToolRuntime.dispatch",
    "ToolDefinition.finalize_content": "ToolRuntime.finish",
    "ToolOutput.presentation_meta": "ToolRuntime.dispatch",
    "ToolOutput.render": "ToolRuntime.dispatch",
    "ToolDefinition.is_concurrency_safe": "ToolRuntime.execution_mode",
    # The five registries P6-29 reached, once the binding could hold a pair.
    "CommandDefinition.run": "CommandRegistry.dispatch",
    "CompactionNote.text": "CompactionSeam.notes",
    "Diagnostic.read": "DiagnosticsRegistry.report",
    "PromptContext.text": "SystemPromptService.assemble",
    "PromptSection.text": "SystemPromptService.assemble",
    # The two buckets whose element type *is* the body, visible since
    # `_Registration` grew its parameter — before that they were `Any` and this
    # walk could not name them at all.
    "SystemPromptService._tools": "SystemPromptService.assemble",
    "SystemPromptService._variables": "SystemPromptService.assemble",
    "StatusField.read": "TuiStatusRegistry.readings",
    "_Sink.export": "SessionTelemetry.record",
    # The single-slot providers, in the same table as the bodies rather than a
    # second beside it: it is the same claim ("this runs inside a binding")
    # checked the same way, and the two walks that find them differ only in
    # where they look. Two dicts meant the parametrized check below had to
    # prefix one set and strip the prefix back off — machinery that existed
    # purely because there were two. The names cannot collide: `FsService.rebase`
    # is the method, `FsService._rebase` the field it fills.
    #
    # P6-29 dissolved the objection that had kept all five unbound — "a provider
    # has no agent" was only ever about the *layer* half, and the owner half
    # needs no agent at all.
    "AdapterHandle.adapter": "LlmRuntime.stream",
    "CodeRuntimeSeam.provider": "CodeRuntimeSeam.run",
    "CompactionSeam.engine": "CompactionSeam.compact_if_needed",
    "SandboxSeam.provider": "SandboxSeam.confine",
    "WorkspaceSeam.provider": "WorkspaceSeam.acquire",
    "_Registered.provider": "SubagentService.start",
    # The one provider slot whose target is already in hand — `ph.seams.fs` has
    # `_scope_of` of its own — so a rebase resolver runs for the agent whose path
    # is being resolved, where the other four take the registration's layer.
    "FsService._rebase": "FsService.root_for",
}
"""Row bodies a registry invokes inside a binding, and where that binding is."""


UNBOUND: dict[str, str] = {
    # --- not a registry-held body at all -------------------------------------
    "AgentRegistry.driver_factory": "builds the agent; runs before its scope exists",
    "FakeAdapter.respond": "a test double's canned reply",
    "StubCodeRuntime.programs": "a test double's canned programs",
    "InboxNotifications.claimed": "a callback back into the caller that supplied it",
    "InboxNotifications.discarded": "a callback back into the caller that supplied it",
    "InboxNotifications.inserted": "a callback back into the caller that supplied it",
    "_Entry.unobserve": "teardown; runs as its scope unwinds",
    # --- teardown ------------------------------------------------------------
    # A disposer runs *while* a scope is being unwound. Binding one would offer
    # a lifetime to register on at the moment that lifetime is ending, which is
    # the opposite of what I2 wants; `add_disposer` refuses an inactive scope
    # for the same reason.
    "Job.release": "teardown; runs as its scope unwinds",
    "SubagentRun.dispose": "teardown; runs as its scope unwinds",
    "Workspace.release": "teardown; runs as its scope unwinds",
    "_Effect.dispose": "teardown; runs as its scope unwinds",
    "_Held.dispose": "teardown; runs as its scope unwinds",
    # --- policy and presentation, called for an answer rather than for effect --
    # These are asked a question and expected to return one. None of them has a
    # reason to register, and two of them run outside any pipeline at all.
    "ToolDefinition.present_call": "TUI presentation, outside the pipeline",
    "ToolDefinition.present_result": "TUI presentation, outside the pipeline",
    "TransportPresentation.present_call": "TUI presentation, outside the pipeline",
    "TransportPresentation.present_result": "TUI presentation, outside the pipeline",
    "ScreenDefinition.build": "TUI presentation, outside the pipeline",
    "_FrontEnd.drawn": "TUI presentation, outside the pipeline",
    "_FrontEnd.present": "TUI presentation, outside the pipeline",
    "_Layer.guards": "a monotonic policy answer, read on the deny path",
    "_Screen.decide": "a policy answer; its own owner is what fs filters on",
    # --- factories and transports --------------------------------------------
    "_Layer.code_namespaces": "a factory, invoked to build bindings for a run",
    "_View.code_namespaces": "the same factories, resolved",
    "CodeBinding.dispatch": "re-enters ToolRuntime.execute, which binds (C1)",
    "SubagentRun.result": "an accessor for the child's outcome, not the child's code",
    # --- handed out rather than invoked --------------------------------------
    # The seam returns the renderer and its caller calls it, so there is no
    # invoke site here to bind. Binding it would mean `sdk_renderer()` returning
    # a wrapper instead of the row's own callable, which changes what a caller
    # holds in order to fix where it runs — a different shape from every entry
    # above, and one to decide on rather than to slip in.
    "CodeRuntimeSeam._sdk_renderers": "handed to the caller; this seam never invokes it",
}
"""Callables the walk finds that are *not* bound, and why each one is not."""


def _invoker_source(name: str) -> str:
    """The source of a named invoker, whether it is a method or a function."""
    if "." in name:
        return _source_of(name)
    for module in _modules():
        found = vars(module).get(name)
        if callable(found) and getattr(found, "__module__", "") == module.__name__:
            return _code_of(inspect.getsource(found))
    raise AssertionError(f"invoker {name} cannot be located")


@pytest.mark.parametrize(
    ("surface", "discovered"),
    [("row body", _row_bodies()), ("provider field", _provider_fields())],
    ids=["row-bodies", "provider-slots"],
)
def test_every_invoked_body_is_accounted_for(surface: str, discovered: set[str]) -> None:
    """A new callable on a registry's value fails until somebody decides.

    The point of the walks. `_scoped_methods` above cannot see this surface at
    all — a body is a *field*, not a method with a `scope=` — so before this the
    only thing standing between a new `ToolDefinition` callback and running as
    an unrelated row was that someone would think to check.

    Parametrized over the two walks rather than written twice: they share the
    tables, and only the noun naming what went unclassified differs.
    """
    unclassified = sorted(discovered - set(BOUND) - set(UNBOUND))
    assert unclassified == [], (
        f"each of these is a {surface} and nothing says whether it runs inside a "
        f"binding: add it to BOUND or to UNBOUND: {unclassified}"
    )


def test_no_classification_outlives_what_it_classifies() -> None:
    """The other direction, asked once against both walks.

    Per-walk it cannot be asked at all — a provider slot is absent from
    `_row_bodies()` by construction, so each walk alone would call every one of
    the other's entries stale. Together they are the whole surface the tables
    are allowed to describe.
    """
    stale = sorted((set(BOUND) | set(UNBOUND)) - _row_bodies() - _provider_fields())
    assert stale == [], f"these are classified but no longer exist: {stale}"


@pytest.mark.parametrize("name", sorted(BOUND))
def test_a_bound_body_names_an_invoker_that_binds(name: str) -> None:
    """`BOUND` is checked against the code, not trusted.

    Without this the table is a promise: names under the claim "these run inside
    a binding" that nothing verifies, and the cheapest way past the accounting
    test above is to type one into the bound half. That is the same
    unfalsifiable-gate shape this module's docstring criticises at P3-23 and
    P3-24, and the same reason `test_the_classification_is_a_check_and_not_a_promise`
    exists one surface over.

    **It proves the invoker binds, not that it binds *this* body**, and that
    bounds what the test is worth. `_invoker_source` concatenates a method with
    the same-class helpers it calls, so a method holding several bindings still
    reads as bound when one is deleted — `SystemPromptService.assemble` enters
    four, and losing one leaves this green. Deleting the *only* binding in a
    method does turn it red, which is the case that catches a whole invoker
    regressing: `ToolRuntime.finish` → `ToolDefinition.finalize_content`, and
    `SandboxSeam.confine` → its provider, both verified. The per-body guarantee
    is the behavioural tests' job, and every binding here has one.
    """
    invoker = BOUND[name]
    source = _invoker_source(invoker)
    assert "running(" in source or "_ACTIVATING.set" in source, (
        f"{name} is classified as bound, but its invoker {invoker} never enters a binding"
    )


def test_the_body_classifications_do_not_overlap() -> None:
    """A name in both tables would be a decision made twice, differently."""
    both = sorted(set(BOUND) & set(UNBOUND))
    assert both == [], f"classified as both bound and unbound: {both}"


async def test_a_command_body_is_told_the_boundary_the_caller_stated(mount: Any) -> None:
    """P6-24's commands half: the boundary reaches the body, not just the binding.

    `dispatch` gained `scope=` and spent it on the ambient binding, and stopped
    there — so a body asking a *scoped* question still had only `agent` and
    re-derived one, a frame below a caller that had just stated it.
    `ph.commands.revert` is the live case: it reads the tool table to decide what
    a revert covers, and its own docstring notes that getting that wrong "ran in
    the *unsafe* direction".

    Asserted through `CommandContext.scope`, because that is the value a body
    actually holds. The agent's own scope is deliberately different from the
    stated one here — that divergence is the whole subject, and the fallback
    would pass if the two were the same.
    """
    from ph.testing import FAKE_OPTIONS

    ctx = await mount()
    agent = ctx.agents.create(ctx.sessions.create("p624-cmd"), FAKE_OPTIONS)
    stated = ctx.scope("the-stated-boundary")
    seen: list[Context] = []

    ctx.commands.register(
        _definition("p624", run=lambda _arg, invocation: seen.append(invocation.scope))
    )
    await ctx.commands.dispatch("/p624", scope=stated, agent=agent)
    assert seen == [stated], "the body was handed the agent's scope, not the stated one"

    seen.clear()
    await ctx.commands.dispatch("/p624", agent=agent)
    assert seen == [agent.ctx], "with no stated scope the agent is still the fallback"
