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
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections.abc import Callable
from typing import Any, get_args

import pytest

import ph
from ph.cordis import Context, events, plugin
from ph.cordis.context import maybe_await
from ph.cordis.events import DispatchMode

pytestmark = pytest.mark.anyio


def _modules() -> list[Any]:
    """Every public module in the `ph` tree, imported once."""
    if not _IMPORTED:
        for found in pkgutil.walk_packages(ph.__path__, prefix="ph."):
            if any(part.startswith("_") for part in found.name.split(".")[1:]):
                continue
            try:
                _IMPORTED.append(importlib.import_module(found.name))
            except Exception:  # pragma: no cover - an optional extra is not installed
                continue
    return _IMPORTED


_IMPORTED: list[Any] = []


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
    modules = _modules()
    names: set[str] = set()
    for module in modules:
        for cls in vars(module).values():
            if not inspect.isclass(cls) or cls.__module__ != module.__name__:
                continue
            for method_name, method in vars(cls).items():
                if method_name.startswith("__") or not callable(method):
                    continue
                try:
                    signature = inspect.signature(method)
                except (TypeError, ValueError):  # pragma: no cover - builtins
                    continue
                parameter = signature.parameters.get("scope")
                if parameter is None or parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
                    continue
                if parameter.default is inspect.Parameter.empty:
                    # `scope` with no default: the caller must say, so there is
                    # no default to get wrong. `FsService.screen` is the one, and
                    # it is required for P6-18's reason — its owner decides
                    # *reach*, not just teardown, so a forgotten scope would
                    # widen policy rather than merely delay cleanup. Partitioned
                    # by introspection rather than named, so the next seam to
                    # make the same call needs no edit here.
                    continue
                names.add(f"{cls.__name__}.{method_name}")
    return names


NOT_A_LIFETIME: dict[str, str] = {
    # `scope=` here names *what to answer for* or *where to dispatch* — never
    # whose lifetime a thing belongs to. Redirecting these to the activating row
    # would change what a caller sees or who hears an event, which is B7's
    # subject and not this row's. Asserted, not just declared: the static check
    # below requires each of these to be free of `owner_for`.
    "CompactionSeam.notes": "visibility — which notes this scope may read",
    "ToolRuntime.get": "a lookup in this scope's view",
    "ToolRuntime.names": "a lookup in this scope's view",
    "ToolRuntime.schemas": "a lookup in this scope's view",
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
    assert Context.current_scope() is None, "no activation is in flight here"


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
        "ToolRuntime.register",
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


def _owner_resolution(name: str) -> str:
    """The source of a method plus the same-class helpers it delegates to.

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
            body = inspect.getsource(member)
            source += body
            pending.extend(re.findall(r"self\.(_\w+)\(", body))
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
        name
        for name in sorted(set(RECIPES) | NOT_EXERCISED)
        if "owner_for" not in _owner_resolution(name)
    ]
    assert missing == [], (
        "these are classified as registrations but never resolve an owner — "
        f"they still default to the seam, which is the P6-12 defect: {missing}"
    )

    confused = [name for name in sorted(NOT_A_LIFETIME) if "owner_for" in _owner_resolution(name)]
    assert confused == [], (
        "these are classified as visibility or dispatch targets but resolve a "
        f"lifetime owner — one of the two is wrong: {confused}"
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


def _definition(name: str) -> Any:
    """The least a `CommandDefinition` needs, for tests that only watch it come and go."""
    from ph.seams.commands import CommandDefinition

    return CommandDefinition(name=name, summary="s", run=lambda *a, **k: None)


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
    `apply` has returned saw `current_scope() is None` and fell back to the seam,
    so the registration outlived its row exactly as it did before P6-12. That is
    why `register_when_composed` carried a hand-written `scope=ctx`: not because
    the call was unusual, but because it was the one place somebody had noticed.

    **A tool body is still not covered**, and an earlier draft of this docstring
    said it was. `definition.execute` is invoked by the tools registry, not
    dispatched as a listener, so nothing binds it — and the same holds for a
    command's `run` and a provider claimed through `claim_slot`. Those are P6-26.
    Naming them here rather than only in the plan, because this module is where a
    reader comes to learn what the rule covers, and a gate that overstates its
    reach is worse than one that admits a gap.
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
    assert tools.get("p625_tool") is not None

    await fork.dispose()
    assert tools.get("p625_tool") is None, "the tool outlived the row that built it"


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
                seen.append(Context.current_scope())

            ctx.on(event, listener)
        else:

            async def listener(*args: Any) -> None:
                seen.append(Context.current_scope())

            ctx.on(event, listener)

    fork = root.plugin(row)
    await root.reconcile()
    activation = fork.ctx

    extra = {"inner": lambda *a: None} if mode == "waterfall" else {}
    await maybe_await(getattr(root, mode)(event, **extra))
    # `emit` schedules an async listener rather than awaiting it.
    await root.drain()

    assert seen == [activation], f"{mode}/{shape} did not run the listener as its own scope"
    assert Context.current_scope() is None, f"{mode}/{shape} left the binding set"


def test_the_declared_modes_and_the_registry_agree() -> None:
    """`DispatchMode` and `events._MODES` are two spellings of one closed set.

    Nothing held them against each other, so a mode added to one and not the
    other would be accepted by `declare` and rejected by `check`, or the reverse.
    The parametrization above reads the `Literal`, which makes this the one
    remaining place the two could drift.
    """
    from ph.cordis.events import _MODES

    assert set(get_args(DispatchMode)) == set(_MODES)
