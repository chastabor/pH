"""What a front end reads off a root it cannot reach into (P5-14).

`PHTuiApp` used to resolve nine seams out of `ctx` — the command registry, the
screen registry, the status fields, the model routes, the presets, the
credentials — because the harness was in its own process. Over a socket none of
that is reachable, and the alternative to projecting it is a UI that quietly does
less when it is remote, which is the split this whole plan exists to avoid.

**Nothing here spells a field.** Each seam describes its own wire form —
`StatusReading.to_wire()`, `CommandDefinition.schema()`, `ScreenDefinition.schema()`
— the way `ToolSchema` already did for tools (P7-11), so a field added to a seam
reaches a browser tab with no edit here. The alternative was a dict per item
written at this edge, whose failure is the quiet kind: add `danger: bool` to a
command and the terminal shows it, the browser does not, and nothing fails.

**Every function here is a fold, not a fact.** Nothing is stored, nothing is
appended, and a projection is computed from the root as it stands at the moment
it is asked for. That is what makes them safe to send repeatedly and safe to
recompute after a restart: a client that reconnects gets today's answer rather
than a cached one that was true when somebody last wrote it down.

**Absence is normal and is not an error.** A profile need not mount `commands`,
`tui_screens`, `tui_status` or `credentials`, and a projection of a seam that is
not there is the empty list — the answer the in-process front end gave too,
where each of these is a `ctx.get(...)` that may return `None`. A daemon that
refused instead would make "this deployment has no screens" indistinguishable
from "this deployment is broken".

`Root` and `Supervisor` are imported under `TYPE_CHECKING` and nowhere else:
`supervisor` imports *this* module, so naming their types at runtime would close
the cycle — and `from __future__ import annotations` makes every annotation below
a string the checker reads and the interpreter never evaluates. Both were `Any`
for want of that import, which is `ph.keys`' own arrangement and the reason it
works there: a projection that takes `Any` is a projection nothing checks, on the
one layer whose whole job is not to say less than the seam does.

@module ph_app.daemon.projections
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from ph.cordis import DEPLOYMENT
from ph.json import JsonValue
from ph.keys import (
    COMMANDS,
    CREDENTIALS,
    PERMISSION_PRESETS,
    SKILLS,
    TOOLS,
    TUI_SCREENS,
    TUI_STATUS,
)
from ph.llm.types import ToolSchema
from ph.seams.commands import CommandSchema
from ph.seams.permission_presets import PresetSchema
from ph.seams.skills import Skill
from ph.seams.tui_screens import ScreenSchema
from ph.seams.tui_status import StatusReading

from ..payloads import ConfigRow
from ..sessions import SessionSummary, session_summaries

if TYPE_CHECKING:
    from .supervisor import Root, Supervisor

__all__ = [
    "browse_of",
    "commands_of",
    "credentials_named",
    "credentials_of",
    "readings_of",
    "screens_of",
    "tools_of",
]

# **These return models, not their wire dicts (P8-08).** A projection is read by
# two callers — the method that answers `commands/list`, and the notice that
# announces a change — and each used to call `.to_wire()` itself, so the shape
# reaching a client depended on which door it came through. The notice models in
# `ph_app.payloads` declare `list[CommandSchema]`, so the dump happens once, in
# `to_wire()`, at the edge.


def readings_of(root: Root) -> list[StatusReading]:
    """The footer, as the status seam currently reads it.

    A reading is a fold of the log, so this is cheap and correct to recompute; it
    rides `session.status` for that reason rather than being polled on the TUI's
    30 Hz tick, which exists for the spinner and would otherwise ask this
    question thirty times a second to get the same answer.
    """
    registry = root.ctx.get(TUI_STATUS)
    if registry is None:
        return []
    return list(registry.readings(root.session))


def commands_of(root: Root) -> list[CommandSchema]:
    """Every slash command a person may run against this root.

    `run` is deliberately not projected: it is a callable, and the client's job
    is to *offer* the command and send the line back, not to run it. The daemon
    runs it, in the root's own context — which is also the only place it could
    work, since a command body reaches for seams that live there.
    """
    registry = root.ctx.get(COMMANDS)
    if registry is None:
        return []
    return [one.schema() for one in registry.list()]


def screens_of(root: Root) -> list[ScreenSchema]:
    """The screens this deployment contributes, without their bodies.

    `build(session)` stays in the client and runs against the session it
    rebuilt from its own snapshot — which is enough while the client *is* the
    TUI, as it is under textual-serve. A declarative screen body on the wire is
    P5-15's other half and is deferred to P7-07; saying so here is the point,
    because a projection that silently dropped `build` would look complete.
    """
    registry = root.ctx.get(TUI_SCREENS)
    if registry is None:
        return []
    return [one.schema() for one in registry.list()]


def tools_of(root: Root) -> list[ToolSchema]:
    """What the model may call here, as `--mode rpc` already answers it.

    `DEPLOYMENT` and not an agent's scope (P6-32): this says what the deployment
    offers, which is the question a front end is asking. An agent's narrowed view
    is that agent's business and is not what a footer or a palette shows.
    """
    tools = root.ctx.get(TOOLS)
    if tools is None:
        return []
    return list(tools.schemas(scope=DEPLOYMENT))


def presets_of(root: Root) -> list[PresetSchema]:
    """The permission postures, with the live one marked.

    Resolved by the seam per call, like every projection here —
    `PermissionPresetService.schemas` says why that matters.
    """
    presets = root.ctx.get(PERMISSION_PRESETS)
    if presets is None:
        return []
    return list(presets.schemas(root.session))


def skills_of(root: Root) -> list[Skill]:
    """What is installed here, for `tools_of`'s reason and against its scope.

    `DEPLOYMENT` again: a child narrowed at spawn sees less, and that narrowing
    is the child's business — a person reading a sidebar is asking what this
    deployment has, which is the same question the tool list answers one
    function up.

    The catalog and not the bodies. `SkillService.list` is what the prompt's own
    catalog renders from, so a panel built on this cannot drift from what the
    model was told exists; reading a body is `ctx.skills.body`, which is a
    request the model makes and not something a front end pages in.
    """
    skills = root.ctx.get(SKILLS)
    if skills is None:
        return []
    return list(skills.list(DEPLOYMENT))


CREDENTIAL_CONFIG_KEY = "apiKeyEnv"
"""What an adapter row calls the environment variable holding its key."""


def credentials_of(root: Root, supervisor: Supervisor) -> dict[str, bool]:
    """Every credential this deployment names, and whether it is held —
    **never the values.**

    Held-ness rather than the secret, and the shape is what enforces it: there is
    no field here a value could travel in, so a future edit cannot leak one by
    forgetting to strip it. The picker only ever needed the boolean.

    **The names are found here, not sent here.** This took a `names` list, which
    meant the client had to know them — so `daemon/config` shipped the *entire
    composed profile* to every attached front end at attach, and a modal module
    in the terminal re-mined it with a recursive walk. A projection is what that
    should have been from the start (P5-14): the daemon holds the profile and the
    credential seam, so it can answer the whole question, and the answer is a
    `{name: bool}` map with no configuration in it.

    Two arguments because it is two facts: the composed profile is the
    *supervisor's* — every root mounts the same one — while the store is the
    root's own. In **row order**, which is the order the picker shows and the
    reason this is a dict rather than a set.

    The service is resolved once for the whole batch. Asking per name walked the
    scope chain per name, which is what the in-process caller did by looping over
    a predicate.
    """
    names = credentials_named(supervisor.profile.dump())
    service = root.ctx.get(CREDENTIALS)
    if service is None:
        return dict.fromkeys(names, False)
    # Plus whatever was handed to this process and is named in no row: the login
    # screen takes free text, so a credential can be *held* without the profile
    # having heard of it — and dropping it here would make it disappear from the
    # picker the moment somebody set it.
    names += [name for name in service.provided() if name not in names]
    return {name: bool(service.has(service.reference(name))) for name in names}


def credentials_named(rows: Iterable[ConfigRow]) -> list[str]:
    """Every credential the composed configuration names, in row order.

    Walks a row's config rather than matching on plugin names: the key is
    declared, so an adapter pH has never heard of is still covered.
    """
    found: list[str] = []
    for row in rows:
        for name in _walk(row.get("config")):
            if name not in found:
                found.append(name)
    return found


def _walk(value: JsonValue) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == CREDENTIAL_CONFIG_KEY and isinstance(item, str) and item:
                yield item
            else:
                yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        # `(list, tuple)` and not `Sequence`, which a `str` also satisfies —
        # every character would recurse forever as a one-character string.
        for item in value:
            yield from _walk(item)


def browse_of(supervisor: Supervisor) -> list[SessionSummary]:
    """Every session a person could open, stored and live, folded here (P5-14).

    **One list from the one process that can see both halves.** The logs are on
    the daemon's disk under the daemon's `$PH_HOME`, and which roots are *mounted*
    is a fact only the supervisor holds — so a client that asked for them
    separately had to be handed a directory path and read the files itself. It
    does not any more: a front end on another machine, or one with no filesystem
    at all, gets the same rows.

    A live root's `status` is its own — `running`, `waiting`, `retrying` — because
    that is what a person is choosing on: joining a session parked on somebody
    else's approval modal is a different act from joining one that is working.
    A stored row keeps `stored`.

    A live root the disk has not seen yet — its log still in a write buffer —
    gets a row of its own, built from what the supervisor knows: the header's
    `cwd`, so the repo it belongs to is on the row even before the file exists.
    """
    directory = supervisor.sessions_directory()
    stored = session_summaries(directory) if directory is not None else []
    held = {root.id: root for root in supervisor.roots.values()}
    rows = [
        summary.model_copy(update={"state": held[summary.session_id].status})
        if summary.session_id in held
        else summary
        for summary in stored
    ]
    known = {summary.session_id for summary in stored}
    rows.extend(
        SessionSummary(
            session_id=root.id,
            modified=0.0,
            size=0,
            cwd=root.session.header.cwd or "",
            state=root.status,
        )
        for root_id, root in sorted(held.items())
        if root_id not in known
    )
    return list(rows)
