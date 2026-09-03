"""What a front end reads off a root it cannot reach into (P5-14).

`PHTuiApp` used to resolve nine seams out of `ctx` — the command registry, the
screen registry, the status fields, the model routes, the presets, the
credentials — because the harness was in its own process. Over a socket none of
that is reachable, and the alternative to projecting it is a UI that quietly does
less when it is remote, which is the split this whole plan exists to avoid.

**Every function here is a fold, not a fact.** Nothing is stored, nothing is
appended, and a projection is computed from the root as it stands at the moment
it is asked for. That is what makes them safe to send repeatedly and safe to
recompute after a restart: a client that reconnects gets today's answer rather
than a cached one that was true when somebody last wrote it down.

**Absence is normal and is not an error.** A profile need not mount `commands`,
`tui_screens`, `tui_status` or `credentials`, and a projection of a seam that is
not there is the empty list — the same answer `HarnessSession` gives in process,
where each of these is a `ctx.get(...)` that may return `None`. A daemon that
refused instead would make "this deployment has no screens" indistinguishable
from "this deployment is broken".

@module ph_app.daemon.projections
"""

from __future__ import annotations

from typing import Any

from ph.cordis import DEPLOYMENT

__all__ = [
    "commands_of",
    "credentials_of",
    "readings_of",
    "screens_of",
    "tools_of",
]


def readings_of(root: Any) -> list[dict[str, Any]]:
    """The footer, as the status seam currently reads it.

    A reading is a fold of the log, so this is cheap and correct to recompute; it
    rides `session.status` for that reason rather than being polled on the TUI's
    30 Hz tick, which exists for the spinner and would otherwise ask this
    question thirty times a second to get the same answer.
    """
    registry = root.ctx.get("tui_status")
    if registry is None:
        return []
    readings = registry.readings(root.session)
    return [{"text": one.text, "level": one.level} for one in readings]


def commands_of(root: Any) -> list[dict[str, Any]]:
    """Every slash command a person may run against this root.

    `run` is deliberately not projected: it is a callable, and the client's job
    is to *offer* the command and send the line back, not to run it. The daemon
    runs it, in the root's own context — which is also the only place it could
    work, since a command body reaches for seams that live there.
    """
    registry = root.ctx.get("commands")
    if registry is None:
        return []
    return [
        {"name": one.name, "summary": one.summary, "argumentHint": one.argument_hint}
        for one in registry.list()
    ]


def screens_of(root: Any) -> list[dict[str, Any]]:
    """The screens this deployment contributes, without their bodies.

    `build(session)` stays in the client and runs against the session it
    rebuilt from its own snapshot — which is enough while the client *is* the
    TUI, as it is under textual-serve. A declarative screen body on the wire is
    P5-15's other half and is deferred to P7-07; saying so here is the point,
    because a projection that silently dropped `build` would look complete.
    """
    registry = root.ctx.get("tui_screens")
    if registry is None:
        return []
    return [
        {"id": one.id, "label": one.label, "order": one.order, "key": one.key}
        for one in registry.list()
    ]


def tools_of(root: Any) -> list[dict[str, Any]]:
    """What the model may call here, as `--mode rpc` already answers it.

    `DEPLOYMENT` and not an agent's scope (P6-32): this says what the deployment
    offers, which is the question a front end is asking. An agent's narrowed view
    is that agent's business and is not what a footer or a palette shows.
    """
    tools = root.ctx.get("tools")
    if tools is None:
        return []
    return [schema.to_wire() for schema in tools.schemas(scope=DEPLOYMENT)]


def credentials_of(root: Any, names: list[str]) -> dict[str, bool]:
    """Which of these credentials the harness already holds — **never the values.**

    Held-ness rather than the secret, and the shape is what enforces it: there is
    no field here a value could travel in, so a future edit cannot leak one by
    forgetting to strip it. The picker only ever needed the boolean.

    The service is resolved once for the whole batch. Asking per name walked the
    scope chain per name, which is what the in-process caller did by looping over
    a predicate.
    """
    service = root.ctx.get("credentials")
    if service is None:
        return dict.fromkeys(names, False)
    return {name: bool(service.has(service.reference(name))) for name in names}
