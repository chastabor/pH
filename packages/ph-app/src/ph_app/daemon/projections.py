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

@module ph_app.daemon.projections
"""

from __future__ import annotations

from typing import Any

from ph.cordis import DEPLOYMENT

from ..sessions import SessionSummary, session_summaries

__all__ = [
    "browse_of",
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
    return [one.to_wire() for one in registry.readings(root.session)]


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
    return [one.schema().to_wire() for one in registry.list()]


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
    return [one.schema().to_wire() for one in registry.list()]


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


def browse_of(supervisor: Any) -> list[dict[str, Any]]:
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
    return [row.to_wire() for row in rows]
