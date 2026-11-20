"""What a front end reads from a root that it cannot reach into (P5-14).

`PHTuiApp` used to resolve nine seams out of `ctx`, because the harness was in
the process of the TUI itself: the command registry, the screen registry, the
status fields, the model routes, the presets, the credentials. Over a socket,
none of that is reachable. Without a projection, the UI would quietly do less
when it is remote, and this whole plan exists to avoid that split.

**Nothing here names a field.** Each seam describes its own wire form:
`StatusReading.to_wire()`, `CommandDefinition.schema()`,
`ScreenDefinition.schema()`. `ToolSchema` already did this for tools (P7-11). So
a field that is added to a seam reaches a browser tab with no edit here. The
alternative was a dict for each item, written at this edge, and it fails
quietly: add `danger: bool` to a command, and the terminal shows it, the browser
does not, and nothing fails.

**Every function here is a fold, not a fact.** Nothing is stored and nothing is
appended. A projection is computed from the root as it is at the moment of the
request. Projections are therefore safe to send repeatedly, and safe to compute
again after a restart. A client that reconnects gets the current answer, and not
a cached answer that was true when somebody last wrote it down.

**Absence is normal and is not an error.** A profile need not mount `commands`,
`tui_screens`, `tui_status` or `credentials`. The projection of a seam that is
absent is the empty list. The in-process front end gave the same answer, because
there each of these is a `ctx.get(...)` that may return `None`. If a daemon
refused, "this deployment has no screens" would look the same as "this
deployment is broken".

`Root` and `Supervisor` are imported under `TYPE_CHECKING` and nowhere else.
`supervisor` imports *this* module, so their types named at runtime would close
the cycle. `from __future__ import annotations` makes every annotation below a
string that the checker reads and the interpreter never evaluates. Before that
import, both were `Any`. `ph.keys` uses the same arrangement, and this is the
reason why it works there. A projection that takes `Any` is a projection that
nothing checks, on the one layer whose job is to say no less than the seam says.

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
    MODELS,
    MOUNT,
    PERMISSION_PRESETS,
    SKILLS,
    SUBAGENTS,
    TOOLS,
    TUI_SCREENS,
    TUI_STATUS,
)
from ph.llm.types import ToolSchema
from ph.seams.commands import CommandSchema
from ph.seams.models import ModelEntry
from ph.seams.permission_presets import PresetSchema
from ph.seams.skills import Skill
from ph.seams.subagents import ChildState
from ph.seams.tui_screens import ScreenSchema
from ph.seams.tui_status import StatusReading

from ..payloads import ChildRow, ConfigRow
from ..sessions import SessionSummary, session_summaries

if TYPE_CHECKING:
    from .supervisor import Root, Supervisor

__all__ = [
    "browse_of",
    "commands_of",
    "credentials_named",
    "credentials_of",
    "family_of",
    "family_rows",
    "models_of",
    "presets_of",
    "readings_of",
    "screens_of",
    "skills_of",
    "tools_of",
]

# **These return models, not their wire dicts (P8-08).** Two callers read a
# projection: the method that answers `commands/list`, and the notice that
# announces a change. Each used to call `.to_wire()` itself, so the shape that
# reached a client depended on which of the two paths it came through. The notice
# models in `ph_app.payloads` declare `list[CommandSchema]`, so the dump happens
# one time, in `to_wire()`, at the edge.


def readings_of(root: Root) -> list[StatusReading]:
    """The footer, as the status seam reads it now.

    A reading is a fold of the log, so it is cheap and correct to compute again.
    For that reason it is sent with `session.status` and is not polled on the
    30 Hz tick of the TUI. That tick exists for the spinner. A poll on it would
    ask this question thirty times a second and get the same answer.
    """
    registry = root.ctx.get(TUI_STATUS)
    if registry is None:
        return []
    return list(registry.readings(root.session))


def commands_of(root: Root) -> list[CommandSchema]:
    """Every slash command that a person may run against this root.

    `run` is deliberately not projected. It is a callable, and the job of the
    client is to *offer* the command and send the line back, not to run it. The
    daemon runs it, in the context of the root. That is also the only place
    where it could work, because a command body uses seams that live there.
    """
    registry = root.ctx.get(COMMANDS)
    if registry is None:
        return []
    return [one.schema() for one in registry.list()]


def screens_of(root: Root) -> list[ScreenSchema]:
    """The screens that this deployment contributes, without their bodies.

    `build(session)` stays in the client, and runs against the mirror of the
    log that the client keeps (`DaemonSession.session`). That is enough while
    the client *is* the TUI, as it is under textual-serve. A declarative screen
    body on the wire is the other half of P5-15, and is deferred to P7-07. This
    docstring says so on purpose, because a projection that silently dropped
    `build` would look complete.
    """
    registry = root.ctx.get(TUI_SCREENS)
    if registry is None:
        return []
    return [one.schema() for one in registry.list()]


def family_of(root: Root) -> list[ChildState]:
    """Every sub-agent beneath `root`, each from its own log (Phase 11). This is
    the one walk of a delegation tree that the seam has
    (`SubagentService.family`): a parent comes before its own children, and
    siblings are in admission order. A tree is drawn in that order.

    **The whole tree** is read in one pass when the log of the root opens
    (`SubagentService.load_children`). This is a fold, like everything in this
    module: a call loads nothing.
    """
    subagents = root.ctx.get(SUBAGENTS)
    return subagents.family(root.id) if subagents is not None else []


def family_rows(root: Root) -> list[ChildRow]:
    """`family_of`, as the rows `session/children` and `session.children` carry."""
    return [ChildRow.of(state) for state in family_of(root)]


def tools_of(root: Root) -> list[ToolSchema]:
    """What the model may call here, in the form that `--mode rpc` already answers.

    The scope is `DEPLOYMENT` and not the scope of an agent (P6-32). This says
    what the deployment offers, which is the question that a front end asks. The
    narrowed view of an agent is the concern of that agent, and a footer or a
    palette does not show it.
    """
    tools = root.ctx.get(TOOLS)
    if tools is None:
        return []
    return list(tools.schemas(scope=DEPLOYMENT))


def models_of(root: Root) -> list[ModelEntry]:
    """The models that the profile of this root lists, with the default first.
    This is what `/model` offers.

    This is the list, not the route in force. The route in force travels in the
    status (`modelKey`), and it changes without a change to the list. An unlisted
    route that a person named is therefore in neither. The free text of the
    picker is the way to name it again.
    """
    models = root.ctx.get(MODELS)
    return models.entries() if models is not None else []


def presets_of(root: Root) -> list[PresetSchema]:
    """The permission postures, with the live one marked.

    The seam resolves them on each call, like every projection here.
    `PermissionPresetService.schemas` says why that matters.
    """
    presets = root.ctx.get(PERMISSION_PRESETS)
    if presets is None:
        return []
    return list(presets.schemas(root.session))


def skills_of(root: Root) -> list[Skill]:
    """What is installed here, for the same reason and with the same scope as
    `tools_of`.

    The scope is `DEPLOYMENT` again. A child that was narrowed at spawn sees
    less, and that narrowing is the concern of the child. A person who reads a
    sidebar asks what this deployment has, which is the same question that
    `tools_of` answers.

    This is the catalog and not the bodies. The catalog in the prompt renders
    from `SkillService.list`, so a panel that is built on this cannot drift from
    what the model was told exists. `ctx.skills.body` reads a body. That is a
    request that the model makes, and not something that a front end pages in.
    """
    skills = root.ctx.get(SKILLS)
    if skills is None:
        return []
    return list(skills.list(DEPLOYMENT))


CREDENTIAL_CONFIG_KEY = "apiKeyEnv"
"""What an adapter row calls the environment variable holding its key."""


def credentials_of(root: Root) -> dict[str, bool]:
    """Every credential that the profile of this root names, and whether it is
    held. **Never the values.**

    This returns whether a credential is held and not the secret, and the shape
    enforces that. No field here could carry a value, so a later edit cannot
    leak one because it forgot to strip it. The picker only ever needed the
    boolean.

    **The names are found here, not sent here.** This function used to take a
    `names` list, so the client had to know the names. `daemon/config` therefore
    sent the *entire composed profile* to every attached front end at attach,
    and a modal module in the terminal searched it again with a recursive walk.
    That should have been a projection from the start (P5-14). The daemon holds
    the profile and the credential seam, so it can answer the whole question.
    The answer is a `{name: bool}` map with no configuration in it.

    **The names come from the profile that the root is mounted on**, and not
    from the default profile of the daemon. Two roots on two profiles can share
    one daemon (S5), and each one runs on the adapters of its own profile. This
    function used to read the profile of the supervisor. A root on another
    profile then got the names of the default, and the credential that its own
    adapter needed was missing from the picker.

    The map is in **row order**. The picker shows that order, and it is the
    reason why this is a dict and not a set.

    The service is resolved one time for the whole batch. A lookup for each name
    walked the scope chain for each name, which is what the in-process caller
    did when it looped over a predicate.
    """
    names = credentials_named(root.ctx.require(MOUNT).profile.dump())
    service = root.ctx.get(CREDENTIALS)
    if service is None:
        return dict.fromkeys(names, False)
    # Add whatever was given to this process and is named in no row. The login
    # screen takes free text, so a credential can be *held* although the profile
    # does not know it. If it were dropped here, it would disappear from the
    # picker at the moment somebody set it.
    names += [name for name in service.provided() if name not in names]
    return {name: bool(service.has(service.reference(name))) for name in names}


def credentials_named(rows: Iterable[ConfigRow]) -> list[str]:
    """Every credential that the composed configuration names, in row order.

    This walks the config of a row and does not match on plugin names. The key is
    declared, so an adapter that pH does not know is still covered.
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
        # `(list, tuple)` and not `Sequence`, because a `str` also satisfies
        # `Sequence`. Every character would recurse forever as a one-character
        # string.
        for item in value:
            yield from _walk(item)


def browse_of(supervisor: Supervisor, *, cwd: str = "") -> list[SessionSummary]:
    """Every session that a person could open, stored and live, folded here
    (P5-14).

    **One list, from the one process that can see both halves.** The logs are on
    the disk of the daemon, under the `$PH_HOME` of the daemon. Only the
    supervisor knows which roots are *mounted*. A client that asked for the two
    separately therefore had to be given a directory path and read the files
    itself. It does not have to any more: a front end on another machine, or one
    with no filesystem at all, gets the same rows.

    The `status` of a live root is its own (`running`, `waiting`, `retrying`),
    because that is what a person chooses on. To join a session that waits on
    the approval modal of somebody else is a different act from joining one that
    is working. A stored row keeps `stored`.

    A live root that the disk has not seen yet, because its log is still in a
    write buffer, gets its own row. The row is built from what the supervisor
    knows: the `cwd` of the header, so the repo that the root belongs to is on
    the row even before the file exists.
    """
    directory = supervisor.sessions_directory()
    stored = session_summaries(directory, cwd=cwd) if directory is not None else []
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
        if root_id not in known and (not cwd or root.session.header.cwd == cwd)
    )
    return list(rows)
