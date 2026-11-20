"""The base profile of a session, recorded in its log (session profiles, S3).

The base is the full environment that a session started in. It holds every
environment row of the named profile that the session started on. Each row goes
through the model of its plugin, so the defaults that the profile never wrote
are in the base too (decision 8). The base is kept in the log itself
(decision 17) as one `profile/base` record. This module writes the record, and
the record is on disk before the first step of the agent. The log alone can
therefore say what the session ran in, and can rebuild it.

```text
profile/base {name, rows, sources, phVersion}
```

* `name`: the named profile, or `""` for a `--profile` path.
* `rows`: the resolved environment rows (`Profile.resolved({"environment"})`).
* `sources`: the layers of the user that the base composed from, each as it was
  written. A comparison reads them to say whether a changed setting was an edit
  by the user or a change in pH.
* `phVersion`: the pH version that resolved the base (decision 14).

**The base is the named profile as it composes, not what the session runs
with.** A command-line start option (`ProfileDocument.override`) is a deviation
of the session from its base. It is logged as a `profile/override`, beside the
overrides that a command or a verb makes (S4). The environment of the log is
therefore always the base composed with its overrides, in order. `opened` does
what opening a root does with both. S5 mounts each root from them.

**A named profile that changed since** (S6) is found only when a session
starts. `profile_change` compares the named profile as it composes now with the
version that the session starts on, and attributes each setting. The decision
of the user about it is recorded beside the base, with the other records of
this module:

```text
profile/adopted {name, rows, sources, phVersion,   a version accepted. The next start
                 clear?}                           makes it the base
profile/declined {…}                               a version that the user declined
profile/withdrawn {…, reason}                      an adopted version that a start could
                                                   not mount, taken back
profile/override-cleared {row, command}            an override that stops applying
profile/refused {seqs, reason}                     records that the log could not hold,
                                                   taken back
profile/saved {name, path, entries}                the environment saved as a named
                                                   profile (S7)
```

`switch_base` is the only way that a base changes after the first one. It writes
the new base and the clears that the new base implies, in one batch. An override
applies from the point where it is logged. It continues to apply across a base
switch, because a swap of base profiles keeps the overrides (decision 5), until
a clear names its row. `fold_environment` implements that rule, for a live log
and for a stored log.

**A setting is recorded as the value that it runs with**, with `${env:...}`
interpolated, because an audit asks for that value. The config of a row names a
credential and never holds one. `ctx.credentials` holds the key, by name. A
resolved row therefore carries no secret to log.

@module ph.session_profile
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Literal, TypeAlias, cast

import yaml

from . import __version__
from .cordis import (
    Context,
    LoaderError,
    Profile,
    ProfileDocument,
    Row,
    compose_rows,
)
from .cordis.loader import Mount, entry_ids, new_declarations, resolve_row
from .json import JsonObject, JsonValue, as_bool, as_int, as_obj, as_seq, as_str, thaw_json
from .keys import MOUNT
from .session import Session, SessionEvent
from .session.store import session_written
from .session.writers import log_writer
from .text import count_of
from .wire import literal_lookup

__all__ = [
    "ADOPTED",
    "BASE",
    "CLEARED",
    "DECLINED",
    "OVERRIDE",
    "REFUSED",
    "SAVED",
    "WITHDRAWN",
    "Declined",
    "Difference",
    "EnvironmentFold",
    "LoggedEnvironment",
    "Override",
    "OverrideNotRecorded",
    "OverrideSource",
    "ProfileBase",
    "ProfileChange",
    "ProfileSource",
    "base_of",
    "clear_overrides",
    "differences",
    "environment",
    "environment_listing",
    "fold_environment",
    "listing",
    "logged_environment",
    "opened",
    "override",
    "override_documents",
    "overrides",
    "profile_change",
    "rebuilt",
    "record_adopted",
    "record_base",
    "record_declined",
    "record_saved",
    "record_summary",
    "resolved_environment",
    "saved_base",
    "switch_base",
    "withdraw_adoption",
    "withdrawn_note",
]

_LOG = log_writer(__name__)

BASE = "profile/base"
"""The record that holds the base profile of a session. It is required, not
ignorable: a reader that skipped it would rebuild a session in an environment
that the session never ran in."""

ChangedBy: TypeAlias = Literal["person", "pH"]
"""Who caused a changed setting: the user (their own file) or pH (its shipped
profiles or the defaults of a plugin). This is the distinction of decision 14."""


@dataclass(frozen=True, slots=True)
class ProfileSource:
    """One layer of the user that a base composed from, as it was written."""

    layer: str
    entries: tuple[JsonValue, ...]

    def about(self, row_id: str) -> list[JsonValue]:
        """Every entry here that adds, patches or removes `row_id` (`entry_ids`)."""
        return [entry for entry in self.entries if row_id in entry_ids(as_obj(entry))]


@dataclass(frozen=True, slots=True)
class ProfileBase:
    """The base of a session, as `profile/base` records it."""

    name: str
    rows: tuple[JsonObject, ...]
    sources: tuple[ProfileSource, ...]
    ph_version: str

    def to_wire(self) -> JsonObject:
        return {
            "name": self.name,
            "rows": [dict(row) for row in self.rows],
            "sources": [
                {"layer": source.layer, "entries": list(source.entries)} for source in self.sources
            ],
            "phVersion": self.ph_version,
        }

    @classmethod
    def of(cls, data: Mapping[str, JsonValue]) -> ProfileBase:
        """The base that a `profile/base` record carries, thawed from the log."""
        return cls(
            name=as_str(data.get("name")),
            rows=tuple(as_obj(thaw_json(row)) for row in as_seq(data.get("rows"))),
            sources=tuple(
                ProfileSource(
                    layer=as_str(as_obj(source).get("layer")),
                    entries=tuple(thaw_json(one) for one in as_seq(as_obj(source).get("entries"))),
                )
                for source in as_seq(data.get("sources"))
            ),
            ph_version=as_str(data.get("phVersion")),
        )

    def said_about(self, row_id: str) -> list[JsonValue]:
        """What the layers of the user said about one row, in order."""
        return [entry for source in self.sources for entry in source.about(row_id)]


def base_of(profile: Profile) -> ProfileBase:
    """`profile` as the base of a session: its environment, resolved, without start options."""
    documents = [document for document in profile.documents if not document.override]
    named = Profile.from_documents(documents, name=profile.name)
    return ProfileBase(
        name=profile.name,
        rows=tuple(named.resolved({"environment"})),
        sources=tuple(
            ProfileSource(layer=document.layer, entries=tuple(as_seq(document.entries)))
            for document in documents
            if document.sets == "environment"
        ),
        ph_version=__version__,
    )


def saved_base(session: Session) -> ProfileBase | None:
    """The base that this session last recorded and did not refuse, or `None` for a
    session that has none. The fold supplies it, so a base that the log could not
    hold is not the base of the session."""
    return logged_environment(session).base


async def record_base(ctx: Context, session: Session) -> ProfileBase | None:
    """Record the base of `session` from the profile that `ctx` mounted, one time
    and durably.

    The function records no base in the log of a sub-agent
    (`SessionHeader.is_subagent`). A child runs on the mount of its root, so the
    base of the root answers for it. A fork or a segment continues the log of
    its root, with the base in the prefix, so it already has a base. The
    exception is a fork or a segment that was cut from a log older than this
    record. It gets its own base, because it is a session of its own.

    One time: a session that is resumed with a base keeps it (S6 decides what a
    changed named profile means for it). A session from before this record gets
    its first base now. The record is appended and the session is written before
    this function returns, so the first step of the agent comes after the log
    can say what the session ran in.

    :raises OverrideNotRecorded: when the base did not reach disk. The start that
        asked is refused, and does not run on a base that its log may never hold.
    """
    mount = ctx.get(MOUNT)
    if mount is None or session.header.is_subagent:
        return None
    if saved_base(session) is not None:
        return None
    base = base_of(mount.profile)
    record = _LOG.append(session, BASE, base.to_wire())
    await _recorded(ctx, session, "this session's base was not recorded", [record])
    return base


def rebuilt(
    base: ProfileBase,
    host: Profile,
    overrides: Sequence[Override] = (),
    *,
    then: Sequence[ProfileDocument] = (),
) -> Profile:
    """The profile that the log says a session runs in (S5).

    The profile is the base, then the overrides in log order, over the rows of
    `host` of every other kind. `then`, the options of this start, comes last.

    The environment comes from the log alone, and not from what the named file
    says now. The deployment and presentation rows of the host belong to this
    daemon, and the base of a session never held them (decision 23). Each
    override is a profile entry, so rebuilding is composing and nothing else.

    **A row that `then` declares and that is already held is not declared
    again** (`new_declarations`). A start option is given at every start, and
    the first start logged it. "Held" means everything before `then`. The rows
    of the host are held too, and `host` was composed with `then` over them.
    """
    kept = [row.to_entry() for row in host.rows_of({"deployment", "presentation"})]
    before = [ProfileDocument("host", list(kept)), *_documents(base, overrides)]
    held = {row.id for row in compose_rows(before)} if then else set()
    return Profile.from_documents([*before, *_beyond(held, then)], name=base.name)


def _beyond(held: Collection[str], start: Sequence[ProfileDocument]) -> list[ProfileDocument]:
    """The documents of `start`, without each row that they declare and `held`
    names (`new_declarations`). An emptied document is left out. Each document
    has already composed one time, in the profile that it came from."""
    documents: list[ProfileDocument] = []
    for document in start:
        entries = [
            new
            for entry in as_seq(document.entries)
            if (new := new_declarations(as_obj(entry), held)) is not None
        ]
        if entries:
            documents.append(replace(document, entries=cast(JsonValue, entries)))
    return documents


def _documents(base: ProfileBase, overrides: Sequence[Override]) -> list[ProfileDocument]:
    """The environment of the log as documents: the base, then each override in order."""
    return [ProfileDocument(BASE, [dict(row) for row in base.rows]), *override_documents(overrides)]


def override_documents(overrides: Sequence[Override]) -> list[ProfileDocument]:
    """Each override as the layer that it is: environment rows, named by what asked
    for them. The layers compose over a base, or over the layers of the user
    (`/profile save`)."""
    return [
        ProfileDocument(f"{OVERRIDE}: {one.command}", [dict(one.entry)], sets="environment")
        for one in overrides
    ]


def _overridden(overrides: Sequence[Override]) -> list[str]:
    """Every row that the overrides address, one time each, in the order of first mention."""
    return list(dict.fromkeys(row for one in overrides for row in entry_ids(one.entry)))


def resolved_environment(env: LoggedEnvironment) -> list[dict[str, JsonValue]]:
    """`environment(env)`, with every row as it runs (`resolve_row`). `--full`
    prints this.

    Only the rows that an override addresses are resolved again. Every other row
    is the row of the base as recorded, which was resolved when it was recorded.
    A listing that is requested at each record of a long log therefore costs
    what its overrides touch, and not the whole profile.
    """
    if env.base is None:
        return []
    touched = set(_overridden(env.overrides))
    recorded = {as_str(row.get("id")): dict(row) for row in env.base.rows}
    return [
        resolve_row(row) if row.id in touched or row.id not in recorded else recorded[row.id]
        for row in environment(env)
    ]


def environment(env: LoggedEnvironment) -> list[Row]:
    """The environment rows that `env` says a session runs in: its base, then its
    overrides, composed. `[]` for a session with no base."""
    return [] if env.base is None else _environment(env.base, env.overrides)


# ---------------------------------------------------------------- overrides --

OVERRIDE = "profile/override"
"""One deviation from the base, in the profile grammar (S4). It is required: a
reader that skipped it would rebuild a session without an allowance that the
session ran with."""

OverrideSource: TypeAlias = Literal["cli", "command", "verb"]
"""Where a deviation came from: a start option, a slash command, or a protocol verb."""

_SOURCES: Mapping[str, OverrideSource] = literal_lookup(OverrideSource)


class OverrideNotRecorded(RuntimeError):
    """A change was not applied, because the record of it could not be written."""

    code = "OVERRIDE_NOT_RECORDED"
    """The failure has a named code, so that the reply of a daemon carries it
    (`ph_app.protocol.respond`)."""


async def _written(
    ctx: Context, session: Session, refused: str, records: Sequence[SessionEvent]
) -> bool:
    """Whether `records`, just appended, reached disk. **Every change in this
    module is written through this one function.**

    When the records did not reach disk, a `REFUSED` record that names them
    follows. The next flush that succeeds then writes the refusal with them, and
    the fold skips them. A change that the log could not hold is never in force.
    A write that was cut off part-way is refused too, because the change that it
    guards is not made."""
    try:
        if await session_written(ctx, session):
            return True
    except BaseException:
        _refuse(session, refused, records)
        raise
    _refuse(session, refused, records)
    return False


def _refuse(session: Session, refused: str, records: Sequence[SessionEvent]) -> None:
    reason = f"{refused}: the session log could not be written"
    _LOG.append(session, REFUSED, {"seqs": [one.seq for one in records], "reason": reason})


async def _recorded(
    ctx: Context, session: Session, refused: str, records: Sequence[SessionEvent]
) -> None:
    """`_written`, or an `OverrideNotRecorded` that says what was not done.

    **A change that the log does not hold is not made.** A function that changes
    the environment writes first, and continues only after this returns."""
    if not await _written(ctx, session, refused, records):
        raise OverrideNotRecorded(
            f"{refused}: the session log could not be written, and a change the log "
            "does not hold is not made"
        )


@dataclass(frozen=True, slots=True)
class Override:
    """One `profile/override`: an entry over the environment of the session, and who asked."""

    row: str
    """The rows that the entry addresses (`entry_ids`), for a reader of the log.
    The entry is the authoritative value."""
    entry: JsonObject
    """A profile entry: `{id, config}` for a live change, or any patch for a start
    option. The environment of the log is therefore the base composed with these
    entries."""
    source: OverrideSource
    command: str
    """What asked, as it was typed: `/sandbox allow host pypi.org`, `--model fast`."""

    def to_wire(self) -> JsonObject:
        return {
            "row": self.row,
            "entry": dict(self.entry),
            "source": self.source,
            "command": self.command,
        }

    @classmethod
    def of(cls, data: Mapping[str, JsonValue]) -> Override:
        return cls(
            row=as_str(data.get("row")),
            entry=as_obj(thaw_json(data.get("entry"))),
            source=_SOURCES.get(as_str(data.get("source")), "command"),
            command=as_str(data.get("command")),
        )


def overrides(session: Session) -> list[Override]:
    """The overrides in force, in log order: every one logged and not cleared since."""
    return list(logged_environment(session).overrides)


async def override(
    ctx: Context,
    session: Session,
    row_id: str,
    config: JsonValue,
    *,
    source: OverrideSource,
    command: str,
) -> bool:
    """Change the config of one row on this mount. Every such change goes through
    this function, so the log always records it.

    The function returns `False` and records nothing when the row already runs
    with that setting (decision 3). It compares through the model of the row, so
    `{}` and the defaults that it stands for are one setting. Otherwise the
    override is appended and the session is written **before** the row is
    reconfigured. A record that could not be written refuses the change. This
    order exists to prevent one failure: a session that ran with a host allowed,
    and a log that never says so.

    :raises OverrideNotRecorded: when the record did not reach disk. Nothing changed.
    :raises LoaderError: for a row that this mount cannot reconfigure live.
    """
    mount = ctx.require(MOUNT)
    row = next((one for one in mount.profile.rows if one.id == row_id), None)
    if row is None:
        raise LoaderError(f'no row with id "{row_id}" to override')
    fork = mount.forks.get(row_id)
    if _settled(row, row.config if fork is None else fork.config) == _settled(row, config):
        return False
    change = Override(
        row=row_id,
        entry={"id": row_id, "config": thaw_json(config)},
        source=source,
        command=command,
    )
    record = _LOG.append(session, OVERRIDE, change.to_wire())
    await _recorded(ctx, session, f'"{row_id}" was not changed', [record])
    await mount.reconfigure(row_id, config)
    return True


async def opened(ctx: Context, session: Session) -> None:
    """What opening a root does to its environment. `open_session` makes this one call.

    0. **A version that the session adopted** (S6) becomes its base
       (`switch_base`). The overrides are kept, or cleared when the adoption
       said `clear`.
    1. **Its base** (S3) is recorded if it has none, before anything runs in it.
    2. **The options of this start** (S4). Each `--patch`
       (`ProfileDocument.override`) that differs from the base and from the
       overrides already logged is logged as an override, with source `cli`. An
       option that is given again at the next start records nothing, because it
       already belongs to the session (decision 3): a row that it declares and
       that the log holds is not declared again (`rebuilt`). **The start is
       refused when the log cannot hold the options** (`OverrideNotRecorded`),
       by the rule of `override`: a change that the log does not hold is not
       made, and the start runs on nothing that it did not record.
    3. **The mount is brought to what the log says.** Each row that an override
       names is set, one time, to the value that the composition of the log
       gives it. The last entry logged decides the value. That includes a start
       option, because it was logged last.

    Step 3 does nothing for a root that is mounted from its own log. Every host
    mounts that way when it can read the log before it mounts (S5,
    `ph_app.profiles.session_profile`). Step 3 is the fallback for a host that
    cannot, because its store is not a file. The root of such a host mounts the
    profile that was requested, and this function brings it to its log. Only a
    config can be set on a live mount. On that mount, an override that disables
    or adds a row therefore waits for the next start that can read the log. Rows
    that no override names are not touched.
    """
    mount = ctx.get(MOUNT)
    if mount is None or session.header.is_subagent:
        return
    # 0. A version that was accepted since this session last ran becomes its base
    # now (S6), so a base record always marks when a base took effect.
    env = logged_environment(session)
    if env.adopted is not None:
        await switch_base(
            ctx,
            session,
            env.adopted,
            command=f"adopt {env.adopted.name}",
            clear_all=env.adopted_clears,
        )
        env = logged_environment(session)
    if await record_base(ctx, session) is not None:
        env = logged_environment(session)
    base = env.base
    if base is None:
        return
    logged = list(env.overrides)
    current = _environment(base, logged)
    options = [
        document
        for document in mount.profile.documents
        if document.override and document.sets == "environment"
    ]
    started = [
        as_obj(entry)
        for document in _beyond({row.id for row in current}, options)
        for entry in as_seq(document.entries)
    ]
    records: list[SessionEvent] = []
    for entry in started:
        change = Override(
            row=", ".join(entry_ids(entry)),
            entry=dict(entry),
            source="cli",
            command=f"--patch {_flow(entry)}",
        )
        after = _environment(base, [*logged, change])
        # Compare by entry, not by `Row`: a row that a start option patched to
        # the value it already had differs only in which layer last set it.
        if [row.to_entry() for row in after] == [row.to_entry() for row in current]:
            continue
        records.append(_LOG.append(session, OVERRIDE, change.to_wire()))
        logged.append(change)
        current = after
    if records:
        await _recorded(ctx, session, "this start's options were not applied", records)
    await _converge(mount, current, _overridden(logged))


async def _converge(mount: Mount, current: Sequence[Row], rows: Collection[str]) -> list[str]:
    """Bring each of `rows` on the live mount to its setting in `current`. Returns
    the rows that it could not bring there. Only a config can be set on a live
    mount, so a row that was added, removed, or turned on or off there waits for
    the next start that can read the log.
    """
    target = {row.id: row for row in current}
    mounted = {row.id: row for row in mount.profile.rows}
    left: list[str] = []
    for row_id in rows:
        want, have = target.get(row_id), mounted.get(row_id)
        if want is None and have is None:
            continue
        if want is None or have is None or want.disabled != have.disabled:
            left.append(row_id)
            continue
        if have.disabled:
            continue
        fork = mount.forks.get(row_id)
        if _settled(have, have.config if fork is None else fork.config) == _settled(
            have, want.config
        ):
            continue
        if fork is None:
            left.append(row_id)
            continue
        await mount.reconfigure(row_id, want.config)
    return left


async def clear_overrides(
    ctx: Context, session: Session, rows: Sequence[str] | None, *, command: str
) -> tuple[list[str], list[str]]:
    """Stop the overrides of `rows` from applying, or every override when `rows` is
    `None` (S7).

    The function writes one `profile/override-cleared` per row, in one batch,
    **before** the mount follows, as for an override. A clear that the log does
    not hold is not made. It returns `(cleared, left)`: the rows that were
    cleared, and those of them that the live mount could not follow. Such a row
    had an override that disabled or added a row, and its clear takes effect at
    the next start.

    :raises OverrideNotRecorded: when the records did not reach disk. Nothing changed.
    """
    env = logged_environment(session)
    named = _overridden(env.overrides)
    cleared = [row for row in (named if rows is None else rows) if row in named]
    if not cleared or env.base is None:
        return [], []
    with session.batch() as batch:
        records = [_LOG.append(batch, CLEARED, {"row": row, "command": command}) for row in cleared]
    await _recorded(ctx, session, f"{', '.join(cleared)}: not cleared", records)
    after = logged_environment(session)
    left = await _converge(ctx.require(MOUNT), environment(after), cleared)
    return cleared, left


def _settled(row: Row, config: object) -> JsonValue:
    """The config of one row through its model. "The same setting" is compared in
    this form.

    The parameter is `object` because `ForkScope` holds the config of a fork as
    that type. The value is the interpolated JSON that the row was mounted with,
    and the cast states that.
    """
    return resolve_row(replace(row, config=cast(JsonValue, config))).get("config")


def _environment(base: ProfileBase, logged: Sequence[Override]) -> list[Row]:
    """The environment rows that the base and `logged` compose to, in order."""
    return compose_rows(_documents(base, logged))


def _flow(value: JsonValue) -> str:
    """An entry, or any container, as `--patch` would spell it: one line of flow YAML."""
    return yaml.safe_dump(thaw_json(value), default_flow_style=True, sort_keys=False).strip()


@dataclass(frozen=True, slots=True)
class Difference:
    """One setting that differs between two bases, and who caused the difference."""

    row: str
    setting: str
    """A dotted path within the row, such as `config.network.hosts` or `disabled`.
    `""` when the whole row is added or gone."""
    before: JsonValue
    after: JsonValue
    by: ChangedBy


def differences(saved: ProfileBase, now: ProfileBase) -> list[Difference]:
    """Every setting that differs between `saved` and `now`, each attributed (decision 13).

    A difference is attributed to the user when their own layers say something
    different about that row. Otherwise it is attributed to pH: a shipped profile
    or a plugin default changed under a file that nobody edited.
    """
    before = {as_str(row.get("id")): row for row in saved.rows}
    after = {as_str(row.get("id")): row for row in now.rows}
    found: list[Difference] = []
    for row_id in dict.fromkeys([*before, *after]):
        by: ChangedBy = "person" if saved.said_about(row_id) != now.said_about(row_id) else "pH"
        was, is_ = before.get(row_id), after.get(row_id)
        if was is None or is_ is None:
            found.append(Difference(row_id, "", was, is_, by))
            continue
        found.extend(Difference(row_id, path, x, y, by) for path, x, y in _changed(was, is_))
    return found


def _changed(
    before: Mapping[str, JsonValue], after: Mapping[str, JsonValue], prefix: str = ""
) -> Iterator[tuple[str, JsonValue, JsonValue]]:
    for key in dict.fromkeys([*before, *after]):
        x, y = before.get(key), after.get(key)
        if x == y:
            continue
        path = f"{prefix}{key}"
        if isinstance(x, Mapping) and isinstance(y, Mapping):
            yield from _changed(x, y, f"{path}.")
        else:
            yield path, x, y


# ------------------------------------------------- a named profile that moved --

ADOPTED = "profile/adopted"
"""A version of the named profile of the session, accepted ahead of the start that
applies it (S6). Three things accept one: `phern profiles adopt`, a "yes" when a
start asked, and `/profile use`, which may carry `clear: true` (`--clear`). The
next start makes it the base, with `switch_base`. It is ignorable: a reader that
skipped it would start the session on the base that it already has, which is what
the session ran in until then."""

DECLINED = "profile/declined"
"""A version that the user declined when a start asked (S6), so that no start asks
about the same version again. It is ignorable: a reader that skipped it asks one
more time."""

WITHDRAWN = "profile/withdrawn"
"""An adopted version that was taken back, and the reason: the start that applied
it could not mount it (`withdraw_adoption`). The session stays on its base. The
version reads as declined, so no start offers it again. A later version of the
profile is asked about like any other. It is required: a reader that skipped it
would apply the adoption at its next start, and fail on it again."""

CLEARED = "profile/override-cleared"
"""An override that stops applying, by the row that it addressed. It is required:
a reader that skipped it would rebuild a session with an override that the
session no longer runs with."""

REFUSED = "profile/refused"
"""Records that a function in this module appended and then refused, because the
log could not hold them (`_written`): their seqs, and the reason. It is required:
a reader that skipped it would start the session on a change that the user was
told was refused."""


@dataclass(frozen=True, slots=True)
class Declined:
    """A version that was declined since the base. Either the user declined it when
    a start asked (`profile/declined`), or a start could not mount it
    (`profile/withdrawn`)."""

    version: ProfileBase
    reason: str = ""
    """What stopped the start that took the version back. `""` when the user declined it."""


@dataclass(frozen=True, slots=True)
class LoggedEnvironment:
    """What the log of a session says about its environment. This is the one
    reading of its `profile/*` records, for a live session and for a stored one
    that is read from disk."""

    base: ProfileBase | None = None
    overrides: tuple[Override, ...] = ()
    adopted: ProfileBase | None = None
    """A version accepted since the base, which the next start makes the base."""
    declined: Declined | None = None
    """The version that was last declined since the base, and what declined it."""
    adopted_clears: bool = False
    """Whether the overrides are cleared with the switch to `adopted`
    (`/profile use --clear`, decision 11), and do not apply over it."""

    @property
    def starts_on(self) -> ProfileBase | None:
        """The base the next start runs on: the version adopted, else the saved one."""
        return self.adopted if self.adopted is not None else self.base


@dataclass(slots=True)
class EnvironmentFold:
    """`fold_environment`, one record at a time. It serves a reader that wants the
    environment at each record of a log (the trajectory view, S8) and must not
    fold the prefix again at every record.

    The rules:

    - A base replaces the base. It also clears what was adopted or declined
      against the base before it.
    - A withdrawal takes back what was adopted, and declines it.
    - An override applies from the point where it is logged, **across a base
      switch** (decision 5), until a `profile/override-cleared` names a row that
      it addresses.
    - A refusal takes back the records that it names, as if they were never
      logged.
    - Other types are skipped, so a caller may pass a whole log.
    """

    base: ProfileBase | None = None
    adopted: ProfileBase | None = None
    declined: Declined | None = None
    adopted_clears: bool = False
    changes: list[Override] = field(default_factory=list)
    _taken: list[tuple[int, str, Mapping[str, JsonValue]]] = field(default_factory=list)
    """Every record folded so far, so that a refusal can fold them again without
    the records that it names. A session has few environment records."""

    def step(self, seq: int, kind: str, data: Mapping[str, JsonValue]) -> None:
        if kind == REFUSED:
            refused = {as_int(one) for one in as_seq(data.get("seqs"))}
            fresh = EnvironmentFold()
            for record in self._taken:
                if record[0] not in refused:
                    fresh.step(*record)
            for one in fields(self):
                setattr(self, one.name, getattr(fresh, one.name))
            return
        if not kind.startswith("profile/"):
            return
        self._taken.append((seq, kind, data))
        if kind == BASE:
            self.base, self.adopted, self.declined = ProfileBase.of(data), None, None
            self.adopted_clears = False
        elif kind == OVERRIDE:
            self.changes.append(Override.of(data))
        elif kind == CLEARED:
            row = as_str(data.get("row"))
            self.changes = [one for one in self.changes if row not in entry_ids(one.entry)]
        elif kind == ADOPTED:
            self.adopted, self.declined = ProfileBase.of(data), None
            self.adopted_clears = as_bool(data.get("clear"))
        elif kind == DECLINED:
            self.declined = Declined(ProfileBase.of(data))
        elif kind == WITHDRAWN:
            # A withdrawal is a decline with a reason: the adoption is removed,
            # and the version that it named is not offered again.
            self.adopted, self.adopted_clears = None, False
            self.declined = Declined(ProfileBase.of(data), as_str(data.get("reason")))

    @property
    def environment(self) -> LoggedEnvironment:
        return LoggedEnvironment(
            self.base, tuple(self.changes), self.adopted, self.declined, self.adopted_clears
        )


def fold_environment(
    records: Iterable[tuple[int, str, Mapping[str, JsonValue]]],
) -> LoggedEnvironment:
    """Fold `(seq, type, data)` records, in log order, into what they say
    (`EnvironmentFold`). A refusal names records by their seq."""
    fold = EnvironmentFold()
    for seq, kind, data in records:
        fold.step(seq, kind, data)
    return fold.environment


def logged_environment(session: Session) -> LoggedEnvironment:
    """`fold_environment` over a live session's `profile/*` records."""
    return fold_environment(
        (event.seq, event.type, event.data) for event in session.select("profile")
    )


async def switch_base(
    ctx: Context,
    session: Session,
    version: ProfileBase,
    *,
    command: str,
    clear_all: bool = False,
) -> list[str]:
    """Make `version` the base of the session and keep its overrides. Returns the
    rows cleared.

    The function writes one batch (item 2): a `profile/base`, and a
    `profile/override-cleared` for each row whose overrides `version` already
    says. That is the rule of decision 3, because a setting that the base has is
    no deviation from it. With `clear_all`, it clears every overridden row
    instead, for a user who chose to start the new base clean (decision 11). A
    crash therefore leaves the old base with its overrides or the new base with
    its own, never a mixture.

    :raises OverrideNotRecorded: when the batch did not reach disk, by the rule of
        `override`: the mount is not brought to a base that the log does not hold.
    """
    kept = logged_environment(session).overrides
    cleared = _overridden(kept) if clear_all else _said_by(version, kept)
    with session.batch() as batch:
        records = [_LOG.append(batch, BASE, version.to_wire())]
        records.extend(
            _LOG.append(batch, CLEARED, {"row": row, "command": command}) for row in cleared
        )
    await _recorded(ctx, session, f"the base was not changed to {version.name}", records)
    return cleared


def _said_by(base: ProfileBase, kept: Sequence[Override]) -> list[str]:
    """The rows whose overrides change nothing over `base`. The environment is
    composed with and without them, with every row through its model. An override
    that was written sparse and a base that states every default are therefore
    compared as the settings that they are."""

    def composed(changes: Sequence[Override]) -> list[dict[str, JsonValue]]:
        return [resolve_row(row) for row in _environment(base, changes)]

    whole = composed(kept)
    return [
        row
        for row in _overridden(kept)
        if composed([one for one in kept if row not in entry_ids(one.entry)]) == whole
    ]


async def record_adopted(
    ctx: Context, session: Session, version: ProfileBase, *, clear: bool = False
) -> bool:
    """Accept `version` for the next start of this session. Returns whether the
    record reached disk.

    `clear` says that the overrides of the session are cleared with the switch,
    and do not apply over the new base: `/profile use --clear` (decision 11). The
    start that applies the version clears them in the same batch as the base
    (`switch_base`)."""
    payload = dict(version.to_wire())
    if clear:
        payload["clear"] = True
    record = _LOG.append(session, ADOPTED, payload)
    return await _written(ctx, session, f"{version.name} was not adopted", [record])


async def withdraw_adoption(ctx: Context, session: Session, *, reason: str) -> None:
    """Take back a version that was adopted for the next start of this session,
    when that start could not mount it, and record why (`reason`). Does nothing
    when no version is pending.

    **The withdrawal is a record of its own** (`profile/withdrawn`), which the
    fold reads as a decline. The session starts on the base that it has, with its
    overrides. Nothing of the session was cleared, because clearing is a step of
    the start itself and that start never opened. The version is not offered
    again, and the reason stays in the log for whoever next attaches or audits.
    Without this record, a version that does not mount is applied again at every
    start, and the session never opens again.
    """
    adopted = logged_environment(session).adopted
    if adopted is None:
        return
    record = _LOG.append(session, WITHDRAWN, {**adopted.to_wire(), "reason": reason})
    # The record goes through the one function that writes every change in this
    # module. A withdrawal that the log could not hold would steer this start
    # from memory and the next start from disk, where the version is still
    # pending and will again fail to mount. `OverrideNotRecorded` names the
    # write instead, as `adopt_version` does for the adoption.
    await _recorded(ctx, session, f"{adopted.name} was not withdrawn", [record])


SAVED = "profile/saved"
"""The environment of the session, written out as a named profile (S7,
decision 5): the name, the file, and the entries written. It is ignorable: it
changes nothing that this session runs with. A base switch to the saved profile
is its own record."""


async def record_saved(
    ctx: Context, session: Session, *, name: str, path: str, entries: Sequence[JsonObject]
) -> bool:
    """Record that the environment of this session was saved as the named profile `name`."""
    _LOG.append(
        session, SAVED, {"name": name, "path": path, "entries": [dict(one) for one in entries]}
    )
    return await session_written(ctx, session)


async def record_declined(ctx: Context, session: Session, version: ProfileBase) -> bool:
    """Decline `version`, so that a later start does not ask about it again.

    This is its own write, and not one helper for both answers. Every write names
    its type at the place where it is made, and `test_log_writers` reads the
    table from those places."""
    record = _LOG.append(session, DECLINED, version.to_wire())
    return await _written(ctx, session, f"{version.name} was not declined", [record])


@dataclass(frozen=True, slots=True)
class ProfileChange:
    """A named profile as it composes now, against the version a session starts on."""

    was: ProfileBase
    now: ProfileBase
    differences: tuple[Difference, ...]
    shadowed: tuple[Override, ...]
    """The overrides of the session that address a row that changed. An override
    is the setting of a whole row (the rule of the loader), so it still applies
    over the new version, and the change of the new version to that row does
    not."""
    declined: bool
    """Whether the user already declined this version, or a start took it back."""
    withdrawn: str = ""
    """Why a start took this version back (`Declined.reason`). `""` when the user
    declined it, or nobody did."""

    @property
    def yours(self) -> int:
        """How many of the settings are an edit by the user, and not a change in pH."""
        return sum(one.by == "person" for one in self.differences)

    @property
    def worth_asking(self) -> bool:
        """Whether the change needs a decision by the user: it holds an edit of
        their own, and it is not already declined (decision 14)."""
        return self.yours > 0 and not self.declined

    @property
    def ph_moved(self) -> str:
        """The part of pH, by version: `pH 0.4.0 → 0.5.0`, or `pH` within one version."""
        was, now = self.was.ph_version, self.now.ph_version
        return f"pH {was} → {now}" if was != now else "pH"


def profile_change(env: LoggedEnvironment, now: ProfileBase) -> ProfileChange | None:
    """How `now`, the named profile as it composes, differs from what `env` starts on.

    `None` when nothing differs: a session with no base, or a session that is
    already on this version.
    """
    was = env.starts_on
    if was is None:
        return None
    found = differences(was, now)
    if not found:
        return None
    moved = {one.row for one in found}
    no = (
        env.declined if env.declined is not None and env.declined.version.rows == now.rows else None
    )
    return ProfileChange(
        was=was,
        now=now,
        differences=tuple(found),
        shadowed=tuple(one for one in env.overrides if moved & set(entry_ids(one.entry))),
        declined=no is not None,
        withdrawn=no.reason if no is not None else "",
    )


def record_summary(kind: str, data: Mapping[str, JsonValue]) -> str:
    """One line for one `profile/*` record, as an audit reads it (S8)."""
    if kind == BASE:
        base = ProfileBase.of(data)
        rows = count_of(len(base.rows), "row")
        return f"base: {base.name or 'a profile file'} (pH {base.ph_version}), {rows}"
    if kind == OVERRIDE:
        one = Override.of(data)
        return f"{one.row} ← {one.command} ({one.source})"
    if kind == CLEARED:
        return f"{as_str(data.get('row'))} no longer overridden ({as_str(data.get('command'))})"
    if kind == ADOPTED:
        return f"adopted {as_str(data.get('name'))}'s version, for the next start"
    if kind == DECLINED:
        return f"kept this session's version over {as_str(data.get('name'))}'s"
    if kind == WITHDRAWN:
        name, reason = as_str(data.get("name")), as_str(data.get("reason"))
        return f"took back {name}'s version: it did not start ({reason})"
    if kind == SAVED:
        return f"saved as {as_str(data.get('name'))}: {as_str(data.get('path'))}"
    if kind == REFUSED:
        return f"refused, so not in force: {as_str(data.get('reason'))}"
    return kind


def withdrawn_note(name: str, reason: str) -> str:
    """The sentence for a version that a start took back (`profile/withdrawn`). The
    command that asked for it, a terminal that attaches to the session, and a
    one-shot start on it all use this one wording."""
    return f"{name} did not start ({reason}), so this session runs on the version it had."


def environment_listing(env: LoggedEnvironment) -> list[str]:
    """What `/profile show` prints (S7): the base, each override and what asked for
    it, and each setting that the session runs with and its base does not say."""
    base = env.base
    if base is None:
        return ["This session has no recorded base: it records one when it next starts."]
    lines = [f"Base: {base.name or 'a profile file'} (pH {base.ph_version})"]
    if env.adopted is not None:
        lines.append(f"Adopted, for the next start: {env.adopted.name}")
    elif env.declined is not None and env.declined.reason:
        taken = withdrawn_note(env.declined.version.name, env.declined.reason)
        lines.append(f"Taken back at its last start: {taken}")
    if not env.overrides:
        lines.append("No overrides: it runs on its base as recorded.")
        return lines
    lines.append("Overrides, in the order they apply:")
    lines += _columns([(one.row, f"{one.command}  ({one.source})") for one in env.overrides])
    moved = differences(base, replace(base, rows=tuple(resolved_environment(env))))
    if moved:
        lines.append("What it runs with, where that is not its base:")
        lines += _columns([(one.row, one.setting or "(the row)", _moved(one)) for one in moved])
    return lines


def listing(change: ProfileChange) -> list[str]:
    """The change as the user reads it before a decision (decision 13).

    The listing gives each setting with its old and new value. For a list, it
    gives what was added and removed. It also says whose change it is: the file
    of the user, or pH. Then it gives the overrides of the session that still
    apply over a changed row.
    """
    name = change.now.name or "this profile"
    lines = [f"{name} has changed since this session started:"]
    lines += _columns(
        [
            (one.row, one.setting or "(the row)", _moved(one), _whose(one, change))
            for one in change.differences
        ]
    )
    if change.shadowed:
        lines.append("Still applied over it, from this session:")
        lines += _columns([(one.row, one.command) for one in change.shadowed])
    if change.declined:
        lines.append("You kept this session's version when last asked.")
    return lines


def _moved(difference: Difference) -> str:
    before, after = difference.before, difference.after
    if not difference.setting:
        return "added" if before is None else "removed"
    if isinstance(before, list) and isinstance(after, list):
        added = [f"+ {_shown(one)}" for one in after if one not in before]
        gone = [f"- {_shown(one)}" for one in before if one not in after]
        return ", ".join([*added, *gone]) or "reordered"
    return f"{_shown(before)} → {_shown(after)}"


def _shown(value: JsonValue) -> str:
    """One value in the listing: a scalar as itself, and a container as flow YAML,
    which is the spelling of the `--patch` and the commands beside it."""
    if value is None:
        return "unset"
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping | list | tuple):
        return _flow(value)
    return json.dumps(value)


def _whose(difference: Difference, change: ProfileChange) -> str:
    return "(your profile)" if difference.by == "person" else f"({change.ph_moved})"


def _columns(rows: Sequence[tuple[str, ...]]) -> list[str]:
    """Rows of cells, every column but the last padded to its widest."""
    if not rows:
        return []
    widths = [max(len(row[at]) for row in rows) for at in range(len(rows[0]) - 1)]
    return [
        "  "
        + "  ".join(
            [
                *(cell.ljust(width) for cell, width in zip(row[:-1], widths, strict=True)),
                row[-1],
            ]
        )
        for row in rows
    ]
