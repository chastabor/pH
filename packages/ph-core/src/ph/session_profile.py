"""A session's base profile, recorded in its log (session profiles, S3).

The environment a session started in, in full: every environment row of the named
profile it was started on, each through its plugin's model, so the defaults it
never wrote are in it too (decision 8). Kept in the log itself (decision 17) as
one `profile/base` record, written through this module's own door and on disk
before the agent's first step, so the log alone can say what the session ran in
and rebuild it.

```text
profile/base {name, rows, sources, phVersion}
```

* `name` — the named profile, `""` for a `--profile` path.
* `rows` — the resolved environment rows (`Profile.resolved({"environment"})`).
* `sources` — the person's own layers it composed from, each as written: what a
  comparison reads to say whether a changed setting was the person's edit or pH's.
* `phVersion` — the pH that resolved it (decision 14).

**The named profile as it composes, not what the session runs with.** A
command-line start option (`ProfileDocument.override`) is what the session
deviates from its base by, and is logged as a `profile/override` beside the ones a
command or a verb makes (S4) — so the log's environment is always the base
composed with its overrides, in order. `opened` is what opening a root does with
both; S5 mounts each root from them.

**A setting is recorded as the value it runs with**, `${env:...}` interpolated,
because that is what an audit asks. A row's config names a credential — the key
is `ctx.credentials`'s, by name — and never holds one, so a resolved row carries
no secret to log.

@module ph.session_profile
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
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
    normalize_plugin,
)
from .cordis.loader import entry_ids, resolve_plugin, resolve_row
from .json import JsonObject, JsonValue, as_obj, as_seq, as_str, thaw_json
from .keys import MOUNT
from .session import Session
from .session.store import session_written
from .session.writers import log_writer
from .wire import literal_lookup

__all__ = [
    "BASE",
    "OVERRIDE",
    "Difference",
    "Override",
    "OverrideNotRecorded",
    "OverrideSource",
    "ProfileBase",
    "ProfileSource",
    "base_of",
    "differences",
    "opened",
    "override",
    "overrides",
    "rebuilt",
    "record_base",
    "saved_base",
]

_LOG = log_writer(__name__)

BASE = "profile/base"
"""The record a session's base profile is kept in. Required, not ignorable: a
reader that skipped it would rebuild a session in an environment it never ran in."""

ChangedBy: TypeAlias = Literal["person", "pH"]
"""Who a changed setting is owed to: the person's own file, or pH (its shipped
profiles or a plugin's defaults) — decision 14's distinction."""


@dataclass(frozen=True, slots=True)
class ProfileSource:
    """One of the person's layers a base composed from, as it was written."""

    layer: str
    entries: tuple[JsonValue, ...]

    def about(self, row_id: str) -> list[JsonValue]:
        """Every entry here that adds, patches or removes `row_id` (`entry_ids`)."""
        return [entry for entry in self.entries if row_id in entry_ids(as_obj(entry))]


@dataclass(frozen=True, slots=True)
class ProfileBase:
    """A session's base, as `profile/base` records it."""

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
        """The base a `profile/base` record carries, thawed out of the log."""
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
        """What the person's layers said about one row, in order."""
        return [entry for source in self.sources for entry in source.about(row_id)]


def base_of(profile: Profile) -> ProfileBase:
    """`profile` as a session's base: its environment, resolved, without start options."""
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
    """The base this session last recorded, or `None` for one that has none."""
    return session.projection(BASE, lambda event: ProfileBase.of(event.data))


async def record_base(ctx: Context, session: Session) -> ProfileBase | None:
    """Record `session`'s base from the profile `ctx` mounted, once, durably.

    For a root session only: a child runs on its root's mount and a segment
    continues its root's log, so both are answered by the root's base. Once:
    a session resumed with a base keeps it (S6 decides what a changed named
    profile means for it), and one from before this record gets its first now.
    Written, then the session written, before this returns — so the agent's first
    step comes after the log can say what it ran in.
    """
    mount = ctx.get(MOUNT)
    if mount is None or session.header.parent_session is not None:
        return None
    if session.latest(BASE) is not None:
        return None
    base = base_of(mount.profile)
    _LOG.append(session, BASE, base.to_wire())
    await session_written(ctx, session)
    return base


def rebuilt(base: ProfileBase, host: Profile, overrides: Sequence[Override] = ()) -> Profile:
    """The profile the log says a session runs in: its base, then its overrides in log
    order, over `host`'s rows of every other kind.

    The environment is the log's alone — not whatever the named file says now — and
    the host's deployment and presentation rows are this daemon's, which a session's
    base never held (decision 23). Each override is a profile entry, so rebuilding is
    composing and nothing else.
    """
    kept = [
        row.to_entry()
        for row in host.rows
        if normalize_plugin(resolve_plugin(row.name)).affects != "environment"
    ]
    documents = [ProfileDocument("host", list(kept)), *_documents(base, overrides)]
    return Profile.from_documents(documents, name=base.name)


def _documents(base: ProfileBase, overrides: Sequence[Override]) -> list[ProfileDocument]:
    """The log's environment as documents: the base, then each override in order."""
    return [
        ProfileDocument(BASE, [dict(row) for row in base.rows]),
        *(ProfileDocument(f"{OVERRIDE}: {one.command}", [dict(one.entry)]) for one in overrides),
    ]


# ---------------------------------------------------------------- overrides --

OVERRIDE = "profile/override"
"""One deviation from the base, in the profile grammar (S4). Required: a reader that
skipped it would rebuild a session without an allowance it ran with."""

OverrideSource: TypeAlias = Literal["cli", "command", "verb"]
"""Where a deviation came from: a start option, a slash command, or a protocol verb."""

_SOURCES: Mapping[str, OverrideSource] = literal_lookup(OverrideSource)


class OverrideNotRecorded(RuntimeError):
    """A change was not applied, because the record of it could not be written."""


@dataclass(frozen=True, slots=True)
class Override:
    """One `profile/override`: an entry over the session's environment, and who asked."""

    row: str
    """The rows it addresses (`entry_ids`), for a reader of the log — the entry is the fact."""
    entry: JsonObject
    """A profile entry — `{id, config}` for a live change, any patch for a start
    option — so that the log's environment is the base composed with these."""
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
    """The overrides in force: every one logged since the latest base, in order."""
    since = session.latest(BASE)
    return [
        Override.of(event.data)
        for event in session.events_from(since.seq + 1 if since is not None else 0)
        if event.type == OVERRIDE
    ]


async def override(
    ctx: Context,
    session: Session,
    row_id: str,
    config: JsonValue,
    *,
    source: OverrideSource,
    command: str,
) -> bool:
    """Change one row's config on this mount — the one door, so the log always says so.

    `False`, recording nothing, when the row already runs with that setting (decision
    3): compared through the row's model, so `{}` and the defaults it stands for are
    one setting. Otherwise the override is appended and the session written **before**
    the row is reconfigured, and a record that could not be written refuses the
    change — a session that ran with a host allowed and a log that never says so is
    the failure this order exists to rule out.

    :raises OverrideNotRecorded: when the record did not reach disk; nothing changed.
    :raises LoaderError: for a row this mount cannot reconfigure live.
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
    _LOG.append(session, OVERRIDE, change.to_wire())
    if not await session_written(ctx, session):
        raise OverrideNotRecorded(
            f'"{row_id}" was not changed: the session log could not be written, and a '
            "change the log does not hold is not made"
        )
    await mount.reconfigure(row_id, config)
    return True


async def opened(ctx: Context, session: Session) -> None:
    """What opening a root does to its environment — the one call `open_session` makes.

    1. **Its base** (S3), recorded if it has none, before anything runs in it.
    2. **This start's options** (S4): each `--patch` (`ProfileDocument.override`) that
       differs from the base and the overrides already logged is logged as one,
       source `cli` — and one given again at the next start records nothing, since it
       is already the session's (decision 3).
    3. **The mount brought to what the log says**: each row an override names is set,
       once, to the value the log's composition gives it — the last word, a start
       option included, since it was logged last.

    Step 3 is how a `/sandbox allow` holds on the next start while the daemon still
    mounts one composition for every root; S5 mounts each root from its own log and
    retires it. Only a config can be set on a live mount, so an override that
    disables or adds a row is logged and waits for S5. Rows no override names are not
    touched: what a changed named profile means for a saved base is S6's question.
    """
    mount = ctx.get(MOUNT)
    if mount is None or session.header.parent_session is not None:
        return
    await record_base(ctx, session)
    base = saved_base(session)
    if base is None:
        return
    logged = overrides(session)
    current = _environment(base, logged)
    started = [
        as_obj(entry)
        for document in mount.profile.documents
        if document.override and document.sets == "environment"
        for entry in as_seq(document.entries)
    ]
    recorded = False
    for entry in started:
        change = Override(
            row=", ".join(entry_ids(entry)),
            entry=dict(entry),
            source="cli",
            command=f"--patch {_flow(entry)}",
        )
        after = _environment(base, [*logged, change])
        # By entry, not by `Row`: a row a start option patched to what it already
        # was differs only in which layer last said so.
        if [row.to_entry() for row in after] == [row.to_entry() for row in current]:
            continue
        _LOG.append(session, OVERRIDE, change.to_wire())
        logged.append(change)
        current, recorded = after, True
    if recorded:
        await session_written(ctx, session)
    named = {row_id for change in logged for row_id in entry_ids(change.entry)}
    for row in current:
        if row.id not in named:
            continue
        mounted = next((one for one in mount.profile.rows if one.id == row.id), None)
        fork = mount.forks.get(row.id)
        if mounted is None or mounted.disabled or fork is None:
            continue
        if _settled(mounted, fork.config) != _settled(mounted, row.config):
            await mount.reconfigure(row.id, row.config)


def _settled(row: Row, config: object) -> JsonValue:
    """One row's config through its model — what "the same setting" is compared as.

    `object` because that is how `ForkScope` holds a fork's config; it is the
    interpolated JSON the row was mounted with, which is what the cast says.
    """
    return resolve_row(replace(row, config=cast(JsonValue, config))).get("config")


def _environment(base: ProfileBase, logged: Sequence[Override]) -> list[Row]:
    """The environment rows the base and `logged` compose to, in order."""
    return compose_rows(_documents(base, logged))


def _flow(entry: JsonObject) -> str:
    """An entry as `--patch` would spell it: one line of flow YAML."""
    return yaml.safe_dump(thaw_json(entry), default_flow_style=True, sort_keys=False).strip()


@dataclass(frozen=True, slots=True)
class Difference:
    """One setting that differs between two bases, and who it is owed to."""

    row: str
    setting: str
    """A dotted path within the row — `config.network.hosts`, `disabled` — or `""`
    when the whole row is added or gone."""
    before: JsonValue
    after: JsonValue
    by: ChangedBy


def differences(saved: ProfileBase, now: ProfileBase) -> list[Difference]:
    """Every setting that differs between `saved` and `now`, each attributed (decision 13).

    To the person when their own layers say something different about that row;
    to pH otherwise — a shipped profile or a plugin default moved under a file
    nobody edited.
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
