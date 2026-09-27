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

**A named profile that moved since** (S6) is found only when a session starts:
the named profile as it composes now against the version the session starts on
(`profile_change`), each setting attributed. What a person decides about it is
recorded beside the base:

```text
profile/adopted {name, rows, sources, phVersion}   a version accepted; the next start
                                                  makes it the base
profile/declined {…}                               a version a person said no to
profile/override-cleared {row, command}            an override that stops applying
```

`switch_base` is the one way a base changes after the first: the new base and the
clears it implies, in one batch. An override applies from where it is logged across
a base switch — swapping base profiles keeps the overrides (decision 5) — until a
clear for its row; `fold_environment` is that rule, for a live log and a stored one.

**A setting is recorded as the value it runs with**, `${env:...}` interpolated,
because that is what an audit asks. A row's config names a credential — the key
is `ctx.credentials`'s, by name — and never holds one, so a resolved row carries
no secret to log.

@module ph.session_profile
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
)
from .cordis.loader import entry_ids, resolve_row
from .json import JsonObject, JsonValue, as_obj, as_seq, as_str, thaw_json
from .keys import MOUNT
from .session import Session
from .session.store import session_written
from .session.writers import log_writer
from .wire import literal_lookup

__all__ = [
    "ADOPTED",
    "BASE",
    "CLEARED",
    "DECLINED",
    "OVERRIDE",
    "Difference",
    "LoggedEnvironment",
    "Override",
    "OverrideNotRecorded",
    "OverrideSource",
    "ProfileBase",
    "ProfileChange",
    "ProfileSource",
    "base_of",
    "differences",
    "fold_environment",
    "listing",
    "logged_environment",
    "opened",
    "override",
    "overrides",
    "profile_change",
    "rebuilt",
    "record_adopted",
    "record_base",
    "record_declined",
    "saved_base",
    "switch_base",
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


def rebuilt(
    base: ProfileBase,
    host: Profile,
    overrides: Sequence[Override] = (),
    *,
    then: Sequence[ProfileDocument] = (),
) -> Profile:
    """The profile the log says a session runs in: its base, then its overrides in log
    order, over `host`'s rows of every other kind — and `then`, this start's own
    options, last (S5).

    The environment is the log's alone — not whatever the named file says now — and
    the host's deployment and presentation rows are this daemon's, which a session's
    base never held (decision 23). Each override is a profile entry, so rebuilding is
    composing and nothing else.
    """
    kept = [row.to_entry() for row in host.rows_of({"deployment", "presentation"})]
    documents = [ProfileDocument("host", list(kept)), *_documents(base, overrides), *then]
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

    0. **A version it adopted** (S6), made its base — `switch_base`, overrides kept.
    1. **Its base** (S3), recorded if it has none, before anything runs in it.
    2. **This start's options** (S4): each `--patch` (`ProfileDocument.override`) that
       differs from the base and the overrides already logged is logged as one,
       source `cli` — and one given again at the next start records nothing, since it
       is already the session's (decision 3).
    3. **The mount brought to what the log says**: each row an override names is set,
       once, to the value the log's composition gives it — the last word, a start
       option included, since it was logged last.

    Step 3 is a no-op for a root mounted from its own log, which every host does when
    it can read the log before mounting (S5, `ph_app.profiles.session_profile`). It is
    the fallback for one that cannot — a store that is not a file — whose root mounts
    the profile it was asked for and is brought to its log here. Only a config can be
    set on a live mount, so there an override that disables or adds a row waits for
    the next start that can read the log. Rows no override names are not touched.
    """
    mount = ctx.get(MOUNT)
    if mount is None or session.header.parent_session is not None:
        return
    # 0. A version accepted since this session last ran becomes its base now (S6),
    # so a base record always marks when a base took effect.
    env = logged_environment(session)
    if env.adopted is not None:
        await switch_base(ctx, session, env.adopted, command=f"adopt {env.adopted.name}")
        env = logged_environment(session)
    if await record_base(ctx, session) is not None:
        env = logged_environment(session)
    base = env.base
    if base is None:
        return
    logged = list(env.overrides)
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


def _flow(value: JsonValue) -> str:
    """An entry, or any container, as `--patch` would spell it: one line of flow YAML."""
    return yaml.safe_dump(thaw_json(value), default_flow_style=True, sort_keys=False).strip()


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


# ------------------------------------------------- a named profile that moved --

ADOPTED = "profile/adopted"
"""A version of the session's named profile, accepted ahead of the start that applies
it (S6): by `phern profiles adopt`, or a "yes" when a start asked. The next start
makes it the base, with `switch_base`. Ignorable: a reader that skipped it would
start the session on the base it already has, which is what it ran in until then."""

DECLINED = "profile/declined"
"""A version a person said no to when a start asked (S6), so the same version is not
asked about again. Ignorable: a reader that skipped it asks once more."""

CLEARED = "profile/override-cleared"
"""An override that stops applying, by the row it addressed. Required: a reader that
skipped it would rebuild a session with an override it no longer runs with."""


@dataclass(frozen=True, slots=True)
class LoggedEnvironment:
    """What a session's log says about its environment — the one reading of its
    `profile/*` records, for a live session and for a stored one read off disk."""

    base: ProfileBase | None = None
    overrides: tuple[Override, ...] = ()
    adopted: ProfileBase | None = None
    """A version accepted since the base, which the next start makes the base."""
    declined: ProfileBase | None = None
    """The version a person last said no to, since the base."""

    @property
    def starts_on(self) -> ProfileBase | None:
        """The base the next start runs on: the version adopted, else the saved one."""
        return self.adopted if self.adopted is not None else self.base


def fold_environment(records: Iterable[tuple[str, Mapping[str, JsonValue]]]) -> LoggedEnvironment:
    """`(type, data)` records, in log order, folded into what they say.

    A base replaces the base and settles what was adopted or declined against the
    one before it. An override applies from where it is logged, **across a base
    switch** (decision 5), until a `profile/override-cleared` names a row it
    addresses. Other types are passed over, so a caller may hand in a whole log.
    """
    base: ProfileBase | None = None
    adopted: ProfileBase | None = None
    declined: ProfileBase | None = None
    changes: list[Override] = []
    for kind, data in records:
        if kind == BASE:
            base, adopted, declined = ProfileBase.of(data), None, None
        elif kind == OVERRIDE:
            changes.append(Override.of(data))
        elif kind == CLEARED:
            row = as_str(data.get("row"))
            changes = [one for one in changes if row not in entry_ids(one.entry)]
        elif kind == ADOPTED:
            adopted, declined = ProfileBase.of(data), None
        elif kind == DECLINED:
            declined = ProfileBase.of(data)
    return LoggedEnvironment(base, tuple(changes), adopted, declined)


def logged_environment(session: Session) -> LoggedEnvironment:
    """`fold_environment` over a live session's `profile/*` records."""
    return fold_environment((event.type, event.data) for event in session.select("profile"))


async def switch_base(
    ctx: Context, session: Session, version: ProfileBase, *, command: str
) -> list[str]:
    """Make `version` the session's base, keeping its overrides. Returns the rows cleared.

    One batch (item 2): `profile/base`, and a `profile/override-cleared` for each row
    whose overrides `version` already says — decision 3's own rule, since a setting
    the base has is no deviation from it. So a crash leaves the old base with its
    overrides or the new one with its own, never a mixture.
    """
    kept = logged_environment(session).overrides
    cleared = _said_by(version, kept)
    with session.batch() as batch:
        _LOG.append(batch, BASE, version.to_wire())
        for row in cleared:
            _LOG.append(batch, CLEARED, {"row": row, "command": command})
    await session_written(ctx, session)
    return cleared


def _said_by(base: ProfileBase, kept: Sequence[Override]) -> list[str]:
    """The rows whose overrides change nothing over `base`: composed with and without
    them, every row through its model — so an override written sparse and a base
    that states every default are compared as the settings they are."""

    def composed(changes: Sequence[Override]) -> list[dict[str, JsonValue]]:
        return [resolve_row(row) for row in _environment(base, changes)]

    whole = composed(kept)
    named = dict.fromkeys(row for one in kept for row in entry_ids(one.entry))
    return [
        row
        for row in named
        if composed([one for one in kept if row not in entry_ids(one.entry)]) == whole
    ]


async def record_adopted(ctx: Context, session: Session, version: ProfileBase) -> bool:
    """Accept `version` for this session's next start. Whether it reached disk."""
    _LOG.append(session, ADOPTED, version.to_wire())
    return await session_written(ctx, session)


async def record_declined(ctx: Context, session: Session, version: ProfileBase) -> bool:
    """Say no to `version`, so a later start does not ask about it again.

    Its own write rather than one helper for both answers: every write names its
    type where it is made, which is what `test_log_writers` reads the table from."""
    _LOG.append(session, DECLINED, version.to_wire())
    return await session_written(ctx, session)


@dataclass(frozen=True, slots=True)
class ProfileChange:
    """A named profile as it composes now, against the version a session starts on."""

    was: ProfileBase
    now: ProfileBase
    differences: tuple[Difference, ...]
    shadowed: tuple[Override, ...]
    """The session's overrides of a row that changed. An override is a whole row's
    setting (the loader's rule), so it still applies over the new version, and the
    new version's change to that row does not."""
    declined: bool
    """Whether a person already said no to this version."""

    @property
    def yours(self) -> int:
        """How many of the settings are the person's own edit, rather than pH's."""
        return sum(one.by == "person" for one in self.differences)

    @property
    def worth_asking(self) -> bool:
        """Worth a person's decision: their own edit, not already declined (decision 14)."""
        return self.yours > 0 and not self.declined

    @property
    def ph_moved(self) -> str:
        """pH's part, by version: `pH 0.4.0 → 0.5.0`, or `pH` within one version."""
        was, now = self.was.ph_version, self.now.ph_version
        return f"pH {was} → {now}" if was != now else "pH"


def profile_change(env: LoggedEnvironment, now: ProfileBase) -> ProfileChange | None:
    """How `now`, the named profile as it composes, differs from what `env` starts on.

    `None` when nothing does — a session with no base, or one on the version as it is.
    """
    was = env.starts_on
    if was is None:
        return None
    found = differences(was, now)
    if not found:
        return None
    moved = {one.row for one in found}
    return ProfileChange(
        was=was,
        now=now,
        differences=tuple(found),
        shadowed=tuple(one for one in env.overrides if moved & set(entry_ids(one.entry))),
        declined=env.declined is not None and env.declined.rows == now.rows,
    )


def listing(change: ProfileChange) -> list[str]:
    """The change as a person reads it before deciding (decision 13).

    Each setting with its old and new value — for a list, what was added and
    removed — and whose it is: the person's file, or pH's. Then the session's own
    overrides that still apply over a changed row.
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
    """One value in the listing: a scalar as itself, a container as flow YAML — the
    spelling of the `--patch` and commands beside it."""
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
