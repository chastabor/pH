"""A row keeps nothing for a session the store has let go (DESIGN.md §2.5,
"Per-session tables").

Gate, in two halves:

* *every listener a shipped module writes on `session/disposed` says why what it
  forgets is not a `SessionTable`* — read from the source, so a new one cannot
  join undecided;
* *in every shipped profile that boots without a key, a session that ran a turn
  and was let go leaves nothing keyed by it, and reading it once more files
  nothing* — found by looking at what the process holds, so a plain dict nobody
  forgets is caught the same as a table.

## The defect this is for

A row keeps something per session: a cached fold, the files it wrote, a count it
reached. Each wrote its own `ctx.on("session/disposed", ...)` to drop it, and the
ways that went wrong were found one at a time:

* a row with no listener at all (`ph_stabilize.limits`) kept one entry per
  session for the life of its mount, which in a long-lived root is one per child
  it ever ran;
* a row with a listener was read once more after it had run (a status line drawn
  again, another listener of the same disposal) and filed the entry again, with
  nothing left to forget it.

`ph.session.SessionTable` is the standard that covers both: the session is its
scope, so an entry goes when the store lets the session go, and none is filed for
one that has gone. `SessionFoldCache` keeps its entries in one.

## What it can and cannot see

The source half sees a listener, not the absence of one: a row that keeps a plain
dict and never subscribes is invisible to it. The live half is what sees that, and
it sees only what a turn and a redraw reach: a table filed by a tool the probe
never calls is out of its sight, and each such table's own tests are what hold it.

@module tests.test_session_tables
"""

from __future__ import annotations

import ast
import gc
from types import CellType, FunctionType, MethodType
from typing import Any

import pytest
from workspace_layout import own_nodes, parsed_modules, qualified

from ph.keys import AGENTS, SESSIONS, TOKEN_METER, TUI_STATUS
from ph.session import Session, SessionFoldCache
from ph.testing import FAKE_OPTIONS, MountProfile
from ph_app.profiles import PROFILES, resolve_profile

pytestmark = pytest.mark.anyio

HAND_WRITTEN: dict[str, str] = {
    # `module:function` → why this is not a `SessionTable`.
    "ph.persistence.protocol:attach": (
        "a backend's own door: `SessionPersistence.forget` is a Protocol method each "
        "store implements over its own medium, and this is the one place it is called"
    ),
    "ph.seams.subagents:apply": (
        "not only a forgetting: `_let_go` moves a let-go child's last state into its "
        "parent's stored copy, which is meant to outlive the child's session, and "
        "takes the child out of its parent's live set"
    ),
    "ph_rlm.messaging:apply": (
        "keyed by a pair of agents, not by one session: every bucket naming the agent "
        "goes, on either side"
    ),
}


def _listeners() -> set[str]:
    """Every `.on("session/disposed", ...)` in shipped source, as `module:function`."""
    found: set[str] = set()
    for module, parsed in parsed_modules().items():
        scopes: list[tuple[str, ast.AST]] = [("", parsed.tree), *qualified(parsed.tree)]
        for name, scope in scopes:
            for node in own_nodes(scope):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "on"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "session/disposed"
                ):
                    found.add(f"{module}:{name}")
    return found


def test_every_disposal_listener_says_why_it_is_not_a_table() -> None:
    """A new `ctx.on("session/disposed", ...)` is a per-session table written by
    hand. It either becomes a `SessionTable` or joins `HAND_WRITTEN` with what the
    standard one could not do for it."""
    found = _listeners()

    undecided = found - set(HAND_WRITTEN)
    assert not undecided, (
        f"{sorted(undecided)} subscribe to `session/disposed` by hand. Keep what they "
        "forget in a `ph.session.SessionTable` (a fold over the log in a "
        "`SessionFoldCache`), or say in HAND_WRITTEN why neither fits."
    )
    gone = set(HAND_WRITTEN) - found
    assert not gone, f"{sorted(gone)} no longer subscribe; take them out of HAND_WRITTEN"


# ----------------------------------------------------------- the live half --

NEEDS_A_KEY = frozenset({"anthropic", "deepseek", "google", "llama"})
"""Shipped profiles that name a real provider, and so cannot run a turn here. Each
is `base` plus a provider's route, and none adds a row of its own that keeps
anything per session."""

PROBED = sorted(set(PROFILES) - NEEDS_A_KEY)
"""Every other shipped profile, read from the table rather than listed, so one
added tomorrow is probed without anybody remembering."""


def _names(key: object, session: Session) -> bool:
    """Whether a key is this session: the object, its id, or a tuple naming its id."""
    if key is session:
        return True
    if type(key) is str:
        return key == session.id
    if type(key) is tuple:
        return any(type(part) is str and part == session.id for part in key)
    return False


_WRAPPERS = ("ph.session.tables", "ph.session.folds")
"""The standard tables, which hold the container for whoever owns them."""


def _where(held: object, holder: dict[int, object]) -> str:
    """Where a container lives, as `module.Class.attribute`: up through the
    containers, cells and standard tables that only wrap it, to the first object
    that is somebody's."""
    inner = held
    for _ in range(8):
        owner = holder.get(id(inner))
        if owner is None:
            break
        kind = type(owner)
        if isinstance(owner, FunctionType):
            # A closure's variable: the function that closes over it.
            return f"{owner.__module__}.{owner.__qualname__}"
        wraps = kind in (dict, list, tuple, set, CellType, MethodType)
        if wraps or kind.__module__ in _WRAPPERS:
            inner = owner
            continue
        names = (*getattr(kind, "__slots__", ()), *getattr(owner, "__dict__", {}))
        attribute = next((name for name in names if getattr(owner, name, None) is inner), "")
        return f"{kind.__module__}.{kind.__qualname__}" + (f".{attribute}" if attribute else "")
    return f"an unowned {type(held).__name__}"


_BUILT_IN = (dict, list, tuple, set, frozenset, FunctionType, MethodType, CellType)
"""The built-in holders worth looking through: the containers, and what makes a
closure's variable reachable."""


def _scannable(owner: object) -> bool:
    """Whether to look at what `owner` holds: an instance of a class written in
    Python, or one of `_BUILT_IN`.

    Not every object, though `gc.get_referents` takes any: asked of some extension
    type's instance it leaves the collector spinning on its next pass (seen on
    CPython 3.12, and not chased further). A row's table hangs off a Python object
    or a closure, so nothing this gate is for is behind the ones left out.
    """
    kind = type(owner)
    return kind in _BUILT_IN or bool(kind.__flags__ & _HEAP_TYPE)


_HEAP_TYPE = 1 << 9
"""`Py_TPFLAGS_HEAPTYPE`: set on a class made by a `class` statement."""


def _heap(session: Session) -> tuple[list[str], list[SessionFoldCache[Any]]]:
    """Every container the process holds with this session as a key or a member,
    by where it lives; and every fold cache, for `_read_once_more`, from the same
    pass.

    Through each object's referents rather than `gc.get_objects()` alone: a dict of
    strings to numbers is one the collector does not track, and that is the shape
    of half the tables this looks for. The same pass records who holds what, which
    is what names a hit: asking the collector for each one's referrers is a walk of
    the whole heap apiece.
    """
    gc.collect()
    holder: dict[int, object] = {}
    hits: list[object] = []
    caches: list[SessionFoldCache[Any]] = []
    for owner in gc.get_objects():
        if not _scannable(owner):
            continue
        if isinstance(owner, SessionFoldCache):
            caches.append(owner)
        for held in gc.get_referents(owner):
            if id(held) in holder:
                continue
            holder[id(held)] = owner
            if type(held) in (dict, set, frozenset) and any(
                _names(key, session) for key in list(held)
            ):
                hits.append(held)
    return sorted({_where(held, holder) for held in hits}), caches


def _read_once_more(ctx: Any, session: Session, caches: list[SessionFoldCache[Any]]) -> None:  # noqa: ANN401
    """What a reader still holding a let-go session does: the footer drawn again,
    the meter asked again, and every fold cache in the process read again."""
    status = ctx.get(TUI_STATUS)
    if status is not None:
        status.readings(session)
    meter = ctx.get(TOKEN_METER)
    if meter is not None:
        meter.baseline(session)
    for cache in caches:
        cache.read(session)


@pytest.mark.parametrize("profile", PROBED)
async def test_a_session_let_go_leaves_nothing_keyed_by_it(
    mount: MountProfile, profile: str
) -> None:
    """The store lets a session go, and no row is left holding anything under it:
    not what its turn filed, and not what one more read of it would file.

    Two passes over the heap, the second after the extra read: a read never takes
    an entry out, so what the disposal left is still there to be named with it.

    Sabotage: drop the releases from `SessionStore.dispose`, or have
    `SessionTable.keep` file whatever it is handed, and every fold cache is named.
    """
    ctx = await mount(profile=resolve_profile(profile))
    session = ctx.require(SESSIONS).create(f"probe-{profile}-7f3a9c1e")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    await agent.prompt("hello")
    status = ctx.get(TUI_STATUS)
    if status is not None:
        status.readings(session)
    live, caches = _heap(session)
    assert live, "the probe found nothing while the session was live"

    await ctx.require(AGENTS).dispose(agent.id)
    ctx.require(SESSIONS).dispose(session.id)
    _read_once_more(ctx, session, caches)

    left, _ = _heap(session)
    assert not left, f"still keyed by a session the store let go: {left}"
