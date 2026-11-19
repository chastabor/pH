"""`SessionTable`: what a row keeps per session, kept only while the store holds it.

The type every per-session table is built on (DESIGN.md §2.5, "Per-session
tables"). The gate that holds the shipped rows to it is repo-level,
`tests/test_session_tables.py`; these hold the type itself.
"""

from __future__ import annotations

import pytest

from ph.keys import SESSIONS
from ph.session import Session, SessionFoldCache, SessionTable
from ph.testing import MountProfile, log_event

pytestmark = pytest.mark.anyio


async def test_an_entry_goes_when_the_store_lets_its_session_go(mount: MountProfile) -> None:
    """The half every row wrote by hand, and one forgot to. No context: the session
    is the scope.

    Sabotage: drop the `mark_disposed` releases from `SessionStore.dispose`, and the
    entry stays.
    """
    ctx = await mount()
    table: SessionTable[int] = SessionTable()
    kept, gone = ctx.require(SESSIONS).create("kept"), ctx.require(SESSIONS).create("gone")
    table.keep(kept, 1)
    table.keep(gone, 2)

    ctx.require(SESSIONS).dispose(gone.id)

    assert len(table) == 1
    assert table.get(kept) == 1
    assert gone not in table


async def test_nothing_is_filed_for_a_session_that_has_gone(mount: MountProfile) -> None:
    """The half no listener can cover: a reader still holding the session asks
    again after the disposal, and the entry it files has nobody left to forget it.

    Sabotage: have `keep` file whatever it is handed, and the session is back.
    """
    ctx = await mount()
    table: SessionTable[list[int]] = SessionTable()
    session = ctx.require(SESSIONS).create("s")
    ctx.require(SESSIONS).dispose(session.id)

    assert table.keep(session, [1]) == [1], "the value is still handed back"
    assert table.setdefault(session, list) == []
    assert len(table) == 0


def test_another_copy_of_a_session_has_its_own_entry() -> None:
    """Keyed by the session, not its id: a log opened twice is two sessions, and
    one copy's entry is not the other's."""
    table: SessionTable[int] = SessionTable()
    first, second = Session("s"), Session("s")

    table.keep(first, 1)

    assert table.get(second) is None
    assert table.get(second, 0) == 0
    assert table.get(first) == 1


async def test_what_an_entry_held_is_released_with_it(mount: MountProfile) -> None:
    """One door for the record and what it stands for (Holdings rule 2): a
    disposal and a `forget` both go through `released`, once each."""
    ctx = await mount()
    released: list[str] = []
    table: SessionTable[str] = SessionTable(released=released.append)
    first, second = ctx.require(SESSIONS).create("a"), ctx.require(SESSIONS).create("b")
    table.keep(first, "a's files")
    table.keep(second, "b's files")

    ctx.require(SESSIONS).dispose(first.id)
    table.forget(second)
    table.forget(second)

    assert released == ["a's files", "b's files"]


async def test_an_entry_filed_again_after_a_forget_still_goes(mount: MountProfile) -> None:
    """A forgotten session asked about again is filed again, and goes with it all
    the same; one table asks the session once however often it files."""
    ctx = await mount()
    table: SessionTable[int] = SessionTable()
    session = ctx.require(SESSIONS).create("s")
    table.keep(session, 1)
    table.forget(session)
    table.keep(session, 2)
    table.keep(session, 3)

    ctx.require(SESSIONS).dispose(session.id)

    assert len(table) == 0


def test_a_session_no_store_held_keeps_its_entries() -> None:
    """A session built by hand: whoever made it owns its lifetime."""
    table: SessionTable[int] = SessionTable()
    session = Session("s")

    table.keep(session, 1)

    assert table.get(session) == 1


async def test_a_fold_cache_answers_for_a_gone_session_and_files_nothing(
    mount: MountProfile,
) -> None:
    """`SessionFoldCache` keeps its entries in a `SessionTable`: read once more for a
    session the store let go, it folds and answers, and holds nothing afterwards."""
    ctx = await mount()
    cache: SessionFoldCache[int] = SessionFoldCache(lambda log: log.seq)
    session = ctx.require(SESSIONS).create("s")
    log_event(session, "turn/start", {"turn": 1})
    before = session.seq
    assert cache.read(session) == before
    assert len(cache) == 1

    ctx.require(SESSIONS).dispose(session.id)
    assert len(cache) == 0

    assert cache.read(session) == before
    assert len(cache) == 0


async def test_a_disposal_listener_still_finds_the_cached_fold(mount: MountProfile) -> None:
    """Entries go after `session/disposed` is told, so a listener that reads a fold on
    the session's way out (`SubagentService._let_go`) reads the cached one rather
    than folding the whole log again."""
    ctx = await mount()
    folded: list[int] = []

    def counted(log: Session) -> int:
        folded.append(log.seq)
        return log.seq

    cache: SessionFoldCache[int] = SessionFoldCache(counted)
    session = ctx.require(SESSIONS).create("s")
    cache.read(session)
    ctx.on("session/disposed", cache.read)

    ctx.require(SESSIONS).dispose(session.id)

    assert len(folded) == 1, "the listener refolded a log whose fold was cached"
