"""`ph.path_watch`: told when one directory entry may have changed (P12-05).

The entry here is a plain file: the watch is about a name in a directory, not
about what kind of thing the name points at, and a plain file is what a test can
make anywhere. What is pinned: removing, replacing or renaming the entry is heard;
another entry in the same directory is not (Linux, where inotify names it); the
directory going away ends the watch; an ancestor renamed is heard, because it
moves the path; a directory that is not there cannot be watched; and closing gives
the descriptors back.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import pytest

from ph.orphans import JOURNAL_NAME
from ph.path_watch import EntryWatch, WatchUnavailable
from ph.testing import open_fds, settled

pytestmark = pytest.mark.anyio

linux_only = pytest.mark.skipif(sys.platform != "linux", reason="inotify names the entry")


@asynccontextmanager
async def _watching(path: Path) -> AsyncIterator[list[str]]:
    """What the watch says from here on: `"change"` per yield, `"end"` when it ends."""
    seen: list[str] = []
    watch = EntryWatch(path)
    async with anyio.create_task_group() as tasks:

        async def read() -> None:
            async for _ in watch.changes():
                seen.append("change")
            seen.append("end")

        tasks.start_soon(read)
        try:
            yield seen
        finally:
            tasks.cancel_scope.cancel()
            watch.close()


def _entry(tmp_path: Path) -> Path:
    directory = tmp_path / "runtime"
    directory.mkdir()
    entry = directory / "daemon.sock"
    entry.touch()
    return entry


def _replace(entry: Path) -> None:
    """Removed and made again: what a second daemon binding the same path does."""
    entry.unlink()
    entry.touch()


@pytest.mark.parametrize(
    "change",
    [Path.unlink, _replace, lambda entry: entry.rename(entry.with_name("elsewhere"))],
    ids=["removed", "replaced", "renamed-away"],
)
async def test_a_change_to_the_entry_is_heard(
    tmp_path: Path, change: Callable[[Path], object]
) -> None:
    entry = _entry(tmp_path)
    async with _watching(entry) as seen:
        change(entry)
        await settled(lambda: "change" in seen, "the change to be heard")


@linux_only
async def test_another_entry_in_the_directory_is_not_heard(tmp_path: Path) -> None:
    """`$PH_RUNTIME` holds more than the socket, and its other files change often."""
    entry = _entry(tmp_path)
    async with _watching(entry) as seen:
        other = entry.with_name(JOURNAL_NAME)
        other.touch()
        other.unlink()
        await anyio.sleep(0.2)
        assert seen == []


async def test_the_directory_going_away_ends_the_watch(tmp_path: Path) -> None:
    """Logout reaps the whole directory: heard, and then there is nothing left to
    watch."""
    entry = _entry(tmp_path)
    async with _watching(entry) as seen:
        shutil.rmtree(entry.parent)
        await settled(lambda: seen[-1:] == ["end"], "the watch to end")
        assert "change" in seen


async def test_an_ancestor_renamed_is_heard(tmp_path: Path) -> None:
    """Moving any directory above the entry moves the path, though nothing in the
    entry's own directory changed."""
    deep = tmp_path / "home" / "runtime"
    deep.mkdir(parents=True)
    entry = deep / "daemon.sock"
    entry.touch()
    async with _watching(entry) as seen:
        (tmp_path / "home").rename(tmp_path / "moved")
        await settled(lambda: "change" in seen, "the ancestor's rename to be heard")


def test_a_directory_that_is_not_there_cannot_be_watched(tmp_path: Path) -> None:
    with pytest.raises(WatchUnavailable):
        EntryWatch(tmp_path / "absent" / "daemon.sock")


def test_closing_gives_the_descriptors_back(tmp_path: Path) -> None:
    """A daemon arms one watch for its life, but a test suite arms hundreds."""
    entry = _entry(tmp_path)
    before = open_fds()

    watch = EntryWatch(entry)
    watch.close()
    watch.close()  # twice is harmless

    assert open_fds() == before
