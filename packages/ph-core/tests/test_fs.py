"""P1-17 — the filesystem seam, and the gate that runs *before* the write.

Gate: *write-intent fires before the write; a veto prevents it.*

"Before" is the entire claim. A hook that ran after the write would be a
reporter, which is precisely the failure the feature map records in
prime-agent's `edit` skill — the diff is emitted after the file changed, so
"there is no point at which anything can say no".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.cordis import Context
from ph.seams.fs import EditIntent, FsDenied, FsService, WriteIntent, read_before_edit
from ph.session import Session

pytestmark = pytest.mark.anyio


def _fs(tmp_path: Path) -> tuple[Context, FsService]:
    root = Context()
    service = FsService(ctx=root, root=tmp_path)
    root.provide("fs", service)
    return root, service


async def test_write_intent_fires_before_the_write_and_a_veto_prevents_it(
    tmp_path: Path,
) -> None:
    root, fs = _fs(tmp_path)
    target = tmp_path / "guarded.txt"
    observed: list[bool] = []

    async def deny(intent: WriteIntent, next_: Any) -> str:
        # The file must not exist yet when policy runs.
        observed.append(intent.path.exists())
        return "writes to this path are not allowed"

    root.on("fs/write-intent", deny)
    with pytest.raises(FsDenied, match="not allowed"):
        await fs.write("guarded.txt", "content")
    assert observed == [False]
    assert not target.exists()


async def test_an_allowed_write_reaches_disk_and_announces_itself(tmp_path: Path) -> None:
    root, fs = _fs(tmp_path)
    changed: list[Path] = []
    root.on("fs/changed", lambda path: changed.append(path))
    written = await fs.write("notes/deep.txt", "hello")
    assert written.read_text() == "hello"
    assert changed == [written]


async def test_reads_record_an_observation(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    session = Session("s")
    window = await fs.read("a.txt", session=session)
    assert window.total_lines == 3
    assert [event.type for event in session.events] == ["fs/observed"]
    assert fs.observed_mtime("a.txt") is not None


async def test_a_read_window_says_how_to_get_the_next_one(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    (tmp_path / "long.txt").write_text("\n".join(str(n) for n in range(100)))
    window = await fs.read("long.txt", offset=10, limit=5)
    assert window.text.splitlines() == ["10", "11", "12", "13", "14"]
    assert window.truncated
    assert window.offset == 10


async def test_edit_requires_a_unique_match_unless_told_otherwise(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    target = tmp_path / "dup.txt"
    target.write_text("x\nx\n")
    with pytest.raises(ValueError, match="2 occurrences"):
        await fs.edit("dup.txt", "x", "y")
    assert await fs.edit("dup.txt", "x", "y", replace_all=True) == 2
    assert target.read_text() == "y\ny\n"


async def test_editing_absent_text_is_an_error(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    (tmp_path / "a.txt").write_text("hello")
    with pytest.raises(ValueError, match="no occurrence"):
        await fs.edit("a.txt", "goodbye", "hi")


async def test_read_before_edit_refuses_an_unread_file(tmp_path: Path) -> None:
    root, fs = _fs(tmp_path)
    (tmp_path / "a.txt").write_text("hello")
    await read_before_edit(root, None)

    with pytest.raises(FsDenied, match=r"read .* before editing"):
        await fs.edit("a.txt", "hello", "goodbye")

    await fs.read("a.txt")
    assert await fs.edit("a.txt", "hello", "goodbye") == 1


async def test_read_before_edit_refuses_a_file_that_changed_underneath(
    tmp_path: Path,
) -> None:
    root, fs = _fs(tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("hello")
    await read_before_edit(root, None)
    await fs.read("a.txt")

    import os
    import time

    time.sleep(0.01)
    target.write_text("changed by someone else")
    os.utime(target, (target.stat().st_atime, target.stat().st_mtime + 10))

    with pytest.raises(FsDenied, match="changed on disk"):
        await fs.edit("a.txt", "changed", "x")


async def test_edit_intent_sees_the_replacement_before_it_lands(tmp_path: Path) -> None:
    root, fs = _fs(tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("before")
    seen: list[EditIntent] = []

    async def observe(intent: EditIntent, next_: Any) -> Any:
        seen.append(intent)
        assert intent.path.read_text() == "before"
        return await next_()

    root.on("fs/edit-intent", observe)
    await fs.edit("a.txt", "before", "after")
    assert seen[0].new_text == "after"
    assert target.read_text() == "after"


async def test_glob_and_grep_skip_the_usual_noise(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "keep.py").write_text("import os\nTARGET = 1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "skip.py").write_text("TARGET = 2\n")

    found = await fs.glob("**/*.py")
    assert [Path(path).name for path in found] == ["keep.py"]

    matches = await fs.grep("TARGET")
    assert [match.path for match in matches] == [str(tmp_path / "src" / "keep.py")]
    assert matches[0].line == 2


async def test_an_invalid_regular_expression_is_reported_not_raised_raw(
    tmp_path: Path,
) -> None:
    _root, fs = _fs(tmp_path)
    with pytest.raises(ValueError, match="invalid regular expression"):
        await fs.grep("(unclosed")


async def test_relative_paths_resolve_against_the_root(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    assert fs.resolve("a/b.txt") == tmp_path / "a" / "b.txt"
    # An absolute path passes through: refusing it here would be a confinement
    # claim this layer cannot make (N2).
    assert fs.resolve("/etc/hosts") == Path("/etc/hosts")
