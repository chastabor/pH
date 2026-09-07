"""P0-16 — the three path roots.

Gate: *a wrong-owner tier-3 directory refuses to start.*

Tiers 1 and 2 get their properties from the OS; tier 3 is a predictable name in
a world-writable directory, which is the classic symlink-hijack shape. pH
refuses rather than adopting a directory it cannot vouch for (F9).

## Why `is_under` is a prefix compare and not `Path.relative_to`

`relative_to` allocates a `Path` per root segment and measured **17.9 µs against
0.16 µs** for the separator-aware compare — the same finding P4-06 recorded for
`_spellings`, which `is_under` would otherwise have quietly reintroduced at every
gated write.

## Why `runtime_source` is a field and not a lookup table

P5-11 first answered "which variable did the runtime tier come from" with a
`{tier: variable}` table in its own module — a second copy of a decision
`_resolve_runtime` had already made. It had drifted (naming `LOCALAPPDATA` for a
Windows tier that falls back to `$XDG_CACHE_HOME` or `~/.cache`) and raised
`KeyError` inside `ph doctor` for a tier added in `paths.py` and not there.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ph.paths import (
    RuntimeDirError,
    _check_private_dir,
    canonical,
    default_home_path,
    resolve_roots,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX path tiers; Windows has its own mapping"
)


def _clear(monkeypatch: Any) -> None:
    for name in ("PH_HOME", "PH_CACHE", "PH_RUNTIME", "XDG_RUNTIME_DIR", "TMPDIR"):
        monkeypatch.delenv(name, raising=False)


def test_explicit_overrides_win(tmp_path: Path, monkeypatch: Any) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("PH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("PH_RUNTIME", str(tmp_path / "run"))
    roots = resolve_roots()
    assert roots.home == tmp_path / "home"
    assert roots.cache == tmp_path / "cache"
    assert roots.runtime == tmp_path / "run"
    assert roots.runtime_tier == "override"


def test_xdg_runtime_dir_is_tier_one(tmp_path: Path, monkeypatch: Any) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    roots = resolve_roots()
    # Tier 1 needs no check at all: logind and the kernel own its properties.
    assert roots.runtime == tmp_path / "ph"
    assert roots.runtime_tier == "xdg-runtime"


def test_a_world_writable_runtime_dir_is_refused(tmp_path: Path) -> None:
    hostile = tmp_path / "hostile"
    hostile.mkdir(mode=0o777)
    with pytest.raises(RuntimeDirError, match="expected 700"):
        _check_private_dir(hostile, require_mode=True)


def test_a_symlinked_runtime_dir_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(RuntimeDirError, match="symlink"):
        _check_private_dir(link, require_mode=True)


def test_a_file_where_a_directory_belongs_is_refused(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.write_text("not a directory")
    with pytest.raises(RuntimeDirError, match="not a directory"):
        _check_private_dir(plain, require_mode=False)


def test_resolution_creates_nothing(tmp_path: Path, monkeypatch: Any) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("PH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "xdg"))
    roots = resolve_roots()
    # `ph doctor` must be able to report without side effects.
    assert not roots.home.exists()
    assert not roots.runtime.exists()


def test_roots_are_created_on_demand(tmp_path: Path, monkeypatch: Any) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("PH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "xdg"))
    (tmp_path / "xdg").mkdir()
    roots = resolve_roots(create=True)
    assert roots.home.is_dir()
    assert roots.cache.is_dir()
    assert roots.runtime.is_dir()
    assert oct(roots.runtime.stat().st_mode)[-3:] == "700"


def test_describe_names_the_tier(tmp_path: Path, monkeypatch: Any) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    rows = dict(resolve_roots().describe())
    assert set(rows) == {"PH_HOME", "PH_CACHE", "PH_RUNTIME"}
    assert "tier: xdg-runtime" in rows["PH_RUNTIME"]


def test_derived_directories_hang_off_home(tmp_path: Path, monkeypatch: Any) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    roots = resolve_roots()
    assert roots.sessions_dir() == tmp_path / "sessions"
    assert roots.harness_dir() == tmp_path / "harness"
    assert roots.profiles_dir() == tmp_path / "profiles"


def test_cache_follows_xdg(tmp_path: Path, monkeypatch: Any) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdgcache"))
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    # The venv is rebuildable and large; deleting it must cost a rebuild and
    # nothing more, which is why it does not live in $PH_HOME.
    assert resolve_roots().cache == tmp_path / "xdgcache" / "ph"
    assert os.environ.get("PH_CACHE") is None


# ------------------------------------------------------------------ tier 2 --
#
# `$TMPDIR` is macOS's per-user scratch (`/var/folders/...`), which the OS
# already creates 0700 and owned by us. It is only a tier at all when it has
# those properties, so the predicate is what decides between adopting it and
# falling through to the checked `/tmp` name.


def test_a_per_user_tmpdir_is_tier_two(tmp_path: Path, monkeypatch: Any) -> None:
    """The macOS shape: a `$TMPDIR` the OS made ours is used as-is."""
    _clear(monkeypatch)
    private = tmp_path / "folders"
    private.mkdir(mode=0o700)
    monkeypatch.setenv("TMPDIR", str(private))

    roots = resolve_roots()

    assert roots.runtime == private / "ph"
    assert roots.runtime_tier == "tmpdir"
    assert roots.runtime_source == "TMPDIR"


def test_a_shared_tmpdir_is_not_a_tier(tmp_path: Path, monkeypatch: Any) -> None:
    """A world-readable `$TMPDIR` is Linux's `/tmp` under another name.

    Adopting it would give tier 2's *trust* to a directory with none of tier 2's
    properties — so the resolver falls through to the tier-3 name it verifies
    rather than one it would merely hope about.
    """
    _clear(monkeypatch)
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    monkeypatch.setenv("TMPDIR", str(shared))

    assert resolve_roots().runtime_tier == "tmp-uid"


def test_a_tmpdir_that_is_not_there_is_not_a_tier(tmp_path: Path, monkeypatch: Any) -> None:
    """`$TMPDIR` naming nothing must not raise on the way to the fallback: the
    variable is somebody's stale export, not a reason to refuse to start."""
    _clear(monkeypatch)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "never-created"))

    assert resolve_roots().runtime_tier == "tmp-uid"


def test_an_existing_tier_two_directory_is_still_checked(tmp_path: Path, monkeypatch: Any) -> None:
    """Tier 2 verifies ownership and not mode, and the asymmetry is deliberate.

    The OS made the *parent* per-user, so `ph` inside it inherits that; what it
    cannot inherit is somebody having swapped a symlink in afterwards. Tier 3
    additionally requires 0700, because there nothing upstream promised anything.
    """
    _clear(monkeypatch)
    private = tmp_path / "folders"
    private.mkdir(mode=0o700)
    (private / "ph").symlink_to(tmp_path)
    monkeypatch.setenv("TMPDIR", str(private))

    with pytest.raises(RuntimeDirError, match="symlink"):
        resolve_roots()


def test_a_tier_two_directory_of_the_wrong_mode_is_kept(tmp_path: Path, monkeypatch: Any) -> None:
    """The other half of that asymmetry, stated so a reader does not assume 0700
    is required everywhere it is desirable."""
    _clear(monkeypatch)
    private = tmp_path / "folders"
    private.mkdir(mode=0o700)
    (private / "ph").mkdir(mode=0o755)
    monkeypatch.setenv("TMPDIR", str(private))

    assert resolve_roots().runtime_tier == "tmpdir"


def test_a_directory_owned_by_somebody_else_is_refused(tmp_path: Path, monkeypatch: Any) -> None:
    """F9's gate, at the check rather than through the resolver.

    Driven by patching `getuid` rather than by finding a directory owned by
    another user, which a test cannot create without privileges — the branch
    compares two integers and that is what is worth pinning.
    """
    directory = tmp_path / "someone-elses"
    directory.mkdir(mode=0o700)
    monkeypatch.setattr(os, "getuid", lambda: os.stat(directory).st_uid + 1)

    with pytest.raises(RuntimeDirError, match="owned by uid"):
        _check_private_dir(directory, require_mode=True)


def test_the_tier_three_directory_is_created_private(tmp_path: Path, monkeypatch: Any) -> None:
    """Tier 3 is created 0700 and never adopted, which is one rule in two halves.

    `resolve_roots` refused anything pre-existing that failed its check, so by
    the time `ensure` runs the only tier-3 directory it can be asked to make is
    one that does not exist — and it makes it private at creation rather than
    `chmod`-ing afterwards, because the window between the two is the whole
    attack.
    """
    _clear(monkeypatch)
    monkeypatch.setenv("PH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PH_CACHE", str(tmp_path / "cache"))
    runtime = tmp_path / "tier3"
    monkeypatch.setattr("ph.paths._resolve_runtime", lambda: (runtime, "tmp-uid", ""))

    roots = resolve_roots(create=True)

    assert roots.runtime.is_dir()
    assert oct(roots.runtime.stat().st_mode)[-3:] == "700"
    # Idempotent: a second start adopts what the first made rather than raising.
    assert resolve_roots(create=True).runtime == runtime


# ----------------------------------------------------------------- windows --
#
# Reachable here because the mapping is a decision about environment variables
# rather than a call into a Windows API — so the branch that picks the directory
# can be driven anywhere. What cannot be driven from here is whether a *socket*
# path works, which is P6-11's to answer with the matrix entry it owns.
#
# **`ph.paths`'s own view of `sys`, never the real one.** Patching `sys.platform`
# for the process made the whole suite intermittently fail somewhere else
# entirely: asyncio and anyio read it to choose backend behaviour, and a test
# that lied about the platform while an event loop was alive produced an
# `InvalidStateError` from a callback in an unrelated daemon test. The module
# reads exactly one attribute, so replacing its `sys` binding says the same thing
# to the code under test and nothing to anybody else.


def _on_windows(monkeypatch: Any) -> None:
    monkeypatch.setattr("ph.paths.sys", SimpleNamespace(platform="win32"))


def test_the_windows_roots_follow_the_platform_variables(monkeypatch: Any) -> None:
    """`%APPDATA%` for home and `%LOCALAPPDATA%` for cache, which is the split
    Windows makes: roaming state follows a user between machines and a cache
    does not. The runtime tier has no equivalent, so it hangs off the cache."""
    from ph.paths import _default_cache, _default_home, _resolve_runtime

    _on_windows(monkeypatch)
    monkeypatch.setenv("APPDATA", r"C:\Users\x\AppData\Roaming")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\x\AppData\Local")

    assert _default_home() == Path(r"C:\Users\x\AppData\Roaming") / "ph"
    assert _default_cache() == Path(r"C:\Users\x\AppData\Local") / "ph"
    runtime, tier, source = _resolve_runtime()
    assert tier == "windows" and source == "LOCALAPPDATA"
    assert runtime == _default_cache() / "runtime"


def test_the_windows_runtime_names_the_variable_it_actually_read(monkeypatch: Any) -> None:
    """The drift this replaced: a `{tier: variable}` table said `LOCALAPPDATA`
    for every Windows tier, including the one that falls through to
    `$XDG_CACHE_HOME`. `_cache_source` reads the same branches `_default_cache`
    does, beside it, so the two cannot disagree."""
    from ph.paths import _resolve_runtime

    _on_windows(monkeypatch)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg")

    _runtime, tier, source = _resolve_runtime()

    assert tier == "windows" and source == "XDG_CACHE_HOME"


def test_a_windows_host_with_neither_variable_still_resolves(monkeypatch: Any) -> None:
    """`~` is the last resort, and it names no variable rather than naming one it
    did not read — which is what `ph doctor` prints under "source"."""
    from ph.paths import _default_cache, _default_home, _resolve_runtime

    _on_windows(monkeypatch)
    for name in ("APPDATA", "LOCALAPPDATA", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)

    assert _default_home() == Path.home() / ".ph"
    assert _default_cache() == Path.home() / ".cache" / "ph"
    assert _resolve_runtime()[2] == ""


# ------------------------------------------------------------------ canonical --


def test_the_roots_are_canonical_because_everything_minted_under_them_must_be(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The kernel matches the path it *resolves*: Seatbelt refused a workspace spelled
    `/var/folders/…` its own writes, because that is `/private/var/…` to it. Every
    root pH mints descends from these three, so this is where one spelling is fixed
    — and why no backend has to re-spell the set privately (E6)."""
    _clear(monkeypatch)
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    monkeypatch.setenv("PH_HOME", str(link / "home"))
    monkeypatch.setenv("PH_CACHE", str(link / "cache"))
    monkeypatch.setenv("PH_RUNTIME", str(link / "run"))

    roots = resolve_roots()

    assert roots.home == real / "home", "resolved through the link, tail kept"
    assert roots.cache == real / "cache"
    assert roots.runtime == real / "run"
    assert default_home_path(str(link / "own"), "x") == real / "own", "a configured path too"
    assert default_home_path(None, "scratch") == real / "home" / "scratch"


def test_canonical_resolves_what_exists_and_keeps_the_rest(tmp_path: Path) -> None:
    """`realpath`, not `resolve(strict=True)`: a scratch about to be created is a
    path whose tail does not exist yet, and it must still come out canonical."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    assert canonical(link / "not" / "yet") == real / "not" / "yet"
    assert canonical(real) == real, "already canonical is a no-op"
