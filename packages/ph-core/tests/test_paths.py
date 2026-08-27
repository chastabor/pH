"""P0-16 — the three path roots.

Gate: *a wrong-owner tier-3 directory refuses to start.*

Tiers 1 and 2 get their properties from the OS; tier 3 is a predictable name in
a world-writable directory, which is the classic symlink-hijack shape. pH
refuses rather than adopting a directory it cannot vouch for (F9).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from ph.paths import RuntimeDirError, _check_private_dir, resolve_roots

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
