"""The three path roots, and why there are three (Q1).

One dotdir mixes four lifecycles. `~/.ph` would accumulate a rebuildable
multi-gigabyte venv, irreplaceable session state, a secret, a unix socket and a
PID journal — so a user cannot back up sessions without the venv, and a `~/.ph`
inside Dropbox or iCloud syncs a socket and a `processes.jsonl` full of another
machine's PIDs. That does not merely waste space: it makes the orphan journal
*wrong*, because those PIDs mean something else here.

| root | default | holds |
|---|---|---|
| `$PH_HOME` | `~/.ph` | sessions, harness state, profiles, credentials, AGENTS.md |
| `$PH_CACHE` | `$XDG_CACHE_HOME/ph` | the runtime venv, bootstrap markers |
| `$PH_RUNTIME` | `$XDG_RUNTIME_DIR/ph` | `daemon.sock`, `processes.jsonl` |

`$PH_RUNTIME` being wiped on reboot is **correct, not a limitation**: PIDs do
not survive a reboot, and a journal that did would be actively dangerous once
they are reused.

Its resolution order is three tiers, and only the last needs defending:

1. `$XDG_RUNTIME_DIR/ph` — the OS already guarantees mode 0700, the right
   owner, tmpfs, and removal at session end. No check needed.
2. a per-user `$TMPDIR/ph` (macOS `/var/folders/…`) — already per-user and
   0700. Ownership assertion only.
3. `/tmp/ph-$UID` — a predictable path in a **world-writable** directory, so
   the classic symlink-hijack shape. Verified to be a real directory, owned by
   this uid, mode 0700, and not a symlink; pH refuses to start rather than
   adopt one that is not (F9).

Resolution is pure and always verifies what already exists; `PathRoots.ensure()`
is the one place directories are created, with the mode each tier requires.

@module ph.paths
"""

from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = ["PathRoots", "RuntimeDirError", "resolve_roots"]

RuntimeTier = Literal["override", "windows", "xdg-runtime", "tmpdir", "tmp-uid"]


class RuntimeDirError(RuntimeError):
    """A candidate `$PH_RUNTIME` failed its ownership or mode check."""


@dataclass(frozen=True, slots=True)
class PathRoots:
    """The three resolved roots, plus how `$PH_RUNTIME` was reached."""

    home: Path
    cache: Path
    runtime: Path
    runtime_tier: RuntimeTier

    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    def harness_dir(self) -> Path:
        return self.home / "harness"

    def profiles_dir(self) -> Path:
        return self.home / "profiles"

    def ensure(self) -> PathRoots:
        """Create whatever is missing, with the mode its tier requires."""
        self.home.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        if self.runtime_tier == "tmp-uid":
            # Tier 3 is only ever created by us, never adopted: `resolve_roots`
            # refused anything pre-existing that failed the check.
            if not self.runtime.exists():
                self.runtime.mkdir(mode=0o700)
        else:
            self.runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
        return self

    def describe(self) -> list[tuple[str, str]]:
        """The rows `ph doctor` prints."""
        return [
            ("PH_HOME", str(self.home)),
            ("PH_CACHE", str(self.cache)),
            ("PH_RUNTIME", f"{self.runtime}  (tier: {self.runtime_tier})"),
        ]


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else None


def _default_home() -> Path:
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "ph"
    return Path.home() / ".ph"


def _default_cache() -> Path:
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "ph"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "ph"
    return Path.home() / ".cache" / "ph"


def _check_private_dir(path: Path, *, require_mode: bool) -> None:
    """Assert a directory is ours: a real dir, our uid, 0700, not a symlink."""
    if path.is_symlink():
        raise RuntimeDirError(f"{path} is a symlink; refusing to use it as $PH_RUNTIME")
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise RuntimeDirError(f"{path} is not a directory; refusing to use it")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise RuntimeDirError(
            f"{path} is owned by uid {info.st_uid}, not {os.getuid()}; refusing to use it"
        )
    if require_mode and stat.S_IMODE(info.st_mode) != 0o700:
        raise RuntimeDirError(
            f"{path} has mode {stat.S_IMODE(info.st_mode):o}, expected 700; refusing to use it"
        )


def _is_per_user_tmpdir(path: Path) -> bool:
    """Whether `$TMPDIR` is already per-user (macOS `/var/folders/...`)."""
    if not hasattr(os, "getuid"):
        return False
    try:
        info = path.lstat()
    except OSError:
        return False
    return info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o700


def _resolve_runtime() -> tuple[Path, RuntimeTier]:
    """Pick the runtime root and verify it if it already exists. Creates nothing."""
    override = _env_path("PH_RUNTIME")
    if override is not None:
        return override, "override"
    if sys.platform == "win32":
        # A named pipe replaces the socket path on Windows; the journal still
        # needs a per-boot directory, and LOCALAPPDATA\ph\runtime is the closest
        # equivalent the platform offers.
        return _default_cache() / "runtime", "windows"
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        # Tier 1: the kernel and logind own this directory's properties.
        return Path(xdg) / "ph", "xdg-runtime"
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir and _is_per_user_tmpdir(Path(tmpdir)):
        path = Path(tmpdir) / "ph"
        if path.exists():
            _check_private_dir(path, require_mode=False)
        return path, "tmpdir"
    # Tier 3: a predictable name inside a world-writable directory. Every
    # property tiers 1 and 2 get for free has to be verified here.
    uid = os.getuid() if hasattr(os, "getuid") else 0
    path = Path("/tmp") / f"ph-{uid}"
    if path.exists():
        _check_private_dir(path, require_mode=True)
    return path, "tmp-uid"


def resolve_roots(*, create: bool = False) -> PathRoots:
    """Resolve all three roots, optionally creating them.

    :raises RuntimeDirError: when the tier-3 `/tmp` fallback fails its check —
        pH refuses to start rather than adopt a directory it cannot vouch for.
    """
    runtime, tier = _resolve_runtime()
    roots = PathRoots(
        home=_env_path("PH_HOME") or _default_home(),
        cache=_env_path("PH_CACHE") or _default_cache(),
        runtime=runtime,
        runtime_tier=tier,
    )
    return roots.ensure() if create else roots
