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
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = [
    "PathRoots",
    "RuntimeDirError",
    "canonical",
    "default_cache_path",
    "default_home_path",
    "is_atomic_temp",
    "is_under",
    "resolve_roots",
    "write_atomic",
    "write_text_under",
]


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
    runtime_source: str = ""
    """The environment variable the runtime tier came from, `""` for tier 3.

    Carried beside the tier because it is decided at the same instant and by the same
    `if`: `_resolve_runtime` picks a tier *because* it read a particular variable, and
    a consumer that wants to print which one has no other way to ask. A
    `{tier: variable}` table anywhere else is a second copy of this decision that can
    drift, and can `KeyError` on a tier added here and not there.
    """

    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    def harness_dir(self) -> Path:
        return self.home / "harness"

    def profiles_dir(self) -> Path:
        return self.home / "profiles"

    def profile_overlay(self, name: str) -> Path:
        """The person's own layer over a shipped profile: `$PH_HOME/profiles/<name>.yaml`."""
        return self.profiles_dir() / f"{name}.yaml"

    def profile_dropins(self, name: str) -> Path:
        """Where pH writes rows on the person's behalf: `$PH_HOME/profiles/<name>.d/`.

        Beside the overlay rather than inside it, because the overlay is a file a
        person edits and comments, and a tool that rewrites YAML drops every comment
        in it. Each drop-in holds what one command owns, says so at the top, and
        composes *after* the overlay, in name order — the most recent decision wins.
        """
        return self.profiles_dir() / f"{name}.d"

    def daemon_socket(self) -> Path:
        """Where the supervisor listens (P5-01).

        Under `$PH_RUNTIME` because that is the tier chosen for exactly this: a
        per-boot, per-user, `0o700` directory that a cloud sync will not carry
        to another machine and a reboot clears. A socket in `$PH_HOME` would be
        synced, and a socket carried between machines is one whose peer is a
        process that never existed here.

        On Windows the same path names a named pipe rather than a filesystem
        socket; `resolve_roots` already picks a runtime tier that can hold one.
        """
        return self.runtime / "daemon.sock"

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
        """The rows `phern doctor` prints."""
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


def _cache_source() -> str:
    """Which variable `_default_cache` will read, for the windows runtime tier.

    Beside the function whose branches it names, so the two cannot disagree —
    which they did while this lived in another module as a `{tier: variable}`
    table that said `LOCALAPPDATA` for all three branches.
    """
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return "LOCALAPPDATA"
    return "XDG_CACHE_HOME" if os.environ.get("XDG_CACHE_HOME") else ""


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


def _resolve_runtime() -> tuple[Path, RuntimeTier, str]:
    """Pick the runtime root, the tier, and the variable that named it.

    Verifies what already exists; creates nothing. The third element is
    `PathRoots.runtime_source` — returned from here rather than reconstructed by
    a consumer, because this is the one place that knows which variable was read
    and it knows it at the moment it reads it.
    """
    override = _env_path("PH_RUNTIME")
    if override is not None:
        return override, "override", "PH_RUNTIME"
    if sys.platform == "win32":
        # A named pipe replaces the socket path on Windows; the journal still
        # needs a per-boot directory, and the cache root's `runtime` is the
        # closest equivalent the platform offers. The source is whichever
        # variable `_default_cache` actually used, which is why it is asked
        # rather than assumed to be `LOCALAPPDATA`.
        return _default_cache() / "runtime", "windows", _cache_source()
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        # Tier 1: the kernel and logind own this directory's properties.
        return Path(xdg) / "ph", "xdg-runtime", "XDG_RUNTIME_DIR"
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir and _is_per_user_tmpdir(Path(tmpdir)):
        path = Path(tmpdir) / "ph"
        if path.exists():
            _check_private_dir(path, require_mode=False)
        return path, "tmpdir", "TMPDIR"
    # Tier 3: a predictable name inside a world-writable directory. Every
    # property tiers 1 and 2 get for free has to be verified here.
    uid = os.getuid() if hasattr(os, "getuid") else 0
    path = Path("/tmp") / f"ph-{uid}"
    if path.exists():
        _check_private_dir(path, require_mode=True)
    return path, "tmp-uid", ""


def canonical(path: Path) -> Path:
    """`path` with every symlink resolved — the one spelling the kernel will match.

    **Every directory that participates in a boundary goes through this, once,
    where it is minted** — the three `$PH_*` roots, `default_home_path` (from which
    every provider's root descends), the workspace seam's `base` and `scratch`, and
    `sandbox-allow`'s directories. The kernel does not see a path the way it was
    typed: Seatbelt matches `subpath` rules against the resolved path, and on macOS
    `/var`, `/tmp` and `/etc` are symlinks into `/private` — so a workspace under
    `$TMPDIR` named as `/var/folders/…` was refused its own writes by a profile that
    named it that way (measured 2026-09-07). The fix belongs here rather than in the
    backend that noticed, because the same set of roots is what `permissions-fs`
    prompts about and what `bwrap` binds: `writable_roots` is documented as "the one
    definition" of that set, and a backend re-spelling it privately would have made
    the *enforced* boundary `/private/var/…` while the *prompt* boundary stayed
    `/var/…` — E6's own failure. Canonical at the source, every consumer agrees.

    Roots that bound nothing are deliberately outside this: `uploads`, the session
    stores and `temporary_directory` mint directories nobody compares against a
    workspace, and canonicalizing them would be ceremony rather than an invariant.
    The *candidate* side of a comparison is not canonical either — `FsService.resolve`
    passes an absolute path through as authored — so a caller comparing against these
    roots resolves the candidate itself where the answer would otherwise differ; see
    `permissions_fs.FsPermissions._outside_workspace`, which does it only once the
    cheap compare has already said "outside".

    `realpath`, not `Path.resolve(strict=True)`: a tail that does not exist yet
    (a scratch about to be created) resolves through what does exist and keeps
    the rest, which is what a path about to be `mkdir`ed needs.
    """
    return Path(os.path.realpath(path))


def resolve_roots(*, create: bool = False) -> PathRoots:
    """Resolve all three roots, optionally creating them.

    Canonical (`canonical`) so that everything minted under them — scratch,
    worktrees, the daemon and egress sockets — carries one spelling.

    :raises RuntimeDirError: when the tier-3 `/tmp` fallback fails its check —
        pH refuses to start rather than adopt a directory it cannot vouch for.
    """
    runtime, tier, source = _resolve_runtime()
    roots = PathRoots(
        home=canonical(_env_path("PH_HOME") or _default_home()),
        cache=canonical(_env_path("PH_CACHE") or _default_cache()),
        runtime=canonical(runtime),
        runtime_tier=tier,
        runtime_source=source,
    )
    return roots.ensure() if create else roots


def default_home_path(configured: str | None, name: str) -> Path:
    """A row's `path` setting, else `$PH_HOME/<name>` — the idiom every seam shares.

    Canonical either way (`canonical`): a configured path is a person's spelling,
    and the roots a seam mints under it must match what the kernel will enforce.
    """
    if configured:
        return canonical(Path(configured).expanduser())
    return resolve_roots().home / name


def default_cache_path(configured: str | None, *fallback: str) -> Path:
    """A row's `path` setting, else `$PH_CACHE/<fallback…>` — `default_home_path`'s twin.

    Canonical either way, and that is the whole reason this exists rather than
    being written per row: the two indexing rows arrived a week apart, one
    applied `canonical` to the operator's spelling and the other did not, so a
    configured path reached through a symlink gave one of them two spellings for
    one index — every other reader resolving to a different one, and `phern doctor`
    printing the unresolved one. `canonical`'s own docstring is the argument.

    Takes the fallback in segments because a cache root is rarely just a name:
    these rows key theirs by a digest under it, and joining that at each call
    site was the other half of the copy.
    """
    if configured:
        return canonical(Path(configured).expanduser())
    return resolve_roots().cache.joinpath(*fallback)


def is_under(candidate: Path, root: Path) -> bool:
    """Whether `candidate` is `root` or inside it, both taken as given.

    A separator-aware prefix compare, not `Path.relative_to`, which allocates a
    `Path` per root segment and runs at every gated write. `normcase` because
    `relative_to` folds case on Windows and a naive compare would not, which is the
    one behavior worth keeping from it.

    **Neither path is resolved.** A caller needing symlink-safety must `resolve()`
    *before* asking: resolving inside would answer a different question than a
    security check means to ask, and would silently pass a path whose parent is a
    link out of the tree.
    """
    base = os.path.normcase(root.as_posix()).rstrip("/")
    target = os.path.normcase(candidate.as_posix())
    return target == base or target.startswith(f"{base}/")


def write_text_under(path: Path, text: str, *, append: bool = False) -> None:
    """Write (or append) text, creating the parent directory first.

    **The append path is what this is for now** (O2). `append=False` truncates
    in place, which for a whole document another process reads without
    coordination is a window where the file is neither the old one nor the new
    one; the five JSON and YAML writers that took it — the trust roots, the TUI
    settings, the theme profile, `$PH_HOME/settings.json` and the sandbox
    profile drop-in — are on `write_atomic` below, and
    `tests/test_atomic_documents.py` is the gate that keeps a sixth from
    arriving. Truncating remains the right call for a file only this process
    reads, which is why the parameter stays.

    Every production caller left is an append, so the gate reads as a rule
    about a parameter nobody sets. That is the shape a narrower signature would
    make unnecessary — see the gate's own docstring for why it is a gate.

    Blocking; call it through `anyio.to_thread.run_sync` from async code.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)


_TEMP_SUFFIX_BYTES = 4
_ATOMIC_TEMP = re.compile(rf".+\.[0-9a-f]{{{_TEMP_SUFFIX_BYTES * 2}}}\.tmp")
"""`write_atomic`'s temp name — `<name>.<hex>.tmp` — built from the same width."""


def is_atomic_temp(path: Path) -> bool:
    """Whether `path` is a `write_atomic` temp: a write in flight, or one a crash left.

    The two cannot be told apart from the file, which is the point of asking. A
    sweep that collects unreferenced files must pass these over, or it deletes
    the temp of a write that is running *now* and the rename fails (D17) — the
    same reason nothing in `spill`'s `.staging` is ever collected. The cost is
    the same leak, one file per write a kill interrupted.
    """
    return _ATOMIC_TEMP.fullmatch(path.name) is not None


def write_atomic(path: Path, payload: bytes | str, *, skip_if_present: bool = False) -> None:
    """Write `payload` to `path` so a reader sees all of it or none of it (L7).

    **Temp-and-rename, because a torn file is worse than a missing one here.**
    Every caller of this writes something another process reads without
    coordination: a content-addressed blob whose name promises its sha256, a
    JSON index read on a daemon tick, a handle cache. `write_bytes` truncates
    and then writes, so an interrupted write leaves a prefix under the final
    name — and for the content-addressed writers that prefix is *permanent*,
    since the digest says the file is already correct and nothing ever rewrites
    it. This is atomicity against a concurrent *reader*, not durability against
    power loss: `replace` without an `fsync` of the file and its directory can
    land the rename with the bytes still in flight.

    Six sites had derived this independently, in five spellings of the temp
    name — two of them a fixed string, which any two concurrent writers collide
    on — and only one of the six removed the temp when the write failed. The
    seventh, `spill._write`, had not derived it at all.

    **A random suffix, not the pid**, because the colliding writers can be
    inside one process: `ph_rlm.harness.service` writes a projection per
    session and its own docstring names "a daemon two sessions project at
    once". `replace` is atomic within a filesystem, and the temp is a sibling
    so it always is one. The temp goes with the failure because a sweep that
    reads identity off a name — `phern attachments gc` takes everything before
    the first `.` — would otherwise count an abandoned `<digest>.png.<hex>.tmp`
    as the blob it is not.

    `skip_if_present` is the content-addressed callers' half, and only theirs:
    where the name *is* the sha256 of the bytes, a file already at that name
    already holds them, so rewriting it is one more chance to truncate
    something a live reader holds, for no gain. It is not safe anywhere the
    name does not promise the contents.

    Parents are created at the default mode. A path under a directory whose
    mode matters — `$PH_RUNTIME` at 0700 — is ensured by its owner first;
    `PathRoots.ensure()` is still the one place that happens.
    """
    if skip_if_present and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{secrets.token_hex(_TEMP_SUFFIX_BYTES)}.tmp")
    try:
        if isinstance(payload, str):
            temporary.write_text(payload, encoding="utf-8")
        else:
            temporary.write_bytes(payload)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
