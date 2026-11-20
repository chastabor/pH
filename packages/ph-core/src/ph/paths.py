"""The three path roots, and why there are three (Q1).

One dot directory mixes four lifecycles. `~/.ph` would collect a rebuildable
multi-gigabyte venv, irreplaceable session state, a secret, a unix socket, and
a PID journal. A user then cannot back up the sessions without the venv. A
`~/.ph` inside Dropbox or iCloud also syncs a socket and a `processes.jsonl`
that is full of the PIDs of another machine. That does more than waste space.
It makes the orphan journal *wrong*, because those PIDs mean something else on
this machine.

| root | default | holds |
|---|---|---|
| `$PH_HOME` | `~/.ph` | sessions, harness state, profiles, AGENTS.md, `daemon.yaml` |
| `$PH_CACHE` | `$XDG_CACHE_HOME/ph` | the runtime venv, bootstrap markers |
| `$PH_RUNTIME` | `$XDG_RUNTIME_DIR/ph` | `daemon.sock`, `processes.jsonl` |

A reboot wipes `$PH_RUNTIME`. That is **correct, not a limitation**. PIDs do
not survive a reboot, and a journal that survived would be dangerous as soon as
the PIDs are reused.

`$PH_RUNTIME` resolves through three tiers. Only the last tier needs a defense:

1. `$XDG_RUNTIME_DIR/ph`. The OS already guarantees mode 0700, the right
   owner, tmpfs, and removal at session end. No check is needed.
2. A per-user `$TMPDIR/ph` (macOS `/var/folders/…`). It is already per-user
   and 0700. pH asserts ownership only.
3. `/tmp/ph-$UID`. This is a predictable path in a **world-writable**
   directory, which is the classic shape of a symlink hijack. pH verifies that
   it is a real directory, owned by this uid, with mode 0700, and not a
   symlink. If it is not, pH refuses to start and does not adopt it (F9).

Resolution is pure and always verifies what already exists.
`PathRoots.ensure()` creates directories, with the mode that each tier
requires. `PathRoots.ensure_runtime()` is the part of it that creates
`$PH_RUNTIME` alone.

@module ph.paths
"""

from __future__ import annotations

import os
import re
import secrets
import stat
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .host import load_host_config

__all__ = [
    "PathRoots",
    "RuntimeDirError",
    "canonical",
    "default_cache_path",
    "default_home_path",
    "file_stamp",
    "filesystem_of",
    "free_sibling",
    "is_under",
    "make_directories",
    "replace_durably",
    "resolve_roots",
    "sync_directory",
    "write_all",
    "write_atomic",
    "write_atomic_all",
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
    """The environment variable that the runtime tier came from, `""` for tier 3.

    This field is carried beside the tier because the same `if` decides both at
    the same instant. `_resolve_runtime` picks a tier *because* it read a
    particular variable. A consumer that wants to print which variable has no
    other way to ask. A `{tier: variable}` table anywhere else is a second copy
    of this decision. The copy can drift, and it can raise `KeyError` on a tier
    that is added here and not there.
    """

    sessions: Path | None = None
    """The `paths.sessions` setting of `daemon.yaml`, resolved. `None` means
    `$PH_HOME/sessions`."""
    profiles: Path | None = None
    """The `paths.profiles` setting of `daemon.yaml`, resolved. `None` means
    `$PH_HOME/profiles`."""

    def sessions_dir(self) -> Path:
        return self.sessions or self.home / "sessions"

    def harness_dir(self) -> Path:
        return self.home / "harness"

    def profiles_dir(self) -> Path:
        return self.profiles or self.home / "profiles"

    def profile_overlay(self, name: str) -> Path:
        """The user's named profile `name`: `$PH_HOME/profiles/<name>.yaml`."""
        return self.profiles_dir() / f"{name}.yaml"

    def profile_dropins(self, name: str) -> Path:
        """Where `/sandbox` wrote rows over `name` before S4: `$PH_HOME/profiles/<name>.d/`.

        pH reads this directory until `phern profiles fold` folds it into the
        named profile. Nothing writes one now: a change that a session makes is
        an override in the log of that session.
        """
        return self.profiles_dir() / f"{name}.d"

    def daemon_socket(self) -> Path:
        """Where the supervisor listens (P5-01).

        The socket is under `$PH_RUNTIME` because that tier was chosen for
        exactly this purpose. It is a per-boot, per-user, `0o700` directory. A
        cloud sync does not carry it to another machine, and a reboot clears
        it. A socket in `$PH_HOME` would be synced. The peer of a socket that
        was carried between machines is a process that never existed on this
        machine.

        On Windows the same path names a named pipe and not a filesystem
        socket. `resolve_roots` already picks a runtime tier that can hold one.
        """
        return self.runtime / "daemon.sock"

    def daemon_log(self) -> Path:
        """Where `phern daemon` writes its log.

        A daemon that a UI spawned has no terminal. `launch._detach` points its
        stderr at a file (`daemon_stderr`), and that file keeps only what comes
        before this log is open. The log is under `$PH_HOME`, with the sessions
        that it reports on. It is not under the runtime directory, which a
        logout reaps.
        """
        return self.home / "logs" / "daemon.log"

    def daemon_stderr(self) -> Path:
        """Where a daemon that a UI spawned writes its stderr (`launch._detach`).

        This file is beside `daemon_log` and covers a gap in it. The daemon
        opens the log only after its profile has composed. A refusal *before*
        that point therefore reached no file at all. Examples: a row whose
        plugin does not import, an option that it does not take, a traceback
        during startup. This file belongs to the launcher. The launcher
        truncates it at each spawn. When the process ends before it listens,
        the launcher reads the file back into its own refusal.
        """
        return self.home / "logs" / "daemon.stderr"

    def ensure(self) -> PathRoots:
        """Create whatever is missing, with the mode its tier requires."""
        self.home.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        return self.ensure_runtime()

    def ensure_runtime(self) -> PathRoots:
        """Create `$PH_RUNTIME` alone if it is missing, with the mode that its tier
        requires. This is for a writer under it that must not create the
        directory itself (`ph.orphans`)."""
        if self.runtime_tier == "tmp-uid":
            # pH creates the tier-3 directory itself and never adopts one that
            # failed the check: `resolve_roots` refused any existing directory
            # that failed it.
            if not self.runtime.exists():
                self.runtime.mkdir(mode=0o700)
        else:
            self.runtime.mkdir(parents=True, exist_ok=True, mode=0o700)
        return self

    def describe(self) -> list[tuple[str, str]]:
        """The rows `phern doctor` prints."""
        moved = [
            (name, f"{where}  (daemon.yaml)")
            for name, where in (("sessions", self.sessions), ("profiles", self.profiles))
            if where is not None
        ]
        return [
            ("PH_HOME", str(self.home)),
            *moved,
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

    This function is beside the function whose branches it names, so that the
    two cannot disagree. They did disagree while this lived in another module
    as a `{tier: variable}` table. That table said `LOCALAPPDATA` for all three
    branches.
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

    This verifies what already exists and creates nothing. The third element is
    `PathRoots.runtime_source`. It is returned from here and not reconstructed
    by a consumer, because this is the one place that knows which variable was
    read, and it knows that at the moment it reads the variable.
    """
    override = _env_path("PH_RUNTIME")
    if override is not None:
        return override, "override", "PH_RUNTIME"
    if sys.platform == "win32":
        # On Windows a named pipe replaces the socket path. The journal still
        # needs a per-boot directory, and `runtime` under the cache root is the
        # closest equivalent that the platform offers. The source is whichever
        # variable `_default_cache` actually used. That is why the code asks
        # for it and does not assume `LOCALAPPDATA`.
        return _default_cache() / "runtime", "windows", _cache_source()
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        # Tier 1: the kernel and logind own the properties of this directory.
        return Path(xdg) / "ph", "xdg-runtime", "XDG_RUNTIME_DIR"
    tmpdir = os.environ.get("TMPDIR")
    if tmpdir and _is_per_user_tmpdir(Path(tmpdir)):
        path = Path(tmpdir) / "ph"
        if path.exists():
            _check_private_dir(path, require_mode=False)
        return path, "tmpdir", "TMPDIR"
    # Tier 3: a predictable name inside a world-writable directory. Tiers 1 and
    # 2 get every property for free. Here each property has to be verified.
    uid = os.getuid() if hasattr(os, "getuid") else 0
    path = Path("/tmp") / f"ph-{uid}"
    if path.exists():
        _check_private_dir(path, require_mode=True)
    return path, "tmp-uid", ""


def canonical(path: Path) -> Path:
    """`path` with every symlink resolved: the one spelling that the kernel matches.

    **Every directory that takes part in a boundary goes through this function
    one time, where its path is first produced.** Those directories are:

    - the three `$PH_*` roots
    - `default_home_path`, from which the root of every provider descends
    - the `base` and `scratch` of the workspace seam
    - the directories of `sandbox-allow`

    The kernel does not see a path the way it was typed. Seatbelt matches
    `subpath` rules against the resolved path, and on macOS `/var`, `/tmp` and
    `/etc` are symlinks into `/private`. So a workspace under `$TMPDIR` that
    was named as `/var/folders/…` was refused its own writes by a profile that
    named it that way (measured 2026-09-07).

    The fix belongs here and not in the backend that noticed the problem. The
    same set of roots is what `permissions-fs` prompts about and what `bwrap`
    binds. `writable_roots` is documented as "the one definition" of that set.
    If a backend spelled the set again in private, the *enforced* boundary
    would be `/private/var/…` while the *prompt* boundary stayed `/var/…`. That
    is the failure of E6. When the roots are canonical at the source, every
    consumer agrees.

    Roots that bound nothing are deliberately outside this rule. `uploads`,
    the session stores and `temporary_directory` produce directories that
    nobody compares against a workspace. Canonicalizing them would be ceremony
    and not an invariant.

    The *candidate* side of a comparison is not canonical either:
    `FsService.resolve` passes an absolute path through as authored. A caller
    that compares against these roots therefore resolves the candidate itself
    where the answer would otherwise differ. See
    `permissions_fs.FsPermissions._outside_workspace`, which resolves only
    after the cheap comparison has already said "outside".

    This uses `realpath` and not `Path.resolve(strict=True)`. A tail that does
    not exist yet (a scratch directory that is about to be created) resolves
    through the part that does exist and keeps the rest. A path that is about
    to be `mkdir`ed needs that.
    """
    return Path(os.path.realpath(path))


def resolve_roots(*, create: bool = False) -> PathRoots:
    """Resolve all three roots, optionally creating them.

    The roots are canonical (`canonical`), so that everything produced under
    them carries one spelling: scratch directories, worktrees, the daemon and
    egress sockets.

    `$PH_HOME/daemon.yaml` may move the sessions and profiles directories
    (`ph.host`). It is read one time per process, so every caller here agrees.

    :raises RuntimeDirError: when an existing `$TMPDIR/ph` (tier 2) or
        `/tmp/ph-$UID` (tier 3) fails its check. pH refuses to start and does
        not adopt a directory that it cannot vouch for.
    :raises LoaderError: when `daemon.yaml` is malformed. The whole file is
        refused, for the reason that `load_host_config` gives.
    """
    runtime, tier, source = _resolve_runtime()
    home = canonical(_env_path("PH_HOME") or _default_home())
    moved = load_host_config(home).paths
    roots = PathRoots(
        home=home,
        cache=canonical(_env_path("PH_CACHE") or _default_cache()),
        runtime=canonical(runtime),
        runtime_tier=tier,
        runtime_source=source,
        sessions=_under_home(home, moved.sessions),
        profiles=_under_home(home, moved.profiles),
    )
    return roots.ensure() if create else roots


def _under_home(home: Path, configured: str | None) -> Path | None:
    """A path from `daemon.yaml`, made canonical. `~` is expanded, and a relative
    path is taken under `home`."""
    return canonical(home / Path(configured).expanduser()) if configured else None


def default_home_path(configured: str | None, name: str) -> Path:
    """The `path` setting of a row, or else `$PH_HOME/<name>`. Every seam shares this idiom.

    The result is canonical in both cases (`canonical`). A configured path is
    spelled the way the user typed it, and the roots that a seam produces under
    it must match what the kernel enforces.
    """
    if configured:
        return canonical(Path(configured).expanduser())
    return resolve_roots().home / name


def default_cache_path(configured: str | None, fallback: str) -> Path:
    """The `path` setting of a row, or else `$PH_CACHE/<fallback>`. The twin of
    `default_home_path`.

    The result is canonical in both cases. That is the reason this function
    exists and is not written again in each row. The two indexing rows arrived a
    week apart. One applied `canonical` to the spelling of the operator and the
    other did not. A configured path that was reached through a symlink then
    gave one of the rows two spellings for one index: every other reader
    resolved to a different spelling, and `phern doctor` printed the unresolved
    one. The docstring of `canonical` gives the argument.
    """
    if configured:
        return canonical(Path(configured).expanduser())
    return resolve_roots().cache / fallback


def file_stamp(path: Path) -> tuple[int, int, int] | None:
    """The `(inode, mtime, size)` of a file, or `None` when the file is absent.
    It tells a caller whether the file changed, without reading the file. The
    inode catches a file that `write_atomic` replaced. The mtime and the size
    catch a file that was appended to in place."""
    try:
        found = path.stat()
    except OSError:
        return None
    return found.st_ino, found.st_mtime_ns, found.st_size


def is_under(candidate: Path, root: Path) -> bool:
    """Whether `candidate` is `root` or inside it, both taken as given.

    This is a separator-aware prefix comparison and not `Path.relative_to`.
    `relative_to` allocates a `Path` for each segment of the root, and this
    function runs at every gated write. It applies `normcase` because
    `relative_to` folds case on Windows and a naive comparison would not. That
    is the one behavior of `relative_to` that is worth keeping.

    **Neither path is resolved.** A caller that needs symlink safety must
    `resolve()` *before* it asks. Resolving inside this function would answer a
    different question from the one that a security check means to ask. It
    would also silently pass a path whose parent is a link out of the tree.
    """
    base = os.path.normcase(root.as_posix()).rstrip("/")
    target = os.path.normcase(candidate.as_posix())
    return target == base or target.startswith(f"{base}/")


def free_sibling(path: Path, suffix: str) -> Path:
    """A name beside `path` that is not yet taken: `<name><suffix>`, then `-2`, `-3`, and so on.

    For a file or directory that is kept aside and not replaced, so that a
    second one never lands on the first. Examples: a folded drop-in directory
    (`phern profiles fold`), a log that a repair cut (`truncate_damaged`).
    """
    candidate = path.with_name(f"{path.name}{suffix}")
    count = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}{suffix}-{count}")
        count += 1
    return candidate


_MOUNT_ESCAPE = re.compile(r"\\([0-7]{3})")
"""How the kernel escapes a space, tab, newline or backslash in a mount table field."""


def filesystem_of(path: Path) -> str:
    """The type of the filesystem that holds `path`, from the mount table, or `unknown`.

    Linux keeps the table at `/proc/self/mountinfo`, with one mount on each
    line. The mount point is the fifth field, and the type is the first field
    after a lone `-`. The mount of `path` is the longest mount point that `path`
    is under. The path need not exist yet. It is resolved through the part that
    does exist (`canonical`), so a directory that is still to be made is on the
    filesystem that holds its nearest existing ancestor.
    """
    if not sys.platform.startswith("linux"):
        return "unknown"
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return "unknown"
    target = canonical(path)
    found = [(0, "unknown")]
    for line in lines:
        head, _, tail = line.partition(" - ")
        fields, after = head.split(), tail.split()
        if len(fields) < 5 or not after:
            continue
        point = _MOUNT_ESCAPE.sub(lambda match: chr(int(match[1], 8)), fields[4])
        if is_under(target, Path(point)):
            found.append((len(point.rstrip("/").split("/")), after[0]))
    return max(found, key=lambda one: one[0])[1]


def write_text_under(path: Path, text: str, *, append: bool = False) -> None:
    """Write (or append) text, creating the parent directory first.

    **This function is now for the append path** (O2). `append=False` truncates
    in place. For a whole document that another process reads without
    coordination, that leaves a window in which the file is neither the old one
    nor the new one. The JSON and YAML writers that used it moved to
    `write_atomic` below: the trust roots, the TUI settings, the theme profile
    and the sandbox profile drop-in. `tests/test_atomic_documents.py` is the
    gate that keeps a sixth from arriving. Truncating is still correct for a
    file that only this process reads, which is why the parameter stays.

    Every production caller that is left appends, so the gate reads as a rule
    about a parameter that nobody sets. A narrower signature would make the
    gate unnecessary. The docstring of the gate says why it is a gate.

    Blocking. Call it through `anyio.to_thread.run_sync` from async code.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.write(text)


_TEMP_SUFFIX_BYTES = 4


def write_atomic(
    path: Path,
    payload: bytes | str,
    *,
    skip_if_present: bool = False,
    durable: bool = True,
    preserve: bool = False,
    private: bool = False,
) -> None:
    """Write `payload` to `path` so that a reader sees all of it or none of it (L7),
    and so that what a reader saw survives a power loss (S6).

    **Temp-and-rename, because a torn file is worse than a missing file here.**
    Every caller writes something that another reader takes whole without
    coordination: a content-addressed blob whose name promises its sha256, a
    JSON index that is read on a daemon tick, a handle cache, a file of the
    user that an agent edited. `write_bytes` truncates and then writes, so an
    interrupted write leaves a prefix under the final name. For the
    content-addressed writers that prefix is *permanent*: the digest says that
    the file is already correct, and nothing ever rewrites it.

    **And durable: the file is `fsync`ed before the rename, and its directory
    after.** A rename alone is atomic against a reader but not against the
    machine. The rename can reach disk before the bytes that it names. The
    blobs written here are the ones that a session log points at (a spilled
    result, an attachment, a kernel variable), and that log *is* `fsync`ed at
    the next barrier. So a log could survive a power cut and name a file that
    came back empty. Every durable caller runs off the event loop or on the
    user's own save, where the syncs are not felt. `durable=False` is for a
    file that does not need to outlive a reboot, or that nothing reads back:
    the orphan journal in `$PH_RUNTIME`, a harness projection. It keeps the
    atomicity and skips both syncs.

    Six sites had derived temp-and-rename independently, with five spellings of
    the temp name. Two of the spellings were a fixed string, on which any two
    concurrent writers collide. Only one of the six sites removed the temp file
    when the write failed. The seventh site, the spill store
    (`SpillStore.try_save_all`), had not derived it at all.

    **A random suffix, not the pid**, because the colliding writers can be
    inside one process. `ph_rlm.harness.service` writes a projection for each
    session, and its docstring says that under the daemon two sessions project
    at the same time. `replace` is atomic within a filesystem, and the temp
    file is a sibling, so it is always on the same filesystem. The temp file is
    removed on a failure because a sweep can read identity from a name.
    `phern attachments gc` takes everything before the first `.`, so it would
    otherwise count an abandoned `<digest>.png.<hex>.tmp` as the blob, which
    it is not.

    `skip_if_present` is for the content-addressed callers, and only for them.
    Where the name *is* the sha256 of the bytes, a file that is already at that
    name already holds them. Rewriting it is one more chance to truncate
    something that a live reader holds, for no gain. **Unless it is the wrong
    size**: a file that was torn before this function synced (by a build that
    did not sync, or a disk that lied) is not the blob that its name promises,
    and skipping it would keep it that way. `skip_if_present` is not safe
    anywhere that the name does not promise the contents. **A skipped write
    still dates the file now** (its mtime): to anything that ages the file, the
    bytes were just written again. Without that, an old unreferenced blob that
    was attached again kept its old age, and `phern attachments gc` could take
    it before the message that names it reached disk.

    `preserve` is for replacing a file that belongs to somebody (`write` and
    `edit` of `FsService`, S5). A link is written *through*: the file that the
    link names is replaced, and not the link. The permission bits of the file
    survive, so an executable script stays executable. A hard link does not
    survive: its other names keep the old bytes, as they do after any editor
    that saves by rename. `preserve` is off by default, because a runtime
    writer or a blob writer must not follow a link that somebody planted at its
    path.

    `private` makes the file belong to its owner alone, 0600, from the moment
    it exists. The temp file is created with that mode, so there is no instant
    at which another account can read the bytes (the context file of ph-clm,
    which is a whole conversation).

    Parents are created at the default mode, and **durably when the file is
    durable** (`make_directories`, S20): a blob that is synced inside a
    directory whose own name was never synced is lost with the directory. A
    path under a directory whose mode matters (`$PH_RUNTIME` at 0700) is
    ensured by its owner first. `PathRoots.ensure_runtime()`, which
    `PathRoots.ensure()` calls, is still the one place that creates that
    directory.
    """
    written = _write_file(
        path,
        payload,
        skip_if_present=skip_if_present,
        durable=durable,
        preserve=preserve,
        private=private,
    )
    if written is not None and durable:
        sync_directory(written.parent)


_WRITERS = 8
"""At most this many files of one batch are written at the same time
(`write_atomic_all`). The file that the calling thread writes is one of them."""


def write_atomic_all(
    items: Iterable[tuple[Path, bytes]], *, skip_if_present: bool = False
) -> list[OSError | None]:
    """`write_atomic`, durable, for several files, with **each directory synced one time**.

    For a producer that names several files in one batch of records: the
    variables of a kernel cell, all under one owner (`SpillStore.try_save_all`).
    Each file is synced as `write_atomic` syncs it. The directories that the
    files went into are synced after the last file, one time each and not one
    time for each file. No rename is durable until this function returns, so
    nothing may name a file before it returns.

    **Several files are written at the same time** (at most `_WRITERS`), so
    their syncs share the commits of the disk and do not each wait for a commit
    of their own. Measured, this is two to four times faster for 5 to 20 files
    (R7, `plans/Persistence_Followups_Todo.md`). The calling thread writes the
    first file itself, and a pool that is made for the call writes the rest. A
    single file is written by the calling thread alone. Writers that make one
    new directory together are safe, because `make_directories` accepts a
    directory that is already made.

    A file that could not be written puts its error at its position in the
    result, and the other files are still written. Every other position holds
    `None`. A file that `skip_if_present` left alone made no directory dirty
    and costs no sync.
    """
    batch = list(items)

    def write(item: tuple[Path, bytes]) -> Path | OSError | None:
        path, payload = item
        try:
            return _write_file(path, payload, skip_if_present=skip_if_present)
        except OSError as error:
            return error

    if len(batch) < 2:
        written = list(map(write, batch))
    else:
        pool = ThreadPoolExecutor(_WRITERS - 1, "ph-write")
        try:
            rest = pool.map(write, batch[1:])
            written = [write(batch[0]), *rest]
        finally:
            # Every result is in hand at this point, so the threads only have to
            # exit. Waiting for them was most of what the pool cost a small batch.
            pool.shutdown(wait=False)
    for directory in dict.fromkeys(one.parent for one in written if isinstance(one, Path)):
        sync_directory(directory)
    return [one if isinstance(one, OSError) else None for one in written]


def _write_file(
    path: Path,
    payload: bytes | str,
    *,
    skip_if_present: bool,
    durable: bool = True,
    preserve: bool = False,
    private: bool = False,
) -> Path | None:
    """`write_atomic` without the sync of the directory, which the caller does.
    Returns the path that was written, which is the target of a link under
    `preserve`. Returns `None` when `skip_if_present` left the file alone."""
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    mode: int | None = None
    if preserve:
        path = canonical(path)
        with suppress(FileNotFoundError):
            mode = stat.S_IMODE(path.stat().st_mode)
    if skip_if_present and _holds(path, len(data)):
        with suppress(OSError):
            os.utime(path)
        return None
    if durable:
        make_directories(path.parent)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{secrets.token_hex(_TEMP_SUFFIX_BYTES)}.tmp")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o666)
        try:
            write_all(fd, data)
            if mode is not None:
                os.fchmod(fd, mode)
            if durable:
                os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def replace_durably(source: Path, target: Path) -> None:
    """`os.replace`, with the rename itself made durable: the directory of `target`
    is synced after it. For a file that is already durable under another name,
    when its new name is what a reader points at."""
    os.replace(source, target)
    sync_directory(target.parent)


def write_all(fd: int, payload: bytes) -> None:
    """Write every byte, however many calls that takes. A short write is not an error."""
    view = memoryview(payload)
    while view:
        view = view[os.write(fd, view) :]


def sync_directory(directory: Path) -> None:
    """Make the entries in `directory` durable: a name that was just created in it
    or renamed into it.

    Best effort. Some networked filesystems refuse a directory handle. Such a
    filesystem has already given the file its own durability, and that is all
    that it can give.
    """
    with suppress(OSError):
        handle = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def make_directories(directory: Path) -> None:
    """Create `directory` and every ancestor that is missing, with **each new name
    durable in its parent** (S20).

    `mkdir(parents=True)` leaves the name of a new directory only in the cache
    of its parent. A power cut can then take the directory back, with every
    file that was synced inside it: a log that was flushed as durable is gone
    with its family directory. Each directory that this function makes has its
    parent synced, from the top down. Blocking. It pays only for the
    directories that it makes.
    """
    missing: list[Path] = []
    current = directory
    while not current.is_dir():
        missing.append(current)
        current = current.parent
    for one in reversed(missing):
        with suppress(FileExistsError):
            one.mkdir()
        sync_directory(one.parent)


def _holds(path: Path, size: int) -> bool:
    """Whether `path` is a regular file of exactly `size` bytes. For a
    content-addressed name, this says whether the file already holds the bytes
    that it names."""
    try:
        found = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(found.st_mode) and found.st_size == size
