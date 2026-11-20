"""Whether a daemon outlives the login session that started it (I-6, P5-11).

`$PH_RUNTIME` resolves to `$XDG_RUNTIME_DIR/ph` on Linux. The argument for
tier 1 in `paths.py` is that **the OS owns the properties of that directory**:
mode, owner, tmpfs, and *removal at session end*. Each of those is a reason to
prefer it, and the last one is the problem that Phase 5 meets.

`phern daemon` exists so that a run that takes an hour does not stop because a
laptop lid closed. But logind reaps `/run/user/$UID` when the last session of a
user ends, unless that user is *lingering*. So a daemon that was started from
an ordinary login session meets one of two ends at logout, and
`loginctl enable-linger` is the one fix for both:

* `KillUserProcesses=no` (the Debian and Ubuntu default). The daemon keeps
  running and **loses its socket**. Every client afterwards reports "no daemon
  socket at …" and is told to start one. The first daemon still holds the
  session leases of this user (I-5), and it will refuse the second daemon
  every root that it already owns. Nothing anywhere says the word "logout".
* `KillUserProcesses=yes` (the default of systemd itself). The daemon is
  signaled at logout. That is at least a teardown that runs, but it is not
  what "long running" promised either.

The first case is the silent failure that this module exists to make loud. It
has two readers:

* `phern doctor` and `phern daemon` ask *before*: is this a deployment where
  the socket will survive?
* The daemon asks *after*, about its own socket (`socket_identity`), whenever
  the kernel says that the entry of the socket moved (`ph.path_watch`). By
  then no new client can connect to be told anything. The only surfaces left
  are the log of the daemon and the transcripts of the roots.

**The linger state is read from the marker directory, not from `loginctl`.**
The row asked for "`loginctl` state", and this is that state. `enable-linger`
creates an empty `/var/lib/systemd/linger/<user>` and `disable-linger` removes
it. The check that logind itself makes at boot is the presence of that file.
So the directory is the state and not a cache of it. Reading it directly has
three benefits:

* It is a `stat`, not a subprocess and a D-Bus round trip, on a command that a
  user runs *because* something is already wrong.
* It needs no `ctx.subprocess` seam. A `Diagnostic.read` is synchronous by
  construction and has no way to await one.
* It answers on a host where `loginctl` is absent from `PATH` or logind is not
  running. That is exactly the host whose answer is least obvious.

What is lost is a host that stores the marker somewhere else. That host is
reported as `unknown` and not guessed at, and `loginctl show-user` is named as
the way to settle it.

**Whether logind is here at all is read the same way, from its own
directory.** It was `sys.platform == "linux"` before, and that answered the
wrong question in two ways. A Linux host without systemd (Alpine, a slim
container) was told to check with a `loginctl` that it does not have. And
every test that set up a simulated logind host on a Mac, marker directory
included, was answered `not-applicable` before the marker was looked at. So
twelve of them failed on macOS for no reason that the code under test had.

The evidence is `SYSTEMD_RUN_DIR`, which is the `sd_booted()` test of systemd
itself. **Booted, not installed**, and the difference is the purpose.
`/var/lib/systemd` is created by `systemd-timesyncd` and similar services. It
survives on a host that was booted under sysvinit, and in a Debian container
where PID 1 is not systemd. Keying off it would give exactly the `loginctl`
advice that this change removes to a narrower set of hosts, and not to none.
`/run/systemd/system` exists only where systemd is the init that booted the
machine, and that is the only case where logind reaps anything.

@module ph.lingering
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .paths import PathRoots, RuntimeTier, canonical, is_under, resolve_roots

__all__ = [
    "LINGER_DIR",
    "SYSTEMD_RUN_DIR",
    "LingerState",
    "RuntimeLifetime",
    "lifetime",
    "linger_state",
    "logind_present",
    "socket_identity",
    "username",
]

SYSTEMD_RUN_DIR = Path("/run/systemd/system")
"""The marker of systemd for "am I the init that booted this host" (`sd_booted(3)`).

Systemd creates this directory at boot and nothing else does. So its presence
is the one cheap answer to "will logind reap a runtime directory here". It is
its **own** constant and not a fact inferred from `LINGER_DIR`. The linger
marker directory answers whether a *user* lingers. Reading a second meaning out
of it made one path carry two questions. It also made the "no logind" branch
impossible to test: a test that pointed `LINGER_DIR` at its own temp directory
necessarily gave that directory a parent that exists. It is a module constant
for the same reason as `LINGER_DIR`: a test needs to point it at a directory
that the test controls."""

LINGER_DIR = Path("/var/lib/systemd/linger")
"""Where logind records the users that it keeps a manager running for.

There is one empty file for each lingering user, named by login name.
`loginctl enable-linger` creates the file and `disable-linger` removes it. This
is a module constant so that a test can point the probe at a directory that the
test controls. The real directory is owned by root, and a test that needed to
write to it could not run."""

LingerState = Literal["on", "off", "unknown", "not-applicable"]
"""`on` or `off` when the marker directory settles it. `unknown` when systemd
booted the host but the marker directory is not where logind keeps it (a host
that stores the state elsewhere). `not-applicable` where systemd did not boot
the host (macOS, Windows, a Linux without it). Nothing there reaps a runtime
directory at logout, because nothing created one."""


def username() -> str:
    """The login name of this process, for the marker file and the advice.

    The marker is named by login name and not by uid, so a numeric fallback
    would name a file that cannot exist. `$USER` is read first because it is
    what the user will type after `enable-linger`. `pwd` is behind it for a
    process that was started without an environment: a cron job, or the
    re-exec of the daemon itself.
    """
    name = os.environ.get("USER") or os.environ.get("LOGNAME")
    if name:
        return name
    try:
        # POSIX-only, and its absence is handled below.
        import pwd  # noqa: PLC0415

        return pwd.getpwuid(os.getuid()).pw_name
    except Exception:  # pragma: no cover - no pwd (Windows) or no passwd entry
        return ""


def linger_state(user: str = "") -> LingerState:
    """Whether logind keeps the processes and the runtime directory of this user
    after logout.

    Never raises. `phern doctor` reads this, and a user runs `phern doctor`
    *because* something is broken. An unreadable `/var/lib/systemd` is a fact
    to report, and not a traceback that replaces the rest of the report.
    """
    if not logind_present():
        return "not-applicable"
    name = user or username()
    if not name:
        return "unknown"
    try:
        if not LINGER_DIR.is_dir():
            return "unknown"
        return "on" if (LINGER_DIR / name).exists() else "off"
    except OSError:
        return "unknown"


def logind_present() -> bool:
    """Whether this host has a logind that can reap anything. This is read from
    disk, not from the platform name.

    One `stat` of `SYSTEMD_RUN_DIR`, which is what `sd_booted(3)` does. Never
    raises, for the same reason as `linger_state`: `phern doctor` reads this.

    The three answers stay distinct downstream, and each gets a different
    sentence:

    - No logind is `not-applicable` and advises nothing.
    - Logind with no marker directory is `unknown` and asks a question.
    - A marker directory settles the answer as `on` or `off`. When it is `off`,
      the sentence gives an instruction.
    """
    try:
        return SYSTEMD_RUN_DIR.is_dir()
    except OSError:
        return False


@dataclass(frozen=True, slots=True)
class RuntimeLifetime:
    """What happens to a path under `$PH_RUNTIME` when this login session ends."""

    path: Path
    """The path whose lifetime was asked about: the runtime root, or the socket
    of a daemon when a daemon is the one that asks."""
    tier: RuntimeTier
    derived_from: str
    """The environment variable behind the tier, `""` for the `/tmp` fallback.

    This is `PathRoots.runtime_source`, copied at construction. It was a
    `{tier: variable}` table in this module until a review found that the table
    named `LOCALAPPDATA` for a windows tier whose path can come from
    `$XDG_CACHE_HOME` or `~/.cache`. The table was a second copy of a decision
    that `paths._resolve_runtime` makes while it reads the variable. So the
    copy could only drift, or raise `KeyError` inside the one command that a
    user runs because something is already broken."""
    runtime_dir: str
    """`$XDG_RUNTIME_DIR` as the environment gives it, or `""` when it is unset.

    It is carried here and not read again where it is printed, because this is
    the directory that the verdict is *about*. A sentence naming `path.parent`
    would name `/run/user/1000/ph` for a runtime root and `/run/user/1000/ph`
    again for a socket inside it. Neither of those is the thing that logind
    removes."""
    linger: LingerState
    user: str

    @property
    def reaped_at_logout(self) -> bool:
        """Whether `path` is inside `$XDG_RUNTIME_DIR`, which logind removes.

        This is a containment test and not `tier == "xdg-runtime"`. An explicit
        `PH_RUNTIME=/run/user/1000/ph-dev` is tier `override` and is reaped in
        exactly the same way. The tier says where the answer came from, not
        where it points.
        """
        # Both sides are canonical. `path` comes from `resolve_roots`, which
        # resolves. Comparing it against a raw `$XDG_RUNTIME_DIR` would answer
        # `False` on any host that reaches the directory through a link
        # (`/var/run/user/N` where `/var/run` is `/run`). That is the same
        # spelling drift that the roots were made canonical to end.
        return bool(self.runtime_dir) and is_under(self.path, canonical(Path(self.runtime_dir)))

    @property
    def survives_logout(self) -> bool | None:
        """`None` where the correct answer is "it depends".

        There are three states and not two, because the failures in the module
        docstring have different certainties. `_outlook` lists the six real
        cases, and this property is the answer column of that table.
        """
        return self._outlook()[0]

    @property
    def advice(self) -> str:
        """The command that settles it, or `""` when there is nothing to settle.

        One command covers both failure shapes, which is why a row here can end
        in an instruction and not in a discussion: lingering keeps the runtime
        directory *and* exempts the user from `KillUserProcesses`. The `unknown`
        host gets a question (`show-user`) and not an instruction, because
        "off" justifies an instruction and "we could not tell" does not.
        """
        return self._outlook()[2]

    @property
    def explanation(self) -> str:
        """Why the answer is what it is, without the answer in front of it.

        This is split from `verdict` because one of its readers is not
        answering that question. `phern agents` reaches this when a *connect*
        failed, and the line on screen above it is "no daemon socket at …". A
        next line that begins "no —" would read as a refusal of something that
        nobody asked.
        """
        return self._outlook()[1]

    def _outlook(self) -> tuple[bool | None, str, str]:
        """`(survives, why, what to run)`: the six cases, decided in one place.

        One `match` over the two facts that carry the information. Before, three
        ladders each walked a tri-state that had already discarded some of it.
        **Unknown linger and outside-the-reaped-tree are separate cases**, not
        one `None`. A host with no `/var/lib/systemd/linger` and a socket inside
        `$XDG_RUNTIME_DIR` is exactly the host that this module says deserves
        the question. It must not be told "the socket outlives logout", which
        is the one thing that nobody could know about it.

        The tuple is returned whole because an answer, its sentence and its
        command are three views of one decision. Deciding them separately is how
        they came to disagree.
        """
        match (self.linger, self.reaped_at_logout):
            case ("not-applicable", _):
                return (
                    True,
                    "no logind on this host, so nothing reaps a runtime directory at logout",
                    "",
                )
            case ("on", _):
                return (
                    True,
                    f"lingering is on for {self.user}, so logind keeps this user's "
                    "runtime directory and processes after the last session ends",
                    "",
                )
            case ("off", True):
                return (
                    False,
                    f"logind removes {self.runtime_dir} when your last session ends, "
                    "and a daemon that keeps running loses its socket with it",
                    f"loginctl enable-linger {self.user}",
                )
            case ("off", False):
                return (
                    None,
                    "the socket outlives logout, but whether the process does "
                    "depends on logind's KillUserProcesses",
                    f"loginctl enable-linger {self.user}",
                )
            case (_, True):
                return (
                    None,
                    f"{self.path} is inside {self.runtime_dir}, which logind removes at "
                    "logout for a user who is not lingering — and this host does not say "
                    "whether this user is",
                    f"loginctl show-user {self.user} --property=Linger",
                )
            case _:
                return (
                    None,
                    "the socket outlives logout, and this host does not say whether "
                    "logind keeps the process",
                    f"loginctl show-user {self.user} --property=Linger",
                )

    def verdict(self) -> str:
        """The `survives logout` row: the answer, then why.

        `phern daemon` prints it at startup when the answer is not yes: one
        line, no table, before it blocks. It takes the sentence from here and
        does not write a second one. That keeps the warning that a user reads
        at 09:00 and the report that the user runs at 17:00 from disagreeing
        about what is wrong.
        """
        answer = {True: "yes", False: "no", None: "unknown"}[self.survives_logout]
        return f"{answer} — {self.explanation}"

    def describe(self) -> list[tuple[str, str]]:
        """The rows that `phern doctor` and `phern agents doctor` print.

        These are `(label, value)` pairs, which is the shape of
        `PathRoots.describe()` and of `ctx.diagnostics`. So this reads as one
        more section, and not as a format that either printer has to learn.
        Rows with nothing to say are left out, for the reason that
        `Diagnostic.read` returns none: in a report that always prints every
        row, the row that matters cannot be found.
        """
        source = f"${self.derived_from}" if self.derived_from else "no runtime dir in the env"
        rows = [
            ("derivation", f"{source} → {self.path}  (tier: {self.tier})"),
            ("survives logout", self.verdict()),
        ]
        if self.linger != "not-applicable":
            rows.append(("linger", self._linger_row()))
        if self.advice:
            rows.append(("enable it" if self.linger != "unknown" else "check it", self.advice))
        return rows

    def _linger_row(self) -> str:
        marker = LINGER_DIR / self.user if self.user else LINGER_DIR
        match self.linger:
            case "on":
                return f"on for {self.user} ({marker})"
            case "off":
                return f"off for {self.user} ({marker} is absent)"
            case _:
                return f"unknown — {LINGER_DIR} does not exist on this host"


def lifetime(path: Path | None = None, *, roots: PathRoots | None = None) -> RuntimeLifetime:
    """Read the environment and answer for one path. Creates nothing.

    `path` defaults to the runtime root, which is what `phern doctor` asks about
    before any daemon exists. A daemon passes **its own socket**. `serve()`
    accepts an explicit path, so the socket that a daemon bound and the root
    that `resolve_roots()` derives are not always the same file. A daemon that
    reported the lifetime of a directory that it does not use would be the
    re-derivation on the client side that `daemon/status` exists to avoid.
    """
    resolved = roots or resolve_roots()
    # The name is read one time and handed to the marker lookup. `linger_state`
    # falls back to `username()` when it is not given a name, so reading the
    # field separately probed the same fact twice: `$USER` twice on an ordinary
    # shell, and two `pwd.getpwuid` calls on the deployment that this module is
    # written about (a unit file, a re-exec, cron), where the passwd backend may
    # be an NSS round trip. It also makes "the linger answer is about `user`"
    # true by construction and not by coincidence.
    user = username()
    return RuntimeLifetime(
        path=path or resolved.runtime,
        tier=resolved.runtime_tier,
        derived_from=resolved.runtime_source,
        runtime_dir=os.environ.get("XDG_RUNTIME_DIR") or "",
        linger=linger_state(user),
        user=user,
    )


def socket_identity(path: Path) -> tuple[int, int] | None:
    """`(st_dev, st_ino)` for whatever is at `path`, or `None` if nothing is there.

    This returns the pair and not "does the file still exist", because the
    socket of the daemon has **two** ways to stop being the socket of the
    daemon, and only one of them is an absence. Logout reaps the directory,
    which is the removal. Then the user logs back in, logind makes
    `/run/user/$UID` again, and `phern daemon` binds a *new* socket at the
    same path, following the advice that a client just printed. From then on
    the path exists and answers, while the first daemon still holds every lease
    that the second one will be refused. An existence check calls that a
    recovery. The inode shows that it is the worse half of the hazard of I-5,
    arriving by a different route.

    `lstat`, so a path that was replaced by a symlink to the socket of somebody
    else reads as replaced, and not as the target that it points at.

    **The pair is sound because the daemon holds its listener open.** An inode
    number can be reused the moment the last reference to it goes. So a path
    that is unlinked and then created again can legitimately come back with
    the number that it just had. That is not a theoretical hazard: it is what
    CI does, and it is what caught the first tests of this module. The bound
    socket is different. `serve()` keeps the listener inside
    `async with listener:` for the life of the process, so the inode stays
    referenced while the daemon runs and cannot be given to the socket that
    replaces it. A caller that compares identities for a path that it does
    *not* hold open has no such guarantee and should not read this as one.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return (info.st_dev, info.st_ino)
