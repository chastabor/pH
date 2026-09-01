"""Whether a daemon outlives the login session that started it (I-6, P5-11).

`$PH_RUNTIME` resolves to `$XDG_RUNTIME_DIR/ph` on Linux, and tier 1's whole
argument in `paths.py` is that **the OS owns that directory's properties** —
mode, owner, tmpfs, and *removal at session end*. Every one of those is a
reason to prefer it, and the last one is what Phase 5 walks into.

`ph daemon` exists so a run that takes an hour does not stop because a laptop
lid closed. But logind reaps `/run/user/$UID` when a user's last session ends
unless that user is *lingering*, so a daemon started from an ordinary login
session meets one of two ends at logout, and `loginctl enable-linger` is the
one fix for both:

* `KillUserProcesses=no` (the Debian and Ubuntu default) — the daemon keeps
  running and **loses its socket**. Every client afterwards reports "no daemon
  socket at …" and is told to start one, while the first daemon is still
  holding this user's session leases (I-5) and will refuse the second every
  root it already owns. Nothing anywhere says the word "logout".
* `KillUserProcesses=yes` (systemd's own default) — the daemon is signalled at
  logout, which is at least a teardown that runs, but is not what "long
  running" promised either.

That first case is the silent failure this module exists to make loud, and it
has two readers. `ph doctor` and `ph daemon` ask *before*: is this deployment
one where the socket will survive? The daemon asks *after*, on a cadence, of
its own socket — `socket_identity` — because by then no new client can connect
to be told anything, and the only surfaces left are the daemon's log and the
roots' own transcripts.

**The linger state is read from the marker directory, not from `loginctl`.**
The row asked for "`loginctl` state" and this is it: `enable-linger` creates an
empty `/var/lib/systemd/linger/<user>`, `disable-linger` removes it, and
logind's own check at boot is the presence of that file — so the directory is
the state rather than a cache of it. Reading it directly is a `stat` instead of
a subprocess and a D-Bus round trip on a command a person runs *because*
something is already wrong; it needs no `ctx.subprocess` seam, which a
`Diagnostic.read` (synchronous, by construction) has no way to await; and it
answers on a host where `loginctl` is absent from `PATH` or logind is not
running, which is exactly the host whose answer is least obvious. What is lost
is a host that stores the marker somewhere else — reported as `unknown` rather
than guessed at, with `loginctl show-user` named as the way to settle it.

@module ph.lingering
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .paths import PathRoots, RuntimeTier, is_under, resolve_roots

__all__ = [
    "LINGER_DIR",
    "LingerState",
    "RuntimeLifetime",
    "lifetime",
    "linger_state",
    "socket_identity",
    "username",
]

LINGER_DIR = Path("/var/lib/systemd/linger")
"""Where logind records the users it keeps a manager running for.

An empty file per lingering user, named by login name. `loginctl enable-linger`
creates one and `disable-linger` removes it; a module constant so a test can
point the probe at a directory it controls, since the real one is root-owned
and a test that needed to write to it could not run."""

LingerState = Literal["on", "off", "unknown", "not-applicable"]
"""`on`/`off` when the marker directory settles it, `unknown` when the
directory is missing (no logind, or a host that keeps the state elsewhere), and
`not-applicable` off Linux, where nothing reaps a runtime directory at logout
because nothing created one."""


def username() -> str:
    """This process's login name, for the marker file and the advice.

    The marker is named by login name rather than uid, so a numeric fallback
    would name a file that cannot exist. `$USER` is consulted first because it
    is what the person will type after `enable-linger`, and `pwd` behind it for
    a process started without an environment — a cron job, or the daemon's own
    re-exec.
    """
    name = os.environ.get("USER") or os.environ.get("LOGNAME")
    if name:
        return name
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_name
    except Exception:  # pragma: no cover - no pwd (Windows) or no passwd entry
        return ""


def linger_state(user: str = "") -> LingerState:
    """Whether logind keeps this user's processes and runtime dir after logout.

    Never raises: this is read by `ph doctor`, which a person runs *because*
    something is broken, and an unreadable `/var/lib/systemd` is a fact to
    report rather than a traceback to replace the rest of the report with.
    """
    if sys.platform != "linux":
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


@dataclass(frozen=True, slots=True)
class RuntimeLifetime:
    """What happens to a path under `$PH_RUNTIME` when this login session ends."""

    path: Path
    """The path whose lifetime was asked about — the runtime root, or a
    daemon's own socket when a daemon is the one asking."""
    tier: RuntimeTier
    derived_from: str
    """The environment variable behind the tier, `""` for the `/tmp` fallback.

    `PathRoots.runtime_source`, copied at construction. It was a `{tier: variable}`
    table in this module until the review that found it naming `LOCALAPPDATA` for
    a windows tier whose path can come from `$XDG_CACHE_HOME` or `~/.cache` — a
    second copy of a decision `paths._resolve_runtime` makes while reading the
    variable, so the copy could only ever drift or raise `KeyError` inside the one
    command a person runs because something is already broken."""
    runtime_dir: str
    """`$XDG_RUNTIME_DIR` as the environment gives it, or `""` when unset.

    Carried rather than re-read where it is printed: this is the directory the
    verdict is *about*, and a sentence naming `path.parent` would name
    `/run/user/1000/ph` for a runtime root and `/run/user/1000/ph` again for a
    socket inside it — neither of which is the thing logind removes."""
    linger: LingerState
    user: str

    @property
    def reaped_at_logout(self) -> bool:
        """Whether `path` sits inside `$XDG_RUNTIME_DIR`, which logind removes.

        A containment test rather than `tier == "xdg-runtime"`, because an
        explicit `PH_RUNTIME=/run/user/1000/ph-dev` is tier `override` and
        reaped exactly the same — the tier says where the answer came from, not
        where it points.
        """
        return bool(self.runtime_dir) and is_under(self.path, Path(self.runtime_dir))

    @property
    def survives_logout(self) -> bool | None:
        """`None` where the honest answer is "it depends".

        Three states rather than two, because the failures in the module
        docstring have different certainties, and `_outlook` is where the four
        real cases are enumerated — this is the answer column of that table.
        """
        return self._outlook()[0]

    @property
    def advice(self) -> str:
        """The command that settles it, or `""` when there is nothing to settle.

        One command for both failure shapes, which is the reason a row here can
        end in an instruction rather than in a discussion: lingering keeps the
        runtime directory *and* exempts the user from `KillUserProcesses`. The
        `unknown` host gets a question instead — `show-user` — because "off"
        earns an instruction and "we could not tell" does not.
        """
        return self._outlook()[2]

    @property
    def explanation(self) -> str:
        """Why the answer is what it is, with no answer in front of it.

        Split from `verdict` because it has a reader that is not answering that
        question. `ph agents` reaches this when a *connect* failed, where the
        thing on screen above it is "no daemon socket at …" — and a next line
        beginning "no —" reads as a refusal of something nobody asked.
        """
        return self._outlook()[1]

    def _outlook(self) -> tuple[bool | None, str, str]:
        """`(survives, why, what to run)` — the four cases, decided in one place.

        One `match` over the two facts that carry the information, rather than three
        ladders re-walking a tri-state that had already thrown some away. **Unknown
        linger and outside-the-reaped-tree are separate cases**, not one `None`: a host
        with no `/var/lib/systemd/linger` and a socket inside `$XDG_RUNTIME_DIR` — exactly
        the host this module says deserves the question — must not be told "the socket
        outlives logout", which is the one thing nobody could know about it.

        The tuple is returned whole because an answer, its sentence and its command are
        three views of one decision, and deciding them separately is how they came to
        disagree.
        """
        match (self.linger, self.reaped_at_logout):
            case ("not-applicable", _):
                return True, "nothing on this platform reaps a runtime directory at logout", ""
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

        `ph daemon` prints it at startup — one line, no table, before it blocks —
        and taking it from here rather than writing a second sentence is what
        keeps the warning a person reads at 09:00 and the report they run at
        17:00 from disagreeing about what is wrong.
        """
        answer = {True: "yes", False: "no", None: "unknown"}[self.survives_logout]
        return f"{answer} — {self.explanation}"

    def describe(self) -> list[tuple[str, str]]:
        """The rows `ph doctor` and `ph agents doctor` print.

        `(label, value)` pairs, which is `PathRoots.describe()`'s shape and
        `ctx.diagnostics`' — so this reads as one more section rather than as a
        format either printer has to learn. Rows with nothing to say are left
        out for the reason `Diagnostic.read` returns none: a report that always
        prints every row is one where the row that matters cannot be found.
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

    `path` defaults to the runtime root, which is what `ph doctor` asks about
    before any daemon exists. A daemon passes **its own socket**: `serve()`
    accepts an explicit path, so the socket a daemon bound and the root
    `resolve_roots()` derives are not always the same file, and a daemon that
    reported the lifetime of a directory it is not using would be the
    re-derivation on the client side that `daemon/status` exists to avoid.
    """
    resolved = roots or resolve_roots()
    # Once, and handed to the marker lookup rather than left to it: `linger_state`
    # falls back to `username()` when it is not told, so reading the field
    # separately probed the same fact twice — `$USER` twice on an ordinary shell,
    # and two `pwd.getpwuid` calls on the deployment this module is written about
    # (a unit file, a re-exec, cron), where the passwd backend may be an NSS round
    # trip. It also makes "the linger answer is about `user`" true by construction
    # rather than by coincidence.
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
    """`(st_dev, st_ino)` for whatever is at `path`, or `None` if nothing is.

    The pair rather than "does the file still exist", because the daemon's
    socket has **two** ways of stopping being the daemon's socket and only one
    of them is an absence. Logout reaps the directory, which is the removal;
    then the person logs back in, logind makes `/run/user/$UID` again, and
    `ph daemon` — following the advice a client just printed — binds a *new*
    socket at the same path. From then on the path exists and answers, while
    the first daemon still holds every lease the second one will be refused.
    An existence check calls that recovery. The inode says it is the worse
    half of I-5's hazard arriving by a different door.

    `lstat`, so a path replaced by a symlink to somebody else's socket reads as
    replaced rather than as the target it points at.

    **What makes the pair sound is that the daemon holds its listener open.**
    An inode number is reusable the moment the last reference to it goes, so a
    path unlinked and then re-created can legitimately come back with the number
    it just had — which is not a theoretical hazard, it is what CI does and what
    caught this module's first tests. The bound socket is different: `serve()`
    keeps the listener inside `async with listener:` for the life of the
    process, so the inode stays referenced while the daemon runs and cannot be
    handed to the socket that replaces it. A caller comparing identities for a
    path it does *not* hold open has no such guarantee and should not read this
    as one.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return (info.st_dev, info.st_ino)
