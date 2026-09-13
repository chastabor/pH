"""Shared pytest configuration.

Every async test runs on asyncio, because Textual requires it (D3) and a
harness that passed its tests on a backend it never ships on would be proving
the wrong thing.

`mount` is the one way a test stands up a profile: base + headless plus any
overlay rows, on a fresh root context that is disposed on teardown. Tests that
took a root of their own and remembered to dispose it were each re-deriving
`ph_app.runtime.mounted`.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from ph.bundles import BASE, HEADLESS
from ph.cordis import Context, Profile, load_profile_documents
from ph.cordis.loader import compose_rows
from ph.testing import ReapedHost

pytest_plugins = ["app_fixtures", "rlm_fixtures"]
"""The two per-package fixture sets, registered from the **one** conftest.

They were `packages/ph-app/tests/conftest.py` and
`packages/ph-rlm/tests/conftest.py`, which is what kept the test trees out
of mypy: two modules named `conftest` are a duplicate mypy refuses, and no
flag fixes it because the unique name it would need —
`packages.ph-app.tests.conftest` — is not an identifier (issue 32).

Registering them here rather than merging their bodies into this file keeps
each set beside the tests it serves, and `pythonpath` in `pyproject.toml` is
what makes them importable by name at startup. It also fixed a latent bug
the trees had documented in five separate comments: `from conftest import
ROW` resolved to whichever conftest won the name under full collection, so
ph-rlm's row constants were reachable from ph-app's tree and vice versa.
`pytest_plugins` is only honoured in a *root* conftest, which is the other
reason the registration lives here."""

MountProfile = Callable[..., Awaitable[Context]]

NEEDS_BINARY = {
    "needs_git": ("git", "the worktree tier needs git"),
    "needs_jj": ("jj", "the jj tier needs jj"),
}
"""The markers that mean "this test drives a real binary", and what it needs.

Registered in `pyproject.toml`; skipped below. Here rather than beside the
fixtures they go with, because `ph.testing.git` and `ph.testing.jj` — where
those fixtures live — ship inside the `ph-core` wheel, and a `pytest.mark`
constant there is an `import pytest` on the import path of three plugin rows a
deployment mounts (`llm-fake` above all, which `headless` carries). An install
without pytest could not compose its default profile. A marker costs the wheel
nothing, and the collection hook is a place the wheel does not reach."""


STRAY_CALLBACK_NOTE = """Raised by a callback the event loop ran *outside* any test.

anyio's `TestRunner` collects those and re-raises them from whichever test next
finishes, and every test here shares one session-wide loop — so **the test this
failed is very likely not the cause**. The `context:` line above names the
callback, which the traceback cannot: it arrives with no application frames at
all, because by then nothing of ours is on the stack.

**If `context:` names `Future.set_result(None)`, this is issue 58** and the
mechanism is known: anyio registers that bound method as an `add_reader`
callback and removes the reader in a *done-callback* one loop iteration later,
so a wait cancelled in the gap can still be fired. Five copies of that shape —
`_RawSocketMixin._wait_until_readable` and `_wait_until_writable`,
`UNIXSocketListener.accept`, and inline in `AsyncIOBackend.connect_unix` and
`create_unix_datagram_socket` — so the exposure is the **daemon** socket:
connecting, accepting, and every `Peer` read. Note the last three repeat the
shape *inline* rather than inheriting the mixin's, which is what a first
attempt at the guard got wrong. It is *not* `ph_rlm.kernel.manager._recv_line`,
which this note used to name: that calls the free `anyio.wait_readable`, a
different implementation that catches `InvalidStateError` for itself. See issue
58 in `plans/Implementation_Plan.md` — no re-investigation needed.

**And it is a *test* failure, not a production one.** The same race in a
daemon reaches asyncio's default exception handler: one ERROR on the `asyncio`
logger, and the process carries on — nothing propagates, and the read it fires
on was already being abandoned by the cancellation. It fails a test only
because anyio's `TestRunner` collects loop-callback exceptions and re-raises
them on the shared session loop. So this is noise to be traced, not a fault to
be chased: measured in issue 58.

**And it should no longer be reachable at all.** `ph_app.daemon.cancelsafe`
guards the event loop against exactly this registration, applied on import of
`ph_app.daemon`. If a `Future.set_result(None)` handle still appears here, the
guard has stopped binding — `test_cancelsafe` is what should have caught that,
so check it before looking anywhere else.

Any *other* callback is a new one, and the name above is the lead."""


def pytest_configure(config: pytest.Config) -> None:
    """Make a stray event-loop callback name its own cause.

    **The flake's worst property is misattribution, and that is what this
    fixes.** A callback that raises after its test has finished is collected by
    anyio's session-wide `TestRunner` and re-raised inside an unrelated test,
    with a traceback holding no application frames — `self = None`, one line of
    `asyncio/events.py`, nothing else. Twice now that has sent someone
    investigating a test that was a bystander (issue 58, and the earlier
    instance `daemon_helpers.close_clients` was written for).

    So the exception gets a note carrying what the traceback lacks: which test
    was running when the callback fired, and the callback itself. The failure is
    left failing — a stray callback is a real defect and silencing it would be
    the wrong trade — but it now points somewhere useful.

    Wrapping anyio's handler rather than `loop.set_exception_handler`, because
    the runner installs its own on every loop it makes and would overwrite ours.

    **A heavier companion to this was tried and removed.** While issue 58's
    callback was unknown, a wrapper on `asyncio.BaseEventLoop.call_soon`
    recorded any `set_result` scheduled onto an already-resolved future,
    schedule-time stack and all — and it is what identified the callback. That
    made it a *hunting* tool: it monkeypatched stdlib for the whole test
    process, and once the answer was known it was buying a second diagnosis of
    a diagnosed bug. If a new stray ever needs the same treatment, it is worth
    re-adding for the hunt and removing again after; leaving it armed is the
    part that was not worth it.
    """
    from anyio._backends._asyncio import TestRunner

    original = TestRunner._exception_handler

    def attributing(runner: Any, loop: Any, context: dict[str, Any]) -> Any:  # noqa: ANN401
        error = context.get("exception")
        if isinstance(error, BaseException):
            named = ", ".join(
                f"{key}={context[key]!r}"
                for key in ("message", "handle", "future", "task", "protocol", "transport")
                if context.get(key) is not None
            )
            error.add_note(f"context: {named or context!r}")
            error.add_note(f"running: {os.environ.get('PYTEST_CURRENT_TEST', '<none>')}")
            error.add_note(STRAY_CALLBACK_NOTE)
        return original(runner, loop, context)

    # `setattr`, because a method assignment is what this is: anyio's handler
    # is being wrapped for the process, and mypy refuses a direct rebind.
    setattr(TestRunner, "_exception_handler", attributing)  # noqa: B010


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the tests whose binary this host does not have.

    One decision per binary for the whole run, rather than the `skipif` constant
    this replaced: two modules imported that constant and the third to drive real
    git forgot it, which on a machine without git turned a clean skip into a
    dozen errors. A marker cannot be forgotten the same way — it is spelled at
    the test, and the hook finds every test that carries it.
    """
    for marker, (binary, reason) in NEEDS_BINARY.items():
        if shutil.which(binary) is not None:
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path) -> Iterator[None]:
    """Every test gets its own `$PH_HOME`, whether or not it asked.

    Autouse because the opt-in version did not hold. `mount` pinned it and
    `make_tui_app` pinned it, so a test that mounted a profile through neither
    wrote its sessions into the developer's real home — which the daemon tests
    did (five logs under fixed ids that later runs append to), and which
    something else had been doing for days before anyone looked.

    That is the *fourth* appearance of this class in this suite: a test repo in
    the checkout, a stray `parent-tree`, a real `ph/*` worktree, and now
    sessions. Each earlier one was fixed where it happened. This is the rule
    stated once, in the one place no test can route around — a test that
    genuinely needs another home sets it after this runs and wins.

    **Not through the shared `monkeypatch`, which is how it got routed around
    anyway.** That fixture is one function-scoped instance shared by every other
    fixture and the test itself, so a test calling `monkeypatch.undo()` — to
    drop a patch of its *own*, entirely reasonably — reverts this one too, and
    everything after that line runs against the developer's real home. P5-04's
    resume test did exactly that and wrote a session into `~/.ph/sessions/`,
    which is the fifth appearance of this class and the second in this fixture.
    A private `MonkeyPatch` of its own is out of reach of anything a test does
    to its patches, and still restores the variable the way every other env pin
    in this suite does.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("PH_HOME", str(tmp_path))
        # `$PH_CACHE` for the same reason, added when P7-03 put a *writer* behind
        # it: the upload handle cache lives there, so without this every test that
        # mounted the row would write into the developer's real `~/.cache/ph`.
        # That is the sixth appearance of this class in this suite, and the first
        # one caught before it happened rather than after.
        patch.setenv("PH_CACHE", str(tmp_path / "cache"))
        # `$XDG_CONFIG_HOME` because **`jj` keeps per-repo state outside the
        # repo**, and that is the seventh appearance of this class — caught by a
        # sandbox denying the write, after the suite had already left **2,213**
        # directories under the developer's real `~/.config/jj/repos/`, one per
        # jj test repo, each holding a `config.toml` and a pointer to a pytest
        # path deleted minutes later.
        #
        # It is outside the repo on purpose: jj calls it the *secure* config, so
        # that cloning a repository cannot inject settings into the client. That
        # is a good reason, and it means `jj config set --repo` in a test fixture
        # can only be made hermetic from out here — no argument to that command
        # keeps it inside `tmp_path`.
        #
        # **Beside `tmp_path`, never inside it.** A test's repo is built *at*
        # `tmp_path`, and jj commits everything the project does not ignore — so a
        # config home under it becomes part of the working copy, and the test that
        # asserts a read adopts nothing found `config/jj/repos/…` in the commit.
        # The parent is pytest's own numbered directory: never a repo, and swept
        # with the run.
        patch.setenv("XDG_CONFIG_HOME", str(tmp_path.parent / "xdg-config"))
        yield


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def reaped_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReapedHost:
    """`reaped_host(linger=…)` → a host whose `$XDG_RUNTIME_DIR` logind reaps.

    Returns `$PH_RUNTIME`, which is placed *inside* `$XDG_RUNTIME_DIR` so that
    removing that directory is what logout does rather than something like it.
    `linger=False` is a user who does not linger, `True` one who does, and
    `None` the third state — a host with no `/var/lib/systemd/linger` at all,
    which must read as "unknown" and never as "off". `logind=False` is the fourth:
    a host systemd did not boot, which must read as "not applicable" and advise
    nothing — the branch that had no test at all while its evidence was inferred
    from the marker directory. `$PH_HOME` is untouched:
    `_isolated_home` above owns it, and a second fixture setting the same
    variable is how a rule comes to be half applied.

    Here rather than in either package's tests because it is needed from both,
    and `packages/ph-app/tests/daemon_helpers.py` is out of reach of the ph-core
    suite. It is the same argument `_isolated_home` above makes at greater
    length, and the same hazard: this patches a **module global**
    (`ph.lingering.LINGER_DIR`), so a copy that got missed after a rename would
    not fail — it would read the developer's or the CI box's real linger
    directory and assert whatever that host happened to say. P5-11 landed with
    five copies of these four lines; this is them stated once.
    """

    def make(*, linger: bool | None = False, user: str = "someone", logind: bool = True) -> Path:
        from ph import lingering

        # The runtime tier's inputs only. `$PH_HOME` is `_isolated_home`'s to
        # own — clearing it here would undo the one pin no test may route
        # around, and setting it would make one rule read as two.
        for name in ("PH_RUNTIME", "XDG_RUNTIME_DIR", "TMPDIR"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("USER", user)
        runtime_dir = tmp_path / "xdg"
        runtime = runtime_dir / "ph"
        runtime.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
        monkeypatch.setenv("PH_RUNTIME", str(runtime))
        markers = tmp_path / "linger"
        if linger is not None:
            markers.mkdir(exist_ok=True)
            if linger:
                (markers / user).touch()
        monkeypatch.setattr(lingering, "LINGER_DIR", markers)
        # Staged separately, because it answers a separate question: `logind=False`
        # is a host systemd did not boot, where nothing reaps a runtime directory
        # and there is nothing to advise. Inferring it from `markers` instead would
        # make the "no logind" branch unreachable in a test — a temp directory
        # always has a parent that exists — which is how it went untested when the
        # two facts shared one constant.
        booted = tmp_path / "systemd-run"
        if logind:
            booted.mkdir(exist_ok=True)
        monkeypatch.setattr(lingering, "SYSTEMD_RUN_DIR", booted)
        return runtime

    return make


@pytest.fixture
async def mount(tmp_path: Path) -> AsyncIterator[MountProfile]:
    """`await mount(*overlay_rows)` → a mounted root; disposed after the test.

    Sessions persist under `tmp_path` unless an overlay says otherwise, because
    `_isolated_home` pins `$PH_HOME` for every test. This fixture used to pin it
    too — the same variable to the same value — which made one rule read as
    three, two of them through the `monkeypatch` route `_isolated_home` was
    rewritten to stop trusting. A change there would have been silently half
    applied.
    """
    roots: list[Context] = []

    async def _mount(*overlay_rows: dict[str, Any], profile: Any = None) -> Context:  # noqa: ANN401
        """`profile` layers a bundle between the base and the overlay.

        One keyword rather than a second fixture, because "mount the shipped
        profile" and "mount base plus these rows" differ by one document and
        should not differ by a lifecycle. A *sequence* of paths is the whole
        profile — what `resolve_profile` returns — so a test can mount exactly
        what a person's `--profile` composes rather than re-deriving it.
        """
        if profile is None:
            paths = [BASE, HEADLESS]
        elif isinstance(profile, (list, tuple)):
            paths = list(profile)
        else:
            paths = [BASE, HEADLESS, profile]
        documents = load_profile_documents(paths)
        # The filesystem root goes to `tmp_path` for the same reason `PH_HOME`
        # does, and it is the stronger of the two: `fs.root` is what a workspace
        # tier *branches a git worktree from*, so a test that left it at the
        # process cwd would make checkouts and `ph/*` branches in the
        # developer's own repository. That has happened three times in this
        # suite — a test repo inside the checkout, a stray `parent-tree`, and a
        # real worktree off `main` — and each time the fix was local. This is
        # the guarantee stated once.
        #
        # **Stated on the rung a daemon takes, not the one an operator takes.**
        # It was a config patch on the `fs` row — `config.root`, the top rung of
        # `fs-local`'s ladder — chosen before `Profile.mount` had a `project=`
        # door. That pinned the root, but it meant the whole suite exercised the
        # one rung production never uses, while the rung a daemon *does* use for
        # every root it holds (P5-14) had a single test. Passing `project=` below
        # runs all ~2,570 mounts through the production path instead. The
        # override story is unchanged: a test that wants another root sets
        # `config.root` in an overlay row, which outranks the project exactly as
        # it outranks it for a deployment — and no shipped profile sets it, so
        # nothing this fixture mounts can win against `tmp_path` by accident.
        patches: list[dict[str, Any]] = []
        # Asked of the loader rather than re-read from the raw documents: which
        # entries are rows, which are patches and what `insert:`/`remove:` mean is
        # `compose_rows`' grammar, and a second copy of it here got `remove:` wrong.
        if any(row.id == "sandbox-local" for row in compose_rows(documents)):
            # `sandbox-local` ships in `ph-base` and probes the host at every mount
            # — two spawns and a proxy — and on a host where bwrap works every
            # `ctx.shell.run` in the suite would then be confined, which is a
            # different measurement from the one most tests make. Off here, once;
            # a test that wants the backend says
            # `{"id": "sandbox-local", "disabled": False}`. Only when the row is
            # there to patch: a test composing its own layers without `ph-base`
            # must not be refused over a row it never had.
            patches.append({"id": "sandbox-local", "disabled": True})
        if patches:
            documents.append(("test-root", patches))
        if overlay_rows:
            documents.append(("test-overlay", list(overlay_rows)))
        ctx = Context()
        await Profile.from_documents(documents).mount(ctx, project=tmp_path)
        roots.append(ctx)
        return ctx

    yield _mount
    for ctx in reversed(roots):
        await ctx.drain()
        await ctx.dispose()
