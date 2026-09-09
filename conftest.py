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

import shutil
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from ph.bundles import BASE, HEADLESS
from ph.cordis import Context, Profile, load_profile_documents
from ph.cordis.loader import compose_rows

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


ReapedHost = Callable[..., Path]


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

    async def _mount(*overlay_rows: dict[str, Any], profile: Any = None) -> Context:
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
        # the guarantee stated once. A test that genuinely needs the cwd
        # overrides the row.
        patches: list[dict[str, Any]] = [{"id": "fs", "config": {"root": str(tmp_path)}}]
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
        documents.append(("test-root", patches))
        if overlay_rows:
            documents.append(("test-overlay", list(overlay_rows)))
        ctx = Context()
        await Profile.from_documents(documents).mount(ctx)
        roots.append(ctx)
        return ctx

    yield _mount
    for ctx in reversed(roots):
        await ctx.drain()
        await ctx.dispose()
