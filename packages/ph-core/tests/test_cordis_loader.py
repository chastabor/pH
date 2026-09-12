"""P0-05 — the loader: rows, patches, interpolation, and no code evaluation.

Gate: *`--dump-config` shows composed rows; a `!!js`-style tag is rejected; a
plugin activates only when its `inject` keys are provided.*

The tag test is the one that matters most (D9, invariant I-8): executing code
from a config file is the single dsh idiom deliberately not ported, and the
refusal has to be at parse time, not at use time.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from ph.cordis import Context, Disposer, LoaderError, plugin
from ph.cordis.loader import (
    PROJECT_ROOT,
    Profile,
    _state,
    compose_rows,
    evaluate_predicate,
    interpolate,
    safe_yaml_load,
)
from ph.keys import MOUNT, SANDBOX
from ph.testing import MountProfile, not_none
from ph.wire import WireModel

pytestmark = pytest.mark.anyio


def _doc(name: str, text: str) -> tuple[str, object]:
    return name, safe_yaml_load(text, origin=name)


def _fake_module(name: str, **plugins: Any) -> None:
    """Register `plugins` under an importable module name for `name:` to resolve.

    Written out four times in this file before it was one helper — each copy
    with its own `import sys; import types`, and only the first cleaning up.
    """
    module = types.ModuleType(name)
    vars(module).update(plugins)
    sys.modules[name] = module


def test_rows_keep_file_order() -> None:
    rows = compose_rows([_doc("base", "- id: a\n  name: mod.a\n- id: b\n  name: mod.b\n")])
    assert [row.id for row in rows] == ["a", "b"]
    assert rows[0].layer == "base"


def test_patch_replaces_a_whole_config_by_id() -> None:
    rows = compose_rows(
        [
            _doc("base", "- id: a\n  name: mod.a\n  config:\n    x: 1\n    y: 2\n"),
            _doc("overlay", "- id: a\n  config:\n    x: 9\n"),
        ]
    )
    # A patch replaces rather than merges, so a row's effective value is always
    # one layer's and readable in one place.
    assert rows[0].config == {"x": 9}
    assert rows[0].layer == "overlay"


def test_insert_appends_new_rows_and_refuses_duplicate_ids() -> None:
    rows = compose_rows(
        [
            _doc("base", "- id: a\n  name: mod.a\n"),
            _doc("overlay", "- insert:\n    - id: b\n      name: mod.b\n"),
        ]
    )
    assert [row.id for row in rows] == ["a", "b"]
    with pytest.raises(LoaderError, match="duplicate row id"):
        compose_rows(
            [
                _doc("base", "- id: a\n  name: mod.a\n"),
                _doc("overlay", "- insert:\n    - id: a\n      name: mod.a\n"),
            ]
        )


def test_patching_an_unknown_id_is_an_error() -> None:
    with pytest.raises(LoaderError, match='no row with id "ghost"'):
        compose_rows([_doc("overlay", "- id: ghost\n  config: {}\n")])


def test_code_tags_are_refused_at_parse_time() -> None:
    for tag in ("!!js process.env.HOME", "!!python/object/apply:os.system ['id']"):
        with pytest.raises(LoaderError):
            safe_yaml_load(f"- id: a\n  name: mod.a\n  config: {tag}\n")


def test_timestamps_stay_strings() -> None:
    # A config file is data. A value that looks like a date is the string the
    # author wrote, not a datetime someone has to guess about.
    parsed = safe_yaml_load("- id: a\n  name: mod.a\n  config:\n    when: 2026-08-26\n")
    assert parsed[0]["config"]["when"] == "2026-08-26"


def test_env_interpolation_with_defaults() -> None:
    env = {"PH_TEST_MODEL": "big"}
    assert interpolate("${env:PH_TEST_MODEL}", env) == "big"
    assert interpolate("${env:PH_TEST_MISSING:-small}", env) == "small"
    assert interpolate({"a": ["${env:PH_TEST_MODEL}"]}, env) == {"a": ["big"]}
    with pytest.raises(LoaderError, match="declares no :- default"):
        interpolate("${env:PH_TEST_MISSING}", env)


def test_disabled_predicates_are_closed() -> None:
    assert evaluate_predicate(True) is True
    assert evaluate_predicate(None) is False
    assert evaluate_predicate("${env:PH_TEST_FLAG}", {"PH_TEST_FLAG": "1"}) is True
    assert evaluate_predicate("${env:PH_TEST_FLAG}", {}) is False
    with pytest.raises(LoaderError, match="not a supported predicate"):
        evaluate_predicate("os.system('id')")


def test_disabled_rows_are_not_mounted() -> None:
    profile = Profile.from_documents(
        [_doc("base", "- id: a\n  name: mod.a\n  disabled: true\n- id: b\n  name: mod.b\n")]
    )
    assert [row.id for row in profile.enabled_rows()] == ["b"]
    assert [row["id"] for row in profile.dump()] == ["a", "b"]
    assert profile.dump()[0]["disabled"] is True


async def test_mounting_activates_only_rows_whose_injections_resolve() -> None:
    applied: list[str] = []

    @plugin("t-provider")
    async def provider(ctx: Context, config: None) -> None:
        applied.append("provider")
        ctx.provide("t_thing", 1)

    @plugin("t-consumer", inject=["t_thing"])
    async def consumer(ctx: Context, config: None) -> None:
        applied.append("consumer")

    @plugin("t-orphan", inject=["t_absent"])
    async def orphan(ctx: Context, config: None) -> None:
        applied.append("orphan")

    import sys
    import types

    module = types.ModuleType("ph_test_rows")
    # `setattr`, because that is what these assignments are: a `ModuleType`
    # built at runtime has no declared members for mypy to check against.
    for name, row in (("provider", provider), ("consumer", consumer), ("orphan", orphan)):
        setattr(module, name, row)
    sys.modules["ph_test_rows"] = module

    # Deliberately mounted consumer-first: file order must not decide.
    profile = Profile.from_documents(
        [
            _doc(
                "base",
                "- id: consumer\n  name: ph_test_rows:consumer\n"
                "- id: provider\n  name: ph_test_rows:provider\n"
                "- id: orphan\n  name: ph_test_rows:orphan\n",
            )
        ]
    )
    root = Context()
    mount = await profile.mount(root)
    assert applied == ["provider", "consumer"]
    assert mount.inactive() == ["orphan"]
    await root.dispose()
    del sys.modules["ph_test_rows"]


async def test_topology_reports_what_the_mount_became_not_what_was_written() -> None:
    """The live half of `--dump-config`, and the distinction it exists for.

    `dump()` is the composition before anything runs and says so. It cannot tell
    a row that activated from one that mounted and never did — an unmet `inject`
    key — because both are simply rows. `inactive()` knew, and for one round
    nothing called it. dsh's rule is that once structure comes from configuration
    the static code no longer says what is running, so the dump has to; this is
    that account, per row.

    Four states, each asserted: active with what it injects, waiting with the
    key it lacks named, disabled with the layer that turned it off, and the
    isolated realms — none before an agent exists, and then the agent's path.
    """

    @plugin("t-provider")
    async def provider(ctx: Context, config: None) -> None:
        ctx.provide("t_thing", 1)

    @plugin("t-consumer", inject=["t_thing"])
    async def consumer(ctx: Context, config: None) -> None:
        pass

    @plugin("t-orphan", inject=["t_thing", "t_absent"])
    async def orphan(ctx: Context, config: None) -> None:
        pass

    _fake_module("ph_test_topology", provider=provider, consumer=consumer, orphan=orphan)

    profile = Profile.from_documents(
        [
            _doc(
                "layers/base.yaml",
                "- id: provider\n  name: ph_test_topology:provider\n"
                "- id: consumer\n  name: ph_test_topology:consumer\n"
                "- id: orphan\n  name: ph_test_topology:orphan\n"
                "- id: switched-off\n  name: ph_test_topology:consumer\n",
            ),
            _doc("layers/site.yaml", "- id: switched-off\n  disabled: true\n"),
        ]
    )
    ctx = Context()
    mount = await profile.mount(ctx)

    rows = dict(mount.topology())

    assert rows["consumer"] == "active · injects t_thing · from layers/base.yaml"
    # Only the *unmet* key is named: `t_thing` resolved, and listing it as
    # waited-on would send a reader to fix a service that is there.
    assert (
        rows["orphan"] == "waiting on t_absent · injects t_thing, t_absent · from layers/base.yaml"
    )
    # `by` names the layer that flipped it, not the one that defined it: a patch
    # re-stamps the row's layer, so "why isn't X running" gets the file to open.
    assert rows["switched-off"] == "disabled · by layers/site.yaml"
    assert rows["isolated realms"].startswith("none")

    agent = ctx.scope("agent:a1")
    assert dict(mount.topology())["isolated realms"] == agent.path
    await ctx.dispose()


async def test_topology_follows_a_fiber_through_a_provider_swap() -> None:
    """The states a *live* reader sees, and `ph doctor` never does.

    `doctor` reads after the fixpoint, so it meets `active`, `waiting on`, and
    `disabled`. A running daemon is asked mid-life: a provider swaps its service
    out, and the fiber that injected it is *unwound* — which is not the same fact
    as never having come up, though `_Dependent` recorded only `active` and the
    two printed identically. Then the replacement arrives and, until the next
    `reconcile`, the fiber is ready and not yet running; a first draft printed
    that as `waiting on ` followed by nothing. Then it is active again. And the
    other way a fiber goes dark: unmounted, which is not "ready".
    """
    withdraw: list[Disposer] = []

    @plugin("t-provider")
    async def provider(ctx: Context, config: None) -> None:
        withdraw.append(ctx.provide("t_thing", 1))

    @plugin("t-consumer", inject=["t_thing"])
    async def consumer(ctx: Context, config: None) -> None:
        pass

    _fake_module("ph_test_swap", provider=provider, consumer=consumer)
    profile = Profile.from_documents(
        [
            _doc(
                "layers/base.yaml",
                "- id: provider\n  name: ph_test_swap:provider\n"
                "- id: consumer\n  name: ph_test_swap:consumer\n",
            )
        ]
    )
    ctx = Context()
    mount = await profile.mount(ctx)

    def consumer_line() -> str:
        return dict(mount.topology())["consumer"]

    assert consumer_line().startswith("active ·")

    withdraw[0]()  # the provider swaps its service out
    await ctx.reconcile()
    assert (
        consumer_line() == "unwound · waiting on t_thing · injects t_thing · from layers/base.yaml"
    )
    assert mount.inactive() == ["consumer"]

    ctx.provide("t_thing", 2)  # a replacement arrives; the fixpoint has not run
    assert consumer_line() == "activating · injects t_thing · from layers/base.yaml"

    await ctx.reconcile()
    assert consumer_line().startswith("active ·")

    await mount.forks["provider"].dispose()
    assert (
        dict(mount.topology())["provider"] == "unmounted · injects nothing · from layers/base.yaml"
    )
    await ctx.dispose()


def test_a_fork_that_never_activated_is_not_reported_as_unwound() -> None:
    """The bit is `ever_active`, not `not active`: a fresh fork prints `waiting on`.

    Direct, on `_state`, because the distinction is the whole of the change: the
    mount-level test above shows the unwound reading, and this pins that a fiber
    that never came up does not borrow it.
    """

    @plugin("t-consumer", inject=["t_absent"])
    async def consumer(ctx: Context, config: None) -> None:
        pass

    fork = Context().plugin(consumer)

    assert fork.ever_active is False
    assert _state(fork) == "waiting on t_absent · injects t_absent"


# ------------------------------------------------------ isolate: private realms --


def _realm_module(name: str) -> None:
    """Two plugins under a fake module: a provider of `t_fs`, and a row that reads it.

    The provider records which `root` it was given so a test can tell the private
    copy from the shared one by something other than identity.
    """

    class FsConfig(WireModel):
        root: str = "shared"

    # A model, because a row that reads config has to say so (P8-04): the loader
    # refuses a `config:` block under a row that declares none.
    @plugin("t-fs", config=FsConfig)
    async def fs_provider(ctx: Context, config: FsConfig) -> None:
        ctx.provide("t_fs", {"root": config.root, "owner": ctx.path})

    @plugin("t-reader", inject=["t_fs"])
    async def reader(ctx: Context, config: None) -> None:
        ctx.provide("t_seen", ctx.require("t_fs"))

    _fake_module(name, fs_provider=fs_provider, reader=reader)


async def test_isolate_gives_a_row_a_private_copy_of_a_service() -> None:
    """dsh's `isolate.fs`, on pH's own realms.

    A row that says `isolate: [fs]` runs in an isolation boundary that is also its
    own provisioning realm — `scope()`, the same thing an agent gets — and a second
    copy of the `fs` row is mounted inside it first. That copy's `provide` lands in
    the realm, so the isolating row resolves the private instance while a sibling
    at root resolves the shared one. Nothing redirects: `_provision` walks up from
    the realm and meets the nearer provision first.

    Both readers assert on the *value* they saw, not on `has()`: `has("t_fs")` is
    true from the realm either way (root has one), which is exactly the race the
    per-realm reconcile in `mount` exists to close.
    """
    _realm_module("ph_test_realm")
    profile = Profile.from_documents(
        [
            _doc(
                "base",
                "- id: fs\n  name: ph_test_realm:fs_provider\n"
                "- id: shared-reader\n  name: ph_test_realm:reader\n"
                "- id: private-reader\n  name: ph_test_realm:reader\n  isolate: [fs]\n",
            )
        ]
    )
    ctx = Context()
    mount = await profile.mount(ctx)

    shared = mount.forks["shared-reader"].ctx
    private = mount.forks["private-reader"].ctx
    assert shared is not None and private is not None
    assert shared.require("t_seen") is ctx.require("t_fs"), (
        "the sibling should see the shared service"
    )
    assert private.require("t_seen") is not ctx.require("t_fs"), (
        "the isolating row saw the shared service"
    )
    assert private.require("t_seen")["owner"].startswith("root/realm:private-reader/")
    # The shared instance is untouched: a realm adds a provision, it does not
    # replace one, so root and every other row keep what they had.
    assert ctx.require("t_fs")["root"] == "shared"
    assert "private-reader/fs" in mount.forks
    await ctx.dispose()


async def test_isolate_with_a_mapping_overrides_the_private_copy_s_config() -> None:
    """The form the feature exists for: a private `fs` rooted somewhere else.

    `isolate: [fs]` is a second instance with identical config — separation and
    nothing more. `isolate: {fs: {root: …}}` is what "process the sensitive data
    through a different filesystem" actually needs, and the override reaches the
    private copy without touching the shared row.
    """
    _realm_module("ph_test_realm_override")
    profile = Profile.from_documents(
        [
            _doc(
                "base",
                "- id: fs\n  name: ph_test_realm_override:fs_provider\n  config: {root: shared}\n"
                "- id: sealed\n  name: ph_test_realm_override:reader\n"
                "  isolate: {fs: {root: /sealed}}\n",
            )
        ]
    )
    ctx = Context()
    mount = await profile.mount(ctx)

    sealed = mount.forks["sealed"].ctx
    assert sealed is not None
    assert sealed.require("t_seen")["root"] == "/sealed"
    assert ctx.require("t_fs")["root"] == "shared"
    # The dump reads back as written, in whichever of the two forms was used.
    dumped = {row["id"]: row for row in profile.dump()}
    assert dumped["sealed"]["isolate"] == {"fs": {"root": "/sealed"}}
    await ctx.dispose()


@pytest.mark.parametrize(
    ("row", "site", "match"),
    [
        ("  isolate: [nope]\n", "", 'isolates "nope", which is not a row'),
        ("  isolate: [r]\n", "", "cannot isolate itself"),
        ("  isolate: [fs]\n", "- id: fs\n  disabled: true\n", "which is disabled"),
        ("  isolate: 7\n", "", "must be a list of row ids or a mapping"),
    ],
)
def test_isolate_is_checked_when_the_layers_compose_not_when_they_mount(
    row: str, site: str, match: str
) -> None:
    """`--dump-config` must refuse the same profile `ph` would.

    Four shapes a person can write that cannot mean anything: a row that is not
    there, a row isolating itself, a private copy of a row a later layer turned
    off — which would make the copy the only one running, under a key that says
    "off" — and a value that is neither list nor mapping. Each is named in the
    error, at compose time, so a dump that looked fine never precedes a mount
    that fails.
    """
    _realm_module("ph_test_realm_checks")
    documents = [
        _doc(
            "base",
            "- id: fs\n  name: ph_test_realm_checks:fs_provider\n"
            "- id: r\n  name: ph_test_realm_checks:reader\n" + row,
        )
    ]
    if site:
        documents.append(_doc("site", site))
    with pytest.raises(LoaderError, match=match):
        Profile.from_documents(documents)


async def test_a_realm_is_reported_and_unwinds_with_the_root() -> None:
    """The topology names the realm and its private copy; disposal takes both.

    A realm is a child scope like any other, so I2 covers it: disposing the root
    disposes the realm, which disposes the private copy, which unprovides. The
    test that matters is the last assertion — a private `fs` that outlived its
    realm would be a provision nothing can reach and nothing can release.
    """
    _realm_module("ph_test_realm_topology")
    profile = Profile.from_documents(
        [
            _doc(
                "base",
                "- id: fs\n  name: ph_test_realm_topology:fs_provider\n"
                "- id: sealed\n  name: ph_test_realm_topology:reader\n  isolate: [fs]\n",
            )
        ]
    )
    ctx = Context()
    mount = await profile.mount(ctx)

    rows = dict(mount.topology())
    assert rows["sealed/fs"].startswith("active · injects nothing · private copy in realm:sealed")
    assert rows["sealed/fs"].endswith("own config")
    assert rows["isolated realms"] == "root/realm:sealed"

    private = mount.forks["sealed/fs"].ctx
    assert private is not None
    realm = private.parent
    assert realm is not None and realm.active and realm.label == "realm:sealed"
    await ctx.dispose()
    assert not realm.active and not private.active
    assert not mount.forks["sealed/fs"].active


async def test_a_private_copy_that_cannot_activate_is_refused_not_fallen_through() -> None:
    """The hole the per-realm reconcile does not close, and why it is a refusal.

    `has(key)` is true from the realm as soon as root provides it, so the
    isolating row is ready at once — against the *shared* instance. A reconcile
    before it mounts fixes the order for a private copy that is ready; it does
    nothing for one whose own `inject` is unmet. That copy stays waiting, the row
    activates against root's service, and `isolate: [fs]` silently means the
    opposite of what it says. A first draft's comment claimed the reconcile
    covered this case; it does not, and the test that would have shown it was the
    one not written.

    So the loader checks, and names the key: what a private copy needs has to
    come from a row above the realm.
    """

    @plugin("t-needy-fs", inject=["t_late"])
    async def needy(ctx: Context, config: None) -> None:
        ctx.provide("t_fs", {"root": "private"})

    @plugin("t-late")
    async def late(ctx: Context, config: None) -> None:
        ctx.provide("t_late", True)

    @plugin("t-reader", inject=["t_fs"])
    async def reader(ctx: Context, config: None) -> None:
        ctx.provide("t_seen", ctx.require("t_fs"))

    _fake_module("ph_test_realm_needy", needy=needy, late=late, reader=reader)

    # `late` provides `t_late` *after* the isolating row, so the private copy
    # cannot be ready when the realm is settled.
    profile = Profile.from_documents(
        [
            _doc(
                "base",
                "- id: fs\n  name: ph_test_realm_needy:needy\n"
                "- id: sealed\n  name: ph_test_realm_needy:reader\n  isolate: [fs]\n"
                "- id: late\n  name: ph_test_realm_needy:late\n",
            )
        ]
    )
    ctx = Context()
    with pytest.raises(LoaderError, match='isolates "fs", whose private copy is waiting on t_late'):
        await profile.mount(ctx)
    await ctx.dispose()


# -------------------------------------------------------------- reconfigure --


async def test_reconfigure_reapplies_one_row_and_touches_nothing_else(mount: MountProfile) -> None:
    """P6-38: a row re-applied live releases and refills its own registrations,
    and every other fork in the mount is the same object it was."""
    ctx = await mount()
    before = {
        row_id: fork
        for row_id, fork in ctx.require(MOUNT).forks.items()
        if row_id != "sandbox-allow"
    }
    old = ctx.require(MOUNT).forks["sandbox-allow"]

    fork = await ctx.require(MOUNT).reconfigure(
        "sandbox-allow", {"network": {"mode": "allowlist", "hosts": ["only.example"]}}
    )

    assert fork is not old and old.unmounted and fork.active
    assert ctx.require(MOUNT).forks["sandbox-allow"] is fork
    assert ctx.require(SANDBOX).allowances is not None
    assert not_none(not_none(ctx.require(SANDBOX).allowances).network).hosts == ["only.example"]
    assert {
        row_id: fork
        for row_id, fork in ctx.require(MOUNT).forks.items()
        if row_id != "sandbox-allow"
    } == before
    assert dict(ctx.require(MOUNT).topology())["sandbox-allow"].endswith(
        "from bundles/base.yaml, reconfigured live"
    )
    assert ctx.require(MOUNT).reconfigured == {"sandbox-allow"}
    # The config each row runs is the fork's, not a parallel dict's.
    assert fork.config == {"network": {"mode": "allowlist", "hosts": ["only.example"]}}


async def test_reconfigure_refuses_what_it_cannot_do_honestly(mount: MountProfile) -> None:
    ctx = await mount({"id": "sandbox-allow", "disabled": True})
    with pytest.raises(LoaderError, match="no row with id"):
        await ctx.require(MOUNT).reconfigure("nonesuch", {})
    with pytest.raises(LoaderError, match="is disabled by"):
        await ctx.require(MOUNT).reconfigure("sandbox-allow", {})


# ------------------------------------------------ where a mount works (P5-14) --


async def test_a_mount_provides_the_project_it_was_given(tmp_path: Path) -> None:
    """`project=` is the door for "where this mount works".

    It was `ctx.provide("project_root", …)` *before* `mount`, done only by
    `ph_app.runtime` — an ordering no signature stated, so every other caller of
    `mount` (the tests, the plugins' own mounts, an embedder) had no way to say
    it and silently got the process's directory. Beside `ctx.mount` now, because
    it is the same kind of fact: true of this mount, needed while rows apply.
    """
    ctx = Context()
    await Profile.from_documents([]).mount(ctx, project=tmp_path)

    assert ctx.require(PROJECT_ROOT) == tmp_path
    await ctx.dispose()


async def test_a_mount_given_no_project_provides_none() -> None:
    """Absent rather than defaulted here: `fs-local` owns the fallback, and a
    `Path.cwd()` frozen at mount time would be a different answer from the one a
    row computes when it applies."""
    ctx = Context()
    await Profile.from_documents([]).mount(ctx)

    # `has` is the stronger of the two: it also rules out a provided `None`.
    assert not ctx.has(PROJECT_ROOT)
    await ctx.dispose()
