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
from typing import Any

import pytest

from ph.cordis import Context, LoaderError, plugin
from ph.cordis.loader import (
    Loader,
    compose_rows,
    evaluate_predicate,
    interpolate,
    safe_yaml_load,
)

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
    loader = Loader.from_documents(
        [_doc("base", "- id: a\n  name: mod.a\n  disabled: true\n- id: b\n  name: mod.b\n")]
    )
    assert [row.id for row in loader.enabled_rows()] == ["b"]
    assert [row["id"] for row in loader.dump()] == ["a", "b"]
    assert loader.dump()[0]["disabled"] is True


async def test_mounting_activates_only_rows_whose_injections_resolve() -> None:
    applied: list[str] = []

    @plugin("t-provider")
    async def provider(ctx: Context, config: object) -> None:
        applied.append("provider")
        ctx.provide("t_thing", 1)

    @plugin("t-consumer", inject=["t_thing"])
    async def consumer(ctx: Context, config: object) -> None:
        applied.append("consumer")

    @plugin("t-orphan", inject=["t_absent"])
    async def orphan(ctx: Context, config: object) -> None:
        applied.append("orphan")

    import sys
    import types

    module = types.ModuleType("ph_test_rows")
    module.provider = provider
    module.consumer = consumer
    module.orphan = orphan
    sys.modules["ph_test_rows"] = module

    # Deliberately mounted consumer-first: file order must not decide.
    loader = Loader.from_documents(
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
    await loader.mount(root)
    assert applied == ["provider", "consumer"]
    assert loader.inactive() == ["orphan"]
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
    async def provider(ctx: Context, config: object) -> None:
        ctx.provide("t_thing", 1)

    @plugin("t-consumer", inject=["t_thing"])
    async def consumer(ctx: Context, config: object) -> None:
        pass

    @plugin("t-orphan", inject=["t_thing", "t_absent"])
    async def orphan(ctx: Context, config: object) -> None:
        pass

    _fake_module("ph_test_topology", provider=provider, consumer=consumer, orphan=orphan)

    loader = Loader.from_documents(
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
    await loader.mount(ctx)

    rows = dict(loader.topology(ctx))

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
    assert dict(loader.topology(ctx))["isolated realms"] == agent.path
    await ctx.dispose()


# ------------------------------------------------------ isolate: private realms --


def _realm_module(name: str) -> None:
    """Two plugins under a fake module: a provider of `t_fs`, and a row that reads it.

    The provider records which `root` it was given so a test can tell the private
    copy from the shared one by something other than identity.
    """

    @plugin("t-fs")
    async def fs_provider(ctx: Context, config: object) -> None:
        root = (config or {}).get("root", "shared") if isinstance(config, dict) else "shared"
        ctx.provide("t_fs", {"root": root, "owner": ctx.path})

    @plugin("t-reader", inject=["t_fs"])
    async def reader(ctx: Context, config: object) -> None:
        ctx.provide("t_seen", ctx.t_fs)

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
    loader = Loader.from_documents(
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
    await loader.mount(ctx)

    shared = loader.forks["shared-reader"].ctx
    private = loader.forks["private-reader"].ctx
    assert shared is not None and private is not None
    assert shared.t_seen is ctx.t_fs, "the sibling should see the shared service"
    assert private.t_seen is not ctx.t_fs, "the isolating row saw the shared service"
    assert private.t_seen["owner"].startswith("root/realm:private-reader/")
    # The shared instance is untouched: a realm adds a provision, it does not
    # replace one, so root and every other row keep what they had.
    assert ctx.t_fs["root"] == "shared"
    assert "private-reader/fs" in loader.forks
    await ctx.dispose()


async def test_isolate_with_a_mapping_overrides_the_private_copy_s_config() -> None:
    """The form the feature exists for: a private `fs` rooted somewhere else.

    `isolate: [fs]` is a second instance with identical config — separation and
    nothing more. `isolate: {fs: {root: …}}` is what "process the sensitive data
    through a different filesystem" actually needs, and the override reaches the
    private copy without touching the shared row.
    """
    _realm_module("ph_test_realm_override")
    loader = Loader.from_documents(
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
    await loader.mount(ctx)

    sealed = loader.forks["sealed"].ctx
    assert sealed is not None
    assert sealed.t_seen["root"] == "/sealed"
    assert ctx.t_fs["root"] == "shared"
    # The dump reads back as written, in whichever of the two forms was used.
    dumped = {row["id"]: row for row in loader.dump()}
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
        Loader.from_documents(documents)


async def test_a_realm_is_reported_and_unwinds_with_the_root() -> None:
    """The topology names the realm and its private copy; disposal takes both.

    A realm is a child scope like any other, so I2 covers it: disposing the root
    disposes the realm, which disposes the private copy, which unprovides. The
    test that matters is the last assertion — a private `fs` that outlived its
    realm would be a provision nothing can reach and nothing can release.
    """
    _realm_module("ph_test_realm_topology")
    loader = Loader.from_documents(
        [
            _doc(
                "base",
                "- id: fs\n  name: ph_test_realm_topology:fs_provider\n"
                "- id: sealed\n  name: ph_test_realm_topology:reader\n  isolate: [fs]\n",
            )
        ]
    )
    ctx = Context()
    await loader.mount(ctx)

    rows = dict(loader.topology(ctx))
    assert rows["sealed/fs"].startswith("active · injects nothing · private copy in realm:sealed")
    assert rows["sealed/fs"].endswith("own config")
    assert rows["isolated realms"] == "root/realm:sealed"

    private = loader.forks["sealed/fs"].ctx
    assert private is not None
    realm = private.parent
    assert realm is not None and realm.active and realm.label == "realm:sealed"
    await ctx.dispose()
    assert not realm.active and not private.active
    assert not loader.forks["sealed/fs"].active


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
    async def needy(ctx: Context, config: object) -> None:
        ctx.provide("t_fs", {"root": "private"})

    @plugin("t-late")
    async def late(ctx: Context, config: object) -> None:
        ctx.provide("t_late", True)

    @plugin("t-reader", inject=["t_fs"])
    async def reader(ctx: Context, config: object) -> None:
        ctx.provide("t_seen", ctx.t_fs)

    _fake_module("ph_test_realm_needy", needy=needy, late=late, reader=reader)

    # `late` provides `t_late` *after* the isolating row, so the private copy
    # cannot be ready when the realm is settled.
    loader = Loader.from_documents(
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
        await loader.mount(ctx)
    await ctx.dispose()
