"""P0-05 — the loader: rows, patches, interpolation, and no code evaluation.

Gate: *`--dump-config` shows composed rows; a `!!js`-style tag is rejected; a
plugin activates only when its `inject` keys are provided.*

The tag test is the one that matters most (D9, invariant I-8): executing code
from a config file is the single dsh idiom deliberately not ported, and the
refusal has to be at parse time, not at use time.
"""

from __future__ import annotations

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
