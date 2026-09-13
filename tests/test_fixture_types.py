"""`Any` on a pytest fixture is checking switched off for nothing.

pytest hands a test exactly one type per fixture and every one of them is known:
`tmp_path` is a `Path`, `monkeypatch` a `pytest.MonkeyPatch`. Annotating one
`Any` does not describe a dynamic value — it discards a type nobody had to
infer, and mypy checks these suites (`pyproject.toml`'s `files` lists every
`packages/*/tests`), so the loss is real. Measured on identical code:

    def probe(tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.set_env("PH_HOME", str(tmp_path))   # typo for setenv
        tmp_path.joinpath(3)                            # an int, not a path

`Any` reports nothing. The real types report both — `"MonkeyPatch" has no
attribute "set_env"; maybe "setenv"?` and `Argument 1 to "joinpath" ... has
incompatible type "int"`. Seventy-nine parameters were spelled the first way,
arriving over eighteen commits in three weeks rather than in one bad batch:
a habit no reviewer caught eighty times, which is what a machine is for.

**The type, not merely "not `Any`".** A wrong-but-concrete annotation is not
something mypy reliably catches either — `tmp_path: str` type-checks clean in a
body that only does `str(tmp_path)` — and the table below already holds the
right answer, so asserting it costs nothing. Compared on the last dotted
segment, so `Path` and `pathlib.Path` are the same answer and a direct
`from pytest import MonkeyPatch` is not a violation.

**Why this gate outlives the lint that now exists.** `pyproject.toml` does now
select ruff's ANN401, and the sites that predate it carry `# noqa: ANN401` until
someone types the real type. That is a different assertion from this one: it
rejects `Any` and accepts everything else, so `tmp_path: str` satisfies it and
still loses every check the paragraph above is about. The two are not layers of
one rule, and this is the half that says *which* type.

**`request` is deliberately absent.** Of its `Any` annotations, one was the
fixture and nine are an LLM or middleware request — `on_pre_step(request,
next_)`. A gate on that name would be wrong nine times out of ten, and a gate
with nine exemptions is a list. pytest agrees: `request` is special-cased in
`FixtureManager` rather than registered, so it is not a fixture object at all.

@module tests.test_fixture_types
"""

from __future__ import annotations

import ast
import pathlib

from workspace_layout import workspace_modules, workspace_tests

FIXTURES = {
    "tmp_path": "Path",
    "tmp_path_factory": "pytest.TempPathFactory",
    "monkeypatch": "pytest.MonkeyPatch",
    "capsys": "pytest.CaptureFixture[str]",
    "capfd": "pytest.CaptureFixture[str]",
    "caplog": "pytest.LogCaptureFixture",
    "recwarn": "pytest.WarningsRecorder",
    "pytestconfig": "pytest.Config",
}
"""Fixture name → the type pytest passes.

Built-ins only, and every one single-valued: what `tmp_path` is does not depend
on the suite, so there is nothing here for a test to decide. pH's own fixtures
are absent because they are typed where they are defined — `ph.testing`'s
`MountProfile`, `rlm_fixtures`' `MountedRuntime` — and a second list of them here
would be the thing this file's neighbour calls a list rather than a rule.

**Names with no site today are kept.** `capfd`, `recwarn` and `pytestconfig`
match nothing yet. Unlike a stale *exemption*, which makes a gate pass what it
should fail, a rule entry that matches nothing cannot weaken anything — it is
what catches the first one written tomorrow, at the cost of one line of data.

**`snap_compare` is excluded on purpose**, though it is a real fixture
(pytest-textual-snapshot) annotated `Any` nine times in `test_tui_snapshot.py`.
Its plugin declares `Callable[[str | PurePath], bool]` and returns a callable
taking `(app, press, terminal_size, run_before)`, so the published type is wrong
and every call here passes `terminal_size=`. `Any` is the honest annotation, and
this is why the table is curated rather than harvested out of `_pytest`: a
derived list has no way to know which upstream annotation is a lie."""


def _leaf(annotation: str) -> str:
    """The last dotted segment — what `Path` and `pathlib.Path` have in common."""
    return annotation.rsplit(".", 1)[-1]


def _wrong(path: pathlib.Path) -> list[tuple[int, str, str]]:
    """Every mis-annotated fixture parameter here: `(line, fixture, what was written)`.

    Walking for `ast.arg` rather than for functions: a parameter node occurs only
    inside an `arguments`, so one pass covers positional, positional-only,
    keyword-only and the `*args`/`**kwargs` a caller could never inject into,
    without the reader wondering which of those the gate forgot.
    """
    source = path.read_text(encoding="utf-8")
    # Three quarters of the workspace cannot hold one of these, and parsing it
    # to find that out is most of the gate's cost — 610 ms against 247 ms over
    # 434 files. Sound rather than merely quick: a parameter *named* `tmp_path`
    # puts that literal in the source, so a file this skips could not have
    # contained an offender.
    if not any(fixture in source for fixture in FIXTURES):
        return []
    tree = ast.parse(source, filename=str(path))
    return [
        (node.lineno, node.arg, written)
        for node in ast.walk(tree)
        if isinstance(node, ast.arg) and node.arg in FIXTURES
        if (written := ast.unparse(node.annotation) if node.annotation else "nothing")
        if _leaf(written) != _leaf(FIXTURES[node.arg])
    ]


def test_every_pytest_fixture_carries_the_type_pytest_passes() -> None:
    """The suites and the helpers they call, which are shipped code.

    `workspace_modules` as well as `workspace_tests` because `ph.testing` ships
    the fixtures the suites use — `ph/testing/git.py` takes a `tmp_path` — and a
    helper that mis-annotates one loses exactly as much checking as a test does.
    """
    offenders = [
        f"{name}:{line}: {fixture} is annotated `{written}` — pytest passes `{FIXTURES[fixture]}`"
        for name, path in [*workspace_tests(), *workspace_modules()]
        for line, fixture, written in _wrong(path)
    ]
    assert offenders == [], (
        "these annotate a pytest fixture with something other than the type "
        "pytest guarantees, which stops mypy checking every use of it:\n  " + "\n  ".join(offenders)
    )
