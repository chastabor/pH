"""`ctx.models` — the models a profile lists, and the one rule for choosing among them.

A route used to be two free strings, so which models a session could reach was
nowhere a profile could say. The row lists them by key; `ModelList.resolve` is
what the command line, `/model` and rpc all ask, so the three cases pinned here —
nothing, a key, a whole route — are the whole of what a choice can mean. `choose`
adds what only a mount knows: whether an adapter serves the provider.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ph.bundles import BASE, HEADLESS
from ph.cordis import Profile, ProfileDocument, load_profile_documents
from ph.keys import MODELS
from ph.seams.models import (
    Config,
    ModelChoice,
    ModelChoiceError,
    ModelList,
    ModelRoute,
    choose,
)
from ph.testing import MountProfile

pytestmark = pytest.mark.anyio

LISTED = ModelList(
    Config(
        default="main",
        models={
            "main": ModelRoute(provider="fake", model="fake-1", reasoning_effort="medium"),
            "fast": ModelRoute(provider="fake", model="fake-2"),
        },
    )
)


# ------------------------------------------------------------------ config --


@pytest.mark.parametrize(
    ("config", "said"),
    [
        ({"default": "gone", "models": {"main": {"provider": "p", "model": "m"}}}, "not a listed"),
        ({"default": "main"}, "none is listed"),
        ({"default": "Main", "models": {"Main": {"provider": "p", "model": "m"}}}, "model key"),
        ({"default": "main", "models": {"main": {"provider": "", "model": "m"}}}, "provider"),
    ],
)
def test_a_list_that_cannot_name_its_default_is_refused(config: object, said: str) -> None:
    """At the row, so a profile that would leave every root without a route is
    refused where it is written, not at the first `session/new`."""
    with pytest.raises(ValidationError, match=said):
        Config.model_validate(config)


def test_an_empty_list_is_a_profile_that_names_no_model() -> None:
    assert Config() == Config(default="", models={})


# ------------------------------------------------------------------ choice --


def test_the_flags_spell_a_key_alone_and_a_route_together() -> None:
    assert ModelChoice.from_flags(None, None) == ModelChoice()
    assert ModelChoice.from_flags(None, "fast") == ModelChoice(key="fast")
    assert ModelChoice.from_flags("fake", "fake-9") == ModelChoice(provider="fake", model="fake-9")
    with pytest.raises(ModelChoiceError, match="needs --model"):
        ModelChoice.from_flags("fake", None)


def test_a_typed_choice_splits_at_the_first_slash() -> None:
    """A key is a slug and has no `/`; a model name behind an OpenAI-compatible
    server may carry several, so only the first one separates the provider."""
    assert ModelChoice.parse(" fast ") == ModelChoice(key="fast")
    assert ModelChoice.parse("llama/org/model") == ModelChoice(provider="llama", model="org/model")
    with pytest.raises(ModelChoiceError, match="not a key or a provider/model"):
        ModelChoice.parse("llama/")


def test_a_choice_is_a_key_or_a_route_and_not_both() -> None:
    with pytest.raises(ValidationError, match="not both"):
        ModelChoice(key="fast", provider="fake", model="fake-1")
    with pytest.raises(ValidationError, match="both a provider and a model"):
        ModelChoice(provider="fake")


# ----------------------------------------------------------------- resolve --


def test_nothing_is_the_default_with_its_settings() -> None:
    chosen = LISTED.resolve(ModelChoice())

    assert chosen.key == "main"
    assert chosen.options().reasoning_effort == "medium" and chosen.options().model_key == "main"


def test_a_key_is_its_entry_and_an_unlisted_one_names_the_list() -> None:
    assert LISTED.resolve(ModelChoice(key="fast")).route.model == "fake-2"
    with pytest.raises(ModelChoiceError, match=r"it lists fast, main"):
        LISTED.resolve(ModelChoice(key="slow"))


def test_a_whole_route_is_its_listed_entry_when_there_is_one() -> None:
    """So `--provider fake --model fake-1` and `--model main` run the same entry,
    settings included, and the footer can name it."""
    chosen = LISTED.resolve(ModelChoice(provider="fake", model="fake-1"))

    assert chosen.key == "main" and chosen.route.reasoning_effort == "medium"


def test_a_whole_route_the_list_does_not_hold_is_run_as_given() -> None:
    """A person's own choice is not bounded by the list; what an *agent* may pick
    is (S7b). The key is empty, which is how every reader tells the two apart."""
    chosen = LISTED.resolve(ModelChoice(provider="fake", model="fake-9"))

    assert chosen.key == "" and chosen.route == ModelRoute(provider="fake", model="fake-9")


def test_a_profile_that_lists_nothing_runs_only_a_named_route() -> None:
    with pytest.raises(ModelChoiceError, match="lists no models"):
        ModelList().resolve(ModelChoice())
    assert ModelList().resolve(ModelChoice(provider="p", model="m")).route.label == "p/m"


def test_entries_put_the_default_first() -> None:
    assert [(one.key, one.default) for one in LISTED.entries()] == [
        ("main", True),
        ("fast", False),
    ]


# ------------------------------------------------------ profile and mount --


def test_the_list_is_read_off_a_composed_profile_as_a_mount_reads_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For the check `phern daemon` makes before it binds. Interpolated, so an
    `${env:...}` route names the model a mount would."""
    monkeypatch.setenv("CHOSEN_MODEL", "fake-7")
    overlay = ProfileDocument(
        "test",
        [
            {
                "id": "models",
                "config": {
                    "default": "main",
                    "models": {"main": {"provider": "fake", "model": "${env:CHOSEN_MODEL:-x}"}},
                },
            }
        ],
    )
    profile = Profile.from_documents([*load_profile_documents([BASE, HEADLESS]), overlay])

    assert ModelList.of(profile).resolve(ModelChoice()).route.model == "fake-7"
    base_only = Profile.from_documents(load_profile_documents([BASE]))
    assert ModelList.of(base_only).default == "", "base lists nothing: it has no adapter"


async def test_a_mount_refuses_a_route_no_adapter_serves(mount: MountProfile) -> None:
    """Refused where it is chosen, rather than at the first request as a turn that
    fails and leaves a session to explain it."""
    ctx = await mount()

    assert ctx.require(MODELS).default == "main"
    assert choose(ctx, ModelChoice()).route.label == "fake/fake-1"
    with pytest.raises(ModelChoiceError, match=r'no adapter here serves "nope".*serves fake'):
        choose(ctx, ModelChoice(provider="nope", model="m"))
