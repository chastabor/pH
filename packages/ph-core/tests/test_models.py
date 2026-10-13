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
from ph.keys import DIAGNOSTICS, LLM_FAKE, MODELS
from ph.llm.adapter import ResolvedModel
from ph.llm.types import Sampling
from ph.seams.models import (
    MODEL_KINDS,
    Config,
    ModelChoice,
    ModelChoiceError,
    ModelList,
    ModelRoute,
    choose,
)
from ph.testing import MountProfile, models_overlay, report_section

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


async def test_doctor_names_a_setting_a_route_cannot_carry(mount: MountProfile) -> None:
    """A1: a route that sets a sampling setting its API does not take works, and `phern
    doctor` says which it drops, so the person learns before reading a warning in a
    log. Sabotage: leave the section out, or name a setting the route carries."""
    route = {"provider": "fake", "model": "fake-1", "sampling": {"temperature": 0.6, "topK": 20}}
    ctx = await mount(models_overlay(main=route))
    ctx.require(LLM_FAKE).route = ResolvedModel(sampling=frozenset({"temperature"}))

    report = dict(ctx.require(DIAGNOSTICS).report())

    assert report["Model settings"] == [("main", "topK dropped: fake/fake-1 takes temperature")]
    entry = ctx.require(MODELS).entry("main", ModelRoute.model_validate(route))
    assert entry.options().sampling == Sampling(temperature=0.6, top_k=20)


async def test_doctor_names_an_effort_a_route_does_not_define(mount: MountProfile) -> None:
    """A route's `efforts` say what each effort sends on its API; one that runs at an
    effort they do not define works, sends none, and `phern doctor` says so, whether or
    not its adapter said what sampling it takes. One that runs at `default`, the
    provider's own, is not named. Sabotage: skip the check where no sampling is dropped,
    or name `default`."""
    low = {"chat_template_kwargs": {"enable_thinking": False}}
    ctx = await mount(
        models_overlay(
            main={
                "provider": "fake",
                "model": "fake-1",
                "reasoningEffort": "high",
                "efforts": {"low": low},
            },
            bare={"provider": "fake", "model": "fake-2", "reasoningEffort": "low"},
            set={
                "provider": "fake",
                "model": "fake-3",
                "reasoningEffort": "low",
                "efforts": {"low": {}},
            },
            theirs={"provider": "fake", "model": "fake-4", "reasoningEffort": "default"},
        )
    )

    applies = "not sent, so the provider's default applies"
    assert report_section(ctx, "Model settings") == {
        "bare": f"reasoningEffort low {applies}: its efforts define none",
        "main": f"reasoningEffort high {applies}: its efforts define low",
    }


def test_a_route_s_efforts_are_its_listed_entry_s_its_own_key_first() -> None:
    """What each effort sends on a route is the entry the list holds it under: the key
    an agent runs under where two keys list one route, else the first, as `resolve`
    matches a whole route; nothing for a route it does not list. Sabotage: ignore the
    key, or answer an unlisted route with another's efforts."""
    off, on = {"think": False}, {"think": True}
    models = ModelList(
        Config.model_validate(
            {
                "default": "quick",
                "models": {
                    "quick": {"provider": "p", "model": "m", "efforts": {"x": off}},
                    "deep": {"provider": "p", "model": "m", "efforts": {"x": on}},
                    "plain": {"provider": "p", "model": "n"},
                },
            }
        )
    )

    assert models.efforts_for("p", "m") == {"x": off}
    assert models.efforts_for("p", "m", key="deep") == {"x": on}
    assert models.efforts_for("p", "m", key="plain") == {"x": off}
    assert models.efforts_for("p", "n") == {}
    assert models.efforts_for("p", "unlisted", key="quick") == {}


@pytest.mark.parametrize(
    ("efforts", "said"),
    [
        ({"default": {}}, "it is the provider's own default"),
        ({"Very High": {}}, "effort"),
    ],
    ids=["the-provider-s-default", "not-a-word"],
)
def test_a_route_cannot_define_the_provider_s_default_or_an_unnamed_effort(
    efforts: dict[str, object], said: str
) -> None:
    """`default` is the provider's, which sends nothing, so a profile cannot make it
    send something; and an effort is named by a word, as a skill names one. Sabotage:
    drop either check from `ModelRoute`."""
    with pytest.raises(ValidationError, match=said):
        ModelRoute.model_validate({"provider": "p", "model": "m", "efforts": efforts})


async def test_doctor_lists_which_shared_kinds_a_profile_serves(mount: MountProfile) -> None:
    """A2: the cue to set a kind up. A shared kind the list holds names its route, and one
    it does not says a skill asking for it runs on its parent's model. Sabotage: leave
    the section out, or call an unlisted kind served."""
    ctx = await mount(models_overlay(classify={"provider": "fake", "model": "fake-9"}))

    kinds = dict(dict(ctx.require(DIAGNOSTICS).report())["Model kinds"])

    assert set(kinds) == set(MODEL_KINDS)
    assert kinds["classify"] == "fake/fake-9"
    assert kinds["light"].startswith("not listed;") and "parent's model" in kinds["light"]
