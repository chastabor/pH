"""P0-06 — wire casing: declare, never derive (Q2).

Gate: *the round-trip property passes over every model; a model without the
shared base fails the assertion.*

The second half is the point. `to_camel` → `to_snake` happens to round-trip for
every field name in this plan, so a runtime string conversion would pass its
tests today and break at the first acronym or digit. Pinning aliases at class
definition means a name is never reconstructed from a wire string at all.

**"Every model" is now literally every model.** The round-trip was parametrised
over all 75 and *skipped* the ones `_sample` could not build — which is how
`StatusReading` and `Egress`, both a required field or two away from trivial, came
to be exempt from the one property this file asserts. A missing sample is a gap in
the table below, not a test with nothing to say, so it fails and names the model.

**And every package.** This file used to live in `packages/ph-core/tests` and
walk `ph` alone while saying "every pH model", so every model outside ph-core was
exempt from all of it. That is not a hypothetical gap:
`test_every_wire_model_is_built_at_import` was written for a `ph_app` model, and
could not have caught the regression that earned it. It is here at the workspace
root now, because a rule about every package cannot be enforced from inside one
of them.

The structural gates — the shared base, the pinned alias, the built schema — run
over every package `_workspace_packages` finds, because they need nothing but the
class. No count is written down here on purpose: the function discovers the
layout precisely so that nobody maintains a list, and a number in this paragraph
would be the same list one indirection away, stale the next time a package lands.

The round-trip is the exception, and it is a deferral rather than a boundary: it
needs a constructible *sample* per model and that table is ph-core's, so it runs
over ph-core's models alone. The honest fix is to stop hand-writing the table —
generate each sample from the model's own schema — and until then
`test_every_field_alias_equals_to_camel_of_its_name` is the casing property the
rest are held to.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic.alias_generators import to_camel
from workspace_layout import workspace_packages

from ph.wire import WireModel, wire_alias


def _workspace_packages() -> list[str]:
    """Every top-level package this workspace ships, read off the layout.

    Discovered, not listed. A list is exactly what went stale here — this file
    walked `ph` and said "every pH model" — and a list would go stale again the
    next time a package is added, in the same silent direction: the gate keeps
    passing while covering less. The walk itself is `workspace_layout`'s, because
    a second gate needed it and two implementations of "discovered" drift the
    same way a list does.
    """
    return [package.name for package in workspace_packages()]


def _models_in(package: str) -> dict[str, type[BaseModel]]:
    """Every pydantic model one package defines, by qualified name.

    `ToolModel` subclasses are excluded because they are the one declared
    exemption (Q2): a tool's parameter names are the model's vocabulary, not
    pH's wire, so they stay snake_case.

    Defined *in* the package, not merely visible from it: `inspect.getmembers`
    sees every imported name, so without the `__module__` test a core model would
    be checked once per package that imports it.
    """
    from ph.tools.definition import ToolModel

    root = importlib.import_module(package)
    found: dict[str, type[BaseModel]] = {}
    for info in pkgutil.walk_packages(root.__path__, prefix=f"{package}."):
        module = importlib.import_module(info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseModel)
                and obj not in (BaseModel, WireModel, ToolModel)
                and not issubclass(obj, ToolModel)
                and obj.__module__.startswith(f"{package}.")
            ):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return found


def _all_ph_models() -> list[type[BaseModel]]:
    """Every pH pydantic model that crosses a pH-owned JSON boundary."""
    found: dict[str, type[BaseModel]] = {}
    for package in _workspace_packages():
        found.update(_models_in(package))
    return list(found.values())


def _core_models() -> list[type[BaseModel]]:
    """The ph-core half, which is what the `_sample` table below covers."""
    return list(_models_in("ph").values())


def test_tool_schemas_stay_snake_case() -> None:
    """The exemption is real, and it is only for tool schemas.

    A tool parameter renamed by an alias generator would change what the model
    has to type — `old_text` becoming `oldText` is an API break dressed as a
    convention.
    """
    from ph.tools.builtin.fs_tools import EditArgs
    from ph.tools.definition import ToolModel

    assert issubclass(EditArgs, ToolModel)
    assert "old_text" in EditArgs.model_json_schema()["properties"]
    assert all(field.alias is None for field in EditArgs.model_fields.values())


def test_every_ph_model_uses_the_shared_wire_base() -> None:
    """Including plugin row configs: a YAML row is a JSON boundary too."""
    offenders = [
        f"{model.__module__}.{model.__qualname__}"
        for model in _all_ph_models()
        if not issubclass(model, WireModel)
    ]
    assert offenders == [], (
        "these models cross a JSON boundary without ph.wire.WireModel, so their "
        f"field names would reach the wire un-aliased: {offenders}"
    )


def test_every_field_alias_equals_to_camel_of_its_name() -> None:
    for model in _all_ph_models():
        for name, field in model.model_fields.items():
            assert field.alias == to_camel(name), (
                f"{model.__name__}.{name} has alias {field.alias!r}; aliases are "
                "pinned to wire_alias(field) so no name is ever re-derived"
            )


@pytest.mark.parametrize("model", _core_models(), ids=lambda m: m.__name__)
def test_models_round_trip_by_alias(model: type[BaseModel]) -> None:
    sample = _sample(model)
    assert sample is not None, (
        f"no constructible sample for {model.__name__}: add one to `_sample`'s table. "
        "This used to `skip`, which meant a model with required fields joined the "
        "suite already exempt from the property this file exists to assert — "
        "`StatusReading` and `Egress` sat unchecked that way. A model pH puts on a "
        "wire is either round-tripped here or argued about; it is not quietly passed."
    )
    wire = sample.model_dump(by_alias=True, exclude_none=True)
    assert model.model_validate(wire) == sample
    # Tolerant readers: the snake_case form validates too, which is what lets
    # `ph session import` ingest a foreign JSONL without a second parser.
    assert model.model_validate(sample.model_dump(exclude_none=True)) == sample


def test_every_wire_model_is_built_at_import() -> None:
    """No model defers its schema to a lazy rebuild at first use.

    pydantic leaves `__pydantic_complete__` false when a field names a type it
    cannot resolve yet — a class defined *below* the model that annotates it,
    say — and quietly rebuilds on the first `model_validate`. That moves a
    structural error off import and onto whatever path validates first, which
    here is config load and mount: the two places a person is waiting.

    It is not a thing anyone writes on purpose; it is what happens when a class
    moves. `WindowProbe` sat 468 lines below the `ProviderProfile` that annotates
    it and this was false for that model and the `Config` embedding it, which is
    how the rule earned a test rather than a note.

    Both of those are `ph_app` models, so the first version of this test — written
    in `packages/ph-core/tests`, walking `ph` — passed with the class moved back
    down. A gate that cannot fail on the case it was written for is worse than no
    gate, and that is why the whole file moved up here.
    """
    deferred = [model.__name__ for model in _all_ph_models() if not model.__pydantic_complete__]
    assert not deferred, (
        f"these models defer their schema to a lazy rebuild: {deferred}. Define the types "
        "they name above them, so the failure is an import error rather than a surprise "
        "the first time something validates."
    )


def _sample(model: type[BaseModel]) -> BaseModel | None:
    from ph.llm.types import LlmCallConfig, TextBlock, ToolSchema, UserSource

    prepared: dict[str, Any] = {
        "Message": {
            "id": "m1",
            "role": "user",
            "content": [TextBlock(text="hi")],
            "source": UserSource(),
        },
        "TextBlock": {"text": "hi"},
        "PresetSchema": {"name": "read-only", "summary": "Reads freely.", "active": True},
        "Schedule": {"id": "s1", "kind": "interval", "spec": "300000", "prompt": "go"},
        "Goal": {"id": "g1", "objective": "make the tests pass"},
        "Budget": {},
        "ReasoningBlock": {"text": "thinking"},
        "ToolResultBlock": {"toolCallId": "c1", "content": [TextBlock(text="out")]},
        "ToolCallBlock": {"id": "c1", "name": "read", "arguments": "{}"},
        "AttachmentRef": {"attachmentId": "sha256:abc", "mime": "image/png", "bytes": 3},
        "FileHandle": {
            "provider": "anthropic",
            "attachmentId": "sha256:abc",
            "handle": "file_01",
            "uploadedAt": 1,
        },
        "MediaBlock": {
            "attachment": {"attachmentId": "sha256:abc", "mime": "image/png", "bytes": 3}
        },
        "PluginSource": {"plugin": "ph.test", "form": "notice", "summary": "did a thing"},
        "ModelSource": {"provider": "fake", "model": "fake-1"},
        "ToolSource": {"callId": "c1"},
        "TokenUsage": {"inputTokens": 3, "outputTokens": 4},
        "LlmFailure": {"message": "nope", "code": "UNKNOWN"},
        "LlmCallConfig": {"provider": "fake", "model": "fake-1"},
        "LlmCallConfigAdapterDefaults": {"maxTokens": True},
        "ToolSchema": {"name": "read", "description": "read a file", "parameters": {}},
        "ContextSnapshotSection": {"name": "workspace", "text": "cwd: /x"},
        "ProvisionEntry": {"source": ".env"},
        "SessionHeader": {"id": "s1", "createdAt": 1},
        "EpochHeader": {
            "config": LlmCallConfig(provider="fake", model="fake-1"),
            "system": "be helpful",
            "tools": [ToolSchema(name="read", description="d", parameters={})],
        },
        "RequestContext": {"provider": "fake", "model": "fake-1", "contextWindow": 8},
        "SurfaceReplace": {"replaces": [1, 2]},
        "UserQuestion": {"question": "which?"},
        "CommandSchema": {"name": "compact", "summary": "Fold the transcript."},
        "ScreenSchema": {"id": "trajectory", "label": "Trajectory"},
        "ToolCallView": {"title": "read"},
        "ToolResultView": {"title": "read"},
        "ApprovalRequest": {"toolName": "edit"},
        "CredentialRef": {"name": "OPENAI_API_KEY"},
        "FileSlice": {"path": "/x", "text": "hi"},
        "GrepMatch": {"path": "/x", "line": 1, "text": "hit"},
        "Skill": {"name": "review", "description": "Review a diff."},
        "SpillRef": {"locator": "/x", "bytes": 3, "retrievalHint": "read /x"},
        "SessionTelemetryRecord": {
            "channel": "ops",
            "time": 1,
            "severity": "info",
            "attributes": {},
            "body": "started",
        },
        "SandboxPolicy": {"mode": "read-only"},
        # Both halves of the egress door, `agent` included: it is the field that
        # carries who a refusal belongs to, and an optional one left out of a
        # sample is a field `exclude_none=True` would drop before the round trip
        # could test it.
        "Egress": {"socket": "/run/user/1/ph/egress.sock", "port": 3128, "agent": "s/1"},
        # A non-default `level`, for the same reason: the default would round-trip
        # through a field the wire never carried.
        "StatusReading": {"text": "3 refused", "level": "warning"},
        "CodeDispatchRef": {
            "rootCallId": "r",
            "parentCallId": "p",
            "subCallId": "p:code:0",
            "name": "read",
        },
        "CodeDispatchLog": {
            "rootCallId": "r",
            "parentCallId": "p",
            "subCallId": "p:code:0",
            "name": "read",
            "isError": False,
        },
        "_EventWire": {"type": "turn/start", "seq": 0, "time": 1, "data": {"turn": 1}},
    }
    fields = prepared.get(model.__name__)
    if fields is None:
        try:
            return model()
        except Exception:
            return None
    return model.model_validate(fields)


def test_wire_alias_is_the_single_alias_function() -> None:
    assert wire_alias("source_event_seqs") == "sourceEventSeqs"
    assert wire_alias("tool_call_id") == "toolCallId"
    assert wire_alias("max_log_bytes") == "maxLogBytes"
