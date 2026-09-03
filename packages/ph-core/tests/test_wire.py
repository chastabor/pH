"""P0-06 — wire casing: declare, never derive (Q2).

Gate: *the round-trip property passes over every model; a model without the
shared base fails the assertion.*

The second half is the point. `to_camel` → `to_snake` happens to round-trip for
every field name in this plan, so a runtime string conversion would pass its
tests today and break at the first acronym or digit. Pinning aliases at class
definition means a name is never reconstructed from a wire string at all.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic.alias_generators import to_camel

from ph.wire import WireModel, wire_alias


def _all_ph_models() -> list[type[BaseModel]]:
    """Every pH pydantic model that crosses a pH-owned JSON boundary.

    `ToolModel` subclasses are excluded because they are the one declared
    exemption (Q2): a tool's parameter names are the model's vocabulary, not
    pH's wire, so they stay snake_case.
    """
    import ph
    from ph.tools.definition import ToolModel

    found: dict[str, type[BaseModel]] = {}
    for info in pkgutil.walk_packages(ph.__path__, prefix="ph."):
        module = importlib.import_module(info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseModel)
                and obj not in (BaseModel, WireModel, ToolModel)
                and not issubclass(obj, ToolModel)
                and obj.__module__.startswith("ph.")
            ):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return list(found.values())


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


@pytest.mark.parametrize("model", _all_ph_models(), ids=lambda m: m.__name__)
def test_models_round_trip_by_alias(model: type[BaseModel]) -> None:
    sample = _sample(model)
    if sample is None:
        pytest.skip(f"no constructible sample for {model.__name__}")
    wire = sample.model_dump(by_alias=True, exclude_none=True)
    assert model.model_validate(wire) == sample
    # Tolerant readers: the snake_case form validates too, which is what lets
    # `ph session import` ingest a foreign JSONL without a second parser.
    assert model.model_validate(sample.model_dump(exclude_none=True)) == sample


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
